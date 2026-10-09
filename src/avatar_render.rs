#![cfg(target_os = "android")]

//! Poses every player's avatar body (`boy.glb`) from tracking data using the
//! `avatar_pose` solver: explicit rig, arm IK from the controllers, a
//! procedural walk cycle from the player's movement, finger curls, and a
//! head-hidden clone for the local player so their own head never sits in
//! front of the camera.

use std::collections::HashMap;
use std::time::Instant;

use glam::{Quat, Vec3};

use space_soup::renderer::xr_renderer::XrRenderer;
use space_soup::renderer::{mesh_pipeline::ModelUniform, GltfMesh};
use space_soup::ControllerState;
use space_soup_protocol::{PlayerId, WireWorld};

use crate::avatar;

/// OpenXR grip pose -> avatar_pose hand frame (fingers -Z, back of the hand
/// +Y), per hand [left, right]. Grip pose: -Z along the held handle's axis,
/// +X out the back of the right hand / the palm of the left. Holding a
/// handle, the fingers (wrist -> knuckles) run along grip -Y and the back of
/// the hand faces +X (right) or -X (left).
pub(crate) fn grip_to_hand(left: bool) -> Quat {
    let m = if left {
        glam::Mat3::from_cols(Vec3::NEG_Z, Vec3::NEG_X, Vec3::Y)
    } else {
        glam::Mat3::from_cols(Vec3::Z, Vec3::X, Vec3::Y)
    };
    Quat::from_mat3(&m)
}

pub(crate) fn build_bodies(
    local_player: PlayerId,
    rig: &space_soup_engine::PlayerRig,
    remotes: &HashMap<PlayerId, avatar::RemotePlayerState>,
) -> Vec<(PlayerId, avatar::RemotePlayerState)> {
    let local_state = avatar::RemotePlayerState {
        head: avatar::Transform {
            position: rig.head().position,
            rotation: rig.head().rotation,
        },
        left_hand: Some(avatar::Transform {
            position: rig.hand_grip(space_soup_engine::Hand::Left).position,
            rotation: rig.hand_grip(space_soup_engine::Hand::Left).rotation,
        }),
        right_hand: Some(avatar::Transform {
            position: rig.hand_grip(space_soup_engine::Hand::Right).position,
            rotation: rig.hand_grip(space_soup_engine::Hand::Right).rotation,
        }),
    };
    std::iter::once((local_player, local_state))
        .chain(remotes.iter().map(|(&id, &state)| (id, state)))
        .collect()
}

/// Per-player solver state: resolved skeleton + rig, the stateful body solver
/// (calibration, yaw smoothing, gait), velocity tracking and hand-pose
/// smoothing.
pub(crate) struct AvatarSolveState {
    skel: avatar_pose::Skeleton,
    rig: avatar_pose::Rig,
    /// The model's head joint height over its full height (both above its
    /// own floor): the player's height (head to toe) times this is what
    /// the solver sizes the body from.
    head_ratio: f32,
    /// The model's own hand length (wrist to middle fingertip, bind pose),
    /// to size gloves to it.
    hand_len: f32,
    solver: avatar_pose::BodySolver,
    last_head: Option<(Vec3, Instant)>,
    velocity: Vec3,
    pose_smooth: [avatar_pose::HandPose; 2],
}

/// Authored `avatar_ik::HandPose` (spread/twist per finger) approximated down
/// to the five curls the body solver animates.
fn curls_from_authored(p: &avatar_ik::HandPose, max_deg: f32) -> avatar_pose::HandPose {
    let f = |fp: &avatar_ik::FingerPose| {
        ((fp.flex[0] + fp.flex[1] + fp.flex[2]) / 3.0 / max_deg.max(1.0)).clamp(0.0, 1.0)
    };
    avatar_pose::HandPose {
        curls: [
            f(&p.thumb),
            f(&p.index),
            f(&p.middle),
            f(&p.ring),
            f(&p.little),
        ],
        // `p` already carries real spread/twist (an authored grip point, or
        // a HandPull's posed grip) -- passed through rather than discarded,
        // now that the solver has somewhere to put them.
        spread_deg: [p.thumb.spread, p.index.spread, p.middle.spread, p.ring.spread, p.little.spread],
        thumb_twist_deg: p.thumb.twist,
        joints: [
            [[0.0; 3]; 3],
            [[0.0, 0.0, p.index.twist], [0.0; 3], [0.0; 3]],
            [[0.0, 0.0, p.middle.twist], [0.0; 3], [0.0; 3]],
            [[0.0, 0.0, p.ring.twist], [0.0; 3], [0.0; 3]],
            [[0.0, 0.0, p.little.twist], [0.0; 3], [0.0; 3]],
        ],
    }
}

/// Grip-point curl map ("thumb" → 0..1, …) with the squeeze as fallback.
fn curls_from_map(m: &HashMap<String, f32>, squeeze: f32) -> avatar_pose::HandPose {
    let g = |k: &str, d: f32| m.get(k).copied().unwrap_or(d);
    avatar_pose::HandPose {
        curls: [
            g("thumb", squeeze * 0.8),
            g("index", squeeze),
            g("middle", squeeze),
            g("ring", squeeze),
            g("little", squeeze),
        ],
        // This map is the legacy `finger_curl` fallback (no `HandPose` was
        // authored for this grip) -- no spread/twist to carry.
        ..Default::default()
    }
}

/// Free hand: trigger curls the index, squeeze curls the rest, a touched
/// face/stick rests the thumb on the controller.
fn curls_free(
    trigger: f32,
    squeeze: f32,
    touching: bool,
    hv: &hand_menu_core::HandValues,
) -> avatar_pose::HandPose {
    let rest = hv.rest;
    let thumb = if touching {
        hv.thumb_touch.max(rest[0])
    } else {
        rest[0]
    }
    .max(squeeze * 0.8);
    avatar_pose::HandPose {
        curls: [
            thumb,
            rest[1].max(trigger),
            rest[2].max(squeeze),
            rest[3].max(squeeze),
            rest[4].max(squeeze),
        ],
        // The live calibration menu's own direction controls.
        spread_deg: hv.finger_spread,
        thumb_twist_deg: hv.thumb_twist,
        ..Default::default()
    }
}

/// A prop's say over one of the local player's hands this frame: where the
/// solver should put it (render space, as solver input -- already through
/// the hand's tuning) and its fingers.
#[derive(Clone, Copy, Default)]
pub(crate) struct HandOverride {
    pub input: Option<(Vec3, Quat)>,
    pub pose: Option<avatar_pose::HandPose>,
}

/// The local player's body as solved this frame (render space): wrists as
/// hand frames [left, right], and the feet and facing.
#[derive(Clone, Copy)]
pub(crate) struct LocalBody {
    pub wrists: [(Vec3, Quat); 2],
    pub root_pos: Vec3,
    pub root_yaw: f32,
}

/// VR gloves worn over the local player's own hands [left, right], posed
/// with the same finger curls. `shown` is set each frame they're drawn.
pub(crate) struct Gloves {
    pub parts: [(GltfMesh, ModelUniform, avatar_pose::GloveRig); 2],
    pub shown: bool,
}

/// Joints of both hands (wrists and fingers).
fn hand_joints(skel: &avatar_pose::Skeleton, rig: &avatar_pose::Rig) -> Vec<usize> {
    let mut j = skel.joint_and_descendants(rig.left_arm.wrist);
    j.extend(skel.joint_and_descendants(rig.right_arm.wrist));
    j
}

pub(crate) fn update_avatar_bodies(
    renderer: &mut XrRenderer,
    avatar_mesh_cache: &mut HashMap<PlayerId, (GltfMesh, ModelUniform)>,
    solve_states: &mut HashMap<PlayerId, AvatarSolveState>,
    // The local player's chosen body, and the body everyone else is drawn
    // with (other players' choices aren't sent over the network yet).
    local_master_mesh: &Option<GltfMesh>,
    avatar_master_mesh: &Option<GltfMesh>,
    gloves: &mut Option<Gloves>,
    local_direct_mesh: &mut Option<(GltfMesh, ModelUniform)>,
    local_player: PlayerId,
    hands: &hand_menu_core::MenuValues,
    offset: Vec3,
    yaw_inv: Quat,
    world: &Option<WireWorld>,
    cs: &ControllerState,
    bodies: &[(PlayerId, avatar::RemotePlayerState)],
    // Local player's hands that are mid-pull, already resolved onto the part
    // they are pulling. [left, right]; None means the hand is drawn as tracked.
    pull_hands: &[Option<crate::part_pull::PullHandPose>; 2],
    // Props holding the local player's hands [left, right].
    prop_hands: &[HandOverride; 2],
    // Out: the local player's posed wrist world transform per hand [L, R] in
    // render space — held objects attach to this exact bone.
    local_hand_world: &mut [Option<avatar_ik::Transform>; 2],
    // Out: the local player's solved body.
    local_body: &mut Option<LocalBody>,
) {
    avatar_mesh_cache.retain(|id, _| bodies.iter().any(|(bid, _)| *bid == *id));
    solve_states.retain(|id, _| bodies.iter().any(|(bid, _)| *bid == *id));

    for (id, state) in bodies.iter().copied() {
        // --- instantiate this player's mesh + solver state once ---
        if !avatar_mesh_cache.contains_key(&id) {
            let master = if id == local_player { local_master_mesh } else { avatar_master_mesh };
            let Some(master) = master else { continue };
            let Some(skin) = master.skin.as_ref() else {
                log::warn!("avatar mesh has no skin — bodies need a rigged mesh");
                continue;
            };
            let skel = avatar_pose::Skeleton::from_skin(
                skin.joint_names.clone(),
                skin.joint_parents.clone(),
                skin.inv_bind_mats.clone(),
            );
            match avatar_pose::Rig::resolve(&skel) {
                Ok(rig) => {
                    // The model's own floor, height and hand length, from its
                    // mesh (some models stand on their origin, others are
                    // rooted at the hips).
                    let (floor, top) = rig
                        .bind_height_range(skin.primitives.iter().flat_map(|p| p.vertices.iter().map(|v| Vec3::from(v.position))))
                        .unwrap_or((0.0, 1.8));
                    let rig = rig.with_floor(floor);
                    let bw = skel.bind_world();
                    let at = |j: usize| bw[j].transform_point3(Vec3::ZERO);
                    let head_h = (rig.canon * at(rig.head)).y - floor;
                    let head_ratio = (head_h / (top - floor).max(0.5)).clamp(0.6, 0.95);
                    let hand_len = rig.right_fingers[2]
                        .iter()
                        .rev()
                        .find_map(|j| j.map(|j| (at(j) - at(rig.right_arm.wrist)).length()))
                        .unwrap_or(0.15);
                    // Wearing gloves: the body's own hands are left out.
                    let mut mesh = if id == local_player && gloves.is_some() {
                        master.clone_with_independent_skin_excluding_joints(renderer.device(), &hand_joints(&skel, &rig))
                    } else {
                        master.clone_with_independent_skin(renderer.device())
                    };
                    mesh.create_skin_bind_group(renderer.device(), renderer.skin_joint_layout());
                    let model_uniform = renderer.create_skinned_model_uniform();
                    solve_states.insert(
                        id,
                        AvatarSolveState {
                            skel,
                            rig,
                            head_ratio,
                            hand_len,
                            solver: avatar_pose::BodySolver::default(),
                            last_head: None,
                            velocity: Vec3::ZERO,
                            pose_smooth: [avatar_pose::HandPose::default(); 2],
                        },
                    );
                    avatar_mesh_cache.insert(id, (mesh, model_uniform));
                }
                Err(e) => {
                    log::error!("avatar rig unusable: {e}");
                    continue;
                }
            }
        }
        let Some(st) = solve_states.get_mut(&id) else { continue };
        let (mesh, _) = avatar_mesh_cache.get_mut(&id).expect("inserted with state");
        mesh.position = Vec3::ZERO;
        mesh.rotation = Quat::IDENTITY;
        mesh.scale = Vec3::ONE;

        // --- tracking into render space ---
        let to_render = |p: Vec3| yaw_inv * (p - offset);
        let head_pos = to_render(state.head.position);
        let head_rot = yaw_inv * state.head.rotation;

        // A hand pulling an authored grip is drawn on that grip instead of on
        // the controller, so it stays wrapped around the handle as the part
        // travels. Only the local player has pull sessions.
        let posed = |h: avatar::Transform, idx: usize| -> (Vec3, Quat) {
            let to_hand = grip_to_hand(idx == 0);
            match pull_hands[idx].as_ref().filter(|_| id == local_player) {
                Some(p) => (to_render(p.position), yaw_inv * p.rotation * to_hand),
                None => (to_render(h.position), yaw_inv * h.rotation * to_hand),
            }
        };
        let mut left_hand = state.left_hand.map(|h| posed(h, 0));
        let mut right_hand = state.right_hand.map(|h| posed(h, 1));
        if id == local_player {
            if let Some(w) = prop_hands[0].input.filter(|_| left_hand.is_some()) {
                left_hand = Some(w);
            }
            if let Some(w) = prop_hands[1].input.filter(|_| right_hand.is_some()) {
                right_hand = Some(w);
            }
        }

        // --- finger curls ---
        let (hv_l, hv_r) = (hands.hand(true), hands.hand(false));
        let (left_target, right_target) = if id == local_player {
            let held_l = world.as_ref().and_then(|w| w.left_hand_held.as_ref());
            let held_r = world.as_ref().and_then(|w| w.right_hand_held.as_ref());
            let l = match (held_l, pull_hands[0].as_ref()) {
                (Some(held), _) => held
                    .hand_pose
                    .map(|p| curls_from_authored(&p, hv_l.curl_max_deg))
                    .unwrap_or_else(|| curls_from_map(&held.finger_curl, cs.l_squeeze)),
                (None, Some(pull)) => curls_from_authored(&pull.hand_pose, hv_l.curl_max_deg),
                (None, None) => curls_free(
                    cs.l_trigger,
                    cs.l_squeeze,
                    cs.l_stick_touch || cs.l_trigger_touch || cs.l_face_touch,
                    hv_l,
                ),
            };
            let r = match (held_r, pull_hands[1].as_ref()) {
                (Some(held), _) => held
                    .hand_pose
                    .map(|p| curls_from_authored(&p, hv_r.curl_max_deg))
                    .unwrap_or_else(|| curls_from_map(&held.finger_curl, cs.r_squeeze)),
                (None, Some(pull)) => curls_from_authored(&pull.hand_pose, hv_r.curl_max_deg),
                (None, None) => curls_free(
                    cs.r_trigger,
                    cs.r_squeeze,
                    cs.r_stick_touch || cs.r_trigger_touch || cs.r_face_touch,
                    hv_r,
                ),
            };
            (prop_hands[0].pose.unwrap_or(l), prop_hands[1].pose.unwrap_or(r))
        } else {
            (avatar_pose::HandPose::default(), avatar_pose::HandPose::default())
        };
        // Blend toward this frame's target so grips read as motion, not swaps.
        const SMOOTH: f32 = 0.4;
        st.pose_smooth[0] = avatar_pose::HandPose::blend(st.pose_smooth[0], left_target, SMOOTH);
        st.pose_smooth[1] = avatar_pose::HandPose::blend(st.pose_smooth[1], right_target, SMOOTH);

        // --- velocity from head motion (drives the walk cycle) ---
        //
        // From WORLD-space head position, not the render-space `head_pos`
        // above. Render space is deliberately player-centric -- the point of
        // `to_render` is that the camera itself never appears to move, so
        // thumbstick/smooth locomotion (which moves `player_offset`, not the
        // physical head) cancels out of it exactly: walking with the stick
        // produced zero velocity here regardless of speed, so the legs never
        // animated at all except from genuine real-world head movement
        // (roomscale walking). World space has no such cancellation -- it
        // carries both.
        let now = Instant::now();
        let head_pos_world = state.head.position;
        if let Some((last_pos, last_t)) = st.last_head {
            let dt = (now - last_t).as_secs_f32().clamp(0.001, 0.25);
            // Rotate the world-space delta into this frame's render-space
            // orientation (same convention PoseInput.velocity is otherwise
            // used in, via head_pos/head_rot above) without re-subtracting
            // `offset`, which is what cancelled thumbstick movement out in
            // the first place.
            let v = yaw_inv * (head_pos_world - last_pos) / dt;
            // A snap-turn (the default turn mode) changes `player_yaw`
            // instantly, and world-space head position is
            // `player_offset + yaw_rot * tracked_head` -- rotating that
            // second term by a sudden 45 degrees moves `head_pos_world` even
            // though the player didn't physically move at all, which this
            // frame's position delta can't tell apart from real motion.
            // Over one frame that reads as tens of m/s, nothing a real human
            // (or this game's fixed 1.6 m/s thumbstick speed) ever produces,
            // so clamping well above real movement but far below a turn
            // spike suppresses it without touching genuine walking. This is
            // what made the legs "helicopter" the instant a turn landed on
            // the same frame as the gait's cadence went distance-locked
            // (uncapped) instead of frequency-clamped.
            const MAX_REALISTIC_SPEED: f32 = 2.5;
            let v = v.clamp_length_max(MAX_REALISTIC_SPEED);
            let k = 0.3;
            st.velocity += (Vec3::new(v.x, 0.0, v.z) - st.velocity) * k;
        }
        let dt = st
            .last_head
            .map(|(_, t)| (now - t).as_secs_f32())
            .unwrap_or(1.0 / 72.0);
        st.last_head = Some((head_pos_world, now));

        // --- solve + upload ---
        // The menu's height is head to toe; the solver sizes the body from
        // the head joint, which sits at this model's own share of it.
        let (lt, rt, height_override_m) = if id == local_player {
            (hv_l.tuning(), hv_r.tuning(), Some(hands.height_cm / 100.0 * st.head_ratio))
        } else {
            (Default::default(), Default::default(), None)
        };
        let input = avatar_pose::PoseInput {
            head_pos,
            head_rot,
            left_hand,
            right_hand,
            left_pose: st.pose_smooth[0],
            right_pose: st.pose_smooth[1],
            velocity: st.velocity,
            ground_y: 0.0,
            left_tuning: lt,
            right_tuning: rt,
            height_override_m,
        };
        let body = st.solver.solve(&st.skel, &st.rig, &input, dt);
        mesh.update_joint_matrices(renderer.queue(), &body.skin_mats);

        if id == local_player {
            *local_body = Some(LocalBody { wrists: body.wrists, root_pos: body.root_pos, root_yaw: body.root_yaw });
            // Exact posed wrist bones for held-object attachment (the
            // editor authors hand offsets against the bone, so the solver's
            // hand frames are turned back into it).
            *local_hand_world = [
                Some(avatar_ik::Transform {
                    position: body.wrists[0].0,
                    rotation: body.wrists[0].1 * st.rig.hand_fix[0],
                }),
                Some(avatar_ik::Transform {
                    position: body.wrists[1].0,
                    rotation: body.wrists[1].1 * st.rig.hand_fix[1],
                }),
            ];

            // The local player sees their own body from the shoulders down
            // -- just the arms and hands (and no hands when wearing gloves);
            // everyone else (and the mirror) gets this player's full mesh.
            // Gloves are only loaded while worn (until they have, the body's
            // own hands show).
            let gloved = gloves.is_some();
            let direct = local_direct_mesh.get_or_insert_with(|| {
                let mut arms = Vec::new();
                for arm in [&st.rig.left_arm, &st.rig.right_arm] {
                    let shoulder = st.skel.joint_parents[arm.upper].unwrap_or(arm.upper);
                    arms.extend(st.skel.joint_and_descendants(shoulder));
                }
                let hand_set = if gloved { hand_joints(&st.skel, &st.rig) } else { Vec::new() };
                let hidden: Vec<usize> =
                    (0..st.skel.joint_names.len()).filter(|j| !arms.contains(j) || hand_set.contains(j)).collect();
                let mut direct_mesh =
                    mesh.clone_with_independent_skin_excluding_joints(renderer.device(), &hidden);
                direct_mesh.create_skin_bind_group(renderer.device(), renderer.skin_joint_layout());
                (direct_mesh, renderer.create_skinned_model_uniform())
            });
            direct.0.position = Vec3::ZERO;
            direct.0.rotation = Quat::IDENTITY;
            direct.0.scale = Vec3::ONE;
            direct.0.update_joint_matrices(renderer.queue(), &body.skin_mats);

            // Gloves ride the solved wrists, sized to this body's own hand
            // and bent with the same finger poses.
            if let Some(g) = gloves.as_mut() {
                g.shown = gloved;
                if gloved {
                    for (i, (gmesh, _, grig)) in g.parts.iter_mut().enumerate() {
                        let scale = grig.span.map_or(1.0, |s| st.hand_len * body.root_scale / s.max(1e-4));
                        let place = grig.place(body.wrists[i], scale);
                        gmesh.position = Vec3::ZERO;
                        gmesh.rotation = Quat::IDENTITY;
                        gmesh.scale = Vec3::ONE;
                        gmesh.update_joint_matrices(renderer.queue(), &grig.skin(place, &st.pose_smooth[i]));
                    }
                }
            }
        }
    }
}
