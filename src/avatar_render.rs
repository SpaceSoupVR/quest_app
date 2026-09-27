#![cfg(target_os = "android")]

use std::collections::HashMap;

use glam::{Quat, Vec3};

use space_soup::renderer::xr_renderer::XrRenderer;
use space_soup::renderer::{mesh_pipeline::ModelUniform, GltfMesh};
use space_soup::ControllerState;
use space_soup_protocol::{PlayerId, WireWorld};

use crate::avatar;

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

#[allow(clippy::too_many_arguments)]
pub(crate) fn update_avatar_bodies(
    renderer: &mut XrRenderer,
    avatar_mesh_cache: &mut HashMap<PlayerId, (GltfMesh, ModelUniform)>,
    avatar_skeleton_cache: &mut HashMap<PlayerId, avatar_ik::SkeletonData>,
    avatar_master_mesh: &Option<GltfMesh>,
    local_direct_mesh: &mut Option<(GltfMesh, ModelUniform)>,
    local_player: PlayerId,
    rig_config: &avatar_ik::RigConfig,
    calibrated_heights: &mut HashMap<PlayerId, avatar_ik::HeightCalibrator>,
    offset: Vec3,
    yaw_inv: Quat,
    world: &Option<WireWorld>,
    cs: &ControllerState,
    bodies: &[(PlayerId, avatar::RemotePlayerState)],
    // Local player's hands that are mid-pull, already resolved onto the part
    // they are pulling. [left, right]; None means the hand is drawn as tracked.
    pull_hands: &[Option<crate::part_pull::PullHandPose>; 2],
    // Out: the local player's posed wrist-joint world transform per hand [Left, Right],
    // in render space. A held object attaches to this exact bone (object = wrist *
    // hand_offset^-1), reproducing the editor's authored pose with no reconstruction --
    // and inheriting the arm IK's reach clamp for free.
    local_hand_world: &mut [Option<avatar_ik::Transform>; 2],
) {
    avatar_mesh_cache.retain(|id, _| bodies.iter().any(|(bid, _)| *bid == *id));
    avatar_skeleton_cache.retain(|id, _| bodies.iter().any(|(bid, _)| *bid == *id));

    for (id, state) in bodies.iter().copied() {
        if !avatar_mesh_cache.contains_key(&id) {
            let Some(master) = avatar_master_mesh else { continue };
            let mut mesh = master.clone_with_independent_skin(renderer.device());
            if mesh.is_skinned() {
                mesh.create_skin_bind_group(renderer.device(), renderer.skin_joint_layout());
                let model_uniform = renderer.create_skinned_model_uniform();
                if let Some(skin) = &mesh.skin {
                    avatar_skeleton_cache.insert(id, avatar::skeleton_data_from_skin(skin));
                }
                avatar_mesh_cache.insert(id, (mesh, model_uniform));
            } else {
                log::warn!("'models/boy/boy.glb' has no skin — avatar bodies need a rigged mesh");
                let model_uniform = renderer.create_model_uniform();
                avatar_mesh_cache.insert(id, (mesh, model_uniform));
            }
        }
        let (mesh, _) = avatar_mesh_cache.get_mut(&id).expect("just inserted above");
        mesh.position = Vec3::ZERO;
        mesh.rotation = Quat::IDENTITY;
        mesh.scale = Vec3::ONE;
        let Some(skin) = &mesh.skin else { continue };
        let Some(skeleton) = avatar_skeleton_cache.get(&id) else { continue };

        let mut rig_cfg = rig_config.clone();
        let up = avatar_ik::detect_up_axis(skeleton);
        rig_cfg.up_axis = up.to_array();
        let raw_bind_head_height = avatar_ik::bind_head_height_along(skeleton, up);
        // The headset reports where the EYES are, not where the rig's Head
        // joint is, and on a humanoid those are 5-10 cm apart -- the Head joint
        // sits at the base of the skull, below and behind the eyes. Calibrating
        // against the head height divided an eye height by a head height, which
        // scaled every avatar slightly too tall; anchoring the Head joint at
        // the headset pose then lifted the whole body until the wearer's own
        // torso rose into their view.
        let raw_bind_eye_height = avatar_ik::bind_eye_height_along(skeleton, up);
        let calibrated_height = calibrated_heights
            .entry(id)
            .or_default()
            .observe(state.head.position.y);
        let root_scale = avatar::height_calibrated_scale(calibrated_height, raw_bind_eye_height);

        let to_render = |p: Vec3| yaw_inv * (p - offset);
        // Still the HEAD height: the drop runs from the head joint to the
        // floor, and pairing it with the eye-derived scale is what puts the
        // eyes at the headset while leaving the feet on the ground.
        let floor_drop = raw_bind_head_height * root_scale;
        let head_rot = yaw_inv * state.head.rotation;
        // Place the head joint one scaled offset BEHIND and BELOW the eyes.
        //
        // Through `world_eye_offset` rather than by rotating the raw offset:
        // that offset is in MODEL space, and boy.glb is Z-up, so multiplying it
        // by a world-space head rotation sent the correction sideways instead
        // of down. It still moved the avatar, which is why it read as a fix
        // that had not gone far enough rather than as one aimed the wrong way.
        // One-shot diagnostic. Three attempts at this have been wrong because
        // each rested on a guess about the rig; these are the numbers the code
        // actually computes, logged once so the log is readable.
        if id == local_player {
            use std::sync::atomic::{AtomicBool, Ordering};
            static LOGGED: AtomicBool = AtomicBool::new(false);
            if !LOGGED.swap(true, Ordering::Relaxed) {
                let raw = avatar_ik::eye_to_head_offset(skeleton);
                log::info!(
                    "AVATARDIAG up={up:?} bind_head={raw_bind_head_height:.3} \
                     bind_eye={raw_bind_eye_height:.3} eye_to_head={raw:?} \
                     calibrated_h={calibrated_height:.3} root_scale={root_scale:.5} \
                     stature={:.3} headset_y={:.3}",
                    skeleton.bind_stature,
                    state.head.position.y,
                );
            }
        }

        let eye_offset = avatar_ik::world_eye_offset(
            skeleton, up, rig_cfg.forward(), head_rot, Vec3::Y,
        );
        let head_pos = to_render(state.head.position) - eye_offset * root_scale;
        if id == local_player {
            use std::sync::atomic::{AtomicBool, Ordering};
            static LOGGED2: AtomicBool = AtomicBool::new(false);
            if !LOGGED2.swap(true, Ordering::Relaxed) {
                log::info!(
                    "AVATARDIAG eye_offset={eye_offset:?} head_pos={head_pos:?} \
                     floor_drop={floor_drop:.3}",
                );
            }
        }
        let root = avatar_ik::body_root_transform_basis(
            avatar::Transform {
                position: head_pos,
                rotation: head_rot,
            },
            floor_drop,
            up,
            rig_cfg.forward(),
        );

        // A hand pulling an authored grip is drawn on that grip instead of on the
        // controller, so it stays wrapped around the handle as the part travels.
        // Only the local player has pull sessions; remote hands come over the wire
        // already posed.
        let posed = |h: avatar::Transform, idx: usize| -> avatar::Transform {
            match pull_hands[idx].as_ref().filter(|_| id == local_player) {
                Some(p) => avatar::Transform {
                    position: to_render(p.position),
                    rotation: yaw_inv * p.rotation,
                },
                None => avatar::Transform {
                    position: to_render(h.position),
                    rotation: yaw_inv * h.rotation,
                },
            }
        };
        let left_hand = state.left_hand.map(|h| posed(h, 0));
        let right_hand = state.right_hand.map(|h| posed(h, 1));

        let (left_curl, right_curl) = if id == local_player {
            let held_l = world.as_ref().and_then(|w| w.left_hand_held.as_ref());
            let held_r = world.as_ref().and_then(|w| w.right_hand_held.as_ref());
            // The authored pose when the server sent one -- it carries spread and
            // twist, which a curl map has no axis for. The curl is the fallback
            // for a server older than that field, and for a hand holding nothing.
            let max = rig_config.finger_curl_max_deg;
            let l = match (held_l, pull_hands[0].as_ref()) {
                (Some(held), _) => held.hand_pose.unwrap_or_else(|| {
                    avatar::HandPose::from_curl(
                        avatar::HandCurl::from_finger_curl(&held.finger_curl, cs.l_squeeze),
                        max,
                    )
                }),
                (None, Some(pull)) => pull.hand_pose,
                (None, None) => avatar::HandPose::from_curl(
                    avatar::HandCurl::free_hand(
                        cs.l_trigger,
                        cs.l_squeeze,
                        cs.l_stick_touch,
                        rig_config.thumb_touch_curl,
                    ),
                    max,
                ),
            };
            let r = match (held_r, pull_hands[1].as_ref()) {
                (Some(held), _) => held.hand_pose.unwrap_or_else(|| {
                    avatar::HandPose::from_curl(
                        avatar::HandCurl::from_finger_curl(&held.finger_curl, cs.r_squeeze),
                        max,
                    )
                }),
                (None, Some(pull)) => pull.hand_pose,
                (None, None) => avatar::HandPose::from_curl(
                    avatar::HandCurl::free_hand(
                        cs.r_trigger,
                        cs.r_squeeze,
                        cs.r_stick_touch,
                        rig_config.thumb_touch_curl,
                    ),
                    max,
                ),
            };
            (Some(l), Some(r))
        } else {
            (None, None)
        };

        let skinned_mats = avatar_ik::body_skin_matrices(
            skeleton,
            &rig_cfg,
            root.position,
            root.rotation,
            head_rot,
            root_scale,
            left_hand,
            right_hand,
            left_curl,
            right_curl,
        );
        skin.update_joint_matrices(renderer.queue(), &skinned_mats);

        if id == local_player {
            // The exact posed wrist a held object attaches to (same solve, render space).
            *local_hand_world = avatar_ik::posed_hand_worlds(
                skeleton,
                &rig_cfg,
                root.position,
                root.rotation,
                head_rot,
                root_scale,
                left_hand,
                right_hand,
                left_curl,
                right_curl,
            );
        }

        if id == local_player {
            let direct = local_direct_mesh.get_or_insert_with(|| {
                let hidden_joints = avatar_ik::head_and_descendant_joints(skeleton);
                // Joints AND a height. Hiding Head, its descendants and Neck
                // still leaves the top of the neck behind, because those
                // vertices are weighted to Chest -- which has to stay, since it
                // is the torso the wearer looks down at. The cutoff removes the
                // open tube of throat that was left sitting at eye level.
                let cutoff = avatar_ik::first_person_cutoff_height(skeleton, up);
                let mut direct_mesh = mesh.clone_with_independent_skin_excluding(
                    renderer.device(),
                    &hidden_joints,
                    up,
                    cutoff,
                );
                if id == local_player {
                    log::info!(
                        "AVATARDIAG first-person cutoff along {up:?} = {cutoff:?} (neck joint)",
                    );
                }
                direct_mesh.create_skin_bind_group(renderer.device(), renderer.skin_joint_layout());
                (direct_mesh, renderer.create_skinned_model_uniform())
            });
            direct.0.position = Vec3::ZERO;
            direct.0.rotation = Quat::IDENTITY;
            direct.0.scale = Vec3::ONE;
            if let Some(direct_skin) = &direct.0.skin {
                let direct_mats = avatar_ik::body_skin_matrices(
                    skeleton,
                    &rig_cfg,
                    root.position,
                    root.rotation,
                    head_rot,
                    root_scale,
                    left_hand,
                    right_hand,
                    left_curl,
                    right_curl,
                );
                direct_skin.update_joint_matrices(renderer.queue(), &direct_mats);
            }
        }
    }
}
