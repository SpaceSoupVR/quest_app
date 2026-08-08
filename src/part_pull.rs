#![cfg(target_os = "android")]

use glam::{Mat4, Quat, Vec3};
use log::{info, warn};
use std::collections::HashMap;

use space_soup::renderer::GltfMesh;
use space_soup::ControllerState;
use space_soup_engine::{ButtonPress, Hand, InputAxes, InputFrame, PartDriver, PlayerRig};
use space_soup_protocol::{WireRenderMesh, WireWorld};

use crate::grab_detect;
use crate::grab_detect::LiveObjects;

pub(crate) fn part_animation_blend(driver: PartDriver, hand: Hand, cs: &ControllerState) -> f32 {
    match (driver, hand) {
        (PartDriver::HoldTrigger, Hand::Left) => cs.l_trigger,
        (PartDriver::HoldTrigger, Hand::Right) => cs.r_trigger,
        (PartDriver::HoldGrip, Hand::Left) => cs.l_squeeze,
        (PartDriver::HoldGrip, Hand::Right) => cs.r_squeeze,
        (PartDriver::HandPull | PartDriver::Manual, _) => 0.0,
    }
}

/// Analog pull at which trigger/grip counts as a button edge. Scripts wanting a
/// different break point read the raw value with get_trigger()/get_grip().
pub(crate) const BUTTON_THRESHOLD: f32 = 0.5;

pub(crate) const PART_PULL_GRAB_RANGE: f32 = 0.09;

/// The blend each of an object's part-animation clips sits at this frame.
///
/// One implementation, used twice: render_prep poses the skeleton with it, and
/// handle_input reports it upward so the engine can evaluate blend-threshold
/// triggers. Computing it twice would let the pose the player sees and the
/// trigger that fires disagree, which is the kind of desync nobody would think
/// to look for.
///
/// The client owns this because it is the only side that can compute it -- a
/// HandPull blend comes from where the hand is relative to the posed part.
pub(crate) fn blends_for_object(
    object_id: &str,
    parts: &[space_soup_engine::PartAnimationDef],
    hand: Hand,
    cs: &ControllerState,
    pull_sessions: &[Option<PullSession>; 2],
    manual: Option<&HashMap<String, f32>>,
) -> HashMap<String, f32> {
    parts
        .iter()
        .map(|pa| {
            let raw = match pa.driver {
                PartDriver::HandPull => pull_sessions
                    .iter()
                    .flatten()
                    .find(|s| s.object_id == object_id && s.clip == pa.clip)
                    .map(|s| s.blend)
                    .unwrap_or(0.0),
                PartDriver::Manual => {
                    manual.and_then(|m| m.get(&pa.clip).copied()).unwrap_or(0.0)
                }
                _ => part_animation_blend(pa.driver, hand, cs),
            };
            (pa.clip.clone(), pa.easing.apply(raw))
        })
        .collect()
}


pub(crate) fn hand_idx(hand: Hand) -> usize {
    match hand {
        Hand::Left => 0,
        Hand::Right => 1,
    }
}

pub(crate) struct PullSession {
    pub(crate) object_id: String,
    pub(crate) clip: String,
    pub(crate) grab_local: Vec3,
    pub(crate) axis_model: Vec3,
    pub(crate) travel: f32,
    pub(crate) b0: f32,
    pub(crate) blend: f32,
}

pub(crate) fn try_start_pull(
    object_id: &str,
    gun_world: Mat4,
    pull_hand_pos: Vec3,
    static_scene: &grab_detect::StaticScene,
    mesh_cache: &HashMap<String, (GltfMesh, space_soup::renderer::mesh_pipeline::ModelUniform)>,
) -> Option<PullSession> {
    let parts = static_scene.part_animations.get(object_id)?;
    let (mesh, _) = mesh_cache.get(object_id)?;
    let skin = mesh.skin.as_ref()?;
    let hand_local = gun_world.inverse().transform_point3(pull_hand_pos);
    for pa in parts.iter().filter(|pa| pa.driver == PartDriver::HandPull) {
        let Some(clip_idx) = skin.animation_index(&pa.clip) else {
            warn!("HandPull '{object_id}': clip '{}' not found in skin", pa.clip);
            continue;
        };
        let Some((anchor_model, axis_model, travel)) = skin.pull_geometry(clip_idx) else {
            warn!("HandPull '{object_id}': clip '{}' has no pull_geometry", pa.clip);
            continue;
        };
        let anchor_world = gun_world.transform_point3(anchor_model);
        let dist = pull_hand_pos.distance(anchor_world);
        info!(
            "HandPull '{object_id}' clip '{}': dist={dist:.3}m (need <={PART_PULL_GRAB_RANGE:.2}) anchor_world=({:.2},{:.2},{:.2}) travel={travel:.3}",
            pa.clip, anchor_world.x, anchor_world.y, anchor_world.z
        );
        if dist <= PART_PULL_GRAB_RANGE {
            return Some(PullSession {
                object_id: object_id.to_string(),
                clip: pa.clip.clone(),
                grab_local: hand_local,
                axis_model,
                travel,
                b0: 0.0,
                blend: 0.0,
            });
        }
    }
    None
}

// Grab/release/button-press detection for one frame -- reads the previous
// frame's trigger/squeeze/button state (to detect just-pressed/just-released
// edges) and updates it for the next frame. Also advances any in-progress
// HandPull sessions.
#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_input(
    cs: &ControllerState,
    rig: &PlayerRig,
    world: &Option<WireWorld>,
    meshes_src: &[WireRenderMesh],
    static_scene: &grab_detect::StaticScene,
    mesh_cache: &HashMap<String, (GltfMesh, space_soup::renderer::mesh_pipeline::ModelUniform)>,
    live_objects: &LiveObjects,
    pull_sessions: &mut [Option<PullSession>; 2],
    prev_r_trigger: &mut bool,
    prev_l_trigger: &mut bool,
    prev_r_squeeze: &mut bool,
    prev_l_squeeze: &mut bool,
    prev_btn_a: &mut bool,
    prev_btn_b: &mut bool,
    prev_btn_x: &mut bool,
    prev_btn_y: &mut bool,
) -> InputFrame {
    let mut input = InputFrame::default();

    let r_trigger_down = cs.r_trigger > 0.5 || cs.r_squeeze > 0.5;
    let l_trigger_down = cs.l_trigger > 0.5 || cs.l_squeeze > 0.5;
    let r_trigger_only = cs.r_trigger > 0.5;
    let l_trigger_only = cs.l_trigger > 0.5;

    let held_gun_world = |hand: Hand| -> Option<(String, Mat4)> {
        let held = world.as_ref().and_then(|w| match hand {
            Hand::Left => w.left_hand_held.as_ref(),
            Hand::Right => w.right_hand_held.as_ref(),
        })?;
        let hand_tf = rig.hand_grip(hand);
        let hand_mat = Mat4::from_rotation_translation(hand_tf.rotation, hand_tf.position);
        let offset_mat = Mat4::from_rotation_translation(
            Quat::from_array(held.point_local_rot),
            Vec3::from(held.point_local_pos),
        );
        let (_, rot, pos) = (hand_mat * offset_mat.inverse()).to_scale_rotation_translation();
        let scale = meshes_src
            .iter()
            .find(|m| m.id == held.object_id)
            .map(|m| Vec3::from(m.scale))
            .unwrap_or(Vec3::ONE);
        Some((held.object_id.clone(), Mat4::from_scale_rotation_translation(scale, rot, pos)))
    };

    if r_trigger_down && !(*prev_r_trigger || *prev_r_squeeze) {
        let p = rig.hand_grip(Hand::Right).position;
        let pull = held_gun_world(Hand::Left)
            .and_then(|(oid, gw)| try_start_pull(&oid, gw, p, static_scene, mesh_cache));
        if let Some(session) = pull {
            info!("GRAB R: started HandPull on '{}'", session.object_id);
            pull_sessions[hand_idx(Hand::Right)] = Some(session);
        } else if let Some((id, point)) =
            grab_detect::nearest_grip_point_to(live_objects, static_scene, p, r_trigger_only, Hand::Right)
        {
            info!("GRAB R: '{id}' via grip point '{point}'");
            input.grabbed.push((id, Hand::Right, point));
        } else if let Some(id) = grab_detect::nearest_object_to(live_objects, p) {
            info!("GRAB R: '{id}' via proximity (no grip point)");
            input.grabbed.push((id, Hand::Right, String::new()));
        } else {
            info!(
                "GRAB R: pressed, nothing in range (hand {:.2},{:.2},{:.2}; {} live objs)",
                p.x, p.y, p.z, live_objects.by_id.len()
            );
        }
    }
    if !r_trigger_down && (*prev_r_trigger || *prev_r_squeeze) {
        if pull_sessions[hand_idx(Hand::Right)].is_none() {
            if let Some(id) =
                grab_detect::nearest_object_to(live_objects, rig.hand_grip(Hand::Right).position)
            {
                input.released.push((id, Hand::Right));
            }
        }
    }
    if l_trigger_down && !(*prev_l_trigger || *prev_l_squeeze) {
        let p = rig.hand_grip(Hand::Left).position;
        let pull = held_gun_world(Hand::Right)
            .and_then(|(oid, gw)| try_start_pull(&oid, gw, p, static_scene, mesh_cache));
        if let Some(session) = pull {
            info!("GRAB L: started HandPull on '{}'", session.object_id);
            pull_sessions[hand_idx(Hand::Left)] = Some(session);
        } else if let Some((id, point)) =
            grab_detect::nearest_grip_point_to(live_objects, static_scene, p, l_trigger_only, Hand::Left)
        {
            info!("GRAB L: '{id}' via grip point '{point}'");
            input.grabbed.push((id, Hand::Left, point));
        } else if let Some(id) = grab_detect::nearest_object_to(live_objects, p) {
            info!("GRAB L: '{id}' via proximity (no grip point)");
            input.grabbed.push((id, Hand::Left, String::new()));
        } else {
            info!(
                "GRAB L: pressed, nothing in range (hand {:.2},{:.2},{:.2}; {} live objs)",
                p.x, p.y, p.z, live_objects.by_id.len()
            );
        }
    }
    if !l_trigger_down && (*prev_l_trigger || *prev_l_squeeze) {
        if pull_sessions[hand_idx(Hand::Left)].is_none() {
            if let Some(id) =
                grab_detect::nearest_object_to(live_objects, rig.hand_grip(Hand::Left).position)
            {
                input.released.push((id, Hand::Left));
            }
        }
    }

    for hand in [Hand::Left, Hand::Right] {
        let idx = hand_idx(hand);
        let Some(session) = pull_sessions[idx].as_mut() else {
            continue;
        };
        let still_pulling = match hand {
            Hand::Left => l_trigger_down,
            Hand::Right => r_trigger_down,
        };
        let gun = held_gun_world(Hand::Left)
            .filter(|(oid, _)| *oid == session.object_id)
            .or_else(|| held_gun_world(Hand::Right).filter(|(oid, _)| *oid == session.object_id));
        let Some((_, gun_world)) = gun else {
            pull_sessions[idx] = None;
            continue;
        };
        if !still_pulling {
            pull_sessions[idx] = None;
            continue;
        }
        let hand_local = gun_world.inverse().transform_point3(rig.hand_grip(hand).position);
        let pulled = (hand_local - session.grab_local).dot(session.axis_model) / session.travel;
        session.blend = (session.b0 + pulled).clamp(0.0, 1.0);
    }

    {
        let held_r = grab_detect::held_object_id(
            world.as_ref().and_then(|w| w.right_hand_held.as_ref()),
            live_objects,
            rig.hand_grip(Hand::Right).position,
        );
        let held_l = grab_detect::held_object_id(
            world.as_ref().and_then(|w| w.left_hand_held.as_ref()),
            live_objects,
            rig.hand_grip(Hand::Left).position,
        );
        // (button, is-down-now, was-down-last-frame, which hand, held object).
        // Both edges are derived from the same pair so a press and its release
        // can never disagree about which object or hand they belong to.
        let edges = [
            ("btn_a", cs.btn_a, *prev_btn_a, Hand::Right, held_r.clone()),
            ("btn_b", cs.btn_b, *prev_btn_b, Hand::Right, held_r.clone()),
            ("btn_x", cs.btn_x, *prev_btn_x, Hand::Left, held_l.clone()),
            ("btn_y", cs.btn_y, *prev_btn_y, Hand::Left, held_l.clone()),
            ("trigger", cs.r_trigger > BUTTON_THRESHOLD, *prev_r_trigger, Hand::Right, held_r.clone()),
            ("trigger", cs.l_trigger > BUTTON_THRESHOLD, *prev_l_trigger, Hand::Left, held_l.clone()),
            ("grip", cs.r_squeeze > BUTTON_THRESHOLD, *prev_r_squeeze, Hand::Right, held_r),
            ("grip", cs.l_squeeze > BUTTON_THRESHOLD, *prev_l_squeeze, Hand::Left, held_l),
        ];
        for (button, down, was_down, hand, object_id) in edges {
            if down && !was_down {
                input
                    .button_presses
                    .push(ButtonPress::new(button, object_id, hand));
            } else if !down && was_down {
                // The up edge. Without it a script can see a trigger pulled but has
                // nothing telling it the trigger was let go -- there is no
                // button-release event at all, and `on_release` means grab release.
                input
                    .button_releases
                    .push(ButtonPress::new(button, object_id, hand));
            }
        }
    }

    // Report every held object's part blends so the engine can evaluate
    // blend-threshold triggers. Those actions -- spawning a magazine, handing it
    // to physics -- are authoritative world state, so the decision cannot live on
    // a headset even though only a headset can compute the input to it.
    for hand in [Hand::Left, Hand::Right] {
        let held = world.as_ref().and_then(|w| match hand {
            Hand::Left => w.left_hand_held.as_ref(),
            Hand::Right => w.right_hand_held.as_ref(),
        });
        let Some(held) = held else { continue };
        let Some(parts) = static_scene.part_animations.get(&held.object_id) else { continue };
        let manual = meshes_src
            .iter()
            .find(|m| m.id == held.object_id)
            .map(|m| &m.manual_part_blends);
        let blends = blends_for_object(&held.object_id, parts, hand, cs, pull_sessions, manual);
        if !blends.is_empty() {
            input.part_blends.insert(held.object_id.clone(), blends);
        }
    }

    // Continuous values, every frame, so a script can poll what an edge cannot say:
    // how hard, and still held.
    input.axes = InputAxes {
        l_trigger: cs.l_trigger,
        r_trigger: cs.r_trigger,
        l_grip: cs.l_squeeze,
        r_grip: cs.r_squeeze,
        l_stick: [cs.l_stick.x, cs.l_stick.y],
        r_stick: [cs.r_stick.x, cs.r_stick.y],
    };

    *prev_r_trigger = cs.r_trigger > 0.5;
    *prev_l_trigger = cs.l_trigger > 0.5;
    *prev_r_squeeze = cs.r_squeeze > 0.5;
    *prev_l_squeeze = cs.l_squeeze > 0.5;
    *prev_btn_a = cs.btn_a;
    *prev_btn_b = cs.btn_b;
    *prev_btn_x = cs.btn_x;
    *prev_btn_y = cs.btn_y;

    input
}
