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

        let mut rig_cfg = *rig_config;
        let up = avatar_ik::detect_up_axis(skeleton);
        rig_cfg.up_axis = up.to_array();
        let raw_bind_head_height = avatar_ik::bind_head_height_along(skeleton, up);
        let calibrated_height = calibrated_heights
            .entry(id)
            .or_default()
            .observe(state.head.position.y);
        let root_scale = avatar::height_calibrated_scale(calibrated_height, raw_bind_head_height);

        let to_render = |p: Vec3| yaw_inv * (p - offset);
        let floor_drop = raw_bind_head_height * root_scale;
        let head_rot = yaw_inv * state.head.rotation;
        let root = avatar_ik::body_root_transform_basis(
            avatar::Transform {
                position: to_render(state.head.position),
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
            let direct = local_direct_mesh.get_or_insert_with(|| {
                let hidden_joints = avatar_ik::head_and_descendant_joints(skeleton);
                let mut direct_mesh = mesh.clone_with_independent_skin_excluding_joints(
                    renderer.device(),
                    &hidden_joints,
                );
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
