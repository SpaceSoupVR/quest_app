#![cfg(target_os = "android")]

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};

use space_soup::renderer::mesh_pipeline::ModelUniform;
use space_soup::renderer::xr_renderer::XrRenderer;
use space_soup::renderer::{Cuboid, GltfMesh, Light, MeshInstance, MirrorSurface};
use space_soup::ControllerState;
use space_soup_engine::{Hand, PartDriver, PlayerRig};
use space_soup_protocol::{
    PlayerId, WireHeldGrip, WireRenderCuboid, WireRenderLight, WireRenderMesh, WireWorld,
};

use crate::convert::{to_space_soup_cuboid, to_space_soup_light};
use crate::grab_detect::StaticScene;
use crate::part_pull::{part_animation_blend, PullSession};

const MAX_RENDER_DIST: f32 = 40.0;

/// Updates mesh transforms in-place (world-driven meshes, plus the one
/// currently held per hand with its part-animation blend), then builds the
/// per-frame render lists that borrow from the mesh caches. The borrow is
/// why this can't just return owned data -- `MeshInstance` is a thin
/// (&GltfMesh, &ModelUniform) pair, not a copy of the mesh itself.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_render_lists<'a>(
    cuboids_src: &[WireRenderCuboid],
    lights_src: &[WireRenderLight],
    meshes_src: &'a [WireRenderMesh],
    mesh_cache: &'a mut HashMap<String, (GltfMesh, ModelUniform)>,
    avatar_mesh_cache: &'a HashMap<PlayerId, (GltfMesh, ModelUniform)>,
    local_direct_mesh: &'a Option<(GltfMesh, ModelUniform)>,
    local_player: PlayerId,
    world: &Option<WireWorld>,
    rig: &PlayerRig,
    static_scene: &StaticScene,
    pull_sessions: &[Option<PullSession>; 2],
    cs: &ControllerState,
    renderer: &XrRenderer,
    offset: Vec3,
    yaw_inv: Quat,
    head_pos: Vec3,
) -> (Vec<Cuboid>, Vec<Light>, Vec<MeshInstance<'a>>, Vec<MeshInstance<'a>>, Option<MirrorSurface>) {
    let cuboids: Vec<Cuboid> = cuboids_src
        .iter()
        .filter(|rc| Vec3::from(rc.position).distance(head_pos) < MAX_RENDER_DIST)
        .map(|rc| to_space_soup_cuboid(rc, offset, yaw_inv))
        .collect();

    let lights: Vec<Light> = lights_src
        .iter()
        .map(|rl| to_space_soup_light(rl, offset, yaw_inv))
        .collect();

    for rm in meshes_src {
        if let Some((mesh, _)) = mesh_cache.get_mut(&rm.id) {
            mesh.position = yaw_inv * (Vec3::from(rm.position) - offset);
            mesh.rotation = yaw_inv * Quat::from_array(rm.rotation);
            mesh.scale = Vec3::from(rm.scale);
        }
    }

    for hand in [Hand::Left, Hand::Right] {
        let held: Option<&WireHeldGrip> = world.as_ref().and_then(|w| match hand {
            Hand::Left => w.left_hand_held.as_ref(),
            Hand::Right => w.right_hand_held.as_ref(),
        });
        let Some(held) = held else { continue };
        let Some((mesh, _)) = mesh_cache.get_mut(&held.object_id) else {
            continue;
        };
        let hand_tf = rig.hand_grip(hand);
        let hand_mat = Mat4::from_rotation_translation(hand_tf.rotation, hand_tf.position);
        let offset_mat = Mat4::from_rotation_translation(
            Quat::from_array(held.point_local_rot),
            Vec3::from(held.point_local_pos),
        );
        let (_, rot, pos) = (hand_mat * offset_mat.inverse()).to_scale_rotation_translation();
        mesh.position = yaw_inv * (pos - offset);
        mesh.rotation = yaw_inv * rot;

        if let Some(parts) = static_scene.part_animations.get(&held.object_id) {
            if let Some(skin) = &mesh.skin {
                let manual_blends = meshes_src
                    .iter()
                    .find(|m| m.id == held.object_id)
                    .map(|m| &m.manual_part_blends);
                let targets: Vec<(usize, f32)> = parts
                    .iter()
                    .filter_map(|pa| {
                        let clip_idx = skin.animation_index(&pa.clip)?;
                        let raw = if pa.driver == PartDriver::HandPull {
                            pull_sessions
                                .iter()
                                .flatten()
                                .find(|s| s.object_id == held.object_id && s.clip == pa.clip)
                                .map(|s| s.blend)
                                .unwrap_or(0.0)
                        } else if pa.driver == PartDriver::Manual {
                            manual_blends
                                .and_then(|m| m.get(&pa.clip).copied())
                                .unwrap_or(0.0)
                        } else {
                            part_animation_blend(pa.driver, hand, cs)
                        };
                        Some((clip_idx, pa.easing.apply(raw)))
                    })
                    .collect();
                if !targets.is_empty() {
                    let mats = skin.skin_matrices_blended_multi(&targets);
                    mesh.update_joint_matrices(renderer.queue(), &mats);
                }
            }
        }
    }

    let mesh_instances: Vec<MeshInstance> = meshes_src
        .iter()
        .filter(|rm| Vec3::from(rm.position).distance(head_pos) < MAX_RENDER_DIST)
        .filter_map(|rm| {
            let (mesh, model) = mesh_cache.get(&rm.id)?;
            Some(MeshInstance { mesh, model, lightmap_key: Some(rm.id.as_str()) })
        })
        .chain(
            avatar_mesh_cache
                .iter()
                .filter(|(&id, _)| id != local_player)
                .map(|(_, (mesh, model))| MeshInstance { mesh, model, lightmap_key: None }),
        )
        .chain(
            local_direct_mesh
                .iter()
                .map(|(mesh, model)| MeshInstance { mesh, model, lightmap_key: None }),
        )
        .collect();

    let mirror_only_mesh_instances: Vec<MeshInstance> = avatar_mesh_cache
        .get(&local_player)
        .map(|(mesh, model)| MeshInstance { mesh, model, lightmap_key: None })
        .into_iter()
        .collect();

    let mirror_surface = cuboids_src.iter().find(|rc| rc.id == "mirror").map(|rc| {
        let rotation = yaw_inv * Quat::from_array(rc.rotation);
        let half_size = Vec3::from(rc.half_size);
        let normal = rotation * Vec3::NEG_Z;
        let position =
            yaw_inv * (Vec3::from(rc.position) - offset) + normal * (half_size.z + 0.005);
        MirrorSurface {
            position,
            rotation,
            half_size,
        }
    });

    (cuboids, lights, mesh_instances, mirror_only_mesh_instances, mirror_surface)
}
