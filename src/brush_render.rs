//! Brush geometry for the on-device renderer.
//!
//! Brushes are plane-set solids -- the level's walls, rooms and stairs. Before
//! this the headset never saw one: the server sent every brush object down the
//! CUBOID list, so a wall arrived as the box around it and a wall fractured into
//! twelve chunks arrived as twelve overlapping crates.
//!
//! Meshed on the client, from scene data that is already on the headset, for
//! exactly the reason terrain is: run.sh pushes game/ wholesale, so sending
//! triangles per snapshot would spend the budget that matters at 64 players on
//! something both ends already have. The engine's `brush_mesh` is the same code
//! the editor's parity test pins, so what the headset draws is what the author
//! saw.
//!
//! Built once per scene load. What changes at runtime is only WHICH brushes are
//! drawn -- a chunk shot out of a wall, a door a script hid -- and that arrives
//! as a list of ids in the snapshot.

use std::collections::HashSet;

use glam::{Quat, Vec3};
use space_soup::renderer::cuboid::SolidVertex;
use space_soup_engine::scene::Scene;

/// One brush object's triangles, in world space.
pub struct BrushObject {
    pub id: String,
    vertices: Vec<SolidVertex>,
    indices: Vec<u32>,
}

/// Every brush in the scene, plus the assembled buffer the renderer is handed.
#[derive(Default)]
pub struct BrushGeometry {
    objects: Vec<BrushObject>,
    /// The last assembly, kept so a frame that changed nothing costs nothing.
    vertices: Vec<SolidVertex>,
    indices: Vec<u32>,
    built_for: Option<(Vec3, f32, u64)>,
}

impl BrushGeometry {
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// Mesh every brush in a scene.
    ///
    /// A brush that produces no geometry is dropped with a warning rather than
    /// failing the load: one broken solid in a level is diagnosable from the
    /// log, and a client that refuses to start is not.
    pub fn load(scene: &Scene) -> Self {
        let mut objects = Vec::new();
        for obj in &scene.objects {
            let Some(def) = obj.brush.as_ref() else { continue };
            let groups = space_soup_engine::brush::brush_mesh(def);

            let colour = space_soup::renderer::Color3(
                obj.cuboid.color.0,
                obj.cuboid.color.1,
                obj.cuboid.color.2,
                obj.cuboid.color.3,
            )
            .to_linear();
            let reflectivity = obj.cuboid.reflectivity.clamp(0.0, 1.0);

            let mut vertices: Vec<SolidVertex> = Vec::new();
            let mut indices: Vec<u32> = Vec::new();
            for g in &groups {
                let base = vertices.len() as u32;
                for i in 0..g.positions.len() / 3 {
                    vertices.push(SolidVertex {
                        position: [g.positions[i * 3], g.positions[i * 3 + 1], g.positions[i * 3 + 2]],
                        normal: [g.normals[i * 3], g.normals[i * 3 + 1], g.normals[i * 3 + 2]],
                        // One colour for the whole object, not one per material.
                        // The face materials ARE carried through brush_mesh and
                        // are what a textured brush pipeline will key on; until
                        // that exists, tinting by material name would invent art
                        // direction, and the object's own authored colour is
                        // what these brushes were already being drawn in.
                        color: colour,
                        // The face's real texture uv. Nothing samples it yet --
                        // brushes have no lightmap, so the default white one is
                        // bound and any coordinate reads white -- but it is the
                        // coordinate a material pass needs and it is free here.
                        uv2: [g.uvs[i * 2], g.uvs[i * 2 + 1]],
                        reflectivity,
                    });
                }
                indices.extend(g.indices.iter().map(|i| i + base));
            }

            if vertices.is_empty() || indices.is_empty() {
                log::warn!("brush_render: '{}' produced no geometry", obj.id);
                continue;
            }
            objects.push(BrushObject { id: obj.id.clone(), vertices, indices });
        }

        let total: usize = objects.iter().map(|o| o.indices.len() / 3).sum();
        if !objects.is_empty() {
            log::info!("brush_render: {} brushes, {total} triangles", objects.len());
        }
        Self { objects, ..Default::default() }
    }

    /// The triangles to draw this frame, in the player's local space.
    ///
    /// Transformed rather than handed over in world space, because everything
    /// else in the solid pass already is: the XR view matrix is the headset
    /// pose alone, so a wall left in world coordinates would stay put while the
    /// crates beside it moved with the player.
    ///
    /// Rebuilt only when the player has actually moved or the hidden set has
    /// changed. Standing still is the common case and costs nothing; the walk
    /// case costs one rotate and one subtract per vertex, over static geometry
    /// that never has to be re-meshed.
    pub fn assemble(
        &mut self,
        hidden: &[String],
        offset: Vec3,
        yaw_inv: Quat,
        player_yaw: f32,
    ) -> Option<(&[SolidVertex], &[u32])> {
        if self.objects.is_empty() {
            return None;
        }
        let key = (offset, player_yaw, hidden_fingerprint(hidden));
        if self.built_for != Some(key) {
            let skip: HashSet<&str> = hidden.iter().map(String::as_str).collect();
            self.vertices.clear();
            self.indices.clear();
            for o in &self.objects {
                if skip.contains(o.id.as_str()) {
                    continue;
                }
                let base = self.vertices.len() as u32;
                for v in &o.vertices {
                    let p = yaw_inv * (Vec3::from(v.position) - offset);
                    // The normal is rotated but NOT translated. Translating it
                    // would leave every surface lit as though it faced the
                    // world origin, which looks like broken lighting rather
                    // than like a broken transform.
                    let n = yaw_inv * Vec3::from(v.normal);
                    self.vertices.push(SolidVertex {
                        position: p.to_array(),
                        normal: n.to_array(),
                        ..*v
                    });
                }
                self.indices.extend(o.indices.iter().map(|i| i + base));
            }
            self.built_for = Some(key);
        }
        if self.vertices.is_empty() || self.indices.is_empty() {
            return None;
        }
        Some((&self.vertices, &self.indices))
    }
}

/// An order-independent fingerprint of the hidden set.
///
/// Order-independent because the server builds the list by walking the scene
/// and a reorder is not a change; hashing the order would rebuild the whole
/// level's geometry for nothing. FNV-1a over each id, summed.
fn hidden_fingerprint(hidden: &[String]) -> u64 {
    let mut acc: u64 = hidden.len() as u64;
    for id in hidden {
        let mut h: u64 = 14695981039346656037;
        for b in id.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(1099511628211);
        }
        acc = acc.wrapping_add(h);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use space_soup_engine::scene::GameObject;

    /// A 4 x 2 x 0.3 wall as a plane set, which is what the editor writes.
    fn wall_object(id: &str) -> GameObject {
        let solid = space_soup_engine::brush::block_solid(
            [-2.0, 0.0, -0.15],
            [2.0, 2.0, 0.15],
            "concrete",
        );
        GameObject {
            id: id.into(),
            brush: Some(space_soup_engine::brush::BrushDef {
                solids: vec![solid],
                subtract: Vec::new(),
            }),
            ..Default::default()
        }
    }

    fn scene_of(objects: Vec<GameObject>) -> Scene {
        Scene { objects, ..Default::default() }
    }

    #[test]
    fn a_wall_meshes_to_its_own_shape_and_not_a_box() {
        let g = BrushGeometry::load(&scene_of(vec![wall_object("wall")]));
        assert_eq!(g.object_count(), 1);

        // Six quads, two triangles each. A bounding cuboid would be the same
        // count, so the shape is checked by EXTENT below, not by triangles.
        let o = &g.objects[0];
        assert_eq!(o.indices.len(), 36);

        let xs: Vec<f32> = o.vertices.iter().map(|v| v.position[0]).collect();
        let zs: Vec<f32> = o.vertices.iter().map(|v| v.position[2]).collect();
        let span = |v: &[f32]| {
            v.iter().fold(f32::MIN, |a, b| a.max(*b)) - v.iter().fold(f32::MAX, |a, b| a.min(*b))
        };
        assert!((span(&xs) - 4.0).abs() < 1e-4, "4m wide: {}", span(&xs));
        assert!((span(&zs) - 0.3).abs() < 1e-4, "and 30cm thick: {}", span(&zs));
    }

    #[test]
    fn every_face_carries_the_normal_of_the_plane_it_came_from() {
        // Shared vertices would average these into a bevel that is not there,
        // and a wall lit as though its corners were rounded reads as a lighting
        // bug rather than as a meshing one.
        let g = BrushGeometry::load(&scene_of(vec![wall_object("wall")]));
        let normals: HashSet<[i32; 3]> = g.objects[0]
            .vertices
            .iter()
            .map(|v| {
                [
                    (v.normal[0] * 100.0).round() as i32,
                    (v.normal[1] * 100.0).round() as i32,
                    (v.normal[2] * 100.0).round() as i32,
                ]
            })
            .collect();
        assert_eq!(normals.len(), 6, "six flat faces, six normals: {normals:?}");
        for n in &normals {
            let len = ((n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) as f32).sqrt() / 100.0;
            assert!((len - 1.0).abs() < 0.02, "normals are unit: {n:?}");
        }
    }

    #[test]
    fn a_scene_with_no_brushes_hands_the_renderer_nothing() {
        let mut g = BrushGeometry::load(&scene_of(vec![GameObject::default()]));
        assert!(g.is_empty());
        assert!(g.assemble(&[], Vec3::ZERO, Quat::IDENTITY, 0.0).is_none());
    }

    #[test]
    fn geometry_arrives_in_the_players_frame_not_the_worlds() {
        // The XR view matrix is the headset pose alone, so a wall left in world
        // coordinates stays put while the crates beside it move with the player.
        let mut g = BrushGeometry::load(&scene_of(vec![wall_object("wall")]));
        let offset = Vec3::new(10.0, 0.0, 0.0);

        let (verts, _) = g.assemble(&[], offset, Quat::IDENTITY, 0.0).expect("geometry");
        let min_x = verts.iter().fold(f32::MAX, |a, v| a.min(v.position[0]));
        assert!((min_x - (-12.0)).abs() < 1e-4, "walked 10m east: {min_x}");
    }

    #[test]
    fn a_yawed_player_turns_the_wall_and_its_normals_together() {
        let mut g = BrushGeometry::load(&scene_of(vec![wall_object("wall")]));
        let yaw = std::f32::consts::FRAC_PI_2;
        let (verts, _) = g
            .assemble(&[], Vec3::ZERO, Quat::from_rotation_y(-yaw), yaw)
            .expect("geometry");

        // A normal left unrotated lights every surface as though the player
        // never turned -- which looks like broken lighting rather than a broken
        // transform, and is the harder bug to find. Unit length does NOT catch
        // it (an unrotated normal is still unit), so this asserts the property
        // that actually breaks: on a convex solid every outward normal points
        // away from the centre, and that only survives if both were turned.
        let centre = verts
            .iter()
            .fold(Vec3::ZERO, |a, v| a + Vec3::from(v.position))
            / verts.len() as f32;
        for v in verts {
            let n = Vec3::from(v.normal);
            assert!((n.length() - 1.0).abs() < 1e-3, "still unit: {n:?}");
            assert!(
                n.dot(Vec3::from(v.position) - centre) > 0.0,
                "normal {n:?} does not face outward from {centre:?} at {:?}",
                v.position
            );
        }
        let spans_z = verts.iter().fold(f32::MIN, |a, v| a.max(v.position[2]))
            - verts.iter().fold(f32::MAX, |a, v| a.min(v.position[2]));
        assert!((spans_z - 4.0).abs() < 1e-3, "the 4m span turned onto z: {spans_z}");
    }

    #[test]
    fn a_hidden_brush_is_left_out_of_the_buffer() {
        let mut g = BrushGeometry::load(&scene_of(vec![wall_object("a"), wall_object("b")]));
        let whole = g.assemble(&[], Vec3::ZERO, Quat::IDENTITY, 0.0).unwrap().1.len();

        let half = g
            .assemble(&["a".to_string()], Vec3::ZERO, Quat::IDENTITY, 0.0)
            .unwrap()
            .1
            .len();
        assert_eq!(half * 2, whole, "one of two walls gone");

        assert!(
            g.assemble(&["a".into(), "b".into()], Vec3::ZERO, Quat::IDENTITY, 0.0).is_none(),
            "a level shot to pieces hands the renderer nothing, not an empty draw"
        );
    }

    #[test]
    fn a_wall_that_came_back_is_drawn_again() {
        // The cache is keyed on the hidden set as well as the pose. Keying it
        // on the pose alone would leave a repaired or reset wall invisible
        // until the player happened to move.
        let mut g = BrushGeometry::load(&scene_of(vec![wall_object("a")]));
        assert!(g.assemble(&["a".into()], Vec3::ZERO, Quat::IDENTITY, 0.0).is_none());
        assert!(
            g.assemble(&[], Vec3::ZERO, Quat::IDENTITY, 0.0).is_some(),
            "standing perfectly still, the wall must reappear"
        );
    }

    #[test]
    fn reordering_the_hidden_list_is_not_a_change() {
        let mut g = BrushGeometry::load(&scene_of(vec![wall_object("a"), wall_object("b")]));
        let first = g
            .assemble(&["a".into(), "b".into()], Vec3::ZERO, Quat::IDENTITY, 0.0)
            .is_none();
        assert!(first);
        // Same set, other order. The server builds it by walking the scene, so
        // a reorder is not news and must not rebuild the level.
        assert_eq!(
            hidden_fingerprint(&["a".into(), "b".into()]),
            hidden_fingerprint(&["b".into(), "a".into()])
        );
    }

    #[test]
    fn two_different_hidden_sets_are_told_apart() {
        assert_ne!(
            hidden_fingerprint(&["a".into()]),
            hidden_fingerprint(&["b".into()])
        );
        assert_ne!(
            hidden_fingerprint(&["a".into()]),
            hidden_fingerprint(&["a".into(), "b".into()])
        );
    }
}
