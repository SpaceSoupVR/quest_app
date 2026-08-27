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
//!
//! MATERIALS ARE PER SCENE, AND THEY ARE THE SCENE'S OWN
//!
//! Terrain's four layers are one art decision for the whole project. A level's
//! walls are not: a warehouse and a bunker share nothing. So the material array
//! is built from exactly the ids the loaded scene's brush faces reference,
//! which also keeps a level well inside the array's limit without anyone
//! curating a list.

use std::collections::HashSet;

use glam::{Quat, Vec3};
use space_soup::renderer::brush_pipeline::{BrushVertex, MAX_BRUSH_MATERIALS};
use space_soup::renderer::terrain_pipeline::TerrainImage;
use space_soup_engine::scene::Scene;

/// One brush object's triangles, in world space.
pub struct BrushObject {
    pub id: String,
    vertices: Vec<BrushVertex>,
    indices: Vec<u32>,
}

/// Every brush in the scene, plus the assembled buffer the renderer is handed.
#[derive(Default)]
pub struct BrushGeometry {
    objects: Vec<BrushObject>,
    /// The last assembly, kept so a frame that changed nothing costs nothing.
    vertices: Vec<BrushVertex>,
    indices: Vec<u32>,
    built_for: Option<(Vec3, f32, u64)>,
    /// Material ids in array-layer order. The renderer's texture array is built
    /// from this, so index i here is layer i there.
    materials: Vec<String>,
}

impl BrushGeometry {
    /// The material ids this scene's brushes reference, in layer order.
    pub fn materials(&self) -> &[String] {
        &self.materials
    }
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
        // Assigned in first-seen order over the scene, which is stable for a
        // given file -- so a level's layer numbering does not shuffle between
        // runs, and a screenshot of the wrong texture stays reproducible.
        let mut materials: Vec<String> = Vec::new();
        let mut layer_of = |name: &str| -> u32 {
            if let Some(i) = materials.iter().position(|m| m == name) {
                return i as u32;
            }
            if materials.len() >= MAX_BRUSH_MATERIALS {
                // Clamped rather than wrapped: wrapping would silently paint
                // this face with an unrelated material, which looks like an
                // authoring mistake in a place nobody edited.
                log::warn!(
                    "brush_render: more than {MAX_BRUSH_MATERIALS} materials in this scene; \
                     '{name}' will draw as '{}'",
                    materials[MAX_BRUSH_MATERIALS - 1]
                );
                return (MAX_BRUSH_MATERIALS - 1) as u32;
            }
            materials.push(name.to_string());
            (materials.len() - 1) as u32
        };

        // One lightmap layout for the whole level, built from the brushes in
        // scene order -- exactly the list the baker walks, so a brush's index
        // means the same thing on both sides. Built once outside the loop
        // because it is a property of the level, not of any one brush.
        let brush_defs: Vec<&space_soup_engine::brush::BrushDef> =
            scene.objects.iter().filter_map(|o| o.brush.as_ref()).collect();
        let lm_layout =
            space_soup_engine::brush_lightmap::scene_brush_lightmap_layout(&brush_defs);

        let mut objects = Vec::new();
        let mut brush_index = 0usize;
        for obj in &scene.objects {
            let Some(def) = obj.brush.as_ref() else { continue };
            let groups =
                space_soup_engine::brush::brush_mesh_in_atlas(def, &lm_layout, brush_index);
            brush_index += 1;

            let colour = space_soup::renderer::Color3(
                obj.cuboid.color.0,
                obj.cuboid.color.1,
                obj.cuboid.color.2,
                obj.cuboid.color.3,
            )
            .to_linear();

            let mut vertices: Vec<BrushVertex> = Vec::new();
            let mut indices: Vec<u32> = Vec::new();
            for g in &groups {
                let material = layer_of(&g.material);
                let base = vertices.len() as u32;
                for i in 0..g.positions.len() / 3 {
                    vertices.push(BrushVertex {
                        position: [g.positions[i * 3], g.positions[i * 3 + 1], g.positions[i * 3 + 2]],
                        normal: [g.normals[i * 3], g.normals[i * 3 + 1], g.normals[i * 3 + 2]],
                        tangent: [
                            g.tangents[i * 4],
                            g.tangents[i * 4 + 1],
                            g.tangents[i * 4 + 2],
                            g.tangents[i * 4 + 3],
                        ],
                        // In TILES, straight from the face's own scale, so a
                        // material tiling every two metres does exactly that.
                        uv: [g.uvs[i * 2], g.uvs[i * 2 + 1]],
                        material,
                        // Multiplied over the texture. For a face whose material
                        // is missing the array holds white, so this is the whole
                        // appearance -- which is what keeps an untextured brush
                        // looking like the colour its object was authored in.
                        tint: colour,
                        // Where this vertex reads the level's baked lighting.
                        uv2: [g.uv2[i * 2], g.uv2[i * 2 + 1]],
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
        Self { objects, materials, ..Default::default() }
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
    ) -> Option<(&[BrushVertex], &[u32])> {
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
                    self.vertices.push(BrushVertex {
                        position: p.to_array(),
                        normal: n.to_array(),
                        // The tangent is a direction in the face, so it turns
                        // with the wall. Left in world space it would rotate the
                        // normal map relative to the surface as the player
                        // turned -- lighting that swims.
                        tangent: {
                            let t = yaw_inv * Vec3::new(v.tangent[0], v.tangent[1], v.tangent[2]);
                            [t.x, t.y, t.z, v.tangent[3]]
                        },
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

/// Load a scene's brush materials from the game's material library.
///
/// `game/materials/<id>/color.jpg` and `normal.jpg`, which is exactly where the
/// editor's material library installs them -- one library, both consumers, so
/// what the author picked in the editor is what the headset binds. `rough.jpg`
/// and `ao.jpg` may also be there and are not read yet; the shader has no use
/// for them until it does more than diffuse.
///
/// Returns arrays parallel to `materials`, with `None` for anything missing. A
/// missing colour map is not an error worth failing a level over: it binds
/// white and the object's authored tint shows through, which is a wall someone
/// can see and file a bug about rather than a client that will not start.
pub fn load_materials(
    game_dir: &std::path::Path,
    materials: &[String],
) -> (Vec<TerrainImage>, Vec<Option<TerrainImage>>) {
    let dir = game_dir.join("materials");
    let mut colours = Vec::with_capacity(materials.len());
    let mut normals = Vec::with_capacity(materials.len());
    for id in materials {
        // `default` is the engine's name for "no material assigned", not a
        // directory anyone installs, so it is expected to be absent and is not
        // worth a warning on every level that has one unpainted face.
        let colour = TerrainImage::load(&dir.join(id).join("color.jpg"));
        if colour.is_none() && id != "default" {
            log::warn!("brush_render: material '{id}' has no colour map; drawing it white");
        }
        colours.push(colour.unwrap_or_else(|| TerrainImage {
            width: 1,
            height: 1,
            rgba: vec![255, 255, 255, 255],
        }));
        normals.push(TerrainImage::load(&dir.join(id).join("normal.jpg")));
    }
    (colours, normals)
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

    /// A wall whose six faces carry three different materials.
    fn painted_wall(id: &str) -> GameObject {
        let mut solid = space_soup_engine::brush::block_solid(
            [-2.0, 0.0, -0.15],
            [2.0, 2.0, 0.15],
            "concrete",
        );
        solid.faces[0].material = "brick".into();
        solid.faces[1].material = "brick".into();
        solid.faces[2].material = "metal".into();
        GameObject {
            id: id.into(),
            brush: Some(space_soup_engine::brush::BrushDef {
                solids: vec![solid],
                subtract: Vec::new(),
            }),
            ..Default::default()
        }
    }

    #[test]
    fn each_face_carries_the_layer_of_its_own_material() {
        let g = BrushGeometry::load(&scene_of(vec![painted_wall("wall")]));
        assert_eq!(g.materials().len(), 3, "three distinct materials: {:?}", g.materials());

        let used: HashSet<u32> = g.objects[0].vertices.iter().map(|v| v.material).collect();
        assert_eq!(used.len(), 3, "and three distinct layers on the geometry");
        for i in &used {
            assert!((*i as usize) < g.materials().len(), "layer {i} is in range");
        }
    }

    #[test]
    fn two_walls_of_the_same_material_share_one_layer() {
        // The point of one array and a per-vertex index: a level of forty walls
        // in four materials is four layers and one draw, not forty of either.
        let g = BrushGeometry::load(&scene_of(vec![painted_wall("a"), painted_wall("b")]));
        assert_eq!(g.materials().len(), 3, "still three: {:?}", g.materials());
    }

    #[test]
    fn a_scene_past_the_material_limit_clamps_rather_than_wrapping() {
        // Wrapping would paint the overflow with an unrelated material, which
        // looks like an authoring mistake somewhere nobody edited.
        let objects: Vec<GameObject> = (0..MAX_BRUSH_MATERIALS + 5)
            .map(|i| {
                let mut solid = space_soup_engine::brush::block_solid(
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    &format!("mat{i}"),
                );
                for f in &mut solid.faces {
                    f.material = format!("mat{i}");
                }
                GameObject {
                    id: format!("w{i}"),
                    brush: Some(space_soup_engine::brush::BrushDef {
                        solids: vec![solid],
                        subtract: Vec::new(),
                    }),
                    ..Default::default()
                }
            })
            .collect();

        let g = BrushGeometry::load(&scene_of(objects));
        assert_eq!(g.materials().len(), MAX_BRUSH_MATERIALS);
        for v in g.objects.iter().flat_map(|o| &o.vertices) {
            assert!(
                (v.material as usize) < MAX_BRUSH_MATERIALS,
                "layer {} would read past the end of the array",
                v.material
            );
        }
    }

    #[test]
    fn the_uv_is_in_tiles_so_a_wall_repeats_its_material() {
        // Not 0..1. A 4m wall of a material tiling every 2m must span two
        // tiles, and normalising it here would make every wall show exactly one
        // copy of its texture however big it is.
        let g = BrushGeometry::load(&scene_of(vec![wall_object("wall")]));
        let us: Vec<f32> = g.objects[0].vertices.iter().map(|v| v.uv[0]).collect();
        // The SPAN, not the largest value: the wall straddles the origin, so u
        // runs -1..1 and the biggest single number is 1 while the wall is two
        // tiles wide.
        let span = us.iter().fold(f32::MIN, |a, b| a.max(*b))
            - us.iter().fold(f32::MAX, |a, b| a.min(*b));
        assert!(
            (span - 2.0).abs() < 1e-4,
            "4m of wall at the default 2m tile is two tiles: {span}"
        );
    }

    #[test]
    fn tangents_survive_the_walk_into_the_players_frame() {
        // A tangent left in world space rotates the normal map relative to the
        // surface as the player turns, which looks like the lighting swimming.
        let mut g = BrushGeometry::load(&scene_of(vec![wall_object("wall")]));
        let yaw = std::f32::consts::FRAC_PI_2;
        let (verts, _) = g
            .assemble(&[], Vec3::ZERO, Quat::from_rotation_y(-yaw), yaw)
            .expect("geometry");
        for v in verts {
            let t = Vec3::new(v.tangent[0], v.tangent[1], v.tangent[2]);
            let n = Vec3::from(v.normal);
            assert!((t.length() - 1.0).abs() < 1e-3, "tangent stays unit: {t:?}");
            assert!(
                t.dot(n).abs() < 1e-3,
                "and stays in the face it belongs to: t={t:?} n={n:?}"
            );
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