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

/// Whether the headset repairs T-junctions in brush geometry.
///
/// OFF, because on the headset it made things WORSE. With it on, room edges
/// showed a sawtooth artefact in both SSR states that was absent before the
/// repair (user report, 2026-09-10) -- even though the repaired mesh passes
/// every geometric check: no flipped or zero-area triangles, lightmap UVs in
/// range, zero T-junctions. The leading suspect is coplanar overlapping faces
/// from CSG that z-fight once the repair gives them different triangulations.
/// Kept as a switch so that can be measured and fixed, not deleted.
pub const REPAIR_T_JUNCTIONS: bool = true;
// ^ ON, AND NOW IT ACTUALLY REPAIRS (2026-09-23). test_room: 0 T-junctions.
//
// What was wrong, in order, so none of it is retried:
//   - fan from an INVENTED centroid vertex: reached zero, but the vertex
//     existed in no other representation and shipped a sawtooth (2026-09-10);
//   - fan from one real corner: dropped the zero-area triangles spanning runs
//     of collinear inserted points, i.e. the repair itself -- 408 -> 300;
//   - ear clipping without a chord check: cut a far corner of the ceiling and
//     ran the new chord straight through both doorway corners on the front
//     edge -- the very T-junctions under the front seam and the doorway floor.
//
// And the repair was only ever half the fix. The CSG emitted faces that ran on
// inside solid geometry (`brush::exposed_fragments`), so junctions were two
// surfaces crossing rather than one shared edge, and 408 T-junctions came from
// that. Trimmed to exposed surface the level has 96; repaired, 0 -- in 180
// triangles against the original 204.
//
// The remaining T-junctions were exactly where the headset showed artefacts:
// the front ceiling/wall edge and the doorway floor edge, where the doorway
// splits the front wall into pieces whose corners land mid-edge on the long
// ceiling and floor polygons. A T-junction leaves pixel gaps along its edge;
// through them the headset showed the buried faces behind, lit from inside the
// wall -- the bright specks the magenta-sky test proved were not sky.

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
    /// Build the headset's brush mesh, with T-junction repair as configured.
    pub fn load(scene: &Scene) -> Self {
        Self::load_with(scene, REPAIR_T_JUNCTIONS)
    }

    /// `load`, with the repair chosen explicitly -- so the repair stays tested
    /// while it is switched off on the headset.
    pub fn load_with(scene: &Scene, repair_t_junctions: bool) -> Self {
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

        // POLYGONS FIRST, from every brush in the level, so T-junctions are
        // repaired ACROSS objects as well as within one: a room's carve and the
        // shell it is carved from are one object, but two buildings that touch
        // are two, and a crack does not care which.
        let mut per_object: Vec<(&space_soup_engine::scene::GameObject, usize)> = Vec::new();
        let mut all_polys: Vec<space_soup_engine::brush::BrushPolygon> = Vec::new();
        let mut brush_index = 0usize;
        for obj in &scene.objects {
            let Some(def) = obj.brush.as_ref() else { continue };
            let polys =
                space_soup_engine::brush::brush_polygons_in_atlas(def, &lm_layout, brush_index);
            brush_index += 1;
            per_object.push((obj, polys.len()));
            all_polys.extend(polys);
        }
        let before = space_soup_engine::brush_tjunction::count_t_junctions(&all_polys);
        let repaired: Vec<(space_soup_engine::brush::BrushPolygon, bool)> = if repair_t_junctions {
            space_soup_engine::brush_tjunction::fix_t_junctions(&all_polys)
        } else {
            all_polys.iter().map(|p| (p.clone(), false)).collect()
        };
        let mut repaired = repaired.into_iter();

        let mut objects = Vec::new();
        for (obj, count) in per_object {
            let colour = space_soup::renderer::Color3(
                obj.cuboid.color.0,
                obj.cuboid.color.1,
                obj.cuboid.color.2,
                obj.cuboid.color.3,
            )
            .to_linear();

            let mut vertices: Vec<BrushVertex> = Vec::new();
            let mut indices: Vec<u32> = Vec::new();
            for (poly, changed) in repaired.by_ref().take(count) {
                let material = layer_of(&poly.material);
                let tri = space_soup_engine::brush_tjunction::triangulate(&poly, changed);
                let base = vertices.len() as u32;
                // THE FACE'S OWN CENTRE, shared by every vertex of it.
                //
                // It selects the reflection probe. A fragment's interpolated
                // position cannot: a room's probe box IS its interior, so wall
                // fragments sit exactly on the box surface where the
                // containment test is a coin toss, and an MSAA edge pixel is
                // shaded at a centre that can lie outside the polygon
                // altogether, EXTRAPOLATING the position past the box. That
                // was the dotted line along every room seam. A centroid is
                // deep inside the room and identical for the whole face, so
                // the face cannot disagree with itself.
                // THE FACE'S OWN LIGHTMAP FOOTPRINT, so the fragment can clamp
                // an extrapolated uv2 back into it. Its own bounding box rather
                // than the chart rect: strictly inside the chart, already in
                // hand here, and correct however the atlas is packed.
                let uv2_rect = {
                    let mut lo = [f32::INFINITY; 2];
                    let mut hi = [f32::NEG_INFINITY; 2];
                    for t in &tri.uv2 {
                        for a in 0..2 {
                            lo[a] = lo[a].min(t[a]);
                            hi[a] = hi[a].max(t[a]);
                        }
                    }
                    // A face with no uv2 at all would leave infinities here and
                    // clamp every sample to nothing; fall back to the whole
                    // atlas, which is exactly the old unclamped behaviour.
                    if lo[0].is_finite() && hi[0].is_finite() {
                        [lo[0], lo[1], hi[0], hi[1]]
                    } else {
                        [0.0, 0.0, 1.0, 1.0]
                    }
                };
                let face_centre = {
                    let n = tri.positions.len().max(1) as f32;
                    let mut c = [0.0f32; 3];
                    for pos in &tri.positions {
                        c[0] += pos[0];
                        c[1] += pos[1];
                        c[2] += pos[2];
                    }
                    [c[0] / n, c[1] / n, c[2] / n]
                };
                // HALF THE FACE'S EXTENT IN ITS OWN BASIS, about `face_centre`.
                //
                // The fragment clamps its interpolated position into this
                // before taking a view direction; see
                // `BrushVertex::face_half_extent` for why that is what stops
                // the seam on polished faces.
                //
                // Measured in the face's tangent frame rather than as a world
                // box, so it survives the player-frame rotation untouched --
                // exactly like `uv2_rect` and unlike `face_centre`, which is a
                // position and must be transformed.
                let face_half_extent = {
                    let n = Vec3::from(poly.normal);
                    let raw = Vec3::new(poly.tangent[0], poly.tangent[1], poly.tangent[2]);
                    let t = (raw - n * n.dot(raw)).normalize_or_zero();
                    let b = n.cross(t) * poly.tangent[3];
                    let c = Vec3::from(face_centre);
                    let mut half = [0.0f32; 2];
                    for pos in &tri.positions {
                        let d = Vec3::from(*pos) - c;
                        half[0] = half[0].max(d.dot(t).abs());
                        half[1] = half[1].max(d.dot(b).abs());
                    }
                    // A degenerate tangent leaves zeros, which would clamp every
                    // fragment onto the face centre. Fall back to something no
                    // clamp can bite on, which is the old unclamped behaviour.
                    if t.length_squared() > 0.5 {
                        half
                    } else {
                        [f32::INFINITY; 2]
                    }
                };
                for i in 0..tri.positions.len() {
                    vertices.push(BrushVertex {
                        position: tri.positions[i],
                        normal: poly.normal,
                        tangent: poly.tangent,
                        uv: tri.uvs[i],
                        material,
                        tint: colour,
                        uv2: tri.uv2[i],
                        face_centre,
                        uv2_rect,
                        face_half_extent,
                    });
                }
                indices.extend(tri.indices.iter().map(|i| i + base));
            }

            if vertices.is_empty() || indices.is_empty() {
                log::warn!("brush_render: '{}' produced no geometry", obj.id);
                continue;
            }
            objects.push(BrushObject { id: obj.id.clone(), vertices, indices });
        }
        if before > 0 {
            if repair_t_junctions {
                log::info!("brush_render: repaired {before} T-junction(s) so walls meet without cracks");
            } else {
                log::info!("brush_render: {before} T-junction(s) left unrepaired (REPAIR_T_JUNCTIONS is off)");
            }
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
                        // A POSITION, so it gets the position transform. Left
                        // to `..*v` it would stay in world space while
                        // `position` moved to the player's frame, and the
                        // probe box test would be comparing two different
                        // frames -- the exact bug the header of this file
                        // warns about for static geometry.
                        face_centre: (yaw_inv * (Vec3::from(v.face_centre) - offset)).to_array(),
                        // NOT transformed: this is an atlas UV rectangle, not a
                        // position. `..*v` copying it through is correct here
                        // and would have been wrong for `face_centre` -- the two
                        // sit next to each other and mean different things.
                        // `face_half_extent` rides through on `..*v` for the
                        // same reason: it is a pair of LENGTHS in the face's own
                        // basis, and a yaw rotation does not change a length.
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
) -> BrushMaterialMaps {
    let dir = game_dir.join("materials");
    let mut colours = Vec::with_capacity(materials.len());
    let mut normals = Vec::with_capacity(materials.len());
    let mut roughs = Vec::with_capacity(materials.len());
    let mut aos = Vec::with_capacity(materials.len());
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
        // Genuinely optional, and silently so. Plenty of materials ship with
        // neither, and a missing one means "fully rough, fully unoccluded" --
        // which is exactly how every brush looked before these were read at
        // all, so an older material library keeps rendering unchanged.
        roughs.push(TerrainImage::load(&dir.join(id).join("rough.jpg")));
        aos.push(TerrainImage::load(&dir.join(id).join("ao.jpg")));
    }
    BrushMaterialMaps { colours, normals, roughs, aos }
}

/// Every map the brush shader reads, one entry per material in order.
///
/// A struct rather than a tuple because it grew to four parallel vectors, and
/// four positional returns is an invitation to pass roughness where ambient
/// occlusion belongs -- which would light perfectly and look merely wrong.
pub struct BrushMaterialMaps {
    pub colours: Vec<TerrainImage>,
    pub normals: Vec<Option<TerrainImage>>,
    pub roughs: Vec<Option<TerrainImage>>,
    pub aos: Vec<Option<TerrainImage>>,
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

    /// Vertex-on-edge incidences at the TRIANGLE level, across every object --
    /// what the rasteriser actually sees. Deliberately independent of the
    /// engine's polygon-level counter, so the two cannot share a blind spot.
    fn triangle_level_t_junctions(g: &BrushGeometry) -> usize {
        let mut verts: Vec<[f32; 3]> = Vec::new();
        let mut edges: Vec<([f32; 3], [f32; 3])> = Vec::new();
        for o in &g.objects {
            verts.extend(o.vertices.iter().map(|v| v.position));
            for t in o.indices.chunks(3) {
                for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    edges.push((o.vertices[a as usize].position, o.vertices[b as usize].position));
                }
            }
        }
        let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let mut hits = 0;
        for (a, b) in &edges {
            let ab = sub(*b, *a);
            let len2 = dot(ab, ab);
            if len2 < 1e-10 {
                continue;
            }
            let len = len2.sqrt();
            for v in &verts {
                let t = dot(sub(*v, *a), ab) / len2;
                if t * len <= 1e-4 || (1.0 - t) * len <= 1e-4 {
                    continue;
                }
                let p = [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t];
                let d = sub(*v, p);
                if dot(d, d) < 1e-8 {
                    hits += 1;
                }
            }
        }
        hits
    }

    /// Bilinear at mip level L reaches about 2^L base texels past a chart's
    /// edge, so a level is only safe while that stays inside the gutter --
    /// past it the filter averages in whatever the atlas packed next door.
    /// The renderer's mip counts and the engine's gutters live in crates that
    /// cannot see each other; this is where they are held together.
    #[test]
    fn the_lightmap_mip_depth_fits_the_gutter() {
        let top = space_soup::renderer::mesh::LIGHTMAP_MIP_LEVELS - 1;
        assert!(1u32 << top <= space_soup_engine::brush_lightmap::GUTTER);
    }

    #[test]
    fn the_sun_mask_mip_depth_fits_its_gutter() {
        use space_soup_engine::brush_lightmap::{GUTTER, SUN_MASK_SCALE};
        let top = space_soup::renderer::mesh::SUN_MASK_MIP_LEVELS - 1;
        assert!(1u32 << top <= GUTTER * SUN_MASK_SCALE);
        // And no shallower than the lightmap's reach at the same distance: the
        // mask is SUN_MASK_SCALE times finer, so it needs log2 of that more.
        let lm_top = space_soup::renderer::mesh::LIGHTMAP_MIP_LEVELS - 1;
        assert!(top >= lm_top + SUN_MASK_SCALE.trailing_zeros());
    }

    /// DIAGNOSTIC (not an assertion): edges of the shipped mesh that no other
    /// triangle shares EXACTLY. A closed surface uses every edge twice, once
    /// each way; an edge used once is a place the rasteriser can leave a gap.
    #[test]
    #[ignore]
    fn diag_unshared_edges_in_the_shipped_level() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../game/scenes/test_room.json");
        let scene = space_soup_engine::scene::Scene::load(std::path::Path::new(path)).unwrap();
        let g = BrushGeometry::load_with(&scene, true);
        let key = |p: [f32; 3]| [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()];
        let mut count: std::collections::HashMap<([u32; 3], [u32; 3]), i32> = Default::default();
        let mut all: Vec<([f32; 3], [f32; 3])> = Vec::new();
        for o in &g.objects {
            for t in o.indices.chunks(3) {
                for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    let (pa, pb) = (o.vertices[a as usize].position, o.vertices[b as usize].position);
                    let (ka, kb) = (key(pa), key(pb));
                    let k = if ka < kb { (ka, kb) } else { (kb, ka) };
                    *count.entry(k).or_default() += 1;
                    all.push((pa, pb));
                }
            }
        }
        let mut once = 0;
        let mut seen = std::collections::HashSet::new();
        for (pa, pb) in &all {
            let (ka, kb) = (key(*pa), key(*pb));
            let k = if ka < kb { (ka, kb) } else { (kb, ka) };
            if count[&k] == 1 && seen.insert(k) {
                once += 1;
                eprintln!("unshared edge {pa:?} -> {pb:?}");
            }
        }
        // Distinct vertices closer than 5 mm: the same corner computed twice.
        let mut verts: Vec<[f32; 3]> = g.objects.iter().flat_map(|o| o.vertices.iter().map(|v| v.position)).collect();
        verts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        verts.dedup();
        let mut near = 0;
        for i in 0..verts.len() {
            for j in i + 1..verts.len() {
                let d = ((verts[i][0] - verts[j][0]).powi(2) + (verts[i][1] - verts[j][1]).powi(2) + (verts[i][2] - verts[j][2]).powi(2)).sqrt();
                if d > 0.0 && d < 5e-3 {
                    near += 1;
                    eprintln!("near-duplicate {:?} {:?} ({d:e} m)", verts[i], verts[j]);
                }
            }
        }
        eprintln!("{once} unshared edges, {near} near-duplicate vertex pairs, {} triangles", all.len() / 3);
    }

    /// THE REAL LEVEL, measured the way the rasteriser meets it: repaired, it
    /// has NO T-junctions. Unrepaired but trimmed to exposed surface it has 96,
    /// every one of them where a doorway splits a wall into pieces; see the
    /// note on `REPAIR_T_JUNCTIONS` for the three triangulations that did not
    /// get here.
    #[test]
    fn the_shipped_level_mesh_has_no_t_junctions() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../game/scenes/test_room.json");
        let scene = space_soup_engine::scene::Scene::load(std::path::Path::new(path))
            .expect("test_room.json should load");
        let g = BrushGeometry::load_with(&scene, true);
        assert!(!g.objects.is_empty(), "the level produced no brush geometry to check");
        let hits = triangle_level_t_junctions(&g);
        assert_eq!(hits, 0, "{hits} vertices still sit inside another triangle's edge");
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