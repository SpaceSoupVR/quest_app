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
    /// The box round its vertices, which a ray [`BrushGeometry::cast`] passes
    /// by without testing a triangle.
    bounds: (Vec3, Vec3),
}

/// Where a ray met the level's brushes: how far along it, the point, the face's
/// normal (toward the ray), its material's layer and its object's tint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrushHit {
    pub distance: f32,
    pub point: Vec3,
    pub normal: Vec3,
    pub material: u32,
    pub tint: Vec3,
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

    /// The first brush face a ray from `origin` along `dir` (a unit vector)
    /// meets from its front within `max` metres, in the WORLD, as the objects
    /// are. `hidden`: objects not drawn this frame, which nothing can meet.
    /// On the CPU, for the few rays a frame the flashlight's bounce casts: an
    /// object's box first, its triangles only when the ray passes through it.
    pub fn cast(&self, origin: Vec3, dir: Vec3, max: f32, hidden: &[String]) -> Option<BrushHit> {
        // Axes the ray runs along exactly divide by a tiny number instead of
        // zero, so the slabs' products stay finite.
        let safe = Vec3::select(dir.abs().cmplt(Vec3::splat(1e-9)), Vec3::splat(1e-9), dir);
        let inv = safe.recip();
        let mut best: Option<BrushHit> = None;
        for o in self.objects.iter().filter(|o| !hidden.iter().any(|h| *h == o.id)) {
            let reach = best.map_or(max, |b| b.distance);
            let (near, far) = {
                let a = (o.bounds.0 - origin) * inv;
                let b = (o.bounds.1 - origin) * inv;
                (a.min(b).max_element(), a.max(b).min_element())
            };
            if far < near.max(0.0) || near > reach {
                continue;
            }
            for tri in o.indices.chunks_exact(3) {
                let [a, b, c] = [tri[0], tri[1], tri[2]].map(|i| &o.vertices[i as usize]);
                let normal = Vec3::from(a.normal);
                // From the front only: the inside of a solid is never seen.
                if normal.dot(dir) >= 0.0 {
                    continue;
                }
                let Some(t) = ray_triangle(origin, dir, a.position.into(), b.position.into(), c.position.into()) else {
                    continue;
                };
                if t < best.map_or(max, |b| b.distance) {
                    best = Some(BrushHit {
                        distance: t,
                        point: origin + dir * t,
                        normal,
                        material: a.material,
                        tint: Vec3::new(a.tint[0], a.tint[1], a.tint[2]),
                    });
                }
            }
        }
        best
    }
}

/// Where a ray from `o` along `d` crosses triangle `a b c`, as its distance
/// along `d`, if ahead of `o` (Moller and Trumbore 1997).
fn ray_triangle(o: Vec3, d: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    let (e1, e2) = (b - a, c - a);
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    let s = o - a;
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = d.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    (t > 1e-4).then_some(t)
}

/// A colour map's mean diffuse reflectance, in LINEAR light: what a surface
/// of it sends back of the light falling on it, on average. Averaged as
/// light, not as sRGB bytes, which would report a surface far darker than it
/// is (the bake's `probe::image_albedo` does the same). Every fourth texel each
/// way is plenty for a mean.
pub fn mean_albedo(img: &TerrainImage) -> Vec3 {
    let to_linear = |b: u8| {
        let c = b as f32 / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    let (mut sum, mut n) = (Vec3::ZERO, 0u32);
    for y in (0..img.height).step_by(4) {
        for x in (0..img.width).step_by(4) {
            let i = ((y * img.width + x) * 4) as usize;
            let Some(px) = img.rgba.get(i..i + 3) else { continue };
            sum += Vec3::new(to_linear(px[0]), to_linear(px[1]), to_linear(px[2]));
            n += 1;
        }
    }
    if n == 0 {
        Vec3::ONE
    } else {
        sum / n as f32
    }
}

/// A material's mean roughness, as the shader reads its map (`rough.jpg`,
/// after the author's range): 1, fully rough, where it has none.
pub fn mean_roughness(img: Option<&TerrainImage>) -> f32 {
    let Some(img) = img else { return 1.0 };
    let (mut sum, mut n) = (0.0f32, 0u32);
    for y in (0..img.height).step_by(4) {
        for x in (0..img.width).step_by(4) {
            let i = ((y * img.width + x) * 4) as usize;
            let Some(px) = img.rgba.get(i) else { continue };
            sum += *px as f32 / 255.0;
            n += 1;
        }
    }
    if n == 0 {
        1.0
    } else {
        sum / n as f32
    }
}

impl BrushGeometry {
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    /// Every distinct plane the level's faces lie in whose material is no
    /// rougher than `max` (`roughness` by layer, mean): `(normal, offset,
    /// roughness)`, the plane `normal . x = offset` facing out of its solid, in
    /// the WORLD. One entry for a floor however many faces and rooms it spans.
    pub fn smooth_planes(&self, roughness: &[f32], max: f32) -> Vec<(Vec3, f32, f32)> {
        let mut out: Vec<(Vec3, f32, f32)> = Vec::new();
        for o in &self.objects {
            for tri in o.indices.chunks_exact(3) {
                let a = &o.vertices[tri[0] as usize];
                let r = roughness.get(a.material as usize).copied().unwrap_or(1.0);
                if r > max {
                    continue;
                }
                let n = Vec3::from(a.normal);
                let d = n.dot(Vec3::from(a.position));
                if !out.iter().any(|(m, e, q)| m.dot(n) > 0.9999 && (e - d).abs() < 1e-3 && *q == r) {
                    out.push((n, d, r));
                }
            }
        }
        out
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
        // Cut where the rooms' boxes cut them; see `split_at_rooms`.
        let brushes: Vec<(&str, &space_soup_engine::brush::BrushDef)> =
            scene.objects.iter().filter_map(|o| o.brush.as_ref().map(|b| (o.id.as_str(), b))).collect();
        let rooms: Vec<(Vec3, Vec3)> =
            space_soup_engine::room_graph::rooms_from_scene(&brushes).0.iter().map(|r| (r.min, r.max)).collect();
        let mut per_object: Vec<(&space_soup_engine::scene::GameObject, usize)> = Vec::new();
        let mut all_polys: Vec<space_soup_engine::brush::BrushPolygon> = Vec::new();
        let mut brush_index = 0usize;
        for obj in &scene.objects {
            let Some(def) = obj.brush.as_ref() else { continue };
            let polys = split_at_rooms(
                space_soup_engine::brush::brush_polygons_in_atlas(def, &lm_layout, brush_index),
                &rooms,
            );
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
            let bounds = vertices.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), v| {
                (lo.min(Vec3::from(v.position)), hi.max(Vec3::from(v.position)))
            });
            objects.push(BrushObject { id: obj.id.clone(), vertices, indices, bounds });
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

/// How far past a room's plane a polygon must reach, on both sides, before it
/// is cut there. A wall flush with the plane, or a few millimetres over it, is
/// left whole: cutting it would only make slivers.
const ROOM_SPLIT_SLACK: f32 = 0.01;

/// EVERY POLYGON CUT WHERE A ROOM'S BOX CUTS IT, so no polygon lies partly in a
/// room and partly out of it.
///
/// A face's reflection traces from the room its CENTROID is in (the shader's
/// `probe_face_room`; see `face_centre` for why not per pixel). A polygon that
/// straddles a room boundary therefore reflects one room over all of it. The
/// hallway has no end walls of its own: both its ends are the outside walls of
/// the buildings it joins, and those are single polygons metres long whose
/// centroids lie outdoors -- so the marble border round each hallway door, a
/// strip of the hall's outside wall seen from inside the hallway, reflected the
/// sky and the hills (headset, 2026-10-01 01:03:41). Cut at the hallway's box,
/// the strip is a polygon of its own whose centroid lies in the hallway.
///
/// The rooms are the level's room carves (`room_graph::rooms_from_scene`), the
/// boxes the probe volumes are made from. Done before the T-junction repair,
/// which then joins the new vertices to the neighbouring polygons.
fn split_at_rooms(
    polys: Vec<space_soup_engine::brush::BrushPolygon>,
    rooms: &[(Vec3, Vec3)],
) -> Vec<space_soup_engine::brush::BrushPolygon> {
    let mut out = Vec::with_capacity(polys.len());
    for poly in polys {
        let mut pieces = vec![poly];
        for &(lo, hi) in rooms {
            let mut next = Vec::with_capacity(pieces.len());
            for piece in pieces {
                split_at_room(piece, lo, hi, &mut next);
            }
            pieces = next;
        }
        out.extend(pieces);
    }
    out
}

/// `poly` cut by the six planes of the box `lo..hi` into the part inside it and
/// the parts outside, all pushed onto `out`. A polygon that does not reach into
/// the box is pushed whole; one lying IN a plane of the box -- a wall the room
/// is flush with -- is cut across that plane's neighbours, never along it.
fn split_at_room(
    poly: space_soup_engine::brush::BrushPolygon,
    lo: Vec3,
    hi: Vec3,
    out: &mut Vec<space_soup_engine::brush::BrushPolygon>,
) {
    let (plo, phi) = poly.positions.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(a, b), p| {
        (a.min(Vec3::from(*p)), b.max(Vec3::from(*p)))
    });
    // Flush with a face of the box counts as reaching it: that is the wall a
    // room's end is made of.
    let slack = Vec3::splat(ROOM_SPLIT_SLACK);
    if plo.cmpgt(hi + slack).any() || phi.cmplt(lo - slack).any() {
        out.push(poly);
        return;
    }
    let mut rest = poly;
    for axis in 0..3 {
        // Below the box's low face is outside it; so is above its high face.
        let (below, above) = split_polygon(rest, axis, lo[axis], true);
        out.extend(below);
        let Some(above) = above else { return };
        let (below, above) = split_polygon(above, axis, hi[axis], false);
        out.extend(above);
        let Some(below) = below else { return };
        rest = below;
    }
    out.push(rest);
}

/// A convex polygon cut by the plane where coordinate `axis` equals `at`, as
/// (the part below, the part above). Uncut -- one side `None` -- unless it
/// reaches `ROOM_SPLIT_SLACK` past the plane on both sides; one lying in the
/// plane goes above if `flush_above`, else below -- the box's side, so a wall
/// flush with either end of a room counts as the room's. The new corners lie
/// exactly on the plane, their uvs and lightmap uvs interpolated along the edge
/// they cut, which is exact: both are affine across a planar face.
fn split_polygon(
    poly: space_soup_engine::brush::BrushPolygon,
    axis: usize,
    at: f32,
    flush_above: bool,
) -> (Option<space_soup_engine::brush::BrushPolygon>, Option<space_soup_engine::brush::BrushPolygon>) {
    let d: Vec<f32> = poly.positions.iter().map(|p| p[axis] - at).collect();
    let below_it = d.iter().any(|&s| s < -ROOM_SPLIT_SLACK);
    let above_it = d.iter().any(|&s| s > ROOM_SPLIT_SLACK);
    match (below_it, above_it) {
        (false, false) if flush_above => return (None, Some(poly)),
        (true, false) | (false, false) => return (Some(poly), None),
        (false, true) => return (None, Some(poly)),
        (true, true) => {}
    }
    let empty = || space_soup_engine::brush::BrushPolygon {
        material: poly.material.clone(),
        normal: poly.normal,
        tangent: poly.tangent,
        ..Default::default()
    };
    let (mut below, mut above) = (empty(), empty());
    let push = |q: &mut space_soup_engine::brush::BrushPolygon, p: [f32; 3], uv: [f32; 2], uv2: [f32; 2]| {
        q.positions.push(p);
        q.uvs.push(uv);
        q.uv2.push(uv2);
    };
    let lerp2 = |a: [f32; 2], b: [f32; 2], t: f32| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
    let n = poly.positions.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let (p, uv, uv2) = (poly.positions[i], poly.uvs[i], poly.uv2[i]);
        // On the plane (to a rounding error): a corner of both pieces.
        if d[i] <= 1e-6 {
            push(&mut below, p, uv, uv2);
        }
        if d[i] >= -1e-6 {
            push(&mut above, p, uv, uv2);
        }
        if (d[i] < -1e-6 && d[j] > 1e-6) || (d[i] > 1e-6 && d[j] < -1e-6) {
            let t = d[i] / (d[i] - d[j]);
            let q = poly.positions[j];
            let mut c = [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t, p[2] + (q[2] - p[2]) * t];
            c[axis] = at;
            let (cuv, cuv2) = (lerp2(uv, poly.uvs[j], t), lerp2(uv2, poly.uv2[j], t));
            push(&mut below, c, cuv, cuv2);
            push(&mut above, c, cuv, cuv2);
        }
    }
    let real = |q: space_soup_engine::brush::BrushPolygon| (q.positions.len() >= 3).then_some(q);
    (real(below), real(above))
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
        let mut rough = TerrainImage::load(&dir.join(id).join("rough.jpg"));
        // THE AUTHOR'S READING OF THE MAP: `"roughness": {"min": a, "max": b}`
        // maps the map's 0..1 onto a..b. A downloaded map is someone else's
        // choice -- Rock063 ships at 0.33-0.59, wet-looking for dry cave rock,
        // and its walls showed mirror-like patches (headset, 2026-09-29).
        // Absent, the map is used as it is. See `roughness_range`.
        if let (Some(img), Some((lo, hi))) = (rough.as_mut(), roughness_range(game_dir, id)) {
            for px in img.rgba.chunks_exact_mut(4) {
                for c in &mut px[..3] {
                    let r = lo + (*c as f32 / 255.0) * (hi - lo);
                    *c = (r.clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
            log::info!("brush_render: material '{id}' roughness read as {lo:.2}..{hi:.2}");
        }
        roughs.push(rough);
        aos.push(TerrainImage::load(&dir.join(id).join("ao.jpg")));
    }
    BrushMaterialMaps { colours, normals, roughs, aos }
}

/// What material `id`'s roughness map's 0 and 1 stand for: its
/// `"roughness": {"min", "max"}` in the game's committed
/// `material_overrides.json` (keyed by material id), else in the material's own
/// `material.json`, else nothing. The overrides file wins because the library
/// is downloaded per machine and not committed (`game/.gitignore`): a setting
/// kept only in `materials/<id>/material.json` would reach nobody else.
pub fn roughness_range(game_dir: &std::path::Path, id: &str) -> Option<(f32, f32)> {
    let read = |path: std::path::PathBuf| -> Option<serde_json::Value> {
        serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
    };
    let range = |v: &serde_json::Value| -> Option<(f32, f32)> {
        let r = v.get("roughness")?;
        let lo = r.get("min")?.as_f64()? as f32;
        let hi = r.get("max")?.as_f64()? as f32;
        (lo.is_finite() && hi.is_finite()).then_some((lo.clamp(0.0, 1.0), hi.clamp(0.0, 1.0)))
    };
    read(game_dir.join("material_overrides.json"))
        .and_then(|o| o.get(id).and_then(range))
        .or_else(|| read(game_dir.join("materials").join(id).join("material.json")).and_then(|m| range(&m)))
}

#[cfg(test)]
mod roughness_tests {
    use super::roughness_range;

    #[test]
    fn the_committed_override_wins_over_the_librarys_own_file() {
        let dir = std::env::temp_dir().join(format!("roughness_range_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("materials/Rock063")).unwrap();
        std::fs::write(dir.join("materials/Rock063/material.json"), r#"{"roughness": {"min": 0.1, "max": 0.2}}"#).unwrap();
        assert_eq!(roughness_range(&dir, "Rock063"), Some((0.1, 0.2)), "the library's own setting");
        std::fs::write(dir.join("material_overrides.json"), r#"{"Rock063": {"roughness": {"min": 0.55, "max": 1.0}}}"#).unwrap();
        assert_eq!(roughness_range(&dir, "Rock063"), Some((0.55, 1.0)), "the committed override");
        assert_eq!(roughness_range(&dir, "Marble020"), None, "a material with neither is left as it is");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The shipped override parses and gives the hallway's rock a dry range.
    #[test]
    fn the_hallway_rock_reads_as_dry_rock() {
        let game = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../game");
        if !game.join("material_overrides.json").exists() {
            eprintln!("skipping: no material_overrides.json");
            return;
        }
        let (lo, hi) = roughness_range(&game, "Rock063").expect("Rock063 has an override");
        // Rock063's map spans 0.33..0.59: read through the override, its
        // middle must land where dry stone is, past the probe trace's fade.
        let mid = lo + 0.46 * (hi - lo);
        assert!(mid > 0.65 && hi <= 1.0, "Rock063 reads as {lo}..{hi}: its middle, {mid}, is still glossy");
    }
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

    /// EVERY BRUSH FACE OF THE SHIPPED LEVEL TAKES A SUN READER THAT FITS THE
    /// QUEST'S INSTRUCTION CACHE: never reached by the sun, or baked throughout
    /// -- none left to the full reader, which carries the level's static sun
    /// map and is over the cliff (`space_soup::renderer::brush_pipeline::SunFaces`).
    /// A bake that left faces unbaked would cost ~1 ms an eye wherever they are
    /// seen, with nothing else to show for it. Prints each class's share.
    #[test]
    fn every_face_of_the_shipped_level_takes_a_reader_without_the_static_sun_map() {
        use space_soup::renderer::brush_pipeline::{dilate_sun_mask, FaceSun, SunFaces};
        let game = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../game");
        let scene =
            Scene::load(&space_soup_engine::Manifest::scene_path(&game, "test_room")).unwrap();
        let mut geometry = BrushGeometry::load_with(&scene, true);
        let (vertices, indices) = geometry.assemble(&[], Vec3::ZERO, Quat::IDENTITY, 0.0).unwrap();
        let maps = space_soup_engine::lightmaps::load_scene_lightmaps(&game, "test_room");
        let mask = maps.iter().find(|m| m.object_id == space_soup_engine::lightmaps::SCENE_BRUSH_SUN_MASK_ID).unwrap();
        let dilated = dilate_sun_mask(&mask.rgba, mask.width, mask.height);
        let faces = SunFaces::from_mask(&dilated, mask.width, mask.height).unwrap();
        let mut count = [0usize; 3];
        let mut area = [0f32; 3];
        for tri in indices.chunks_exact(3) {
            let class = faces.class_of(vertices[tri[0] as usize].uv2_rect) as usize;
            let p = |i: u32| Vec3::from(vertices[i as usize].position);
            count[class] += 1;
            area[class] += 0.5 * (p(tri[1]) - p(tri[0])).cross(p(tri[2]) - p(tri[0])).length();
        }
        let total: f32 = area.iter().sum();
        for class in [FaceSun::Never, FaceSun::Baked, FaceSun::Unbaked] {
            let i = class as usize;
            println!("{class:?}: {} triangles, {:.1} m2 ({:.1}%)", count[i], area[i], 100.0 * area[i] / total);
        }
        assert_eq!(count[FaceSun::Unbaked as usize], 0, "faces left to the full reader");
        // And the partition the frame draws agrees.
        let (_, ends) = faces.partition(vertices, indices);
        assert_eq!(ends, [count[0] as u32 * 3, (count[0] + count[1]) as u32 * 3]);
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

    /// NO FACE OF THE REAL LEVEL LIES PARTLY IN A ROOM AND PARTLY OUT OF IT: a
    /// face reflects the room its centre is in, so one that straddles reflects
    /// the wrong room over part of itself. Before `split_at_rooms` the hall's
    /// outside wall at x = 3 ran from the hallway out to the front of the
    /// building in one polygon, and the hallway saw it reflect the sky.
    #[test]
    fn no_face_of_the_shipped_level_straddles_a_room() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../game/scenes/test_room.json");
        let scene = space_soup_engine::scene::Scene::load(std::path::Path::new(path)).unwrap();
        let brushes: Vec<(&str, &space_soup_engine::brush::BrushDef)> =
            scene.objects.iter().filter_map(|o| o.brush.as_ref().map(|b| (o.id.as_str(), b))).collect();
        let rooms = space_soup_engine::room_graph::rooms_from_scene(&brushes).0;
        assert!(rooms.len() >= 3, "test_room has the hall, the hallway and the brick hall: {}", rooms.len());
        let g = BrushGeometry::load_with(&scene, true);
        let key = |c: [f32; 3]| c.map(f32::to_bits);
        let mut faces: std::collections::HashMap<[u32; 3], (Vec3, Vec3, Vec3)> = Default::default();
        for v in g.objects.iter().flat_map(|o| o.vertices.iter()) {
            let p = Vec3::from(v.position);
            let e = faces.entry(key(v.face_centre)).or_insert((p, p, Vec3::from(v.face_centre)));
            e.0 = e.0.min(p);
            e.1 = e.1.max(p);
        }
        let s = ROOM_SPLIT_SLACK;
        let mut straddling = Vec::new();
        for (lo, hi, centre) in faces.values() {
            for r in &rooms {
                // Reaching into the room's box -- by more than the slack along
                // every axis the face spans, and within it along the one it is
                // flat in (a wall flush with the room's end)...
                let reaches = (0..3).all(|a| {
                    if hi[a] - lo[a] < 1e-4 {
                        lo[a] >= r.min[a] - s && lo[a] <= r.max[a] + s
                    } else {
                        hi[a].min(r.max[a]) - lo[a].max(r.min[a]) > s
                    }
                });
                // ...while not lying inside it.
                let inside = (0..3).all(|a| lo[a] >= r.min[a] - s && hi[a] <= r.max[a] + s);
                if reaches && !inside {
                    straddling.push((r.id.clone(), *lo, *hi, *centre));
                }
            }
        }
        assert!(straddling.is_empty(), "{} face(s) straddle a room: {:?}", straddling.len(), &straddling[..straddling.len().min(4)]);
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

    /// A ray meets the first face it comes to from the front, with that face's
    /// normal and material -- not the back of one it starts behind, not a
    /// wall past its reach, not a wall the frame does not draw.
    #[test]
    fn a_ray_meets_the_nearest_face_from_its_front() {
        let mut far = painted_wall("far");
        if let Some(b) = far.brush.as_mut() {
            b.solids = vec![space_soup_engine::brush::block_solid([-2.0, 0.0, -3.15], [2.0, 2.0, -2.85], "concrete")];
        }
        let g = BrushGeometry::load(&scene_of(vec![painted_wall("near"), far]));
        let hit = g.cast(Vec3::new(0.3, 1.0, 3.0), Vec3::NEG_Z, 20.0, &[]).expect("the near wall");
        assert!((hit.distance - 2.85).abs() < 1e-4 && (hit.point.z - 0.15).abs() < 1e-4, "{hit:?}");
        assert!((hit.normal - Vec3::Z).length() < 1e-5);
        // The material of the face it met: the one facing +z.
        let facing: HashSet<u32> = g.objects[0]
            .vertices
            .iter()
            .filter(|v| Vec3::from(v.normal).dot(Vec3::Z) > 0.99)
            .map(|v| v.material)
            .collect();
        assert_eq!(facing, HashSet::from([hit.material]));
        let c = GameObject::default().cuboid.color;
        let tint = space_soup::renderer::Color3(c.0, c.1, c.2, c.3).to_linear();
        assert_eq!(hit.tint, Vec3::new(tint[0], tint[1], tint[2]), "the object's colour");
        // Started inside the near wall, it meets the far wall's front.
        let inside = g.cast(Vec3::new(0.3, 1.0, 0.0), Vec3::NEG_Z, 20.0, &[]).unwrap();
        assert!((inside.point.z + 2.85).abs() < 1e-4, "{inside:?}");
        // Hidden, the near wall is not there; past the reach, nothing is.
        let through = g.cast(Vec3::new(0.3, 1.0, 3.0), Vec3::NEG_Z, 20.0, &["near".to_string()]).unwrap();
        assert!((through.point.z + 2.85).abs() < 1e-4);
        assert!(g.cast(Vec3::new(0.3, 1.0, 3.0), Vec3::NEG_Z, 2.0, &[]).is_none());
        assert!(g.cast(Vec3::new(0.3, 1.0, 3.0), Vec3::Z, 20.0, &[]).is_none(), "away from both");
        assert!(g.cast(Vec3::new(5.0, 1.0, 3.0), Vec3::NEG_Z, 20.0, &[]).is_none(), "past their ends");
    }

    /// A colour map's mean albedo is taken in linear light.
    #[test]
    fn a_colour_maps_albedo_is_its_mean_in_linear_light() {
        let image = |rgba: Vec<u8>, width: u32, height: u32| TerrainImage { width, height, rgba };
        let white = image(vec![255; 4 * 16], 4, 4);
        assert!((mean_albedo(&white) - Vec3::ONE).length() < 1e-5);
        // Mid grey in sRGB is a fifth of white's light, not a half.
        let grey = image([128u8, 128, 128, 255].repeat(64), 8, 8);
        let a = mean_albedo(&grey);
        assert!((a.x - 0.2158).abs() < 1e-3 && a.x == a.y && a.y == a.z, "{a}");
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