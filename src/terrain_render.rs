//! Terrain geometry for the on-device renderer.
//!
//! The samples are static scene data that is ALREADY on the headset -- run.sh
//! pushes game/ wholesale -- so the client loads them itself rather than
//! receiving them over the wire. Terrain therefore costs nothing per snapshot,
//! which matters when the target is 64 players and the snapshot budget is the
//! binding constraint.
//!
//! Built once per scene load. A heightfield does not change at runtime yet, and
//! when it does (craters), the right shape is to re-cook the affected patch
//! rather than to stream vertices every frame.

use glam::{Quat, Vec3};
use space_soup::renderer::cuboid::SolidVertex;
use space_soup_engine::brush::{contains_point, BrushDef, BrushSolid};
use space_soup_engine::terrain::{TerrainDef, TerrainSource};

/// How far below the terrain surface to look for a structure that buries it.
///
/// BELOW, not above. `BrushDef::evaluate` yields a building's SOLID pieces --
/// slab, walls, ceiling -- and not the carved room, so a point just above the
/// ground inside a room is open air and would find nothing. Five centimetres
/// down lands inside a floor slab resting on the ground, and inside nothing at
/// all under a structure raised off it, like a bridge.
pub const BURIED_PROBE_DEPTH: f32 = 0.05;

/// How far a buried terrain vertex is lowered, in the RENDERED copy only.
///
/// A floor laid on flattened ground is coplanar with it: `test_room` measures
/// the terrain 0.1 mm under the room floor, well inside depth precision, so the
/// two z-fight and grass flickers through the floor. It only showed with SSR
/// off, because the SSR eye pass redraws every brush over the finished scene
/// and paints the fight over.
///
/// Physics is untouched: the terrain stays the ground you collide with.
pub const BURIED_DROP: f32 = 0.02;

/// How finely a candidate triangle is sampled for coverage: a barycentric grid
/// of `(n + 1)(n + 2) / 2` points (28 at 6). At the 0.86 m terrain spacing that
/// is a sample every ~14 cm, finer than any wall.
const COVERAGE_SAMPLES: u32 = 6;

/// What `bury_under_structures` changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Burial {
    pub hidden_triangles: usize,
    pub lowered_vertices: usize,
}

/// Render-only treatment of terrain covered by structure floors.
///
/// A triangle whose three vertices are ALL buried, and which is covered across
/// its whole area, is dropped: drawing it is pure wasted fill and the source of
/// the z-fight. (Three buried corners alone are not enough at an inside corner;
/// see the note in the body.)
/// A triangle that crosses a wall line cannot be dropped without opening a gap
/// outside the wall, so it is kept and only its buried vertices are lowered by
/// `BURIED_DROP` -- enough to put the part under the room well below the floor,
/// while the dip outside is hidden by the wall's foot below ground.
pub fn bury_under_structures(
    positions: &mut [Vec3],
    indices: &[u32],
    solids: &[BrushSolid],
) -> (Vec<u32>, Burial) {
    let buried: Vec<bool> = positions
        .iter()
        .map(|p| {
            let probe = [p.x as f64, (p.y - BURIED_PROBE_DEPTH) as f64, p.z as f64];
            solids.iter().any(|s| contains_point(s, probe, 1e-4))
        })
        .collect();
    // COVERED ACROSS ITS WHOLE AREA, not just at its corners. Three buried
    // corners prove a triangle covered only under a CONVEX footprint. Where the
    // hallway meets a hall the footprint is an L, and a triangle with a corner
    // under each building spans the open ground in the inside corner between
    // them: hiding it opened a hole in the grass (headset, 2026-09-24). So a
    // triangle whose corners are all buried is also sampled across its area,
    // and kept -- lowered, like any triangle crossing a wall -- if any sample
    // is open ground. Only triangles already buried at every corner pay for it.
    let covered = |q: Vec3| {
        let probe = [q.x as f64, (q.y - BURIED_PROBE_DEPTH) as f64, q.z as f64];
        solids.iter().any(|s| contains_point(s, probe, 1e-4))
    };
    let fully_covered = |a: Vec3, b: Vec3, c: Vec3| {
        (0..=COVERAGE_SAMPLES).all(|i| {
            (0..=(COVERAGE_SAMPLES - i)).all(|j| {
                let (u, v) = (i as f32 / COVERAGE_SAMPLES as f32, j as f32 / COVERAGE_SAMPLES as f32);
                covered(a + (b - a) * u + (c - a) * v)
            })
        })
    };
    let mut kept = Vec::with_capacity(indices.len());
    let mut burial = Burial::default();
    for tri in indices.chunks_exact(3) {
        let [a, b, c] = [tri[0], tri[1], tri[2]].map(|i| positions[i as usize]);
        if tri.iter().all(|&i| buried[i as usize]) && fully_covered(a, b, c) {
            burial.hidden_triangles += 1;
        } else {
            kept.extend_from_slice(tri);
        }
    }
    for (p, b) in positions.iter_mut().zip(&buried) {
        if *b {
            p.y -= BURIED_DROP;
            burial.lowered_vertices += 1;
        }
    }
    (kept, burial)
}

/// Ground vertices and indices in the renderer's own format.
///
/// `vertices` are WORLD space, as loaded. `assemble` produces the player-frame
/// copy the renderer actually draws.
pub struct TerrainGeometry {
    pub vertices: Vec<SolidVertex>,
    /// Triangles REORDERED so each spatial chunk is one contiguous run.
    ///
    /// The order is what makes frustum culling possible at all: a chunk has to
    /// be drawable as a single `draw_indexed` range, and a row-major grid
    /// scatters any square patch across the whole buffer.
    pub indices: Vec<u32>,
    /// Player-frame copy, rebuilt only when the player has actually moved.
    posed: Vec<SolidVertex>,
    built_for: Option<(Vec3, f32)>,
    /// `(first_index, index_count)` per chunk, into `indices`.
    chunk_ranges: Vec<(u32, u32)>,
    /// Player-frame bounds per chunk, rebuilt alongside `posed`.
    chunk_bounds: Vec<(Vec3, Vec3)>,
}

impl TerrainGeometry {
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() || self.indices.is_empty()
    }

    /// Ground in the player's frame, the way every other kind of geometry is
    /// handed to the renderer.
    ///
    /// This was missing, and the symptom pointed at the wrong thing entirely.
    /// Brushes were transformed and terrain was passed through in world space,
    /// so walking left the ground glued to the player while the room slid past
    /// -- which reads as "the room brush is moving", not as "the ground is
    /// stuck". Terrain was the only geometry in the scene without a frame.
    ///
    /// Cached on (offset, yaw) exactly like `BrushGeometry::assemble`: standing
    /// still costs nothing, and a heightfield is static so there is never a
    /// re-mesh, only a rotate and a subtract per vertex.
    pub fn assemble(
        &mut self,
        offset: Vec3,
        yaw_inv: Quat,
        player_yaw: f32,
    ) -> Option<(&[SolidVertex], &[u32])> {
        if self.is_empty() {
            return None;
        }
        let key = (offset, player_yaw);
        if self.built_for != Some(key) {
            self.posed.clear();
            self.posed.reserve(self.vertices.len());
            for v in &self.vertices {
                let p = yaw_inv * (Vec3::from(v.position) - offset);
                // The normal is rotated but NOT translated -- it is a
                // direction. Subtracting the offset from it too would tilt the
                // ground's lighting by however far the player had walked.
                let n = yaw_inv * Vec3::from(v.normal);
                let mut out = *v;
                out.position = p.to_array();
                out.normal = n.to_array();
                self.posed.push(out);
            }
            // Bounds follow the vertices into the player's frame. Recomputed
            // here rather than rotated: an AABB does not survive a rotation --
            // rotating the eight corners and re-fitting grows the box every
            // time, and a box that grows every frame stops culling anything.
            self.chunk_bounds.clear();
            for &(first, count) in &self.chunk_ranges {
                let mut lo = Vec3::splat(f32::INFINITY);
                let mut hi = Vec3::splat(f32::NEG_INFINITY);
                for i in first..first + count {
                    let p = Vec3::from(self.posed[self.indices[i as usize] as usize].position);
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
                self.chunk_bounds.push((lo, hi));
            }
            self.built_for = Some(key);
        }
        Some((&self.posed, &self.indices))
    }

    /// The chunks a shadow pass may cull against a light's frustum.
    ///
    /// Empty until `assemble` has run, because the bounds are in the player's
    /// frame and there is no frame before then.
    /// World-space bounds of the terrain, from the unposed vertices.
    ///
    /// The UNPOSED ones deliberately: this is the footprint the baked occlusion
    /// map was generated over, which is fixed in the world and does not move
    /// with the player.
    pub fn world_bounds(&self) -> (Vec3, Vec3) {
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for v in &self.vertices {
            let p = Vec3::from(v.position);
            lo = lo.min(p);
            hi = hi.max(p);
        }
        (lo, hi)
    }

    pub fn caster_chunks(&self) -> Vec<space_soup::renderer::shadow::CasterChunk> {
        self.chunk_ranges
            .iter()
            .zip(self.chunk_bounds.iter())
            .map(|(&(first_index, index_count), &(min, max))| {
                space_soup::renderer::shadow::CasterChunk { first_index, index_count, min, max }
            })
            .collect()
    }
}

/// Reorder triangles so that spatially-near ones are adjacent in the buffer.
///
/// A grid-bucket sort on each triangle's centroid in x/z. Ground is a
/// heightfield, so two triangles near each other on the map are near each other
/// in space, and `CHUNK_TARGET` buckets per axis is enough resolution that a
/// small spot cone touches only a few of them.
///
/// Deliberately not a full spatial tree: this runs once per level load, the
/// gain is bounded by how tight the chunks are rather than by how clever the
/// structure is, and a flat grid is trivially correct to re-derive.
fn chunk_indices(
    vertices: &[SolidVertex],
    indices: &[u32],
    buckets_per_axis: u32,
) -> (Vec<u32>, Vec<(u32, u32)>) {
    let n = buckets_per_axis.max(1);
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for v in vertices {
        let p = Vec3::from(v.position);
        lo = lo.min(p);
        hi = hi.max(p);
    }
    let extent = hi - lo;
    let sx = if extent.x.abs() > 1e-6 { n as f32 / extent.x } else { 0.0 };
    let sz = if extent.z.abs() > 1e-6 { n as f32 / extent.z } else { 0.0 };

    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); (n * n) as usize];
    for tri in indices.chunks_exact(3) {
        let c = (Vec3::from(vertices[tri[0] as usize].position)
            + Vec3::from(vertices[tri[1] as usize].position)
            + Vec3::from(vertices[tri[2] as usize].position))
            / 3.0;
        let bx = (((c.x - lo.x) * sx) as u32).min(n - 1);
        let bz = (((c.z - lo.z) * sz) as u32).min(n - 1);
        buckets[(bz * n + bx) as usize].extend_from_slice(tri);
    }

    let mut out = Vec::with_capacity(indices.len());
    let mut ranges = Vec::new();
    for b in buckets {
        if b.is_empty() {
            continue;
        }
        ranges.push((out.len() as u32, b.len() as u32));
        out.extend_from_slice(&b);
    }
    (out, ranges)
}

/// Load a scene's terrain and convert it for the solid pipeline.
///
/// Returns `None` rather than failing the frame when the asset is missing: a
/// level that renders without ground is diagnosable from the log, and a client
/// that refuses to start is not.
pub fn load(
    def: &TerrainDef,
    game_dir: &std::path::Path,
    step: u32,
    // Every brush in the level. Terrain buried under their floors is hidden or
    // lowered in the rendered copy; see `bury_under_structures`.
    structures: &[&BrushDef],
) -> Option<TerrainGeometry> {
    let source = match space_soup_engine::terrain::load(def, game_dir) {
        Ok(s) => s,
        Err(e) => {
            log::warn!("terrain_render: {e}");
            return None;
        }
    };

    let bounds = source.bounds();
    let patch = source.patch(bounds, step.max(1));
    if patch.positions.is_empty() || patch.indices.is_empty() {
        log::warn!("terrain_render: terrain produced no geometry");
        return None;
    }

    let normals = vertex_normals(&patch.positions, &patch.indices);

    // Normalised position over the footprint, for sampling the splat map. Taken
    // from `bounds` rather than from the TerrainDef's declared size so it stays
    // correct for any TerrainSource -- the trait is what a future voxel or
    // chunked representation implements, and the def's shape is not.
    //
    // Guarded against a zero-extent axis: a degenerate terrain would otherwise
    // produce NaN uvs, and a NaN texture coordinate does not error, it just
    // samples something arbitrary and paints the ground with it.
    let extent = bounds.max - bounds.min;
    let inv_x = if extent.x.abs() > 1e-6 { 1.0 / extent.x } else { 0.0 };
    let inv_z = if extent.z.abs() > 1e-6 { 1.0 / extent.z } else { 0.0 };
    // Normals come from the ground as authored, before any vertex is lowered:
    // two centimetres under a floor must not tilt the lighting of the ground
    // beside the wall.
    let solids: Vec<BrushSolid> = structures.iter().flat_map(|b| b.evaluate()).collect();
    let mut positions = patch.positions.clone();
    let (kept_indices, burial) = bury_under_structures(&mut positions, &patch.indices, &solids);
    if burial.hidden_triangles > 0 || burial.lowered_vertices > 0 {
        log::info!(
            "terrain_render: {} triangle(s) under structure floors hidden, {} vertex(es) lowered {} m (render only)",
            burial.hidden_triangles,
            burial.lowered_vertices,
            BURIED_DROP,
        );
    }
    let vertices: Vec<SolidVertex> = positions
        .iter()
        .zip(normals.iter())
        .map(|(p, n)| SolidVertex {
            position: [p.x, p.y, p.z],
            normal: [n.x, n.y, n.z],
            color: ground_colour(n.y),
            // Terrain is lit dynamically and has no lightmap, so the slot the
            // cuboid path uses for atlas coordinates is free. It carries the
            // splat uv instead -- one less vertex attribute, and the two can
            // never both be wanted on the same draw.
            uv2: [
                (p.x - bounds.min.x) * inv_x,
                (p.z - bounds.min.z) * inv_z,
            ],
            reflectivity: 0.0,
        })
        .collect();

    log::info!(
        "terrain_render: {} vertices, {} triangles (step {step})",
        patch.positions.len(),
        patch.indices.len() / 3
    );
    // 8x8 buckets over the footprint. On a 110 m field that is ~14 m a side --
    // comfortably smaller than the sun's box and larger than a small spot's
    // cone, so a 4 m lamp touches one or two chunks instead of all 32k
    // triangles.
    let (indices, chunk_ranges) = chunk_indices(&vertices, &kept_indices, CHUNK_TARGET);
    log::info!(
        "terrain_render: {} caster chunks over {} triangles",
        chunk_ranges.len(),
        indices.len() / 3,
    );
    Some(TerrainGeometry {
        vertices,
        indices,
        posed: Vec::new(),
        built_for: None,
        chunk_ranges,
        chunk_bounds: Vec::new(),
    })
}

/// Buckets per axis for shadow-caster chunking.
///
/// A tradeoff with no sharp optimum: more chunks cull tighter and cost more
/// per-frame bounds work and more draw calls. Eight squares the field into 64,
/// which on a tile GPU is a handful of extra draws against tens of thousands of
/// triangles saved.
const CHUNK_TARGET: u32 = 8;

/// Area-weighted vertex normals from the triangles.
///
/// Computed here rather than taken from the heightfield gradient because this
/// runs on whatever `patch` produced: at a coarse LOD step the triangles are not
/// the sample grid any more, and shading them by the fine-grained gradient would
/// light a surface that is not the one being drawn.
fn vertex_normals(positions: &[Vec3], indices: &[u32]) -> Vec<Vec3> {
    let mut normals = vec![Vec3::ZERO; positions.len()];
    for tri in indices.chunks_exact(3) {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        // Not normalised: the cross product's magnitude is twice the triangle
        // area, which is exactly the weighting a shared vertex should get.
        let face = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
        normals[a] += face;
        normals[b] += face;
        normals[c] += face;
    }
    for n in &mut normals {
        *n = n.normalize_or_zero();
        if *n == Vec3::ZERO {
            *n = Vec3::Y;
        }
    }
    normals
}

/// Flat ground reads greener, steep faces read as rock.
///
/// A single colour makes a sculpted landscape unreadable in the headset --
/// without a slope cue there is nothing to tell a gentle rise from a cliff until
/// you walk into it.
fn ground_colour(normal_y: f32) -> [f32; 4] {
    let steepness = (1.0 - normal_y.clamp(0.0, 1.0)).clamp(0.0, 1.0);
    let grass = Vec3::new(0.30, 0.38, 0.22);
    let rock = Vec3::new(0.34, 0.32, 0.30);
    let c = grass.lerp(rock, steepness.powf(0.6));
    [c.x, c.y, c.z, 1.0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_ground_normals_point_up() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
        ];
        // Counter-clockwise seen from above, matching patch().
        let indices = vec![0, 2, 1, 1, 2, 3];
        for n in vertex_normals(&positions, &indices) {
            assert!((n.y - 1.0).abs() < 1e-5, "expected up, got {n:?}");
        }
    }

    #[test]
    fn a_vertex_with_no_triangles_still_gets_a_usable_normal() {
        // normalize_or_zero would otherwise hand the shader a zero vector.
        let positions = vec![Vec3::ZERO, Vec3::X, Vec3::Z, Vec3::new(9.0, 0.0, 9.0)];
        let normals = vertex_normals(&positions, &[0, 2, 1]);
        assert_eq!(normals[3], Vec3::Y);
    }

    #[test]
    fn steep_ground_is_coloured_differently_from_flat() {
        assert_ne!(ground_colour(1.0), ground_colour(0.1));
    }

    #[test]
    fn colours_stay_in_range() {
        for y in [0.0, 0.25, 0.5, 0.75, 1.0] {
            for c in ground_colour(y) {
                assert!((0.0..=1.0).contains(&c), "colour component {c} out of range");
            }
        }
    }

    /// The splat uv must span the whole footprint, corner to corner. A uv that
    /// only ever reaches 0.5 samples a quarter of the map over the whole level
    /// and looks like a painting mistake rather than a coordinate bug.
    #[test]
    fn splat_uvs_span_the_footprint() {
        use space_soup_engine::terrain::{Heightfield, TerrainSource};

        // 5x5 samples over 40x40 metres, offset so a bug that forgets the
        // origin cannot pass by accident.
        let field = Heightfield::new(
            vec![0u16; 25],
            [5, 5],
            [40.0, 40.0],
            [0.0, 10.0],
            Vec3::new(100.0, 0.0, -60.0),
        )
        .expect("build test heightfield");

        let bounds = field.bounds();
        let patch = field.patch(bounds, 1);
        let extent = bounds.max - bounds.min;

        let uvs: Vec<[f32; 2]> = patch
            .positions
            .iter()
            .map(|p| [
                (p.x - bounds.min.x) / extent.x,
                (p.z - bounds.min.z) / extent.z,
            ])
            .collect();

        let min_u = uvs.iter().map(|c| c[0]).fold(f32::INFINITY, f32::min);
        let max_u = uvs.iter().map(|c| c[0]).fold(f32::NEG_INFINITY, f32::max);
        let min_v = uvs.iter().map(|c| c[1]).fold(f32::INFINITY, f32::min);
        let max_v = uvs.iter().map(|c| c[1]).fold(f32::NEG_INFINITY, f32::max);

        assert!(min_u.abs() < 1e-5, "u should reach 0, got {min_u}");
        assert!((max_u - 1.0).abs() < 1e-5, "u should reach 1, got {max_u}");
        assert!(min_v.abs() < 1e-5, "v should reach 0, got {min_v}");
        assert!((max_v - 1.0).abs() < 1e-5, "v should reach 1, got {max_v}");

        for uv in &uvs {
            assert!(uv[0].is_finite() && uv[1].is_finite(), "uv must be finite: {uv:?}");
        }
    }
}

#[cfg(test)]
mod player_frame_tests {
    use super::*;

    fn geometry(positions: &[[f32; 3]]) -> TerrainGeometry {
        TerrainGeometry {
            vertices: positions
                .iter()
                .map(|p| SolidVertex {
                    position: *p,
                    normal: [0.0, 1.0, 0.0],
                    color: [1.0, 1.0, 1.0, 1.0],
                    uv2: [0.0, 0.0],
                    reflectivity: 0.0,
                })
                .collect(),
            indices: vec![0, 1, 2],
            posed: Vec::new(),
            built_for: None,
            chunk_ranges: Vec::new(),
            chunk_bounds: Vec::new(),
        }
    }

    #[test]
    fn walking_moves_the_ground_past_the_player() {
        // The bug this fixes. Terrain was handed to the renderer in world space
        // while everything else was transformed, so the ground travelled with
        // the player and the room appeared to slide instead.
        let mut g = geometry(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
        let at_origin = g
            .assemble(Vec3::ZERO, Quat::IDENTITY, 0.0)
            .map(|(v, _)| v[0].position)
            .unwrap();
        let walked = g
            .assemble(Vec3::new(0.0, 0.0, 5.0), Quat::IDENTITY, 0.0)
            .map(|(v, _)| v[0].position)
            .unwrap();
        assert_eq!(at_origin, [0.0, 0.0, 0.0]);
        assert_eq!(walked, [0.0, 0.0, -5.0], "ground must fall behind the player");
    }

    #[test]
    fn turning_rotates_the_ground_about_the_player() {
        let mut g = geometry(&[[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
        let turned = g
            .assemble(
                Vec3::ZERO,
                Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
                std::f32::consts::FRAC_PI_2,
            )
            .map(|(v, _)| v[0].position)
            .unwrap();
        assert!((turned[0] - -1.0).abs() < 1e-5, "got {turned:?}");
        assert!(turned[2].abs() < 1e-5, "got {turned:?}");
    }

    #[test]
    fn normals_are_rotated_but_never_translated() {
        // A normal is a direction. Subtracting the player's position from it
        // would tilt the ground's lighting by however far they had walked --
        // the ground would get darker the further you went.
        let mut g = geometry(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
        let n = g
            .assemble(Vec3::new(12.0, 3.0, -7.0), Quat::IDENTITY, 0.0)
            .map(|(v, _)| v[0].normal)
            .unwrap();
        assert_eq!(n, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn standing_still_reuses_the_previous_build() {
        // 16k vertices rebuilt every frame would be real cost for no change.
        let mut g = geometry(&[[1.0, 2.0, 3.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
        g.assemble(Vec3::ONE, Quat::IDENTITY, 0.0);
        let first = g.built_for;
        g.assemble(Vec3::ONE, Quat::IDENTITY, 0.0);
        assert_eq!(first, g.built_for);
        assert_eq!(g.posed.len(), 3);
    }

    #[test]
    fn empty_terrain_asks_for_no_draw() {
        let mut g = TerrainGeometry {
            vertices: Vec::new(),
            indices: Vec::new(),
            posed: Vec::new(),
            built_for: None,
            chunk_ranges: Vec::new(),
            chunk_bounds: Vec::new(),
        };
        assert!(g.assemble(Vec3::ZERO, Quat::IDENTITY, 0.0).is_none());
    }
}


#[cfg(test)]
mod burial_tests {
    use super::*;
    use space_soup_engine::brush::block_solid;

    /// An n x n grid of vertices at height `y`, spacing 1 m, centred on the origin.
    fn flat(n: u32, y: f32) -> (Vec<Vec3>, Vec<u32>) {
        let half = (n - 1) as f32 * 0.5;
        let mut p = Vec::new();
        for iz in 0..n {
            for ix in 0..n {
                p.push(Vec3::new(ix as f32 - half, y, iz as f32 - half));
            }
        }
        let mut idx = Vec::new();
        for iz in 0..n - 1 {
            for ix in 0..n - 1 {
                let a = iz * n + ix;
                let (b, c, d) = (a + 1, a + n, a + n + 1);
                idx.extend([a, c, b, b, c, d]);
            }
        }
        (p, idx)
    }

    #[test]
    fn terrain_under_a_floor_is_hidden_and_the_edge_is_lowered() {
        let (mut p, idx) = flat(9, -0.0001);
        let before = p.clone();
        let slab = block_solid([-2.0, -0.3, -2.0], [2.0, 0.0, 2.0], "m");
        let (kept, burial) = bury_under_structures(&mut p, &idx, &[slab]);
        assert!(burial.hidden_triangles > 0, "nothing under the floor was hidden");
        assert!(kept.len() < idx.len());
        for (i, (a, b)) in before.iter().zip(&p).enumerate() {
            let inside = a.x.abs() <= 2.0 && a.z.abs() <= 2.0;
            let dy = a.y - b.y;
            if inside {
                assert!((dy - BURIED_DROP).abs() < 1e-6, "buried vertex {i} at {a:?} was not lowered");
            } else {
                assert_eq!(dy, 0.0, "vertex {i} outside the structure moved");
            }
        }
    }

    #[test]
    fn no_hidden_triangle_reaches_outside_the_structure() {
        let (mut p, idx) = flat(9, -0.0001);
        let original = p.clone();
        let slab = block_solid([-2.0, -0.3, -2.0], [2.0, 0.0, 2.0], "m");
        let (kept, _) = bury_under_structures(&mut p, &idx, &[slab]);
        let kept_set: std::collections::HashSet<[u32; 3]> =
            kept.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
        for t in idx.chunks_exact(3) {
            if kept_set.contains(&[t[0], t[1], t[2]]) {
                continue;
            }
            for &i in t {
                let v = original[i as usize];
                assert!(v.x.abs() <= 2.0 + 1e-4 && v.z.abs() <= 2.0 + 1e-4,
                    "a hidden triangle has a vertex outside the floor at {v:?}: that is a gap");
            }
        }
    }

    #[test]
    fn a_structure_raised_off_the_ground_hides_nothing() {
        let (mut p, idx) = flat(9, 0.0);
        let before = p.clone();
        let bridge = block_solid([-2.0, 2.0, -2.0], [2.0, 2.3, 2.0], "m");
        let (kept, burial) = bury_under_structures(&mut p, &idx, &[bridge]);
        assert_eq!(burial, Burial::default());
        assert_eq!(kept, idx);
        assert_eq!(p, before);
    }

    /// THE REAL LEVEL: nowhere inside either room may drawn terrain come within
    /// 2 mm of the floor at y = 0 -- about twice the depth precision at 20 m with
    /// this near plane, so a fight there is impossible. Before this existed the
    /// terrain sat 0.1 mm under the floor.
    /// NO HOLES: a triangle is hidden only if structure covers ALL of it, not
    /// just its three corners. At an inside corner -- the hallway meeting a hall
    /// -- a triangle can have every corner under a building and still reach
    /// out over open ground, and hiding it opens a hole in the grass there
    /// (headset, 2026-09-24).
    #[test]
    fn no_hidden_terrain_triangle_leaves_a_hole_in_the_shipped_level() {
        let game = concat!(env!("CARGO_MANIFEST_DIR"), "/../game");
        let scene = space_soup_engine::scene::Scene::load(std::path::Path::new(&format!("{game}/scenes/test_room.json")))
            .expect("test_room.json should load");
        let def = scene.terrain.as_ref().expect("test_room has terrain");
        let source = space_soup_engine::terrain::load(def, std::path::Path::new(game)).expect("terrain loads");
        let patch = source.patch(source.bounds(), 1);
        let solids: Vec<BrushSolid> = scene.objects.iter().filter_map(|o| o.brush.as_ref()).flat_map(|b| b.evaluate()).collect();
        let mut p = patch.positions.clone();
        let original = p.clone();
        let (kept, _) = bury_under_structures(&mut p, &patch.indices, &solids);
        let kept_set: std::collections::HashSet<[u32; 3]> = kept.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
        let covered = |q: Vec3| {
            let probe = [q.x as f64, (q.y - BURIED_PROBE_DEPTH) as f64, q.z as f64];
            solids.iter().any(|s| contains_point(s, probe, 1e-4))
        };
        let mut holes = Vec::new();
        for t in patch.indices.chunks_exact(3) {
            if kept_set.contains(&[t[0], t[1], t[2]]) {
                continue;
            }
            let (a, b, c) = (original[t[0] as usize], original[t[1] as usize], original[t[2] as usize]);
            'samples: for i in 0..=6 {
                for j in 0..=(6 - i) {
                    let q = a + (b - a) * (i as f32 / 6.0) + (c - a) * (j as f32 / 6.0);
                    if !covered(q) {
                        holes.push((q.x, q.z));
                        break 'samples;
                    }
                }
            }
        }
        assert!(holes.is_empty(), "{} hidden triangle(s) reach open ground, first at {:?}", holes.len(), &holes[..holes.len().min(6)]);
    }

    #[test]
    fn inside_the_shipped_rooms_no_drawn_terrain_is_near_the_floor() {
        let game = concat!(env!("CARGO_MANIFEST_DIR"), "/../game");
        let scene = space_soup_engine::scene::Scene::load(std::path::Path::new(&format!("{game}/scenes/test_room.json")))
            .expect("test_room.json should load");
        let def = scene.terrain.as_ref().expect("test_room has terrain");
        let source = space_soup_engine::terrain::load(def, std::path::Path::new(game)).expect("terrain loads");
        let patch = source.patch(source.bounds(), 1);
        let structures: Vec<&BrushDef> = scene.objects.iter().filter_map(|o| o.brush.as_ref()).collect();
        let solids: Vec<BrushSolid> = structures.iter().flat_map(|b| b.evaluate()).collect();
        let mut p = patch.positions.clone();
        let (kept, burial) = bury_under_structures(&mut p, &patch.indices, &solids);
        assert!(burial.hidden_triangles > 0, "no terrain under the buildings was hidden");

        // THE ROOMS AS THE LEVEL'S OWN CARVES DEFINE THEM, not as coordinates
        // written here: this listed the brick room's interior by number, and
        // when the room moved east the test went on checking the open lawn
        // where it used to stand.
        let named: Vec<(&str, &BrushDef)> =
            scene.objects.iter().filter_map(|o| o.brush.as_ref().map(|b| (o.id.as_str(), b))).collect();
        let (rooms, _) = space_soup_engine::room_graph::rooms_from_scene(&named);
        assert!(rooms.len() >= 3, "expected the hall, the hallway and the brick room, got {}", rooms.len());
        let in_room = |x: f32, z: f32| {
            rooms.iter().any(|r| x > r.min.x && x < r.max.x && z > r.min.z && z < r.max.z)
        };
        let mut worst = f32::NEG_INFINITY;
        for t in kept.chunks_exact(3) {
            let (a, b, c) = (p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]);
            // Sample the triangle's interior, not just its corners: the height
            // that matters is wherever the triangle passes under the room.
            for i in 0..=8 {
                for j in 0..=(8 - i) {
                    let (u, v) = (i as f32 / 8.0, j as f32 / 8.0);
                    let q = a + (b - a) * u + (c - a) * v;
                    if in_room(q.x, q.z) {
                        worst = worst.max(q.y);
                    }
                }
            }
        }
        assert!(worst <= -0.002, "drawn terrain reaches {worst} m inside a room -- within 2 mm of the floor, so it can z-fight");
    }
}
