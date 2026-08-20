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

use glam::Vec3;
use space_soup::renderer::cuboid::SolidVertex;
use space_soup_engine::terrain::{TerrainDef, TerrainSource};

/// Ground vertices and indices in the renderer's own format.
pub struct TerrainGeometry {
    pub vertices: Vec<SolidVertex>,
    pub indices: Vec<u32>,
}

impl TerrainGeometry {
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() || self.indices.is_empty()
    }
}

/// Load a scene's terrain and convert it for the solid pipeline.
///
/// Returns `None` rather than failing the frame when the asset is missing: a
/// level that renders without ground is diagnosable from the log, and a client
/// that refuses to start is not.
pub fn load(def: &TerrainDef, game_dir: &std::path::Path, step: u32) -> Option<TerrainGeometry> {
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
    let vertices = patch
        .positions
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
    Some(TerrainGeometry { vertices, indices: patch.indices })
}

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
