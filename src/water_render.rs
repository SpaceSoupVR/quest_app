//! Turning a scene's `WaterDef` into something the renderer can draw.
//!
//! The two things this does that the engine crate cannot: sample the loaded
//! terrain for depth, and carry the surface into the PLAYER's frame every time
//! the player moves -- the same treatment brushes and terrain get, and for the
//! same reason. Geometry left in world space stays glued to the player while
//! everything else slides past.

use glam::{Quat, Vec3};
use space_soup::renderer::water_pipeline::{WaterUniform, WaterVertex};
use space_soup_engine::water::WaterDef;

/// One body's world-space surface, plus the optics the shader needs.
pub struct WaterBody {
    world: Vec<WaterVertex>,
    pub indices: Vec<u32>,
    pub uniform: WaterUniform,
    posed: Vec<WaterVertex>,
    built_for: Option<(Vec3, f32)>,
}

impl WaterBody {
    /// The surface in the player's frame, or `None` if it has not changed.
    ///
    /// `None` rather than the unchanged slice, so the caller cannot accidentally
    /// re-upload a lake's worth of vertices every frame for a player standing
    /// still. Water is large, static and cheap to leave alone -- the only thing
    /// that moves is the frame it is expressed in.
    pub fn assemble(
        &mut self,
        offset: Vec3,
        yaw_inv: Quat,
        player_yaw: f32,
    ) -> Option<&[WaterVertex]> {
        let key = (offset, player_yaw);
        if self.built_for == Some(key) {
            return None;
        }
        self.posed.clear();
        self.posed.reserve(self.world.len());
        for v in &self.world {
            let p = yaw_inv * (Vec3::from(v.position) - offset);
            self.posed.push(WaterVertex { position: p.to_array(), depth: v.depth });
        }
        self.built_for = Some(key);
        Some(&self.posed)
    }

    /// The most recent player-frame surface, whether or not it was just rebuilt.
    pub fn posed(&self) -> &[WaterVertex] {
        &self.posed
    }

    /// The surface in WORLD space, as it was tessellated.
    ///
    /// What the renderer's buffers are sized and seeded from: `posed` is empty
    /// until the first frame, and a zero-length upload would have the body
    /// dropped as if the level had no water in it.
    pub fn world(&self) -> &[WaterVertex] {
        &self.world
    }
}

fn srgb_to_linear(c: u8) -> f32 {
    let s = c as f32 / 255.0;
    if s <= 0.040_45 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
}

/// Build every body of water in a scene.
///
/// Colours are converted to LINEAR here. They are authored as sRGB bytes,
/// because that is what a colour picker produces, and the shader mixes and
/// tonemaps in linear -- feeding it the raw bytes makes shallow water far too
/// bright and the depth gradient the wrong shape.
pub fn build(
    defs: &[WaterDef],
    terrain: Option<&dyn space_soup_engine::terrain::TerrainSource>,
) -> Vec<WaterBody> {
    let mut out = Vec::new();
    for def in defs {
        let ground = |x: f32, z: f32| terrain.and_then(|t| t.height_at(x, z));
        let Some((verts, indices)) = space_soup_engine::water::build_surface(def, ground) else {
            log::info!(
                "water at y={:.2}: nothing of it is below the ground, skipped",
                def.height,
            );
            continue;
        };
        let world: Vec<WaterVertex> = verts
            .iter()
            .map(|v| WaterVertex { position: v.position, depth: v.depth })
            .collect();
        let lin = |c: [u8; 3]| {
            [srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2]), 1.0]
        };
        log::info!(
            "water at y={:.2}: {} vertices, {} triangles",
            def.height,
            world.len(),
            indices.len() / 3,
        );
        out.push(WaterBody {
            posed: Vec::with_capacity(world.len()),
            world,
            indices,
            uniform: WaterUniform {
                shallow: lin(def.shallow),
                deep: lin(def.deep),
                params: [def.depth_scale, def.shore_fade, def.wave_scale, def.wave_strength],
                anim: [0.0, def.wave_speed, def.opacity, 0.0],
            },
            built_for: None,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn def() -> WaterDef {
        serde_json::from_str(r#"{"height": 1.0, "bounds": [-4, -4, 4, 4]}"#).unwrap()
    }

    #[test]
    fn a_body_with_no_terrain_still_builds() {
        // No heightfield means the ground sampler always returns None, which
        // `build_surface` reads as "off the edge of the terrain" -- deep, not
        // dry. A scene with water and no terrain is a flooded room, and it must
        // not silently render nothing.
        let bodies = build(&[def()], None);
        assert_eq!(bodies.len(), 1);
        assert!(!bodies[0].indices.is_empty());
    }

    #[test]
    fn colours_are_converted_out_of_srgb() {
        let mut d = def();
        d.shallow = [188, 188, 188];
        let bodies = build(&[d], None);
        let r = bodies[0].uniform.shallow[0];
        // sRGB 188 is linear ~0.5. Passing the byte through unconverted would
        // give 0.737, which is why unconverted water reads as far too bright.
        assert!(
            (r - 0.5).abs() < 0.02,
            "sRGB 188 must land near linear 0.5, got {r}",
        );
    }

    #[test]
    fn the_surface_moves_into_the_players_frame() {
        let mut bodies = build(&[def()], None);
        let body = &mut bodies[0];
        let offset = Vec3::new(10.0, 0.0, -5.0);
        let posed = body.assemble(offset, Quat::IDENTITY, 0.0).expect("first call rebuilds").to_vec();
        for (w, p) in body.world.iter().zip(posed.iter()) {
            let expect = Vec3::from(w.position) - offset;
            assert!(
                (Vec3::from(p.position) - expect).length() < 1e-5,
                "vertex was not carried into the player frame",
            );
            assert_eq!(p.depth, w.depth, "depth is a measurement, not a coordinate");
        }
    }

    #[test]
    fn standing_still_reuses_the_last_assembly() {
        let mut bodies = build(&[def()], None);
        let body = &mut bodies[0];
        assert!(body.assemble(Vec3::ZERO, Quat::IDENTITY, 0.0).is_some(), "first call must build");
        assert!(
            body.assemble(Vec3::ZERO, Quat::IDENTITY, 0.0).is_none(),
            "a stationary player must not re-pose the surface",
        );
        assert!(
            body.assemble(Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY, 0.0).is_some(),
            "moving must re-pose it",
        );
    }
}
