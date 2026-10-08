//! Turning a scene's `WaterDef` into something the renderer can draw.
//!
//! The thing this does that the engine crate cannot: sample the loaded terrain
//! for depth. The surface stays in the WORLD -- the water's vertex shader poses
//! it into the player's frame -- so it is built once and never re-uploaded as
//! the player walks.

use space_soup::renderer::water_pipeline::{WaterOptics, WaterUniform, WaterVertex};
use space_soup::renderer::water_waves::WaveParams;
use space_soup_engine::water::{WaterDef, OPEN_WATER_DEPTH};

/// One body's world-space surface, plus its optics and the sea it raises.
pub struct WaterBody {
    pub world: Vec<WaterVertex>,
    pub indices: Vec<u32>,
    pub uniform: WaterUniform,
    pub waves: WaveParams,
}

fn srgb_to_linear(c: u8) -> f32 {
    let s = c as f32 / 255.0;
    if s <= 0.040_45 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
}

/// The sea a body's wind raises: its own seed, so two bodies never wave in
/// step, and its typical depth for how fast the waves run.
fn wave_params(def: &WaterDef, depth: f32, seed: u64) -> WaveParams {
    WaveParams {
        wind_speed: def.wind_speed.max(0.0),
        wind_dir: def.wind_direction.to_radians(),
        fetch: def.fetch.max(1.0),
        depth,
        choppiness: def.choppiness.clamp(0.0, 1.5),
        seed: WaveParams::default().seed ^ seed.wrapping_mul(0x9e37_79b9_7f4a_7c15),
        ..WaveParams::default()
    }
}

/// Build every body of water in a scene.
///
/// Colours are converted to LINEAR here. They are authored as sRGB bytes,
/// because that is what a colour picker produces, and the shader works in
/// linear -- feeding it the raw bytes makes the water far too clear and too
/// bright.
pub fn build(
    defs: &[WaterDef],
    terrain: Option<&dyn space_soup_engine::terrain::TerrainSource>,
) -> Vec<WaterBody> {
    let mut out = Vec::new();
    for (index, def) in defs.iter().enumerate() {
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
        // The waves' dispersion depth: the water's typical depth where it is
        // over known ground, as a wave feels the bottom a few metres down.
        let known: Vec<f32> = verts.iter().map(|v| v.depth).filter(|&d| d > 0.0 && d < OPEN_WATER_DEPTH).collect();
        let depth = if known.is_empty() { 30.0 } else { (known.iter().sum::<f32>() / known.len() as f32).clamp(0.5, 30.0) };
        let waves = wave_params(def, depth, index as u64);
        let lin = |c: [u8; 3]| [srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2])];
        let uniform = WaterUniform::new(
            &WaterOptics {
                height: def.height,
                shallow: lin(def.shallow),
                deep: lin(def.deep),
                depth_scale: def.depth_scale,
                shore_fade: def.shore_fade,
                swash: def.swash,
                swell: def.swell,
            },
            &waves,
        );
        log::info!(
            "water at y={:.2}: {} vertices, {} triangles; {:.1} m/s over {:.0} m, waves {:.2} m high, {:.1} m deep",
            def.height,
            world.len(),
            indices.len() / 3,
            waves.wind_speed,
            waves.fetch,
            waves.significant_height(),
            depth,
        );
        out.push(WaterBody { world, indices, uniform, waves });
    }
    out
}

/// WET SAND: the line below which the ground is wet, world y, and the damp
/// band above it -- from the bodies whose waves wash a shore (a lake's still
/// edge leaves no band): their surface plus the wash's run-up, which reaches
/// further up a beach than the wash rises, and the highest of them.
pub fn wet_shore(defs: &[WaterDef]) -> Option<(f32, f32)> {
    defs.iter()
        .filter(|d| d.swash > 0.0)
        .map(|d| (d.height + 1.3 * d.swash, 0.25))
        .reduce(|a, b| if a.0 >= b.0 { a } else { b })
}

#[cfg(test)]
mod tests {
    use super::*;
    use space_soup::renderer::water_pipeline::LIGHT_PATH;

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
        // sRGB 188 is linear ~0.5: a white bed through `depth_scale` metres
        // must come out at half, not at 0.737, which is what reading the byte
        // unconverted would make it -- water far too clear.
        let mut d = def();
        d.shallow = [188, 188, 188];
        let bodies = build(&[d.clone()], None);
        let k = bodies[0].uniform.extinction[0];
        let seen = (-k * d.depth_scale * (1.0 + LIGHT_PATH)).exp();
        assert!((seen - 0.5).abs() < 0.02, "sRGB 188 must see through to linear ~0.5, got {seen}");
    }

    #[test]
    fn the_waves_follow_the_authored_wind() {
        let mut d = def();
        d.wind_speed = 9.0;
        d.wind_direction = 90.0;
        d.fetch = 3000.0;
        let w = build(&[d], None).remove(0).waves;
        assert_eq!((w.wind_speed, w.fetch), (9.0, 3000.0));
        assert!((w.wind_dir - std::f32::consts::FRAC_PI_2).abs() < 1e-6, "degrees must become radians: {}", w.wind_dir);
    }

    #[test]
    fn only_a_washing_shore_leaves_wet_sand() {
        let lake = def();
        assert_eq!(wet_shore(&[lake.clone()]), None);
        let mut sea = def();
        sea.swash = 0.2;
        let (line, band) = wet_shore(&[lake, sea]).unwrap();
        assert!((line - 1.26).abs() < 1e-5 && band > 0.0);
    }

    #[test]
    fn two_bodies_do_not_wave_in_step() {
        let bodies = build(&[def(), def()], None);
        assert_ne!(bodies[0].waves.seed, bodies[1].waves.seed);
    }
}
