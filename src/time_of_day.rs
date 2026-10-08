//! The level's time of day as the app hands it to the renderer: the scene's
//! authored day, and the brush atlas's daylight layers checked against the
//! shipped bake. Not android-gated: the offline harness lights with the same.
//! See `space_soup::renderer::time_of_day`.

use space_soup::renderer::sky::{AtlasLayers, TimeOfDayParams};

/// How closely the daylight layers must add back to the shipped atlas to be
/// trusted: they are three bakes of the same level, so a level re-baked after
/// them (moved walls, new lamps) fails by far more than this.
pub const LAYERS_MATCH_TOLERANCE: f32 = 0.05;

/// The scene's own time of day, its north turned to the photographed sky's
/// sun (`photographed_sun`, world) when it does not say where north is.
pub fn scene_params(sky: Option<&space_soup_engine::SkyDef>, photographed_sun: Option<[f32; 3]>) -> Option<TimeOfDayParams> {
    sky.and_then(|s| s.time_of_day).map(|t| t.params(photographed_sun))
}

/// The time of day's parameters for a level with none of its own -- what the
/// `time_of_day_hour` lever and the offline harness's `TIME_OF_DAY` use: the
/// defaults, at `hour`, with north turned so the sun at the DEFAULT start hour
/// stands where the photograph's does (as the renderer turns it). Turned at
/// `hour` instead, every hour put the sun at the photograph's azimuth: it rose
/// and set in the same place (2026-10-08).
#[allow(dead_code)] // the offline harness's (not built for the headset)
pub fn default_params(hour: f32, photographed_sun: Option<[f32; 3]>) -> TimeOfDayParams {
    let p = TimeOfDayParams::default();
    let p = match photographed_sun {
        Some(s) => p.with_sun_azimuth_of(s),
        None => p,
    };
    TimeOfDayParams { start_hour: hour, ..p }
}

/// The brush atlas's daylight layers, when the scene has them and they belong
/// to `shipped` (its loaded `__brushes__` light). Says which, in the log.
pub fn daylight_layers(game_dir: &std::path::Path, scene_name: &str, shipped: Option<(&[f32], u32, u32)>) -> Option<AtlasLayers> {
    let layers = space_soup_engine::daylight::load_brush_layers(game_dir, scene_name)?;
    match shipped {
        Some((light, w, h)) if layers.matches(light, w, h, LAYERS_MATCH_TOLERANCE) => {
            log::info!("TIMEOFDAY daylight layers {}x{}: lamps, sky, sun", layers.width, layers.height);
            Some(layers)
        }
        _ => {
            log::warn!(
                "TIMEOFDAY daylight layers of '{scene_name}' do not add back to its shipped brush atlas (re-baked since?): \
                 not used; a moving sun keeps the bake's bounce. Re-run `bake daylight` and `BAKE_DAYLIGHT=off bake probe`."
            );
            None
        }
    }
}

/// THE OFFLINE HARNESS'S HOUR: `TIME_OF_DAY=hours` renders the level at that
/// local solar time under the time-of-day sky (its own, or the defaults with
/// north turned to its photograph's sun). Unset, the photograph, as shipped.
#[allow(dead_code)] // the offline harness's (not built for the headset)
pub fn offline_hour() -> Option<f32> {
    std::env::var("TIME_OF_DAY").ok().and_then(|v| v.trim().parse().ok())
}

/// The time of day the harness renders at `hour`: its parameters and the
/// sky's state then, and the stars' extinction.
#[allow(dead_code)] // the offline harness's (not built for the headset)
pub fn offline_snapshot(
    sky: Option<&space_soup_engine::SkyDef>,
    photographed_sun: Option<[f32; 3]>,
    hour: f32,
) -> (TimeOfDayParams, space_soup::renderer::sky::SkySnapshot, [f32; 3]) {
    let params = match scene_params(sky, photographed_sun) {
        Some(p) => TimeOfDayParams { start_hour: hour, ..p },
        None => default_params(hour, photographed_sun),
    };
    let tod = space_soup::renderer::sky::TimeOfDaySky::new(params);
    let snap = tod.snapshot(hour, 0.0);
    let zenith = tod.atmosphere.eye_transmittance([0.0, 1.0, 0.0]).map(|t| -t.max(1e-6).ln());
    (params, snap, zenith)
}

/// A probe photograph relit, as the headset relights it. See
/// `space_soup::renderer::time_of_day::relight_probe_faces`.
#[allow(unused_imports)] // the offline harness's (not built for the headset)
pub use space_soup::renderer::time_of_day::relight_probe_faces as relight_faces;

/// The lamps-only photographs (`BAKE_DAYLIGHT=off bake probe`), as a probe
/// level read from the daylight directory, when the level has them.
pub fn lamps_probe_level(game_dir: &std::path::Path, scene_name: &str) -> Option<crate::probe_level::ProbeLevel> {
    crate::probe_level::ProbeLevel::load(&space_soup_engine::daylight::daylight_dir(game_dir, scene_name), scene_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE SHIPPED LEVEL'S LAYERS ADD BACK TO ITS ATLAS, and at night leave the
    /// lamps' bounce alone. Skipped where the level has no layers.
    #[test]
    fn test_rooms_daylight_layers_belong_to_its_bake() {
        let game = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../game");
        let Some(layers) = space_soup_engine::daylight::load_brush_layers(&game, "test_room") else {
            eprintln!("skipping: test_room has no daylight layers");
            return;
        };
        let maps = space_soup_engine::lightmaps::load_scene_lightmaps(&game, "test_room");
        let full = maps.iter().find(|m| m.object_id == "__brushes__").unwrap();
        let light = full.linear.as_ref().unwrap();
        for tol in [0.01f32, 0.02, 0.05] {
            eprintln!("within {:.0}%: {}", tol * 100.0, layers.matches(light, full.width, full.height, tol));
        }
        assert!(layers.matches(light, full.width, full.height, LAYERS_MATCH_TOLERANCE));
        let sum = |v: &[f32]| v.chunks_exact(4).map(|p| (p[0] + p[1] + p[2]) as f64).sum::<f64>();
        let (lamps, sky, sun) = (sum(&layers.lamps), sum(&layers.sky), sum(&layers.sun));
        eprintln!("atlas light: lamps {lamps:.1}, sky {sky:.1}, sun {sun:.1}");
        assert!(lamps > 0.0 && sky > 0.0 && sun > 0.0);
    }

    /// The sun rises and sets on opposite sides of the meridian whatever hour
    /// is asked for, and at the default hour stands at the photograph's azimuth.
    #[test]
    fn the_harness_turns_north_once_not_per_hour() {
        let photo = [0.3775f32, 0.7417, 0.5544];
        let sun = |h: f32| {
            let p = default_params(h, Some(photo));
            space_soup::renderer::sky::TimeOfDaySky::new(p).snapshot(h, 0.0).sun.direction
        };
        let (rise, set) = (sun(5.0), sun(19.0));
        let angle = (rise[0] * set[0] + rise[2] * set[2]) / ((rise[0].hypot(rise[2])) * (set[0].hypot(set[2])));
        assert!(angle < 0.0, "sunrise {rise:?} and sunset {set:?} on the same side");
        let at = sun(TimeOfDayParams::default().start_hour);
        let (want, have) = (photo[0].atan2(photo[2]), at[0].atan2(at[2]));
        assert!((want - have).abs() < 0.01, "default hour's sun {at:?} is not the photograph's {photo:?}");
    }
}
