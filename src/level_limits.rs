//! A LEVEL AGAINST THE ENGINE'S FIXED TABLES, SAID OUT LOUD.
//!
//! Several things a level has are kept in tables of a fixed size -- the
//! lights a frame shades, the stationary lamps' shadow masks, the buildings
//! reflections can meet, the weather's map layers, the brush materials -- and
//! a level past one of them used to lose the rest IN SILENCE: lamp 65 shone
//! through walls, the fifth area of weather never rained, the ninth building
//! was missing from reflections (`docs/scale-audit-2026-10-13.md`, F5-F7,
//! and its budget line: "a level over the limit is REFUSED at bake or load
//! with a message, never silently thinned").
//!
//! [`table_use`] counts a scene against each table; [`report`] logs every
//! table it is over as an ERROR, once, at load -- with what the player will
//! see of it -- and the scale bench prints the same lines
//! (`scale_bench budget`: `table.*`), so a level that outgrows a table is
//! known before anyone looks for the missing lamp.
//!
//! It does not make the tables larger, and it does not refuse the level: a
//! level that loads with a lamp unshadowed is diagnosable, one that will not
//! start is not. The tables that follow what is near and seen instead
//! (the probes' sixteen, by `select_resident_probes`) are not in it.

use space_soup_engine::scene::Scene;
use space_soup_engine::scene_light::LightMode;

/// One table, and a level's use of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableUse {
    pub table: &'static str,
    pub used: usize,
    pub limit: usize,
    /// What a level over the limit shows.
    pub over: &'static str,
}

impl TableUse {
    pub fn is_over(&self) -> bool {
        self.used > self.limit
    }
}

/// How many stationary lamps a level's shadow masks can tell apart at all
/// (`space_soup_engine::stationary::assign_channels` names lamps by bit).
pub const MAX_STATIONARY_LAMPS: usize = 64;

/// `scene` against every fixed table. What it costs: one pass over the
/// level's objects, at load.
pub fn table_use(scene: &Scene) -> Vec<TableUse> {
    let lamps = |mode: fn(LightMode) -> bool| scene.objects.iter().flat_map(|o| o.lights.iter()).filter(|l| mode(l.mode)).count();
    let mut materials: Vec<String> = scene.objects.iter().filter_map(|o| o.brush.as_ref()).flat_map(|b| b.materials()).collect();
    materials.sort();
    materials.dedup();
    let washed_shores = usize::from(crate::water_render::sea_line(&scene.water).is_some());
    vec![
        TableUse {
            table: "lights shaded a frame",
            used: lamps(|m| m != LightMode::Baked),
            limit: space_soup::renderer::lights::MAX_LIGHTS,
            over: "the level has more live lamps than a frame shades (the sun, the torch and every fire take slots too): the 16 are chosen by distance from the player, so where more are in reach, which lamps are lit changes as the player walks",
        },
        TableUse {
            table: "stationary lamps with a shadow mask",
            used: lamps(|m| m == LightMode::Stationary),
            limit: MAX_STATIONARY_LAMPS,
            over: "lamps past the 64th have no baked shadow and shine through walls",
        },
        TableUse {
            table: "buildings in outdoor reflections",
            used: scene.objects.iter().filter(|o| o.brush.is_some()).count(),
            limit: space_soup::renderer::uniforms::MAX_BUILDINGS,
            over: "only the 8 largest buildings appear in reflections seen from outdoors",
        },
        TableUse {
            table: "weather map layers (areas + washed shore)",
            used: scene.weather.len() + washed_shores,
            limit: space_soup_engine::weather::MAX_AREAS,
            over: "the shore keeps its layer and the areas nearest the player's start the rest; the other areas have no rain, snow or wet ground",
        },
        TableUse {
            table: "brush materials",
            used: materials.len(),
            limit: space_soup::renderer::brush_pipeline::MAX_BRUSH_MATERIALS,
            over: "faces of the materials past the 24th draw with the 24th",
        },
    ]
}

/// Every table `scene` is over, logged as an error with what it shows.
/// Returns how many.
pub fn report(scene_name: &str, scene: &Scene) -> usize {
    let over: Vec<TableUse> = table_use(scene).into_iter().filter(TableUse::is_over).collect();
    for t in &over {
        log::error!("LEVEL OVER A FIXED TABLE: '{scene_name}' has {} for '{}', which holds {}: {}", t.used, t.table, t.limit, t.over);
    }
    over.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene(json: serde_json::Value) -> Scene {
        serde_json::from_value(json).expect("a scene")
    }

    fn lamp(id: &str, mode: &str, n: usize) -> serde_json::Value {
        serde_json::json!({
            "id": id,
            "cuboid": { "position": [0.0, 2.0, 0.0], "half_size": [0.1, 0.1, 0.1] },
            "lights": (0..n).map(|_| serde_json::json!({ "kind": "Point", "mode": mode, "intensity": 5.0, "range": 6.0 })).collect::<Vec<_>>(),
        })
    }

    /// The shipped level is inside every table: nothing is reported for it.
    #[test]
    fn the_shipped_level_is_inside_every_table() {
        let game = crate::offline_frame::offline_game_dir();
        let Ok(scene) = Scene::load(&space_soup_engine::Manifest::scene_path(&game, "test_room")) else {
            eprintln!("skipping: no test_room");
            return;
        };
        let use_ = table_use(&scene);
        assert!(use_.iter().all(|t| !t.is_over()), "{use_:?}");
        assert!(use_.iter().any(|t| t.used > 0));
    }

    /// A LEVEL PAST A TABLE IS NAMED, table by table: 70 stationary lamps and
    /// 9 live ones, 9 buildings' worth of weather -- each over its own table
    /// and no other -- where the same level loaded with nothing said.
    #[test]
    fn a_level_past_a_table_is_named() {
        let mut objects = vec![lamp("stationary", "Stationary", 70), lamp("live", "Realtime", 9), lamp("baked", "Baked", 40)];
        objects.push(serde_json::json!({ "id": "plain", "cuboid": { "position": [0.0, 0.0, 0.0], "half_size": [1.0, 1.0, 1.0] } }));
        let weather: Vec<serde_json::Value> = (0..5).map(|k| serde_json::json!({ "name": format!("a{k}"), "bounds": [k as f32 * 50.0, 0.0, k as f32 * 50.0 + 20.0, 20.0] })).collect();
        let s = scene(serde_json::json!({ "name": "big", "objects": objects, "weather": weather }));
        let by_name = |name: &str| table_use(&s).into_iter().find(|t| t.table.starts_with(name)).unwrap();
        // Baked lamps are in the lightmaps: they take no slot.
        assert_eq!((by_name("lights").used, by_name("lights").is_over()), (79, true));
        assert_eq!((by_name("stationary").used, by_name("stationary").is_over()), (70, true));
        assert_eq!((by_name("weather").used, by_name("weather").is_over()), (5, true));
        assert!(!by_name("buildings").is_over() && !by_name("brush materials").is_over());
        assert_eq!(report("big", &s), 3);
        // Four areas and no sea fit; a washed sea beside them does not.
        let fits = scene(serde_json::json!({ "name": "fits", "objects": [], "weather": weather[..4] }));
        assert_eq!(report("fits", &fits), 0);
        let sea = scene(serde_json::json!({ "name": "sea", "objects": [], "weather": weather[..4], "water": [{ "height": 0.0, "bounds": [-20.0, -20.0, 20.0, 20.0], "swash": 0.3 }] }));
        assert_eq!(report("sea", &sea), 1);
    }
}
