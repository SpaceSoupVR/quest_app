//! The level's special effects -- fire, coals, smoke, embers, dust -- loaded
//! from the scene file by the client, as its lights are (`scene_lights`).
//!
//! An effect is level dressing: the headset already has the scene file that
//! run.sh pushed, so nothing is streamed, and a game with no server draws them
//! all the same. Each particle is a function of its emitter and the clock
//! (`space_soup::renderer::effects`), so the emitter's few numbers are the
//! whole of it.
use glam::Vec3;
use space_soup::renderer::effects::stack::{Placed, PlacedEffect};
use space_soup::renderer::effects::{EffectEmitter, EffectKind, Variation};
use space_soup::renderer::levers::Levers;
use space_soup_engine::scene::{EffectKindDef, Scene};
use space_soup_engine::Manifest;

/// Every object's effect in a scene, in the WORLD's frame: it rises along the
/// object's up, and dust fills the object's box. Hidden objects keep theirs --
/// an effect's object is usually a hidden marker, the effect its only look;
/// `rate` 0 is how one is switched off. Nothing for a scene that will not load.
pub(crate) fn load(game_dir: &std::path::Path, scene_name: &str) -> Vec<EffectEmitter> {
    let Ok(mut scene) = Scene::load(&Manifest::scene_path(game_dir, scene_name)) else {
        return Vec::new();
    };
    scene.resolve_world_transforms();
    scene
        .objects
        .iter()
        .filter_map(|o| {
            let e = o.effect.as_ref()?;
            // An effect that names an asset `game/effects` holds is drawn by
            // it (`placed`); its kind stays the shorthand for one that does not.
            if e.asset.as_deref().is_some_and(|a| names_an_asset(game_dir, a)) {
                return None;
            }
            let c = &o.cuboid;
            Some(EffectEmitter {
                id: o.id.clone(),
                kind: match e.kind {
                    EffectKindDef::Fire => EffectKind::Fire,
                    EffectKindDef::Smoke => EffectKind::Smoke,
                    EffectKindDef::Embers => EffectKind::Embers,
                    EffectKindDef::Dust => EffectKind::Dust,
                    EffectKindDef::Coals => EffectKind::Coals,
                },
                position: c.position,
                direction: c.rotation * Vec3::Y,
                extent: [
                    c.rotation * Vec3::X * c.half_size.x,
                    c.rotation * Vec3::Y * c.half_size.y,
                    c.rotation * Vec3::Z * c.half_size.z,
                ],
                scale: e.scale.max(0.0),
                rate: e.rate.max(0.0),
                tint: [srgb_to_linear(e.tint.0), srgb_to_linear(e.tint.1), srgb_to_linear(e.tint.2)],
                ceiling: None,
                // As the editor's preview reads them (`effectsSim.js`
                // `variationOf`): clamped to their ranges.
                variation: Variation {
                    intensity: e.intensity,
                    size_variation: e.size_variation,
                    life_variation: e.life_variation,
                    speed_variation: e.speed_variation,
                    temperature_variation: e.temperature_variation,
                    flicker: e.flicker,
                    flicker_rate: e.flicker_rate,
                    turbulence: e.turbulence,
                    seed: e.seed,
                }
                .clamped(),
            })
        })
        .collect()
}

/// EFFECTS BEHIND A LEVER, for an A/B on the headset: a scene object (by
/// id) whose effect is drawn only while the renderer lever says so. Off by
/// default, so the shipped level loads without them.
pub(crate) const LEVER_GATED: &[(&str, fn(&Levers) -> bool)] = &[
    // The dust round the cave's bonfire, to judge with the torch through it.
    ("fx_cave_dust", |l| l.cave_dust),
];

/// Which of [`LEVER_GATED`]'s levers are on.
pub(crate) fn gate(levers: &Levers) -> Vec<bool> {
    LEVER_GATED.iter().map(|(_, on)| on(levers)).collect()
}

/// The level's emitters less those whose lever is off.
pub(crate) fn gated(emitters: &[EffectEmitter], levers: &Levers) -> Vec<EffectEmitter> {
    emitters.iter().filter(|e| LEVER_GATED.iter().all(|(id, on)| e.id != *id || on(levers))).cloned().collect()
}

/// Whether `game/effects/<name>.json` exists: an asset the library can hold.
fn names_an_asset(game_dir: &std::path::Path, name: &str) -> bool {
    !name.is_empty() && !name.contains(['/', '\\', '.']) && game_dir.join("effects").join(format!("{name}.json")).is_file()
}

/// THE LEVEL'S PLACED EFFECTS: every object whose effect names an asset of
/// `game/effects` (`effect.asset`), at the object along its up, in the WORLD's
/// frame. Its `scale`, `rate` and `tint` multiply the asset's sprites, streams
/// and colour (the tint decoded from sRGB). One that names no such file is
/// left to `load`, which draws its kind.
pub(crate) fn placed(game_dir: &std::path::Path, scene_name: &str) -> Vec<PlacedEffect> {
    let Ok(mut scene) = Scene::load(&Manifest::scene_path(game_dir, scene_name)) else {
        return Vec::new();
    };
    scene.resolve_world_transforms();
    scene
        .objects
        .iter()
        .filter_map(|o| {
            let e = o.effect.as_ref()?;
            let named = e.asset.as_deref().filter(|a| !a.is_empty())?;
            let Some(asset) = Some(named).filter(|a| names_an_asset(game_dir, a)) else {
                // Said out loud: a missing file draws the built-in kind, which
                // looks like a working effect (the pond's fog drew as smoke on
                // the headset for a day because run.sh never pushed effects/).
                log::warn!(
                    "EFFECTS: '{}' names asset '{named}' but {} is missing -- drawing its built-in kind instead",
                    o.id,
                    game_dir.join("effects").join(format!("{named}.json")).display(),
                );
                return None;
            };
            let c = &o.cuboid;
            Some(PlacedEffect {
                object: o.id.clone(),
                asset: asset.to_string(),
                position: c.position,
                direction: c.rotation * Vec3::Y,
                source: object_key(&o.id),
                placed: Placed {
                    scale: e.scale.max(0.0),
                    rate: e.rate.max(0.0),
                    tint: [srgb_to_linear(e.tint.0), srgb_to_linear(e.tint.1), srgb_to_linear(e.tint.2)],
                    // (`find_sunlit`, once the physics scene is up.)
                    sunlit: 0.0,
                    // Its box across the world's x and z: what a disc that
                    // fills its object spans (a pond's mist, by the pond's box).
                    footprint: {
                        let ext = [c.rotation * Vec3::X * c.half_size.x, c.rotation * Vec3::Y * c.half_size.y, c.rotation * Vec3::Z * c.half_size.z];
                        let across = |k: usize| ext.iter().map(|v| v[k].abs()).sum::<f32>();
                        [across(0), across(2)]
                    },
                    // Its `intensity` (x[0, 4]): a fog slab's density.
                    strength: e.intensity.clamp(0.0, 4.0),
                },
            })
        })
        .collect()
}

/// THE SCENE'S FILL BUDGET at load, by the check the editor runs
/// (`authoring::scene_json`): the kinds' templates and the placed assets
/// within 8 m of one another, and the level's live particles. Empty when
/// inside it; otherwise what is over, for the log. The editor refuses to
/// save such a scene, so this catches only one written by hand.
pub(crate) fn budget_problems(game_dir: &std::path::Path, emitters: &[EffectEmitter], placed: &[PlacedEffect]) -> Vec<String> {
    use serde_json::json;
    let mut items: Vec<serde_json::Value> = emitters
        .iter()
        .map(|e| {
            let volume = 8.0 * e.extent[0].length() * e.extent[1].length() * e.extent[2].length();
            json!({ "object": e.id, "kind": format!("{:?}", e.kind).to_ascii_lowercase(), "position": e.position.to_array(), "scale": e.scale, "rate": e.rate, "volume": volume })
        })
        .collect();
    let mut assets = std::collections::HashMap::new();
    for p in placed {
        if !assets.contains_key(&p.asset) {
            if let Ok(text) = std::fs::read_to_string(game_dir.join("effects").join(format!("{}.json", p.asset))) {
                assets.insert(p.asset.clone(), text);
            }
        }
        items.push(json!({ "object": p.object, "asset": p.asset, "position": p.position.to_array(), "scale": p.placed.scale, "rate": p.placed.rate, "footprint": p.placed.footprint }));
    }
    if items.is_empty() {
        return Vec::new();
    }
    let verdict = space_soup::renderer::effects::authoring::scene_json(&serde_json::Value::Array(items), &assets);
    ["over", "issues"].iter().flat_map(|k| verdict[*k].as_array().cloned().unwrap_or_default()).filter_map(|v| v.as_str().map(str::to_string)).collect()
}

/// HOW MUCH SUN REACHES each placed effect (`Placed::sunlit`), found once at
/// load: a ray toward the sun (`toward`, world) from where its particles are
/// -- the object and 0.5 m and 1.5 m over it -- through the physics scene,
/// as `find_ceilings` casts up (`blocked(from, toward)`: true when it meets
/// something). The share that reach open sky; 0 with no sun. A cave or a room
/// gives 0, so their particles stay lit by their lamps alone.
pub(crate) fn find_sunlit(placed: &mut [PlacedEffect], toward: Option<Vec3>, blocked: impl Fn(Vec3, Vec3) -> bool) {
    for p in placed.iter_mut() {
        p.placed.sunlit = match toward.and_then(|d| d.try_normalize()) {
            Some(d) => [0.05, 0.5, 1.5].iter().filter(|&&h| !blocked(p.position + Vec3::Y * h, d)).count() as f32 / 3.0,
            None => 0.0,
        };
    }
}

/// A placed effect's source: its object's id, hashed (FNV-1a), so its seed
/// and the governor's key stay the object's across loads.
fn object_key(id: &str) -> u64 {
    id.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
}

/// The ceiling over each fire and smoke emitter, by a ray straight up from it
/// (`cast` gives the height it meets, if any): a cave's roof or a hall's
/// ceiling, which smoke spreads out under. Dust needs none.
pub(crate) fn find_ceilings(emitters: &mut [EffectEmitter], cast: impl Fn(Vec3) -> Option<f32>) {
    for e in emitters.iter_mut() {
        if matches!(e.kind, EffectKind::Smoke | EffectKind::Embers | EffectKind::Fire) {
            e.ceiling = cast(e.position + Vec3::Y * 0.05);
        }
    }
}

fn srgb_to_linear(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scene_without_effects_has_none_and_a_missing_one_is_not_fatal() {
        let game = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../game");
        assert!(load(&game, "no_such_scene").is_empty());
    }

    #[test]
    fn a_scenes_effects_carry_their_variation() {
        // test_room's fires name none: each at the preset's.
        let game = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../game");
        let fx = load(&game, "test_room");
        assert!(fx.iter().any(|e| e.kind == EffectKind::Fire));
        assert!(fx.iter().all(|e| e.variation == Variation::default()), "{:?}", fx.iter().map(|e| e.variation).collect::<Vec<_>>());
    }

    /// A scratch game: `game/effects/steam_vent.json` and a scene of `objects`.
    fn scratch_game(tag: &str, objects: serde_json::Value) -> std::path::PathBuf {
        let real = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../game");
        let dir = std::env::temp_dir().join(format!("scene_effects_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("effects")).unwrap();
        std::fs::create_dir_all(dir.join("scenes")).unwrap();
        std::fs::copy(real.join("effects/steam_vent.json"), dir.join("effects/steam_vent.json")).unwrap();
        let scene = serde_json::json!({ "name": "s", "sky": null, "objects": objects });
        std::fs::write(dir.join("scenes/s.json"), serde_json::to_string(&scene).unwrap()).unwrap();
        dir
    }

    fn marker(id: &str, x: f32, effect: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "id": id, "uuid": format!("0f000000-0000-4000-8000-{:012x}", object_key(id) & 0xffff_ffff_ffff),
            "cuboid": { "color": [200, 200, 200, 255], "half_size": [0.1, 0.1, 0.1], "position": [x, 0.1, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0], "style": "Solid", "wire_color": [0, 0, 0, 0] },
            "hidden": true, "effect": effect,
        })
    }

    #[test]
    fn an_effect_naming_an_asset_is_placed_and_its_kind_is_the_fallback() {
        let dir = scratch_game(
            "place",
            serde_json::json!([
                marker("vent", 1.0, serde_json::json!({ "kind": "smoke", "asset": "steam_vent", "scale": 1.5, "rate": 0.5, "tint": [128, 255, 255, 255] })),
                marker("lost", 3.0, serde_json::json!({ "kind": "smoke", "asset": "no_such_effect" })),
                marker("plain", 5.0, serde_json::json!({ "kind": "embers" })),
            ]),
        );
        let placed = placed(&dir, "s");
        assert_eq!(placed.len(), 1, "{placed:?}");
        let p = &placed[0];
        assert_eq!((p.object.as_str(), p.asset.as_str()), ("vent", "steam_vent"));
        assert_eq!(p.position, Vec3::new(1.0, 0.1, 0.0));
        assert_eq!(p.direction, Vec3::Y);
        assert_eq!((p.placed.scale, p.placed.rate), (1.5, 0.5));
        // The tint decoded from sRGB, as the kinds' are.
        assert!((p.placed.tint[0] - 0.2158).abs() < 1e-3 && p.placed.tint[1] == 1.0, "{:?}", p.placed.tint);
        assert_eq!(p.source, object_key("vent"));
        // The placed one is not drawn twice; one naming no file draws its kind.
        let kinds = load(&dir, "s");
        assert_eq!(kinds.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(), ["lost", "plain"]);
        assert!(budget_problems(&dir, &kinds, &placed).is_empty(), "{:?}", budget_problems(&dir, &kinds, &placed));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_scene_crowded_with_effects_is_reported_at_load() {
        // Forty vents, each at forty times its rate, within a metre: far over
        // the neighbourhood's fill and the level's particles.
        let objects: Vec<_> = (0..40).map(|k| marker(&format!("v{k}"), k as f32 * 0.02, serde_json::json!({ "kind": "smoke", "asset": "steam_vent", "rate": 40.0 }))).collect();
        let dir = scratch_game("crowd", serde_json::Value::Array(objects));
        let placed = placed(&dir, "s");
        assert_eq!(placed.len(), 40);
        let problems = budget_problems(&dir, &load(&dir, "s"), &placed);
        assert!(problems.iter().any(|p| p.contains("megapixel-layers")), "{problems:?}");
        assert!(problems.iter().any(|p| p.contains("particles alive")), "{problems:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// WHERE TO STAND AN EFFECT: for each `POINTS=x,y,z;..` (y: a height
    /// inside the room), the floor and the ceiling a ray down and up meets
    /// through the physics scene, as `find_ceilings` casts. `SCENE` (test_room).
    #[test]
    #[ignore]
    fn print_floor_and_ceiling() {
        let game = crate::offline_frame::offline_game_dir();
        let name = std::env::var("SCENE").unwrap_or_else(|_| "test_room".into());
        let scene = Scene::load(&Manifest::scene_path(&game, &name)).expect("the scene");
        let mut physics = space_soup_engine::rigid_physics::PhysicsWorld::new();
        physics.rebuild(&scene, &game);
        for p in std::env::var("POINTS").unwrap_or_default().split(';').filter(|s| !s.is_empty()) {
            let v: Vec<f32> = p.split(',').map(|x| x.trim().parse().unwrap()).collect();
            let at = Vec3::new(v[0], v[1], v[2]);
            let down = physics.raycast(at, Vec3::NEG_Y, 40.0).map(|(hit, _)| hit.y);
            let up = physics.raycast(at, Vec3::Y, 40.0).map(|(hit, _)| hit.y);
            eprintln!("POINT {p}: floor {down:?} ceiling {up:?}");
        }
    }

    /// The cave's dust is behind the lever `cave_dust`, off as shipped.
    #[test]
    fn the_cave_dust_waits_for_its_lever() {
        let dust = |id: &str| EffectEmitter { id: id.into(), ..any_emitter() };
        let all = [dust("fx_cave_dust"), dust("hall_dust")];
        let off = Levers::default();
        assert!(!off.cave_dust);
        assert_eq!(gated(&all, &off).iter().map(|e| e.id.as_str()).collect::<Vec<_>>(), ["hall_dust"]);
        let on = Levers { cave_dust: true, ..Levers::default() };
        assert_eq!(gated(&all, &on).len(), 2);
        assert_ne!(gate(&off), gate(&on));
    }

    fn any_emitter() -> EffectEmitter {
        EffectEmitter {
            id: String::new(),
            kind: EffectKind::Dust,
            position: Vec3::ZERO,
            direction: Vec3::Y,
            extent: [Vec3::X, Vec3::Y, Vec3::Z],
            scale: 1.0,
            rate: 1.0,
            tint: [1.0; 3],
            ceiling: None,
            variation: Variation::default(),
        }
    }

    #[test]
    fn white_tints_nothing() {
        assert_eq!(srgb_to_linear(255), 1.0);
        assert!((srgb_to_linear(128) - 0.2158).abs() < 1e-3);
    }
}
