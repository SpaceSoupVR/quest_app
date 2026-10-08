//! The level's special effects -- fire, coals, smoke, embers, dust -- loaded
//! from the scene file by the client, as its lights are (`scene_lights`).
//!
//! An effect is level dressing: the headset already has the scene file that
//! run.sh pushed, so nothing is streamed, and a game with no server draws them
//! all the same. Each particle is a function of its emitter and the clock
//! (`space_soup::renderer::effects`), so the emitter's few numbers are the
//! whole of it.
use glam::Vec3;
use space_soup::renderer::effects::{EffectEmitter, EffectKind, Variation};
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

    #[test]
    fn white_tints_nothing() {
        assert_eq!(srgb_to_linear(255), 1.0);
        assert!((srgb_to_linear(128) - 0.2158).abs() < 1e-3);
    }
}
