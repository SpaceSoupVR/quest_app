//! Realtime scene lights, loaded from disk by the client.
//!
//! WHY THIS EXISTS
//!
//! A level's lights are static scene data sitting in the game directory that
//! run.sh already pushed -- exactly like brushes and terrain, and for exactly
//! the reason terrain_render gives: data the headset already has should not be
//! streamed to it. Sending it per snapshot costs the wire budget that matters
//! when the target is 64 players.
//!
//! But the reason it is not merely an optimisation is the standalone path. The
//! client took every light from the server snapshot, so a game that does not
//! run a multiplayer server got NO realtime lights at all -- not dim ones, none
//! -- and every surface fell back to sky ambient. That reads as "the lighting
//! is broken": whole walls evenly lit, no cone from any spot, and no shadows,
//! because a scene with no lights has nothing to cast them.
//!
//! Baked lighting already had its standalone path (lightmaps load from disk).
//! This is the other half.
//!
//! WHY ONLY WHEN THE SERVER IS SILENT
//!
//! A connected server runs `collect_render_lights` over the same scene file and
//! sends the same lights. Using both would light every level twice.
use glam::Vec3;
use space_soup_engine::{scene::Scene, LightKind, LightMode, Manifest};
use space_soup_protocol::{WireColor3, WireLightKind, WireRenderLight};

/// Every realtime light in a scene, in the shape a server snapshot would carry.
///
/// The wire shape rather than the renderer's, so this feeds the identical
/// conversion the networked path uses -- one place decides how a light becomes
/// a `Light`, and the standalone path cannot drift away from the multiplayer
/// one without the drift being visible in both.
pub(crate) fn load(game_dir: &std::path::Path, scene_name: &str) -> Vec<WireRenderLight> {
    // LIVE: realtime and stationary lamps both shade their direct light every
    // frame; a stationary one only takes its shadows from the bake.
    load_matching(game_dir, scene_name, "live", LightMode::is_live)
}

/// Every BAKED light in a scene, in the same shape. Their light is in the
/// lightmaps; the renderer shades them only on what has no lightmap -- the
/// characters and the ground. See `XrRenderer::set_baked_lights`.
pub(crate) fn load_baked(game_dir: &std::path::Path, scene_name: &str) -> Vec<WireRenderLight> {
    load_matching(game_dir, scene_name, "baked", |m| m == LightMode::Baked)
}

/// WHICH MASK CHANNEL each stationary lamp's baked shadows are in, by render
/// light id (`object#index`). Computed from the scene exactly as the baker
/// computed it -- see `space_soup_engine::stationary` -- so nothing has to
/// carry it. A lamp that found no channel is absent, and shades unshadowed.
pub(crate) fn stationary_channels(game_dir: &std::path::Path, scene_name: &str) -> std::collections::HashMap<String, u8> {
    let path = Manifest::scene_path(game_dir, scene_name);
    let Ok(mut scene) = Scene::load(&path) else {
        return Default::default();
    };
    scene.resolve_world_transforms();
    let assigned = space_soup_engine::stationary::scene_stationary_channels(&scene.objects);
    for (lamp, channel) in &assigned {
        match channel {
            Some(c) => log::info!("stationary: '{}' shadows from mask channel {c}", lamp.id()),
            None => log::warn!("stationary: '{}' has no mask channel; it shades unshadowed", lamp.id()),
        }
    }
    assigned.into_iter().filter_map(|(lamp, c)| Some((lamp.id(), c?))).collect()
}

/// The stationary masks a bake wrote, if they are the ones this scene's lamps
/// read: as many layers as their channels need. A bake written for other
/// lamps -- or in the older format, one byte and four lamps a layer -- would
/// hand each lamp another lamp's shadow, so it is dropped with a warning and
/// the stationary lamps shade unshadowed until the level is re-baked.
pub(crate) fn usable_stationary_masks<T>(baked: Vec<T>, channels: &std::collections::HashMap<String, u8>) -> Vec<T> {
    let needed = space_soup_engine::stationary::mask_layers(&channels.values().map(|&c| Some(c)).collect::<Vec<_>>());
    if baked.len() == needed {
        return baked;
    }
    log::warn!(
        "lightmaps: the bake has {} stationary mask layer(s) and this scene's lamps need {needed}: it was baked \
         for other lamps, or before masks took two bytes a lamp -- stationary lamps shade unshadowed until the \
         level is re-baked",
        baked.len()
    );
    Vec::new()
}

fn load_matching(
    game_dir: &std::path::Path,
    scene_name: &str,
    what: &str,
    keep: impl Fn(LightMode) -> bool,
) -> Vec<WireRenderLight> {
    let path = Manifest::scene_path(game_dir, scene_name);
    let scene = match Scene::load(&path) {
        Ok(s) => s,
        Err(e) => {
            log::warn!("scene lights: {} did not load: {e:#}", path.display());
            return Vec::new();
        }
    };

    let mut out = Vec::new();
    for o in &scene.objects {
        // Index over the object's FULL light list, so a light's id does not
        // change when a sibling is switched to baked -- the same rule
        // `collect_render_lights` follows.
        for (i, l) in o.lights.iter().enumerate() {
            if !keep(l.mode) {
                continue;
            }
            // A lamp that is switched off contributes nothing, and is skipped
            // rather than sent at zero intensity: the realtime light budget is
            // small, and an off lamp taking one of its slots is a lit lamp
            // somewhere else going dark.
            if !l.enabled {
                continue;
            }
            let pose = space_soup_engine::scene_light::resolve_light_pose(
                l,
                o.cuboid.position,
                o.cuboid.rotation,
                l.socket.as_deref().and_then(|n| o.socket(n)),
            );
            // Exhaustive on purpose, unlike the tests: a new field must fail to
            // compile here rather than silently not cross the wire.
            out.push(WireRenderLight {
                id: format!("{}#{i}", o.id),
                position: pose.position.to_array(),
                direction: pose.direction().to_array(),
                kind: match l.kind {
                    LightKind::Point => WireLightKind::Point,
                    LightKind::Spot => WireLightKind::Spot,
                    LightKind::Directional => WireLightKind::Directional,
                },
                color: WireColor3(l.color.0, l.color.1, l.color.2, l.color.3),
                intensity: l.intensity,
                range: l.range,
                cone_angle_deg: l.cone_angle_deg,
                inner_cone_angle_deg: l.inner_cone_angle_deg,
            });
        }
    }
    log::info!(
        "scene lights: {} {what} light(s) loaded from disk for '{scene_name}'",
        out.len()
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn game_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../game")
    }

    /// Every lamp in the scene, whichever mode it is authored in.
    fn all_modes(scene: &str) -> Vec<WireRenderLight> {
        let mut all = load(&game_dir(), scene);
        all.extend(load_baked(&game_dir(), scene));
        all
    }

    /// test_room's lamps are STATIONARY (since 2026-09-27): shaded live, their
    /// shadows from the baked masks, so they come through `load` with the
    /// live lights. The two loaders must still split the scene cleanly -- a
    /// lamp through both would be lit twice.
    #[test]
    fn the_two_loaders_split_the_scene_without_overlap() {
        let live = load(&game_dir(), "test_room");
        let baked = load_baked(&game_dir(), "test_room");
        if live.is_empty() && baked.is_empty() {
            eprintln!("skipping: test_room not present");
            return;
        }
        assert!(live.len() >= 7, "test_room's stationary lamps must load as live: {}", live.len());
        for b in &baked {
            assert!(live.iter().all(|l| l.id != b.id), "{} loaded as both live and baked", b.id);
        }
    }

    /// Every stationary lamp in test_room found a shadow channel, so none of
    /// them shades unshadowed through the walls.
    #[test]
    fn every_test_room_stationary_lamp_has_a_mask_channel() {
        let channels = stationary_channels(&game_dir(), "test_room");
        if channels.is_empty() {
            eprintln!("skipping: test_room not present");
            return;
        }
        assert_eq!(channels.len(), 7, "{channels:?}");
        assert!(channels.values().all(|&c| (c as usize) < space_soup_engine::stationary::MAX_STATIONARY_CHANNELS));
    }

    /// Masks are read only when the bake wrote as many layers as the lamps'
    /// channels need. Three channels take two layers at two lamps a layer; a
    /// one-layer bake is the older four-lamp format (or other lamps) and would
    /// hand channel 1 the penumbra of channel 0.
    #[test]
    fn a_bake_for_other_lamps_or_the_old_format_is_not_read() {
        let channels: std::collections::HashMap<String, u8> =
            [("a#0", 0u8), ("b#0", 1), ("c#0", 2)].into_iter().map(|(k, c)| (k.to_string(), c)).collect();
        assert_eq!(usable_stationary_masks(vec!["layer0", "layer1"], &channels), vec!["layer0", "layer1"]);
        assert!(usable_stationary_masks(vec!["layer0"], &channels).is_empty(), "a one-layer bake was read");
        assert!(usable_stationary_masks(vec!["l0", "l1", "l2"], &channels).is_empty(), "a bake for more lamps was read");
        assert!(usable_stationary_masks(Vec::<&str>::new(), &Default::default()).is_empty());
    }

    #[test]
    fn a_scene_with_lights_yields_them_without_a_server() {
        // The standalone path. Before this, a game with no multiplayer server
        // rendered every level with zero realtime lights and nothing said so.
        let lights = all_modes("test_room");
        if lights.is_empty() {
            eprintln!("skipping: test_room not present");
            return;
        }
        assert!(lights.iter().any(|l| l.kind == WireLightKind::Spot));
        assert!(lights.iter().all(|l| l.intensity > 0.0));
    }

    #[test]
    fn a_spot_points_somewhere() {
        // A zero direction would make every spot's cone test meaningless.
        let lights = all_modes("test_room");
        for l in lights.iter().filter(|l| l.kind == WireLightKind::Spot) {
            let d = Vec3::from(l.direction);
            assert!((d.length() - 1.0).abs() < 1e-3, "{}: {d:?}", l.id);
        }
    }

    #[test]
    fn the_inner_cone_survives_the_trip() {
        // It reaches the GPU through this struct; dropping it here would put
        // every standalone spot back to a soft blob with no rim.
        let lights = all_modes("test_room");
        assert!(
            lights.iter().any(|l| l.inner_cone_angle_deg > 0.0),
            "test_room authors an inner cone; it must arrive",
        );
    }

    #[test]
    fn a_missing_scene_is_not_fatal() {
        assert!(load(&game_dir(), "no_such_scene_at_all").is_empty());
    }
}
