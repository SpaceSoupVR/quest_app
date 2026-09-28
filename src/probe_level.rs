//! A level's reflection probes as the renderer streams them: described from
//! the bake's index, pixels read from disk on demand, doorways named by room.
//!
//! Shared by the app and `offline_frame`, so the harness exercises exactly the
//! room numbering and doorway matching the headset does.

use std::path::Path;
use std::sync::Arc;

use glam::Vec3;
use space_soup::renderer::probe_stream::{ProbeDepthSource, ProbeDesc, ProbeSource};
use space_soup::renderer::uniforms::{ProbePortal, ProbeProxy};
use space_soup_engine::reflection_probe::{self, ProbeEntry};
use space_soup_engine::scene::GameObject;

pub struct ProbeLevel {
    pub resolution: u32,
    pub descs: Vec<ProbeDesc>,
    pub portals: Vec<ProbePortal>,
    /// The rooms (by the numbers `descs` and `portals` use) that are CLOSED:
    /// walled all round but their baked doorways, so nothing outside the
    /// building can be seen from inside one except through a doorway. See
    /// `space_soup_engine::room_graph::closed_room_boxes` and the renderer's
    /// `portal_cull`.
    pub closed_rooms: Vec<u32>,
    /// The index entries, in `descs` order, for the source.
    pub entries: Arc<Vec<ProbeEntry>>,
}

/// The probe rooms that are closed: those whose box is one of the level's
/// closed room carves (`room_graph::closed_room_boxes`), matched to within
/// 5 cm. A room none of whose photographs has depth is the outdoor volume,
/// never closed; a room box that matches no carve -- authored by hand, or
/// from a stale bake -- is left open, which only costs drawing more.
fn closed_probe_rooms(
    objects: &[GameObject],
    descs: &[ProbeDesc],
    entries: &[ProbeEntry],
    portals: &[ProbePortal],
) -> Vec<u32> {
    let brushes: Vec<(&str, &space_soup_engine::brush::BrushDef)> =
        objects.iter().filter_map(|o| o.brush.as_ref().map(|b| (o.id.as_str(), b))).collect();
    let doorways: Vec<(Vec3, Vec3)> = portals.iter().map(|p| (p.min, p.max)).collect();
    let closed_boxes = space_soup_engine::room_graph::closed_room_boxes(&brushes, &doorways);
    let mut closed: Vec<u32> = descs
        .iter()
        .zip(entries)
        .filter(|(d, e)| {
            e.depth.is_some()
                && closed_boxes.iter().any(|(lo, hi)| {
                    lo.abs_diff_eq(d.min, 0.05) && hi.abs_diff_eq(d.max, 0.05)
                })
        })
        .map(|(d, _)| d.volume)
        .collect();
    closed.sort_unstable();
    closed.dedup();
    closed
}

impl ProbeLevel {
    /// The level's probes, or `None` when it has none.
    pub fn load(game_dir: &Path, scene: &str) -> Option<Self> {
        let index = reflection_probe::load_scene_probe_index(game_dir, scene);
        if index.is_empty() {
            return None;
        }
        // ONE FACE SIZE, because they share one cube array. The index says
        // what each was baked at; an older index that does not is read once.
        let resolution = index
            .iter()
            .map(|e| e.resolution)
            .find(|&r| r > 0)
            .or_else(|| index.iter().find_map(|e| reflection_probe::decode_probe(e).map(|(r, _)| r)))?;
        let (entries, dropped): (Vec<ProbeEntry>, Vec<ProbeEntry>) =
            index.into_iter().partition(|e| e.resolution == resolution || e.resolution == 0);
        if !dropped.is_empty() {
            log::warn!(
                "reflection probes: {} dropped for a face size other than {resolution}px; re-bake the level's probes",
                dropped.len(),
            );
        }

        // ROOMS NUMBERED IN ORDER OF FIRST APPEARANCE. The number only has to
        // agree between the probes and the doorways, both of which come from
        // here.
        let mut rooms: Vec<String> = Vec::new();
        let mut room_of = |name: &str| -> u32 {
            match rooms.iter().position(|r| r == name) {
                Some(i) => i as u32,
                None => {
                    rooms.push(name.to_string());
                    (rooms.len() - 1) as u32
                }
            }
        };
        let descs: Vec<ProbeDesc> = entries
            .iter()
            .map(|e| ProbeDesc {
                centre: Vec3::from(e.centre),
                min: Vec3::from(e.min),
                max: Vec3::from(e.max),
                volume: room_of(&e.volume),
            })
            .collect();
        // A doorway naming a room with no probe (a stale bake) is left out:
        // it would blend toward nothing.
        let portals: Vec<ProbePortal> = reflection_probe::load_scene_portals(game_dir, scene)
            .into_iter()
            .filter_map(|p| {
                let low = rooms.iter().position(|r| *r == p.low)? as u32;
                let high = rooms.iter().position(|r| *r == p.high)? as u32;
                Some(ProbePortal { min: Vec3::from(p.min), max: Vec3::from(p.max), axis: p.axis, low, high, wall: None })
            })
            .collect();
        // How deep each doorway's jambs are, from the walls it is cut through.
        // See `portal_wall_extent`: the carve alone overstates it.
        let mut portals = portals;
        let mut closed_rooms = Vec::new();
        if let Ok(mut scene) = space_soup_engine::scene::Scene::load(&space_soup_engine::Manifest::scene_path(game_dir, scene)) {
            scene.resolve_world_transforms();
            for p in &mut portals {
                p.wall = space_soup_engine::reflection_proxy::portal_wall_extent(&scene.objects, p.min, p.max, p.axis as usize);
            }
            closed_rooms = closed_probe_rooms(&scene.objects, &descs, &entries, &portals);
        }
        Some(Self { resolution, descs, portals, closed_rooms, entries: Arc::new(entries) })
    }

    /// WHAT STANDS INSIDE THE ROOMS -- a pillar, a lamp -- for the reflection
    /// trace, named by the same room numbers as the probes and doorways.
    /// `objects` must already be in world space (`resolve_world_transforms`).
    /// See `space_soup_engine::reflection_proxy`.
    pub fn proxies(&self, game_dir: &Path, objects: &[GameObject]) -> Vec<ProbeProxy> {
        // One box per room, indexed by room number: every cell of a room
        // carries the room's box.
        //
        // A room none of whose photographs has depth -- the outdoor volume,
        // whose box is a stand-in for the sky dome and contains the whole
        // building -- gets an EMPTY box: the trace never walks it, so nothing
        // stands in it.
        let count = self.descs.iter().map(|d| d.volume as usize + 1).max().unwrap_or(0);
        let empty = (Vec3::splat(1.0), Vec3::splat(-1.0));
        let mut rooms = vec![empty; count];
        for (d, e) in self.descs.iter().zip(self.entries.iter()) {
            if e.depth.is_some() {
                rooms[d.volume as usize] = (d.min, d.max);
            }
        }
        space_soup_engine::reflection_proxy::reflection_proxies(game_dir, objects, &rooms)
            .into_iter()
            .map(|p| ProbeProxy {
                centre: p.centre,
                half_size: p.half_size,
                rotation: p.rotation,
                volume: p.room as u32,
                solid: p.solid,
            })
            .collect()
    }

    /// [`ProbeLevel::proxies`] for the scene file itself, its transforms
    /// resolved to world space. Empty when the scene does not load.
    pub fn scene_proxies(&self, game_dir: &Path, scene: &str) -> Vec<ProbeProxy> {
        let path = space_soup_engine::Manifest::scene_path(game_dir, scene);
        match space_soup_engine::scene::Scene::load(&path) {
            Ok(mut s) => {
                s.resolve_world_transforms();
                self.proxies(game_dir, &s.objects)
            }
            Err(e) => {
                log::warn!("reflection proxies: {} did not load: {e:#}", path.display());
                Vec::new()
            }
        }
    }

    /// Distances for probe `i`, from its depth file, when the bake wrote one.
    /// See `space_soup::renderer::probe_stream::ProbeDepthSource`.
    pub fn depth_source(&self) -> ProbeDepthSource {
        let entries = self.entries.clone();
        let res = self.resolution;
        Arc::new(move |i| reflection_probe::decode_probe_depth(entries.get(i)?, res))
    }

    /// Pixels for probe `i`, read from its file each time. Nothing keeps them.
    pub fn source(&self) -> ProbeSource {
        let entries = self.entries.clone();
        let res = self.resolution;
        Arc::new(move |i| {
            let (r, faces) = reflection_probe::decode_probe(entries.get(i)?)?;
            (r == res).then_some(faces)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// test_room's rooms hold the pillar, three hanging lamps and two wall
    /// sconces -- and nothing of the rooms' own shells. Each lamp's box is the
    /// MODEL's, hanging below its ceiling mount, not the 20 cm handle the
    /// editor gives the object.
    #[test]
    fn test_room_proxies_are_the_pillar_and_the_fixtures() {
        let game = Path::new(env!("CARGO_MANIFEST_DIR")).join("../game");
        let Some(level) = ProbeLevel::load(&game, "test_room") else {
            eprintln!("skipping: no test_room probes");
            return;
        };
        let proxies = level.scene_proxies(&game, "test_room");
        for p in &proxies {
            eprintln!("proxy room {} centre {:?} half {:?}", p.volume, p.centre, p.half_size);
            assert!(p.half_size.max_element() < 2.0, "a room's shell became a proxy: {p:?}");
        }
        assert_eq!(proxies.len(), 6, "pillar + 3 hanging lamps + 2 sconces");
        let pillar = proxies.iter().find(|p| (p.centre - Vec3::new(0.0, 1.55, -7.0)).length() < 1e-3).expect("the pillar");
        assert!((pillar.half_size - Vec3::new(0.45, 1.55, 0.45)).length() < 1e-3);
        // Hanging lamps: mounted at y = 3.1, the shade below it.
        for x in [0.0, 14.0] {
            let lamp = proxies.iter().find(|p| (p.centre.x - x).abs() < 0.5 && p.centre.y > 1.5).expect("a hanging lamp");
            assert!(lamp.centre.y + lamp.half_size.y <= 3.2, "the lamp reaches through the ceiling: {lamp:?}");
            assert!(lamp.half_size.y > 0.15, "the lamp's box is the editor handle, not the model: {lamp:?}");
        }
    }
}

#[cfg(test)]
mod closed_room_tests {
    use super::*;

    /// test_room's hall, hallway and brick hall are shells walled all round
    /// but their doorways; the outdoor volume is not a room at all.
    #[test]
    fn test_rooms_indoor_rooms_are_closed_and_the_outdoors_is_not() {
        let game = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../game");
        let Some(level) = ProbeLevel::load(&game, "test_room") else {
            eprintln!("skipping: test_room has no probes here");
            return;
        };
        let named = |room: u32| {
            level
                .descs
                .iter()
                .zip(level.entries.iter())
                .find(|(d, _)| d.volume == room)
                .map(|(_, e)| e.volume.clone())
                .unwrap()
        };
        let closed: Vec<String> = level.closed_rooms.iter().map(|&r| named(r)).collect();
        for room in ["hall_probe", "hallway#0", "brick_probe"] {
            assert!(closed.iter().any(|c| c == room), "{room} should be closed: {closed:?}");
        }
        assert!(!closed.iter().any(|c| c == "outdoors"), "{closed:?}");
    }
}
