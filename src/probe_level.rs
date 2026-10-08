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
use space_soup::renderer::proxy_cards::ProxyCards;
use space_soup::renderer::proxy_field::ProxyField;
use space_soup_engine::reflection_probe::{self, ProbeEntry};
use space_soup_engine::scene::GameObject;

/// WHAT STANDS IN A LEVEL'S ROOMS for the reflection trace, as
/// [`ProbeLevel::proxies`] builds it: the proxies, the models' distance
/// fields (`ProbeProxy::field` indexes `fields`) and the models' cards
/// (`ProbeProxy::cards` indexes `cards`).
#[derive(Default)]
pub struct ReflectionProxies {
    pub proxies: Vec<ProbeProxy>,
    pub fields: Vec<ProxyField>,
    pub cards: Vec<ProxyCards>,
    /// Which sides each lamp's bright source shows from, measured from its
    /// cards, by object id with the frame they were taken in. See
    /// `glare_fixtures`.
    pub glare: std::collections::HashMap<String, crate::glare_fixtures::MeasuredGlare>,
}

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
                has_depth: e.depth.is_some(),
                room_light: e.irradiance,
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
    /// trace, named by the same room numbers as the probes and doorways; each
    /// model's distance field, one per model and scale however many objects
    /// use it; and each model's cards, from `scene`'s probe bake. `objects`
    /// must already be in world space (`resolve_world_transforms`). See
    /// `space_soup_engine::reflection_proxy`, `space_soup::renderer::proxy_field`
    /// and `space_soup::renderer::proxy_cards`.
    pub fn proxies(&self, game_dir: &Path, scene: &str, objects: &[GameObject]) -> ReflectionProxies {
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
        let mut standing = space_soup_engine::reflection_proxy::reflection_proxies(game_dir, objects, &rooms);
        // The doors' leaves, after everything else, so the shaped models'
        // numbers the baker gives the rest are unchanged. Turned with their
        // leaves each frame: see `space_soup::renderer::doors::posed_proxies`.
        standing.extend(space_soup_engine::reflection_proxy::door_proxies(game_dir, objects, &rooms));
        // WHICH MODELS ARE TRACED BY THEIR OWN SHAPE: the baker's rule, so
        // the models it left out of the photographs are exactly the ones
        // given a field here. See `shaped_models`.
        let shaped = space_soup_engine::reflection_proxy::shaped_models(&standing, objects);
        let loaded_cards = space_soup_engine::reflection_cards::load_scene_cards(game_dir, scene);
        let mut out = ReflectionProxies::default();
        let mut pictured = Vec::new();
        let mut field_of: Vec<Option<Option<u32>>> = vec![None; space_soup_engine::reflection_proxy::MAX_SHAPED_MODELS];
        for (p, shape) in standing.iter().zip(&shaped) {
            let object = objects.get(p.object);
            let mesh = if p.solid { None } else { object.and_then(|o| o.mesh.as_ref()) };
            let field = match (mesh, shape) {
                (Some(mesh), Some(k)) => *field_of[*k as usize].get_or_insert_with(|| {
                    let started = std::time::Instant::now();
                    let f = space_soup_engine::reflection_proxy::model_field(
                        &game_dir.join(&mesh.path),
                        mesh.scale,
                        space_soup_engine::reflection_proxy::FIELD_SAMPLES,
                    )?;
                    log::info!(
                        "reflection proxies: '{}' field {:?} ({:.0} mm reach) in {} ms",
                        mesh.path,
                        f.dims,
                        f.max_distance * 1000.0,
                        started.elapsed().as_millis(),
                    );
                    // Its mean colour, for what its cards do not show. See
                    // `probe_model_colour` in the renderer's lights block.
                    let albedo = space_soup_engine::mesh_lightmap::model_albedo(&game_dir.join(&mesh.path))
                        .map_or([0.2; 3], |a| a.to_array());
                    out.fields.push(ProxyField { dims: f.dims, max_distance: f.max_distance, distances: f.distances, albedo });
                    Some(out.fields.len() as u32 - 1)
                }),
                (Some(mesh), None) => {
                    log::warn!(
                        "reflection proxies: more than {} models; '{}' is traced by its bounds",
                        space_soup_engine::reflection_proxy::MAX_SHAPED_MODELS,
                        mesh.path,
                    );
                    None
                }
                (None, _) => None,
            };
            // ITS CARDS, where the bake pictured it. See `proxy_cards`.
            let cards = match (mesh, object) {
                (Some(_), Some(o)) => loaded_cards.iter().find(|c| c.object_id == o.id).map(|c| {
                    pictured.push((p.object, c, p.centre, p.half_size, p.rotation));
                    out.cards.push(ProxyCards {
                        resolution: c.resolution,
                        texels: c.texels.clone(),
                        normals: c.normals.clone(),
                        albedo: c.albedo.clone(),
                    });
                    out.cards.len() as u32 - 1
                }),
                _ => None,
            };
            out.proxies.push(ProbeProxy {
                centre: p.centre,
                half_size: p.half_size,
                rotation: p.rotation,
                volume: p.room as u32,
                solid: p.solid,
                field,
                cards,
            });
        }
        out.glare = crate::glare_fixtures::measure(objects, &pictured);
        out
    }

    /// [`ProbeLevel::proxies`] for the scene file itself, its transforms
    /// resolved to world space. Empty when the scene does not load.
    pub fn scene_proxies(&self, game_dir: &Path, scene: &str) -> ReflectionProxies {
        let path = space_soup_engine::Manifest::scene_path(game_dir, scene);
        match space_soup_engine::scene::Scene::load(&path) {
            Ok(mut s) => {
                s.resolve_world_transforms();
                self.proxies(game_dir, scene, &s.objects)
            }
            Err(e) => {
                log::warn!("reflection proxies: {} did not load: {e:#}", path.display());
                ReflectionProxies::default()
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

    /// THE BUILDINGS' OUTSIDES the bake wrote beside the probes: each one's
    /// world box and its six outside faces, at this level's face size -- one
    /// baked at another is left out and said so. See the renderer's
    /// `set_building_outsides`.
    pub fn buildings(&self, game_dir: &Path, scene: &str) -> Vec<(Vec3, Vec3, Vec<u8>)> {
        reflection_probe::load_scene_buildings(game_dir, scene)
            .iter()
            .filter_map(|e| {
                let (res, faces) = reflection_probe::decode_probe(e)?;
                if res != self.resolution {
                    log::warn!("building outside '{}': {res}px faces, the probes' are {}; re-bake", e.object_id, self.resolution);
                    return None;
                }
                Some((Vec3::from(e.min), Vec3::from(e.max), faces))
            })
            .collect()
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
    /// sconces -- and nothing of the rooms' own shells -- and the four door
    /// leaves, each once for both rooms of its doorway: a shut leaf stands in
    /// the wall between two boxes, and the trace meets it from either side
    /// only as a proxy of the room it is entering (`door_proxies`). Without
    /// them the marble showed the lit hallway through a shut door (headset,
    /// 2026-10-08). Each lamp's box is the
    /// MODEL's, hanging below its ceiling mount, not the 20 cm handle the
    /// editor gives the object.
    #[test]
    fn test_room_proxies_are_the_pillar_and_the_fixtures() {
        let game = Path::new(env!("CARGO_MANIFEST_DIR")).join("../game");
        let Some(level) = ProbeLevel::load(&game, "test_room") else {
            eprintln!("skipping: no test_room probes");
            return;
        };
        let ReflectionProxies { proxies, fields, cards, .. } = level.scene_proxies(&game, "test_room");
        for p in &proxies {
            eprintln!("proxy room {} centre {:?} half {:?}", p.volume, p.centre, p.half_size);
            assert!(p.half_size.max_element() < 2.0, "a room's shell became a proxy: {p:?}");
        }
        assert_eq!(proxies.len(), 6 + 8, "pillar + 3 hanging lamps + 2 sconces + 4 leaves x 2 rooms");
        let leaves: Vec<_> = proxies[6..].iter().collect();
        assert!(leaves.iter().all(|p| p.half_size.min_element() < 0.05 && !p.solid), "a door's proxy is its thin leaf: {leaves:?}");
        let fixtures = &proxies[..6];
        let pillar = proxies.iter().find(|p| (p.centre - Vec3::new(0.0, 1.55, -7.0)).length() < 1e-3).expect("the pillar");
        assert!((pillar.half_size - Vec3::new(0.45, 1.55, 0.45)).length() < 1e-3);
        // Hanging lamps: mounted at y = 3.1, the shade below it.
        for x in [0.0, 14.0] {
            let lamp = proxies.iter().find(|p| (p.centre.x - x).abs() < 0.5 && p.centre.y > 1.5).expect("a hanging lamp");
            assert!(lamp.centre.y + lamp.half_size.y <= 3.2, "the lamp reaches through the ceiling: {lamp:?}");
            assert!(lamp.half_size.y > 0.15, "the lamp's box is the editor handle, not the model: {lamp:?}");
        }
        // Every fixture is traced by its own shape: one field per model and
        // scale -- the three hanging lamps share one, the two sconces another
        // -- and the pillar, a brush, is its box.
        assert!(pillar.field.is_none(), "the pillar is a brush: {pillar:?}");
        let models: Vec<_> = fixtures.iter().filter(|p| !p.solid).collect();
        assert_eq!(models.len(), 5);
        assert!(leaves.iter().all(|p| p.field.is_some()), "a leaf without a field: {leaves:?}");
        assert!(models.iter().all(|p| p.field.is_some()), "a fixture without a field: {models:?}");
        let mut distinct: Vec<u32> = models.iter().chain(&leaves).filter_map(|p| p.field).collect();
        distinct.sort();
        distinct.dedup();
        assert_eq!(distinct.len(), fields.len(), "a field no proxy uses, or a proxy naming no field");
        assert!(fields.len() <= 5, "{} fields for test_room's two or three fixture models and two leaves", fields.len());
        for f in &fields {
            assert_eq!(f.distances.len() as u32, f.dims.iter().product::<u32>());
        }
        // A bake with cards pictures every fixture on its own, and a proxy
        // names its own cards; the pillar has none.
        if !cards.is_empty() {
            assert!(pillar.cards.is_none(), "the pillar is in the photographs: {pillar:?}");
            assert!(models.iter().all(|p| p.cards.is_some()), "a fixture without cards: {models:?}");
            assert_eq!(cards.len(), 5, "one set of cards a placed fixture");
        }
    }

    /// The engine numbers the models the baker leaves out of the photographs;
    /// the renderer holds a field for each. The two budgets are one number.
    #[test]
    fn the_shaped_model_budget_is_the_renderers_field_budget() {
        assert_eq!(
            space_soup_engine::reflection_proxy::MAX_SHAPED_MODELS,
            space_soup::renderer::proxy_field::MAX_PROXY_FIELDS,
        );
        assert_eq!(space_soup_engine::reflection_cards::CARD_FACES, space_soup::renderer::proxy_cards::CARD_FACES);
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
