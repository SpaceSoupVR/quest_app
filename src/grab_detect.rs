//! Client-side grab-detection heuristics, now that physics/attachments live
//! entirely on the server. `nearest_object_to`/`nearest_grip_point_to` used
//! to read a local `GameRuntime`'s scene directly (see the now-deleted
//! `space_soup_hands::query`); they need the same two pieces of data, just
//! sourced differently: grip-point *definitions* are static per-scene
//! authored data (loaded once via a plain JSON parse, no PhysX/Rhai
//! involved), while object *positions* are live and come from whatever the
//! server most recently broadcast in its `WireWorld`.

use std::collections::HashMap;
use std::path::Path;

use glam::{Quat, Vec3};

use space_soup_engine::{GripKind, GripPointDef, Hand, Manifest, Scene};
use space_soup_protocol::{WireHeldGrip, WireObjectBounds};

const GRAB_RANGE: f32 = 0.15;

pub struct LiveCuboid {
    pub position: Vec3,
    pub rotation: Quat,
    pub half_size: Vec3,
}

/// This tick's object interaction bounds, refreshed from the latest
/// `WireWorld` broadcast's `object_bounds` — replaces reading
/// `runtime.scene().objects` directly. Deliberately *not* sourced from
/// `WireWorld::cuboids`: that list is render-only and skips any object with
/// a mesh (a rifle, say), which would make mesh-rendered objects invisible
/// to grab detection.
#[derive(Default)]
pub struct LiveObjects {
    pub by_id: HashMap<String, LiveCuboid>,
}

impl LiveObjects {
    pub fn update(&mut self, object_bounds: &[WireObjectBounds]) {
        self.by_id.clear();
        for b in object_bounds {
            self.by_id.insert(
                b.id.clone(),
                LiveCuboid {
                    position: Vec3::from(b.position),
                    rotation: Quat::from_array(b.rotation),
                    half_size: Vec3::from(b.half_size),
                },
            );
        }
    }
}

/// Static grip-point definitions for the scene currently reported by the
/// server (`WireWorld::scene_name`). Reloaded only when that name changes —
/// this is authored content, not simulation, so a plain `Scene::load` (pure
/// JSON parsing) is all it needs; no `GameRuntime`/PhysX/Rhai required.
pub struct StaticScene {
    pub scene_name: String,
    pub grip_points: HashMap<String, Vec<GripPointDef>>,
}

impl StaticScene {
    pub fn load(game_dir: &Path, scene_name: &str) -> Self {
        let path = Manifest::scene_path(game_dir, scene_name);
        let grip_points = match Scene::load(&path) {
            Ok(scene) => scene
                .objects
                .into_iter()
                .filter(|o| !o.grip_points.is_empty())
                .map(|o| (o.id, o.grip_points))
                .collect(),
            Err(e) => {
                log::warn!(
                    "grab_detect: failed to load scene '{scene_name}' for grip points: {e}"
                );
                HashMap::new()
            }
        };
        Self {
            scene_name: scene_name.to_string(),
            grip_points,
        }
    }
}

pub fn nearest_object_to(live: &LiveObjects, point: Vec3) -> Option<String> {
    live.by_id
        .iter()
        .map(|(id, c)| {
            let closest = point.clamp(c.position - c.half_size, c.position + c.half_size);
            (id.clone(), point.distance(closest))
        })
        .filter(|(_, d)| *d <= GRAB_RANGE)
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(id, _)| id)
}

pub fn nearest_grip_point_to(
    live: &LiveObjects,
    static_scene: &StaticScene,
    point: Vec3,
    trigger_only: bool,
    hand: Hand,
) -> Option<(String, String)> {
    static_scene
        .grip_points
        .iter()
        .flat_map(|(id, points)| {
            let live_c = live.by_id.get(id);
            // A hand only considers points tagged for it — so a left hand
            // never snaps into a pose authored for the right hand.
            points.iter().filter(move |gp| gp.hand == hand).filter_map(move |gp| {
                let live_c = live_c?;
                let obj_mat =
                    glam::Mat4::from_rotation_translation(live_c.rotation, live_c.position);
                let world_pos = obj_mat.transform_point3(Vec3::from(gp.local_pos));
                Some((id.clone(), gp.name.clone(), gp.kind, point.distance(world_pos)))
            })
        })
        .filter(|(_, _, kind, d)| *d <= GRAB_RANGE && (*kind != GripKind::Pinch || trigger_only))
        .min_by(|a, b| a.3.partial_cmp(&b.3).unwrap())
        .map(|(id, name, _, _)| (id, name))
}

/// What a hand is holding, for stamping `ButtonPress::object_id` — the
/// server's own held-grip report if there is one, otherwise the same
/// proximity fallback used for grab detection.
pub fn held_object_id(
    held: Option<&WireHeldGrip>,
    live: &LiveObjects,
    hand_pos: Vec3,
) -> Option<String> {
    if let Some(held) = held {
        return Some(held.object_id.clone());
    }
    nearest_object_to(live, hand_pos)
}
