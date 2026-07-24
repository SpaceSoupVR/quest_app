
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
            points.iter().filter(move |gp| gp.hand == hand).filter_map(move |gp| {
                let live_c = live_c?;
                let obj_mat =
                    glam::Mat4::from_rotation_translation(live_c.rotation, live_c.position);
                let world_pos = obj_mat.transform_point3(Vec3::from(gp.local_pos));
                let range = gp.grab_range.unwrap_or(GRAB_RANGE);
                Some((id.clone(), gp.name.clone(), gp.kind, point.distance(world_pos), range))
            })
        })
        .filter(|(_, _, kind, d, range)| *d <= *range && (*kind != GripKind::Pinch || trigger_only))
        .min_by(|a, b| a.3.partial_cmp(&b.3).unwrap())
        .map(|(id, name, _, _, _)| (id, name))
}

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

