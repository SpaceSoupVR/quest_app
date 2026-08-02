
use std::collections::HashMap;
use std::path::Path;

use glam::{Quat, Vec3};

use space_soup_engine::scene::OpticDef;
use space_soup_engine::{GripKind, GripPointDef, Hand, Manifest, PartAnimationDef, Scene};
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
    pub part_animations: HashMap<String, Vec<PartAnimationDef>>,
    pub optics: HashMap<String, OpticDef>,
}

impl StaticScene {
    pub fn load(game_dir: &Path, scene_name: &str) -> Self {
        let path = Manifest::scene_path(game_dir, scene_name);
        let (grip_points, part_animations, optics) = match Scene::load(&path) {
            Ok(scene) => {
                let mut grip_points = HashMap::new();
                let mut part_animations = HashMap::new();
                let mut optics = HashMap::new();
                let total_objs = scene.objects.len();
                for o in scene.objects {
                    if let Some(optic) = o.optic.clone() {
                        optics.insert(o.id.clone(), optic);
                    }
                    if !o.grip_points.is_empty() {
                        let summary: Vec<String> = o
                            .grip_points
                            .iter()
                            .map(|gp| format!("{}({:?},{:?})", gp.name, gp.hand, gp.kind))
                            .collect();
                        log::info!(
                            "grab_detect: '{}' loaded {} grip point(s): [{}]",
                            o.id,
                            o.grip_points.len(),
                            summary.join(", ")
                        );
                        grip_points.insert(o.id.clone(), o.grip_points);
                    }
                    if !o.part_animations.is_empty() {
                        part_animations.insert(o.id, o.part_animations);
                    }
                }
                log::info!(
                    "grab_detect: scene '{scene_name}' loaded — {total_objs} objects, {} with grip points, {} with part animations",
                    grip_points.len(),
                    part_animations.len()
                );
                (grip_points, part_animations, optics)
            }
            Err(e) => {
                log::warn!(
                    "grab_detect: failed to load scene '{scene_name}' for grip points/part animations: {e}"
                );
                (HashMap::new(), HashMap::new(), HashMap::new())
            }
        };
        Self {
            scene_name: scene_name.to_string(),
            grip_points,
            part_animations,
            optics,
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

pub struct GrabDiag {
    pub nearest_obj: Option<(String, f32)>,
    pub nearest_grip: Option<(String, String, f32, f32)>,
    pub live_count: usize,
}

impl GrabDiag {
    pub fn summary(&self, hand: Hand, hand_pos: Vec3) -> String {
        let obj = match &self.nearest_obj {
            Some((id, d)) => format!("nearest_obj='{id}' surf_d={d:.3}m (grab<={GRAB_RANGE:.2})"),
            None => "nearest_obj=none".to_string(),
        };
        let grip = match &self.nearest_grip {
            Some((id, name, d, range)) => {
                let reach = if d <= range { "IN-RANGE" } else { "too-far" };
                format!("nearest_grip='{id}'.{name} d={d:.3}m (need<={range:.2}) {reach}")
            }
            None => "nearest_grip=none-for-hand".to_string(),
        };
        format!(
            "GRABDIAG {:?}: hand=({:.2},{:.2},{:.2}) live={} {obj} {grip}",
            hand, hand_pos.x, hand_pos.y, hand_pos.z, self.live_count
        )
    }
}

pub fn grab_diagnostic(
    live: &LiveObjects,
    static_scene: &StaticScene,
    hand_pos: Vec3,
    hand: Hand,
) -> GrabDiag {
    let nearest_obj = live
        .by_id
        .iter()
        .map(|(id, c)| {
            let closest = hand_pos.clamp(c.position - c.half_size, c.position + c.half_size);
            (id.clone(), hand_pos.distance(closest))
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let nearest_grip = static_scene
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
                Some((id.clone(), gp.name.clone(), hand_pos.distance(world_pos), range))
            })
        })
        .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    GrabDiag {
        nearest_obj,
        nearest_grip,
        live_count: live.by_id.len(),
    }
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

