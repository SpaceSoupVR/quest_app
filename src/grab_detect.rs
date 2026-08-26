
use std::collections::HashMap;
use std::path::Path;

use glam::{Quat, Vec3};

use space_soup_engine::rigid_physics::PhysicsWorld;
use space_soup_engine::{
    distance_to_oriented_box, GripKind, GripPointDef, Hand, Manifest, PartAnimationDef, Scene,
};
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
    // Client-local physics world (static rigid_body colliders only -- walls/floor/ramps),
    // rebuilt from the same scene JSON the server loads. Lets the client run its own wall
    // and ground collision for player movement without depending on the server (see
    // movement::step_locomotion) -- the same PhysicsWorld type and rebuild() call the
    // server uses, so client and server never disagree about where geometry is.
    pub physics: PhysicsWorld,
    /// Which sky the scene asked for, if any. The panorama itself lives in the
    /// shared library under game/skies/, so only the reference travels here.
    pub sky: Option<space_soup_engine::SkyDef>,
}

impl StaticScene {
    pub fn load(game_dir: &Path, scene_name: &str) -> Self {
        let path = Manifest::scene_path(game_dir, scene_name);
        let mut physics = PhysicsWorld::new();
        let mut sky = None;
        let (grip_points, part_animations) = match Scene::load(&path) {
            Ok(scene) => {
                physics.rebuild(&scene, game_dir);
                sky = scene.sky.clone();

                let mut grip_points = HashMap::new();
                let mut part_animations = HashMap::new();
                let total_objs = scene.objects.len();
                for o in scene.objects {
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
                (grip_points, part_animations)
            }
            Err(e) => {
                log::warn!(
                    "grab_detect: failed to load scene '{scene_name}' for grip points/part animations: {e}"
                );
                (HashMap::new(), HashMap::new())
            }
        };
        Self {
            scene_name: scene_name.to_string(),
            grip_points,
            part_animations,
            physics,
            sky,
        }
    }
}

/// Distance from a point to this object's box, honouring its rotation.
///
/// The geometry lives in the engine (`distance_to_oriented_box`) because every
/// module in this crate is `#[cfg(target_os = "android")]` -- nothing here can
/// be tested without building an APK and putting on a headset. See
/// scene_tests_cuboid_geom.rs for what it is pinned to.
fn distance_to_box(c: &LiveCuboid, point: Vec3) -> f32 {
    distance_to_oriented_box(c.position, c.rotation, c.half_size, point)
}

pub fn nearest_object_to(live: &LiveObjects, point: Vec3) -> Option<String> {
    live.by_id
        .iter()
        .map(|(id, c)| (id.clone(), distance_to_box(c, point)))
        .filter(|(_, d)| *d <= GRAB_RANGE)
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(id, _)| id)
}

/// The nearest object a bare-proximity grab is allowed to take.
///
/// Objects with authored grip points are excluded. If someone placed grips on a
/// weapon, those grips ARE the contract for holding it -- and a proximity grab
/// is not a near-miss version of one, it is a different thing: it attaches the
/// object at whatever offset it already had instead of snapping the grip onto
/// the hand. That is why a rifle could be picked up by the muzzle and carried
/// hanging half a metre off the palm.
///
/// The m4a1 makes the gap concrete: a 90 cm box with two 15 cm grip spheres on
/// it, so most of the weapon's length was only reachable through this fallback.
/// Excluding it means such a grab now fails, visibly, instead of succeeding
/// wrongly -- which is the right failure, and points at the authoring (another
/// grip point, or a wider grab_range) rather than hiding it.
pub fn nearest_grabbable_object_to(
    live: &LiveObjects,
    static_scene: &StaticScene,
    point: Vec3,
) -> Option<String> {
    live.by_id
        .iter()
        .filter(|(id, _)| !static_scene.grip_points.contains_key(id.as_str()))
        .map(|(id, c)| (id.clone(), distance_to_box(c, point)))
        .filter(|(_, d)| *d <= GRAB_RANGE)
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(id, _)| id)
}

/// The posed transform of the part a grip rides, if it rides one.
///
/// `part_transforms` holds world-space poses published by render_prep last frame
/// -- the headset is the only place a skinned pose exists. The composition
/// itself lives on `GripPointDef` in the engine, where it is unit-tested; this
/// is only the map lookup, which is all the client knows that the engine does
/// not.
pub fn part_pose(
    gp: &GripPointDef,
    object_id: &str,
    part_transforms: &HashMap<String, HashMap<String, ([f32; 3], [f32; 4])>>,
) -> Option<(Vec3, Quat)> {
    let name = gp.part.as_ref()?;
    let (pos, rot) = part_transforms.get(object_id)?.get(name)?;
    Some((Vec3::from(*pos), Quat::from_array(*rot)))
}

pub fn nearest_grip_point_to(
    live: &LiveObjects,
    static_scene: &StaticScene,
    point: Vec3,
    trigger_only: bool,
    hand: Hand,
    part_transforms: &HashMap<String, HashMap<String, ([f32; 3], [f32; 4])>>,
) -> Option<(String, String)> {
    static_scene
        .grip_points
        .iter()
        .flat_map(|(id, points)| {
            let live_c = live.by_id.get(id);
            points.iter().filter(move |gp| gp.hand == hand).filter_map(move |gp| {
                let live_c = live_c?;
                let (world_pos, _) = gp.anchor_world(
                    live_c.position,
                    live_c.rotation,
                    part_pose(gp, id, part_transforms),
                );
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

/// What this hand is holding.
///
/// The server's answer first, then what we grabbed and have not released. The
/// second is not redundant: a proximity grab stores no grip point name, so
/// `resolve_held_grip` finds nothing to report unless the object also has a
/// legacy `grip_pose` for that hand -- the m4a1 has neither, so the server says
/// "empty hand" for a rifle the player is visibly carrying.
///
/// Proximity is the last resort and answers a different question ("what is near
/// my hand") than the one being asked ("what am I holding").
pub fn held_object_id(
    held: Option<&WireHeldGrip>,
    grabbed: Option<&str>,
    live: &LiveObjects,
    hand_pos: Vec3,
) -> Option<String> {
    if let Some(held) = held {
        return Some(held.object_id.clone());
    }
    if let Some(id) = grabbed {
        return Some(id.to_string());
    }
    nearest_object_to(live, hand_pos)
}

