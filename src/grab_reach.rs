//! WHAT EACH HAND IS OFFERED TO TAKE, on this headset: the loose things
//! round it, from the grid the live objects are filed in (`GrabGrid`), and
//! the handles of the doors beside it. `space_soup_engine::grab_target`
//! chooses among them, keeps the choice steady and shares it between the
//! hands; the choice is lit (`ModelUniform::set_highlight`) and a grip press
//! takes it (`part_pull::handle_input`).
//!
//! The query is the cells within [`grab_target::QUERY_REACH`] of the grip --
//! what is near the hand, never the level -- plus the scene's doors, a short
//! list each looked at with one distance.
//!
//! Host-compiled, so it is tested off the headset on the shipped level.

use std::collections::HashSet;

use glam::Vec3;

use space_soup_engine::grab_index::GrabGrid;
use space_soup_engine::grab_target::{self, Offer, TakeKind};

use crate::client_doors::ClientDoors;

/// A hand as the targets see it: its grip (where it holds) and its aim (unit,
/// where it points), world.
#[derive(Debug, Clone, Copy)]
pub struct HandProbe {
    pub grip: Vec3,
    pub aim: Vec3,
}

/// THE SCENE'S OBJECTS A HAND MAY NOT TAKE BY REACHING: everything authored
/// but its loose bodies (a Dynamic rigid body that is not a door). What the
/// scene does not name -- spawned while it runs -- is not in here, and can
/// be taken.
pub fn fixed_objects(scene: &space_soup_engine::Scene) -> HashSet<String> {
    scene
        .objects
        .iter()
        .filter(|o| {
            let loose = o.rigid_body.as_ref().is_some_and(|b| b.mode == space_soup_engine::BodyMode::Dynamic);
            !loose || o.door.is_some()
        })
        .map(|o| o.id.clone())
        .collect()
}

/// What a hand at `hand` is offered: each loose thing (`may_take`) whose box
/// is within reach in `grid`, and each door handle within its own reach
/// (`door::HANDLE_REACH`), lit by the door's lever where it has one.
pub fn offers(grid: &GrabGrid, may_take: impl Fn(&str) -> bool, doors: &ClientDoors, hand: HandProbe) -> Vec<Offer> {
    let mut out = Vec::new();
    grid.each_within(hand.grip, grab_target::QUERY_REACH, |id, centre, _rot, half, distance| {
        if may_take(id) {
            out.push(Offer::of_box(id, TakeKind::Loose, hand.grip, hand.aim, centre, half, distance));
        }
    });
    for d in doors.doors.iter().filter(|d| !d.geom.fixed()) {
        let Some(latch) = d.latch else { continue };
        let distance = latch.distance(d.geom.pose_at(d.drawn), d.geom.half.z, hand.grip);
        // Kept a little past the reach (`grab_target::KEEP_SLACK`) so the
        // lever's glow holds steady at the edge.
        if distance <= space_soup_engine::door::HANDLE_REACH * grab_target::KEEP_SLACK {
            out.push(Offer {
                id: d.id.clone(),
                glow: d.lever.as_ref().map_or_else(|| d.id.clone(), |(lever, _, _)| lever.clone()),
                kind: TakeKind::Handle,
                distance,
                angle: 0.0,
                reach: Some(space_soup_engine::door::HANDLE_REACH),
            });
        }
    }
    out
}

/// Both hands' offers, `[left, right]` (`None` for a hand not tracked).
pub fn offers_for(
    grid: &GrabGrid,
    may_take: impl Fn(&str) -> bool + Copy,
    doors: &ClientDoors,
    hands: [Option<HandProbe>; 2],
) -> [Option<Vec<Offer>>; 2] {
    hands.map(|h| h.map(|h| offers(grid, may_take, doors, h)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use space_soup_engine::grab_target::{GlowFade, GrabTargets};

    struct Level {
        scene: space_soup_engine::Scene,
        grid: GrabGrid,
        fixed: HashSet<String>,
        doors: ClientDoors,
    }

    /// The shipped level as the headset holds it: every object's box in the
    /// grid as the server sends them (`LiveObjects::update`).
    fn test_room() -> Option<Level> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../game/scenes/test_room.json");
        let mut scene = space_soup_engine::scene::Scene::load(&path).ok()?;
        scene.resolve_world_transforms();
        let mut grid = GrabGrid::default();
        grid.begin();
        for o in &scene.objects {
            grid.set(&o.id, o.cuboid.position, o.cuboid.rotation, o.cuboid.half_size);
        }
        grid.end();
        let fixed = fixed_objects(&scene);
        let doors = ClientDoors::from_scene(&scene);
        Some(Level { scene, grid, fixed, doors })
    }

    fn at(level: &Level, id: &str) -> Vec3 {
        level.scene.find_object(id).unwrap_or_else(|| panic!("no {id}")).cuboid.position
    }

    fn choose(level: &Level, targets: &mut GrabTargets, hands: [Option<HandProbe>; 2], held: [Option<&str>; 2]) {
        let offers = offers_for(&level.grid, |id| !level.fixed.contains(id), &level.doors, hands);
        targets.update([offers[0].as_deref(), offers[1].as_deref()], held);
    }

    /// THE SHIPPED BEACH: a hand reaching down to the rocks is offered the
    /// one under it, and the one it points at from a short lean away; the
    /// sand, the sea and the walls are never offered; a full hand offers
    /// nothing and the other hand is never offered what it holds.
    #[test]
    fn the_shipped_rocks_are_chosen_by_reach_and_aim() {
        let Some(level) = test_room() else {
            eprintln!("skipping: no test_room");
            return;
        };
        let pebble = at(&level, "beach_pebble_3");
        let rock = at(&level, "beach_rock_2");
        let mut t = GrabTargets::default();
        // A hand 10 cm over the pebble, palm down: the pebble.
        let over = HandProbe { grip: pebble + Vec3::Y * 0.10, aim: Vec3::NEG_Y };
        choose(&level, &mut t, [None, Some(over)], [None, None]);
        assert_eq!(t.choice(1).map(|c| c.id.as_str()), Some("beach_pebble_3"));
        assert_eq!(t.choice(1).unwrap().glow, "beach_pebble_3");
        // Leaning, 45 cm off the fist rock and pointing at it: that rock,
        // though the pebble is a little nearer the grip than it is off-aim.
        let grip = rock + Vec3::new(0.0, 0.35, -0.28);
        let lean = HandProbe { grip, aim: (rock - grip).normalize() };
        let mut t = GrabTargets::default();
        choose(&level, &mut t, [None, Some(lean)], [None, None]);
        assert_eq!(t.choice(1).map(|c| c.id.as_str()), Some("beach_rock_2"), "{:?}", t.choice(1));
        // Pointing away from it from there: nothing (no force grab, no
        // reach behind the hand past touching).
        let away = HandProbe { grip, aim: -(rock - grip).normalize() };
        let mut t = GrabTargets::default();
        choose(&level, &mut t, [None, Some(away)], [None, None]);
        assert!(t.choice(1).is_none(), "{:?}", t.choice(1));
        // The left hand holds the pebble: the right, over it, is offered the
        // next thing, never the pebble.
        let mut t = GrabTargets::default();
        choose(&level, &mut t, [Some(over), Some(over)], [Some("beach_pebble_3"), None]);
        assert!(t.choice(0).is_none());
        assert_ne!(t.choice(1).map(|c| c.id.as_str()), Some("beach_pebble_3"));
        // Out in the hall, with walls, floor and lamps all round: nothing.
        let hall = HandProbe { grip: Vec3::new(0.0, 1.2, -6.0), aim: Vec3::NEG_Z };
        let mut t = GrabTargets::default();
        choose(&level, &mut t, [Some(hall), Some(hall)], [None, None]);
        assert!(t.choice(0).is_none() && t.choice(1).is_none(), "{:?}", t.hands);
    }

    /// THE SHIPPED BRICK DOOR: a hand at its handle is offered the door, and
    /// what is lit is the lever; a hand half a metre off is not.
    #[test]
    fn a_hand_at_the_shipped_door_handle_lights_the_lever() {
        let Some(level) = test_room() else {
            eprintln!("skipping: no test_room");
            return;
        };
        let (_, lever, _) = level.doors.rider_poses().next().expect("the lever");
        let mut t = GrabTargets::default();
        let hand = HandProbe { grip: lever + Vec3::new(-0.08, 0.0, 0.0), aim: Vec3::X };
        choose(&level, &mut t, [Some(hand), None], [None, None]);
        let c = t.choice(0).expect("the handle");
        assert_eq!((c.id.as_str(), c.glow.as_str(), c.kind), ("brick_door_north", "brick_door_lever", TakeKind::Handle));
        let far = HandProbe { grip: lever + Vec3::new(-0.5, 0.0, 0.0), aim: Vec3::X };
        let mut t = GrabTargets::default();
        choose(&level, &mut t, [Some(far), None], [None, None]);
        assert!(t.choice(0).is_none(), "{:?}", t.choice(0));
    }

    /// ONLY THE CHOICE IS LIT, and the light follows the hand from one rock
    /// to the next and goes out: what is written each frame is the lit
    /// thing's level, and nothing once it is out.
    #[test]
    fn only_the_choice_is_lit_and_the_light_follows_the_hand() {
        let Some(level) = test_room() else {
            eprintln!("skipping: no test_room");
            return;
        };
        let mut t = GrabTargets::default();
        let mut glow = GlowFade::default();
        let written: std::cell::RefCell<std::collections::HashMap<String, f32>> = Default::default();
        let mut frame = |grip: Option<Vec3>, time: f64| {
            let hand = grip.map(|g| HandProbe { grip: g, aim: Vec3::NEG_Y });
            choose(&level, &mut t, [None, hand], [None, None]);
            let wanted: Vec<&str> = t.hands.iter().flatten().map(|c| c.glow.as_str()).collect();
            let out = glow.step(1.0 / 72.0, time, &wanted);
            for (id, l) in &out {
                written.borrow_mut().insert(id.clone(), *l);
            }
            out.len()
        };
        let (p3, b) = (at(&level, "beach_pebble_3"), at(&level, "beach_boulder"));
        let mut time = 0.0;
        for _ in 0..30 {
            time += 1.0 / 72.0;
            frame(Some(p3 + Vec3::Y * 0.08), time);
        }
        let w = written.borrow().clone();
        assert!(w["beach_pebble_3"] > 0.6);
        assert_eq!(w.len(), 1, "something else was lit: {w:?}");
        for _ in 0..60 {
            time += 1.0 / 72.0;
            frame(Some(b + Vec3::Y * 0.12), time);
        }
        let w = written.borrow().clone();
        assert!(w["beach_boulder"] > 0.6, "{w:?}");
        assert_eq!(w["beach_pebble_3"], 0.0, "the pebble kept its light: {w:?}");
        let mut idle = 0;
        for _ in 0..60 {
            time += 1.0 / 72.0;
            idle = frame(Some(Vec3::new(0.0, 1.2, -6.0)), time);
        }
        assert_eq!(written.borrow()["beach_boulder"], 0.0);
        assert_eq!(idle, 0, "with nothing lit, something was still written");
    }
}
