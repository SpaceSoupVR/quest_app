//! THE DOORS ON THIS HEADSET: where each leaf is drawn, collided and shadowed.
//!
//! The server owns a door (`space_soup_engine::door`; its PhysX hinge and the
//! bodies pushing it). Its answer reaches the headset a network round trip
//! after the hand that pushed it, so a door drawn where the server last put it
//! trails the hand pushing it -- by 30-60 ms on a home network, and a hand
//! passes through the drawn leaf before it moves. So the headset draws each
//! leaf at the server's angle turned just far enough that no local hand sits
//! inside it (`DoorGeom::clear_spheres`): the leaf moves with the hand at once,
//! and the server, given the same hand, arrives at the same push a few frames
//! later and takes over. Away from any hand the drawn angle eases to the
//! server's in `EASE_SECONDS`, which also smooths the snapshots' steps.
//!
//! Host-compiled, so it is tested off the headset; the frame calls it.

use glam::{Quat, Vec3};

use space_soup::renderer::doors::DoorView;
use space_soup_engine::door::{DoorGeom, HingeLoad, HingeState};

/// How long the drawn angle takes to close most of its gap to the server's,
/// seconds (a time constant). A snapshot arrives every 1/60 s; easing over
/// a little longer hides their steps without visible lag.
pub const EASE_SECONDS: f32 = 0.04;

/// A hand's radius as it pushes a leaf, metres: a palm, a little fuller than
/// the server's 3.5 cm grip anchor so the drawn leaf never shows the hand's
/// mesh poking through it.
pub const HAND_RADIUS: f32 = 0.05;

#[derive(Clone, Debug)]
pub struct ClientDoor {
    pub id: String,
    pub geom: DoorGeom,
    /// The server's angle, from its last snapshot.
    pub server: f32,
    /// The angle drawn this frame.
    pub drawn: f32,
    /// What its hinge resists with, and how fast it turns on the headset's
    /// own hinge (`step_hinge`) -- used until a server sends this door.
    pub load: HingeLoad,
    pub spin: f32,
    /// A server has sent this door: it is the server's to move.
    pub served: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ClientDoors {
    pub doors: Vec<ClientDoor>,
    /// Last frame's hand spheres, so each hand's push is its own speed.
    last_hands: Vec<(Vec3, f32)>,
    /// Frames since the last `DOORDIAG` line.
    frames: u32,
}

/// How often the headset logs one `DOORDIAG` line, frames: once a second.
const DIAG_EVERY: u32 = 72;

/// THE PARTS OF A HAND THAT PUSH A DOOR, world spheres: each hand's grip and
/// palm, a fist's worth, and its five finger tips, as tracked hands report
/// them -- a controller reports the grip alone. `joint` gives a joint's
/// position in the world, or none where this rig has no such joint.
pub fn hand_spheres(joint: impl Fn(space_soup_engine::JointId) -> Option<Vec3>) -> Vec<(Vec3, f32)> {
    use space_soup_engine::rig::FingerJoint as F;
    use space_soup_engine::Hand;
    use space_soup_engine::JointId as J;
    let mut out = Vec::new();
    for h in [Hand::Left, Hand::Right] {
        out.extend(joint(J::HandGrip(h)).map(|p| (p, HAND_RADIUS)));
        out.extend(joint(J::Finger(h, F::Palm)).map(|p| (p, HAND_RADIUS)));
        for tip in [F::ThumbTip, F::IndexTip, F::MiddleTip, F::RingTip, F::LittleTip] {
            out.extend(joint(J::Finger(h, tip)).map(|p| (p, FINGER_RADIUS)));
        }
    }
    out
}

/// A finger tip's radius as it pushes a leaf, metres.
pub const FINGER_RADIUS: f32 = 0.012;

impl ClientDoors {
    /// The scene's doors, closed (or at their authored start).
    pub fn from_scene(scene: &space_soup_engine::Scene) -> Self {
        let doors = scene
            .objects
            .iter()
            .filter_map(|o| {
                let geom = DoorGeom::of(o)?;
                let def = o.door.clone().unwrap_or_default();
                let start = def.start_deg.to_radians().clamp(geom.lo, geom.hi);
                let load = HingeLoad::of(&geom, &def);
                Some(ClientDoor { id: o.id.clone(), geom, server: start, drawn: start, load, spin: 0.0, served: false })
            })
            .collect();
        Self { doors, ..Default::default() }
    }

    pub fn is_door(&self, id: &str) -> bool {
        self.doors.iter().any(|d| d.id == id)
    }

    /// Take the server's latest poses (`id`, rotation), this frame's local
    /// hands (world centres) and the player's head (world), and set each
    /// leaf's drawn angle.
    ///
    /// A DOOR NO SERVER SENDS -- a game played alone, with no server at all --
    /// is turned here, on the headset's own hinge (`DoorGeom::step_hinge`),
    /// by the player's body and hands: the same pushes, friction and closer
    /// as the server's PhysX hinge. Before this it only followed a server, and
    /// alone the player walked into a door that never moved (headset,
    /// 2026-10-08).
    ///
    /// A SERVER CLAIMS A DOOR ONLY ONCE IT HAS MOVED IT. A server that sends
    /// the door at rest and never turns it -- one older than the doors, or
    /// one whose players' bodies never reach them -- claimed every door with
    /// its first snapshot and held it shut against the headset's own pushes
    /// (headset, Checkpoint 52).
    ///
    /// `hands`: world spheres (`hand_spheres`); each pushes with its speed
    /// since the last frame.
    pub fn update<'a>(&mut self, server: impl IntoIterator<Item = (&'a str, Quat)>, hands: &[(Vec3, f32)], head: Option<Vec3>, dt: f32) {
        for (id, rot) in server {
            if let Some(d) = self.doors.iter_mut().find(|d| d.id == id) {
                d.server = d.geom.angle_of(rot);
                d.served |= d.server.abs() > 1f32.to_radians();
            }
        }
        let spheres: Vec<(Vec3, f32)> = hands.to_vec();
        let paths: Vec<(Vec3, Vec3, f32)> = hands
            .iter()
            .enumerate()
            .map(|(i, (p, r))| {
                // From where this hand was, unless it jumped (a teleport, a
                // hand found again): then it pushes from where it is.
                let from = self.last_hands.get(i).map_or(*p, |(q, _)| *q);
                (if (from - *p).length() < 0.3 && self.last_hands.len() == hands.len() { from } else { *p }, *p, *r)
            })
            .collect();
        self.last_hands = spheres.clone();
        let heads: Vec<Vec3> = head.into_iter().collect();
        let k = if dt > 0.0 { 1.0 - (-dt / EASE_SECONDS).exp() } else { 1.0 };
        for d in self.doors.iter_mut().filter(|d| !d.served) {
            let s = d.geom.step_hinge(HingeState { angle: d.drawn, spin: d.spin }, &d.load, &heads, &paths, dt);
            d.drawn = s.angle;
            d.spin = s.spin;
            d.server = s.angle;
        }
        for d in self.doors.iter_mut().filter(|d| d.served) {
            // Ease toward the server, then clear the hands from wherever that
            // leaves the leaf: a hand inside it moves it at once.
            let eased = d.drawn + (d.server - d.drawn) * k;
            d.drawn = d.geom.clear_spheres(eased, &spheres);
        }
        self.frames += 1;
        if self.frames >= DIAG_EVERY {
            self.frames = 0;
            log::info!("{}", self.diag(hands, head));
        }
    }

    /// ONE `DOORDIAG` LINE: per door its angle, who moves it, how near the
    /// nearest hand sphere is to its leaf and how deep the body reaches into
    /// it (negative: short of it) -- what says on the device whether a hand
    /// or the body ever touches a leaf, and whether a server holds it.
    pub fn diag(&self, hands: &[(Vec3, f32)], head: Option<Vec3>) -> String {
        let mut line = format!("DOORDIAG hands {} head {:?}", hands.len(), head.map(|h| h.to_array().map(|v| (v * 100.0).round() / 100.0)));
        for d in &self.doors {
            let gap = |p: Vec3, r: f32| {
                let probe = r + 5.0;
                d.geom.body_contact(d.drawn, p, (p.y, p.y), probe).map_or(f32::MAX, |(_, _, depth)| probe - depth - r)
            };
            let hand = hands.iter().map(|(p, r)| gap(*p, *r)).fold(f32::MAX, f32::min);
            let body = head.map_or(f32::MAX, |h| {
                let span = (h.y - space_soup_engine::door::BODY_PUSH_SPAN.0, h.y - space_soup_engine::door::BODY_PUSH_SPAN.1);
                let probe = space_soup_engine::door::BODY_PUSH_RADIUS + 5.0;
                d.geom.body_contact(d.drawn, h, span, probe).map_or(f32::MAX, |(_, _, depth)| depth - 5.0)
            });
            line += &format!(
                " | {} {:.1}deg {} hand {:.2}m body {:+.2}m",
                d.id,
                d.drawn.to_degrees(),
                if d.served { "server" } else { "local" },
                hand,
                body
            );
        }
        line
    }

    /// Each leaf's drawn pose, world: `(id, centre, rotation)`.
    pub fn poses(&self) -> impl Iterator<Item = (&str, Vec3, Quat)> {
        self.doors.iter().map(|d| {
            let (p, r) = d.geom.pose_at(d.drawn);
            (d.id.as_str(), p, r)
        })
    }

    /// The doors as the renderer takes them (`XrRenderer::set_doors`).
    pub fn views(&self) -> Vec<DoorView> {
        self.doors
            .iter()
            .map(|d| DoorView {
                corners: d.geom.corners_at(d.drawn),
                closed_centre: d.geom.closed_pos,
                shut: DoorGeom::shut(d.drawn),
                rotation: d.geom.pose_at(d.drawn).1,
                closed_rotation: d.geom.closed_rot,
            })
            .collect()
    }

    /// Doors at given angles, degrees -- `DOORS=a,b,c,d` in the offline
    /// harness, in the scene's order -- with nothing to ease from.
    pub fn set_angles_deg(&mut self, angles: &[f32]) {
        for (d, a) in self.doors.iter_mut().zip(angles) {
            let a = a.to_radians().clamp(d.geom.lo, d.geom.hi);
            d.server = a;
            d.drawn = a;
            d.spin = 0.0;
            self.last_hands.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use space_soup_engine::door::DoorDef;

    fn doors() -> ClientDoors {
        let def = DoorDef { open_min_deg: -90.0, open_max_deg: 90.0, ..Default::default() };
        let geom = DoorGeom::new(Vec3::new(0.0, 1.1, 0.0), Quat::IDENTITY, Vec3::new(0.4, 1.09, 0.02), &def);
        let load = HingeLoad::of(&geom, &def);
        ClientDoors { doors: vec![ClientDoor { id: "d".into(), geom, server: 0.0, drawn: 0.0, load, spin: 0.0, served: true }], ..Default::default() }
    }

    #[test]
    fn a_local_hand_moves_the_drawn_leaf_before_the_server_does() {
        let mut d = doors();
        // The server still has it shut; a hand is 1 cm into its +z face.
        d.update([("d", Quat::IDENTITY)], &[(Vec3::new(0.2, 1.0, 0.06), HAND_RADIUS)], None, 1.0 / 72.0);
        let a = d.doors[0].drawn;
        // Turned away from the hand (positive: +x toward -z) just clear of it.
        assert!(a > 0.5f32.to_radians(), "the hand is inside the drawn leaf: {}", a.to_degrees());
        // With no hand, it eases back toward the server within a few frames.
        for _ in 0..30 {
            d.update(std::iter::empty(), &[], None, 1.0 / 72.0);
        }
        assert!(d.doors[0].drawn.abs() < 0.1f32.to_radians());
    }

    #[test]
    fn the_drawn_leaf_follows_the_server_and_says_when_it_seals_its_doorway() {
        let mut d = doors();
        let geom = d.doors[0].geom;
        let (_, r) = geom.pose_at(1.0);
        for _ in 0..40 {
            d.update([("d", r)], &[], None, 1.0 / 72.0);
        }
        assert!((d.doors[0].drawn - 1.0).abs() < 1e-3);
        assert!(!d.views()[0].shut);
        d.set_angles_deg(&[0.5]);
        assert!(d.views()[0].shut);
        let (id, p, _) = d.poses().next().unwrap();
        assert_eq!(id, "d");
        assert!((p - geom.pose_at(0.5f32.to_radians()).0).length() < 1e-5);
    }

    /// ALONE, WITH NO SERVER: the shipped hallway's first door, walked into
    /// from the hall at walking pace by a player the walk stops 25 cm from
    /// it, swings open in front of them and lets them through.
    #[test]
    fn alone_a_player_walking_into_the_shipped_hall_door_opens_it() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../game/scenes/test_room.json");
        let Ok(mut scene) = space_soup_engine::scene::Scene::load(&path) else {
            eprintln!("skipping: no test_room");
            return;
        };
        scene.resolve_world_transforms();
        let mut doors = ClientDoors::from_scene(&scene);
        let i = doors.doors.iter().position(|d| d.id == "hall_door_south").expect("the hall's south leaf");
        let g = doors.doors[i].geom;
        // From the hall (x < 2.85) toward the hallway, in line with the
        // leaf's middle.
        let (_, closed_rot) = g.pose_at(0.0);
        let _ = closed_rot;
        let mut head = Vec3::new(1.6, 1.6, g.closed_pos.z);
        let dt = 1.0 / 72.0;
        let mut through = false;
        for _ in 0..(72 * 6) {
            let mut next = head + Vec3::new(1.2 * dt, 0.0, 0.0);
            for d in &doors.doors {
                if let Some((_, n, depth)) = d.geom.body_contact(d.drawn, next, (next.y - 1.3, next.y - 0.25), 0.25) {
                    next += n * depth;
                }
            }
            head = next;
            doors.update(std::iter::empty(), &[], Some(head), dt);
            through |= head.x > 3.4;
        }
        let d = &doors.doors[i];
        assert!(!d.served);
        assert!(through, "never got through: head {head}, leaf at {:.1} deg", d.drawn.to_degrees());
    }

    /// A HAND PUSHES THE SHIPPED HALL DOOR OPEN, and a server that keeps
    /// sending it at rest (older than the doors) does not hold it shut.
    /// Replayed as the headset reports a tracked open hand -- grip, palm and
    /// five finger tips -- reaching into the leaf near its free edge at
    /// 0.8 m/s from the hall side, then a controller (grip alone) pushing the
    /// other leaf from the hallway side. No recorded device poses exist yet;
    /// this is the pose layout `hand_spheres` reads, moved as a push moves.
    #[test]
    fn a_tracked_hand_and_a_controller_push_the_shipped_hall_doors() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../game/scenes/test_room.json");
        let Ok(mut scene) = space_soup_engine::scene::Scene::load(&path) else {
            eprintln!("skipping: no test_room");
            return;
        };
        scene.resolve_world_transforms();
        let mut doors = ClientDoors::from_scene(&scene);
        let south = doors.doors.iter().position(|d| d.id == "hall_door_south").unwrap();
        let north = doors.doors.iter().position(|d| d.id == "hall_door_north").unwrap();
        let at_rest: Vec<(String, Quat)> = doors.doors.iter().map(|d| (d.id.clone(), d.geom.pose_at(0.0).1)).collect();
        let dt = 1.0 / 72.0;
        // The south leaf's free edge is toward the doorway's middle, z = -3.0.
        let free_z = -3.05;
        let open_hand = |palm: Vec3| -> Vec<(Vec3, f32)> {
            let mut h = vec![(palm + Vec3::new(-0.08, 0.0, 0.0), HAND_RADIUS), (palm, HAND_RADIUS)];
            for (k, dz) in [-0.06f32, -0.025, 0.0, 0.02, 0.04].iter().enumerate() {
                let reach = if k == 0 { 0.06 } else { 0.09 };
                h.push((palm + Vec3::new(reach, 0.02 * k as f32, *dz), FINGER_RADIUS));
            }
            h
        };
        let mut palm = Vec3::new(2.3, 1.2, free_z - 0.1);
        for _ in 0..(72 * 2) {
            palm.x = (palm.x + 0.8 * dt).min(3.3);
            let rest = at_rest.iter().map(|(id, r)| (id.as_str(), *r));
            doors.update(rest, &open_hand(palm), None, dt);
        }
        let s = doors.doors[south].drawn.to_degrees();
        assert!(!doors.doors[south].served, "a server that never moved it claimed it");
        assert!(s > 30.0, "the tracked hand pushed the south leaf to {s:.1} deg");
        // A controller from the hallway side pushes the north leaf the other way.
        let mut grip = Vec3::new(3.4, 1.2, -2.85);
        for _ in 0..(72 * 2) {
            grip.x = (grip.x - 0.8 * dt).max(2.4);
            let rest = at_rest.iter().map(|(id, r)| (id.as_str(), *r));
            doors.update(rest, &[(grip, HAND_RADIUS)], None, dt);
        }
        let n = doors.doors[north].drawn.to_degrees();
        assert!(n.abs() > 30.0, "the controller pushed the north leaf to {n:.1} deg");
        assert!(doors.diag(&[(grip, HAND_RADIUS)], None).starts_with("DOORDIAG"));
    }
}
