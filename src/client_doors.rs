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
//! A LATCHED DOOR'S BOLT is worked out here by the rule the server uses
//! (`DoorLatch::bolt`: drawn while a hand is at the handle), so a bolted leaf
//! is not drawn giving way to a hand the server will not let move it. A HAND
//! HOLDING THE HANDLE swings the drawn leaf on the headset's own hinge by the
//! spring the server's physics uses (`DoorGeom::step_hinge_held`), so it
//! follows the hand at once; let go, it eases to the server's. The handle's
//! LEVER, an object of its own, rides the drawn leaf and turns down while a
//! hand is at it (`rider_poses`).
//!
//! Host-compiled, so it is tested off the headset; the frame calls it.

use glam::{Quat, Vec3};

use space_soup::renderer::doors::DoorView;
use space_soup_engine::door::{DoorDef, DoorGeom, DoorLatch, HingeLoad, HingeState};

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
    /// Its handle and latch, if it has them, and whether the bolt is home.
    pub latch: Option<DoorLatch>,
    pub bolted: bool,
    /// The lever drawn on its handle: that object's id, and where it sits in
    /// the leaf's own frame.
    pub lever: Option<(String, Vec3, Quat)>,
    /// How far the lever is turned down, radians.
    pub lever_turn: f32,
    /// Where on the leaf (its own frame) each local hand holds its handle.
    held: [Option<Vec3>; 2],
}

/// How far a worked lever turns down, radians, and how long it takes.
pub const LEVER_TURN: f32 = 0.5;
const LEVER_SECONDS: f32 = 0.05;
/// A hand this far from the handle it held has let go of it, metres (the
/// server's `HOLD_BREAK`).
const HOLD_BREAK: f32 = 0.75;

impl ClientDoor {
    /// A door hung as `def` says, standing where it starts.
    pub fn new(id: &str, geom: DoorGeom, def: &DoorDef) -> Self {
        let start = def.start_deg.to_radians().clamp(geom.lo, geom.hi);
        let latch = DoorLatch::of(def);
        Self {
            id: id.to_string(),
            geom,
            server: start,
            drawn: start,
            load: HingeLoad::of(&geom, def),
            spin: 0.0,
            served: false,
            latch,
            bolted: latch.is_some_and(|l| l.bolt(false, start, 0.0, false)),
            lever: None,
            lever_turn: 0.0,
            held: [None, None],
        }
    }

    /// Whether a local hand holds this door's handle.
    pub fn is_held(&self) -> bool {
        self.held.iter().any(Option::is_some)
    }
}

#[derive(Clone, Debug, Default)]
pub struct ClientDoors {
    pub doors: Vec<ClientDoor>,
    /// Last frame's hand spheres, so each hand's push is its own speed.
    last_hands: Vec<(Vec3, f32)>,
    /// Last frame's grips [left, right], for a held handle's pull.
    last_grips: [Option<Vec3>; 2],
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
                let mut door = ClientDoor::new(&o.id, geom, &def);
                // Its lever, where the scene put it on the closed leaf.
                door.lever = def.lever.as_ref().and_then(|id| scene.find_object(id)).map(|l| {
                    let inv = geom.closed_rot.inverse();
                    (l.id.clone(), inv * (l.cuboid.position - geom.closed_pos), inv * l.cuboid.rotation)
                });
                Some(door)
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
        self.update_held(server, hands, head, [None, None], [None, None], dt)
    }

    /// The door whose handle a hand at `grip` (world) can take: within
    /// `HANDLE_REACH` of it, as the leaf is drawn. What a grip press asks
    /// before it looks for anything loose.
    pub fn handle_near(&self, grip: Vec3) -> Option<&str> {
        self.doors
            .iter()
            .filter(|d| !d.geom.fixed())
            .filter_map(|d| Some((d, d.latch?.distance(d.geom.pose_at(d.drawn), d.geom.half.z, grip))))
            .filter(|(_, dist)| *dist <= space_soup_engine::door::HANDLE_REACH)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(d, _)| d.id.as_str())
    }

    /// [`Self::update`] with the hands' grips, `[left, right]` (world), and
    /// what each has GRABBED (an object's id, as the grab was sent): a hand
    /// whose grab is a door holds that door's handle until it lets go.
    pub fn update_held<'a>(
        &mut self,
        server: impl IntoIterator<Item = (&'a str, Quat)>,
        hands: &[(Vec3, f32)],
        head: Option<Vec3>,
        grips: [Option<Vec3>; 2],
        grabbed: [Option<&str>; 2],
        dt: f32,
    ) {
        for d in self.doors.iter_mut() {
            let pose = d.geom.pose_at(d.drawn);
            for h in 0..2 {
                let holds = grabbed[h] == Some(d.id.as_str()) && d.latch.is_some() && !d.geom.fixed();
                d.held[h] = match (holds, grips[h], d.latch) {
                    (true, Some(grip), Some(latch)) => {
                        let local = d.held[h].unwrap_or_else(|| latch.hold_point(pose, d.geom.half.z, grip));
                        ((pose.0 + pose.1 * local).distance(grip) <= HOLD_BREAK).then_some(local)
                    }
                    _ => None,
                };
            }
            // THE BOLT, by the server's rule, and the lever a hand works.
            let worked = d.latch.is_some_and(|l| d.is_held() || l.worked(pose, d.geom.half.z, grips.into_iter().flatten()));
            if let Some(l) = d.latch {
                d.bolted = l.bolt(d.bolted, d.drawn, d.spin, worked);
            }
            let to = if worked { LEVER_TURN } else { 0.0 };
            d.lever_turn += (to - d.lever_turn) * if dt > 0.0 { 1.0 - (-dt / LEVER_SECONDS).exp() } else { 1.0 };
        }
        let last_grips = std::mem::replace(&mut self.last_grips, grips);
        for (id, rot) in server {
            if let Some(d) = self.doors.iter_mut().find(|d| d.id == id) {
                d.server = d.geom.angle_of(rot);
                d.served |= d.server.abs() > 1f32.to_radians();
                // A door the server has open is not bolted, whoever opened it.
                if d.served && d.server.abs() > 2.0 * space_soup_engine::door::LATCH_DEG.to_radians() {
                    d.bolted = false;
                }
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
        for d in self.doors.iter_mut().filter(|d| !d.geom.fixed()) {
            if d.is_held() {
                // BY ITS HANDLE: the headset's own hinge, pulled as the
                // server's physics will pull it, whoever owns the door.
                let pulls: Vec<(Vec3, Vec3, Vec3)> = (0..2)
                    .filter_map(|h| {
                        let (local, now) = (d.held[h]?, grips[h]?);
                        Some((last_grips[h].filter(|was| was.distance(now) < 0.3).unwrap_or(now), now, local))
                    })
                    .collect();
                let s = d.geom.step_hinge_held(HingeState { angle: d.drawn, spin: d.spin }, &d.load, &heads, &paths, &pulls, dt);
                (d.drawn, d.spin) = (s.angle, s.spin);
                if !d.served {
                    d.server = s.angle;
                }
            } else if d.bolted {
                // Bolted: it gives way to nothing. (A served one is the
                // server's, which holds it by the same rule.)
                d.drawn = if d.served { d.drawn + (d.server - d.drawn) * k } else { d.drawn };
                d.spin = 0.0;
            } else if !d.served {
                let s = d.geom.step_hinge(HingeState { angle: d.drawn, spin: d.spin }, &d.load, &heads, &paths, dt);
                d.drawn = s.angle;
                d.spin = s.spin;
                d.server = s.angle;
            } else {
                // Ease toward the server, then clear the hands from wherever
                // that leaves the leaf: a hand inside it moves it at once.
                let eased = d.drawn + (d.server - d.drawn) * k;
                d.drawn = d.geom.clear_spheres(eased, &spheres);
            }
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
                " | {} {:.1}deg {}{} hand {:.2}m body {:+.2}m",
                d.id,
                d.drawn.to_degrees(),
                if d.served { "server" } else { "local" },
                if d.is_held() { " HELD" } else if d.bolted { " bolted" } else { "" },
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

    /// WHAT RIDES A LEAF, where it is drawn: each door's lever, on the leaf
    /// as it stands and turned down as far as a hand has worked it --
    /// `(id, centre, rotation)`, world.
    pub fn rider_poses(&self) -> impl Iterator<Item = (&str, Vec3, Quat)> {
        self.doors.iter().filter_map(|d| {
            let (id, local, turned) = d.lever.as_ref()?;
            let (p, r) = d.geom.pose_at(d.drawn);
            Some((id.as_str(), p + r * *local, r * *turned * Quat::from_rotation_z(d.lever_turn)))
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
        ClientDoors { doors: vec![ClientDoor { served: true, ..ClientDoor::new("d", geom, &def) }], ..Default::default() }
    }

    /// A latched door: the same leaf (hinged on its -x edge), opening toward
    /// -z, with a handle 31 cm out from its middle and a lever object on it.
    fn latched() -> ClientDoors {
        let def = DoorDef { open_min_deg: 0.0, open_max_deg: 100.0, latch: true, handle: Some([0.31, -0.08, 0.0]), mass: 26.0, friction: 4.0, ..Default::default() };
        let geom = DoorGeom::new(Vec3::new(0.0, 1.1, 0.0), Quat::IDENTITY, Vec3::new(0.4, 1.09, 0.02), &def);
        let mut door = ClientDoor::new("d", geom, &def);
        door.lever = Some(("lever".into(), Vec3::new(0.31, -0.08, 0.0), Quat::IDENTITY));
        ClientDoors { doors: vec![door], ..Default::default() }
    }

    /// ALONE, A LATCHED DOOR: a hand shoving the leaf away from its handle,
    /// and the player's body walking into it, leave it shut; a hand at the
    /// handle draws the bolt and turns the lever down, and the same shove
    /// opens it; pushed home with no hand at the handle, it latches again.
    #[test]
    fn a_latched_door_gives_way_only_to_a_hand_at_its_handle() {
        let mut d = latched();
        let dt = 1.0 / 72.0;
        assert!(d.doors[0].bolted);
        // A palm pressed into the leaf 30 cm from the handle, for a second.
        let palm = Vec3::new(0.05, 1.5, 0.03);
        for _ in 0..72 {
            d.update_held(std::iter::empty(), &[(palm, HAND_RADIUS)], Some(Vec3::new(0.1, 1.6, 0.2)), [None, Some(palm)], [None, None], dt);
        }
        assert!(d.doors[0].bolted && d.doors[0].drawn.abs() < 1e-4, "a bolted leaf gave way: {} deg", d.doors[0].drawn.to_degrees());
        assert!(d.rider_poses().next().unwrap().2.angle_between(Quat::IDENTITY) < 1e-3, "the lever turned with no hand at it");
        // The hand moves to the handle and pushes on through.
        let mut hand = Vec3::new(0.31, 1.02, 0.06);
        for _ in 0..72 {
            hand.z -= 0.5 * dt;
            d.update_held(std::iter::empty(), &[(hand, HAND_RADIUS)], None, [None, Some(hand)], [None, None], dt);
        }
        let opened = d.doors[0].drawn.to_degrees();
        assert!(!d.doors[0].bolted && opened > 30.0, "a hand at the handle opened it {opened:.1} deg");
        // The lever went down under the hand, and rides the leaf.
        let (id, at, _) = d.rider_poses().next().unwrap();
        let (p, r) = d.doors[0].geom.pose_at(d.doors[0].drawn);
        assert_eq!(id, "lever");
        assert!((at - (p + r * Vec3::new(0.31, -0.08, 0.0))).length() < 1e-5);
        // Pushed home by a hand on the leaf's other face, away from the
        // handle, it latches again.
        // (Open, the leaf lies along -z from its hinge at x = -0.4; the palm
        // comes at its outer face from -x and stops well short of the handle.)
        let mut palm = Vec3::new(-0.8, 1.6, -0.5);
        for _ in 0..(72 * 4) {
            palm.x = (palm.x + 0.6 * dt).min(-0.1);
            d.update_held(std::iter::empty(), &[(palm, HAND_RADIUS)], None, [None, Some(palm)], [None, None], dt);
        }
        // ... and then presses it home, high on the leaf, far from the handle.
        let mut palm = Vec3::new(0.0, 1.6, -0.6);
        for _ in 0..(72 * 2) {
            palm.z = (palm.z + 0.5 * dt).min(-0.072);
            d.update_held(std::iter::empty(), &[(palm, HAND_RADIUS)], None, [None, Some(palm)], [None, None], dt);
        }
        for _ in 0..72 {
            d.update_held(std::iter::empty(), &[], None, [None, None], [None, None], dt);
        }
        assert!(d.doors[0].bolted, "it stands at {:.2} deg, unlatched", d.doors[0].drawn.to_degrees());
        assert!(d.doors[0].lever_turn < 0.02);
    }

    /// BY ITS HANDLE: a hand that grabs the handle and walks its arc swings
    /// the drawn leaf with it at once, on a door a server owns and has not
    /// yet moved; let go, the leaf eases to where the server has it.
    #[test]
    fn a_held_handle_swings_the_drawn_leaf_with_the_hand() {
        let mut d = latched();
        d.doors[0].served = true;
        let geom = d.doors[0].geom;
        let dt = 1.0 / 72.0;
        let rest = geom.pose_at(0.0).1;
        let handle = |angle: f32| {
            let (p, r) = geom.pose_at(angle);
            p + r * Vec3::new(0.31, -0.08, -(0.02 + space_soup_engine::door::HANDLE_STANDOFF))
        };
        assert_eq!(d.handle_near(handle(0.0)), Some("d"));
        assert_eq!(d.handle_near(handle(0.0) + Vec3::new(-0.4, 0.3, 0.0)), None);
        let mut worst = 0.0f32;
        for k in 0..72 {
            let a = 70f32.to_radians() * (k as f32 / 71.0);
            let hand = handle(a);
            d.update_held([("d", rest)], &[(hand, HAND_RADIUS)], None, [None, Some(hand)], [None, Some("d")], dt);
            worst = worst.max((d.doors[0].drawn - a).abs());
        }
        assert!(d.doors[0].is_held() && !d.doors[0].bolted);
        assert!(worst < 12f32.to_radians(), "the leaf trailed the hand by {:.1} deg", worst.to_degrees());
        assert!(d.doors[0].drawn > 55f32.to_radians(), "{:.1} deg", d.doors[0].drawn.to_degrees());
        // Let go: it is the server's again (which still has it shut here).
        for _ in 0..30 {
            d.update_held([("d", rest)], &[], None, [None, None], [None, None], dt);
        }
        assert!(!d.doors[0].is_held() && d.doors[0].drawn.abs() < 0.5f32.to_radians());
    }

    /// THE SHIPPED LEVEL: the brick hall's north leaf latches and carries
    /// its lever; its south leaf is bolted for good; the hall's pair swing.
    #[test]
    fn the_shipped_brick_door_latches_and_carries_its_lever() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../game/scenes/test_room.json");
        let Ok(mut scene) = space_soup_engine::scene::Scene::load(&path) else {
            eprintln!("skipping: no test_room");
            return;
        };
        scene.resolve_world_transforms();
        let doors = ClientDoors::from_scene(&scene);
        let get = |id: &str| doors.doors.iter().find(|d| d.id == id).unwrap_or_else(|| panic!("no {id}"));
        let (north, south) = (get("brick_door_north"), get("brick_door_south"));
        assert!(north.bolted && north.latch.is_some() && !north.geom.fixed());
        assert!(south.geom.fixed() && south.latch.is_none());
        assert!(!get("hall_door_north").bolted && get("hall_door_south").latch.is_none());
        // The lever sits on the handle, a metre off the floor, and a hand
        // there on either side of the leaf can take it.
        let (id, at, _) = doors.rider_poses().next().expect("the lever");
        assert_eq!(id, "brick_door_lever");
        assert!((at.y - 1.0).abs() < 0.01, "{at}");
        for side in [-0.07, 0.07] {
            assert_eq!(doors.handle_near(at + Vec3::new(side, 0.0, 0.0)), Some("brick_door_north"));
        }
        assert_eq!(doors.handle_near(at + Vec3::new(0.0, 0.5, 0.0)), None);
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
