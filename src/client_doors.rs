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
}

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
        Self { doors }
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
    pub fn update<'a>(&mut self, server: impl IntoIterator<Item = (&'a str, Quat)>, hands: &[Vec3], head: Option<Vec3>, dt: f32) {
        for (id, rot) in server {
            if let Some(d) = self.doors.iter_mut().find(|d| d.id == id) {
                d.server = d.geom.angle_of(rot);
                d.served = true;
            }
        }
        let spheres: Vec<(Vec3, f32)> = hands.iter().map(|h| (*h, HAND_RADIUS)).collect();
        let heads: Vec<Vec3> = head.into_iter().collect();
        let k = if dt > 0.0 { 1.0 - (-dt / EASE_SECONDS).exp() } else { 1.0 };
        for d in self.doors.iter_mut().filter(|d| !d.served) {
            let s = d.geom.step_hinge(HingeState { angle: d.drawn, spin: d.spin }, &d.load, &heads, &spheres, dt);
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
        ClientDoors { doors: vec![ClientDoor { id: "d".into(), geom, server: 0.0, drawn: 0.0, load, spin: 0.0, served: true }] }
    }

    #[test]
    fn a_local_hand_moves_the_drawn_leaf_before_the_server_does() {
        let mut d = doors();
        // The server still has it shut; a hand is 1 cm into its +z face.
        d.update([("d", Quat::IDENTITY)], &[Vec3::new(0.2, 1.0, 0.06)], None, 1.0 / 72.0);
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
}
