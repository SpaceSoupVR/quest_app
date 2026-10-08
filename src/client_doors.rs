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
use space_soup_engine::door::DoorGeom;

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
                let start = o.door.as_ref().map_or(0.0, |d| d.start_deg.to_radians()).clamp(geom.lo, geom.hi);
                Some(ClientDoor { id: o.id.clone(), geom, server: start, drawn: start })
            })
            .collect();
        Self { doors }
    }

    pub fn is_door(&self, id: &str) -> bool {
        self.doors.iter().any(|d| d.id == id)
    }

    /// Take the server's latest poses (`id`, rotation) and this frame's local
    /// hands (world centres), and set each leaf's drawn angle.
    pub fn update<'a>(&mut self, server: impl IntoIterator<Item = (&'a str, Quat)>, hands: &[Vec3], dt: f32) {
        for (id, rot) in server {
            if let Some(d) = self.doors.iter_mut().find(|d| d.id == id) {
                d.server = d.geom.angle_of(rot);
            }
        }
        let spheres: Vec<(Vec3, f32)> = hands.iter().map(|h| (*h, HAND_RADIUS)).collect();
        let k = if dt > 0.0 { 1.0 - (-dt / EASE_SECONDS).exp() } else { 1.0 };
        for d in &mut self.doors {
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
        ClientDoors { doors: vec![ClientDoor { id: "d".into(), geom, server: 0.0, drawn: 0.0 }] }
    }

    #[test]
    fn a_local_hand_moves_the_drawn_leaf_before_the_server_does() {
        let mut d = doors();
        // The server still has it shut; a hand is 1 cm into its +z face.
        d.update([("d", Quat::IDENTITY)], &[Vec3::new(0.2, 1.0, 0.06)], 1.0 / 72.0);
        let a = d.doors[0].drawn;
        // Turned away from the hand (positive: +x toward -z) just clear of it.
        assert!(a > 0.5f32.to_radians(), "the hand is inside the drawn leaf: {}", a.to_degrees());
        // With no hand, it eases back toward the server within a few frames.
        for _ in 0..30 {
            d.update(std::iter::empty(), &[], 1.0 / 72.0);
        }
        assert!(d.doors[0].drawn.abs() < 0.1f32.to_radians());
    }

    #[test]
    fn the_drawn_leaf_follows_the_server_and_says_when_it_seals_its_doorway() {
        let mut d = doors();
        let geom = d.doors[0].geom;
        let (_, r) = geom.pose_at(1.0);
        for _ in 0..40 {
            d.update([("d", r)], &[], 1.0 / 72.0);
        }
        assert!((d.doors[0].drawn - 1.0).abs() < 1e-3);
        assert!(!d.views()[0].shut);
        d.set_angles_deg(&[0.5]);
        assert!(d.views()[0].shut);
        let (id, p, _) = d.poses().next().unwrap();
        assert_eq!(id, "d");
        assert!((p - geom.pose_at(0.5f32.to_radians()).0).length() < 1e-5);
    }
}
