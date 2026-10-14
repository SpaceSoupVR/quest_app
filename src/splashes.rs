//! SPLASHES: where something strikes the water, seen by the client and handed
//! to the renderer (`XrRenderer::add_splash`), which works out from how fast,
//! at what angle, how wide and how heavy it was everything the splash is
//! (`space_soup::renderer::effects::splash_shape`).
//!
//! Three things strike water here:
//! - FEET, wading: a splash a stride, from alternate feet, each foot swung
//!   forward and down into the water at a shallow angle, as fast as the walk
//!   -- and slower in deep water, which a leg pushes through rather than
//!   steps into.
//! - HANDS, slapped down through the surface, and pulled out of it fast.
//! - THINGS, a level's moving objects, falling in: as fast as they fall, as
//!   wide and as heavy as they are. Seen by the engine's own rule
//!   (`space_soup_engine::water::WaterWatch`), so one floating at the
//!   waterline does not go in again each time it bobs. The server knows their
//!   true speed and mass, and tells of each going-in in its snapshots
//!   (`WireWorld::water_entries`): those are raised by [`SplashWatch::told`],
//!   once each. Only from a server before that does the headset work them
//!   out itself ([`SplashWatch::step`]'s `things`), the speed from how the
//!   thing moved and the mass from its size, as physics does for a body with
//!   none authored -- never both, or each would splash twice.
//!
//! Everything is in the WORLD's frame, on the water's clock. Only the still
//! surface is known here (the footprints' height); the waves move it a hand
//! or so, which a splash's look does not need.
use glam::{Quat, Vec3};
use space_soup::renderer::effects::Splash;
use space_soup_engine::water::{world_half_extents, WaterWatch};
use space_soup_protocol::WireWaterEntry;
use std::collections::HashMap;

/// A stride, metres: a splash each, alternate feet.
const STRIDE: f32 = 0.65;
/// The slowest walk that splashes, m/s; slower, the feet are set down.
const WADE_SPEED: f32 = 0.25;
/// Water shallower than this under the feet is a wet floor, metres.
const WET: f32 = 0.02;
/// A move this far in one frame is a teleport or a respawn, not a walk.
const JUMP: f32 = 1.5;
/// The slowest hand that splashes going in, and coming out, m/s.
const SLAP_IN: f32 = 0.5;
const PULL_OUT: f32 = 1.0;
/// No hand splashes twice within this, seconds.
const COOLDOWN: f64 = 0.15;
/// A FOOT, as the water meets it: as wide as a disc of this radius, metres
/// (a foot and the ankle above it are some 150 cm2 to the water they push);
/// with the shank that drives it, this heavy, kilograms; and coming down
/// into the water this far below level, a wading step's shallow angle.
const FOOT_RADIUS: f32 = 0.07;
const FOOT_MASS: f32 = 3.0;
const FOOT_ENTRY_DEG: f32 = 25.0;
/// A HAND, likewise, with the forearm behind it.
const HAND_RADIUS: f32 = 0.045;
const HAND_MASS: f32 = 0.6;

#[derive(Default)]
pub struct SplashWatch {
    feet: Option<Vec3>,
    walked: f32,
    left_foot: bool,
    hands: [Option<(Vec3, f64)>; 2],
    hand_last: [f64; 2],
    /// Each object's centre when it last moved, when, and how fast it was
    /// going then.
    things: HashMap<String, (Vec3, f64, Vec3)>,
    /// Which of them are in the water.
    water: WaterWatch,
    /// The server's entries in the last snapshot, by number: those already
    /// raised.
    told: Vec<u64>,
    count: u64,
    /// FLOATING THINGS: each one's height and when it was last seen to
    /// move, how fast it was going up or down then, and the fastest it has
    /// sunk since its last bob.
    bobbing: HashMap<String, Bob>,
}

#[derive(Clone, Copy)]
struct Bob {
    y: f32,
    at: f64,
    vy: f32,
    sunk: f32,
}

/// A FLOATING THING'S BOB makes its own small rings: the slowest sinking,
/// m/s, that does, and how near the waterline its middle must be (in its
/// half heights). A crate settling after it is thrown in bobs at a few
/// tenths of a metre a second for a few seconds.
const BOB_SPEED: f32 = 0.06;
const BOB_REACH: f32 = 1.3;

/// A splash older than this when first heard of is not raised: its drops
/// would appear in mid-air. The server tells of one for half a second.
const TOLD_LATE: f32 = 1.0;

impl SplashWatch {
    /// This frame's splashes, at `now` (the water's clock), `dt` after the
    /// last call. `surface` is the still water's height at a world x, z (none
    /// off the water), `ground` the height of what lies under a point;
    /// `sunlit` how much sun reaches a point. `feet` is the
    /// floor under the head and the way the player faces, `hands` the grips,
    /// `things` each moving object's id, centre, half size and rotation.
    #[allow(clippy::too_many_arguments)]
    pub fn step<'a>(
        &mut self,
        now: f64,
        dt: f32,
        surface: &dyn Fn(f32, f32) -> Option<f32>,
        ground: &dyn Fn(Vec3) -> Option<f32>,
        sunlit: &dyn Fn(Vec3) -> f32,
        feet: Option<(Vec3, Vec3)>,
        hands: [Option<Vec3>; 2],
        things: impl Iterator<Item = (&'a str, Vec3, Vec3, Quat)>,
    ) -> Vec<Splash> {
        let mut out = Vec::new();
        let dt = dt.max(1e-3);
        let mut emit = |position: Vec3, depth: Option<f32>, velocity: Vec3, size: f32, mass: f32, held: f32, count: &mut u64| {
            *count += 1;
            out.push(Splash {
                position,
                born: now,
                velocity,
                size,
                mass,
                sunlit: sunlit(position + Vec3::Y * 0.05),
                // Off anything known, as deep as the open sea.
                depth: depth.or_else(|| ground(position).map(|g| position.y - g)).unwrap_or(30.0).max(0.0),
                // The renderer finds how still the water was (`effects::strike`).
                calm: 1.0,
                held,
                seed: count.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (now.to_bits()),
            });
        };

        // FEET.
        match feet {
            Some((floor, facing)) => {
                let moved = self.feet.map_or(0.0, |f| Vec3::new(floor.x - f.x, 0.0, floor.z - f.z).length());
                let depth = surface(floor.x, floor.z).map_or(0.0, |s| s - floor.y);
                if moved > JUMP || depth <= WET {
                    self.walked = 0.0;
                } else if moved / dt >= WADE_SPEED {
                    self.walked += moved;
                    if self.walked >= STRIDE {
                        self.walked -= STRIDE;
                        self.left_foot = !self.left_foot;
                        let ahead = Vec3::new(facing.x, 0.0, facing.z).normalize_or(Vec3::Z);
                        let side = ahead.cross(Vec3::Y) * if self.left_foot { -0.11 } else { 0.11 };
                        let at = floor + side + ahead * 0.25;
                        let water = surface(at.x, at.z).unwrap_or(floor.y + depth);
                        // A foot swings at about twice the walk; a deep step
                        // pushes water aside more than it steps into it.
                        let walk = (moved / dt).min(3.0);
                        let kick = 1.0 - 0.6 * smoothstep(0.25, 0.7, depth);
                        let swing = 2.0 * walk * kick;
                        let (down, along) = FOOT_ENTRY_DEG.to_radians().sin_cos();
                        emit(Vec3::new(at.x, water, at.z), Some(water - floor.y), ahead * (swing * along) - Vec3::Y * (swing * down), FOOT_RADIUS, FOOT_MASS, 1.0, &mut self.count);
                    }
                }
                self.feet = Some(floor);
            }
            None => {
                self.feet = None;
                self.walked = 0.0;
            }
        }

        // HANDS.
        for (i, hand) in hands.iter().enumerate() {
            let Some(p) = *hand else {
                self.hands[i] = None;
                continue;
            };
            if let Some((was, _)) = self.hands[i] {
                if was.distance(p) < JUMP && now - self.hand_last[i] > COOLDOWN {
                    if let Some(s) = surface(p.x, p.z) {
                        let v = ((p - was) / dt).clamp_length_max(8.0);
                        let into = was.y > s && p.y <= s && -v.y >= SLAP_IN;
                        let out_of = was.y <= s && p.y > s && v.y >= PULL_OUT;
                        if into || out_of {
                            self.hand_last[i] = now;
                            // A hand pulled out drags water up after it and
                            // lets it fall back: as one going in half as fast.
                            let velocity = if into { v } else { Vec3::new(v.x, -v.y, v.z) * 0.5 };
                            emit(Vec3::new(p.x, s, p.z), None, velocity, HAND_RADIUS, HAND_MASS, 1.0, &mut self.count);
                        }
                    }
                }
            }
            self.hands[i] = Some((p, now));
        }

        // THINGS. A networked object moves when an update arrives, not every
        // frame: its speed is the move over the time since it last moved.
        let mut seen: Vec<String> = Vec::new();
        for (id, centre, half, rotation) in things {
            seen.push(id.to_string());
            let velocity = match self.things.get(id).copied() {
                Some((was, _, velocity)) if was == centre => velocity,
                Some((was, moved_at, _)) => {
                    let since = (now - moved_at).max(dt as f64) as f32;
                    let velocity = if was.distance(centre) < JUMP * 4.0 { ((centre - was) / since).clamp_length_max(60.0) } else { Vec3::ZERO };
                    self.things.insert(id.to_string(), (centre, now, velocity));
                    velocity
                }
                None => {
                    self.things.insert(id.to_string(), (centre, now, Vec3::ZERO));
                    Vec3::ZERO
                }
            };
            let mass = space_soup_engine::rigid_physics::default_mass(half);
            if let Some(e) = self.water.see(id, centre, world_half_extents(half, rotation), velocity, mass, surface(centre.x, centre.z)) {
                emit(e.position, None, e.velocity, e.radius.clamp(0.01, 2.0), e.mass, 0.0, &mut self.count);
            }
        }
        self.things.retain(|id, _| seen.contains(id));
        self.water.keep(|id| seen.iter().any(|s| s == id));
        out
    }

    /// THINGS FLOATING, BOBBING: each time a thing at the waterline turns
    /// from sinking to rising -- the bottom of a bob -- its push into the
    /// water is raised as a gentle splash (`held`: no jet, too slow for a
    /// crown or drops), whose rings the renderer works out as for any other
    /// (`effects::splash_rings`). Every object, whoever tells of entries:
    /// the server tells of going in, not of bobbing. `things` as in
    /// [`Self::step`].
    pub fn bob<'a>(&mut self, now: f64, surface: &dyn Fn(f32, f32) -> Option<f32>, sunlit: &dyn Fn(Vec3) -> f32, things: impl Iterator<Item = (&'a str, Vec3, Vec3, Quat)>) -> Vec<Splash> {
        let mut out = Vec::new();
        let mut seen: Vec<String> = Vec::new();
        for (id, centre, half, rotation) in things {
            seen.push(id.to_string());
            let extent = world_half_extents(half, rotation);
            let Some(level) = surface(centre.x, centre.z) else {
                self.bobbing.remove(id);
                continue;
            };
            let afloat = (centre.y - level).abs() <= BOB_REACH * extent.y.max(0.01);
            let Some(was) = self.bobbing.get(id).copied() else {
                self.bobbing.insert(id.to_string(), Bob { y: centre.y, at: now, vy: 0.0, sunk: 0.0 });
                continue;
            };
            // A networked thing moves when an update arrives: its speed is
            // the move over the time since it last moved.
            if centre.y == was.y {
                continue;
            }
            let vy = ((centre.y - was.y) / (now - was.at).max(1e-3) as f32).clamp(-20.0, 20.0);
            let mut next = Bob { y: centre.y, at: now, vy, sunk: was.sunk.max(-vy) };
            if afloat && was.vy < 0.0 && vy >= 0.0 && next.sunk >= BOB_SPEED {
                self.count += 1;
                let position = Vec3::new(centre.x, level, centre.z);
                out.push(Splash {
                    sunlit: sunlit(position + Vec3::Y * 0.05),
                    depth: 30.0,
                    calm: 1.0,
                    held: 1.0,
                    seed: self.count.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0xb0b,
                    ..Splash::new(position, now, Vec3::new(0.0, -next.sunk, 0.0), (extent.x * extent.z).sqrt().clamp(0.01, 2.0), space_soup_engine::rigid_physics::default_mass(half))
                });
                next.sunk = 0.0;
            }
            if !afloat {
                next.sunk = 0.0;
            }
            self.bobbing.insert(id.to_string(), next);
        }
        self.bobbing.retain(|id, _| seen.contains(id));
        out
    }

    /// THINGS THE SERVER SAW GO IN: the entries in its latest snapshot
    /// (`WireWorld::water_entries`; none while there is no snapshot). Each is
    /// in every snapshot for half a second, under its own number: a splash is
    /// raised for each number not in the snapshot before, born when the thing
    /// went in -- `age` before `now` -- not when word of it came. `ground` and
    /// `sunlit` as in [`Self::step`].
    pub fn told(&mut self, now: f64, entries: &[WireWaterEntry], ground: &dyn Fn(Vec3) -> Option<f32>, sunlit: &dyn Fn(Vec3) -> f32) -> Vec<Splash> {
        let mut out = Vec::new();
        for e in entries {
            let sound = e.position.iter().chain(&e.velocity).chain([&e.radius, &e.mass, &e.age]).all(|v| v.is_finite());
            if self.told.contains(&e.serial) || !sound || !(0.0..TOLD_LATE).contains(&e.age) {
                continue;
            }
            let position = Vec3::from(e.position);
            self.count += 1;
            out.push(Splash {
                position,
                born: now - e.age as f64,
                velocity: Vec3::from(e.velocity).clamp_length_max(60.0),
                size: e.radius.clamp(0.01, 2.0),
                mass: e.mass.max(1e-3),
                sunlit: sunlit(position + Vec3::Y * 0.05),
                depth: ground(position).map(|g| position.y - g).unwrap_or(30.0).max(0.0),
                calm: 1.0,
                held: 0.0,
                // By its number, so every headset draws the same splash.
                seed: e.serial.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x5eed,
            });
        }
        // Only what the server still tells of: a number is not given twice
        // while it does, and a server started again starts its numbers again.
        self.told.clear();
        self.told.extend(entries.iter().map(|e| e.serial));
        out
    }
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sea at y = 0 for x > 0, dry land at x <= 0.
    fn sea(x: f32, _z: f32) -> Option<f32> {
        (x > 0.0).then_some(0.0)
    }

    fn sun(_: Vec3) -> f32 {
        1.0
    }

    fn bed(_: Vec3) -> Option<f32> {
        Some(-0.4)
    }

    fn none() -> std::iter::Empty<(&'static str, Vec3, Vec3, Quat)> {
        std::iter::empty()
    }

    #[test]
    fn wading_splashes_once_a_stride_from_alternate_feet_and_walking_on_land_does_not() {
        let mut w = SplashWatch::default();
        let dt = 1.0 / 72.0;
        let mut steps = Vec::new();
        // Walk 6.5 m along z at 1.3 m/s, 0.3 m deep.
        for k in 0..360 {
            let floor = Vec3::new(5.0, -0.3, k as f32 * 1.3 * dt);
            steps.extend(w.step(k as f64 * dt as f64, dt, &sea, &bed, &sun, Some((floor, Vec3::Z)), [None, None], none()));
        }
        assert!((9..=11).contains(&steps.len()), "{} splashes over ten strides", steps.len());
        assert!(steps.windows(2).all(|p| (p[0].position.x - 5.0).signum() != (p[1].position.x - 5.0).signum()), "alternate feet");
        assert!(steps.iter().all(|s| s.position.y == 0.0 && s.velocity.length() > 1.5), "on the surface, kicked");
        // Swung forward and down at a shallow angle, by a foot.
        assert!(steps.iter().all(|s| s.velocity.z > 2.0 * -s.velocity.y && s.velocity.y < -0.5 && s.size == FOOT_RADIUS && s.mass == FOOT_MASS), "{:?}", steps[0]);
        // On land, nothing.
        let mut w = SplashWatch::default();
        let dry: usize = (0..360)
            .map(|k| w.step(k as f64 * dt as f64, dt, &sea, &bed, &sun, Some((Vec3::new(-5.0, 0.0, k as f32 * 1.3 * dt), Vec3::Z)), [None, None], none()).len())
            .sum();
        assert_eq!(dry, 0);
    }

    /// A crate bobbing at the waterline raises a gentle splash at the bottom
    /// of each bob -- rings, no jet -- and one resting, or high and dry,
    /// none.
    #[test]
    fn a_floating_thing_bobbing_makes_its_own_rings() {
        let mut w = SplashWatch::default();
        let half = Vec3::splat(0.2);
        let mut bobs = Vec::new();
        // Updates at 20 Hz: a bob of 4 cm at 1.2 Hz, dying away over 4 s.
        for k in 0..200 {
            let t = k as f64 * 0.05;
            let y = -0.05 + 0.04 * (-(t as f32) / 4.0).exp() * (std::f32::consts::TAU * 1.2 * t as f32).cos();
            bobs.extend(w.bob(t, &sea, &sun, std::iter::once(("crate", Vec3::new(5.0, y, 0.0), half, Quat::IDENTITY))));
        }
        assert!((3..=12).contains(&bobs.len()), "{} bobs", bobs.len());
        assert!(bobs.iter().all(|s| s.held == 1.0 && s.velocity.y < -BOB_SPEED && s.velocity.y > -0.5 && s.position.y == 0.0), "{:?}", bobs[0]);
        assert!(bobs.windows(2).all(|p| p[1].born - p[0].born > 0.6), "one a bob");
        // Its rings, and nothing thrown.
        let shape = space_soup::renderer::effects::splash_shape(&bobs[0]);
        assert!(shape.ring_height > 1e-4 && shape.jet_height == 0.0 && shape.drops == 0 && shape.petals == 0, "{shape:?}");
        // Resting, or on land: nothing.
        let mut w = SplashWatch::default();
        let still: usize = (0..100).map(|k| w.bob(k as f64 * 0.05, &sea, &sun, std::iter::once(("crate", Vec3::new(5.0, -0.05, 0.0), half, Quat::IDENTITY))).len()).sum();
        let mut w = SplashWatch::default();
        let dry: usize = (0..100)
            .map(|k| w.bob(k as f64 * 0.05, &sea, &sun, std::iter::once(("crate", Vec3::new(-5.0, 0.04 * (k as f32 * 0.4).cos(), 0.0), half, Quat::IDENTITY))).len())
            .sum();
        assert_eq!((still, dry), (0, 0));
    }

    #[test]
    fn standing_still_or_teleporting_through_water_does_not_splash() {
        let mut w = SplashWatch::default();
        let dt = 1.0 / 72.0;
        let mut n = 0;
        for k in 0..200 {
            n += w.step(k as f64 * dt as f64, dt, &sea, &bed, &sun, Some((Vec3::new(5.0, -0.3, 0.0), Vec3::Z)), [None, None], none()).len();
        }
        for k in 0..20 {
            let floor = Vec3::new(5.0, -0.3, k as f32 * 3.0);
            n += w.step(k as f64 * dt as f64, dt, &sea, &bed, &sun, Some((floor, Vec3::Z)), [None, None], none()).len();
        }
        assert_eq!(n, 0);
    }

    #[test]
    fn a_hand_slapped_down_splashes_and_one_lowered_gently_does_not() {
        let dt = 1.0 / 72.0;
        let run = |speed: f32| {
            let mut w = SplashWatch::default();
            let mut out = Vec::new();
            for k in 0..60 {
                let hand = Vec3::new(3.0, 0.3 - k as f32 * speed * dt, 1.0);
                out.extend(w.step(k as f64 * dt as f64, dt, &sea, &bed, &sun, None, [Some(hand), None], none()));
            }
            out
        };
        let slap = run(2.0);
        assert_eq!(slap.len(), 1);
        assert!((slap[0].velocity.y + 2.0).abs() < 0.1 && slap[0].position.y == 0.0);
        assert!(run(0.2).is_empty());
    }

    #[test]
    fn a_thing_falling_in_splashes_by_its_speed_and_size() {
        let dt = 1.0 / 72.0;
        let mut w = SplashWatch::default();
        let mut out = Vec::new();
        // Dropped from 2 m, updated every third frame as a network update is.
        for k in 0..60 {
            let t = (k / 3 * 3) as f32 * dt;
            let y = (2.0 - 0.5 * 9.81 * t * t).max(-0.5);
            let things = std::iter::once(("crate", Vec3::new(4.0, y + 0.2, 0.0), Vec3::splat(0.2), Quat::IDENTITY));
            out.extend(w.step(k as f64 * dt as f64, dt, &sea, &bed, &sun, None, [None, None], things));
        }
        assert_eq!(out.len(), 1, "{out:?}");
        // As fast as it fell, as wide as a disc its footprint's size, as heavy
        // as physics makes a box that size.
        assert!(-out[0].velocity.y > 4.0 && (out[0].size - 0.2257).abs() < 1e-3 && (out[0].mass - 32.0).abs() < 1e-3, "{:?}", out[0]);
    }

    fn entry(serial: u64, age: f32) -> WireWaterEntry {
        WireWaterEntry { serial, id: "crate".into(), position: [4.0, 0.0, 1.0], velocity: [0.5, -5.6, 0.0], radius: 0.23, mass: 32.0, age }
    }

    /// The server tells of a going-in in thirty snapshots; the headset reads
    /// one a frame, some twice and some never. One splash, with the server's
    /// numbers, born when the thing went in.
    #[test]
    fn what_the_server_tells_of_splashes_once_as_it_was_and_when_it_was() {
        let mut w = SplashWatch::default();
        let dt = 1.0 / 72.0;
        let mut out = Vec::new();
        let mut latest: Vec<WireWaterEntry> = Vec::new();
        for k in 0..72 {
            let now = 10.0 + k as f64 * dt;
            // Snapshots at 60 a second from a going-in at 10.1 s; every
            // seventh frame none has come, and the one before is read again.
            let age = (now - 10.1) as f32;
            if k % 7 != 3 {
                latest = if (0.0..0.5).contains(&age) { vec![entry(12, (age * 60.0).floor() / 60.0)] } else { vec![] };
            }
            out.extend(w.told(now, &latest, &bed, &sun));
        }
        assert_eq!(out.len(), 1, "{out:?}");
        let s = &out[0];
        assert!((s.born - 10.1).abs() < 0.02, "born at {}", s.born);
        assert_eq!((s.position, s.velocity, s.size, s.mass, s.held), (Vec3::new(4.0, 0.0, 1.0), Vec3::new(0.5, -5.6, 0.0), 0.23, 32.0, 0.0));
        assert!((s.depth - 0.4).abs() < 1e-6 && s.sunlit == 1.0);
        // The same thing going in again later is another number: another splash.
        assert_eq!(w.told(12.0, &[entry(13, 0.0)], &bed, &sun).len(), 1);
        // Two at once, and one heard of too late to draw from its start.
        let mut w = SplashWatch::default();
        assert_eq!(w.told(1.0, &[entry(0, 0.1), entry(1, 0.0), entry(2, 1.5)], &bed, &sun).len(), 2);
        // Every headset draws the same splash for the same number.
        let (a, b) = (SplashWatch::default().told(1.0, &[entry(5, 0.0)], &bed, &sun), SplashWatch::default().told(7.0, &[entry(5, 0.2)], &bed, &sun));
        assert_eq!(a[0].seed, b[0].seed);
        // Nonsense off the wire is not drawn.
        let mut bad = entry(9, 0.0);
        bad.velocity[1] = f32::NAN;
        assert!(SplashWatch::default().told(1.0, &[bad], &bed, &sun).is_empty());
    }

    /// A server started again numbers its entries from nothing again: the
    /// first of them must not be taken for one already drawn.
    #[test]
    fn a_number_is_only_remembered_while_the_server_tells_of_it() {
        let mut w = SplashWatch::default();
        assert_eq!(w.told(1.0, &[entry(0, 0.0)], &bed, &sun).len(), 1);
        assert_eq!(w.told(1.1, &[entry(0, 0.1)], &bed, &sun).len(), 0);
        // The connection drops: no snapshot, no entries.
        assert_eq!(w.told(1.2, &[], &bed, &sun).len(), 0);
        assert_eq!(w.told(4.0, &[entry(0, 0.0)], &bed, &sun).len(), 1);
    }

    /// Floating at the waterline, bobbing: it went in once, and that was all.
    #[test]
    fn a_thing_bobbing_at_the_waterline_does_not_splash_again() {
        let dt = 1.0 / 72.0;
        let mut w = SplashWatch::default();
        let mut n = 0;
        for k in 0..720 {
            let t = k as f32 * dt;
            // Dropped in over the first third of a second, then bobbing 8 cm
            // either side of afloat at 1.5 Hz -- fast enough, each time down,
            // to have splashed by the old rule.
            let y = if t < 0.37 { 0.6 - 0.5 * 9.81 * t * t } else { 0.08 * (t * 9.4).sin() };
            let things = std::iter::once(("buoy", Vec3::new(4.0, y + 0.2, 0.0), Vec3::splat(0.2), Quat::IDENTITY));
            n += w.step(k as f64 * dt as f64, dt, &sea, &bed, &sun, None, [None, None], things).len();
        }
        assert_eq!(n, 1);
    }
}
