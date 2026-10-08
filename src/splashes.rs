//! SPLASHES: where something strikes the water, seen by the client and handed
//! to the renderer (`XrRenderer::add_splash`), which throws the drops and
//! spray and spreads the rings (`space_soup::renderer::effects::Splash`).
//!
//! Three things strike water here:
//! - FEET, wading: a splash a stride, from alternate feet, as hard as the
//!   walk and strongest in shallows -- deep water swallows a step's kick.
//! - HANDS, slapped down through the surface, and pulled out of it fast.
//! - THINGS, a level's moving objects, falling in: as hard as they fall, as
//!   big as they are.
//!
//! Everything is in the WORLD's frame, on the water's clock. Only the still
//! surface is known here (the footprints' height); the waves move it a hand
//! or so, which a splash's look does not need.
use glam::Vec3;
use space_soup::renderer::effects::Splash;
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
/// The slowest falling thing that splashes, m/s.
const DROP_IN: f32 = 0.6;
/// No source splashes twice within this, seconds.
const COOLDOWN: f64 = 0.15;

#[derive(Default)]
pub struct SplashWatch {
    feet: Option<Vec3>,
    walked: f32,
    left_foot: bool,
    hands: [Option<(Vec3, f64)>; 2],
    hand_last: [f64; 2],
    /// Each object's bottom when it last moved, and when.
    things: HashMap<String, (Vec3, f64, f64)>,
    count: u64,
}

impl SplashWatch {
    /// This frame's splashes, at `now` (the water's clock), `dt` after the
    /// last call. `surface` is the still water's height at a world x, z (none
    /// off the water), `ground` the height of what lies under a point;
    /// `sunlit` how much sun reaches a point. `feet` is the
    /// floor under the head and the way the player faces, `hands` the grips,
    /// `things` each moving object's id, centre and half size.
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
        things: impl Iterator<Item = (&'a str, Vec3, Vec3)>,
    ) -> Vec<Splash> {
        let mut out = Vec::new();
        let dt = dt.max(1e-3);
        let mut emit = |position: Vec3, depth: Option<f32>, speed: f32, size: f32, count: &mut u64| {
            *count += 1;
            out.push(Splash {
                position,
                born: now,
                speed,
                size,
                sunlit: sunlit(position + Vec3::Y * 0.05),
                // Off anything known, as deep as the open sea.
                depth: depth.or_else(|| ground(position).map(|g| position.y - g)).unwrap_or(30.0).max(0.0),
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
                        // pushes water aside more than it throws it.
                        let walk = (moved / dt).min(3.0);
                        let kick = 1.0 - 0.6 * smoothstep(0.25, 0.7, depth);
                        emit(Vec3::new(at.x, water, at.z), Some(water - floor.y), 2.0 * walk * kick, 0.07, &mut self.count);
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
                        let v = (p - was) / dt;
                        let into = was.y > s && p.y <= s && -v.y >= SLAP_IN;
                        let out_of = was.y <= s && p.y > s && v.y >= PULL_OUT;
                        if into || out_of {
                            self.hand_last[i] = now;
                            let speed = if into { v.length() } else { 0.5 * v.length() };
                            emit(Vec3::new(p.x, s, p.z), None, speed.min(8.0), 0.06, &mut self.count);
                        }
                    }
                }
            }
            self.hands[i] = Some((p, now));
        }

        // THINGS. A networked object moves when an update arrives, not every
        // frame: its speed is the move over the time since it last moved.
        let mut seen: Vec<String> = Vec::new();
        for (id, centre, half) in things {
            let bottom = centre - Vec3::Y * half.y.abs();
            seen.push(id.to_string());
            match self.things.get(id).copied() {
                Some((was, moved_at, last)) if was != bottom => {
                    let since = (now - moved_at).max(dt as f64) as f32;
                    let v = (bottom - was) / since;
                    if was.distance(bottom) < JUMP * 4.0 && now - last > COOLDOWN {
                        if let Some(s) = surface(bottom.x, bottom.z) {
                            if was.y > s && bottom.y <= s && -v.y >= DROP_IN {
                                let size = half.x.abs().max(half.z.abs()).clamp(0.03, 1.5);
                                emit(Vec3::new(bottom.x, s, bottom.z), None, v.length().min(15.0), size, &mut self.count);
                                self.things.insert(id.to_string(), (bottom, now, now));
                                continue;
                            }
                        }
                    }
                    self.things.insert(id.to_string(), (bottom, now, last));
                }
                Some(_) => {}
                None => {
                    self.things.insert(id.to_string(), (bottom, now, f64::MIN));
                }
            }
        }
        self.things.retain(|id, _| seen.contains(id));
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

    fn none() -> std::iter::Empty<(&'static str, Vec3, Vec3)> {
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
        assert!(steps.iter().all(|s| s.position.y == 0.0 && s.speed > 1.5), "on the surface, kicked");
        // On land, nothing.
        let mut w = SplashWatch::default();
        let dry: usize = (0..360)
            .map(|k| w.step(k as f64 * dt as f64, dt, &sea, &bed, &sun, Some((Vec3::new(-5.0, 0.0, k as f32 * 1.3 * dt), Vec3::Z)), [None, None], none()).len())
            .sum();
        assert_eq!(dry, 0);
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
        assert!((slap[0].speed - 2.0).abs() < 0.1 && slap[0].position.y == 0.0);
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
            let things = std::iter::once(("crate", Vec3::new(4.0, y + 0.2, 0.0), Vec3::splat(0.2)));
            out.extend(w.step(k as f64 * dt as f64, dt, &sea, &bed, &sun, None, [None, None], things));
        }
        assert_eq!(out.len(), 1, "{out:?}");
        assert!(out[0].speed > 4.0 && (out[0].size - 0.2).abs() < 1e-6, "{:?}", out[0]);
    }
}
