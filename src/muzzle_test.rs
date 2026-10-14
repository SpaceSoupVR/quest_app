//! THE MUZZLE TEST: firing the firearm effects (`game/effects`) from the right
//! hand, to look at them in the headset before any weapon's script raises
//! them. Behind the lever `muzzle_test`.
//!
//! - **A** (right hand): a tap is one shot; held past [`AUTO_AFTER`] it is
//!   automatic fire at the selected effect's cyclic rate.
//! - **B** (right hand): the next effect (rifle with flash hider -> bare 9 mm
//!   -> suppressed -> ...).
//!
//! A and B still reach a held object's script (`part_pull`); nothing in
//! test_room's scripts answers them, and the grab and door controls are the
//! triggers and the grips.
//!
//! The muzzle is the right hand's aim pose, [`MUZZLE_AHEAD`] along it; the
//! ejection port a little back and to the right. Each shot carries the
//! weapon's `heat` -- shots in the last two seconds over [`HEAT_SHOTS`] -- as
//! the effects' first input, which is what thickens the haze of a string
//! and starts the barrel's thread.
//!
//! The decisions are here, apart from the app's loop, so they are tested on
//! the desk; and the offline harness's `BURST=` is read here too.

use glam::{Quat, Vec3};
use space_soup::renderer::effects::stack::{BurstCount, BurstSet, BurstStats, EffectBurst, EffectId, EffectLibrary};

/// The effects B steps through, and each one's cyclic rate, shots a second.
pub(crate) const EFFECTS: [(&str, f32); 3] = [("muzzle_556_a2", 12.5), ("muzzle_9mm_bare", 6.0), ("muzzle_suppressed", 12.5)];
/// How long A is held before a tap becomes automatic fire, seconds.
const AUTO_AFTER: f32 = 0.22;
/// Shots in two seconds at which a weapon is as hot as the effects know.
const HEAT_SHOTS: f32 = 12.0;
const HEAT_SECONDS: f64 = 2.0;
/// Where the muzzle is along the hand's aim, metres: a pistol's barrel.
pub(crate) const MUZZLE_AHEAD: f32 = 0.28;
/// The source every burst of the test names: one weapon, the right hand's.
const SOURCE: u64 = 0x6d75_7a7a_6c65;

#[derive(Default)]
pub(crate) struct MuzzleTest {
    selected: usize,
    prev_a: bool,
    prev_b: bool,
    /// The test's own clock, seconds.
    now: f64,
    /// When the next shot of a held string is due.
    next_auto: Option<f64>,
    /// When each recent shot was fired: the weapon's heat.
    shots: Vec<f64>,
    fired: u64,
    prev_muzzle: Option<Vec3>,
}

/// One shot to raise: how long ago within this frame it was due, and how hot
/// the weapon was.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Shot {
    pub age: f32,
    pub heat: f32,
    pub seed: u64,
}

impl MuzzleTest {
    /// The effect selected.
    pub(crate) fn selected(&self) -> &'static str {
        EFFECTS[self.selected].0
    }

    /// One frame of the buttons: the shots due in it. A press fires at once;
    /// a hold fires again every cycle from [`AUTO_AFTER`]; B (on its press)
    /// selects the next effect and ends a string.
    pub(crate) fn buttons(&mut self, a: bool, b: bool, dt: f32) -> Vec<Shot> {
        self.now += dt.clamp(0.0, 0.25) as f64;
        let now = self.now;
        if b && !self.prev_b {
            self.selected = (self.selected + 1) % EFFECTS.len();
            self.next_auto = None;
            log::info!("MUZZLE TEST -> {} by B", self.selected());
        }
        let mut due: Vec<f64> = Vec::new();
        if a && !self.prev_a {
            due.push(now);
            self.next_auto = Some(now + AUTO_AFTER as f64);
        } else if a {
            let cycle = 1.0 / EFFECTS[self.selected].1 as f64;
            while let Some(t) = self.next_auto.filter(|t| *t <= now && due.len() < 4) {
                due.push(t);
                self.next_auto = Some(t + cycle);
            }
        } else {
            self.next_auto = None;
        }
        (self.prev_a, self.prev_b) = (a, b);
        self.shots.retain(|t| now - t < HEAT_SECONDS);
        due.into_iter()
            .map(|t| {
                let heat = (self.shots.len() as f32 / HEAT_SHOTS).min(1.0);
                self.shots.push(t);
                self.fired += 1;
                Shot { age: (now - t) as f32, heat, seed: 0x9e37_79b9_7f4a_7c15u64.wrapping_mul(self.fired) }
            })
            .collect()
    }

    /// One frame: the buttons, and the bursts their shots raise from the
    /// right hand's aim pose (`aim_position`, `aim_rotation`, in the WORLD).
    /// `effect` resolves a name in the renderer's library.
    pub(crate) fn step(&mut self, a: bool, b: bool, dt: f32, aim_position: Vec3, aim_rotation: Quat, effect: impl Fn(&str) -> Option<EffectId>) -> Vec<EffectBurst> {
        let muzzle = aim_position + aim_rotation * Vec3::NEG_Z * MUZZLE_AHEAD;
        // How the muzzle is moving: its flash rides it in the frame SpaceWarp
        // makes up.
        let velocity = match self.prev_muzzle.replace(muzzle) {
            Some(was) if dt > 1e-4 && was.distance(muzzle) < 1.0 => (muzzle - was) / dt,
            _ => Vec3::ZERO,
        };
        let shots = self.buttons(a, b, dt);
        let Some(id) = effect(self.selected()) else {
            if !shots.is_empty() {
                log::warn!("MUZZLE TEST: no effect `{}` in game/effects", self.selected());
            }
            return Vec::new();
        };
        if shots.iter().any(|s| s.heat == 0.0) {
            log::info!("MUZZLE TEST: {} fired", self.selected());
        }
        shots.iter().map(|s| burst(id, aim_position, aim_rotation, velocity, *s)).collect()
    }
}

/// A shot's burst from an aim pose in the world: the muzzle ahead of the
/// hand, the port back and to the right of it, throwing right and up.
pub(crate) fn burst(effect: EffectId, aim_position: Vec3, aim_rotation: Quat, velocity: Vec3, shot: Shot) -> EffectBurst {
    let (forward, up, right) = (aim_rotation * Vec3::NEG_Z, aim_rotation * Vec3::Y, aim_rotation * Vec3::X);
    EffectBurst {
        age: shot.age,
        up,
        velocity,
        port: Some((aim_position + forward * 0.06 + right * 0.035 + up * 0.02, (right + up * 0.35 - forward * 0.2).normalize())),
        inputs: [shot.heat, 0.0, 0.0, 0.0],
        ..EffectBurst::new(effect, aim_position + forward * MUZZLE_AHEAD, forward, SOURCE, shot.seed)
    }
}

/// What the bursts came to, for the log.
pub(crate) fn stats_line(s: &BurstStats, c: &BurstCount) -> String {
    format!(
        "BURSTS live {} seen {} culled {} | drawn over {} screened {} shed {} | fill {:.3} of {:.3} wanted, thinned {} | lights {} gated {} | merged {}",
        s.live, s.seen, s.culled, c.over, c.screened, c.shed, s.fill, s.fill_wanted, s.thinned, s.lights, s.lights_gated, s.merged
    )
}

/// THE OFFLINE HARNESS'S SHOTS: `BURST=name,x,y,z,dx,dy,dz,age[,heat[,seed]]`
/// -- an effect of `game/effects`, its muzzle and the way it points in the
/// world, seconds since the shot (0: the frame of the flash), the weapon's
/// heat 0-1 -- several with `;` between them, oldest first, all one weapon's.
/// `None` without `BURST`. The set is ready for `BurstSet::lights` and
/// `BurstSet::frame`.
#[cfg_attr(target_os = "android", allow(dead_code))] // the offline harness's, which the headset build leaves out
pub(crate) fn offline_bursts(game: &std::path::Path) -> Option<(EffectLibrary, BurstSet)> {
    let spec = std::env::var("BURST").ok()?;
    let (lib, errors) = EffectLibrary::from_dir(&game.join("effects"));
    for e in &errors {
        eprintln!("offline frame: {e}");
    }
    let mut set = BurstSet::default();
    for (k, one) in spec.split(';').enumerate() {
        let mut parts = one.split(',').map(str::trim);
        let name = parts.next().unwrap_or("");
        let n: Vec<f32> = parts.filter_map(|x| x.parse().ok()).collect();
        let Some(id) = lib.id(name).filter(|_| n.len() >= 7) else {
            eprintln!("offline frame: BURST wants name,x,y,z,dx,dy,dz,age[,heat[,seed]] with a name from game/effects; got {one}");
            continue;
        };
        let forward = Vec3::new(n[3], n[4], n[5]).try_normalize().unwrap_or(Vec3::NEG_Z);
        let rotation = Quat::from_rotation_arc(Vec3::NEG_Z, forward);
        let shot = Shot { age: n[6].max(0.0), heat: n.get(7).copied().unwrap_or(0.0), seed: n.get(8).map_or(7 + 0x9e37 * k as u64, |s| *s as u64) };
        set.add(&lib, burst(id, Vec3::new(n[0], n[1], n[2]) - forward * MUZZLE_AHEAD, rotation, Vec3::ZERO, shot));
    }
    // A shot already in the past has had its frame: no flash, no light.
    for l in &mut set.live {
        if l.burst.age > 1e-4 {
            l.frames = 1 + (l.burst.age * 36.0) as u32;
        }
    }
    Some((lib, set))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tap is one shot; a hold becomes automatic fire at the effect's
    /// rate, however long the frames are; B steps the effect and wraps.
    #[test]
    fn a_tap_is_one_shot_and_a_hold_is_automatic_fire() {
        let mut m = MuzzleTest::default();
        let dt = 1.0 / 36.0;
        // A tap of three frames.
        let tapped: usize = [true, true, true, false, false].iter().map(|a| m.buttons(*a, false, dt).len()).sum();
        assert_eq!(tapped, 1);
        // Held two seconds: the first at once, then 12.5 a second from 0.22 s.
        let mut held = Vec::new();
        for _ in 0..72 {
            held.extend(m.buttons(true, false, dt));
        }
        let expected = 1.0 + ((2.0 - AUTO_AFTER) * 12.5).floor();
        assert!((held.len() as f32 - expected).abs() <= 1.0, "{} shots in 2 s; about {expected}", held.len());
        assert!(held.iter().all(|s| s.age >= 0.0 && s.age < dt + 1e-4), "each within its frame");
        assert_eq!(held[0].heat, 1.0 / HEAT_SHOTS, "the tap before it still warms the barrel");
        assert!(held.last().unwrap().heat == 1.0, "hot by the end of a string");
        assert!(m.buttons(false, false, dt).is_empty());
        // The same string at 12 frames a second: as many shots.
        let mut slow = MuzzleTest::default();
        let n: usize = (0..24).map(|_| slow.buttons(true, false, 1.0 / 12.0).len()).sum();
        assert!((n as f32 - expected).abs() <= 2.0, "{n} at 12 fps");
        // Cold again after a pause; no two shots share a seed.
        for _ in 0..80 {
            m.buttons(false, false, dt);
        }
        assert_eq!(m.buttons(true, false, dt)[0].heat, 0.0);
        let mut seeds: Vec<u64> = held.iter().map(|s| s.seed).collect();
        seeds.dedup();
        assert_eq!(seeds.len(), held.len());
        // B: the next effect, on its press only, round to the first.
        assert_eq!(m.selected(), "muzzle_556_a2");
        m.buttons(false, true, dt);
        m.buttons(false, true, dt);
        assert_eq!(m.selected(), "muzzle_9mm_bare");
        m.buttons(false, false, dt);
        m.buttons(false, true, dt);
        m.buttons(false, false, dt);
        m.buttons(false, true, dt);
        assert_eq!(m.selected(), "muzzle_556_a2");
    }

    /// Every effect the test steps through is in `game/effects`, and a shot
    /// leaves the muzzle ahead of the hand along its aim, moving with it.
    #[test]
    fn the_tests_effects_ship_and_a_shot_leaves_the_muzzle() {
        let game = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../game");
        let (lib, errors) = EffectLibrary::from_dir(&game.join("effects"));
        assert!(errors.is_empty(), "{errors:?}");
        for (name, rate) in EFFECTS {
            assert!(lib.id(name).is_some() && rate > 0.0, "{name}");
        }
        let mut m = MuzzleTest::default();
        let aim = Quat::from_rotation_y(0.4);
        let hand = Vec3::new(1.0, 1.3, -2.0);
        assert!(m.step(false, false, 0.03, hand, aim, |n| lib.id(n)).is_empty());
        let bursts = m.step(true, false, 0.03, hand + Vec3::X * 0.03, aim, |n| lib.id(n));
        assert_eq!(bursts.len(), 1);
        let b = bursts[0];
        assert!((b.position - (hand + Vec3::X * 0.03 + aim * Vec3::NEG_Z * MUZZLE_AHEAD)).length() < 1e-5);
        assert!((b.direction - aim * Vec3::NEG_Z).length() < 1e-5 && (b.up - Vec3::Y).length() < 1e-5);
        assert!((b.velocity - Vec3::X).length() < 1e-3, "the hand's velocity: {}", b.velocity);
        assert!(b.port.is_some() && b.inputs[0] == 0.0 && b.age == 0.0);
        assert!(m.step(true, false, 0.03, hand, aim, |_| None).is_empty(), "an effect that is not there fires nothing");
    }

    /// A FLASH LIGHTS THE ROOM FOR ITS FRAME, in a dim hall, through the real
    /// pipelines: the floor under the muzzle is brighter on the frame of
    /// the shot than with the light gated away. `cargo test --release
    /// --lib a_flash_lights_the_dim_hall -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn a_flash_lights_the_dim_hall() {
        use crate::offline_frame::{offline_game_dir, offline_scene, render_brushes, View};
        let (lib, _) = EffectLibrary::from_dir(&offline_game_dir().join("effects"));
        let (eye, at) = (Vec3::new(14.0, 1.6, 2.5), Vec3::new(14.0, 1.0, -7.0));
        let f = (at - eye).normalize();
        let muzzle = eye + f * 0.7 + Vec3::X * 0.14 - Vec3::Y * 0.16;
        let seed = (1..200u64)
            .find(|seed| {
                let mut set = BurstSet::default();
                set.add(&lib, burst(lib.id("muzzle_556_a2").unwrap(), muzzle, Quat::IDENTITY, Vec3::ZERO, Shot { age: 0.0, heat: 0.0, seed: *seed }));
                !set.lights(&lib, 9.0, Vec3::ZERO, Quat::IDENTITY).is_empty()
            })
            .unwrap();
        std::env::set_var("BURST", format!("muzzle_556_a2,{},{},{},{},{},{},0,0,{seed}", muzzle.x, muzzle.y, muzzle.z, f.x, f.y, f.z));
        std::env::remove_var("TIME_OF_DAY");
        let view = View { adapt: true, terrain: true, effects: true, water: true, weather: true, ..View::headset(eye, at) };
        // The floor in the lower third of the picture, its mean level.
        let floor = |exposure: &str| {
            std::env::set_var("BURST_EXPOSURE", exposure);
            let shot = render_brushes(&offline_scene(), view)?;
            let rows = shot.height * 2 / 3..shot.height;
            let sum: u64 = rows.clone().flat_map(|y| (0..shot.width).map(move |x| (x, y))).map(|(x, y)| shot.px(x, y)[..3].iter().map(|c| *c as u64).sum::<u64>()).sum();
            Some(sum as f64 / (3 * rows.len() as u64 * shot.width as u64) as f64)
        };
        let (Some(lit), Some(gated)) = (floor("9.16"), floor("1.4")) else {
            eprintln!("skipping: no GPU or no scene");
            return;
        };
        eprintln!("the hall's floor: {gated:.2} levels with the light gated, {lit:.2} with it");
        assert!(lit > gated + 2.0, "the flash's light shows on the floor: {gated} -> {lit}");
        std::env::remove_var("BURST");
        std::env::remove_var("BURST_EXPOSURE");
    }

    /// THE STRIPS: each firearm effect fired from a shouldered weapon, 0.7 m
    /// ahead of the eye, through the real pipelines of the offline frame --
    /// in daylight, at dusk, in the dim brick hall and down the stone hallway
    /// with the torch lit. A single shot at six moments (the frame of the
    /// flash, the next frame, then its smoke), and a string of eight at 12.5
    /// a second. `OUT=dir cargo test --release --lib render_the_muzzle_strips
    /// -- --ignored --nocapture`; `ONLY=day` (or `dusk`, `hall`, `torch`) for
    /// one place. Each picture is cropped round the muzzle at full size.
    #[test]
    #[ignore]
    fn render_the_muzzle_strips() {
        use crate::offline_frame::{offline_game_dir, offline_scene, render_brushes, View};
        let out = std::path::PathBuf::from(std::env::var("OUT").unwrap_or_else(|_| "/tmp".into()));
        let only = std::env::var("ONLY").ok();
        let (lib, _) = EffectLibrary::from_dir(&offline_game_dir().join("effects"));
        // (name, eye, at, hour, the exposure the harness meters there, torch)
        let places: [(&str, Vec3, Vec3, Option<&str>, f32, bool); 4] = [
            ("day", Vec3::new(0.0, 1.6, 10.0), Vec3::new(0.0, 1.8, 3.7), Some("13"), 1.40, false),
            ("dusk", Vec3::new(0.0, 1.6, 10.0), Vec3::new(0.0, 1.8, 3.7), Some("18.6"), 3.78, false),
            ("hall", Vec3::new(14.0, 1.6, 2.5), Vec3::new(14.0, 1.0, -7.0), None, 9.16, false),
            ("torch", Vec3::new(3.4, 1.6, -3.0), Vec3::new(9.0, 1.4, -3.0), None, 3.20, true),
        ];
        const CROP: u32 = 600;
        for (place, eye, at, hour, exposure, torch) in places {
            if only.as_deref().is_some_and(|o| o != place) {
                continue;
            }
            let f = (at - eye).normalize();
            let right = f.cross(Vec3::Y).normalize();
            let up = right.cross(f);
            let muzzle = eye + f * 0.7 + right * 0.14 - up * 0.16;
            match hour {
                Some(h) => std::env::set_var("TIME_OF_DAY", h),
                None => std::env::remove_var("TIME_OF_DAY"),
            }
            std::env::set_var("BURST_EXPOSURE", exposure.to_string());
            let view = View {
                adapt: true,
                terrain: true,
                effects: true,
                water: true,
                weather: true,
                flashlight: torch.then(|| (eye + right * 0.25 - up * 0.35 + f * 0.2, eye + f * 5.0 - up * 0.4)),
                ..View::headset(eye, at)
            };
            // Where the muzzle is in the picture: the crop's middle.
            let focal = 0.5 * view.height as f32 / (0.5 * view.fov_y.to_radians()).tan();
            let (cx, cy) = (0.5 * view.width as f32 + focal * 0.14 / 0.7, 0.5 * view.height as f32 + focal * 0.16 / 0.7);
            let (x0, y0) = ((cx as u32).saturating_sub(CROP / 2).min(view.width - CROP), (cy as u32).saturating_sub(CROP / 2).min(view.height - CROP));
            // A seed whose shot flashes, where the effect can.
            let flashing = |name: &str| {
                (1..200u64)
                    .find(|seed| {
                        let mut set = BurstSet::default();
                        set.add(&lib, burst(lib.id(name).unwrap(), muzzle, Quat::IDENTITY, Vec3::ZERO, Shot { age: 0.0, heat: 0.0, seed: *seed }));
                        !set.lights(&lib, 9.0, Vec3::ZERO, Quat::IDENTITY).is_empty()
                    })
                    .unwrap_or(1)
            };
            let spec = |name: &str, age: f32, heat: f32, seed: u64| format!("{name},{},{},{},{},{},{},{age},{heat},{seed}", muzzle.x, muzzle.y, muzzle.z, f.x, f.y, f.z);
            let mut strips: Vec<(String, Vec<String>)> = Vec::new();
            for (name, _) in EFFECTS {
                let seed = flashing(name);
                strips.push((format!("{place}_{name}_single"), [0.0, 0.028, 0.12, 0.35, 0.8, 1.4].iter().map(|age| spec(name, *age, 0.0, seed)).collect()));
            }
            // Eight shots of the rifle 80 ms apart: the last one's frame, the
            // next, and the haze and the barrel's thread after the string.
            let name = EFFECTS[0].0;
            let last = flashing(name);
            strips.push((
                format!("{place}_{name}_string"),
                [0.0, 0.028, 0.3, 0.9, 2.0, 3.5]
                    .iter()
                    .map(|after| {
                        (0..8).map(|k| spec(name, (7 - k) as f32 * 0.08 + after, k as f32 / HEAT_SHOTS, if k == 7 { last } else { 300 + k as u64 })).collect::<Vec<_>>().join(";")
                    })
                    .collect(),
            ));
            for (label, frames) in strips {
                let mut strip = image::RgbaImage::new(CROP * frames.len() as u32, CROP);
                for (k, burst_spec) in frames.iter().enumerate() {
                    std::env::set_var("BURST", burst_spec);
                    let Some(shot) = render_brushes(&offline_scene(), view) else {
                        eprintln!("skipping: no GPU or no scene");
                        return;
                    };
                    let whole = image::RgbaImage::from_raw(shot.width, shot.height, shot.rgba).expect("a frame");
                    image::imageops::replace(&mut strip, &image::imageops::crop_imm(&whole, x0, y0, CROP, CROP).to_image(), (k as u32 * CROP) as i64, 0);
                }
                let path = out.join(format!("{label}.png"));
                strip.save(&path).expect("write the strip");
                eprintln!("wrote {}", path.display());
            }
        }
        std::env::remove_var("BURST");
    }
}
