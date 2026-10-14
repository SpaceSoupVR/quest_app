//! MEASUREMENTS for the frame drops the user felt on 2026-10-14
//! (`docs/grab-and-lag-status-2026-10-14.md`). Tests only, host CPU; each
//! prints a table and asserts nothing about speed.

#![cfg(test)]

use glam::{Quat, Vec3};
use space_soup::renderer::effects::{simulate_all, EffectEmitter, EffectKind, Surroundings};
use space_soup::renderer::{Color3, Light, LightKind};

/// The marble hall's dust (`hall_dust` in test_room: centre, half size, rate).
fn hall_dust() -> EffectEmitter {
    EffectEmitter {
        id: "hall_dust".into(),
        kind: EffectKind::Dust,
        position: Vec3::new(0.0, 1.55, -4.5),
        direction: Vec3::Y,
        extent: [Vec3::X * 1.2, Vec3::Y * 1.45, Vec3::Z * 1.2],
        scale: 1.0,
        rate: 0.6,
        tint: [1.0; 3],
        ceiling: None,
        variation: Default::default(),
    }
}

/// Four pendant-like spots over the hall, lighting the dust.
fn pendants() -> Vec<Light> {
    [(-1.5, -2.0), (1.5, -2.0), (-1.5, -7.0), (1.5, -7.0)]
        .iter()
        .map(|&(x, z)| Light {
            mask_channel: None,
            shadow_near: None,
            source_radius: 0.0,
            in_level_bake: false,
            position: Vec3::new(x, 3.2, z),
            direction: Vec3::NEG_Y,
            kind: LightKind::Spot,
            color: Color3(255, 240, 220, 255),
            intensity: 30.0,
            range: 8.0,
            cone_angle_deg: 70.0,
            inner_cone_angle_deg: 30.0,
        })
        .collect()
}

/// THE DUST MOTES, by where the head stands: what one frame's simulation
/// costs the CPU (median of 120), how many motes it draws, how many of them
/// are drawn with nothing to show (alpha under 1/255: inside the near focus
/// fade, or not yet found) and how many stand within 0.35 m of the eye.
/// `cargo test --release --lib dust_motes_by_where_the_head_stands -- --ignored --nocapture`.
#[test]
#[ignore]
fn dust_motes_by_where_the_head_stands() {
    let emitters = [hall_dust()];
    let lights = pendants();
    let reaches = |_: &EffectEmitter, _: usize| true;
    let ambient = |_: Vec3| Vec3::splat(0.4);
    println!("| head | instances | invisible (alpha < 1/255) | within 0.35 m | CPU ms (median of 120) |");
    println!("|---|---|---|---|---|");
    for (name, head) in [
        ("centre of the dust", Vec3::new(0.0, 1.6, -4.5)),
        ("edge of the dust", Vec3::new(1.4, 1.6, -4.5)),
        ("3 m out", Vec3::new(0.0, 1.6, -0.5)),
        ("hall_back (10 m)", Vec3::new(-1.3, 1.6, -14.5)),
    ] {
        let at = Surroundings { head, offset: Vec3::ZERO, yaw_inv: Quat::IDENTITY, lights: &lights, reaches: &reaches, ambient: &ambient, exposure: 1.0 };
        let mut times = Vec::new();
        let mut last = None;
        for k in 0..120 {
            let t = 10.0 + k as f64 / 72.0;
            let start = std::time::Instant::now();
            let frame = simulate_all(&emitters, &|_| true, &[], None, t, &at);
            times.push(start.elapsed().as_secs_f64() * 1000.0);
            last = Some(frame);
        }
        times.sort_by(f64::total_cmp);
        let frame = last.unwrap();
        let invisible = frame.instances.iter().filter(|i| i.colour[3] < 1.0 / 255.0).count();
        let near = frame.instances.iter().filter(|i| (Vec3::new(i.centre[0], i.centre[1], i.centre[2]) - head).length() < 0.35).count();
        println!("| {name} | {} | {invisible} | {near} | {:.3} |", frame.instances.len(), times[60]);
    }
}
