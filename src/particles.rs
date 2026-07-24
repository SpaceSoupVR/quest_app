
use glam::{Quat, Vec3};

use space_soup::renderer::{Color3, Particle};
use space_soup_protocol::WireRenderParticleEmitter;

fn hash_to_unit_floats(id: &str, slot: usize) -> (f32, f32) {
    let id_hash = id
        .bytes()
        .fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3));

    let mix = |mut x: u64| -> u64 {
        x ^= x >> 30;
        x = x.wrapping_mul(0xbf58476d1ce4e5b9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94d049bb133111eb);
        x ^= x >> 31;
        x
    };
    let h1 = mix(id_hash ^ (slot as u64).wrapping_mul(2));
    let h2 = mix(id_hash ^ (slot as u64).wrapping_mul(2).wrapping_add(1));
    (
        (h1 % 1_000_000) as f32 / 1_000_000.0,
        (h2 % 1_000_000) as f32 / 1_000_000.0,
    )
}

fn cone_direction(forward: Vec3, spread_deg: f32, u1: f32, u2: f32) -> Vec3 {
    let forward = forward.normalize_or_zero();
    let axis_ref = if forward.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let right = forward.cross(axis_ref).normalize_or_zero();
    let up = forward.cross(right);

    let max_theta = spread_deg.to_radians();
    let theta = u1 * max_theta;
    let phi = u2 * std::f32::consts::TAU;

    let (sin_t, cos_t) = theta.sin_cos();
    let (sin_p, cos_p) = phi.sin_cos();
    right * (sin_t * cos_p) + up * (sin_t * sin_p) + forward * cos_t
}

pub fn simulate(
    emitters: &[WireRenderParticleEmitter],
    sim_time: f32,
    offset: Vec3,
    yaw_inv: Quat,
) -> Vec<Particle> {
    let mut out = Vec::new();

    for e in emitters {
        let spawn_rate = e.spawn_rate.max(0.01);
        let lifetime = e.lifetime.max(0.01);
        let capacity = (spawn_rate * lifetime).ceil() as usize;
        let direction = Vec3::from(e.direction);
        let position = Vec3::from(e.position);
        let base_alpha = e.color.3 as f32 / 255.0;

        for i in 0..capacity {
            let age = (sim_time - i as f32 / spawn_rate).rem_euclid(lifetime);
            let t = age / lifetime;
            let (u1, u2) = hash_to_unit_floats(&e.id, i);
            let dir = cone_direction(direction, e.spread_deg, u1, u2);
            let local_pos = position + dir * e.speed * age;
            let alpha = (base_alpha * (1.0 - t) * 255.0).round() as u8;

            out.push(Particle {
                position: yaw_inv * (local_pos - offset),
                size: e.particle_size,
                color: Color3(e.color.0, e.color.1, e.color.2, alpha),
            });
        }
    }

    out
}

