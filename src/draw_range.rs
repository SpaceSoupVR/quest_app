//! WHAT IS HANDED TO THE RENDERER AT ALL, by how far it is and how big.
//!
//! Apart from `render_prep`, which uses it, so that it is built -- and
//! tested -- on the development machine too.

use glam::Vec3;

/// Anything is drawn while its NEAREST point can be within this of the head,
/// metres, whatever its size.
pub const MAX_RENDER_DIST: f32 = 40.0;

/// ... and a big thing for as long as it is big to the eye: out to
/// `Levers::size_reach` of its own bounding radii -- 229 where its sphere is
/// half a degree across (a dozen pixels in the headset), `1 / tan(0.25
/// degrees)`.
#[cfg_attr(not(test), allow(dead_code))]
pub const SIZE_REACH: f32 = 229.0;

/// WHETHER A THING IS HANDED TO THE RENDERER AT ALL: by its bounding sphere
/// (`origin`, `radius`, world metres) against the head, never by its origin
/// alone. Until 2026-10-13 it was `origin within 40 m`, so a model, a
/// fixture or a baked cave -- a model instance 40 m across whose origin is
/// 39 m from test_room's spawn -- vanished whole the moment its ORIGIN was
/// 40 m off, standing in it or not, and popped back on the way in
/// (`docs/scale-audit-2026-10-13.md`, F8). What is in view of the eye is the
/// renderer's own cull (`scene_cull`), by the same sphere.
pub fn drawn_at(head: Vec3, origin: Vec3, radius: f32, size_reach: f32) -> bool {
    let radius = radius.max(0.0);
    origin.distance(head) - radius < MAX_RENDER_DIST.max(radius * size_reach.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A thing is drawn by its bounding sphere, never its origin alone: a
    /// cave 40 m across does not vanish while the head is in it or near it,
    /// and nothing pops in from 40 m at a size that shows.
    #[test]
    fn a_thing_is_drawn_by_its_bounds_and_its_size_not_its_origin() {
        let head = Vec3::new(0.0, 1.6, 0.0);
        // A small prop: to 40 m of its nearest point, as before.
        assert!(drawn_at(head, Vec3::new(39.0, 1.6, 0.0), 0.05, SIZE_REACH));
        assert!(!drawn_at(head, Vec3::new(41.0, 1.6, 0.0), 0.05, SIZE_REACH));
        // test_room's cave: its origin 45 m off, its rock reaching 22 m from
        // it. The old rule dropped it; the head may be 23 m from its mouth.
        assert!(Vec3::new(45.0, 1.6, 0.0).distance(head) > MAX_RENDER_DIST);
        assert!(drawn_at(head, Vec3::new(45.0, 1.6, 0.0), 22.0, SIZE_REACH));
        // ... and from across a large level, while it is big to the eye.
        assert!(drawn_at(head, Vec3::new(800.0, 1.6, 0.0), 22.0, SIZE_REACH));
        assert!(!drawn_at(head, Vec3::new(6000.0, 1.6, 0.0), 22.0, SIZE_REACH));
        // A lamp 0.3 m across its radius is drawn out to where it is half a
        // degree wide, 69 m, not cut at 40.
        assert!(drawn_at(head, Vec3::new(60.0, 3.0, 0.0), 0.3, SIZE_REACH));
        assert!(!drawn_at(head, Vec3::new(75.0, 3.0, 0.0), 0.3, SIZE_REACH));
        // Never by a step: walking toward anything, once drawn it stays drawn.
        for radius in [0.0f32, 0.05, 0.3, 2.0, 22.0] {
            let mut was = false;
            for step in (0..4000).rev() {
                let is = drawn_at(head, Vec3::new(step as f32 * 0.5, 1.6, 0.0), radius, SIZE_REACH);
                assert!(is || !was, "radius {radius}: it vanished {} m off on the way in", step as f32 * 0.5);
                was = is;
            }
            assert!(was);
        }
        // With no reach by size, the 40 m rule alone -- but by bounds still.
        assert!(drawn_at(head, Vec3::new(45.0, 1.6, 0.0), 22.0, 0.0) && !drawn_at(head, Vec3::new(70.0, 1.6, 0.0), 22.0, 0.0));
        assert!(!drawn_at(head, Vec3::new(60.0, 3.0, 0.0), 0.3, 76.0) && drawn_at(head, Vec3::new(800.0, 1.6, 0.0), 22.0, 76.0));
        // A radius that is not one is no radius.
        assert!(drawn_at(head, Vec3::new(39.0, 1.6, 0.0), -5.0, SIZE_REACH) && !drawn_at(head, Vec3::new(41.0, 1.6, 0.0), -5.0, SIZE_REACH));
    }
}
