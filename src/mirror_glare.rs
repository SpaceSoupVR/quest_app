//! THE FLASHLIGHT'S GLARE IN POLISHED SURFACES. A torch's glass is drawn into
//! the reflections (`flashlight::torch_capsules`) as bright as it glows, but
//! the renderer keeps no HDR image to bloom: its glare is drawn source by
//! source (`space_soup::renderer::glare`). So each polished plane the glass
//! shines on gives it one more source -- its IMAGE there, standing behind the
//! mirror as far as the glass stands in front, as bright as the surface sends
//! its light on toward the eye -- whose veil is tested at the point on the
//! mirror the eye sees it in (`GlareSource::mirror`). The user, 2026-10-02:
//! the flashlight "will need to have the hdr bloom effect on it, at least in
//! reflections or to other players".
//!
//! Flat brush faces only: a plane's image of a point is exact. A rough
//! surface's is a smear as wide as its lobe, whose veil is not a bulb's, so
//! images fade out by [`MIRROR_MAX_ROUGHNESS`], as the scene shader's sharp
//! reflection gives way to the lightmap's blur.

use glam::Vec3;

/// Images are drawn in full on surfaces this smooth, fading to none at
/// [`MIRROR_MAX_ROUGHNESS`] (marble is 0.048).
pub(crate) const MIRROR_FULL_ROUGHNESS: f32 = 0.1;
pub(crate) const MIRROR_MAX_ROUGHNESS: f32 = 0.25;

/// The brightest this many images a torch: the floor and a pillar beside it,
/// say. Each is a quad of glare.
pub(crate) const MAX_IMAGES: usize = 2;

/// A polished plane of the level: `normal . x = offset`, facing out of its
/// solid, in the WORLD, and its material's mean roughness. See
/// `BrushGeometry::smooth_planes`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MirrorPlane {
    pub normal: Vec3,
    pub offset: f32,
    pub roughness: f32,
}

/// Where a ray from the eye meets the level: how far, the face's normal and
/// its material's roughness.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Meeting {
    pub distance: f32,
    pub normal: Vec3,
    pub roughness: f32,
}

/// The glass's image in one plane, in the WORLD.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Image {
    /// Where the image stands, behind the mirror, and the way its beam goes.
    pub glass: Vec3,
    pub forward: Vec3,
    /// The point on the mirror the eye sees it at.
    pub mirror: Vec3,
    /// How much of the glass's light the surface sends on toward the eye:
    /// the scene shader's Fresnel, faded by roughness.
    pub reflectance: f32,
}

/// The level's polished planes: its brushes' faces no rougher than
/// [`MIRROR_MAX_ROUGHNESS`], by their materials' mean `roughness` by layer.
pub(crate) fn planes_of(brushes: &crate::brush_render::BrushGeometry, roughness: &[f32]) -> Vec<MirrorPlane> {
    brushes
        .smooth_planes(roughness, MIRROR_MAX_ROUGHNESS)
        .into_iter()
        .map(|(normal, offset, roughness)| MirrorPlane { normal, offset, roughness })
        .collect()
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The share of light a surface of `roughness` reflects toward a direction
/// `cos_view` off its normal: Schlick from a dielectric's 0.04, its sweep
/// capped at `1 - roughness`, as the scene shader has it
/// (`shade_material_env_part`).
pub(crate) fn reflectance(roughness: f32, cos_view: f32) -> f32 {
    let f0 = 0.04;
    let f_max = (1.0 - roughness).max(f0);
    let g = 1.0 - cos_view.clamp(0.0, 1.0);
    f0 + (f_max - f0) * g * g * g * g * g
}

/// THE IMAGES of a glass at `glass` shining along `forward` (unit) into a
/// cone of half angles `cos_outer`..`cos_inner`, seen by an eye at `eye`, in
/// the level's polished `planes` -- the brightest [`MAX_IMAGES`], in the
/// WORLD. `cast(from, dir, max)` is where a ray first meets the level within
/// `max`: an image is drawn only where the eye's ray to its mirror point lands
/// on that plane's polished face, and nothing stands between that point and
/// the glass.
pub(crate) fn images(
    glass: Vec3,
    forward: Vec3,
    (cos_outer, cos_inner): (f32, f32),
    eye: Vec3,
    planes: &[MirrorPlane],
    cast: impl Fn(Vec3, Vec3, f32) -> Option<Meeting>,
) -> Vec<Image> {
    let mut out: Vec<(f32, Image)> = Vec::new();
    for p in planes {
        // Both in front of it.
        let (in_front, eye_in_front) = (p.normal.dot(glass) - p.offset, p.normal.dot(eye) - p.offset);
        if in_front <= 1e-3 || eye_in_front <= 1e-3 {
            continue;
        }
        let image = glass - p.normal * (2.0 * in_front);
        let mirror = eye + (image - eye) * (eye_in_front / (eye_in_front + in_front));
        // Lit by the beam, as much as the beam sends that way.
        let Some(lit) = (mirror - glass).try_normalize() else { continue };
        let share = smoothstep(cos_outer, cos_inner.max(cos_outer + 1e-3), forward.dot(lit));
        if share <= 0.0 {
            continue;
        }
        // The eye's ray lands on this plane's polished face there...
        let to_mirror = mirror - eye;
        let reach = to_mirror.length();
        let Some(m) = cast(eye, to_mirror / reach, reach + 0.05) else { continue };
        if (m.distance - reach).abs() > 0.02 || m.normal.dot(p.normal) < 0.999 || m.roughness > MIRROR_MAX_ROUGHNESS {
            continue;
        }
        // ...and sees the glass from it.
        let to_glass = glass - mirror;
        let apart = to_glass.length();
        if cast(mirror + p.normal * 2e-3, to_glass / apart, apart - 0.01).is_some() {
            continue;
        }
        let r = reflectance(m.roughness, p.normal.dot(-to_mirror / reach))
            * (1.0 - smoothstep(MIRROR_FULL_ROUGHNESS, MIRROR_MAX_ROUGHNESS, m.roughness));
        if r <= 0.0 {
            continue;
        }
        let forward = forward - p.normal * (2.0 * forward.dot(p.normal));
        // The light it brings the eye, to choose the brightest by.
        let strength = r * share / (image - eye).length_squared().max(1e-4);
        out.push((strength, Image { glass: image, forward, mirror, reflectance: r }));
    }
    out.sort_by(|a, b| b.0.total_cmp(&a.0));
    out.into_iter().take(MAX_IMAGES).map(|(_, i)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLOOR: MirrorPlane = MirrorPlane { normal: Vec3::Y, offset: 0.0, roughness: 0.05 };
    const CONE: (f32, f32) = (0.9063, 0.9903); // 50 and 16 degrees across

    /// An open floor: every downward ray meets it.
    fn floor_only(from: Vec3, dir: Vec3, max: f32) -> Option<Meeting> {
        let t = -from.y / dir.y;
        (dir.y < 0.0 && t > 0.0 && t <= max).then_some(Meeting { distance: t, normal: Vec3::Y, roughness: 0.05 })
    }

    /// A torch two metres off, held a metre up and aimed at the floor halfway
    /// back toward an eye standing at 1.6 m: just where that eye sees the
    /// glass's image in it -- as someone across a polished floor aims one.
    /// `(glass, forward, eye)`.
    fn aimed_between() -> (Vec3, Vec3, Vec3) {
        let (glass, eye) = (Vec3::new(0.0, 1.0, -2.0), Vec3::new(0.0, 1.6, 0.0));
        let image = Vec3::new(glass.x, -glass.y, glass.z);
        let on_floor = eye + (image - eye) * (eye.y / (eye.y - image.y));
        (glass, (on_floor - glass).normalize(), eye)
    }

    /// THE GLASS'S IMAGE IN THE FLOOR: under it as far as the glass is above,
    /// its beam turned up, seen at the point where the eye's line to the image
    /// crosses the floor, as bright as Fresnel sends the light that way.
    #[test]
    fn a_torch_aimed_at_the_floor_has_its_image_in_it() {
        let (glass, forward, eye) = aimed_between();
        let found = images(glass, forward, CONE, eye, &[FLOOR], floor_only);
        assert_eq!(found.len(), 1, "{found:?}");
        let i = found[0];
        assert!((i.glass - Vec3::new(glass.x, -glass.y, glass.z)).length() < 1e-5);
        assert!((i.forward - Vec3::new(forward.x, -forward.y, forward.z)).length() < 1e-5);
        assert!(i.mirror.y.abs() < 1e-5, "on the floor: {}", i.mirror);
        assert!(((i.mirror - eye).normalize() - (i.glass - eye).normalize()).length() < 1e-5, "on the line of sight");
        let cos = (eye - i.mirror).normalize().y;
        assert!((i.reflectance - reflectance(0.05, cos)).abs() < 1e-6);
        assert!(i.reflectance > 0.04 && i.reflectance < 0.2, "{}", i.reflectance);
    }

    /// NONE WHERE IT CANNOT BE: the beam aimed past the point the eye would
    /// see it at -- the glass bright only inside its beam -- a slab between
    /// that point and the glass, the eye's ray landing short on something
    /// else, a rough face in the polished plane, the glass under the floor.
    #[test]
    fn no_image_where_the_beam_the_glass_or_the_eye_cannot_reach() {
        let (glass, forward, eye) = aimed_between();
        assert_eq!(images(glass, forward, CONE, eye, &[FLOOR], floor_only).len(), 1, "the case each below takes away");
        let away = Vec3::new(0.0, -1.0, -0.6).normalize();
        assert!(images(glass, away, CONE, eye, &[FLOOR], floor_only).is_empty(), "beam aimed away");
        let blocked = |from: Vec3, dir: Vec3, max: f32| {
            // A slab hanging between the floor point and the glass.
            if from.y < 0.01 && dir.y > 0.0 && max > 0.5 {
                return Some(Meeting { distance: 0.5, normal: -dir, roughness: 1.0 });
            }
            floor_only(from, dir, max)
        };
        assert!(images(glass, forward, CONE, eye, &[FLOOR], blocked).is_empty(), "the glass hidden from the floor");
        let short = |from: Vec3, dir: Vec3, max: f32| {
            floor_only(from, dir, max).map(|m| Meeting { distance: m.distance * 0.5, normal: -dir, ..m })
        };
        assert!(images(glass, forward, CONE, eye, &[FLOOR], short).is_empty(), "the eye's ray lands on something else");
        let rough = |from: Vec3, dir: Vec3, max: f32| floor_only(from, dir, max).map(|m| Meeting { roughness: 0.6, ..m });
        assert!(images(glass, forward, CONE, eye, &[FLOOR], rough).is_empty(), "a rough face in a polished plane");
        let under = Vec3::new(glass.x, -0.1, glass.z);
        assert!(images(under, forward, CONE, eye, &[FLOOR], floor_only).is_empty(), "under the floor");
    }

    /// FADED OUT BY ROUGHNESS, not cut: a polished floor in full, one a
    /// little rougher dimmer, and none at the limit.
    #[test]
    fn an_image_fades_with_the_surfaces_roughness() {
        let (glass, forward, eye) = aimed_between();
        let at = |r: f32| {
            let floor = MirrorPlane { roughness: r, ..FLOOR };
            let cast = move |from: Vec3, dir: Vec3, max: f32| floor_only(from, dir, max).map(|m| Meeting { roughness: r, ..m });
            images(glass, forward, CONE, eye, &[floor], cast).first().map_or(0.0, |i| i.reflectance)
        };
        let (full, part, none) = (at(0.05), at(0.18), at(MIRROR_MAX_ROUGHNESS));
        assert!(full > part && part > 0.0 && none == 0.0, "{full} {part} {none}");
    }

    /// THE BRIGHTEST FEW: of a floor and three walls, all lit, the ones that
    /// bring the eye the most light -- the two walls beside the torch, whose
    /// images stand nearest -- and at most [`MAX_IMAGES`].
    #[test]
    fn only_the_brightest_images_are_kept() {
        let glass = Vec3::new(0.0, 1.2, -1.0);
        let eye = Vec3::new(0.0, 1.6, 0.0);
        // A beam lighting every way, so that every plane holds an image.
        let everywhere = (-1.0, -0.999);
        let planes = [
            FLOOR,
            MirrorPlane { normal: Vec3::X, offset: -1.0, roughness: 0.05 },
            MirrorPlane { normal: Vec3::NEG_X, offset: -1.0, roughness: 0.05 },
            MirrorPlane { normal: Vec3::Z, offset: -3.0, roughness: 0.05 },
        ];
        // Whichever plane a ray meets first, from the front.
        let first = |from: Vec3, dir: Vec3, max: f32| {
            planes
                .iter()
                .filter_map(|p| {
                    let t = (p.offset - p.normal.dot(from)) / p.normal.dot(dir);
                    (p.normal.dot(dir) < 0.0 && t > 0.0 && t <= max).then_some((t, p.normal))
                })
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .map(|(t, n)| Meeting { distance: t, normal: n, roughness: 0.05 })
        };
        let all: Vec<Image> = planes
            .iter()
            .flat_map(|p| images(glass, Vec3::NEG_Y, everywhere, eye, std::slice::from_ref(p), first))
            .collect();
        assert_eq!(all.len(), 4, "each plane alone holds one: {all:?}");
        let found = images(glass, Vec3::NEG_Y, everywhere, eye, &planes, first);
        assert_eq!(found.len(), MAX_IMAGES);
        assert!(found.iter().all(|i| (i.mirror.x.abs() - 1.0).abs() < 1e-4), "the side walls: {found:?}");
    }
}

/// THE SHIPPED LEVEL: its polished planes are its marble, and a torch aimed
/// at the floor just ahead of the player's feet shows its glass in it.
#[cfg(test)]
mod level_tests {
    use super::*;
    use crate::brush_render::{load_materials, mean_roughness, BrushGeometry};

    #[test]
    fn test_rooms_marble_holds_the_glass_and_its_rough_faces_do_not() {
        let game = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../game");
        let Ok(scene) = space_soup_engine::Scene::load(&space_soup_engine::Manifest::scene_path(&game, "test_room")) else {
            eprintln!("no test_room; skipping");
            return;
        };
        let brushes = BrushGeometry::load(&scene);
        let maps = load_materials(&game, brushes.materials());
        let rough: Vec<f32> = maps.roughs.iter().map(|r| mean_roughness(r.as_ref())).collect();
        for (id, r) in brushes.materials().iter().zip(&rough) {
            eprintln!("material {id}: mean roughness {r:.3}");
        }
        let planes = planes_of(&brushes, &rough);
        for p in &planes {
            eprintln!("polished plane n {:.3} at {:.3}, roughness {:.3}", p.normal, p.offset, p.roughness);
        }
        assert!(planes.iter().any(|p| p.normal.y > 0.999 && p.offset.abs() < 0.05), "the marble floor");
        assert!(planes.iter().all(|p| p.roughness <= MIRROR_MAX_ROUGHNESS));
        let meet = |from: Vec3, dir: Vec3, max: f32| {
            brushes.cast(from, dir, max, &[]).map(|b| Meeting {
                distance: b.distance,
                normal: b.normal,
                roughness: rough.get(b.material as usize).copied().unwrap_or(1.0),
            })
        };
        let cone = crate::flashlight::cone_cosines();
        for (name, eye, glass, aim) in [
            ("down at the feet", Vec3::new(0.0, 1.6, -1.0), Vec3::new(0.2, 1.3, -1.3), Vec3::new(0.1, 0.0, -1.5)),
            ("torch_facing", Vec3::new(0.0, 1.6, -1.0), Vec3::new(0.1, 1.45, -3.0), Vec3::new(0.0, 1.6, -1.0)),
            ("at the pillar", Vec3::new(0.3, 1.6, -3.0), Vec3::new(0.45, 1.3, -3.2), Vec3::new(0.3, 1.2, -6.4)),
            ("torch_in_hand", Vec3::new(0.0, 1.6, -1.0), Vec3::new(0.2, 1.25, -1.45), Vec3::new(0.0, 0.3, -4.0)),
        ] {
            let (lens, rotation) = crate::flashlight::lens_from_bench(glass, aim);
            let found = images(lens, rotation * Vec3::NEG_Z, cone, eye, &planes, meet);
            eprintln!("{name}: {found:?}");
            if name == "down at the feet" {
                assert!(!found.is_empty(), "{name}: the glass in the marble floor");
            }
        }
    }
}
