//! THE FLASHLIGHT'S BOUNCE: the light its beam throws back off whatever it
//! lands on. The user, 2026-10-02, on the headset: the flashlight's light
//! does not bounce. A beam on a wall lights the room round it -- the floor,
//! the ceiling, the hand holding the torch -- in the wall's colour; without
//! this, everything outside the pool stayed as dark as before the torch came
//! on.
//!
//! AS ONE LIGHT STANDING AT THE LIT PATCH, the way games have done it since a
//! virtual point light was a research idea (Keller 1997, "Instant
//! Radiosity"). The beam is traced as rays across its cone ([`beam_rays`]),
//! each carrying its share of the beam's light to what it meets; what that
//! surface sends back is its albedo times the light that arrived. A diffuse
//! patch sends `Phi / pi` straight out of itself and less as the cosine away,
//! so the bounce is a spot facing out of the patch with a half-space cone:
//! the renderer's cone falls as a smoothstep in the cosine, which sends
//! exactly `pi` times its peak in all, as a cosine does. Hits that face
//! several ways -- a corner, both walls of a hallway -- send their light round
//! more of the sphere, and the cone opens toward a full one as their normals
//! disagree, its peak lowered to keep the total ([`light`]). It stands at the
//! flux-weighted middle of the hits, lifted off them by half the patch's size
//! -- a point light ON the wall would light the wall round it from a grazing
//! angle and nothing past its own plane; lifted, it lights the room the way
//! the patch's width does.
//!
//! ONE LIGHT A PATCH, AND AT MOST TWO PATCHES ([`patches`]). A beam past a
//! pillar's edge lands on the pillar 3.4 m off and on the floor, walls and
//! ceiling 5.6 to 12.6 m off, and the middle of all those hits is the air
//! between them: one light there lit nothing near the player, nor the pillar
//! (test_room's `torch_pillar`, 2026-10-02). So the hits are joined to their
//! neighbours on the same surface -- near each other, at nearly the same
//! depth -- and the brightest such patch has a light of its own; whatever
//! else the beam lights shares the second. A wall, a corner or a strip of
//! floor is one patch; a pillar and the room behind it are two.
//!
//! It casts no shadow -- a source a metre wide casts none sharp enough to map,
//! and a half-space cone does not fit in a shadow map -- and makes no point
//! highlight, its highlight being the reflection of a patch, not a bulb
//! (`space_soup::renderer::lights::SURFACE_LIGHT`). So it can light a little
//! through a wall within its range: kept short for that reason
//! ([`BOUNCE_RANGE`]).
//!
//! The rays and the patch are in the WORLD, as the brushes and the physics
//! scene are; [`light`] hands it over in the player's frame, as
//! `flashlight::beam` does the beam.

use glam::{Quat, Vec3};
use space_soup::renderer::{Color3, Light, LightKind};

use crate::flashlight;

/// How far the bounce reaches, metres: the room the patch is in. Its light
/// casts no shadow, so past the room it would reach the next one through the
/// wall -- the inverse square has fallen to a fortieth of its light at a
/// metre by then.
pub(crate) const BOUNCE_RANGE: f32 = 6.0;

/// How quickly the patch follows the beam, seconds. A beam swept across a
/// corner moves its rays from one wall to the next a few at a time; held this
/// long, the light slides rather than steps -- and a still beam is unchanged.
pub(crate) const SMOOTHING_SECONDS: f32 = 0.08;

/// How far the light stands off the patch: half its size, between these.
const MIN_LIFT: f32 = 0.05;
const MAX_LIFT: f32 = 0.4;

/// Two hits lie on one surface when they are no farther apart than this share
/// of the nearer one's distance from the glass -- neighbouring rays are 9
/// degrees apart, so 0.16 of it on a surface square to the beam and up to
/// half at a grazing one -- and the deeper lies no deeper than `LINK_DEPTH`
/// times the nearer. Neighbours on one surface meet both, a grazing floor's
/// at up to 1.3x; a pillar's base and the floor behind it are 1.5x apart.
const LINK_SPACING: f32 = 0.6;
const LINK_DEPTH: f32 = 1.35;

/// The least share of the beam's bounce the second light may carry. A ray or
/// two past an edge is left out rather than given a light of its own, or
/// pulling the patch's light off its surface toward it.
const MIN_SHARE: f32 = 0.1;

/// The most patches held at once: two, and two more fading out.
const MAX_HELD: usize = 4;

/// The albedo of a surface that is not a brush -- the ground, a model --
/// whose material the rays cannot read: a mid grey.
pub(crate) const UNKNOWN_ALBEDO: f32 = 0.3;

/// What one ray across the beam met: where, which way the surface faces
/// (toward the ray), and how much of the light it sends back, linear RGB.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Landing {
    pub point: Vec3,
    pub normal: Vec3,
    pub albedo: Vec3,
}

/// One ray across the beam: its angle from the axis and round it, radians,
/// and the share of the beam's light it carries -- steradians weighted by the
/// cone's falloff, so times the beam's intensity it is the light it carries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BeamRay {
    pub theta: f32,
    pub phi: f32,
    pub flux: f32,
}

/// The beam as rays: the centre, then rings round it, each ray standing for
/// its share of a ring -- inside the hot spot, across its edge, and out to the
/// spill's -- at the angle that ring's light is centred on. The cone's falloff
/// is the renderer's: a smoothstep in the cosine from the spill's edge to the
/// hot spot's (`spot_cone`). Full angles in degrees, as a `Light` has them.
pub(crate) fn beam_rays(cone_deg: f32, hotspot_deg: f32) -> Vec<BeamRay> {
    let outer = 0.5 * cone_deg.to_radians();
    let inner = (0.5 * hotspot_deg.to_radians()).min(outer);
    let (cos_outer, cos_inner) = (outer.cos(), inner.cos().max(outer.cos() + 1e-4));
    let falloff = |theta: f32| {
        let t = ((theta.cos() - cos_outer) / (cos_inner - cos_outer)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    let edges = [0.0, 0.7 * inner, inner + 0.35 * (outer - inner), outer];
    let counts = [1usize, 6, 12];
    let mut out = Vec::new();
    for ring in 0..3 {
        let (a, b) = (edges[ring], edges[ring + 1]);
        const STEPS: usize = 48;
        let (mut flux, mut moment) = (0.0f32, 0.0f32);
        for s in 0..STEPS {
            let theta = a + (b - a) * (s as f32 + 0.5) / STEPS as f32;
            let w = falloff(theta) * std::f32::consts::TAU * theta.sin() * (b - a) / STEPS as f32;
            flux += w;
            moment += w * theta;
        }
        let theta = if ring == 0 || flux <= 0.0 { 0.5 * (a + b) * (ring as f32).min(1.0) } else { moment / flux };
        let n = counts[ring];
        for k in 0..n {
            // Each ring turned half a step against the last, so no ray hides
            // behind another along the same azimuth.
            let phi = std::f32::consts::TAU * (k as f32 + 0.5 * (ring % 2) as f32) / n as f32;
            out.push(BeamRay { theta, phi, flux: flux / n as f32 });
        }
    }
    out
}

/// The patch the beam lights, as one source, in the WORLD.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Patch {
    /// Where its light stands: the middle of what the beam lit, lifted off it.
    pub position: Vec3,
    /// Which way the lit surface faces.
    pub normal: Vec3,
    /// The light it sends back, linear RGB, in the lamps' units of intensity
    /// times steradians.
    pub flux: Vec3,
    /// How widely its hits lie round their middle: the flux-weighted RMS
    /// distance, metres.
    pub spread: f32,
    /// How much its hits face one way: the length of their flux-weighted
    /// mean normal. 1 for a flat patch, 0.7 for a corner, near 0 across a
    /// hallway.
    pub directionality: f32,
}

/// One ray's landing as the bounce counts it.
struct Hit {
    point: Vec3,
    normal: Vec3,
    /// The light it sends back, linear RGB.
    sent: Vec3,
    /// The light it brought, before the surface took its share: `sent` is
    /// the albedo times this.
    brought: f32,
    /// Its luminance, which weighs it.
    weight: f32,
    /// How far from the glass.
    along: f32,
}

/// What the beam lands on this instant, ray by ray: what its bounce
/// ([`Landings::patches`]) and the surfaces its pool shows on in reflections
/// ([`Landings::surfaces`]) are both made from.
pub(crate) struct Landings {
    hits: Vec<Hit>,
    /// The glass, and the beam's axis and across: the frame its surfaces'
    /// pool maps are seen in ([`LitPlane`]).
    lens: Vec3,
    forward: Vec3,
    right: Vec3,
}

/// The patches the beam from the glass at `lens`, turned by `rotation` (the
/// beam along its -Z), lights this instant -- one, or two where the beam
/// lands on surfaces apart (see the module notes): each of `rays` cast with
/// `land`, carrying its light as far as the beam's windowed inverse square
/// lets it. Empty when the beam lands on nothing.
pub(crate) fn patches(
    lens: Vec3,
    rotation: Quat,
    rays: &[BeamRay],
    land: impl Fn(Vec3, Vec3) -> Option<Landing>,
) -> Vec<Patch> {
    landings(lens, rotation, rays, land).patches()
}

/// Where each of `rays` from the glass at `lens`, turned by `rotation`, lands
/// (`land`), and what it brings there. See [`patches`].
pub(crate) fn landings(
    lens: Vec3,
    rotation: Quat,
    rays: &[BeamRay],
    land: impl Fn(Vec3, Vec3) -> Option<Landing>,
) -> Landings {
    let forward = (rotation * Vec3::NEG_Z).normalize_or_zero();
    let right = (rotation * Vec3::X).normalize_or_zero();
    let up = (rotation * Vec3::Y).normalize_or_zero();
    let mut hits: Vec<Hit> = Vec::with_capacity(rays.len());
    for r in rays {
        let dir = (forward * r.theta.cos() + (right * r.phi.cos() + up * r.phi.sin()) * r.theta.sin()).normalize_or_zero();
        let Some(l) = land(lens, dir) else { continue };
        // The beam's own falloff with distance, as the shader windows it.
        let along = (l.point - lens).length();
        let x = along / flashlight::RANGE;
        let window = (1.0 - x * x * x * x).clamp(0.0, 1.0).powi(2);
        let brought = flashlight::INTENSITY * r.flux * window;
        let sent = l.albedo * brought;
        let weight = luminance(sent);
        if weight > 0.0 {
            hits.push(Hit { point: l.point, normal: l.normal, sent, brought, weight, along });
        }
    }
    Landings { hits, lens, forward, right }
}

impl Landings {
    /// The patches its bounce stands at. See [`patches`].
    pub(crate) fn patches(&self) -> Vec<Patch> {
        patches_of(&self.hits, self.forward)
    }

    /// THE SURFACES THE BEAM LIGHTS, brightest first, at most
    /// [`MAX_SURFACES`]: its hits grouped by the plane they share -- normals
    /// within `SURFACE_COS`, planes within `SURFACE_GAP` -- each with its
    /// flux-weighted albedo, and the beam's frame for its pool map. A
    /// reflection meeting one shows the beam's pool there
    /// (`space_soup::renderer::lights::LitSurface`).
    pub(crate) fn surfaces(&self) -> Vec<LitPlane> {
        let mut order: Vec<usize> = (0..self.hits.len()).collect();
        order.sort_by(|a, b| self.hits[*b].weight.total_cmp(&self.hits[*a].weight));
        // (the first hit's plane, the hits on it)
        let mut groups: Vec<(Vec3, f32, Vec<usize>)> = Vec::new();
        for i in order {
            let h = &self.hits[i];
            match groups.iter_mut().find(|(n, off, _)| n.dot(h.normal) > SURFACE_COS && (n.dot(h.point) - off).abs() < SURFACE_GAP) {
                Some((_, _, on)) => on.push(i),
                None => groups.push((h.normal, h.normal.dot(h.point), vec![i])),
            }
        }
        let mut out: Vec<LitPlane> = groups
            .iter()
            .filter_map(|(_, _, on)| {
                let (mut sent, mut brought, mut weight, mut middle, mut facing) = (Vec3::ZERO, 0.0f32, 0.0f32, Vec3::ZERO, Vec3::ZERO);
                for h in on.iter().map(|&i| &self.hits[i]) {
                    sent += h.sent;
                    brought += h.brought;
                    weight += h.weight;
                    middle += h.point * h.weight;
                    facing += h.normal * h.weight;
                }
                if !(weight > 0.0 && brought > 0.0) {
                    return None;
                }
                let normal = facing.try_normalize()?;
                Some(LitPlane {
                    normal,
                    offset: normal.dot(middle / weight),
                    albedo: sent / brought,
                    weight,
                    lens: self.lens,
                    forward: self.forward,
                    right: self.right,
                })
            })
            .collect();
        out.sort_by(|a, b| b.weight.total_cmp(&a.weight));
        out.truncate(MAX_SURFACES);
        out
    }
}

/// A SURFACE THE BEAM LIGHTS, in the WORLD: the plane its hits share
/// (`normal . p == offset`, facing the glass) and its albedo, linear RGB --
/// for the reflections to show the beam's pool on it ([`Landings::surfaces`])
/// -- and the beam that found it, as its pool map sees the plane: from the
/// glass (`lens`) along `forward`, `right` across. `weight`: the luminance it
/// sends back, which ranks it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LitPlane {
    pub normal: Vec3,
    pub offset: f32,
    pub albedo: Vec3,
    pub weight: f32,
    pub lens: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
}

/// How far off its axis a surface's pool map reaches, as a tangent: the
/// beam's cone and a twentieth more, where the shader's cone still softens
/// (`spot_cone`).
pub(crate) fn pool_map_tan() -> f32 {
    (0.5 * flashlight::CONE_DEG * 1.05).to_radians().tan()
}

/// The most surfaces a beam names: as many as the renderer has pool maps for,
/// in all.
pub(crate) const MAX_SURFACES: usize = space_soup::renderer::lights::MAX_LIT_SURFACES;

/// How long a surface is held past the last instant a ray found it, seconds.
/// The edge of a beam swept across a surface lands a ray on it, then not, from
/// frame to frame; dropped at once, the beam's light in its reflection
/// blinked with it. Held, nothing is lit wrongly: its pool map, made from the
/// lamps as they are, still decides what is lit; the held surface only says
/// which plane to map.
pub(crate) const SURFACE_HOLD_SECONDS: f32 = 1.0;

/// The beams' surfaces from frame to frame. See [`SURFACE_HOLD_SECONDS`].
#[derive(Default)]
pub(crate) struct HeldSurfaces {
    held: Vec<(LitPlane, f32)>,
}

impl HeldSurfaces {
    /// This frame's surfaces, `dt` seconds after the last, from every beam's
    /// this instant (`now`): each found again takes its new place, beam and
    /// albedo; each not found ages, and goes once older than the hold. The
    /// freshest, then the brightest, at most [`MAX_SURFACES`].
    pub(crate) fn follow(&mut self, now: &[LitPlane], dt: f32) -> Vec<LitPlane> {
        for (_, age) in self.held.iter_mut() {
            *age += dt.max(0.0);
        }
        for n in now {
            let same = |s: &LitPlane| s.normal.dot(n.normal) > SURFACE_COS && (s.offset - n.offset).abs() < SURFACE_GAP;
            match self.held.iter_mut().find(|(s, _)| same(s)) {
                Some(found) => *found = (*n, 0.0),
                None => self.held.push((*n, 0.0)),
            }
        }
        self.held.retain(|(_, age)| *age <= SURFACE_HOLD_SECONDS);
        self.held.sort_by(|a, b| a.1.total_cmp(&b.1).then(b.0.weight.total_cmp(&a.0.weight)));
        self.held.truncate(MAX_SURFACES);
        self.held.iter().map(|(s, _)| *s).collect()
    }
}

/// `s` in the PLAYER's frame, as the renderer takes it: the world turned by
/// `yaw_inv` about `offset`, as every light is.
pub(crate) fn lit_surface(s: &LitPlane, offset: Vec3, yaw_inv: Quat) -> space_soup::renderer::lights::LitSurface {
    space_soup::renderer::lights::LitSurface {
        normal: yaw_inv * s.normal,
        offset: s.offset - s.normal.dot(offset),
        albedo: s.albedo,
        lens: yaw_inv * (s.lens - offset),
        forward: yaw_inv * s.forward,
        right: yaw_inv * s.right,
        tan_half: pool_map_tan(),
    }
}

/// Two hits lie on one surface when their normals agree within about 11
/// degrees and their planes within 5 cm: a wall, not a wall and the pillar
/// standing a metre in front of it.
const SURFACE_COS: f32 = 0.98;
const SURFACE_GAP: f32 = 0.05;

/// The patches the beam's `hits` light, the beam along `forward`. See
/// [`patches`].
fn patches_of(hits: &[Hit], forward: Vec3) -> Vec<Patch> {
    let all: Vec<usize> = (0..hits.len()).collect();
    let back = -forward;
    let Some(one) = patch_of(hits, &all, back) else {
        return Vec::new();
    };
    // Each hit joined to its neighbours on its surface: union-find over the
    // pairs, 171 of them for 19 rays.
    let mut root: Vec<usize> = all.clone();
    fn find(root: &mut [usize], mut i: usize) -> usize {
        while root[i] != i {
            root[i] = root[root[i]];
            i = root[i];
        }
        i
    }
    for i in 0..hits.len() {
        for j in i + 1..hits.len() {
            let (a, b) = (&hits[i], &hits[j]);
            let (near, deep) = (a.along.min(b.along), a.along.max(b.along));
            if deep <= LINK_DEPTH * near && a.point.distance(b.point) <= LINK_SPACING * near {
                let (ri, rj) = (find(&mut root, i), find(&mut root, j));
                root[ri.max(rj)] = ri.min(rj);
            }
        }
    }
    let roots: Vec<usize> = (0..hits.len()).map(|i| find(&mut root, i)).collect();
    let weight_of = |part: &[usize]| part.iter().map(|&i| hits[i].weight).sum::<f32>();
    let surfaces: Vec<Vec<usize>> = {
        let mut seen: Vec<usize> = roots.clone();
        seen.sort_unstable();
        seen.dedup();
        seen.iter().map(|&r| all.iter().copied().filter(|&i| roots[i] == r).collect()).collect()
    };
    if surfaces.len() < 2 {
        return vec![one];
    }
    // The brightest patch, ties to the first found; the rest together.
    let brightest = surfaces
        .iter()
        .max_by(|a, b| weight_of(a).total_cmp(&weight_of(b)))
        .expect("two or more");
    let rest: Vec<usize> = all.iter().copied().filter(|&i| roots[i] != roots[brightest[0]]).collect();
    let mut out: Vec<Patch> = patch_of(hits, brightest, back).into_iter().collect();
    if weight_of(&rest) >= MIN_SHARE * weight_of(&all) {
        out.extend(patch_of(hits, &rest, back));
    }
    out
}

/// The patch the hits `part` light: at their flux-weighted middle, facing out
/// of them, lifted off them by half their spread -- as far as they face one
/// way: the middle of hits facing each other is already in the open. Facing
/// `back` when their normals cancel.
fn patch_of(hits: &[Hit], part: &[usize], back: Vec3) -> Option<Patch> {
    let (mut flux, mut weight, mut middle, mut facing) = (Vec3::ZERO, 0.0f32, Vec3::ZERO, Vec3::ZERO);
    for h in part.iter().map(|&i| &hits[i]) {
        flux += h.sent;
        weight += h.weight;
        middle += h.point * h.weight;
        facing += h.normal * h.weight;
    }
    if weight <= 0.0 {
        return None;
    }
    let middle = middle / weight;
    let directionality = (facing.length() / weight).min(1.0);
    let normal = facing.try_normalize().unwrap_or(back);
    let spread = (part.iter().map(|&i| hits[i].weight * hits[i].point.distance_squared(middle)).sum::<f32>() / weight).sqrt();
    let lift = (0.5 * spread).clamp(MIN_LIFT, MAX_LIFT) * directionality;
    Some(Patch { position: middle + normal * lift, normal, flux, spread, directionality })
}

/// The bounce from frame to frame: each patch followed over
/// [`SMOOTHING_SECONDS`]; one the beam has left faded out where it was, and
/// one it has newly found faded in where it is -- never slid between them
/// through the air.
#[derive(Default)]
pub(crate) struct Bounce {
    held: Vec<Patch>,
}

impl Bounce {
    /// This frame's patches, `dt` seconds after the last, from this instant's.
    ///
    /// Each of `now` follows the held patch nearest it, if one lies within
    /// reach of it -- half a metre and twice the wider spread -- nearest pairs
    /// first and each held patch once. A new beam starts at full strength.
    pub(crate) fn follow(&mut self, now: &[Patch], dt: f32) -> &[Patch] {
        let k = 1.0 - (-dt.max(0.0) / SMOOTHING_SECONDS).exp();
        let fresh = self.held.is_empty();
        let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
        for (n, p) in now.iter().enumerate() {
            for (h, q) in self.held.iter().enumerate() {
                let d = p.position.distance(q.position);
                if d <= 0.5 + 2.0 * p.spread.max(q.spread) {
                    pairs.push((d, n, h));
                }
            }
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (mut now_done, mut held_done) = (vec![false; now.len()], vec![false; self.held.len()]);
        let mut next: Vec<Patch> = Vec::with_capacity(MAX_HELD + now.len());
        for (_, n, h) in pairs {
            if now_done[n] || held_done[h] {
                continue;
            }
            (now_done[n], held_done[h]) = (true, true);
            let (p, q) = (now[n], self.held[h]);
            next.push(Patch {
                position: q.position.lerp(p.position, k),
                normal: q.normal.lerp(p.normal, k).try_normalize().unwrap_or(p.normal),
                flux: q.flux.lerp(p.flux, k),
                spread: q.spread + (p.spread - q.spread) * k,
                directionality: q.directionality + (p.directionality - q.directionality) * k,
            });
        }
        for (q, _) in self.held.iter().zip(&held_done).filter(|(_, &done)| !done) {
            let flux = q.flux * (1.0 - k);
            if flux.max_element() > 1e-4 {
                next.push(Patch { flux, ..*q });
            }
        }
        for (p, _) in now.iter().zip(&now_done).filter(|(_, &done)| !done) {
            next.push(Patch { flux: if fresh { p.flux } else { p.flux * k }, ..*p });
        }
        next.sort_by(|a, b| luminance(b.flux).total_cmp(&luminance(a.flux)));
        next.truncate(MAX_HELD);
        self.held = next;
        &self.held
    }

    /// The torch is off: nothing is lit, and the next beam starts afresh.
    pub(crate) fn clear(&mut self) {
        self.held.clear();
    }
}

/// The patch as the renderer's light, in the PLAYER's frame, like every light
/// `build_render_lists` hands over. `None` when it sends nothing back.
pub(crate) fn light(p: &Patch, offset: Vec3, yaw_inv: Quat) -> Option<Light> {
    let most = p.flux.max_element();
    if !(most > 0.0) {
        return None;
    }
    let c = p.flux / most;
    // The half space for a flat patch, opening to the whole sphere as the
    // hits' normals cancel. The smoothstep in the cosine from the cone's edge
    // to its axis sends `pi (1 - cos_outer)` times its peak in all.
    let half = 90.0 + 90.0 * (1.0 - p.directionality.clamp(0.0, 1.0));
    let cos_outer = half.to_radians().cos();
    Some(Light {
        position: yaw_inv * (p.position - offset),
        direction: yaw_inv * p.normal,
        kind: LightKind::Spot,
        color: Color3(srgb_byte(c.x), srgb_byte(c.y), srgb_byte(c.z), 255),
        // Straight out of the patch, Phi / pi for a flat one; see the module
        // notes.
        intensity: most / (std::f32::consts::PI * (1.0 - cos_outer)),
        range: BOUNCE_RANGE,
        cone_angle_deg: 2.0 * half,
        inner_cone_angle_deg: 0.0,
        mask_channel: None,
        // Past its range: no shadow (`Light::casts_shadow`).
        shadow_near: Some(f32::INFINITY),
        // AS WIDE AS THE PATCH: the disc whose hits spread as these do (a
        // uniform disc's RMS radius is its radius over root two). A point
        // here lit the stone beside it hundreds of times over -- a hot spot
        // by a corner's edge, low on a doorway's jamb (headset, 2026-10-02).
        source_radius: std::f32::consts::SQRT_2 * p.spread,
        in_level_bake: false,
    })
}

fn luminance(c: Vec3) -> f32 {
    c.dot(Vec3::new(0.2126, 0.7152, 0.0722))
}

/// Where a ray from `origin` along `dir` (unit) first meets one of the
/// characters' `capsules` within `max`: how far, and the capsule's outward
/// normal there. The capsules the ray starts inside are left out: the hand
/// holding the torch, round the glass. So a beam's rays stop at a hand held
/// up in front of it and bounce off the hand, as its shadow map stops the
/// light itself -- the wall in the hand's shadow sends nothing back.
pub(crate) fn capsule_hit(origin: Vec3, dir: Vec3, capsules: &[(Vec3, Vec3, f32)], max: f32) -> Option<(f32, Vec3)> {
    let mut best: Option<(f32, Vec3)> = None;
    for &(a, b, r) in capsules {
        let ab = b - a;
        let len2 = ab.length_squared();
        let axis_at = |p: Vec3| if len2 > 0.0 { a + ab * ((p - a).dot(ab) / len2).clamp(0.0, 1.0) } else { a };
        if (origin - axis_at(origin)).length_squared() <= r * r {
            continue;
        }
        // The capsule as a swept sphere: march the ray's nearest approach in
        // closed form against its cylinder, then its two end caps.
        let mut t_hit = f32::INFINITY;
        let ao = origin - a;
        if len2 > 0.0 {
            let (ab_d, ab_ao) = (ab.dot(dir), ab.dot(ao));
            let qa = len2 - ab_d * ab_d;
            let qb = len2 * ao.dot(dir) - ab_ao * ab_d;
            let qc = len2 * ao.dot(ao) - ab_ao * ab_ao - r * r * len2;
            let disc = qb * qb - qa * qc;
            if qa > 1e-12 && disc >= 0.0 {
                let t = (-qb - disc.sqrt()) / qa;
                let along = ab_ao + t * ab_d;
                if t >= 0.0 && along >= 0.0 && along <= len2 {
                    t_hit = t;
                }
            }
        }
        for c in [a, b] {
            let oc = origin - c;
            let (hb, hc) = (oc.dot(dir), oc.dot(oc) - r * r);
            let disc = hb * hb - hc;
            if disc >= 0.0 {
                let t = -hb - disc.sqrt();
                if t >= 0.0 && t < t_hit {
                    t_hit = t;
                }
            }
        }
        if t_hit <= max && best.is_none_or(|(t, _)| t_hit < t) {
            let p = origin + dir * t_hit;
            best = Some((t_hit, (p - axis_at(p)).normalize_or_zero()));
        }
    }
    best
}

/// A linear 0..1 channel as the sRGB byte `Color3` carries.
fn srgb_byte(linear: f32) -> u8 {
    let l = linear.clamp(0.0, 1.0);
    let s = if l <= 0.0031308 { l * 12.92 } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rays() -> Vec<BeamRay> {
        beam_rays(flashlight::CONE_DEG, flashlight::HOTSPOT_DEG)
    }

    /// The one patch a beam lights.
    fn only(ps: Vec<Patch>) -> Patch {
        assert_eq!(ps.len(), 1, "{ps:?}");
        ps[0]
    }

    /// A ray meets a capsule across it at its near side, an end cap head on,
    /// nothing beside it or past `max`, and nothing of the capsule it starts
    /// inside: the hand round the torch.
    #[test]
    fn a_ray_stops_at_a_hand_and_not_at_the_one_holding_the_torch() {
        let across = (Vec3::new(-0.1, 0.0, -1.0), Vec3::new(0.1, 0.0, -1.0), 0.04);
        let (t, n) = capsule_hit(Vec3::ZERO, Vec3::NEG_Z, &[across], 8.0).expect("across the ray");
        assert!((t - 0.96).abs() < 1e-4, "{t}");
        assert!((n - Vec3::Z).length() < 1e-4, "{n}");
        let end_on = (Vec3::new(0.0, 0.0, -2.0), Vec3::new(0.0, 0.0, -2.3), 0.05);
        let (t, n) = capsule_hit(Vec3::ZERO, Vec3::NEG_Z, &[end_on], 8.0).expect("its cap");
        assert!((t - 1.95).abs() < 1e-4 && (n - Vec3::Z).length() < 1e-4, "{t} {n}");
        assert!(capsule_hit(Vec3::ZERO, Vec3::NEG_Z, &[(Vec3::new(0.2, 0.0, -1.0), Vec3::new(0.3, 0.0, -1.0), 0.04)], 8.0).is_none());
        assert!(capsule_hit(Vec3::ZERO, Vec3::NEG_Z, &[across], 0.5).is_none(), "past max");
        let holding = (Vec3::new(0.0, 0.0, 0.1), Vec3::new(0.0, 0.0, -0.02), 0.04);
        assert_eq!(capsule_hit(Vec3::ZERO, Vec3::NEG_Z, &[holding], 8.0), None, "the hand round the glass");
        let (t, _) = capsule_hit(Vec3::ZERO, Vec3::NEG_Z, &[holding, across, end_on], 8.0).unwrap();
        assert!((t - 0.96).abs() < 1e-4, "the nearest: {t}");
    }

    /// A wall facing the glass `d` metres off along -Z, of albedo `a`.
    fn wall(d: f32, a: Vec3) -> impl Fn(Vec3, Vec3) -> Option<Landing> {
        move |o: Vec3, dir: Vec3| {
            (dir.z < 0.0).then(|| {
                let t = (o.z + d) / -dir.z;
                Landing { point: o + dir * t, normal: Vec3::Z, albedo: a }
            })
        }
    }

    /// The rays carry the whole beam: their shares add up to the cone's
    /// light, integrated finely over its falloff.
    #[test]
    fn the_rays_carry_the_whole_beam() {
        let outer = (0.5 * flashlight::CONE_DEG).to_radians();
        let (cos_outer, cos_inner) =
            (outer.cos(), (0.5 * flashlight::HOTSPOT_DEG).to_radians().cos());
        let n = 20_000;
        let whole: f32 = (0..n)
            .map(|i| {
                let theta = outer * (i as f32 + 0.5) / n as f32;
                let t = ((theta.cos() - cos_outer) / (cos_inner - cos_outer)).clamp(0.0, 1.0);
                t * t * (3.0 - 2.0 * t) * std::f32::consts::TAU * theta.sin() * outer / n as f32
            })
            .sum();
        let carried: f32 = rays().iter().map(|r| r.flux).sum();
        assert!((carried - whole).abs() < 0.002 * whole, "{carried} vs {whole}");
        assert_eq!(rays().len(), 19);
        assert!(rays().iter().all(|r| r.theta <= outer && r.flux > 0.0));
    }

    /// A white wall square to the beam sends back all the light that reached
    /// it, from a light in front of where the beam lands, facing out of the
    /// wall; a darker wall sends back its share, in its colour.
    #[test]
    fn a_wall_sends_back_its_share_of_the_beam_in_its_colour() {
        let lens = Vec3::new(0.0, 1.5, 0.0);
        let p = only(patches(lens, Quat::IDENTITY, &rays(), wall(2.0, Vec3::ONE)));
        let x = 2.0 / flashlight::RANGE;
        let window = (1.0 - x.powi(4)).powi(2);
        let beam: f32 = rays().iter().map(|r| r.flux).sum::<f32>() * flashlight::INTENSITY * window;
        assert!((p.flux - Vec3::splat(beam)).abs().max_element() < 1e-3 * beam, "{} vs {beam}", p.flux);
        assert!((p.normal - Vec3::Z).length() < 1e-5);
        assert!(p.position.z > -2.0 + MIN_LIFT - 1e-4 && p.position.z < -2.0 + MAX_LIFT + 1e-4, "{}", p.position);
        assert!((p.position.x).abs() < 1e-4 && (p.position.y - 1.5).abs() < 1e-4, "in front of the hot spot");
        let red = only(patches(lens, Quat::IDENTITY, &rays(), wall(2.0, Vec3::new(0.5, 0.1, 0.1))));
        assert!((red.flux.x - 0.5 * beam).abs() < 1e-3 * beam && (red.flux.y - 0.1 * beam).abs() < 1e-3 * beam);
        let l = light(&red, Vec3::ZERO, Quat::IDENTITY).unwrap();
        assert_eq!(l.color.0, 255);
        assert!(l.color.1 < 130 && l.color.1 == l.color.2, "{:?}", l.color);
        assert!((l.intensity - red.flux.x / std::f32::consts::PI).abs() < 1e-5);
    }

    /// The bounce is a diffuse patch: a spot facing out of the wall with a
    /// half-space cone, no shadow, short reach -- in the player's frame.
    #[test]
    fn the_bounce_is_a_half_space_spot_without_a_shadow() {
        let p = Patch {
            position: Vec3::new(1.0, 1.0, -3.0),
            normal: Vec3::Z,
            flux: Vec3::splat(3.0),
            spread: 0.3,
            directionality: 1.0,
        };
        let offset = Vec3::new(2.0, 0.0, 1.0);
        let yaw_inv = Quat::from_rotation_y(-0.5);
        let l = light(&p, offset, yaw_inv).unwrap();
        assert_eq!(l.kind, LightKind::Spot);
        assert!((l.position - yaw_inv * (p.position - offset)).length() < 1e-5);
        assert!((l.direction - yaw_inv * Vec3::Z).length() < 1e-5);
        assert_eq!((l.cone_angle_deg, l.inner_cone_angle_deg), (180.0, 0.0));
        let (cos_outer, cos_inner) = l.cone_cosines();
        assert!(cos_outer.abs() < 1e-6 && (cos_inner - 1.0).abs() < 1e-6, "falls as the cosine over the half space");
        assert!(!l.casts_shadow());
        assert_eq!(l.range, BOUNCE_RANGE);
        assert_eq!(l.color, Color3(255, 255, 255, 255));
        assert!((l.source_radius - 0.3 * std::f32::consts::SQRT_2).abs() < 1e-6, "as wide as the patch: {}", l.source_radius);
        assert!(light(&Patch { flux: Vec3::ZERO, ..p }, offset, yaw_inv).is_none());
    }

    /// Hits facing several ways open the cone toward the whole sphere, and
    /// whatever its width the light sends out the patch's whole flux.
    #[test]
    fn a_patch_facing_several_ways_lights_round_more_of_the_sphere() {
        let sent = |l: &Light| {
            let (cos_outer, cos_inner) = l.cone_cosines();
            let n = 20_000;
            (0..n)
                .map(|i| {
                    let mu = -1.0 + 2.0 * (i as f32 + 0.5) / n as f32;
                    let t = ((mu - cos_outer) / (cos_inner - cos_outer)).clamp(0.0, 1.0);
                    t * t * (3.0 - 2.0 * t) * std::f32::consts::TAU * 2.0 / n as f32
                })
                .sum::<f32>()
                * l.intensity
        };
        let mut widths = Vec::new();
        for d in [1.0, 0.7, 0.3, 0.0] {
            let p = Patch { directionality: d, ..at(0.0, 2.0) };
            let l = light(&p, Vec3::ZERO, Quat::IDENTITY).unwrap();
            assert!((sent(&l) - 2.0).abs() < 2e-3, "directionality {d}: sends {}", sent(&l));
            widths.push(l.cone_angle_deg);
        }
        assert_eq!((widths[0], widths[3]), (180.0, 360.0));
        assert!(widths.windows(2).all(|w| w[0] < w[1]), "{widths:?}");
        // Down a hallway between walls at x = -1 and 1: each ring meets both
        // walls, at its own depth. Every patch stays between the walls,
        // unlifted, its light round the whole sphere.
        let hallway = |o: Vec3, d: Vec3| {
            (d.x != 0.0).then(|| {
                let t = if d.x > 0.0 { (1.0 - o.x) / d.x } else { (o.x + 1.0) / -d.x };
                Landing { point: o + d * t, normal: Vec3::new(-d.x.signum(), 0.0, 0.0), albedo: Vec3::splat(0.5) }
            })
        };
        let ps = patches(Vec3::ZERO, Quat::IDENTITY, &rays(), hallway);
        assert!(!ps.is_empty());
        for p in &ps {
            assert!(p.directionality < 0.2 && p.position.x.abs() < 0.3, "between the walls: {p:?}");
            assert!(light(p, Vec3::ZERO, Quat::IDENTITY).unwrap().cone_angle_deg > 320.0);
        }
    }

    /// A beam into nothing lights nothing; a beam into a corner lights from
    /// between the two walls, facing out of both.
    #[test]
    fn a_beam_into_a_corner_lights_from_between_its_walls() {
        assert!(patches(Vec3::ZERO, Quat::IDENTITY, &rays(), |_, _| None).is_empty());
        // Aimed into the corner of a wall at x = 1 and one at z = -1, rolled a
        // little about its axis so no ray but the centre's runs exactly along
        // the corner's plane, where the two walls tie.
        let rotation = Quat::from_rotation_y(-std::f32::consts::FRAC_PI_4) * Quat::from_rotation_z(0.1);
        let corner = |o: Vec3, d: Vec3| {
            let tx = if d.x > 0.0 { (1.0 - o.x) / d.x } else { f32::MAX };
            let tz = if d.z < 0.0 { (o.z + 1.0) / -d.z } else { f32::MAX };
            let (t, n) = if tx < tz { (tx, Vec3::NEG_X) } else { (tz, Vec3::Z) };
            (t < f32::MAX).then(|| Landing { point: o + d * t, normal: n, albedo: Vec3::splat(0.5) })
        };
        let p = only(patches(Vec3::ZERO, rotation, &rays(), corner));
        let between = Vec3::new(-1.0, 0.0, 1.0).normalize();
        assert!(p.normal.dot(between) > 0.99, "{}", p.normal);
        assert!(p.position.x < 1.0 && p.position.z > -1.0, "in the room: {}", p.position);
    }

    /// A beam past a pillar's edge lights from the pillar and from the wall
    /// behind it -- each patch where its own hits are, neither in the air
    /// between -- carrying the beam's light between them.
    #[test]
    fn a_beam_past_an_edge_lights_from_both_surfaces_not_the_air_between() {
        // A pillar's face 2 m off covering the left half of the beam, and a
        // wall 10 m off behind it.
        let edge = |o: Vec3, d: Vec3| {
            if d.z >= 0.0 {
                return None;
            }
            let t = (o.z + 2.0) / -d.z;
            let (t, n) = if (o + d * t).x < 0.0 { (t, Vec3::Z) } else { ((o.z + 10.0) / -d.z, Vec3::Z) };
            Some(Landing { point: o + d * t, normal: n, albedo: Vec3::splat(0.5) })
        };
        let lens = Vec3::new(0.0, 1.5, 0.0);
        let ps = patches(lens, Quat::IDENTITY, &rays(), edge);
        assert_eq!(ps.len(), 2, "{ps:?}");
        let (near, far) = if ps[0].position.z > ps[1].position.z { (ps[0], ps[1]) } else { (ps[1], ps[0]) };
        assert!(near.position.z > -2.0 && near.position.z < -2.0 + MAX_LIFT + 1e-4, "{}", near.position);
        assert!(near.position.x < 0.0, "on the pillar: {}", near.position);
        assert!(far.position.z > -10.0 && far.position.z < -10.0 + MAX_LIFT + 1e-4, "{}", far.position);
        let whole = only(patches(lens, Quat::IDENTITY, &rays(), wall(2.0, Vec3::splat(0.5))));
        // The far half arrives through the beam's window at 10 m.
        assert!(near.flux.x + far.flux.x < whole.flux.x && near.flux.x + far.flux.x > 0.7 * whole.flux.x);
    }

    /// THE SURFACES A BEAM LIGHTS: a wall square to it is one, on the wall's
    /// plane, in the wall's colour, its pool map seen from the glass along the
    /// beam and reaching past the cone's edge; a pillar's face in front of a
    /// wall is a second surface, not the same one.
    #[test]
    fn the_beam_names_the_surfaces_it_lights_in_their_colours() {
        let lens = Vec3::new(0.0, 1.5, 0.0);
        let red = Vec3::new(0.5, 0.1, 0.1);
        let turn = Quat::from_rotation_z(0.3);
        let one = landings(lens, turn, &rays(), wall(2.0, red)).surfaces();
        assert_eq!(one.len(), 1, "{one:?}");
        let s = one[0];
        assert!((s.normal - Vec3::Z).length() < 1e-5 && (s.offset + 2.0).abs() < 1e-4, "{s:?}");
        assert!((s.albedo - red).abs().max_element() < 1e-5, "{:?}", s.albedo);
        // The map's frame: the glass, the beam's axis, and across it as the
        // torch is turned -- square to the axis, so the shader's two halves of
        // the map agree.
        assert_eq!(s.lens, lens);
        assert!((s.forward - Vec3::NEG_Z).length() < 1e-6, "{}", s.forward);
        assert!((s.right - turn * Vec3::X).length() < 1e-6 && s.right.dot(s.forward).abs() < 1e-6, "{}", s.right);
        assert!(pool_map_tan() > (0.5 * flashlight::CONE_DEG).to_radians().tan(), "past the cone's edge");
        // A pillar's face 2 m off over the beam's left half, a wall 6 m off
        // behind it.
        let pillar = |o: Vec3, d: Vec3| {
            if d.z >= 0.0 {
                return None;
            }
            let t = (o.z + 2.0) / -d.z;
            let t = if (o + d * t).x < 0.0 { t } else { (o.z + 6.0) / -d.z };
            Some(Landing { point: o + d * t, normal: Vec3::Z, albedo: Vec3::splat(0.5) })
        };
        let two = landings(lens, Quat::IDENTITY, &rays(), pillar).surfaces();
        assert_eq!(two.len(), 2, "{two:?}");
        let mut offsets: Vec<f32> = two.iter().map(|s| s.offset).collect();
        offsets.sort_by(f32::total_cmp);
        assert!((offsets[0] + 6.0).abs() < 1e-4 && (offsets[1] + 2.0).abs() < 1e-4, "{offsets:?}");
        assert!(two[0].weight >= two[1].weight, "brightest first");
        assert!(landings(lens, Quat::IDENTITY, &rays(), |_, _| None).surfaces().is_empty());
    }

    /// A surface the beam's rays leave is held a moment, then let go; found
    /// again it takes its new beam, and is not named twice.
    #[test]
    fn a_surface_the_beam_leaves_is_held_a_moment_then_let_go() {
        let wall = LitPlane {
            normal: Vec3::Z,
            offset: -2.0,
            albedo: Vec3::ONE,
            weight: 1.0,
            lens: Vec3::new(0.0, 1.5, 0.0),
            forward: Vec3::NEG_Z,
            right: Vec3::X,
        };
        let floor = LitPlane { normal: Vec3::Y, offset: 0.0, weight: 0.5, ..wall };
        let mut held = HeldSurfaces::default();
        assert_eq!(held.follow(&[wall, floor], 0.016), vec![wall, floor]);
        assert_eq!(held.follow(&[wall], 0.5), vec![wall, floor], "held");
        assert_eq!(held.follow(&[wall], 0.6), vec![wall], "let go past the hold");
        let moved = LitPlane { offset: -2.01, lens: Vec3::new(0.3, 1.5, 0.0), ..wall };
        assert_eq!(held.follow(&[moved], 0.016), vec![moved], "found again: replaced, not doubled");
        let many: Vec<LitPlane> = (0..6).map(|k| LitPlane { offset: -2.0 - k as f32, weight: k as f32, ..wall }).collect();
        let kept = held.follow(&many, 0.016);
        assert_eq!(kept.len(), MAX_SURFACES);
        assert_eq!(kept[0].weight, 5.0, "the freshest, brightest first");
    }

    /// A lit surface goes to the player's frame as the lights do: a point on
    /// the wall in the world lies on its plane as the shaders see it, and its
    /// map's glass and axes turn with it.
    #[test]
    fn a_lit_surface_goes_to_the_players_frame_as_the_lights_do() {
        let s = LitPlane {
            normal: Vec3::X,
            offset: 3.0,
            albedo: Vec3::splat(0.4),
            weight: 1.0,
            lens: Vec3::new(0.5, 1.4, -1.0),
            forward: Vec3::X,
            right: Vec3::Z,
        };
        let (offset, yaw_inv) = (Vec3::new(1.0, 0.5, -4.0), Quat::from_rotation_y(0.7));
        let p = lit_surface(&s, offset, yaw_inv);
        let on = Vec3::new(3.0, 2.2, -0.5);
        assert!((p.normal.dot(yaw_inv * (on - offset)) - p.offset).abs() < 1e-5);
        assert!((p.lens - yaw_inv * (s.lens - offset)).length() < 1e-6);
        assert!((p.forward - yaw_inv * s.forward).length() < 1e-6 && (p.right - yaw_inv * s.right).length() < 1e-6);
        assert_eq!((p.albedo, p.tan_half), (s.albedo, pool_map_tan()));
    }

    /// A ray past an edge takes no light of its own, nor pulls the rest's off
    /// their wall.
    #[test]
    fn a_sliver_past_an_edge_stays_with_its_wall() {
        let rs = rays();
        let stray = rs[rs.len() - 1];
        let dir_of = |r: &BeamRay| {
            Vec3::new(r.theta.sin() * r.phi.cos(), r.theta.sin() * r.phi.sin(), -r.theta.cos()).normalize()
        };
        let past = dir_of(&stray);
        let sliver = |o: Vec3, d: Vec3| {
            let far = d.dot(past) > 0.9999;
            let z = if far { 10.0 } else { 2.0 };
            (d.z < 0.0).then(|| Landing { point: o + d * ((o.z + z) / -d.z), normal: Vec3::Z, albedo: Vec3::splat(0.5) })
        };
        let p = only(patches(Vec3::ZERO, Quat::IDENTITY, &rs, sliver));
        assert!(p.position.z > -2.0 && p.position.z < -2.0 + MAX_LIFT + 1e-4, "{}", p.position);
        // Its spread and its light are the wall's alone: the stray, 8 m
        // behind, would have stretched the one to over a metre.
        assert!(p.spread < 0.6, "{}", p.spread);
        let mut rest = rs.clone();
        rest.pop();
        let wall_only = only(patches(Vec3::ZERO, Quat::IDENTITY, &rest, wall(2.0, Vec3::splat(0.5))));
        assert!((p.flux - wall_only.flux).abs().max_element() < 1e-4 * wall_only.flux.x, "{} vs {}", p.flux, wall_only.flux);
    }

    fn at(x: f32, flux: f32) -> Patch {
        Patch { position: Vec3::new(x, 0.0, 0.0), normal: Vec3::Z, flux: Vec3::splat(flux), spread: 0.2, directionality: 1.0 }
    }

    /// A still beam's bounce stays put; a moved one slides there over a few
    /// frames rather than jumping; a beam that leaves everything fades out.
    #[test]
    fn the_bounce_follows_the_beam_smoothly() {
        let (a, b) = (at(0.0, 1.0), Patch { normal: Vec3::Y, ..at(0.5, 2.0) });
        let mut bounce = Bounce::default();
        assert_eq!(bounce.follow(&[a], 0.014), &[a], "a new beam starts where it lands");
        assert_eq!(bounce.follow(&[a], 0.014), &[a]);
        let first = bounce.follow(&[b], 0.014)[0];
        assert!(first.position.x > 0.02 && first.position.x < 0.15, "{}", first.position);
        assert!((first.normal.length() - 1.0).abs() < 1e-5);
        let mut last = first;
        for _ in 0..60 {
            last = bounce.follow(&[b], 0.014)[0];
        }
        assert!((last.position - b.position).length() < 1e-3 && (last.flux - b.flux).length() < 1e-3);
        let mut fading = bounce.follow(&[], 0.014)[0];
        assert!(fading.flux.x < b.flux.x && fading.position == last.position, "fades where it was");
        for _ in 0..200 {
            match bounce.follow(&[], 0.014).first() {
                Some(&p) => fading = p,
                None => break,
            }
        }
        assert!(bounce.follow(&[], 0.014).is_empty(), "faded out: {}", fading.flux);
        bounce.clear();
        assert_eq!(bounce.follow(&[b], 0.014), &[b]);
    }

    /// A patch the beam jumps away from -- past an edge, to a wall metres
    /// behind -- fades out where it was while the new one fades in where it
    /// is: no light ever stands in the air between them.
    #[test]
    fn a_jump_fades_across_and_never_slides_through_the_air() {
        let (near, far) = (at(0.0, 1.0), at(8.0, 1.0));
        let mut bounce = Bounce::default();
        bounce.follow(&[near], 0.014);
        let mut was_near = 1.0f32;
        for frame in 0..100 {
            let held = bounce.follow(&[far], 0.014).to_vec();
            for p in &held {
                assert!(p.position.x == 0.0 || p.position.x == 8.0, "frame {frame}: in the air at {}", p.position);
            }
            let near_now = held.iter().find(|p| p.position.x == 0.0).map_or(0.0, |p| p.flux.x);
            assert!(near_now <= was_near, "the old patch only fades");
            was_near = near_now;
            let far_now = held.iter().find(|p| p.position.x == 8.0).map_or(0.0, |p| p.flux.x);
            if frame == 0 {
                assert!(far_now > 0.0 && far_now < 0.3, "the new one fades in: {far_now}");
            }
        }
        assert_eq!(bounce.follow(&[far], 0.014).len(), 1);
        // Two patches held at once are followed each by its own.
        let (l, r) = (at(-3.0, 1.0), at(3.0, 0.5));
        let mut two = Bounce::default();
        two.follow(&[l, r], 0.014);
        let moved = two.follow(&[at(3.1, 0.5), at(-3.1, 1.0)], 0.014).to_vec();
        assert_eq!(moved.len(), 2);
        assert!(moved.iter().any(|p| p.position.x < -3.0 && p.position.x > -3.1 && p.flux.x == 1.0), "{moved:?}");
        assert!(moved.iter().any(|p| p.position.x > 3.0 && p.position.x < 3.1 && p.flux.x == 0.5), "{moved:?}");
    }

    /// DIAGNOSTIC: the bounce at the bench's flashlight views of test_room, cast
    /// as the frame casts it -- brushes for their material, the physics scene
    /// for anything nearer -- each ray's landing, the patch, its light, and the
    /// light it and the beam put on a few surfaces round the player.
    #[test]
    #[ignore]
    fn print_the_bench_bounces() {
        use crate::brush_render::{load_materials, mean_albedo, BrushGeometry};
        let game = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../game");
        let scene = space_soup_engine::Scene::load(&space_soup_engine::Manifest::scene_path(&game, "test_room")).unwrap();
        let brushes = BrushGeometry::load(&scene);
        let albedos: Vec<Vec3> = load_materials(&game, brushes.materials()).colours.iter().map(mean_albedo).collect();
        let mut physics = space_soup_engine::rigid_physics::PhysicsWorld::new();
        physics.rebuild(&scene, &game);
        let live = crate::scene_lights::load(&game, "test_room");
        println!(
            "live lights {} (+ {} baked): {:?}",
            live.len(),
            crate::scene_lights::load_baked(&game, "test_room").len(),
            live.iter().map(|l| (l.id.as_str(), l.intensity, l.range)).collect::<Vec<_>>()
        );
        for (name, eye, glass, aim) in [
            ("torch_pillar", Vec3::new(0.3, 1.6, -3.0), Vec3::new(0.55, 1.25, -3.2), Vec3::new(0.0, 1.0, -6.4)),
            ("torch_floor", Vec3::new(0.0, 1.6, -1.0), Vec3::new(0.25, 1.2, -1.2), Vec3::new(0.0, 0.0, -2.6)),
            ("torch_hallway", Vec3::new(3.4, 1.6, -3.0), Vec3::new(3.65, 1.25, -2.8), Vec3::new(8.5, 1.0, -3.2)),
            ("torch_in_hand", Vec3::new(0.0, 1.6, -1.0), Vec3::new(0.2, 1.25, -1.45), Vec3::new(0.0, 0.3, -4.0)),
            ("torch_wall", Vec3::new(0.2, 1.6, -9.6), Vec3::new(0.0, 1.25, -9.85), Vec3::new(-2.7, 1.1, -10.4)),
        ] {
            let (lens, rotation) = flashlight::lens_from_bench(glass, aim);
            println!("== {name}: lens {lens:.3} forward {:.3}", rotation * Vec3::NEG_Z);
            let land = |origin: Vec3, dir: Vec3| -> Option<Landing> {
                let brush = brushes.cast(origin, dir, flashlight::RANGE, &[]);
                let solid = physics.raycast(origin, dir, flashlight::RANGE);
                println!(
                    "   ray {dir:.3}: brush {} | physics {}",
                    brush.as_ref().map_or("-".into(), |b| format!(
                        "{:.3} m mat {} ({}) n {:.2} albedo {:.3}",
                        b.distance,
                        b.material,
                        brushes.materials().get(b.material as usize).map_or("?", |s| s.as_str()),
                        b.normal,
                        albedos.get(b.material as usize).copied().unwrap_or(Vec3::ONE) * b.tint
                    )),
                    solid.map_or("-".into(), |(p, n)| format!("{:.3} m n {n:.2}", (p - origin).length())),
                );
                let other = |(point, normal): (Vec3, Vec3)| Landing {
                    point,
                    normal: if normal.dot(dir) > 0.0 { -normal } else { normal },
                    albedo: Vec3::splat(UNKNOWN_ALBEDO),
                };
                match (brush, solid) {
                    (Some(b), Some(s)) if (s.0 - origin).length() + 0.05 < b.distance => Some(other(s)),
                    (Some(b), _) => Some(Landing {
                        point: b.point,
                        normal: b.normal,
                        albedo: albedos.get(b.material as usize).copied().unwrap_or(Vec3::ONE) * b.tint,
                    }),
                    (None, s) => s.map(other),
                }
            };
            let ps = patches(lens, rotation, &rays(), land);
            // What the casts cost the frame, quiet: brushes and physics, every ray.
            let quiet = |origin: Vec3, dir: Vec3| -> Option<Landing> {
                let b = brushes.cast(origin, dir, flashlight::RANGE, &[])?;
                let _ = physics.raycast(origin, dir, flashlight::RANGE);
                Some(Landing { point: b.point, normal: b.normal, albedo: Vec3::splat(0.3) })
            };
            let started = std::time::Instant::now();
            for _ in 0..200 {
                std::hint::black_box(patches(lens, rotation, &rays(), quiet));
            }
            println!("   patches(): {:.1} us a frame", started.elapsed().as_secs_f64() * 1e6 / 200.0);
            for p in &ps {
                println!(
                    "   PATCH at {:.3} ({:.2} m from the lens) facing {:.2}, spread {:.2} m, flux {:.3}",
                    p.position,
                    (p.position - lens).length(),
                    p.normal,
                    p.spread,
                    p.flux
                );
            }
            let Some(&p) = ps.first() else {
                println!("   NO PATCH");
                continue;
            };
            let l = light(&p, Vec3::ZERO, Quat::IDENTITY).unwrap();
            println!(
                "   patch at {:.3} ({:.2} m from the lens) facing {:.2}, flux {:.3}; light {:.3} x {:?}",
                p.position,
                (p.position - lens).length(),
                p.normal,
                p.flux,
                l.intensity,
                l.color
            );
            // What the bounce puts on the floor and the walls round the
            // player, as the shader's diffuse has it (window, inverse square,
            // the half-space cone, N.L), beside the beam's own hot spot.
            let bounce_at = |at: Vec3, n: Vec3| {
                let to = l.position - at;
                let d = to.length();
                let dir = to / d;
                let x = d / l.range;
                let window = (1.0 - x.powi(4)).clamp(0.0, 1.0).powi(2);
                let (co, ci) = l.cone_cosines();
                let t = ((-dir.dot(l.direction) - co) / (ci - co)).clamp(0.0, 1.0);
                l.intensity * window / (d * d).max(0.01) * t * t * (3.0 - 2.0 * t) * n.dot(dir).max(0.0)
            };
            // The frame's ranking by `influence_score`, from the rig's origin
            // under the eye: the sky's sun first, then what fits of the rest.
            let origin = Vec3::new(eye.x, 0.0, eye.z);
            let mut ranked: Vec<(String, f32)> = live
                .iter()
                .map(|w| {
                    let ll = crate::convert::to_space_soup_light(w, origin, Quat::IDENTITY);
                    (w.id.clone(), space_soup::renderer::lights::influence_score(&ll))
                })
                .collect();
            let beam = flashlight::beam(lens, rotation * Vec3::NEG_Z, origin, Quat::IDENTITY);
            ranked.push(("FLASHLIGHT".into(), space_soup::renderer::lights::influence_score(&beam)));
            let bl = light(&p, origin, Quat::IDENTITY).unwrap();
            ranked.push(("BOUNCE".into(), space_soup::renderer::lights::influence_score(&bl)));
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let fits = space_soup::renderer::lights::MAX_LIGHTS - 1;
            println!(
                "   kept (with the sun): {:?}\n   DROPPED: {:?}",
                ranked.iter().take(fits).map(|(n, s)| format!("{n} {s:.3}")).collect::<Vec<_>>(),
                ranked.iter().skip(fits).map(|(n, s)| format!("{n} {s:.3}")).collect::<Vec<_>>()
            );
            let hot = brushes.cast(lens, rotation * Vec3::NEG_Z, flashlight::RANGE, &[]).map(|h| h.distance);
            println!(
                "   beam hot spot {:?} m: irradiance {:?}",
                hot,
                hot.map(|d| flashlight::INTENSITY / (d * d))
            );
            for (what, at, n) in [
                ("floor under the eye", Vec3::new(eye.x, 0.0, eye.z), Vec3::Y),
                ("floor 1 m ahead", Vec3::new(eye.x, 0.0, eye.z) + (lens - eye).with_y(0.0).normalize_or_zero() * 1.0, Vec3::Y),
                ("the hand", glass, -(rotation * Vec3::NEG_Z)),
            ] {
                println!("   bounce on {what}: {:.4}", bounce_at(at, n));
            }
        }
    }
}
