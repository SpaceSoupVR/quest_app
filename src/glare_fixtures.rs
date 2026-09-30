//! WHICH LAMPS GLARE, AND FROM WHICH SIDES, for the veil the renderer draws
//! round each one (`space_soup::renderer::glare`).
//!
//! Only a light with a fixture -- an object with a model -- glares: a bare
//! light is an author's fill with no bulb anywhere in the scene, and a veil
//! round empty air would read as a smudge, not a lamp.
//!
//! FROM ITS CARDS. A wall sconce's bulb hangs in a shade that opens downward:
//! from below the bulb and the shade's lit inside are in plain view, from above
//! and from the wall only the shade's dark outside is. The fixture's reflection
//! cards already know this -- six orthographic pictures of it, its bulb glowing
//! at the radiance of its light (`scene_light::emissive_drive`) -- so the light
//! each card's bright texels send toward its side, over what a bare lamp of the
//! same intensity would send, is how much of the source shows from that side.
//! What they send is the eye's own measure: a patch of displayed value `L` and
//! projected area `A` lights a card facing it at distance `d` to `L A / (pi d^2)`
//! in the renderer's units, where a lamp of intensity `I` lights it to `I / d^2`.
//!
//! FROM THE AUTHOR where the cards do not carry the fixture's glow -- a bulb lit
//! by an emissive TEXTURE, like the hanging lamps', is not on the cards yet --
//! or the fixture has no cards: the light's "glare visible from" faces, around
//! its own direction exactly as the editor draws them.

use std::collections::HashMap;

use glam::{Mat3, Quat, Vec3};
use space_soup::renderer::glare::GlareSource;
use space_soup_engine::reflection_cards::{LoadedCards, CARD_FACES};
use space_soup_engine::scene::GameObject;
use space_soup_engine::scene_light::{GlareFacesDef, LightDef};
use space_soup_engine::LightKind;
use space_soup_protocol::{WireLightKind, WireRenderLight};

/// A displayed value this many times the renderer's white is past anything the
/// display shows at any exposure the eye adapts to; only light beyond it
/// counts toward glare. The rest -- a shade's outside, lit like a wall -- the
/// display draws, and the eye's own veil already answers it.
const BRIGHT: f32 = 4.0;

/// Cards whose brightest side sends less than this share of a bare lamp's light
/// are not carrying the fixture's glow (see the module notes); the author's
/// faces decide instead.
const CARRIES_GLOW: f32 = 0.02;

/// The most a side may send, as a share of a bare lamp: a shade's lit inside
/// adds to its bulb from below, and a bad bake must not light a veil the size
/// of the room.
const MAX_SIDE: f32 = 4.0;

const LUMA: Vec3 = Vec3::new(0.2126, 0.7152, 0.0722);

/// How one lamp glares: the share of its light that shows from each side of
/// `frame` -- +x, -x, +y, -y, +z, -z -- in the world. `frame` is `None` for
/// sides around the light's own direction, taken every frame (the author's
/// faces; see [`light_frame`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FixtureGlare {
    pub sides: [f32; 6],
    pub frame: Option<Quat>,
}

/// Each side's share of a bare lamp's light, from a fixture's cards: card `k`
/// looks in along side `k` (see `reflection_cards`), so what it saw is what an
/// eye on that side sees. `half_size` is the box the cards were taken through;
/// `luminance` the light's -- its linear colour's luminance times its
/// intensity. `None` when the cards carry no glow.
pub(crate) fn card_sides(cards: &LoadedCards, half_size: Vec3, luminance: f32) -> Option<[f32; 6]> {
    let res = cards.resolution as usize;
    if luminance <= 0.0 || res == 0 || cards.texels.len() < CARD_FACES * res * res {
        return None;
    }
    let mut sides = [0.0f32; 6];
    for (k, side) in sides.iter_mut().enumerate() {
        let a = k / 2;
        let (u, v) = ((a + 1) % 3, (a + 2) % 3);
        let texel_area = (2.0 * half_size[u] / res as f32) * (2.0 * half_size[v] / res as f32);
        let excess: f32 = cards.texels[k * res * res..(k + 1) * res * res]
            .iter()
            .map(|t| (Vec3::new(t[0], t[1], t[2]).dot(LUMA) - BRIGHT).max(0.0))
            .sum();
        *side = (excess * texel_area / (std::f32::consts::PI * luminance)).min(MAX_SIDE);
    }
    (sides.iter().copied().fold(0.0, f32::max) >= CARRIES_GLOW).then_some(sides)
}

/// The author's faces as sides of [`light_frame`]: right and left along +x and
/// -x, top and bottom along +y and -y, back and front along +z and -z.
pub(crate) fn authored_sides(f: &GlareFacesDef) -> [f32; 6] {
    let on = |b: bool| if b { 1.0 } else { 0.0 };
    [on(f.right), on(f.left), on(f.top), on(f.bottom), on(f.back), on(f.front)]
}

/// The frame the editor puts a light's glare faces in (`objectRenderer.js`):
/// front along the light's direction, right across it and the world's up
/// (its right when the light points straight up or down), top over both --
/// as a rotation whose +x is right, +y top and -z front.
pub(crate) fn light_frame(direction: Vec3) -> Quat {
    let front = direction.try_normalize().unwrap_or(Vec3::NEG_Z);
    let world_up = if front.y.abs() < 0.99 { Vec3::Y } else { Vec3::X };
    let right = front.cross(world_up).normalize();
    let top = right.cross(front).normalize();
    Quat::from_mat3(&Mat3::from_cols(right, top, -front)).normalize()
}

/// The linear luminance of a light's colour times its intensity.
fn light_luminance(l: &LightDef) -> f32 {
    let c = space_soup::renderer::Color3(l.color.0, l.color.1, l.color.2, l.color.3).to_linear();
    Vec3::new(c[0], c[1], c[2]).dot(LUMA) * l.intensity
}

/// Every lamp that glares, by render light id (`object#index`, as
/// `scene_lights` names them): each point and spot light of an object with a
/// model. `card_sides` holds what [`card_sides`] measured, by object id with
/// the frame its cards were taken in; a fixture without them takes its
/// light's authored faces.
pub(crate) fn fixtures(objects: &[GameObject], measured: &HashMap<String, ([f32; 6], Quat)>) -> HashMap<String, FixtureGlare> {
    let mut out = HashMap::new();
    for o in objects.iter().filter(|o| o.mesh.is_some()) {
        for (i, l) in o.lights.iter().enumerate() {
            if !matches!(l.kind, LightKind::Point | LightKind::Spot) {
                continue;
            }
            let glare = match measured.get(&o.id) {
                Some((sides, frame)) => FixtureGlare { sides: *sides, frame: Some(*frame) },
                None => FixtureGlare { sides: authored_sides(&l.glare_faces), frame: None },
            };
            out.insert(format!("{}#{i}", o.id), glare);
        }
    }
    out
}

/// [`fixtures`] for a scene file, its transforms resolved to world space as
/// the proxies' were. Empty when the scene does not load.
pub(crate) fn load(game_dir: &std::path::Path, scene_name: &str, measured: &HashMap<String, ([f32; 6], Quat)>) -> HashMap<String, FixtureGlare> {
    let path = space_soup_engine::Manifest::scene_path(game_dir, scene_name);
    match space_soup_engine::scene::Scene::load(&path) {
        Ok(mut s) => {
            s.resolve_world_transforms();
            let out = fixtures(&s.objects, measured);
            log::info!("glare: {} lamp(s) with a fixture, {} measured by their cards", out.len(), measured.len());
            out
        }
        Err(e) => {
            log::warn!("glare: {} did not load: {e:#}", path.display());
            HashMap::new()
        }
    }
}

/// What [`card_sides`] measures for every model on cards, by object id, from
/// the proxies' boxes: the light it is measured against is the brightest of
/// the object's lamps, as the bake drove its glow by the brightest
/// (`emissive_drive`).
pub(crate) fn measure(
    objects: &[GameObject],
    cards: &[(usize, &LoadedCards, Vec3, Quat)],
) -> HashMap<String, ([f32; 6], Quat)> {
    let mut out = HashMap::new();
    for &(object, c, half_size, rotation) in cards {
        let Some(o) = objects.get(object) else { continue };
        let luminance = o
            .lights
            .iter()
            .filter(|l| l.enabled && matches!(l.kind, LightKind::Point | LightKind::Spot))
            .map(light_luminance)
            .fold(0.0, f32::max);
        match card_sides(c, half_size, luminance) {
            Some(sides) => {
                log::info!(
                    "glare: '{}' shows its light {:?} (+x -x +y -y +z -z) by its cards",
                    o.id,
                    sides.map(|s| (s * 100.0).round() / 100.0),
                );
                out.insert(o.id.clone(), (sides, rotation));
            }
            None if luminance > 0.0 => {
                log::info!("glare: '{}' cards carry no glow; its lights' authored faces decide", o.id)
            }
            None => {}
        }
    }
    out
}

/// This frame's glare sources: every lamp in `lights` that [`fixtures`] names,
/// in the player's frame (as `convert::to_space_soup_light` puts lights).
pub(crate) fn sources<'a>(
    lights: impl IntoIterator<Item = &'a WireRenderLight>,
    fixtures: &HashMap<String, FixtureGlare>,
    offset: Vec3,
    yaw_inv: Quat,
) -> Vec<GlareSource> {
    lights
        .into_iter()
        .filter_map(|l| {
            let f = fixtures.get(&l.id)?;
            let direction = Vec3::from(l.direction);
            let c = space_soup::renderer::Color3(l.color.0, l.color.1, l.color.2, l.color.3).to_linear();
            let cone = match l.kind {
                WireLightKind::Spot => Some((
                    yaw_inv * direction.normalize_or_zero(),
                    (l.cone_angle_deg.to_radians() * 0.5).cos(),
                    (l.inner_cone_angle_deg.to_radians() * 0.5).cos(),
                )),
                WireLightKind::Point => None,
                WireLightKind::Directional => return None,
            };
            Some(GlareSource {
                position: yaw_inv * (Vec3::from(l.position) - offset),
                radiance: Vec3::new(c[0], c[1], c[2]) * l.intensity,
                sides: f.sides,
                rotation: yaw_inv * f.frame.unwrap_or_else(|| light_frame(direction)),
                cone,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn blank_cards(res: u32) -> LoadedCards {
        let n = CARD_FACES * (res * res) as usize;
        LoadedCards {
            object_id: "lamp".into(),
            resolution: res,
            texels: vec![[0.0, 0.0, 0.0, 2.0]; n],
            normals: vec![[0.0; 3]; n],
        }
    }

    /// A bulb's disc on one card is that side's share: radiance `I / R^2` over
    /// a disc of radius `R` sends exactly a bare lamp's light; the other sides,
    /// which saw only a dim shade, send none.
    #[test]
    fn a_bulb_seen_whole_from_below_sends_a_bare_lamps_light_that_way() {
        const RES: u32 = 64;
        let (half, radius, intensity) = (0.2f32, 0.05f32, 3.0f32);
        let mut c = blank_cards(RES);
        let texel = 2.0 * half / RES as f32;
        let below = 3; // looks in through -y: what an eye below sees
        let mut lit_texels = 0;
        for y in 0..RES {
            for x in 0..RES {
                let at = |i: u32| (i as f32 + 0.5) * texel - half;
                let lit = at(x).hypot(at(y)) <= radius;
                lit_texels += usize::from(lit);
                let value = if lit { intensity / (radius * radius) } else { 0.5 };
                c.texels[below * (RES * RES) as usize + (y * RES + x) as usize] = [value, value, value, 0.5];
            }
        }
        let sides = card_sides(&c, Vec3::splat(half), intensity).expect("carries its glow");
        // The disc as the texels drew it, its light past `BRIGHT`.
        let drawn = (intensity / (radius * radius) - BRIGHT) * lit_texels as f32 * texel * texel;
        let expected = drawn / (std::f32::consts::PI * intensity);
        assert!((sides[below] - expected).abs() < 1e-3 * expected, "{sides:?} vs {expected}");
        assert!((expected - 1.0).abs() < 0.05, "and that is a bare lamp's light: {expected}");
        assert!(sides.iter().enumerate().all(|(k, &s)| k == below || s == 0.0), "{sides:?}");
    }

    /// Cards that saw nothing brighter than a lit wall carry no glow: the
    /// author decides.
    #[test]
    fn cards_without_a_glow_leave_it_to_the_author() {
        let mut c = blank_cards(16);
        for t in &mut c.texels {
            *t = [1.5, 1.5, 1.5, 0.5];
        }
        assert_eq!(card_sides(&c, Vec3::splat(0.2), 3.0), None);
    }

    /// The editor's frame: a lamp aimed down -z has right +x and top +y; one
    /// aimed straight down still has a frame, and every frame is a rotation.
    #[test]
    fn the_authored_faces_sit_where_the_editor_draws_them() {
        let q = light_frame(Vec3::NEG_Z);
        assert!((q * Vec3::X - Vec3::X).length() < 1e-5 && (q * Vec3::Y - Vec3::Y).length() < 1e-5);
        for d in [Vec3::NEG_Y, Vec3::Y, Vec3::new(0.3, -0.8, 0.5)] {
            let q = light_frame(d);
            assert!((q.length() - 1.0).abs() < 1e-5);
            // Front is -z of the frame.
            assert!((q * Vec3::NEG_Z - d.normalize()).length() < 1e-4, "{d}");
        }
        let only_front = GlareFacesDef { front: true, back: false, left: false, right: false, top: false, bottom: false };
        assert_eq!(authored_sides(&only_front), [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    }

    /// A lamp becomes a source in the player's frame, with its spot's beam;
    /// a light with no fixture is not one.
    #[test]
    fn a_fixtures_lamp_becomes_a_source_in_the_players_frame() {
        let mut fixtures = HashMap::new();
        fixtures.insert("lamp#0".to_string(), FixtureGlare { sides: [1.0; 6], frame: None });
        let spot = WireRenderLight {
            position: [2.0, 3.0, 0.0],
            direction: [0.0, -1.0, 0.0],
            color: space_soup_protocol::WireColor3(255, 255, 255, 255),
            intensity: 5.0,
            range: 5.0,
            cone_angle_deg: 60.0,
            inner_cone_angle_deg: 20.0,
            ..WireRenderLight::new("lamp#0", WireLightKind::Spot)
        };
        let bare = WireRenderLight { id: "fill#0".into(), ..spot.clone() };
        let yaw_inv = Quat::from_rotation_y(0.5);
        let s = sources([&spot, &bare], &fixtures, Vec3::new(1.0, 0.0, 0.0), yaw_inv);
        assert_eq!(s.len(), 1, "only the fixture's lamp");
        assert!((s[0].position - yaw_inv * Vec3::new(1.0, 3.0, 0.0)).length() < 1e-5);
        assert!((s[0].radiance - Vec3::splat(5.0)).length() < 1e-4);
        let (axis, outer, inner) = s[0].cone.expect("a spot's beam");
        assert!((axis - Vec3::NEG_Y).length() < 1e-5 && (outer - 30f32.to_radians().cos()).abs() < 1e-5);
        assert!((inner - 10f32.to_radians().cos()).abs() < 1e-5);
    }

    /// THE LEVEL'S OWN: the sconces' cards carry their bulbs, which show from
    /// below and not from above; the hanging lamps', whose bulbs glow by a
    /// texture, do not, and their lights' faces decide.
    #[test]
    fn test_rooms_sconces_glare_from_below_by_their_cards() {
        let game = Path::new(env!("CARGO_MANIFEST_DIR")).join("../game");
        let Some(level) = crate::probe_level::ProbeLevel::load(&game, "test_room") else {
            eprintln!("test_room probes not baked; skipping");
            return;
        };
        let standing = level.scene_proxies(&game, "test_room");
        let south = standing.glare.get("hallway_sconce_south").expect("the south sconce's cards carry its bulb");
        let (sides, _) = south;
        eprintln!("south sconce sides {sides:?}");
        assert!(sides[3] > 0.2, "from below: {sides:?}");
        assert!(sides[2] < 0.25 * sides[3], "from above, far less: {sides:?}");
        assert!(!standing.glare.contains_key("hall_spot_1"), "the hanging lamp's glow is a texture, not on its cards");
    }
}
