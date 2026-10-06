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
use std::sync::Arc;

use glam::{Mat3, Quat, Vec3};
use space_soup::renderer::glare::{GlareAir, GlareBulbRows, GlareSource, GlareTable, GlareTableSplit};
use space_soup_engine::reflection_cards::{card_point, LoadedCards, CARD_FACES, GLARE_BRIGHT};
use space_soup_engine::scene::GameObject;
use space_soup_engine::scene_light::{GlareFacesDef, LightDef};
use space_soup_engine::LightKind;
use space_soup_protocol::{WireLightKind, WireRenderLight};

/// Only light past this counts toward glare: see `GLARE_BRIGHT`, which the
/// bake's glare tables measure against too.
const BRIGHT: f32 = GLARE_BRIGHT;

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
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FixtureGlare {
    pub sides: [f32; 6],
    pub frame: Option<Quat>,
    /// Where each side sees the light come out, in the world: see
    /// [`card_centres`]. `None` for the author's faces.
    pub centres: Option<[Vec3; 6]>,
    /// The fixture's glare from every direction, with the middle of its box
    /// in the world, which the table's centres are measured from: see
    /// [`glare_table`]. Where there is one it decides alone.
    pub table: Option<(Arc<GlareTable>, Vec3)>,
}

/// What a fixture's cards say about its glare: each side's share
/// ([`card_sides`]), the frame they were taken in, and where on the fixture
/// each side sees its light, in the world ([`card_centres`]).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MeasuredGlare {
    pub sides: [f32; 6],
    pub frame: Quat,
    pub centres: [Vec3; 6],
    /// The bake's glare table, as shares of the lamp's light, and the box's
    /// middle in the world. See [`glare_table`].
    pub table: Option<(Arc<GlareTable>, Vec3)>,
}

/// THE BAKE'S GLARE TABLE AS SHARES OF THE LAMP'S LIGHT: each direction's
/// light past `BRIGHT` over `pi` times the lamp's `luminance`, exactly as
/// [`card_sides`] takes a card's -- what an eye that way gets of a bare lamp's
/// light. Capped like a side, and the bulb's own part of it (the table's
/// split, where the bake told them apart) never more than the whole. `None`
/// without a table, or with one that shows nothing. See
/// `reflection_cards::GlareTable`.
pub(crate) fn glare_table(cards: &LoadedCards, luminance: f32) -> Option<GlareTable> {
    let t = cards.glare.as_ref()?;
    if luminance <= 0.0 {
        return None;
    }
    let of_lamp = |f: &f32| f / (std::f32::consts::PI * luminance);
    let share: Vec<f32> = t.flux.iter().map(|f| of_lamp(f).min(MAX_SIDE)).collect();
    let split = t.split.as_ref().map(|s| GlareTableSplit {
        bulb: s.bulb.iter().zip(&share).map(|(b, whole)| of_lamp(b).clamp(0.0, *whole)).collect(),
        bulb_centre: s.bulb_centre.clone(),
        lit_centre: s.lit_centre.clone(),
        lit_spread: s.lit_spread.clone(),
        // The bulb in finer rows, capped as the whole is.
        bulb_fine: s.bulb_fine.as_ref().map(|f| {
            let share: Vec<f32> = f.flux.iter().map(|b| of_lamp(b).clamp(0.0, MAX_SIDE)).collect();
            let whole = share.iter().copied().fold(0.0, f32::max);
            GlareBulbRows { rows: f.rows, share, centre: f.centre.clone(), whole }
        }),
    });
    // The air round the bulb, as the bake measured it.
    let air = t.air.as_ref().map(|a| GlareAir { min: a.min, max: a.max, dims: a.dims, bulb: a.bulb, seen: a.seen.clone(), reach: a.reach });
    (share.iter().copied().fold(0.0, f32::max) >= CARRIES_GLOW).then(|| GlareTable {
        rows: t.rows,
        cols: t.cols,
        share,
        centre: t.centre.clone(),
        split,
        air,
    })
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

/// WHERE each side sees the light come out, in the box's frame: the middle of
/// the light card `k` saw past `BRIGHT`, each texel at the depth it saw it
/// (`card_point`), weighed by how far past. From below a sconce that is its
/// open mouth; the bulb hangs up inside the shade, and a veil grown from there
/// lit the dark shade over the mouth (headset, 2026-09-30). A side that saw no
/// such light keeps the box's middle; it shows nothing, so nothing weighs it.
pub(crate) fn card_centres(cards: &LoadedCards, half_size: Vec3) -> [Vec3; 6] {
    let res = cards.resolution as usize;
    let mut centres = [Vec3::ZERO; 6];
    if res == 0 || cards.texels.len() < CARD_FACES * res * res {
        return centres;
    }
    for (k, centre) in centres.iter_mut().enumerate() {
        let (mut sum, mut weight) = (Vec3::ZERO, 0.0f32);
        for (i, t) in cards.texels[k * res * res..(k + 1) * res * res].iter().enumerate() {
            let excess = (Vec3::new(t[0], t[1], t[2]).dot(LUMA) - BRIGHT).max(0.0);
            if excess > 0.0 {
                let (u, v) = (((i % res) as f32 + 0.5) / res as f32, ((i / res) as f32 + 0.5) / res as f32);
                sum += card_point(k, u, v, t[3].clamp(0.0, 1.0), half_size) * excess;
                weight += excess;
            }
        }
        if weight > 0.0 {
            *centre = sum / weight;
        }
    }
    centres
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
pub(crate) fn fixtures(objects: &[GameObject], measured: &HashMap<String, MeasuredGlare>) -> HashMap<String, FixtureGlare> {
    let mut out = HashMap::new();
    for o in objects.iter().filter(|o| o.mesh.is_some()) {
        for (i, l) in o.lights.iter().enumerate() {
            if !matches!(l.kind, LightKind::Point | LightKind::Spot) {
                continue;
            }
            let glare = match measured.get(&o.id) {
                Some(m) => FixtureGlare { sides: m.sides, frame: Some(m.frame), centres: Some(m.centres), table: m.table.clone() },
                None => FixtureGlare { sides: authored_sides(&l.glare_faces), frame: None, centres: None, table: None },
            };
            out.insert(format!("{}#{i}", o.id), glare);
        }
    }
    out
}

/// [`fixtures`] for a scene file, its transforms resolved to world space as
/// the proxies' were. Empty when the scene does not load.
pub(crate) fn load(game_dir: &std::path::Path, scene_name: &str, measured: &HashMap<String, MeasuredGlare>) -> HashMap<String, FixtureGlare> {
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
    cards: &[(usize, &LoadedCards, Vec3, Vec3, Quat)],
) -> HashMap<String, MeasuredGlare> {
    let mut out = HashMap::new();
    for &(object, c, centre, half_size, rotation) in cards {
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
                let centres = card_centres(c, half_size).map(|p| centre + rotation * p);
                let table = glare_table(c, luminance).map(|t| (Arc::new(t), centre));
                if table.is_none() {
                    log::info!("glare: '{}' has no glare table; its six cards' sides decide", o.id);
                }
                out.insert(o.id.clone(), MeasuredGlare { sides, frame: rotation, centres, table });
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
            // A SPOT'S BEAM decides only where its cards do not: the cards saw
            // where its bulb shows, which is wider than where its light goes --
            // the hanging lamps' bulbs are in plain view from well outside
            // their beams.
            let cone = match l.kind {
                WireLightKind::Spot if f.frame.is_none() => Some((
                    yaw_inv * direction.normalize_or_zero(),
                    (l.cone_angle_deg.to_radians() * 0.5).cos(),
                    (l.inner_cone_angle_deg.to_radians() * 0.5).cos(),
                )),
                WireLightKind::Spot | WireLightKind::Point => None,
                WireLightKind::Directional => return None,
            };
            Some(GlareSource {
                position: yaw_inv * (Vec3::from(l.position) - offset),
                radiance: Vec3::new(c[0], c[1], c[2]) * l.intensity,
                sides: f.sides,
                rotation: yaw_inv * f.frame.unwrap_or_else(|| light_frame(direction)),
                cone,
                centres: f.centres.map(|cs| cs.map(|c| yaw_inv * (c - offset))),
                table: f.table.as_ref().map(|(t, middle)| (t.clone(), yaw_inv * (*middle - offset))),
                halo_only: false,
                mirror: None,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use space_soup::renderer::glare;
    use std::path::Path;

    fn blank_cards(res: u32) -> LoadedCards {
        let n = CARD_FACES * (res * res) as usize;
        LoadedCards {
            object_id: "lamp".into(),
            resolution: res,
            texels: vec![[0.0, 0.0, 0.0, 2.0]; n],
            normals: vec![[0.0; 3]; n],
            albedo: Vec::new(),
            glare: None,
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
        fixtures.insert("lamp#0".to_string(), FixtureGlare { sides: [1.0; 6], frame: None, centres: None, table: None });
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
        // Measured by its cards, the bulb shows wherever they saw it, beam or
        // no beam.
        fixtures.insert("lamp#0".to_string(), FixtureGlare { sides: [1.0; 6], frame: Some(Quat::IDENTITY), centres: None, table: None });
        let measured = sources([&spot], &fixtures, Vec3::new(1.0, 0.0, 0.0), yaw_inv);
        assert_eq!(measured[0].cone, None);
    }

    /// A bright patch on one card puts that side's centre where the card
    /// saw it -- at its texels' place and depth (`card_point`) -- and the dim
    /// texels round it weigh nothing.
    #[test]
    fn a_sides_centre_is_where_its_card_saw_the_light() {
        let res = 8u32;
        let half = Vec3::new(0.2, 0.3, 0.1);
        let mut cards = blank_cards(res);
        // Card 3 (from below): texels (2,5) and (3,5) glow, 0.25 of the way in.
        for x in [2usize, 3] {
            cards.texels[3 * 64 + 5 * 8 + x] = [20.0, 20.0, 20.0, 0.25];
        }
        // A dim shade elsewhere on it counts for nothing.
        cards.texels[3 * 64] = [1.0, 1.0, 1.0, 0.1];
        let c = card_centres(&cards, half)[3];
        let want = card_point(3, 3.0 / 8.0, 5.5 / 8.0, 0.25, half);
        assert!((c - want).length() < 1e-5, "{c} vs {want}");
        assert_eq!(card_centres(&cards, half)[2], Vec3::ZERO, "a side that saw no light keeps the middle");
    }

    /// THE LEVEL'S SCONCES SHOW THEIR LIGHT FROM THEIR MOUTHS: seen from below,
    /// each one's centre lies below its bulb -- in the opening, not up in the
    /// shade -- and on the fixture.
    #[test]
    fn test_rooms_sconces_glare_from_their_mouths() {
        let game = Path::new(env!("CARGO_MANIFEST_DIR")).join("../game");
        let Some(level) = crate::probe_level::ProbeLevel::load(&game, "test_room") else {
            eprintln!("test_room probes not baked; skipping");
            return;
        };
        let standing = level.scene_proxies(&game, "test_room");
        let mut scene =
            space_soup_engine::scene::Scene::load(&space_soup_engine::Manifest::scene_path(&game, "test_room")).unwrap();
        scene.resolve_world_transforms();
        for lamp in ["hallway_sconce_south", "hallway_sconce_north"] {
            let m = standing.glare.get(lamp).unwrap_or_else(|| panic!("{lamp}'s cards carry its bulb"));
            let o = scene.objects.iter().find(|o| o.id == lamp).unwrap();
            let l = &o.lights[0];
            let bulb = space_soup_engine::scene_light::resolve_light_pose(
                l,
                o.cuboid.position,
                o.cuboid.rotation,
                l.socket.as_deref().and_then(|n| o.socket(n)),
            )
            .position;
            let below = m.centres[3];
            eprintln!("{lamp}: bulb {bulb}, light from below shows at {below}");
            assert!(below.y < bulb.y - 0.02, "{lamp}: {below} not below the bulb {bulb}");
            assert!((below - bulb).length() < 0.25, "{lamp}: {below} is not on the fixture round {bulb}");
        }
    }

    /// The bake's table directions and the renderer's are one mapping: the
    /// engine and the renderer each have it, and the renderer may not depend
    /// on the engine.
    #[test]
    fn the_bakes_glare_directions_are_the_renderers() {
        use space_soup_engine::reflection_cards::{glare_direction, GLARE_COLS, GLARE_ROWS};
        let n = GLARE_ROWS * GLARE_COLS;
        let t = GlareTable { rows: GLARE_ROWS, cols: GLARE_COLS, share: vec![0.0; n], centre: vec![Vec3::ZERO; n], split: None, air: None };
        for row in 0..GLARE_ROWS {
            for col in 0..GLARE_COLS {
                let (bake, render) = (glare_direction(row, col, GLARE_ROWS, GLARE_COLS), t.direction(row, col));
                assert!((bake - render).length() < 1e-6, "({row}, {col}): {bake} vs {render}");
            }
        }
        // And a lookup straight along an entry's direction reads that entry.
        let mut one = t.clone();
        one.share[5 * GLARE_COLS + 7] = 1.0;
        assert!((one.sample(glare_direction(5, 7, GLARE_ROWS, GLARE_COLS)).0 - 1.0).abs() < 1e-4);
    }

    /// A table's light is a share of its lamp's exactly as a card's is --
    /// over `pi` times the lamp's luminance -- capped like a side; its bulb's
    /// part the same, and never past the capped whole; a table that shows
    /// nothing is no table.
    #[test]
    fn a_glare_tables_light_is_a_share_of_its_lamps() {
        use space_soup_engine::reflection_cards::{GlareSplit, GlareTable as BakedTable};
        let luminance = 3.0f32;
        let mut cards = blank_cards(4);
        let pi_l = std::f32::consts::PI * luminance;
        cards.glare = Some(BakedTable {
            rows: 1,
            cols: 3,
            flux: vec![0.5 * pi_l, 0.0, 9.0 * pi_l],
            centre: vec![Vec3::new(0.0, -0.1, 0.0), Vec3::ZERO, Vec3::X],
            split: Some(GlareSplit {
                bulb: vec![0.4 * pi_l, 0.0, 8.0 * pi_l],
                bulb_centre: vec![Vec3::new(0.0, -0.12, 0.0), Vec3::ZERO, Vec3::X],
                lit_centre: vec![Vec3::new(0.0, -0.05, 0.0), Vec3::ZERO, Vec3::X],
                lit_spread: vec![0.03, 0.0, 0.1],
                bulb_fine: Some(space_soup_engine::reflection_cards::GlareBulbRows {
                    rows: 2,
                    flux: vec![0.2 * pi_l, 0.0, 9.5 * pi_l, 0.3 * pi_l, 0.0, 0.0],
                    centre: vec![Vec3::new(0.0, -0.12, 0.0); 6],
                }),
            }),
            air: Some(space_soup_engine::reflection_cards::GlareAir {
                min: Vec3::splat(-0.6),
                max: Vec3::splat(0.6),
                dims: [2, 2, 2],
                bulb: Vec3::new(0.0, -0.1, 0.0),
                seen: vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5],
                reach: 0.38,
            }),
        });
        let t = glare_table(&cards, luminance).expect("it shows");
        // The air round the bulb comes over as it was measured.
        let air = t.air.as_ref().expect("the air carried over");
        assert_eq!((air.min, air.max, air.dims, air.bulb), (Vec3::splat(-0.6), Vec3::splat(0.6), [2, 2, 2], Vec3::new(0.0, -0.1, 0.0)));
        assert_eq!((air.seen.clone(), air.reach), (vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5], 0.38));
        // The bulb's finer rows come over as shares too, capped as a side is,
        // the most of them its whole.
        let fine = t.split.as_ref().unwrap().bulb_fine.clone().unwrap();
        assert_eq!((fine.rows, fine.whole), (2, MAX_SIDE));
        let s = &fine.share;
        assert!((s[0] - 0.2).abs() < 1e-6 && s[2] == MAX_SIDE && (s[3] - 0.3).abs() < 1e-6, "{s:?}");
        assert!((t.share[0] - 0.5).abs() < 1e-6 && t.share[1] == 0.0 && t.share[2] == MAX_SIDE, "{:?}", t.share);
        assert_eq!(t.centre[0], Vec3::new(0.0, -0.1, 0.0));
        let split = t.split.as_ref().expect("the split carried over");
        assert!((split.bulb[0] - 0.4).abs() < 1e-6 && split.bulb[2] == MAX_SIDE, "{:?}", split.bulb);
        assert_eq!((split.bulb_centre[0], split.lit_centre[0], split.lit_spread[0]), (Vec3::new(0.0, -0.12, 0.0), Vec3::new(0.0, -0.05, 0.0), 0.03));
        cards.glare.as_mut().unwrap().flux = vec![0.001; 3];
        assert_eq!(glare_table(&cards, luminance), None, "below CARRIES_GLOW everywhere");
    }

    /// THE LEVEL'S LAMPS BY THEIR GLARE TABLES: none shows its bulb from
    /// level with it or from above -- the hanging lamps' shades hide theirs,
    /// which the six cards' blend let glare through from the side (headset,
    /// 2026-09-30) -- and every one shows from below.
    #[test]
    fn test_rooms_lamps_hide_their_bulbs_where_their_shades_do() {
        let game = Path::new(env!("CARGO_MANIFEST_DIR")).join("../game");
        let Some(level) = crate::probe_level::ProbeLevel::load(&game, "test_room") else {
            eprintln!("test_room probes not baked; skipping");
            return;
        };
        let standing = level.scene_proxies(&game, "test_room");
        for lamp in ["hallway_sconce_south", "hallway_sconce_north", "hall_spot_1", "hall_spot_2", "brick_lamp"] {
            let m = standing.glare.get(lamp).unwrap_or_else(|| panic!("{lamp} is measured"));
            let (table, middle) = m.table.clone().unwrap_or_else(|| panic!("{lamp} has a glare table"));
            let s = GlareSource {
                position: middle,
                radiance: Vec3::ONE,
                sides: m.sides,
                rotation: m.frame,
                cone: None,
                centres: Some(m.centres),
                table: Some((table, middle)),
                halo_only: false,
                mirror: None,
            };
            let blended = GlareSource { table: None, ..s.clone() };
            let around = |elevation: f32, azimuth: f32| {
                let (e, a) = (elevation.to_radians(), azimuth.to_radians());
                middle + m.frame * Vec3::new(e.cos() * a.cos(), e.sin(), e.cos() * a.sin()) * 3.0
            };
            let below = glare::visible_share(&s, around(-80.0, 0.0));
            let (mut level_most, mut leak) = (0.0f32, 0.0f32);
            for azimuth in (0..360).step_by(15) {
                let a = azimuth as f32;
                level_most = level_most.max(glare::visible_share(&s, around(0.0, a))).max(glare::visible_share(&s, around(40.0, a)));
                leak = leak.max(glare::visible_share(&blended, around(0.0, a)));
            }
            eprintln!("{lamp}: from below {below:.3}, level or above at most {level_most:.4} (the sides' blend: {leak:.3})");
            assert!(below > 0.1, "{lamp} shows from below: {below}");
            assert!(level_most < 0.02 * below, "{lamp} shows {level_most} from level or above, against {below} below");
        }
    }

    /// THE HANGING LAMPS' BULBS COME OUT FROM UNDER THEIR RIMS AS SLIVERS: going
    /// down from level with a lamp, the first of its bulb's veil is small -- a
    /// little of the bulb, a little of its glare -- and comes from low on the
    /// bulb, under the rim. Read from the table's rows, 10 degrees apart, it came
    /// from where a view 10 degrees lower sees the bulb, as wide as the whole
    /// bulb, and its bright core lay on the shade's outside, above the rim
    /// (headset, 2026-10-01).
    #[test]
    fn test_rooms_hanging_lamps_glare_from_the_sliver_under_the_rim() {
        let game = Path::new(env!("CARGO_MANIFEST_DIR")).join("../game");
        let Some(level) = crate::probe_level::ProbeLevel::load(&game, "test_room") else {
            eprintln!("test_room probes not baked; skipping");
            return;
        };
        let standing = level.scene_proxies(&game, "test_room");
        for lamp in ["hall_spot_1", "hall_spot_2", "brick_lamp"] {
            let m = standing.glare.get(lamp).unwrap_or_else(|| panic!("{lamp} is measured"));
            let (table, middle) = m.table.clone().unwrap_or_else(|| panic!("{lamp} has a glare table"));
            assert!(table.split.as_ref().and_then(|s| s.bulb_fine.as_ref()).is_some(), "{lamp}: re-bake its glare table");
            let s = GlareSource {
                position: middle,
                radiance: Vec3::ONE,
                sides: m.sides,
                rotation: m.frame,
                cone: None,
                centres: Some(m.centres),
                table: Some((table, middle)),
                halo_only: false,
                mirror: None,
            };
            let at = |elevation: f32| {
                let e = elevation.to_radians();
                middle + m.frame * Vec3::new(e.cos(), e.sin(), 0.0) * 3.0
            };
            let bulb = |elevation: f32| glare::glare_lobes(&s, at(elevation)).into_iter().find(|l| l.radius <= glare::LAMP_RADIUS);
            let whole = bulb(-40.0).unwrap_or_else(|| panic!("{lamp} shows its bulb from 40 degrees below"));
            let first = (0..60).map(|i| -2.0 - 0.5 * i as f32).find_map(|e| bulb(e).filter(|l| l.share > 0.0).map(|l| (e, l)));
            let (e, sliver) = first.unwrap_or_else(|| panic!("{lamp}'s bulb never shows"));
            eprintln!("{lamp}: first shows {e} degrees below: {sliver:?}; whole: {whole:?}");
            let up = m.frame * Vec3::Y;
            assert!(sliver.share < 0.2 * whole.share, "{lamp}: {} of {} as it first shows", sliver.share, whole.share);
            assert!(sliver.radius < 0.5 * glare::LAMP_RADIUS, "{lamp}: a sliver's veil {} m across", sliver.radius);
            assert!(
                (sliver.centre - whole.centre).dot(up) < -0.005,
                "{lamp}: the sliver shows from {} m above the whole bulb's middle",
                (sliver.centre - whole.centre).dot(up)
            );
        }
    }

    /// THE LEVEL'S OWN: every fixture's cards carry its bulb -- the sconces'
    /// glowing whole, the hanging lamps' placed by their emissive texture --
    /// and each shows from below and not from above.
    #[test]
    fn test_rooms_lamps_glare_from_below_by_their_cards() {
        let game = Path::new(env!("CARGO_MANIFEST_DIR")).join("../game");
        let Some(level) = crate::probe_level::ProbeLevel::load(&game, "test_room") else {
            eprintln!("test_room probes not baked; skipping");
            return;
        };
        let standing = level.scene_proxies(&game, "test_room");
        for lamp in ["hallway_sconce_south", "hallway_sconce_north", "hall_spot_1", "hall_spot_2", "brick_lamp"] {
            let sides = standing.glare.get(lamp).unwrap_or_else(|| panic!("{lamp}'s cards carry its bulb")).sides;
            eprintln!("{lamp} sides {sides:?}");
            assert!(sides[3] > 0.1, "{lamp} from below: {sides:?}");
            assert!(sides[2] < 0.25 * sides[3], "{lamp} from above, far less: {sides:?}");
        }
    }

    /// DIAGNOSTIC: hall_spot_1's glare from the bench's pendant views -- each
    /// part's share, size and where it sits against the bulb, and its veil at
    /// exposure `EXPOSURE` (default 3, the hall's).
    #[test]
    #[ignore]
    fn print_the_pendant_glare() {
        let game = Path::new(env!("CARGO_MANIFEST_DIR")).join("../game");
        let Some(level) = crate::probe_level::ProbeLevel::load(&game, "test_room") else {
            eprintln!("test_room probes not baked; skipping");
            return;
        };
        let exposure = std::env::var("EXPOSURE").ok().and_then(|e| e.parse().ok()).unwrap_or(3.0f32);
        let standing = level.scene_proxies(&game, "test_room");
        let m = standing.glare.get("hall_spot_1").unwrap();
        let (table, middle) = m.table.clone().unwrap();
        let bulb = Vec3::new(0.0, 3.1 - 1.065, -4.5);
        let warm = space_soup::renderer::Color3(255, 244, 214, 255).to_linear();
        let s = GlareSource {
            position: bulb,
            radiance: Vec3::new(warm[0], warm[1], warm[2]) * 9.0,
            sides: m.sides,
            rotation: m.frame,
            cone: None,
            centres: Some(m.centres),
            table: Some((table, middle)),
            halo_only: false,
            mirror: None,
        };
        eprintln!("bulb {bulb}, box middle {middle}");
        for (name, eye) in [
            ("pendant_far", Vec3::new(0.3, 1.6, 1.0)),
            ("pendant_mid", Vec3::new(0.6, 1.6, -2.2)),
            ("pendant_rim", Vec3::new(0.55, 1.58, -3.45)),
            ("pendant_close", Vec3::new(0.5, 1.5, -3.6)),
            ("pendant_below", Vec3::new(0.35, 1.2, -3.85)),
        ] {
            let to = bulb - eye;
            let elevation = (to.y / Vec3::new(to.x, 0.0, to.z).length()).atan().to_degrees();
            eprintln!("{name}: {:.2} m, {elevation:.1} degrees up", to.length());
            for l in glare::glare_lobes(&s, eye) {
                let along = (l.centre - eye).dot(to.normalize()) - to.length();
                let q = glare::glare_quad(&s, &l, eye, exposure, 1.0);
                eprintln!(
                    "  share {:.3} radius {:.3} centre-bulb {:?} depth vs bulb {along:+.3} m{}",
                    l.share,
                    l.radius,
                    (l.centre - bulb).to_array().map(|x| (x * 1000.0).round() / 1000.0),
                    q.map_or(String::new(), |q| format!(
                        "; a {:.3} core {:.2} deg reach {:.1} deg (cut by depth to {:.1}) peak {:.2} veil at 2/4/8 deg {:.3}/{:.3}/{:.3}",
                        q.a,
                        q.core2.sqrt(),
                        q.degrees,
                        q.core_degrees,
                        q.peak,
                        q.a * glare::cie_veil(4.0 + q.core2) - q.edge,
                        q.a * glare::cie_veil(16.0 + q.core2) - q.edge,
                        q.a * glare::cie_veil(64.0 + q.core2) - q.edge,
                    ))
                );
                // Where the core's depth test stands, against the bulb.
                let plane = l.centre + (eye - l.centre).normalize() * glare::LAMP_RADIUS.max(l.radius);
                eprintln!("    core tested from {:+.3} m against the bulb", (plane - eye).dot(to.normalize()) - to.length());
            }
        }
    }

    /// DIAGNOSTIC: how much of the glare's drawn area at the bench's glare views
    /// can show nothing -- cells whose corners all lie in dark air, or wholly
    /// past the veil's reach -- as shares of the left eye's view, every lamp of
    /// test_room at exposure `EXPOSURE` (default 3), the torch where the view
    /// has one.
    #[test]
    #[ignore]
    fn print_the_glare_fill() {
        let game = Path::new(env!("CARGO_MANIFEST_DIR")).join("../game");
        let Some(level) = crate::probe_level::ProbeLevel::load(&game, "test_room") else {
            eprintln!("test_room probes not baked; skipping");
            return;
        };
        let exposure = std::env::var("EXPOSURE").ok().and_then(|e| e.parse().ok()).unwrap_or(3.0f32);
        let standing = level.scene_proxies(&game, "test_room");
        let fixtures = load(&game, "test_room", &standing.glare);
        let lights: Vec<_> = crate::scene_lights::load(&game, "test_room")
            .into_iter()
            .chain(crate::scene_lights::load_baked(&game, "test_room"))
            .collect();
        let lamps = sources(lights.iter(), &fixtures, Vec3::ZERO, Quat::IDENTITY);
        // The left eye's frustum in tangents, left/right/down/up (bench logs).
        let (l, r, d, u) = (1.376f32, 0.839f32, 0.966f32, 1.428f32);
        let view_area = (l + r) * (d + u);
        for (name, eye, at, torch) in [
            ("pendant_close", Vec3::new(0.5, 1.5, -3.6), Vec3::new(0.0, 1.95, -4.5), None),
            ("pendant_below", Vec3::new(0.35, 1.2, -3.85), Vec3::new(0.0, 1.95, -4.5), None),
            (
                "torch_facing",
                Vec3::new(0.0, 1.6, -1.0),
                Vec3::new(0.0, 1.5, -3.0),
                Some((Vec3::new(0.1, 1.45, -3.0), Vec3::new(0.0, 1.6, -1.0))),
            ),
        ] {
            let forward = (at - eye).normalize();
            let right = forward.cross(Vec3::Y).normalize();
            let up = right.cross(forward);
            let eyes = [eye - right * 0.032, eye + right * 0.032];
            let mut all = lamps.clone();
            let mut adapted = vec![true; all.len()];
            if let Some((glass, aim)) = torch {
                let (lens, rotation) = crate::flashlight::lens_from_bench(glass, aim);
                all.push(crate::flashlight::glare_source(lens, rotation * Vec3::NEG_Z, Vec3::ZERO, Quat::IDENTITY));
                adapted.push(false);
            }
            let (verts, idx, halos) = glare::build_glare(&all, eyes, right, up, exposure, 1.0, true, &[], &adapted);
            // On the left eye's tangent plane, clipped to nothing finer than
            // what is in front of the eye.
            let tangent = |p: Vec3| {
                let v = p - eyes[0];
                let z = v.dot(forward).max(1e-3);
                Vec3::new(v.dot(right) / z, v.dot(up) / z, 0.0)
            };
            // Every cell of every quad, drawn or not: the quads' points come in
            // square blocks, `cells + 1` a side; a core follows its halo with
            // the same veil and a smaller reach.
            let drawn: std::collections::HashMap<usize, bool> =
                idx.chunks(6).enumerate().map(|(k, c)| (c[0] as usize, 6 * k < halos as usize)).collect();
            // [halo, core] x [whole grid, drawn now, dark air, past reach].
            let mut area_of = [[0.0f32; 4]; 2];
            let (mut base, mut last_halo) = (0usize, None::<[f32; 4]>);
            while base < verts.len() {
                let cells = (2.0 / (verts[base + 1].uv[0] + 1.0)).round() as usize;
                let side = cells + 1;
                let shape = verts[base].shape;
                let core = last_halo.is_some_and(|h| h[0] == shape[0] && h[1] == shape[1] && h[3] == shape[3] && shape[2] < h[2]);
                last_halo = if core { None } else { Some(shape) };
                let part = usize::from(core);
                for j in 0..cells {
                    for i in 0..cells {
                        let a = base + j * side + i;
                        let corners = [a, a + 1, a + 1 + side, a + side].map(|k| &verts[k]);
                        let p = corners.map(|c| tangent(Vec3::from(c.position)));
                        // Only what lands in the eye's view is drawn.
                        if !p.iter().any(|q| q.x > -l && q.x < r && q.y > -d && q.y < u) {
                            continue;
                        }
                        let area = 0.5
                            * (0..4)
                                .map(|k| {
                                    let (a, b) = (p[k], p[(k + 1) % 4]);
                                    a.x * b.y - b.x * a.y
                                })
                                .sum::<f32>()
                                .abs();
                        let (lo, hi) = (corners[0].uv, corners[2].uv);
                        let (nu, nv) = (0.0f32.clamp(lo[0], hi[0]), 0.0f32.clamp(lo[1], hi[1]));
                        area_of[part][0] += area;
                        if drawn.contains_key(&a) {
                            assert_eq!(drawn[&a], !core, "cell {a} drawn as the wrong part");
                            area_of[part][1] += area;
                        }
                        if corners.iter().all(|c| c.reach[3] <= 0.0) {
                            area_of[part][2] += area;
                        } else if nu * nu + nv * nv >= 1.0 {
                            area_of[part][3] += area;
                        }
                    }
                }
                base += side * side;
            }
            for (part, a) in ["halo", "core"].iter().zip(area_of) {
                eprintln!(
                    "{name} {part}: the whole grid {:.1}% of the view, drawn now {:.1}% (dark air {:.1}%, past reach {:.1}%)",
                    100.0 * a[0] / view_area,
                    100.0 * a[1] / view_area,
                    100.0 * a[2] / view_area,
                    100.0 * a[3] / view_area,
                );
            }
        }
    }
}
