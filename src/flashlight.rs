//! THE PLAYER'S FLASHLIGHT: a shadow-casting spot carried in the right hand,
//! for trying every lighting and shadow system against a light that moves with
//! the player. The user, 2026-10-01: "a player toggle flashlight (with or
//! without model) that we can use to test more dynamic moving lights with all
//! our other lighting and shadow systems".
//!
//! The right thumbstick's click cycles it: off, the beam, the beam and the
//! torch in the hand, off. A bench view can carry one at a fixed place
//! ([`space_soup::renderer::bench::BenchFlashlight`]), so the headset can be
//! measured with it unattended.
//!
//! IT IS AN ORDINARY LIVE SPOT. Nothing in the renderer knows it is a
//! flashlight, so whatever it shows about the lighting is what any moving lamp
//! would show. The one thing it needs that a ceiling fixture does not is a
//! shadow map that starts at the glass ([`SHADOW_NEAR`]): a fixture's map
//! starts 30 cm out to leave its own housing behind it, and a hand held in
//! front of a flashlight is closer than that.

use glam::{Quat, Vec3};
use space_soup::renderer::glare::GlareSource;
use space_soup::renderer::{Color3, Light, LightKind};

/// The torch model, under the game folder, and the name the mesh loader
/// files it under -- one no scene object can have.
pub(crate) const TORCH_MODEL: &str = "models/flashlight.glb";
pub(crate) const TORCH_ID: &str = "__player_flashlight__";

/// OTHER PLAYERS' TORCHES, as many as are drawn at once: the mesh loader
/// gives each of these names its own instance of the model. Their beams and
/// their glare are not limited by this -- only the torches in their hands.
pub(crate) const REMOTE_TORCH_IDS: [&str; 4] =
    ["__remote_flashlight_0__", "__remote_flashlight_1__", "__remote_flashlight_2__", "__remote_flashlight_3__"];

/// Which of [`REMOTE_TORCH_IDS`] the mesh loader's `id` is.
pub(crate) fn remote_torch_slot(id: &str) -> Option<usize> {
    REMOTE_TORCH_IDS.iter().position(|r| *r == id)
}

/// Another player's glass as their client sent it, if it is a place and its
/// turn is a turn -- a beam from nowhere would light the level from nowhere
/// -- the turn made exact. In the WORLD, as `lens_from_aim` gives the local
/// player's.
pub(crate) fn remote_lens(position: Vec3, rotation: Quat) -> Option<(Vec3, Quat)> {
    let usable = position.is_finite() && rotation.is_finite() && rotation.length() > 0.5;
    usable.then(|| (position, rotation.normalize()))
}

/// What the clicks have made of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum FlashlightMode {
    #[default]
    Off,
    /// The beam alone, from where the hand points.
    Beam,
    /// The beam and the torch it comes from.
    BeamAndTorch,
}

impl FlashlightMode {
    /// The mode after one more click.
    pub(crate) fn next(self) -> Self {
        match self {
            Self::Off => Self::Beam,
            Self::Beam => Self::BeamAndTorch,
            Self::BeamAndTorch => Self::Off,
        }
    }

    pub(crate) fn lit(self) -> bool {
        self != Self::Off
    }

    pub(crate) fn shows_torch(self) -> bool {
        self == Self::BeamAndTorch
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Beam => "on",
            Self::BeamAndTorch => "on, with the torch",
        }
    }
}

/// How far ahead of the controller's aim origin the glass sits, metres. The
/// aim origin is at the front of the controller already; another 4 cm puts the
/// fingers wrapped round it behind the shadow map's near plane, so the hand
/// holding the light never shadows its own beam.
pub(crate) const LENS_AHEAD: f32 = 0.04;

/// The beam's centre, in the lamps' units (a surface `d` metres away is lit
/// `albedo x intensity / d^2`). A modest LED torch: test_room's ceiling spots
/// are 3-9, and a torch's hot spot is a few times a room spot's centre.
pub(crate) const INTENSITY: f32 = 16.0;

/// Full angle of the spill's edge, degrees.
pub(crate) const CONE_DEG: f32 = 50.0;

/// Full angle of the hot spot, degrees: the beam falls from full here to
/// nothing at [`CONE_DEG`].
pub(crate) const HOTSPOT_DEG: f32 = 16.0;

/// Where the inverse square is windowed to zero, metres.
pub(crate) const RANGE: f32 = 20.0;

/// Where its shadow map begins, metres from the glass. See the module doc.
pub(crate) const SHADOW_NEAR: f32 = 0.02;

/// A white LED's 5600 K.
pub(crate) const COLOUR: Color3 = Color3(255, 241, 228, 255);

/// The glass and the way the beam leaves it, in the WORLD, from the hand's
/// aim pose there: forward is the pose's -Z, an OpenXR aim ray's.
pub(crate) fn lens_from_aim(aim_position: Vec3, aim_rotation: Quat) -> (Vec3, Vec3) {
    let forward = (aim_rotation * Vec3::NEG_Z).normalize_or_zero();
    (aim_position + forward * LENS_AHEAD, forward)
}

/// The glass aimed from `at` toward `aim` in the world, as a bench view gives it.
pub(crate) fn lens_from_bench(at: Vec3, aim: Vec3) -> (Vec3, Quat) {
    let forward = (aim - at).normalize_or_zero();
    // Up stays up unless the beam is nearly vertical.
    let up = if forward.dot(Vec3::Y).abs() > 0.99 { Vec3::Z } else { Vec3::Y };
    let right = forward.cross(up).normalize_or_zero();
    let true_up = right.cross(forward);
    let rotation = Quat::from_mat3(&glam::Mat3::from_cols(right, true_up, -forward));
    (at, rotation)
}

/// The beam as the renderer takes it: in the PLAYER's frame, like every light
/// `build_render_lists` hands over.
pub(crate) fn beam(lens: Vec3, forward: Vec3, offset: Vec3, yaw_inv: Quat) -> Light {
    Light {
        position: yaw_inv * (lens - offset),
        direction: yaw_inv * forward,
        kind: LightKind::Spot,
        color: COLOUR,
        intensity: INTENSITY,
        range: RANGE,
        cone_angle_deg: CONE_DEG,
        inner_cone_angle_deg: HOTSPOT_DEG,
        mask_channel: None,
        shadow_near: Some(SHADOW_NEAR),
        source_radius: 0.0,
        in_level_bake: false,
    }
}

/// Where the torch model goes, in the player's frame: its glass on the beam's
/// origin, turned like the hand, so its body trails back along the beam into
/// the fist (`tools/models/flashlight.py` builds it with the beam along -Z
/// from the origin).
pub(crate) fn torch_pose(lens: Vec3, rotation: Quat, offset: Vec3, yaw_inv: Quat) -> (Vec3, Quat) {
    (yaw_inv * (lens - offset), (yaw_inv * rotation).normalize())
}

/// How brightly the glass glows: the rule every fixture's bulb follows,
/// intensity over the lamp radius squared
/// (`space_soup_engine::scene_light::emissive_drive`).
pub(crate) fn lens_drive() -> f32 {
    let r = space_soup_engine::scene_light::LAMP_RADIUS;
    INTENSITY / (r * r)
}

/// The torch's glass, metres across its radius (`tools/models/flashlight.py`).
pub(crate) const GLASS_RADIUS: f32 = 0.0168;
/// The torch from its glass back to its tail cap, metres, and as one capsule
/// that long: the head's 19 mm radius and the grip's 14.5 mm, weighed by
/// their lengths.
pub(crate) const TORCH_LENGTH: f32 = 0.15;
pub(crate) const TORCH_RADIUS: f32 = 0.0165;
/// Its anodised body's albedo (the model's 0.035-0.04).
pub(crate) const TORCH_ALBEDO: f32 = 0.037;

/// The colour the glass glows, as the model's emission gives it: the light's
/// own [`COLOUR`], each channel a fraction of full.
pub(crate) fn glass_colour() -> [f32; 3] {
    [COLOUR.0 as f32 / 255.0, COLOUR.1 as f32 / 255.0, COLOUR.2 as f32 / 255.0]
}

/// THE TORCH AS CAPSULES, for its reflections, in the PLAYER's frame: its
/// body from the glass back to the tail, as dark as it is anodised, and its
/// glass, glowing toward everything in front of it as brightly as it is
/// drawn ([`lens_drive`]). The user, 2026-10-02: the flashlight "does not
/// show as a reflection on any surface ... nor does the front of the
/// flashlight light up or show lit in the avatar's reflection". See
/// `space_soup::renderer::uniforms::CapsuleGroup::surfaces`.
pub(crate) fn torch_capsules(
    lens: Vec3,
    rotation: Quat,
    offset: Vec3,
    yaw_inv: Quat,
) -> space_soup::renderer::uniforms::CapsuleGroup {
    let (at, turn) = torch_pose(lens, rotation, offset, yaw_inv);
    // The model's body trails back along +Z from the glass.
    let back = (turn * Vec3::Z).normalize_or_zero();
    space_soup::renderer::uniforms::CapsuleGroup {
        capsules: vec![
            (at + back * (TORCH_LENGTH - TORCH_RADIUS), at + back * TORCH_RADIUS, TORCH_RADIUS),
            // A disc at its second end, facing away from its first.
            (at + back * 0.01, at, GLASS_RADIUS),
        ],
        colour: glass_colour(),
        surfaces: vec![-TORCH_ALBEDO, lens_drive()],
    }
}

/// THE GLASS AS A SOURCE OF GLARE, in the player's frame: seen from inside
/// its beam it glares as every lamp's bulb does, as much as the beam sends
/// that way -- full in the hot spot, nothing past the spill -- and from
/// behind or beside it, where its own body hides the glass, not at all (user,
/// 2026-10-02: "the flashlight will need to have the hdr bloom effect on
/// it"). As a halo alone: its bezel and the hand round it stand at the glass,
/// and a core showed the bezel as a dark ring in the glare. See
/// `space_soup::renderer::glare`.
pub(crate) fn glare_source(lens: Vec3, forward: Vec3, offset: Vec3, yaw_inv: Quat) -> GlareSource {
    let c = COLOUR.to_linear();
    let (outer, inner) = cone_cosines();
    GlareSource {
        position: yaw_inv * (lens - offset),
        radiance: Vec3::new(c[0], c[1], c[2]) * INTENSITY,
        sides: [1.0; 6],
        rotation: Quat::IDENTITY,
        cone: Some(((yaw_inv * forward).normalize_or_zero(), outer, inner)),
        centres: None,
        table: None,
        halo_only: true,
        mirror: None,
    }
}

/// The cosines of the beam's half angles: the spill's edge and the hot spot's.
pub(crate) fn cone_cosines() -> (f32, f32) {
    ((0.5 * CONE_DEG).to_radians().cos(), (0.5 * HOTSPOT_DEG).to_radians().cos())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_click_cycles_off_beam_torch_and_back() {
        let mut m = FlashlightMode::default();
        assert_eq!(m, FlashlightMode::Off);
        assert!(!m.lit());
        m = m.next();
        assert!(m.lit() && !m.shows_torch());
        m = m.next();
        assert!(m.lit() && m.shows_torch());
        m = m.next();
        assert_eq!(m, FlashlightMode::Off);
    }

    #[test]
    fn the_glass_is_ahead_of_the_hand_along_its_aim() {
        // A hand at (1, 1.2, -2) turned 90 degrees left: its aim runs along -X.
        let aim = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let (lens, forward) = lens_from_aim(Vec3::new(1.0, 1.2, -2.0), aim);
        assert!((forward - Vec3::NEG_X).length() < 1e-5, "{forward}");
        assert!((lens - Vec3::new(1.0 - LENS_AHEAD, 1.2, -2.0)).length() < 1e-5, "{lens}");
    }

    #[test]
    fn the_beam_reaches_the_renderer_in_the_players_frame() {
        // The rig turned 90 degrees and moved: a world point ahead of the glass
        // must stay ahead of it after both go through the same transform.
        let offset = Vec3::new(3.0, 0.0, -1.0);
        let yaw = 0.7_f32;
        let yaw_inv = Quat::from_rotation_y(-yaw);
        let lens = Vec3::new(4.0, 1.4, -2.5);
        let forward = Vec3::new(0.3, -0.2, -0.93).normalize();
        let l = beam(lens, forward, offset, yaw_inv);
        let ahead_world = lens + forward * 2.0;
        let ahead_player = yaw_inv * (ahead_world - offset);
        let along = (ahead_player - l.position).normalize();
        assert!((along - l.direction).length() < 1e-5, "{along} vs {}", l.direction);
        assert_eq!(l.kind, LightKind::Spot);
        assert_eq!(l.mask_channel, None, "a moving light has no baked mask");
        assert_eq!(l.shadow_near, Some(SHADOW_NEAR));
        assert!(l.inner_cone_angle_deg > 0.0 && l.inner_cone_angle_deg < l.cone_angle_deg);
    }

    #[test]
    fn a_bench_flashlight_points_where_it_is_told() {
        let (at, rot) = lens_from_bench(Vec3::new(0.2, 1.1, 0.0), Vec3::new(0.2, 0.0, -3.0));
        let forward = rot * Vec3::NEG_Z;
        let want = (Vec3::new(0.2, 0.0, -3.0) - at).normalize();
        assert!((forward - want).length() < 1e-5, "{forward} vs {want}");
        // And straight down does not divide by zero.
        let (_, down) = lens_from_bench(Vec3::Y, Vec3::ZERO);
        assert!(((down * Vec3::NEG_Z) - Vec3::NEG_Y).length() < 1e-5);
    }

    /// The glass glares at an eye in its beam -- fully down the hot spot,
    /// partly in the spill -- and not at one behind the torch or beside it.
    #[test]
    fn the_glass_glares_only_into_its_beam() {
        use space_soup::renderer::glare::{glare_lobes, visible_share};
        let lens = Vec3::new(0.0, 1.5, 0.0);
        let s = glare_source(lens, Vec3::NEG_Z, Vec3::ZERO, Quat::IDENTITY);
        assert!(visible_share(&s, lens + Vec3::NEG_Z * 2.0) > 0.99, "down the beam");
        assert_eq!(visible_share(&s, lens + Vec3::Z * 2.0), 0.0, "behind the torch");
        assert_eq!(visible_share(&s, lens + Vec3::X * 2.0), 0.0, "beside it");
        assert!(!glare_lobes(&s, lens + Vec3::NEG_Z * 2.0).is_empty());
        let spill = visible_share(&s, lens + Quat::from_rotation_y(15f32.to_radians()) * Vec3::NEG_Z * 2.0);
        assert!(spill > 0.0 && spill < 1.0, "15 degrees off, in the spill: {spill}");
    }

    /// Another player's glass is used as sent when it is a place and a turn,
    /// the turn made exact; anything else -- a NaN, a zero turn -- is no torch.
    #[test]
    fn another_players_glass_is_used_only_when_it_is_a_place_and_a_turn() {
        let (at, turn) = remote_lens(Vec3::new(1.0, 1.2, -3.0), Quat::from_rotation_y(0.4) * 1.01).unwrap();
        assert_eq!(at, Vec3::new(1.0, 1.2, -3.0));
        assert!((turn.length() - 1.0).abs() < 1e-6 && turn.dot(Quat::from_rotation_y(0.4)) > 0.9999);
        assert!(remote_lens(Vec3::new(f32::NAN, 1.0, 0.0), Quat::IDENTITY).is_none());
        assert!(remote_lens(Vec3::ZERO, Quat::from_xyzw(0.0, 0.0, 0.0, 0.0)).is_none());
        assert!(remote_lens(Vec3::ZERO, Quat::from_xyzw(f32::INFINITY, 0.0, 0.0, 1.0)).is_none());
        assert_eq!(remote_torch_slot(REMOTE_TORCH_IDS[2]), Some(2));
        assert_eq!(remote_torch_slot(TORCH_ID), None, "the player's own torch is not another's");
    }

    /// THE TORCH IS REFLECTED WHERE IT IS DRAWN: its glass a disc on the
    /// beam's origin facing down the beam, glowing as the drawn glass does;
    /// its body behind it, from the glass to the tail -- in the player's frame.
    #[test]
    fn the_torch_is_reflected_where_it_is_drawn() {
        let offset = Vec3::new(-2.0, 0.0, 5.0);
        let yaw_inv = Quat::from_rotation_y(-1.1);
        let aim = Quat::from_rotation_x(-0.4);
        let (lens, forward) = lens_from_aim(Vec3::new(0.0, 1.0, 0.0), aim);
        let l = beam(lens, forward, offset, yaw_inv);
        let g = torch_capsules(lens, aim, offset, yaw_inv);
        let (ga, gb, gr) = g.capsules[1];
        assert!((gb - l.position).length() < 1e-5, "the glass on the beam's origin");
        assert!(((gb - ga).normalize() - l.direction).length() < 1e-4, "facing down the beam");
        assert_eq!((gr, g.surfaces[1]), (GLASS_RADIUS, lens_drive()));
        let (ba, bb, br) = g.capsules[0];
        assert!(g.surfaces[0] < 0.0, "the body a solid");
        assert!(((ba - l.position).length() + br - TORCH_LENGTH).abs() < 1e-5, "its tail");
        assert!(((bb - l.position).length() - br).abs() < 1e-5, "its front at the glass");
        assert!((ba - l.position).dot(l.direction) < 0.0 && (bb - l.position).dot(l.direction) < 0.0, "behind it");
    }

    #[test]
    fn the_torch_sits_on_the_beam() {
        let offset = Vec3::new(-2.0, 0.0, 5.0);
        let yaw_inv = Quat::from_rotation_y(-1.1);
        let aim = Quat::from_rotation_x(-0.4);
        let (lens, forward) = lens_from_aim(Vec3::new(0.0, 1.0, 0.0), aim);
        let l = beam(lens, forward, offset, yaw_inv);
        let (pos, rot) = torch_pose(lens, aim, offset, yaw_inv);
        assert!((pos - l.position).length() < 1e-5);
        assert!(((rot * Vec3::NEG_Z) - l.direction).length() < 1e-5, "the glass faces along the beam");
    }
}
