use glam::{Quat, Vec3};
use space_soup_engine::{Color3, CuboidStyle, RenderCuboid};

/// Plain position/rotation pair — deliberately not tied to `network.rs`'s
/// android-only wire conversion, so this module (and its tests) compile and
/// run on any host.
#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
}

#[derive(Debug, Clone, Copy)]
pub struct LocalPose {
    pub head: Transform,
    pub left_hand: Option<Transform>,
    pub right_hand: Option<Transform>,
}

#[derive(Debug, Clone, Copy)]
pub struct RemotePlayerState {
    pub head: Transform,
    pub left_hand: Option<Transform>,
    pub right_hand: Option<Transform>,
}

#[derive(Debug, Clone, Copy)]
pub struct ArmLengths {
    pub upper: f32,
    pub forearm: f32,
}

impl Default for ArmLengths {
    fn default() -> Self {
        Self {
            upper: 0.28,
            forearm: 0.26,
        }
    }
}

const TORSO_DROP: f32 = 0.35;
const SHOULDER_OFFSET: Vec3 = Vec3::new(0.20, 0.05, 0.0);
const AVATAR_COLOR: Color3 = Color3(200, 200, 210, 255);
const AVATAR_WIRE: Color3 = Color3(0, 0, 0, 0);

/// Correction applied to `models/man.glb` so it sits upright at a realistic
/// scale: the source asset (Sketchfab/FBX, per its embedded node names) is
/// Z-up with a ~4.2-unit "height", so it needs a -90°-about-X rotation and a
/// ~0.43 scale-down to look right under this engine's Y-up, meters convention.
/// Estimated from the raw mesh bounding box, not visually verified — expect
/// to retune both once this can actually be seen on-headset.
pub fn man_glb_orientation_offset() -> Quat {
    Quat::from_rotation_x(-90_f32.to_radians())
}
pub const MAN_GLB_SCALE: f32 = 0.43;

/// Elbow position for a 2-bone (shoulder-elbow-hand) IK chain, via the
/// standard law-of-cosines solve. The bend direction is picked from a fixed
/// "elbow points down" hint projected perpendicular to the shoulder->hand
/// axis, so it's deterministic and doesn't flicker frame to frame.
pub fn solve_arm(shoulder: Vec3, hand: Vec3, lengths: ArmLengths) -> Vec3 {
    let to_hand = hand - shoulder;
    let raw_dist = to_hand.length();
    let min_reach = (lengths.upper - lengths.forearm).abs().max(0.01);
    let max_reach = lengths.upper + lengths.forearm;
    let dist = raw_dist.clamp(min_reach, max_reach);

    let axis = if raw_dist > 1e-5 {
        to_hand / raw_dist
    } else {
        Vec3::NEG_Y
    };

    let cos_theta = ((lengths.upper * lengths.upper + dist * dist
        - lengths.forearm * lengths.forearm)
        / (2.0 * lengths.upper * dist))
        .clamp(-1.0, 1.0);
    let theta = cos_theta.acos();

    let bend_dir = perpendicular_component(Vec3::NEG_Y, axis)
        .unwrap_or_else(|| perpendicular_component(Vec3::X, axis).unwrap_or(Vec3::X));

    shoulder + axis * (lengths.upper * theta.cos()) + bend_dir * (lengths.upper * theta.sin())
}

fn perpendicular_component(hint: Vec3, axis: Vec3) -> Option<Vec3> {
    let component = hint - axis * hint.dot(axis);
    (component.length_squared() > 1e-6).then(|| component.normalize())
}

/// Torso position/orientation derived from the head alone: dropped straight
/// down by a fixed height, with **yaw-only** rotation — pitch/roll are
/// discarded so looking up/down doesn't tilt the torso (same trick already
/// used in this codebase's frustum-culling fix, for the same reason: pitch
/// shouldn't leak into a computation that's conceptually about heading).
pub fn torso_transform(head: Transform) -> Transform {
    let forward = head.rotation * Vec3::NEG_Z;
    let forward_h = Vec3::new(forward.x, 0.0, forward.z);
    let rotation = if forward_h.length_squared() < 1e-6 {
        Quat::IDENTITY
    } else {
        Quat::from_rotation_arc(Vec3::NEG_Z, forward_h.normalize())
    };
    Transform {
        position: head.position - Vec3::Y * TORSO_DROP,
        rotation,
    }
}

fn bone_cuboid(id: String, from: Vec3, to: Vec3, thickness: f32) -> RenderCuboid {
    let delta = to - from;
    let length = delta.length();
    let dir = if length > 1e-5 { delta / length } else { Vec3::NEG_Y };
    RenderCuboid {
        id,
        position: from + dir * (length / 2.0),
        half_size: Vec3::new(thickness, thickness, (length / 2.0).max(0.001)),
        rotation: Quat::from_rotation_arc(Vec3::Z, dir),
        color: AVATAR_COLOR,
        wire_color: AVATAR_WIRE,
        style: CuboidStyle::Solid,
    }
}

#[allow(clippy::too_many_arguments)]
fn push_arm(
    out: &mut Vec<RenderCuboid>,
    id_str: &str,
    label: &str,
    torso: Transform,
    local_shoulder: Vec3,
    hand: Transform,
    lengths: ArmLengths,
) {
    let shoulder = torso.position + torso.rotation * local_shoulder;
    let elbow = solve_arm(shoulder, hand.position, lengths);

    out.push(bone_cuboid(
        format!("avatar_{id_str}_{label}_upper"),
        shoulder,
        elbow,
        0.045,
    ));
    out.push(bone_cuboid(
        format!("avatar_{id_str}_{label}_forearm"),
        elbow,
        hand.position,
        0.04,
    ));
    out.push(RenderCuboid {
        id: format!("avatar_{id_str}_{label}_hand"),
        position: hand.position,
        half_size: Vec3::splat(0.045),
        rotation: hand.rotation,
        color: AVATAR_COLOR,
        wire_color: AVATAR_WIRE,
        style: CuboidStyle::Solid,
    });
}

/// Builds a small cuboid arm rig (arms only — no legs, standard for VR
/// avatars with no leg tracking) for one remote player. The head/torso are
/// no longer plain cuboids here; they're rendered as a `models/man.glb` mesh
/// instance positioned via `torso_transform` (see `lib.rs`'s avatar mesh
/// cache). `id_str` should be unique per player (e.g. their `PlayerId`'s
/// string form) since it's used to make each cuboid's id unique across
/// players.
pub fn build_avatar_cuboids(id_str: &str, remote: &RemotePlayerState) -> Vec<RenderCuboid> {
    let mut out = Vec::with_capacity(6);

    let torso = torso_transform(remote.head);
    let lengths = ArmLengths::default();
    if let Some(hand) = remote.left_hand {
        let local_shoulder = Vec3::new(-SHOULDER_OFFSET.x, SHOULDER_OFFSET.y, SHOULDER_OFFSET.z);
        push_arm(&mut out, id_str, "left", torso, local_shoulder, hand, lengths);
    }
    if let Some(hand) = remote.right_hand {
        push_arm(&mut out, id_str, "right", torso, SHOULDER_OFFSET, hand, lengths);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_arm_matches_law_of_cosines() {
        let shoulder = Vec3::ZERO;
        let hand = Vec3::new(0.0, 0.0, -0.5);
        let lengths = ArmLengths {
            upper: 0.28,
            forearm: 0.26,
        };

        let elbow = solve_arm(shoulder, hand, lengths);

        assert!(
            (shoulder.distance(elbow) - lengths.upper).abs() < 1e-4,
            "shoulder-elbow distance should equal upper arm length, got {}",
            shoulder.distance(elbow)
        );
        assert!(
            (elbow.distance(hand) - lengths.forearm).abs() < 1e-4,
            "elbow-hand distance should equal forearm length, got {}",
            elbow.distance(hand)
        );
    }

    #[test]
    fn solve_arm_clamps_unreachable_targets() {
        let shoulder = Vec3::ZERO;
        let lengths = ArmLengths {
            upper: 0.28,
            forearm: 0.26,
        };
        // Target far beyond max reach (upper + forearm = 0.54).
        let hand = Vec3::new(0.0, 0.0, -5.0);

        let elbow = solve_arm(shoulder, hand, lengths);

        assert!(elbow.is_finite(), "elbow position should stay finite: {elbow:?}");
        assert!(
            (shoulder.distance(elbow) - lengths.upper).abs() < 1e-3,
            "should still respect upper arm length when clamped"
        );
    }

    #[test]
    fn torso_transform_ignores_pitch_and_roll() {
        let base_yaw = 30.0_f32.to_radians();
        let head_level = Transform {
            position: Vec3::new(1.0, 1.7, 2.0),
            rotation: Quat::from_rotation_y(base_yaw),
        };
        let head_pitched = Transform {
            position: head_level.position,
            rotation: Quat::from_rotation_y(base_yaw) * Quat::from_rotation_x(60.0_f32.to_radians()),
        };

        let torso_level = torso_transform(head_level);
        let torso_pitched = torso_transform(head_pitched);

        let fwd_level = torso_level.rotation * Vec3::NEG_Z;
        let fwd_pitched = torso_pitched.rotation * Vec3::NEG_Z;

        assert!(
            fwd_level.distance(fwd_pitched) < 1e-4,
            "pitching the head should not change torso orientation: {fwd_level:?} vs {fwd_pitched:?}"
        );
        assert!(
            fwd_level.y.abs() < 1e-5,
            "torso forward should be perfectly horizontal, got y={}",
            fwd_level.y
        );
    }

    #[test]
    fn build_avatar_cuboids_head_only_has_no_arms() {
        let remote = RemotePlayerState {
            head: Transform {
                position: Vec3::new(0.0, 1.7, 0.0),
                rotation: Quat::IDENTITY,
            },
            left_hand: None,
            right_hand: None,
        };

        let cuboids = build_avatar_cuboids("test-player", &remote);

        // Head/torso are the man.glb mesh now (see lib.rs), not cuboids —
        // with no hands tracked there should be no arm cuboids either.
        assert!(cuboids.is_empty());
    }
}
