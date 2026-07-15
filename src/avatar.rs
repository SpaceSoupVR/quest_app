use glam::{Mat4, Quat, Vec3};

use space_soup::renderer::mesh::GltfSkin;

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

/// `models/man.glb`'s bind-pose vertex data is already Y-up (head sits
/// above torso above feet along +Y — verified directly against the file,
/// not assumed), but at roughly 4.2x real-world scale — the same size
/// issue the original unrigged version had, just no longer also needing a
/// Z-up rotation fix now that it's rigged. Estimated from bind-pose
/// head/foot heights (~3.58 units apart), not visually verified — expect
/// to retune once this can actually be seen on-headset.
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

/// Root placement for the whole skeleton, derived from the head alone:
/// dropped straight down by `floor_drop` (already scaled to match
/// `root_scale`, see call site), with **yaw-only** rotation — pitch/roll
/// are discarded so looking up/down doesn't tilt the body.
pub fn body_root_transform(head: Transform, floor_drop: f32) -> Transform {
    let forward = head.rotation * Vec3::NEG_Z;
    let forward_h = Vec3::new(forward.x, 0.0, forward.z);
    let rotation = if forward_h.length_squared() < 1e-6 {
        Quat::IDENTITY
    } else {
        Quat::from_rotation_arc(Vec3::NEG_Z, forward_h.normalize())
    };
    Transform {
        position: head.position - Vec3::Y * floor_drop,
        rotation,
    }
}

fn joint_index(skin: &GltfSkin, name: &str) -> Option<usize> {
    skin.joint_names.iter().position(|n| n == name)
}

fn joint_index_any(skin: &GltfSkin, names: &[&str]) -> Option<usize> {
    names.iter().find_map(|n| joint_index(skin, n))
}

/// "Head" (Mixamo-style rigs) or "head" (this asset's own rig) — tried in
/// that order so this works for either naming convention without needing
/// to know in advance which one a given asset uses.
fn find_head_joint(skin: &GltfSkin) -> Option<usize> {
    joint_index_any(skin, &["Head", "head"])
}

/// The rigged skeleton's own bind-pose head height (mesh-local, *before*
/// `root_scale` is applied — multiply by it at the call site), used so
/// `body_root_transform` can put the feet at floor level under the tracked
/// head regardless of this particular rig's proportions. Falls back to a
/// generic human height if this skeleton has no head joint under either
/// naming convention.
pub fn bind_head_height(skin: &GltfSkin) -> f32 {
    let Some(head_ji) = find_head_joint(skin) else {
        return 1.6;
    };
    let bind_transforms = skin.hierarchical_transforms(&skin.joint_local_bind);
    bind_transforms[head_ji].transform_point3(Vec3::ZERO).y
}

struct ArmChain {
    shoulder: usize,
    upper: usize,
    forearm: usize,
    hand: usize,
}

/// Looks up an arm chain by trying two observed naming conventions:
/// Mixamo/Blender-style (`{Side}Shoulder` -> `{Side}Arm` -> `{Side}ForeArm`
/// -> `{Side}Hand`, e.g. "LeftShoulder") and this asset's own shorter style
/// (`{s}shoulder` -> `{s}arm1` -> `{s}arm2` -> `{s}hand`, e.g. "lshoulder").
/// Returns `None` if a skeleton uses neither, so callers can fall back to
/// leaving that arm in its bind pose instead of panicking.
fn find_arm_chain(skin: &GltfSkin, side: &str) -> Option<ArmChain> {
    let short = side.chars().next()?.to_ascii_lowercase();
    let shoulder = joint_index_any(
        skin,
        &[&format!("{side}Shoulder"), &format!("{short}shoulder")],
    )?;
    let upper = joint_index_any(skin, &[&format!("{side}Arm"), &format!("{short}arm1")])?;
    let forearm = joint_index_any(skin, &[&format!("{side}ForeArm"), &format!("{short}arm2")])?;
    let hand = joint_index_any(skin, &[&format!("{side}Hand"), &format!("{short}hand")])?;
    Some(ArmChain {
        shoulder,
        upper,
        forearm,
        hand,
    })
}

/// Aims the upper-arm and forearm bones at `target` (mesh-local space,
/// already divided by `root_scale` — see call site) via a 2-bone IK solve,
/// writing the result into `local`'s rotations for those two joints. Uses
/// each bone's own bind-pose child offset as the "aim from" direction, so
/// this works for any rig's bind orientation without hardcoding which axis
/// points "along the bone" — a hardcoded axis would silently produce
/// twisted limbs on a rig authored with different conventions.
fn apply_arm_ik(
    local: &mut [(Vec3, Quat, Vec3)],
    skin: &GltfSkin,
    bind_transforms: &[Mat4],
    chain: &ArmChain,
    target: Vec3,
) {
    let upper_len = skin.joint_local_bind[chain.forearm].0.length();
    let forearm_len = skin.joint_local_bind[chain.hand].0.length();
    if upper_len < 1e-5 || forearm_len < 1e-5 {
        return;
    }

    let (_, shoulder_rot, _) = bind_transforms[chain.shoulder].to_scale_rotation_translation();
    let shoulder_pos = bind_transforms[chain.upper].transform_point3(Vec3::ZERO);

    let elbow_pos = solve_arm(
        shoulder_pos,
        target,
        ArmLengths {
            upper: upper_len,
            forearm: forearm_len,
        },
    );

    let bind_dir_upper = skin.joint_local_bind[chain.forearm].0.normalize();
    let desired_dir_upper = (shoulder_rot.inverse() * (elbow_pos - shoulder_pos)).normalize_or_zero();
    if desired_dir_upper.length_squared() > 1e-8 {
        local[chain.upper].1 = Quat::from_rotation_arc(bind_dir_upper, desired_dir_upper);
    }

    let upper_world_rot = shoulder_rot * local[chain.upper].1;
    let bind_dir_forearm = skin.joint_local_bind[chain.hand].0.normalize();
    let desired_dir_forearm = (upper_world_rot.inverse() * (target - elbow_pos)).normalize_or_zero();
    if desired_dir_forearm.length_squared() > 1e-8 {
        local[chain.forearm].1 = Quat::from_rotation_arc(bind_dir_forearm, desired_dir_forearm);
    }
}

/// Full skinning-matrix set for one avatar body: legs/spine/head stay in
/// bind pose (no tracking data for them — the whole body's heading instead
/// comes from `root_rot`), arms are 2-bone-IK'd toward the tracked hand
/// positions when tracked, left in bind pose when not. `root_scale`
/// uniformly scales the whole skeleton (see `MAN_GLB_SCALE`) — hand IK
/// targets get divided by it internally so bone-length math stays in the
/// same (unscaled) units as the bind pose data.
///
/// Always builds the *full* body, head included — the local player's own
/// body is never actually drawn in their own direct view at all (see
/// `lib.rs`, which keeps it out of the main render list entirely and only
/// includes it for the mirror pass and other players' clients), so there's
/// no local near-clipping concern left to hide the head for.
pub fn body_skin_matrices(
    skin: &GltfSkin,
    root_pos: Vec3,
    root_rot: Quat,
    root_scale: f32,
    left_hand: Option<Transform>,
    right_hand: Option<Transform>,
) -> Vec<Mat4> {
    let mut local = skin.joint_local_bind.clone();
    let bind_transforms = skin.hierarchical_transforms(&skin.joint_local_bind);
    let root_rot_inv = root_rot.inverse();
    let scale = root_scale.max(1e-5);

    if let (Some(hand), Some(chain)) = (left_hand, find_arm_chain(skin, "Left")) {
        let target = (root_rot_inv * (hand.position - root_pos)) / scale;
        apply_arm_ik(&mut local, skin, &bind_transforms, &chain, target);
    }
    if let (Some(hand), Some(chain)) = (right_hand, find_arm_chain(skin, "Right")) {
        let target = (root_rot_inv * (hand.position - root_pos)) / scale;
        apply_arm_ik(&mut local, skin, &bind_transforms, &chain, target);
    }

    let final_transforms = skin.hierarchical_transforms(&local);
    let root = Mat4::from_scale_rotation_translation(Vec3::splat(root_scale), root_rot, root_pos);
    skin.inv_bind_mats
        .iter()
        .enumerate()
        .map(|(ji, inv_bind)| root * final_transforms[ji] * *inv_bind)
        .collect()
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
    fn body_root_transform_ignores_pitch_and_roll() {
        let base_yaw = 30.0_f32.to_radians();
        let head_level = Transform {
            position: Vec3::new(1.0, 1.7, 2.0),
            rotation: Quat::from_rotation_y(base_yaw),
        };
        let head_pitched = Transform {
            position: head_level.position,
            rotation: Quat::from_rotation_y(base_yaw) * Quat::from_rotation_x(60.0_f32.to_radians()),
        };

        let root_level = body_root_transform(head_level, 1.6);
        let root_pitched = body_root_transform(head_pitched, 1.6);

        let fwd_level = root_level.rotation * Vec3::NEG_Z;
        let fwd_pitched = root_pitched.rotation * Vec3::NEG_Z;

        assert!(
            fwd_level.distance(fwd_pitched) < 1e-4,
            "pitching the head should not change root orientation: {fwd_level:?} vs {fwd_pitched:?}"
        );
        assert!(
            fwd_level.y.abs() < 1e-5,
            "root forward should be perfectly horizontal, got y={}",
            fwd_level.y
        );
    }

    #[test]
    fn body_root_transform_drops_to_floor() {
        let head = Transform {
            position: Vec3::new(0.0, 1.7, 0.0),
            rotation: Quat::IDENTITY,
        };
        let root = body_root_transform(head, 1.6);
        assert!(
            (root.position.y - 0.1).abs() < 1e-4,
            "root should sit floor_drop below the head, got y={}",
            root.position.y
        );
    }
}
