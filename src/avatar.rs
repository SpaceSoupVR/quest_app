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

/// Tunable rig-calibration numbers that don't have a mathematically
/// "correct" value derivable from the mesh data alone — they depend on
/// which way *this specific rig's* bind pose happens to face, which side of
/// its bend plane counts as "forward," and how the OpenXR grip pose's own
/// axis convention happens to line up with its wrist bone. Getting these
/// right is inherently a "look at it on-headset and adjust" process (see
/// git history — several rounds of exactly that), so they're loaded from
/// `game/avatar_rig.json` (via [`load_rig_config`]) instead of being
/// hardcoded, letting them be retuned by editing and re-pushing that one
/// JSON file rather than recompiling and reinstalling the whole app.
/// Missing fields (or a missing/unparseable file) fall back to
/// [`RigConfig::default`]'s values, which match what was last confirmed
/// correct on-headset for `models/boy/boy.glb`.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(default)]
pub struct RigConfig {
    /// `solve_arm`'s `bend_hint` for arms — which side of the shoulder-hand
    /// axis the elbow bends toward.
    pub arm_bend_hint: [f32; 3],
    /// `solve_arm`'s `bend_hint` for legs — which side the knee bends
    /// toward.
    pub leg_bend_hint: [f32; 3],
    /// Fixed offset applied (in the hand's own local frame, i.e.
    /// post-multiplied) to the tracked controller rotation before it's used
    /// as the wrist bone's target orientation — see `wrist_calibration_offset`.
    /// Some such offset is inherent to wiring a *tracked controller's
    /// absolute orientation* onto *a bone from a different rig* (there's no
    /// reason the two agree on which local axis is "forward"). Two prior
    /// guesses for `models/boy/boy.glb` (before this was JSON-tunable): no
    /// offset reported "backwards"; 180° around local `NEG_Z` (roll around
    /// the arm's own length axis) reported "upside down" instead — a
    /// *different* symptom, the signature of overshooting past the right
    /// axis rather than the wrong category of fix. `Ry(π)⁻¹ * Rz(π)`
    /// reduces to `Rx(π)`, i.e. swapping the `Rz(π)` guess for `Ry(π)`
    /// changes the result by exactly a flip of both up/down and
    /// forward/back, consistent with "upside down" being one axis-flip
    /// away — hence the current default below.
    pub wrist_calibration_offset_axis: [f32; 3],
    pub wrist_calibration_offset_deg: f32,
    /// Additional roll around the *controller's own* forward axis (`NEG_Z`,
    /// applied before `wrist_calibration_offset_axis`/`_deg`'s flip — i.e.
    /// in the physical controller's own frame, not the post-flip one) —
    /// confirmed on-headset that the flip alone (`Rx(180)`) got the
    /// fingers/palm facing the right general directions but left the palm
    /// twisted; this corrects that residual twist without disturbing the
    /// flip that was already right.
    pub wrist_roll_deg: f32,
    /// Offset added to the tracked hand position, in the controller's own
    /// local frame (so "backward" always means "back along wherever the
    /// controller is currently pointing," not a fixed world direction),
    /// before it's used as the wrist's IK target. Confirmed needed
    /// on-headset — the grip pose OpenXR reports sits at the controller's
    /// handle, not at the physical wrist, so the avatar's hand rendered
    /// noticeably further forward than the real hand.
    pub wrist_position_offset: [f32; 3],
    /// Axis the Head joint tilts around for look up/down (see the head-tilt
    /// code in `body_skin_matrices`) and its sign (1.0 or -1.0 — flip if
    /// looking up tilts the head down or vice versa).
    pub head_pitch_axis: [f32; 3],
    pub head_pitch_sign: f32,
    /// Total curl angle (degrees), split evenly across a finger's 3 joints,
    /// at `HandCurl` amount `1.0` — see `apply_finger_curl`.
    pub finger_curl_max_deg: f32,
    /// `HandCurl::free_hand`'s thumb curl while resting on the thumb-stick,
    /// `[0, 1]`.
    pub thumb_touch_curl: f32,
    /// Max forward spine lean (degrees) at full crouch — see the crouch-lean
    /// code in `body_skin_matrices`.
    pub max_lean_deg: f32,
    /// Crouch depth (meters, how far `root_pos` has dropped below floor
    /// level) at which `max_lean_deg` is fully reached.
    pub full_lean_crouch_m: f32,
}

impl Default for RigConfig {
    fn default() -> Self {
        Self {
            arm_bend_hint: [0.0, -1.0, 0.0],
            leg_bend_hint: [0.0, 0.0, 1.0],
            wrist_calibration_offset_axis: [1.0, 0.0, 0.0],
            wrist_calibration_offset_deg: 180.0,
            wrist_roll_deg: 45.0,
            wrist_position_offset: [0.0, 0.0, 0.05],
            head_pitch_axis: [1.0, 0.0, 0.0],
            head_pitch_sign: -1.0,
            finger_curl_max_deg: 80.0,
            thumb_touch_curl: 0.35,
            max_lean_deg: 30.0,
            full_lean_crouch_m: 0.4,
        }
    }
}

impl RigConfig {
    pub fn arm_bend_hint(&self) -> Vec3 {
        Vec3::from_array(self.arm_bend_hint)
    }

    pub fn leg_bend_hint(&self) -> Vec3 {
        Vec3::from_array(self.leg_bend_hint)
    }

    /// Roll (in the controller's own frame) composed first, then the
    /// axis/angle flip — see the two fields' own doc comments for why each
    /// exists and which frame it's expressed in.
    pub fn wrist_calibration_offset(&self) -> Quat {
        let flip_axis = Vec3::from_array(self.wrist_calibration_offset_axis).normalize_or_zero();
        let flip = if flip_axis == Vec3::ZERO {
            Quat::IDENTITY
        } else {
            Quat::from_axis_angle(flip_axis, self.wrist_calibration_offset_deg.to_radians())
        };
        let roll = Quat::from_axis_angle(Vec3::NEG_Z, self.wrist_roll_deg.to_radians());
        flip * roll
    }

    pub fn wrist_position_offset(&self) -> Vec3 {
        Vec3::from_array(self.wrist_position_offset)
    }

    pub fn head_pitch_rotation(&self, pitch_rad: f32) -> Quat {
        let axis = Vec3::from_array(self.head_pitch_axis).normalize_or_zero();
        if axis == Vec3::ZERO {
            return Quat::IDENTITY;
        }
        Quat::from_axis_angle(axis, self.head_pitch_sign * pitch_rad)
    }
}

/// Reads and parses `path` as a `RigConfig`; falls back to
/// [`RigConfig::default`] (logging why) if the file is missing or
/// malformed, so a typo or not-yet-pushed file degrades to "last known
/// good" instead of a hard failure.
pub fn load_rig_config(path: &std::path::Path) -> RigConfig {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) => {
            log::warn!("avatar_rig.json not found at {} ({e}) — using defaults", path.display());
            return RigConfig::default();
        }
    };
    match serde_json::from_str(&text) {
        Ok(config) => config,
        Err(e) => {
            log::warn!("avatar_rig.json at {} failed to parse ({e}) — using defaults", path.display());
            RigConfig::default()
        }
    }
}

/// A real adult's standing head height should land in this range — clamps a
/// single bad/glitched tracking sample (e.g. exactly 0 before tracking has
/// settled after connecting) from producing a degenerate avatar scale.
const MIN_CALIBRATED_HEIGHT_M: f32 = 1.2;
const MAX_CALIBRATED_HEIGHT_M: f32 = 2.2;


/// Per-player avatar scale, calculated from that player's own real tracked
/// height rather than one fixed constant for everyone. `calibrated_height`
/// should be a running *maximum* of that player's observed (raw, HMD-
/// tracked, pre-locomotion-offset) head height — see call site — since a
/// crouch or momentary glitch reads low and only the tallest sample seen so
/// far is a trustworthy "standing" reference; there's no explicit
/// calibration step/gesture, this is a passive running estimate instead.
/// `raw_bind_head_height` is this skeleton's own unscaled bind-pose head
/// height (`bind_head_height`) — dividing by it is what makes the *result*
/// land at the player's real height regardless of this particular rig's
/// own proportions.
pub fn height_calibrated_scale(calibrated_height: f32, raw_bind_head_height: f32) -> f32 {
    calibrated_height.clamp(MIN_CALIBRATED_HEIGHT_M, MAX_CALIBRATED_HEIGHT_M) / raw_bind_head_height
}

/// Middle-joint position for a 2-bone (shoulder-elbow-hand, or hip-knee-
/// ankle) IK chain, via the standard law-of-cosines solve. The bend
/// direction is picked from a `bend_hint` projected perpendicular to the
/// shoulder/hip->target axis, so it's deterministic and doesn't flicker
/// frame to frame — arms bend with elbows pointing down (`Vec3::NEG_Y`),
/// legs bend with knees pointing forward (`Vec3::Z`, this rig's own
/// bind-pose forward — see `body_root_transform`), so this takes it as a
/// parameter rather than hardcoding one.
pub fn solve_arm(shoulder: Vec3, hand: Vec3, lengths: ArmLengths, bend_hint: Vec3) -> Vec3 {
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

    let bend_dir = perpendicular_component(bend_hint, axis)
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
        // models/boy/boy.glb's own bind pose faces +Z, not the -Z this
        // engine otherwise treats as "forward" (verified directly against
        // the file: bind-pose "Right toe" sits at z=+10.96 vs "Right
        // ankle"'s z=-3.53, i.e. toes point toward +Z) — aligning -Z here
        // instead would face the whole body 180 degrees backwards.
        Quat::from_rotation_arc(Vec3::Z, forward_h.normalize())
    };
    Transform {
        position: head.position - Vec3::Y * floor_drop,
        rotation,
    }
}

fn joint_index(joint_names: &[String], name: &str) -> Option<usize> {
    joint_names.iter().position(|n| n == name)
}

fn joint_index_any(joint_names: &[String], names: &[&str]) -> Option<usize> {
    names.iter().find_map(|n| joint_index(joint_names, n))
}

/// "Head" (Mixamo-style rigs) or "head" (this asset's own rig) — tried in
/// that order so this works for either naming convention without needing
/// to know in advance which one a given asset uses.
fn find_head_joint(joint_names: &[String]) -> Option<usize> {
    joint_index_any(joint_names, &["Head", "head"])
}

/// The head joint plus every joint beneath it in the hierarchy (hair, eyes,
/// etc.) — the full set of joints whose geometry should disappear when
/// hiding the head, since e.g. hair strands are their own joints, not part
/// of the head joint's own mesh weight. Falls back to an empty set if this
/// skeleton has no recognizable head joint, so callers just render
/// everything rather than guessing.
pub fn head_and_descendant_joints(skin: &GltfSkin) -> Vec<usize> {
    head_and_descendant_joints_of(&skin.joint_names, &skin.joint_parents)
}

fn head_and_descendant_joints_of(joint_names: &[String], joint_parents: &[Option<usize>]) -> Vec<usize> {
    let Some(head_ji) = find_head_joint(joint_names) else {
        return Vec::new();
    };
    let mut joints = vec![head_ji];
    // `joint_parents` only ever points a joint at an *earlier* ancestor (see
    // mesh.rs's loader), so a single forward pass already sees every
    // joint's parent decision before that joint could itself be a parent —
    // no need to loop until fixpoint.
    for ji in 0..joint_names.len() {
        if let Some(parent) = joint_parents[ji] {
            if joints.contains(&parent) {
                joints.push(ji);
            }
        }
    }
    joints
}

/// The rigged skeleton's own bind-pose head height (mesh-local, *before*
/// `root_scale` is applied — multiply by it at the call site), used so
/// `body_root_transform` can put the feet at floor level under the tracked
/// head regardless of this particular rig's proportions. Falls back to a
/// generic human height if this skeleton has no head joint under either
/// naming convention.
pub fn bind_head_height(skin: &GltfSkin) -> f32 {
    let Some(head_ji) = find_head_joint(&skin.joint_names) else {
        return 1.6;
    };
    let bind_transforms = skin.hierarchical_transforms(&skin.joint_local_bind);
    bind_transforms[head_ji].transform_point3(Vec3::ZERO).y
}

/// A generic 2-bone limb chain: `root` is the immovable parent joint whose
/// bind rotation the IK'd rotations are expressed relative to (shoulder for
/// an arm, hip socket for a leg), `upper`/`lower` are the two IK-rotated
/// joints (upper-arm/forearm, or thigh/shin), and `end` is the effector
/// joint used only for its bind-pose offset (forearm/shin length). Shared
/// by `find_arm_chain` and `find_leg_chain` — same law-of-cosines math
/// either way, see `apply_arm_ik`.
struct LimbChain {
    root: usize,
    upper: usize,
    lower: usize,
    end: usize,
}

/// Looks up an arm chain by trying three observed naming conventions:
/// Mixamo/Blender-style (`{Side}Shoulder` -> `{Side}Arm` -> `{Side}ForeArm`
/// -> `{Side}Hand`, e.g. "LeftShoulder"), this asset's own shorter style
/// (`{s}shoulder` -> `{s}arm1` -> `{s}arm2` -> `{s}hand`, e.g. "lshoulder"),
/// and a third, space-separated style (`{Side} shoulder` -> `{Side} arm` ->
/// `{Side} elbow` -> `{Side} wrist`, e.g. "Right shoulder" — `models/boy/
/// boy.glb`'s own rig; note its "elbow"/"wrist" joints play the forearm/hand
/// roles respectively, since each names the bone *starting* at that joint,
/// same convention as the other two styles). Returns `None` if a skeleton
/// uses none of the three, so callers can fall back to leaving that arm in
/// its bind pose instead of panicking.
fn find_arm_chain(joint_names: &[String], side: &str) -> Option<LimbChain> {
    let short = side.chars().next()?.to_ascii_lowercase();
    let root = joint_index_any(
        joint_names,
        &[
            &format!("{side}Shoulder"),
            &format!("{short}shoulder"),
            &format!("{side} shoulder"),
        ],
    )?;
    let upper = joint_index_any(
        joint_names,
        &[&format!("{side}Arm"), &format!("{short}arm1"), &format!("{side} arm")],
    )?;
    let lower = joint_index_any(
        joint_names,
        &[
            &format!("{side}ForeArm"),
            &format!("{short}arm2"),
            &format!("{side} elbow"),
        ],
    )?;
    let end = joint_index_any(
        joint_names,
        &[&format!("{side}Hand"), &format!("{short}hand"), &format!("{side} wrist")],
    )?;
    Some(LimbChain { root, upper, lower, end })
}

/// Looks up a leg chain: `Hips` (root/hip-socket) -> `{Side} leg` (thigh) ->
/// `{Side} knee` (shin) -> `{Side} ankle` (foot) — `models/boy/boy.glb`'s
/// own naming, the only convention this rig uses for legs (unlike arms,
/// which also try Mixamo/this-asset's-shorter styles, since no leg data
/// existed to observe those conventions against before now).
fn find_leg_chain(joint_names: &[String], side: &str) -> Option<LimbChain> {
    let root = joint_index(joint_names, "Hips")?;
    let upper = joint_index(joint_names, &format!("{side} leg"))?;
    let lower = joint_index(joint_names, &format!("{side} knee"))?;
    let end = joint_index(joint_names, &format!("{side} ankle"))?;
    Some(LimbChain { root, upper, lower, end })
}

/// Standalone equivalent of `GltfSkin::hierarchical_transforms`, taking
/// `joint_parents` directly instead of a full `GltfSkin` — which needs a
/// real `wgpu::Device` to construct, so functions that only need parent
/// relationships (not e.g. bind poses or GPU buffers) take `joint_parents`
/// as its own parameter instead, staying callable without one (including
/// from host-only unit tests).
fn hierarchical_transforms_of(local: &[(Vec3, Quat, Vec3)], joint_parents: &[Option<usize>]) -> Vec<Mat4> {
    let mut out = vec![Mat4::IDENTITY; local.len()];
    for ji in 0..local.len() {
        let (t, r, s) = local[ji];
        let local_mat = Mat4::from_scale_rotation_translation(s, r, t);
        out[ji] = match joint_parents[ji] {
            Some(pi) => out[pi] * local_mat,
            None => local_mat,
        };
    }
    out
}

/// Sets `local[ji].1` so this joint's *mesh-local* (i.e. pre-`root_rot`)
/// world orientation becomes `desired_mesh_local_rot`, given its exact
/// current ancestor chain — recomputed fresh from `local` every call rather
/// than assumed from the static bind pose, so it stays correct even when an
/// ancestor (e.g. the forearm, already aimed by `apply_arm_ik`, or the
/// spine mid-crouch-lean) has itself moved away from bind pose. This is
/// deliberately more rigorous than the `root_rot * local[chain.X].1`
/// shortcut used elsewhere in this file for *aiming* a bone roughly toward
/// a direction (which tolerates skipping intermediate ancestors just fine,
/// since a rough direction match still looks reasonable) — matching a full
/// absolute orientation doesn't tolerate that same slop, or the result
/// visibly twists the wrong way.
fn set_world_rotation(
    local: &mut [(Vec3, Quat, Vec3)],
    joint_parents: &[Option<usize>],
    ji: usize,
    desired_mesh_local_rot: Quat,
) {
    let current_hier = hierarchical_transforms_of(local, joint_parents);
    let ancestors_world_rot = match joint_parents[ji] {
        Some(pi) => current_hier[pi].to_scale_rotation_translation().1,
        None => Quat::IDENTITY,
    };
    local[ji].1 = ancestors_world_rot.inverse() * desired_mesh_local_rot;
}

/// Aims the upper and lower bones of a 2-bone limb chain — upper-arm/
/// forearm, or thigh/shin — at `target` (mesh-local space, already divided
/// by `root_scale` — see call site) via a law-of-cosines IK solve, writing
/// the result into `local`'s rotations for those two joints. Uses each
/// bone's own bind-pose child offset as the "aim from" direction, so this
/// works for any rig's bind orientation without hardcoding which axis
/// points "along the bone" — a hardcoded axis would silently produce
/// twisted limbs on a rig authored with different conventions.
fn apply_arm_ik(
    local: &mut [(Vec3, Quat, Vec3)],
    skin: &GltfSkin,
    bind_transforms: &[Mat4],
    chain: &LimbChain,
    target: Vec3,
    bend_hint: Vec3,
    end_rot: Option<Quat>,
) {
    let (_, root_rot, _) = bind_transforms[chain.root].to_scale_rotation_translation();
    let upper_pos = bind_transforms[chain.upper].transform_point3(Vec3::ZERO);
    let lower_pos = bind_transforms[chain.lower].transform_point3(Vec3::ZERO);
    let end_pos = bind_transforms[chain.end].transform_point3(Vec3::ZERO);

    // Measured via the *hierarchical* (root-to-joint) bind transforms, not
    // `joint_local_bind[...].0.length()`'s raw immediate-parent-relative
    // translation — `boy.glb`'s skin root ("Hips") bakes in a ~100x scale
    // from its non-joint "Armature" ancestor (see the loader's ancestor-mat
    // comment), which `upper_pos`/`target` (both hierarchical) reflect but a
    // raw local translation never picks up. Mixing an inflated `upper_pos`
    // with un-inflated lengths made `solve_arm`'s `dist` always clamp to
    // `max_reach`, i.e. a permanently near-fully-extended, barely-bending
    // arm regardless of the real target — confirmed on-headset via a
    // `bend_angle_deg` that stayed ~5 degrees no matter how far the
    // controller reached.
    let upper_len = (lower_pos - upper_pos).length();
    let lower_len = (end_pos - lower_pos).length();
    if upper_len < 1e-5 || lower_len < 1e-5 {
        return;
    }

    let mid_pos = solve_arm(
        upper_pos,
        target,
        ArmLengths {
            upper: upper_len,
            forearm: lower_len,
        },
        bend_hint,
    );

    // Aims each joint at its target by adding the *minimal extra* rotation
    // on top of that joint's own bind-pose local rotation, rather than
    // replacing it outright with `from_rotation_arc(raw_bind_offset,
    // desired)` — the latter silently assumes the bind-pose local rotation
    // is near-identity, which boy.glb's leg joints very much aren't ("Right
    // leg"/hip has a ~181 degree bind rotation, "Right ankle" ~65 degrees,
    // measured directly off the file); discarding that produced visibly
    // twisted legs. `Quat::from_rotation_arc(from, to) * bind_rot` still
    // maps the bone's current (already bind-rotated) direction to `to`
    // exactly, while leaving whatever twist/roll `bind_rot` carried around
    // that axis untouched — arms happen to have near-identity bind
    // rotations, so this is a no-op improvement for them, not a regression.
    let bind_rot_upper = skin.joint_local_bind[chain.upper].1;
    let bind_dir_upper = skin.joint_local_bind[chain.lower].0.normalize();
    let current_dir_upper = bind_rot_upper * bind_dir_upper;
    let desired_dir_upper = (root_rot.inverse() * (mid_pos - upper_pos)).normalize_or_zero();
    if desired_dir_upper.length_squared() > 1e-8 {
        local[chain.upper].1 = Quat::from_rotation_arc(current_dir_upper, desired_dir_upper) * bind_rot_upper;
    }

    let upper_world_rot = root_rot * local[chain.upper].1;
    let bind_rot_lower = skin.joint_local_bind[chain.lower].1;
    let bind_dir_lower = skin.joint_local_bind[chain.end].0.normalize();
    let current_dir_lower = bind_rot_lower * bind_dir_lower;
    let desired_dir_lower = (upper_world_rot.inverse() * (target - mid_pos)).normalize_or_zero();
    if desired_dir_lower.length_squared() > 1e-8 {
        local[chain.lower].1 = Quat::from_rotation_arc(current_dir_lower, desired_dir_lower) * bind_rot_lower;
    }

    // Wrist rotation: unlike the upper/lower bones above (which only *aim*
    // at a target position), a tracked controller also reports its own
    // orientation — twisting/tilting the hand should carry through to the
    // wrist bone directly, not just bend the elbow toward its position.
    // `None` for legs, which have no equivalent tracked foot/ankle
    // orientation to follow. `end_rot` is expected in the same mesh-local
    // (pre-`root_rot`) space `target` is in — see call site.
    if let Some(end_rot) = end_rot {
        set_world_rotation(local, &skin.joint_parents, chain.end, end_rot);
    }
}

/// One finger's bones, meta/proximal -> distal — `models/boy/boy.glb`'s own
/// fingers are exactly 3 joints each (no separate fingertip bone).
struct FingerChain([usize; 3]);

struct HandFingers {
    thumb: FingerChain,
    index: FingerChain,
    middle: FingerChain,
    ring: FingerChain,
    little: FingerChain,
}

/// Curl amount per finger, `[0, 1]` — index tracks the controller trigger,
/// thumb tracks thumb-rest touch, the rest track the grip squeeze (same
/// split the old hand.glb-specific free-hand heuristic used).
#[derive(Debug, Clone, Copy, Default)]
pub struct HandCurl {
    pub thumb: f32,
    pub index: f32,
    pub middle: f32,
    pub ring: f32,
    pub little: f32,
}

impl HandCurl {
    pub fn free_hand(trigger: f32, squeeze: f32, thumb_touch: bool, thumb_touch_curl: f32) -> Self {
        let squeeze = squeeze.clamp(0.0, 1.0);
        Self {
            thumb: if thumb_touch { thumb_touch_curl } else { 0.0 },
            index: trigger.clamp(0.0, 1.0),
            middle: squeeze,
            ring: squeeze,
            little: squeeze,
        }
    }

    /// Approximates gripping a held object: all five fingers close together
    /// on the squeeze amount, rather than the trigger/thumb-touch split
    /// `free_hand` uses for an empty hand (holding something moves every
    /// finger, not just the ones a free hand's trigger/thumb-rest track).
    /// A held object's *authored* per-joint grip curl (`grip_pose_*.
    /// finger_curl`, e.g. the rifle's) was hand-tuned for hand.glb's own
    /// joint layout — porting that exactly onto boy.glb's differently-named
    /// bones isn't done here, so this is a reasonable uniform approximation
    /// rather than that exact authored pose.
    pub fn held(squeeze: f32) -> Self {
        let squeeze = squeeze.clamp(0.0, 1.0);
        Self {
            thumb: squeeze,
            index: squeeze,
            middle: squeeze,
            ring: squeeze,
            little: squeeze,
        }
    }
}

/// `models/boy/boy.glb`'s own finger joint naming: `Thumb0/1/2_{R,L}`,
/// `{Index,Middle,Ring,Little}Finger{1,2,3}_{R,L}`.
fn find_hand_fingers(joint_names: &[String], side: &str) -> Option<HandFingers> {
    let suffix = if side == "Right" { "_R" } else { "_L" };
    let idx = |name: String| joint_index(joint_names, &name);
    Some(HandFingers {
        thumb: FingerChain([
            idx(format!("Thumb0{suffix}"))?,
            idx(format!("Thumb1{suffix}"))?,
            idx(format!("Thumb2{suffix}"))?,
        ]),
        index: FingerChain([
            idx(format!("IndexFinger1{suffix}"))?,
            idx(format!("IndexFinger2{suffix}"))?,
            idx(format!("IndexFinger3{suffix}"))?,
        ]),
        middle: FingerChain([
            idx(format!("MiddleFinger1{suffix}"))?,
            idx(format!("MiddleFinger2{suffix}"))?,
            idx(format!("MiddleFinger3{suffix}"))?,
        ]),
        ring: FingerChain([
            idx(format!("RingFinger1{suffix}"))?,
            idx(format!("RingFinger2{suffix}"))?,
            idx(format!("RingFinger3{suffix}"))?,
        ]),
        little: FingerChain([
            idx(format!("LittleFinger1{suffix}"))?,
            idx(format!("LittleFinger2{suffix}"))?,
            idx(format!("LittleFinger3{suffix}"))?,
        ]),
    })
}

/// Curls `fingers` toward a fist by rotating each joint around the normal
/// of the plane the spread fingers lie in at bind pose (derived from this
/// rig's own index/ring/wrist bind positions — measured directly against
/// `models/boy/boy.glb`'s real bind pose, not assumed: a positive rotation
/// around this axis reliably pulls the fingertip closer to the wrist for
/// both hands using the identical sign, i.e. this self-corrects for
/// left/right mirroring without needing a separate per-side sign flip).
/// There's no baked "fist" animation clip to blend to instead (boy.glb ships
/// with zero animations) — this is the only way to curl these bones at all.
fn apply_finger_curl(
    local: &mut [(Vec3, Quat, Vec3)],
    bind_transforms: &[Mat4],
    joint_parents: &[Option<usize>],
    fingers: &HandFingers,
    wrist_ji: usize,
    curl: HandCurl,
    finger_curl_max_deg: f32,
) {
    // Recomputed fresh from `local` (rather than reusing the static
    // `bind_transforms`) so a finger's parent orientation — chiefly the
    // wrist, which now tracks the controller's real rotation instead of
    // staying at bind pose — is accounted for. Using the stale bind-pose
    // orientation here made fingers curl in the wrong plane (e.g. "up"
    // instead of "inward toward the palm") any time the wrist itself was
    // rotated away from bind pose, i.e. essentially always once wrist
    // tracking was added. A standalone function taking `joint_parents`
    // directly, rather than `GltfSkin::hierarchical_transforms`, so this
    // stays callable (and testable) without a full `GltfSkin` — which needs
    // a real `wgpu::Device` to construct — the same reason `joint_parents`
    // is already threaded through as its own parameter instead of `&GltfSkin`.
    let current_hier = hierarchical_transforms_of(local, joint_parents);
    let wrist_pos = bind_transforms[wrist_ji].transform_point3(Vec3::ZERO);
    let index_pos = bind_transforms[fingers.index.0[0]].transform_point3(Vec3::ZERO);
    let ring_pos = bind_transforms[fingers.ring.0[0]].transform_point3(Vec3::ZERO);
    let bend_axis_world = (index_pos - wrist_pos).cross(ring_pos - wrist_pos).normalize_or_zero();
    if bend_axis_world == Vec3::ZERO {
        return;
    }

    // On-headset testing showed `bend_axis_world`'s sign (derived from
    // index/ring/wrist) curls the thumb correctly but the other four
    // fingers backwards — the thumb's bind-pose bend plane sits roughly
    // perpendicular to the other fingers' shared plane, so the same shared
    // axis' sign happens to land right for it and wrong for the rest,
    // rather than either being uniformly right or wrong. Per-finger sign
    // flip rather than picking a single "corrected" axis, since flipping
    // the axis outright would just reverse which of the two groups is wrong.
    for (chain, amount, sign) in [
        (&fingers.thumb, curl.thumb, 1.0),
        (&fingers.index, curl.index, -1.0),
        (&fingers.middle, curl.middle, -1.0),
        (&fingers.ring, curl.ring, -1.0),
        (&fingers.little, curl.little, -1.0),
    ] {
        let per_joint_deg = amount.clamp(0.0, 1.0) * finger_curl_max_deg / chain.0.len() as f32 * sign;
        if per_joint_deg == 0.0 {
            continue;
        }
        for &ji in &chain.0 {
            let Some(parent) = joint_parents[ji] else { continue };
            let (_, parent_rot, _) = current_hier[parent].to_scale_rotation_translation();
            let local_axis = (parent_rot.inverse() * bend_axis_world).normalize_or_zero();
            if local_axis == Vec3::ZERO {
                continue;
            }
            local[ji].1 = Quat::from_axis_angle(local_axis, per_joint_deg.to_radians()) * local[ji].1;
        }
    }
}

/// Full skinning-matrix set for one avatar body: spine stays in bind pose
/// except for crouch lean (see below), arms are 2-bone-IK'd toward the
/// tracked hand positions and rotated to match the tracked hand orientation
/// when tracked (left in bind pose when not), and legs are always
/// 2-bone-IK'd toward a floor-planted foot target under each hip (see the
/// loop below) — there's no per-hand "not tracked" case for legs since
/// there's no leg tracking hardware to begin with, only ever this same
/// floor-plant target. `root_scale` uniformly scales the whole skeleton (see
/// `height_calibrated_scale`, called at the call site — every player gets
/// their own scale from their own tracked height, not one fixed constant) —
/// IK targets get divided by it internally so bone-length math stays in the
/// same (unscaled) units as the bind pose data.
///
/// `root_rot` is yaw-only (see `body_root_transform`) — the whole body's
/// heading — while `head_rot` is the *real*, un-stripped tracked head
/// rotation, in that same render space; the difference between the two
/// (computed below) is applied to the Neck/Head joints so looking up/down
/// or tilting your head visibly tilts the neck instead of just reorienting
/// the whole body.
///
/// Hiding the local player's own head for their first-person view is done at
/// the mesh level instead (see `head_and_descendant_joints` and
/// `clone_with_independent_skin_excluding_joints`, called from `lib.rs` when
/// building that separate direct-view mesh instance) — head/hair/eye
/// triangles are dropped outright, so this function always produces the
/// full-body pose and doesn't need to know which mesh it's driving.
///
/// `config` holds the rig-specific calibration numbers (bend hint signs,
/// wrist offset, head tilt axis) that don't have one mathematically
/// "correct" value — see [`RigConfig`] and `game/avatar_rig.json`.
pub fn body_skin_matrices(
    skin: &GltfSkin,
    config: &RigConfig,
    root_pos: Vec3,
    root_rot: Quat,
    head_rot: Quat,
    root_scale: f32,
    left_hand: Option<Transform>,
    right_hand: Option<Transform>,
    left_curl: Option<HandCurl>,
    right_curl: Option<HandCurl>,
) -> Vec<Mat4> {
    let mut local = skin.joint_local_bind.clone();
    let bind_transforms = skin.hierarchical_transforms(&skin.joint_local_bind);
    let root_rot_inv = root_rot.inverse();
    let scale = root_scale.max(1e-5);

    if let (Some(hand), Some(chain)) = (left_hand, find_arm_chain(&skin.joint_names, "Left")) {
        let hand_pos_calibrated = hand.position + hand.rotation * config.wrist_position_offset();
        let target = (root_rot_inv * (hand_pos_calibrated - root_pos)) / scale;
        let hand_rot = root_rot_inv * hand.rotation * config.wrist_calibration_offset();
        apply_arm_ik(&mut local, skin, &bind_transforms, &chain, target, config.arm_bend_hint(), Some(hand_rot));
        if let (Some(curl), Some(fingers)) = (left_curl, find_hand_fingers(&skin.joint_names, "Left")) {
            apply_finger_curl(&mut local, &bind_transforms, &skin.joint_parents, &fingers, chain.end, curl, config.finger_curl_max_deg);
        }
    }
    if let (Some(hand), Some(chain)) = (right_hand, find_arm_chain(&skin.joint_names, "Right")) {
        let hand_pos_calibrated = hand.position + hand.rotation * config.wrist_position_offset();
        let target = (root_rot_inv * (hand_pos_calibrated - root_pos)) / scale;
        let hand_rot = root_rot_inv * hand.rotation * config.wrist_calibration_offset();
        apply_arm_ik(&mut local, skin, &bind_transforms, &chain, target, config.arm_bend_hint(), Some(hand_rot));
        if let (Some(curl), Some(fingers)) = (right_curl, find_hand_fingers(&skin.joint_names, "Right")) {
            apply_finger_curl(&mut local, &bind_transforms, &skin.joint_parents, &fingers, chain.end, curl, config.finger_curl_max_deg);
        }
    }

    // Legs: no tracking hardware gives us knee/ankle data on Quest, so each
    // foot is targeted straight down from its own (possibly crouch-lowered,
    // since `root_pos` follows head height) hip to floor level in render
    // space — the same convention the rest of the scene's floor geometry
    // uses. The knee bends to close whatever gap crouching opens up between
    // the lowered hip and the floor-planted foot; without this the legs
    // stayed rigid in bind pose and the whole body just sank through the
    // floor as the head dropped.
    for side in ["Left", "Right"] {
        let Some(chain) = find_leg_chain(&skin.joint_names, side) else { continue };
        let hip_pos = bind_transforms[chain.upper].transform_point3(Vec3::ZERO);
        let hip_world = root_pos + root_rot * (hip_pos * scale);
        let foot_target_world = Vec3::new(hip_world.x, 0.0, hip_world.z);
        let target = (root_rot_inv * (foot_target_world - root_pos)) / scale;
        apply_arm_ik(&mut local, skin, &bind_transforms, &chain, target, config.leg_bend_hint(), None);
    }

    // Crouch-driven spine lean: no waist/torso tracking exists either, so
    // this approximates a natural forward hunch purely from the same
    // crouch signal already driving the leg IK above — how far `root_pos`
    // (which follows head height, minus the fixed standing floor_drop) has
    // dropped below floor level. Split across Spine + Chest (both present
    // in this rig, both near-identity bind rotations around local X —
    // measured directly off the file) for a smoother bend than
    // concentrating it in one joint, same idea as `apply_finger_curl`
    // spreading curl across a finger's joints. Composed on top of each
    // joint's own bind rotation, not replacing it, for the same reason
    // `apply_arm_ik` does — see its own comment on the twist bug this
    // avoids repeating.
    let crouch = (-root_pos.y).max(0.0);
    let lean_deg =
        (crouch / config.full_lean_crouch_m * config.max_lean_deg).min(config.max_lean_deg);
    if lean_deg > 0.0 {
        let torso_joints: Vec<usize> = ["Spine", "Chest"]
            .iter()
            .filter_map(|n| joint_index(&skin.joint_names, n))
            .collect();
        if !torso_joints.is_empty() {
            let per_joint_deg = lean_deg / torso_joints.len() as f32;
            let lean_rot = Quat::from_axis_angle(Vec3::X, per_joint_deg.to_radians());
            for ji in torso_joints {
                local[ji].1 = lean_rot * skin.joint_local_bind[ji].1;
            }
        }
    }

    // Head tilt: `head_rot` carries the *real* pitch the yaw-only `root_rot`
    // deliberately drops (see `body_root_transform`) — tilting the Head
    // joint to match means looking up/down visibly tilts the neck, instead
    // of that motion only ever reorienting the whole body's heading.
    // Deliberately pitch-only (roll dropped entirely) via a single fixed
    // axis, same approach as the crouch lean above — composing the *full*
    // tracked rotation (including roll) via `set_world_rotation` was tried
    // first and produced a visibly twisted neck; a full-orientation match
    // has no protection against an unconstrained roll component the way a
    // single known axis does, the same reasoning `apply_arm_ik`'s
    // direction-based aiming uses to avoid twisted limbs.
    //
    // Axis/sign for the tilt itself come from `config.head_pitch_axis`/
    // `head_pitch_sign` — confirmed correct on-headset for the current
    // defaults (local +X, sign -1.0).
    if let Some(head_ji) = find_head_joint(&skin.joint_names) {
        let head_forward = head_rot * Vec3::NEG_Z;
        let pitch_rad = head_forward.y.clamp(-1.0, 1.0).asin();
        let pitch_rot = config.head_pitch_rotation(pitch_rad);
        local[head_ji].1 = pitch_rot * skin.joint_local_bind[head_ji].1;
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
    fn load_rig_config_parses_a_partial_override() {
        let dir = std::env::temp_dir().join(format!("rig_config_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("avatar_rig.json");
        std::fs::write(&path, r#"{"leg_bend_hint": [1.0, 2.0, 3.0]}"#).unwrap();

        let config = load_rig_config(&path);
        assert_eq!(config.leg_bend_hint(), Vec3::new(1.0, 2.0, 3.0), "overridden field should parse");
        assert_eq!(
            config.arm_bend_hint(),
            RigConfig::default().arm_bend_hint(),
            "fields absent from the JSON should fall back to defaults, not zero"
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn load_rig_config_falls_back_to_defaults_for_a_missing_file() {
        let config = load_rig_config(std::path::Path::new("/nonexistent/avatar_rig.json"));
        assert_eq!(config.arm_bend_hint(), RigConfig::default().arm_bend_hint());
    }

    #[test]
    fn load_rig_config_falls_back_to_defaults_for_malformed_json() {
        let dir = std::env::temp_dir().join(format!("rig_config_test_malformed_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("avatar_rig.json");
        std::fs::write(&path, "{ not valid json").unwrap();

        let config = load_rig_config(&path);
        assert_eq!(config.arm_bend_hint(), RigConfig::default().arm_bend_hint());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn wrist_calibration_offset_matches_configured_axis_and_angle() {
        let config = RigConfig {
            wrist_calibration_offset_axis: [0.0, 0.0, 1.0],
            wrist_calibration_offset_deg: 90.0,
            wrist_roll_deg: 0.0,
            ..RigConfig::default()
        };
        let offset = config.wrist_calibration_offset();
        let rotated = offset * Vec3::X;
        assert!(
            rotated.distance(Vec3::Y) < 1e-4,
            "90 degrees around Z should rotate +X to +Y, got {rotated:?}"
        );
    }

    #[test]
    fn wrist_calibration_offset_composes_roll_before_flip() {
        let config = RigConfig {
            wrist_calibration_offset_axis: [0.0, 1.0, 0.0],
            wrist_calibration_offset_deg: 0.0,
            wrist_roll_deg: 90.0,
            ..RigConfig::default()
        };
        // No flip, just a 90-degree roll around the controller's own
        // forward (`NEG_Z`) — should rotate +X to +Y, same law as any
        // right-hand-rule rotation around that axis.
        let offset = config.wrist_calibration_offset();
        let rotated = offset * Vec3::X;
        assert!(
            rotated.distance(Vec3::NEG_Y) < 1e-4,
            "90 degrees around NEG_Z should rotate +X to -Y, got {rotated:?}"
        );
    }

    #[test]
    fn wrist_position_offset_reads_back_configured_vector() {
        let config = RigConfig {
            wrist_position_offset: [0.0, 0.0, 0.05],
            ..RigConfig::default()
        };
        assert_eq!(config.wrist_position_offset(), Vec3::new(0.0, 0.0, 0.05));
    }

    #[test]
    fn find_arm_chain_matches_boy_glb_space_separated_naming() {
        // Exact joint names measured off models/boy/boy.glb's real skin.
        let joint_names: Vec<String> = [
            "Hips", "Right leg", "Right knee", "Right ankle", "Right toe", "Left leg",
            "Left knee", "Left ankle", "Left toe", "Spine", "Chest", "Right shoulder",
            "Right arm", "Right elbow", "Right wrist", "Left shoulder", "Left arm",
            "Left elbow", "Left wrist", "Neck", "Head",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let right = find_arm_chain(&joint_names, "Right").expect("Right arm chain should resolve");
        assert_eq!(joint_names[right.root], "Right shoulder");
        assert_eq!(joint_names[right.upper], "Right arm");
        assert_eq!(joint_names[right.lower], "Right elbow");
        assert_eq!(joint_names[right.end], "Right wrist");

        let left = find_arm_chain(&joint_names, "Left").expect("Left arm chain should resolve");
        assert_eq!(joint_names[left.root], "Left shoulder");
        assert_eq!(joint_names[left.upper], "Left arm");
        assert_eq!(joint_names[left.lower], "Left elbow");
        assert_eq!(joint_names[left.end], "Left wrist");
    }

    #[test]
    fn find_leg_chain_matches_boy_glb_naming() {
        let joint_names: Vec<String> = [
            "Hips", "Right leg", "Right knee", "Right ankle", "Right toe", "Left leg",
            "Left knee", "Left ankle", "Left toe",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let right = find_leg_chain(&joint_names, "Right").expect("Right leg chain should resolve");
        assert_eq!(joint_names[right.root], "Hips");
        assert_eq!(joint_names[right.upper], "Right leg");
        assert_eq!(joint_names[right.lower], "Right knee");
        assert_eq!(joint_names[right.end], "Right ankle");

        let left = find_leg_chain(&joint_names, "Left").expect("Left leg chain should resolve");
        assert_eq!(joint_names[left.upper], "Left leg");
        assert_eq!(joint_names[left.lower], "Left knee");
        assert_eq!(joint_names[left.end], "Left ankle");
    }

    #[test]
    fn find_leg_chain_returns_none_for_an_unrecognized_rig() {
        let joint_names: Vec<String> = ["Hips", "Spine"].iter().map(|s| s.to_string()).collect();
        assert!(find_leg_chain(&joint_names, "Left").is_none());
    }

    /// Exact right-hand joint names (and bind-pose order) measured off
    /// `models/boy/boy.glb`'s real skin, wrist first.
    fn boy_glb_right_hand_joint_names() -> Vec<String> {
        [
            "Right wrist", "Thumb0_R", "Thumb1_R", "Thumb2_R", "RingFinger1_R", "RingFinger2_R",
            "RingFinger3_R", "MiddleFinger1_R", "MiddleFinger2_R", "MiddleFinger3_R",
            "LittleFinger1_R", "LittleFinger2_R", "LittleFinger3_R", "IndexFinger1_R",
            "IndexFinger2_R", "IndexFinger3_R",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    #[test]
    fn find_hand_fingers_matches_boy_glb_naming() {
        let joint_names = boy_glb_right_hand_joint_names();
        let fingers = find_hand_fingers(&joint_names, "Right").expect("Right hand fingers should resolve");
        assert_eq!(joint_names[fingers.thumb.0[0]], "Thumb0_R");
        assert_eq!(joint_names[fingers.thumb.0[2]], "Thumb2_R");
        assert_eq!(joint_names[fingers.index.0[0]], "IndexFinger1_R");
        assert_eq!(joint_names[fingers.middle.0[1]], "MiddleFinger2_R");
        assert_eq!(joint_names[fingers.ring.0[2]], "RingFinger3_R");
        assert_eq!(joint_names[fingers.little.0[0]], "LittleFinger1_R");
    }

    #[test]
    fn find_hand_fingers_returns_none_for_an_unrecognized_rig() {
        let joint_names: Vec<String> = ["Hips", "Right wrist"].iter().map(|s| s.to_string()).collect();
        assert!(find_hand_fingers(&joint_names, "Right").is_none());
    }

    #[test]
    fn apply_finger_curl_pulls_fingertip_closer_to_the_wrist() {
        // Real bind-pose (mesh-space) positions measured directly off
        // models/boy/boy.glb via GltfMesh::load + hierarchical_transforms —
        // not invented — so this test protects against silently regressing
        // to a bend axis that doesn't actually curl the rig.
        let joint_names = boy_glb_right_hand_joint_names();
        let wrist_pos = Vec3::new(-0.67543733, -0.031071994, 1.5096515);
        let index1_pos = Vec3::new(-0.7532355, -0.057480514, 1.5156413);
        let index2_pos = index1_pos + Vec3::new(-0.058, -0.010, 0.0); // approx bone continuation
        let ring1_pos = Vec3::new(-0.7566287, -0.018173985, 1.5204992);

        // joint_parents indices line up with boy_glb_right_hand_joint_names():
        // 0=wrist(no parent modeled here), 1..3=thumb chain, 13..15=index chain.
        let joint_parents: Vec<Option<usize>> =
            vec![None, Some(0), Some(1), Some(2), Some(0), Some(4), Some(5), Some(0), Some(7), Some(8), Some(0), Some(10), Some(11), Some(0), Some(13), Some(14)];

        let mut bind_transforms = vec![Mat4::IDENTITY; joint_names.len()];
        bind_transforms[0] = Mat4::from_translation(wrist_pos);
        bind_transforms[13] = Mat4::from_translation(index1_pos);
        bind_transforms[14] = Mat4::from_translation(index2_pos);
        bind_transforms[4] = Mat4::from_translation(ring1_pos);

        let fingers = find_hand_fingers(&joint_names, "Right").unwrap();
        let mut local = vec![(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE); joint_names.len()];
        local[13] = (index2_pos - index1_pos, Quat::IDENTITY, Vec3::ONE);

        apply_finger_curl(
            &mut local,
            &bind_transforms,
            &joint_parents,
            &fingers,
            0,
            HandCurl { thumb: 0.0, index: 1.0, middle: 0.0, ring: 0.0, little: 0.0 },
            RigConfig::default().finger_curl_max_deg,
        );

        let curled_dir = local[13].1 * (index2_pos - index1_pos).normalize();
        let curled_tip = index1_pos + curled_dir * (index2_pos - index1_pos).length();
        assert!(
            curled_tip.distance(wrist_pos) < index2_pos.distance(wrist_pos),
            "curling the index finger should pull the fingertip closer to the wrist"
        );
    }

    #[test]
    fn find_arm_chain_still_matches_mixamo_style_naming() {
        let joint_names: Vec<String> = ["Hips", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let chain = find_arm_chain(&joint_names, "Left").expect("Mixamo-style chain should still resolve");
        assert_eq!(joint_names[chain.end], "LeftHand");
    }

    #[test]
    fn find_arm_chain_returns_none_for_an_unrecognized_rig() {
        let joint_names: Vec<String> = ["Hips", "Spine", "SomeOtherConvention_L_UpperArm"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(find_arm_chain(&joint_names, "Left").is_none());
    }

    #[test]
    fn find_head_joint_matches_boy_glb() {
        let joint_names: Vec<String> = ["Neck", "Head"].iter().map(|s| s.to_string()).collect();
        assert_eq!(find_head_joint(&joint_names), Some(1));
    }

    #[test]
    fn head_and_descendant_joints_includes_grandchildren_but_not_unrelated_joints() {
        // Hips(0) -> Spine(1) -> Neck(2) -> Head(3) -> HairA(4) -> HairA_tip(5)
        //                                          \-> Eye(6)
        // Hips(0) -> Right leg(7) — unrelated branch, should never be included.
        let joint_names: Vec<String> = [
            "Hips", "Spine", "Neck", "Head", "HairA", "HairA_tip", "Eye", "Right leg",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let joint_parents: Vec<Option<usize>> = vec![
            None,
            Some(0),
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(3),
            Some(0),
        ];

        let mut hidden = head_and_descendant_joints_of(&joint_names, &joint_parents);
        hidden.sort_unstable();
        assert_eq!(hidden, vec![3, 4, 5, 6], "should include Head and every joint beneath it, nothing else");
    }

    #[test]
    fn head_and_descendant_joints_of_is_empty_for_an_unrecognized_rig() {
        let joint_names: Vec<String> = ["Hips", "Spine"].iter().map(|s| s.to_string()).collect();
        let joint_parents: Vec<Option<usize>> = vec![None, Some(0)];
        assert!(head_and_descendant_joints_of(&joint_names, &joint_parents).is_empty());
    }

    #[test]
    fn solve_arm_matches_law_of_cosines() {
        let shoulder = Vec3::ZERO;
        let hand = Vec3::new(0.0, 0.0, -0.5);
        let lengths = ArmLengths {
            upper: 0.28,
            forearm: 0.26,
        };

        let elbow = solve_arm(shoulder, hand, lengths, Vec3::NEG_Y);

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

        let elbow = solve_arm(shoulder, hand, lengths, Vec3::NEG_Y);

        assert!(elbow.is_finite(), "elbow position should stay finite: {elbow:?}");
        assert!(
            (shoulder.distance(elbow) - lengths.upper).abs() < 1e-3,
            "should still respect upper arm length when clamped"
        );
    }

    /// Regression test for the exact bug found on-headset: boy.glb's "Right
    /// leg" (hip) joint has a bind-pose local rotation of ~181.5 degrees
    /// (measured directly off the file) — nowhere near identity. Aiming a
    /// joint by outright replacing its rotation with
    /// `from_rotation_arc(raw_unrotated_offset, desired)` silently assumes
    /// near-identity bind rotation and discarded that ~181.5 degrees
    /// entirely, producing a visibly twisted leg. This exercises the same
    /// "add the minimal corrective rotation on top of the bind rotation"
    /// formula `apply_arm_ik` now uses, standalone (no GltfSkin/wgpu device
    /// needed, unlike `apply_arm_ik` itself).
    #[test]
    fn preserving_bind_rotation_still_aims_at_the_target_despite_large_twist() {
        let bind_dir = Vec3::Y; // raw, unrotated child-offset direction
        let bind_rot = Quat::from_axis_angle(
            Vec3::new(0.9995, -0.0048, 0.0314).normalize(),
            181.521_f32.to_radians(),
        );
        let current_dir = bind_rot * bind_dir;

        let desired_dir = Vec3::new(0.3, -0.9, 0.2).normalize();
        let new_rot = Quat::from_rotation_arc(current_dir, desired_dir) * bind_rot;

        let result_dir = new_rot * bind_dir;
        assert!(
            result_dir.distance(desired_dir) < 1e-4,
            "bone should end up aimed at the desired direction despite the large bind rotation, got {result_dir:?} vs {desired_dir:?}"
        );

        // Guard against silently reverting to the old (broken) formula: for
        // a bind rotation this far from identity, replacing it outright
        // gives a visibly different (and wrong) result.
        let naive_rot = Quat::from_rotation_arc(bind_dir, desired_dir);
        assert!(
            naive_rot.angle_between(new_rot) > 1.0_f32.to_radians(),
            "this test's bind rotation should be large enough that preserving it matters"
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

    #[test]
    fn height_calibrated_scale_matches_a_tall_and_a_short_player() {
        let raw_bind_head_height = 1.6619; // boy.glb's own, in meters (166.19cm)

        let tall = height_calibrated_scale(1.9, raw_bind_head_height);
        assert!(
            (tall * raw_bind_head_height - 1.9).abs() < 1e-4,
            "scaling the rig by this factor should land its own head height at the tall player's 1.9m"
        );

        let short = height_calibrated_scale(1.4, raw_bind_head_height);
        assert!(
            (short * raw_bind_head_height - 1.4).abs() < 1e-4,
            "scaling the rig by this factor should land its own head height at the short player's 1.4m"
        );
        assert!(short < tall, "a shorter player should get a smaller avatar scale");
    }

    #[test]
    fn height_calibrated_scale_clamps_glitched_readings() {
        let raw_bind_head_height = 1.6619;
        // A tracking glitch (or the brief window before tracking settles)
        // reading exactly 0 shouldn't collapse the avatar to zero size.
        let scale = height_calibrated_scale(0.0, raw_bind_head_height);
        assert!(
            scale * raw_bind_head_height >= MIN_CALIBRATED_HEIGHT_M - 1e-4,
            "a bogus zero reading should clamp to the minimum sane height, got scale={scale}"
        );
    }
}
