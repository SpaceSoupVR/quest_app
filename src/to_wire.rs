
use space_soup_engine::{FingerJoint, Hand, InputFrame, JointId, LocomotionInput, PlayerRig, TeleportTarget, Transform};
use space_soup_protocol::{
    Pose, WireFingerJoint, WireHand, WireInputFrame, WireJointId, WireLocomotionInput,
    WirePlayerRig, WireTeleportTarget,
};

fn hand_to_wire(h: Hand) -> WireHand {
    match h {
        Hand::Left => WireHand::Left,
        Hand::Right => WireHand::Right,
    }
}

fn finger_joint_to_wire(j: FingerJoint) -> WireFingerJoint {
    match j {
        FingerJoint::Palm => WireFingerJoint::Palm,
        FingerJoint::Wrist => WireFingerJoint::Wrist,
        FingerJoint::ThumbMeta => WireFingerJoint::ThumbMeta,
        FingerJoint::ThumbProx => WireFingerJoint::ThumbProx,
        FingerJoint::ThumbDist => WireFingerJoint::ThumbDist,
        FingerJoint::ThumbTip => WireFingerJoint::ThumbTip,
        FingerJoint::IndexMeta => WireFingerJoint::IndexMeta,
        FingerJoint::IndexProx => WireFingerJoint::IndexProx,
        FingerJoint::IndexInter => WireFingerJoint::IndexInter,
        FingerJoint::IndexDist => WireFingerJoint::IndexDist,
        FingerJoint::IndexTip => WireFingerJoint::IndexTip,
        FingerJoint::MiddleMeta => WireFingerJoint::MiddleMeta,
        FingerJoint::MiddleProx => WireFingerJoint::MiddleProx,
        FingerJoint::MiddleInter => WireFingerJoint::MiddleInter,
        FingerJoint::MiddleDist => WireFingerJoint::MiddleDist,
        FingerJoint::MiddleTip => WireFingerJoint::MiddleTip,
        FingerJoint::RingMeta => WireFingerJoint::RingMeta,
        FingerJoint::RingProx => WireFingerJoint::RingProx,
        FingerJoint::RingInter => WireFingerJoint::RingInter,
        FingerJoint::RingDist => WireFingerJoint::RingDist,
        FingerJoint::RingTip => WireFingerJoint::RingTip,
        FingerJoint::LittleMeta => WireFingerJoint::LittleMeta,
        FingerJoint::LittleProx => WireFingerJoint::LittleProx,
        FingerJoint::LittleInter => WireFingerJoint::LittleInter,
        FingerJoint::LittleDist => WireFingerJoint::LittleDist,
        FingerJoint::LittleTip => WireFingerJoint::LittleTip,
    }
}

fn joint_id_to_wire(j: JointId) -> WireJointId {
    match j {
        JointId::Head => WireJointId::Head,
        JointId::HandGrip(h) => WireJointId::HandGrip(hand_to_wire(h)),
        JointId::HandAim(h) => WireJointId::HandAim(hand_to_wire(h)),
        JointId::Finger(h, fj) => WireJointId::Finger(hand_to_wire(h), finger_joint_to_wire(fj)),
    }
}

fn transform_to_wire(t: Transform) -> Pose {
    Pose {
        position: t.position.to_array(),
        rotation: t.rotation.to_array(),
    }
}

pub fn player_rig_to_wire(rig: &PlayerRig) -> WirePlayerRig {
    WirePlayerRig {
        joints: rig
            .joints
            .iter()
            .map(|(&joint, &tf)| (joint_id_to_wire(joint), transform_to_wire(tf)))
            .collect(),
        hand_tracking_active: rig
            .hand_tracking_active
            .iter()
            .map(|(&hand, &active)| (hand_to_wire(hand), active))
            .collect(),
    }
}

pub fn input_frame_to_wire(input: &InputFrame) -> WireInputFrame {
    WireInputFrame {
        pointed: input
            .pointed
            .iter()
            .map(|(id, h)| (id.clone(), hand_to_wire(*h)))
            .collect(),
        grabbed: input
            .grabbed
            .iter()
            .map(|(id, h, point)| (id.clone(), hand_to_wire(*h), point.clone()))
            .collect(),
        released: input
            .released
            .iter()
            .map(|(id, h)| (id.clone(), hand_to_wire(*h)))
            .collect(),
        button_presses: input.button_presses.iter().map(button_press_to_wire).collect(),
        // Button-up edges travel separately from `released` above, which is GRAB
        // release and has meant that since before buttons had an up edge at all.
        button_releases: input.button_releases.iter().map(button_press_to_wire).collect(),
        axes: space_soup_protocol::WireInputAxes {
            l_trigger: input.axes.l_trigger,
            r_trigger: input.axes.r_trigger,
            l_grip: input.axes.l_grip,
            r_grip: input.axes.r_grip,
            l_stick: input.axes.l_stick,
            r_stick: input.axes.r_stick,
        },
        // Blends go up so the server can evaluate blend-threshold triggers. Only
        // this side can compute them -- a HandPull blend depends on where the hand
        // is relative to the posed part -- but the actions they fire (spawning a
        // magazine, handing it to physics) are authoritative world state.
        part_blends: input.part_blends.clone(),
    }
}

fn button_press_to_wire(b: &space_soup_engine::ButtonPress) -> space_soup_protocol::WireButtonPress {
    space_soup_protocol::WireButtonPress {
        button: b.button.clone(),
        object_id: b.object_id.clone(),
        hand: b.hand.map(hand_to_wire),
    }
}

pub fn locomotion_input_to_wire(input: &LocomotionInput) -> WireLocomotionInput {
    WireLocomotionInput {
        move_stick: input.move_stick,
        turn_stick_x: input.turn_stick_x,
        teleport_pressed: input.teleport_pressed,
        teleport_released: input.teleport_released,
        teleport_hand: hand_to_wire(input.teleport_hand),
        // Filled in by movement::step_locomotion after the local sim step.
        player_offset: None,
        player_yaw: None,
    }
}

pub fn teleport_target_to_wire(target: TeleportTarget) -> WireTeleportTarget {
    WireTeleportTarget {
        position: target.position.to_array(),
        valid: target.valid,
    }
}

