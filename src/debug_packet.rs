#![cfg(target_os = "android")]

use log::info;

use space_soup::{ControllerState, HandTrackers};
use space_soup_engine::{
    debug_sender, DebugPacket, HandSample, JointSample, Locomotion, LocomotionSample, Pose,
    SceneSample, TimingSample,
};

use crate::convert::{xr_quat, xr_vec3};
use crate::grab_detect::StaticScene;
use crate::platform::JOINT_NAMES;

#[allow(clippy::too_many_arguments)]
pub(crate) fn maybe_send(
    debug_stream: &mut Option<std::net::TcpStream>,
    debug_reconnect_timer: &mut u64,
    hands: &HandTrackers,
    cs: &ControllerState,
    eye_views: &[openxr::View],
    locomotion: &Locomotion,
    static_scene: &StaticScene,
    cuboid_count: usize,
    mesh_count: usize,
    dt: f32,
    frame_count: u64,
) {
    let Some(stream) = debug_stream else {
        return;
    };

    let to_joint_samples = |joints: &[space_soup::HandJoint]| -> Vec<JointSample> {
        joints
            .iter()
            .enumerate()
            .map(|(i, j)| JointSample {
                name: JOINT_NAMES.get(i).unwrap_or(&"unknown").to_string(),
                pose: Pose::new(xr_vec3(j.pose.position), xr_quat(j.pose.orientation)),
                valid: j.valid,
            })
            .collect()
    };

    let left_joints = to_joint_samples(&hands.left_joints);
    let right_joints = to_joint_samples(&hands.right_joints);

    let packet = DebugPacket {
        head: eye_views
            .first()
            .map(|ev| Pose::new(xr_vec3(ev.pose.position), xr_quat(ev.pose.orientation)))
            .unwrap_or_default(),

        left_hand: HandSample {
            tracking_active: !hands.left_joints.is_empty(),
            grip: cs
                .l_grip_pose
                .map(|p| Pose::new(xr_vec3(p.position), xr_quat(p.orientation))),
            aim: cs
                .l_aim_pose
                .map(|p| Pose::new(xr_vec3(p.position), xr_quat(p.orientation))),
            joints: left_joints,
            trigger: cs.l_trigger,
            squeeze: cs.l_squeeze,
            stick: [cs.l_stick.x, cs.l_stick.y],
            stick_click: cs.l_stick_click,
            btn_a: false,
            btn_b: false,
            btn_x: cs.btn_x,
            btn_y: cs.btn_y,
        },

        right_hand: HandSample {
            tracking_active: !hands.right_joints.is_empty(),
            grip: cs
                .r_grip_pose
                .map(|p| Pose::new(xr_vec3(p.position), xr_quat(p.orientation))),
            aim: cs
                .r_aim_pose
                .map(|p| Pose::new(xr_vec3(p.position), xr_quat(p.orientation))),
            joints: right_joints,
            trigger: cs.r_trigger,
            squeeze: cs.r_squeeze,
            stick: [cs.r_stick.x, cs.r_stick.y],
            stick_click: cs.r_stick_click,
            btn_a: cs.btn_a,
            btn_b: cs.btn_b,
            btn_x: false,
            btn_y: false,
        },

        locomotion: LocomotionSample {
            mode: "client-predicted".to_string(),
            player_offset: locomotion.player_offset.into(),
            player_yaw_deg: locomotion.player_yaw.to_degrees(),
            teleport_aiming: false,
        },

        scene: SceneSample {
            scene_name: static_scene.scene_name.clone(),
            object_count: cuboid_count + mesh_count,
            render_cuboids: cuboid_count,
            render_meshes: mesh_count,
            active_animations: vec![],
        },

        timing: TimingSample {
            dt_seconds: dt,
            fps: if dt > 0.0 { 1.0 / dt } else { 0.0 },
            frame_count,
        },

        log_lines: vec![],
    };

    if debug_sender::send(stream, &packet).is_err() {
        info!("debug_viewer disconnected — will retry");
        *debug_stream = None;
        *debug_reconnect_timer = 0;
    }
}
