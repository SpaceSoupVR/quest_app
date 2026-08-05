#![cfg(target_os = "android")]

use log::info;

use space_soup_engine::{Hand, PlayerRig};

use crate::avatar;
use crate::grab_detect;
use crate::network;

pub(crate) fn send_local_pose(net: &network::NetworkHandle, rig: &PlayerRig) {
    let to_avatar_transform = |tf: space_soup_engine::Transform| avatar::Transform {
        position: tf.position,
        rotation: tf.rotation,
    };
    let _ = net.local_pose_tx.send(avatar::LocalPose {
        head: to_avatar_transform(rig.head()),
        left_hand: Some(to_avatar_transform(rig.hand_grip(Hand::Left))),
        right_hand: Some(to_avatar_transform(rig.hand_grip(Hand::Right))),
    });
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn log_frame_status(
    frame_count: u64,
    cuboid_count: usize,
    mesh_count: usize,
    light_count: usize,
    live_objects: &grab_detect::LiveObjects,
    static_scene: &grab_detect::StaticScene,
    rig: &PlayerRig,
    world_connected: bool,
) {
    if frame_count % 90 == 0 {
        info!(
            "Frame {frame_count}: cuboids={} meshes={} lights={} bounds={} connected={}",
            cuboid_count,
            mesh_count,
            light_count,
            live_objects.by_id.len(),
            world_connected,
        );
    }

    if frame_count % 30 == 0 {
        for hand in [Hand::Left, Hand::Right] {
            let hp = rig.hand_grip(hand).position;
            let diag = grab_detect::grab_diagnostic(live_objects, static_scene, hp, hand);
            info!("{}", diag.summary(hand, hp));
        }
    }
}
