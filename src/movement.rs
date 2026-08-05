#![cfg(target_os = "android")]

use glam::Vec3;
use log::info;

use space_soup::ControllerState;
use space_soup_engine::rigid_physics::PhysicsWorld;
use space_soup_engine::{Hand, InputFrame, Locomotion, LocomotionInput, PlayerRig, TeleportTarget};

use crate::network;
use crate::to_wire;

#[allow(clippy::too_many_arguments)]
pub(crate) fn step_locomotion(
    cs: &ControllerState,
    dt: f32,
    rig: &PlayerRig,
    frame_count: u64,
    server_player_offset: Option<Vec3>,
    server_player_yaw: Option<f32>,
    world_connected: bool,
    locomotion: &mut Locomotion,
    physics: &PhysicsWorld,
    prev_r_trigger: bool,
    input: &InputFrame,
    net: &network::NetworkHandle,
) {
    let locomotion_input = LocomotionInput {
        move_stick: (cs.l_stick.x, cs.l_stick.y),
        turn_stick_x: cs.r_stick.x,
        teleport_pressed: cs.r_trigger > 0.5 && !prev_r_trigger,
        teleport_released: !(cs.r_trigger > 0.5) && prev_r_trigger,
        teleport_hand: Hand::Right,
    };

    let teleport_target: Option<TeleportTarget> = None;

    // Player position/turning is fully client-authoritative: this is the ONLY place that
    // ever moves the player. The server's echoed offset (server_player_offset/_yaw,
    // still received below purely for the diagnostic log) is NEVER applied to
    // `locomotion` -- it exists only so the server can place this player for OTHER
    // players (we report our own pose in the input message at the end of this
    // function). Wall/ground collision reuses the exact same PhysicsWorld-backed logic
    // the server runs (Locomotion::apply_collision, in space_soup_engine), built from
    // the same local scene JSON already loaded into `physics` -- so the client doesn't
    // need the server for collision either, and never disagrees with it about geometry.
    let prev_xz = (locomotion.player_offset.x, locomotion.player_offset.z);
    locomotion.update(dt, &locomotion_input, rig, teleport_target);
    locomotion.apply_collision(physics, prev_xz);

    if frame_count % 30 == 0
        && (cs.l_stick.x.abs() > 0.1 || cs.l_stick.y.abs() > 0.1 || cs.r_stick.x.abs() > 0.1)
    {
        let logged_offset = server_player_offset.unwrap_or_default();
        let logged_yaw = server_player_yaw.unwrap_or_default();
        info!(
            "LOCO: Lstick=({:.2},{:.2}) Rstick.x={:.2} -> server off=({:.2},{:.2},{:.2}) yaw={:.1}deg conn={}",
            cs.l_stick.x, cs.l_stick.y, cs.r_stick.x,
            logged_offset.x, logged_offset.y, logged_offset.z, logged_yaw.to_degrees(),
            world_connected
        );
    }

    let mut locomotion_wire = to_wire::locomotion_input_to_wire(&locomotion_input);
    locomotion_wire.player_offset = Some(locomotion.player_offset.to_array());
    locomotion_wire.player_yaw = Some(locomotion.player_yaw);

    let _ = net.input_tx.send(network::PendingInput {
        input: to_wire::input_frame_to_wire(input),
        locomotion_input: locomotion_wire,
        rig: to_wire::player_rig_to_wire(rig),
        teleport_target: teleport_target.map(to_wire::teleport_target_to_wire),
    });
}
