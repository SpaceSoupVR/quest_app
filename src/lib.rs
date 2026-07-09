use log::{error, info};

mod avatar;
#[cfg(target_os = "android")]
mod network;

#[cfg(target_os = "android")]
use glam::{Quat, Vec3};
#[cfg(target_os = "android")]
use openxr;
#[cfg(target_os = "android")]
use space_soup::renderer::xr_renderer::XrRenderer;
#[cfg(target_os = "android")]
use space_soup::renderer::{
    Color3, Cuboid, CuboidStyle as SsCuboidStyle, GltfMesh, Light, LightKind as SsLightKind,
    MeshInstance,
};
#[cfg(target_os = "android")]
use space_soup::{Controllers, HandTrackers, Headset, VkContext, XrContext};
#[cfg(target_os = "android")]
use space_soup_engine::{
    debug_sender, ButtonPress, CuboidStyle as EngineCuboidStyle, DebugPacket, FingerJoint,
    GameRuntime, Hand, HandSample, InputFrame, JointSample, LightKind as EngineLightKind,
    Locomotion, LocomotionInput, LocomotionMode, LocomotionSample, PlayerFrameInput, Pose,
    RenderCuboid, RenderLight, RenderMesh, SceneSample, TeleportTarget, TimingSample,
};
#[cfg(target_os = "android")]
use space_soup_hands::{
    build_player_rig, free_hand_finger_curl, hand_for_mesh_id, hand_skin_matrices,
    held_grip_pose, held_object_id, nearest_grip_point_to, nearest_object_to,
};
#[cfg(target_os = "android")]
use space_soup_protocol::PlayerId;
#[cfg(target_os = "android")]
use log::warn;
#[cfg(target_os = "android")]
use std::collections::HashMap;
#[cfg(target_os = "android")]
use std::path::PathBuf;

#[cfg(target_os = "android")]
const ANDROID_LOOPER_ID_MAIN: u32 = 0;
#[cfg(target_os = "android")]
const ANDROID_LOOPER_ID_INPUT: u32 = 1;

#[cfg(target_os = "android")]
fn pump_android_events(exit: &mut bool) {
    use ndk::looper::{Poll, ThreadLooper};
    let Some(looper) = ThreadLooper::for_thread() else {
        return;
    };
    loop {
        let Ok(Poll::Event { ident, .. }) = looper.poll_all_timeout(std::time::Duration::ZERO)
        else {
            break;
        };
        match ident as u32 {
            ANDROID_LOOPER_ID_MAIN => match ndk_glue::poll_events() {
                Some(ndk_glue::Event::Destroy) => {
                    info!("pump_android_events: activity destroyed");
                    *exit = true;
                }
                Some(_) => {}
                None => break,
            },
            ANDROID_LOOPER_ID_INPUT => {
                let Some(queue) = ndk_glue::input_queue() else {
                    break;
                };
                match queue.get_event() {
                    Ok(Some(event)) => queue.finish_event(event, false),
                    _ => break,
                }
            }
            _ => break,
        }
    }
}

#[cfg(target_os = "android")]
#[no_mangle]
pub unsafe extern "C" fn ANativeActivity_onCreate(
    activity: *mut std::ffi::c_void,
    saved_state: *mut std::ffi::c_void,
    saved_state_size: usize,
) {
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Debug)
            .with_tag("quest_app"),
    );
    info!("ANativeActivity_onCreate started");

    // Note: `ndk_glue::init` already registers the JVM/Activity with
    // `ndk-context` internally (needed by cpal's AAudio backend for kira
    // sound playback) — don't call `ndk_context::initialize_android_context`
    // here too, it aborts the process on double-initialization.
    ndk_glue::init(activity as _, saved_state as _, saved_state_size, run);
}

pub fn run() {
    match run_inner() {
        Ok(()) => info!("App exited cleanly"),
        Err(e) => error!("App error: {e}"),
    }
}

#[cfg(not(target_os = "android"))]
fn run_inner() -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}

#[cfg(target_os = "android")]
fn game_dir() -> PathBuf {
    PathBuf::from("/sdcard/Android/data/com.example.questapp/files/game")
}

#[cfg(target_os = "android")]
fn run_inner() -> Result<(), Box<dyn std::error::Error>> {
    std::panic::set_hook(Box::new(|info| {
        error!("PANIC: {info}");
    }));

    info!("init: waiting for activity resume");
    'wait_resume: loop {
        while let Some(event) = ndk_glue::poll_events() {
            match event {
                ndk_glue::Event::Resume => {
                    info!("init: activity resumed");
                    break 'wait_resume;
                }
                ndk_glue::Event::Destroy => return Ok(()),
                _ => {}
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    info!("init: creating XR context");
    let xr = {
        let mut attempts = 0u32;
        loop {
            match XrContext::new() {
                Ok(ctx) => break ctx,
                Err(e) if e.to_string().contains("no more") && attempts < 25 => {
                    warn!(
                        "xr: limit reached — previous session still cleaning up \
                           (attempt {}/25), retrying in 200ms",
                        attempts + 1
                    );
                    std::thread::sleep(std::time::Duration::from_millis(200));
                    attempts += 1;
                }
                Err(e) => return Err(e),
            }
        }
    };
    info!("init: creating Vulkan context");
    let vk = VkContext::new(&xr)?;
    info!("init: creating headset session");
    let mut headset = Headset::new(&xr, &vk)?;
    info!("init: creating controllers");
    let mut controllers = Controllers::new(&xr.instance, &headset.session)?;
    info!("init: creating hand trackers");
    let mut hands = HandTrackers::new(&xr, &headset.session)?;
    info!("init: creating XR renderer");
    let mut renderer = XrRenderer::new(&vk, &xr, &headset.session)?;
    info!("init: all subsystems ready");

    renderer.device().on_uncaptured_error(Box::new(|error| {
        error!("=== WGPU UNCAPTURED ERROR ===\n{error}\n=============================");
    }));

    let mut debug_stream: Option<std::net::TcpStream> = None;

    let dir = game_dir();
    let mut runtime = match GameRuntime::load(&dir) {
        Ok(rt) => {
            info!(
                "Loaded game from {} — scene '{}'",
                dir.display(),
                rt.scene_name()
            );
            rt
        }
        Err(e) => {
            error!("Failed to load game from {}: {e}", dir.display());
            error!("adb push your game folder to that path and relaunch.");
            return Err(e.into());
        }
    };

    info!(
        "Object count right after load: {}",
        runtime.scene().objects.len()
    );

    // quest_app still simulates its own single local player (Phase 4 of the
    // server-authoritative rework — actually sending this to a remote
    // server and rendering *its* results instead — hasn't happened yet);
    // GameRuntime itself is multi-player-capable now, so it needs a fixed
    // id to address that one local player by.
    let local_player = PlayerId::local();
    runtime
        .locomotions
        .insert(local_player, Locomotion::new(LocomotionMode::Smooth));

    info!(
        "Object count after load (incl. hand bones from scene JSON): {}",
        runtime.scene().objects.len()
    );

    let net = network::spawn(network::server_url());

    let mut mesh_cache: HashMap<
        String,
        (GltfMesh, space_soup::renderer::mesh_pipeline::ModelUniform),
    > = HashMap::new();

    // Remote-player avatar bodies: one man.glb instance per connected
    // player, keyed by PlayerId rather than scene-object id (see avatar.rs
    // for why head/torso/arms are split between this mesh and cuboids).
    let mut avatar_mesh_cache: HashMap<
        PlayerId,
        (GltfMesh, space_soup::renderer::mesh_pipeline::ModelUniform),
    > = HashMap::new();
    let man_glb_path = runtime.game_dir().join("models/man.glb");

    let (mesh_tx, mesh_rx) = std::sync::mpsc::channel::<(String, GltfMesh)>();
    {
        let device = renderer.device().clone();
        let queue = renderer.queue().clone();
        let layout = renderer.mesh_texture_layout().clone();
        let gdir = runtime.game_dir().to_path_buf();

        let mut path_to_ids: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        for obj in runtime.scene().objects.iter() {
            if let Some(mesh_ref) = obj.mesh.as_ref() {
                path_to_ids
                    .entry(mesh_ref.path.clone())
                    .or_default()
                    .push(obj.id.clone());
            }
        }

        info!(
            "Background mesh load starting: {} unique meshes",
            path_to_ids.len()
        );
        std::thread::Builder::new()
            .name("mesh_loader".into())
            .spawn(move || {
                for (path, obj_ids) in path_to_ids {
                    let full_path = gdir.join(&path);
                    match GltfMesh::load(&device, &queue, &layout, &full_path) {
                        Ok(mesh) => {
                            let prim_count = mesh.primitives.len()
                                + mesh.skin.as_ref().map(|s| s.primitives.len()).unwrap_or(0);
                            info!("Mesh loaded: '{path}' ({prim_count} primitives, {} object(s), skinned={})",
                                obj_ids.len(), mesh.is_skinned());
                            for obj_id in &obj_ids {
                                let instance = mesh.clone_with_independent_skin(&device);
                                if mesh_tx.send((obj_id.clone(), instance)).is_err() { return; }
                            }
                        }
                        Err(e) => warn!("Failed to load mesh '{path}': {e}"),
                    }
                }
                info!("Background mesh loading complete");
            })
            .expect("failed to spawn mesh_loader");
    }

    info!("All resources ready — entering event loop");

    let mut exit = false;
    let mut frame_count: u64 = 0;
    let mut input_log_timer: u64 = 0;
    let mut debug_reconnect_timer: u64 = 0;
    let mut last_time: Option<std::time::Instant> = None;

    let mut prev_r_trigger = false;
    let mut prev_l_trigger = false;
    let mut prev_r_squeeze = false;
    let mut prev_l_squeeze = false;
    let mut prev_btn_a = false;
    let mut prev_btn_b = false;
    let mut prev_btn_x = false;
    let mut prev_btn_y = false;

    const JOINT_NAMES: [&str; 26] = [
        "palm",
        "wrist",
        "thumb_meta",
        "thumb_prox",
        "thumb_dist",
        "thumb_tip",
        "index_meta",
        "index_prox",
        "index_inter",
        "index_dist",
        "index_tip",
        "middle_meta",
        "middle_prox",
        "middle_inter",
        "middle_dist",
        "middle_tip",
        "ring_meta",
        "ring_prox",
        "ring_inter",
        "ring_dist",
        "ring_tip",
        "little_meta",
        "little_prox",
        "little_inter",
        "little_dist",
        "little_tip",
    ];

    'main: loop {
        pump_android_events(&mut exit);
        if exit {
            break 'main;
        }

        if debug_stream.is_none() {
            debug_reconnect_timer += 1;
            if debug_reconnect_timer >= 60 {
                debug_reconnect_timer = 0;
                if let Ok(s) = std::net::TcpStream::connect("127.0.0.1:7778") {
                    info!("debug_viewer connected");
                    debug_stream = Some(s);
                }
            }
        }

        let mut event_buf = openxr::EventDataBuffer::new();
        loop {
            match xr.instance.poll_event(&mut event_buf)? {
                Some(openxr::Event::SessionStateChanged(e)) => {
                    if headset.handle_state_change(e.state())? {
                        exit = true;
                    }
                }
                Some(openxr::Event::InstanceLossPending(_)) => exit = true,
                Some(_) => {}
                None => break,
            }
        }

        if exit {
            break 'main;
        }
        if !headset.running {
            if frame_count % 50 == 0 {
                info!(
                    "idle: waiting for XR session READY ({}s elapsed)",
                    frame_count / 10
                );
            }
            frame_count += 1;
            std::thread::sleep(std::time::Duration::from_millis(100));
            continue;
        }

        let frame_state = headset.frame_waiter.wait()?;
        headset.frame_stream.begin()?;

        for (obj_id, mut mesh) in mesh_rx.try_iter() {
            if mesh.is_skinned() {
                mesh.create_skin_bind_group(renderer.device(), renderer.skin_joint_layout());
                let model_uniform = renderer.create_skinned_model_uniform();
                mesh_cache.insert(obj_id, (mesh, model_uniform));
            } else {
                let model_uniform = renderer.create_model_uniform();
                mesh_cache.insert(obj_id, (mesh, model_uniform));
            }
        }

        let time = frame_state.predicted_display_time;

        controllers.sync(&headset.session, &headset.stage, time)?;
        hands.sync(&headset.stage, time)?;

        input_log_timer += 1;
        if input_log_timer >= 90 {
            input_log_timer = 0;
            controllers.log();
            hands.log();
        }

        if !frame_state.should_render {
            if frame_count % 50 == 0 {
                info!(
                    "waiting: session running but should_render=false ({}s elapsed)",
                    frame_count / 10
                );
            }
            frame_count += 1;
            headset
                .frame_stream
                .end(time, openxr::EnvironmentBlendMode::OPAQUE, &[])?;
            continue;
        }

        let (_, eye_views) = headset.session.locate_views(
            openxr::ViewConfigurationType::PRIMARY_STEREO,
            time,
            &headset.stage,
        )?;

        let now = std::time::Instant::now();
        let dt = last_time
            .map(|t| now.duration_since(t).as_secs_f32())
            .unwrap_or(1.0 / 90.0);
        last_time = Some(now);

        let cs = &controllers.state;

        let rig = build_player_rig(&eye_views, &runtime.locomotions[&local_player], cs, &hands);

        let mut input = InputFrame::default();

        let r_trigger_down = cs.r_trigger > 0.5 || cs.r_squeeze > 0.5;
        let l_trigger_down = cs.l_trigger > 0.5 || cs.l_squeeze > 0.5;
        let r_trigger_only = cs.r_trigger > 0.5;
        let l_trigger_only = cs.l_trigger > 0.5;

        if r_trigger_down && !(prev_r_trigger || prev_r_squeeze) {
            let p = rig.hand_grip(Hand::Right).position;
            if let Some((id, point)) = nearest_grip_point_to(&runtime, p, r_trigger_only) {
                input.grabbed.push((id, Hand::Right, point));
            } else if let Some(id) = nearest_object_to(&runtime, p) {
                input.grabbed.push((id, Hand::Right, String::new()));
            }
        }
        if !r_trigger_down && (prev_r_trigger || prev_r_squeeze) {
            if let Some(id) = nearest_object_to(&runtime, rig.hand_grip(Hand::Right).position) {
                input.released.push((id, Hand::Right));
            }
        }
        if l_trigger_down && !(prev_l_trigger || prev_l_squeeze) {
            let p = rig.hand_grip(Hand::Left).position;
            if let Some((id, point)) = nearest_grip_point_to(&runtime, p, l_trigger_only) {
                input.grabbed.push((id, Hand::Left, point));
            } else if let Some(id) = nearest_object_to(&runtime, p) {
                input.grabbed.push((id, Hand::Left, String::new()));
            }
        }
        if !l_trigger_down && (prev_l_trigger || prev_l_squeeze) {
            if let Some(id) = nearest_object_to(&runtime, rig.hand_grip(Hand::Left).position) {
                input.released.push((id, Hand::Left));
            }
        }

        // Controller-button presses (rising edges) for animation bindings.
        // A/B live on the right controller, X/Y on the left; trigger/grip are
        // emitted per hand. `object_id` carries the held object (if any) so
        // ContextualHold bindings can match.
        {
            let held_r = held_object_id(&runtime, local_player, &rig, Hand::Right);
            let held_l = held_object_id(&runtime, local_player, &rig, Hand::Left);
            let presses = [
                ("btn_a", cs.btn_a && !prev_btn_a, held_r.clone()),
                ("btn_b", cs.btn_b && !prev_btn_b, held_r.clone()),
                ("btn_x", cs.btn_x && !prev_btn_x, held_l.clone()),
                ("btn_y", cs.btn_y && !prev_btn_y, held_l.clone()),
                ("trigger", cs.r_trigger > 0.5 && !prev_r_trigger, held_r.clone()),
                ("trigger", cs.l_trigger > 0.5 && !prev_l_trigger, held_l.clone()),
                ("grip", cs.r_squeeze > 0.5 && !prev_r_squeeze, held_r),
                ("grip", cs.l_squeeze > 0.5 && !prev_l_squeeze, held_l),
            ];
            for (button, pressed, object_id) in presses {
                if pressed {
                    input.button_presses.push(ButtonPress {
                        button: button.to_string(),
                        object_id,
                    });
                }
            }
        }

        prev_r_trigger = cs.r_trigger > 0.5;
        prev_l_trigger = cs.l_trigger > 0.5;
        prev_r_squeeze = cs.r_squeeze > 0.5;
        prev_l_squeeze = cs.l_squeeze > 0.5;
        prev_btn_a = cs.btn_a;
        prev_btn_b = cs.btn_b;
        prev_btn_x = cs.btn_x;
        prev_btn_y = cs.btn_y;

        let locomotion_input = LocomotionInput {
            move_stick: (cs.l_stick.x, cs.l_stick.y),
            turn_stick_x: cs.r_stick.x,
            teleport_pressed: cs.r_trigger > 0.5 && !prev_r_trigger,
            teleport_released: !(cs.r_trigger > 0.5) && prev_r_trigger,
            teleport_hand: Hand::Right,
        };

        let teleport_target: Option<TeleportTarget> = None;

        let mut frame_inputs = HashMap::new();
        frame_inputs.insert(
            local_player,
            PlayerFrameInput {
                rig,
                input,
                locomotion_input,
                teleport_target,
            },
        );
        let (render_cuboids, render_meshes, render_lights, scene_change) =
            runtime.update(dt, &frame_inputs);

        // Multiplayer: publish this frame's world-space head/hand transforms
        // (rig.head()/hand_grip() are already locomotion-adjusted, unlike
        // the raw eye_views/grip poses used for the debug packet below) for
        // network.rs's background thread to forward to the server.
        {
            let to_transform = |tf: space_soup_engine::Transform| avatar::Transform {
                position: tf.position,
                rotation: tf.rotation,
            };
            let local_rig = &runtime.rigs[&local_player];
            let _ = net.local_pose_tx.send(avatar::LocalPose {
                head: to_transform(local_rig.head()),
                left_hand: Some(to_transform(local_rig.hand_grip(Hand::Left))),
                right_hand: Some(to_transform(local_rig.hand_grip(Hand::Right))),
            });
        }

        if frame_count % 90 == 0 {
            info!(
                "Frame {frame_count}: render_cuboids={} render_meshes={} render_lights={} scene.objects={} dt={:.4}",
                render_cuboids.len(),
                render_meshes.len(),
                render_lights.len(),
                runtime.scene().objects.len(),
                dt,
            );
        }

        if let Some(next_scene) = scene_change {
            if let Err(e) = runtime.load_scene(&next_scene) {
                warn!("Failed to switch scene to '{next_scene}': {e}");
            } else {
                info!(
                    "After scene reload: {} objects",
                    runtime.scene().objects.len()
                );

                let new_objs: Vec<(String, String)> = runtime
                    .scene()
                    .objects
                    .iter()
                    .filter_map(|o| o.mesh.as_ref().map(|m| (o.id.clone(), m.path.clone())))
                    .collect();
                for (obj_id, path) in new_objs {
                    if mesh_cache.contains_key(&obj_id) {
                        continue;
                    }
                    let full_path = runtime.game_dir().join(&path);
                    info!(
                        "Attempting to load mesh (scene reload): '{path}' -> {}",
                        full_path.display()
                    );
                    match GltfMesh::load(
                        renderer.device(),
                        renderer.queue(),
                        renderer.mesh_texture_layout(),
                        &full_path,
                    ) {
                        Ok(mesh) => {
                            let model_uniform = renderer.create_model_uniform();
                            info!("Preloaded mesh '{obj_id}' ('{path}') for new scene");
                            mesh_cache.insert(obj_id, (mesh, model_uniform));
                        }
                        Err(e) => warn!("Failed to preload mesh '{path}': {e}"),
                    }
                }
            }
        }

        if let Some(ref mut stream) = debug_stream {
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
                    mode: format!("{:?}", runtime.locomotions[&local_player].mode),
                    player_offset: runtime.locomotions[&local_player].player_offset.into(),
                    player_yaw_deg: runtime.locomotions[&local_player].player_yaw.to_degrees(),
                    teleport_aiming: runtime.locomotions[&local_player].is_teleport_aiming(),
                },

                scene: SceneSample {
                    scene_name: runtime.scene_name().to_string(),
                    object_count: runtime.scene().objects.len(),
                    render_cuboids: render_cuboids.len(),
                    render_meshes: render_meshes.len(),
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
                debug_stream = None;
                debug_reconnect_timer = 0;
            }
        }

        let offset = runtime.locomotions[&local_player].player_offset;
        let yaw_inv = Quat::from_rotation_y(-runtime.locomotions[&local_player].player_yaw);

        let (head_pos, _) = runtime.world_head_transform(local_player);
        const MAX_RENDER_DIST: f32 = 40.0;

        let remotes = net.remote_players.lock().unwrap().clone();

        let avatar_cuboids: Vec<RenderCuboid> = remotes
            .iter()
            .flat_map(|(id, state)| avatar::build_avatar_cuboids(&id.0.to_string(), state))
            .collect();

        // Bodies to render as a man.glb instance: every remote player, plus
        // the local player themselves (so you can look down and see your
        // own body, not just floating hands) — the local head pose comes
        // straight from this frame's rig rather than a network round-trip.
        let local_head_tf = runtime.rigs[&local_player].head();
        let local_head = avatar::Transform {
            position: local_head_tf.position,
            rotation: local_head_tf.rotation,
        };
        let bodies: Vec<(PlayerId, avatar::Transform)> = std::iter::once((local_player, local_head))
            .chain(remotes.iter().map(|(&id, state)| (id, state.head)))
            .collect();

        // Drop avatar mesh instances for players/bodies no longer present.
        avatar_mesh_cache.retain(|id, _| bodies.iter().any(|(bid, _)| bid == id));

        for (id, head) in bodies.iter().copied() {
            if !avatar_mesh_cache.contains_key(&id) {
                match GltfMesh::load(
                    renderer.device(),
                    renderer.queue(),
                    renderer.mesh_texture_layout(),
                    &man_glb_path,
                ) {
                    Ok(mesh) => {
                        let model_uniform = renderer.create_model_uniform();
                        avatar_mesh_cache.insert(id, (mesh, model_uniform));
                    }
                    Err(e) => {
                        warn!("Failed to load avatar mesh 'models/man.glb' for player {id:?}: {e}");
                        continue;
                    }
                }
            }
            let (mesh, _) = avatar_mesh_cache.get_mut(&id).expect("just inserted above");
            let torso = avatar::torso_transform(head);
            mesh.position = yaw_inv * (torso.position - offset);
            mesh.rotation = yaw_inv * torso.rotation * avatar::man_glb_orientation_offset();
            mesh.scale = Vec3::splat(avatar::MAN_GLB_SCALE);
        }

        let cuboids: Vec<Cuboid> = render_cuboids
            .iter()
            .chain(avatar_cuboids.iter())
            .filter(|rc| rc.position.distance(head_pos) < MAX_RENDER_DIST)
            .map(|rc| to_space_soup_cuboid(rc, offset, yaw_inv))
            .collect();

        let lights: Vec<Light> = render_lights
            .iter()
            .map(|rl| to_space_soup_light(rl, offset, yaw_inv))
            .collect();

        for rm in &render_meshes {
            if let Some((mesh, _)) = mesh_cache.get_mut(&rm.id) {
                mesh.position = yaw_inv * (rm.position - offset);
                mesh.rotation = yaw_inv * rm.rotation;
                mesh.scale = rm.scale;
            }
        }

        for rm in &render_meshes {
            if let Some((mesh, _)) = mesh_cache.get(&rm.id) {
                if let Some(skin) = &mesh.skin {
                    let Some(hand) = hand_for_mesh_id(&rm.id) else {
                        continue;
                    };
                    let (has_grip, squeeze, trigger, thumb_touch) = match hand {
                        Hand::Left => (
                            cs.l_grip_pose.is_some(),
                            cs.l_squeeze,
                            cs.l_trigger,
                            cs.l_stick_touch,
                        ),
                        Hand::Right => (
                            cs.r_grip_pose.is_some(),
                            cs.r_squeeze,
                            cs.r_trigger,
                            cs.r_stick_touch,
                        ),
                    };
                    let (mut root_pos, mut root_rot, curl) = if has_grip {
                        let tf = runtime.rigs[&local_player].hand_grip(hand);
                        (tf.position, tf.rotation, squeeze)
                    } else {
                        let tf = runtime.rigs[&local_player].finger(hand, FingerJoint::Wrist);
                        (tf.position, tf.rotation, 0.0)
                    };

                    let held_point = runtime.held_grip_point(local_player, hand);
                    let held_pose = if held_point.is_none() {
                        held_grip_pose(&runtime, local_player, hand)
                    } else {
                        None
                    };

                    if let Some((obj, point)) = held_point {
                        let obj_mat = glam::Mat4::from_rotation_translation(
                            obj.cuboid.rotation,
                            obj.cuboid.position,
                        );
                        let offset_mat = glam::Mat4::from_rotation_translation(
                            Quat::from_array(point.local_rot),
                            Vec3::from(point.local_pos),
                        );
                        let (_, rot, pos) = (obj_mat * offset_mat).to_scale_rotation_translation();
                        root_pos = pos;
                        root_rot = rot;
                    } else if let Some((obj, grip)) = held_pose {
                        let obj_mat = glam::Mat4::from_rotation_translation(
                            obj.cuboid.rotation,
                            obj.cuboid.position,
                        );
                        let offset_mat = glam::Mat4::from_rotation_translation(
                            Quat::from_array(grip.hand_offset_rot),
                            Vec3::from(grip.hand_offset_pos),
                        );
                        let (_, rot, pos) = (obj_mat * offset_mat).to_scale_rotation_translation();
                        root_pos = pos;
                        root_rot = rot;
                    }

                    let held_curl = held_point
                        .map(|(_, p)| &p.finger_curl)
                        .or_else(|| held_pose.map(|(_, g)| &g.finger_curl));
                    let free_curl = if held_curl.is_none() {
                        Some(free_hand_finger_curl(skin, trigger, squeeze, thumb_touch))
                    } else {
                        None
                    };
                    let finger_curl = held_curl.or(free_curl.as_ref());
                    let skinned_mats =
                        hand_skin_matrices(skin, root_pos, root_rot, curl, finger_curl);
                    skin.update_joint_matrices(renderer.queue(), &skinned_mats);
                }
            }
        }

        let mesh_instances: Vec<MeshInstance> = render_meshes
            .iter()
            .filter(|rm| rm.position.distance(head_pos) < MAX_RENDER_DIST)
            .filter_map(|rm| {
                let (mesh, model) = mesh_cache.get(&rm.id)?;
                Some(MeshInstance { mesh, model })
            })
            .chain(
                avatar_mesh_cache
                    .values()
                    .map(|(mesh, model)| MeshInstance { mesh, model }),
            )
            .collect();

        let proj_views = renderer.render_frame_with_meshes(
            &headset.session,
            &headset.stage,
            time,
            &cuboids,
            &mesh_instances,
            &lights,
        )?;
        let proj_layer = openxr::CompositionLayerProjection::new()
            .space(&headset.stage)
            .views(&proj_views);
        headset
            .frame_stream
            .end(time, openxr::EnvironmentBlendMode::OPAQUE, &[&proj_layer])?;

        frame_count += 1;
        if frame_count % 500 == 0 {
            info!("Frame {frame_count}");
        }
    }

    renderer.cleanup();
    Ok(())
}

#[cfg(target_os = "android")]
fn xr_vec3(p: openxr::Vector3f) -> Vec3 {
    Vec3::new(p.x, p.y, p.z)
}

#[cfg(target_os = "android")]
fn xr_quat(o: openxr::Quaternionf) -> Quat {
    Quat::from_xyzw(o.x, o.y, o.z, o.w)
}

#[cfg(target_os = "android")]
fn to_space_soup_cuboid(rc: &RenderCuboid, offset: Vec3, yaw_inv: Quat) -> Cuboid {
    let position = yaw_inv * (rc.position - offset);
    let rotation = yaw_inv * rc.rotation;

    let style = match rc.style {
        EngineCuboidStyle::Solid => SsCuboidStyle::Solid,
        EngineCuboidStyle::Wireframe => SsCuboidStyle::Wireframe,
        EngineCuboidStyle::SolidAndWire => SsCuboidStyle::SolidAndWire,
    };

    let mut c = match style {
        SsCuboidStyle::Solid => Cuboid::solid(position, rc.half_size, ss_color(rc.color)),
        SsCuboidStyle::Wireframe => {
            Cuboid::wireframe(position, rc.half_size, ss_color(rc.wire_color))
        }
        SsCuboidStyle::SolidAndWire => Cuboid::solid_and_wire(
            position,
            rc.half_size,
            ss_color(rc.color),
            ss_color(rc.wire_color),
        ),
    };
    c.rotation = rotation;
    c
}

/// Same "world moves under a fixed camera" render-space transform applied to
/// `render_cuboids`/`render_meshes` above — direction is a vector, so only
/// the yaw rotation applies to it, not the positional offset.
#[cfg(target_os = "android")]
fn to_space_soup_light(rl: &RenderLight, offset: Vec3, yaw_inv: Quat) -> Light {
    Light {
        position: yaw_inv * (rl.position - offset),
        direction: yaw_inv * rl.direction,
        kind: match rl.kind {
            EngineLightKind::Point => SsLightKind::Point,
            EngineLightKind::Spot => SsLightKind::Spot,
        },
        color: ss_color(rl.color),
        intensity: rl.intensity,
        range: rl.range,
        cone_angle_deg: rl.cone_angle_deg,
    }
}

#[cfg(target_os = "android")]
fn ss_color(c: space_soup_engine::Color3) -> Color3 {
    Color3(c.0, c.1, c.2, c.3)
}

