use log::{error, info, warn};

#[cfg(target_os = "android")]
use openxr;
#[cfg(target_os = "android")]
use space_soup::{XrContext, VkContext, Headset, Controllers, HandTrackers};
#[cfg(target_os = "android")]
use space_soup::renderer::xr_renderer::XrRenderer;
#[cfg(target_os = "android")]
use space_soup::renderer::{Cuboid, Color3, CuboidStyle as SsCuboidStyle, GltfMesh, MeshInstance};
#[cfg(target_os = "android")]
use glam::{Vec3, Quat};
#[cfg(target_os = "android")]
use space_soup_engine::{
    GameRuntime, InputFrame, PlayerRig, Hand,
    LocomotionInput, LocomotionMode, TeleportTarget,
    RenderCuboid, RenderMesh, CuboidStyle as EngineCuboidStyle,
    spawn_both_hand_rigs,
    DebugPacket, Pose, HandSample, JointSample, LocomotionSample, SceneSample, TimingSample,
    debug_sender,
};
#[cfg(target_os = "android")]
use std::path::PathBuf;
#[cfg(target_os = "android")]
use std::collections::HashMap;

#[cfg(target_os = "android")]
#[no_mangle]
pub unsafe extern "C" fn ANativeActivity_onCreate(
    activity:         *mut std::ffi::c_void,
    saved_state:      *mut std::ffi::c_void,
    saved_state_size: usize,
) {
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Debug)
            .with_tag("quest_app"),
    );
    info!("ANativeActivity_onCreate started");

    let activity    = activity as usize;
    let saved_state = saved_state as usize;

    std::thread::Builder::new()
        .name("xr_main".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            ndk_glue::init(activity as _, saved_state as _, saved_state_size, || {
                run();
            });
        })
        .expect("failed to spawn xr_main");
}

pub fn run() {
    match run_inner() {
        Ok(())  => info!("App exited cleanly"),
        Err(e)  => error!("App error: {e}"),
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
    let xr              = XrContext::new()?;
    let vk              = VkContext::new(&xr)?;
    let mut headset     = Headset::new(&xr, &vk)?;
    let mut controllers = Controllers::new(&xr.instance, &headset.session)?;
    let mut hands       = HandTrackers::new(&xr, &headset.session)?;
    let mut renderer    = XrRenderer::new(&vk, &xr, &headset.session)?;

    let mut debug_stream: Option<std::net::TcpStream> = None;

    // ── Load the game ───────────────────────────────────────────────────────
    let dir = game_dir();
    let mut runtime = match GameRuntime::load(&dir) {
        Ok(rt) => {
            info!("Loaded game from {} — scene '{}'", dir.display(), rt.scene_name());
            rt
        }
        Err(e) => {
            error!("Failed to load game from {}: {e}", dir.display());
            error!("adb push your game folder to that path and relaunch.");
            return Err(e.into());
        }
    };

    info!("Object count right after load: {}", runtime.scene().objects.len());

    spawn_both_hand_rigs(&mut runtime);
    runtime.locomotion.set_mode(LocomotionMode::Smooth);

    info!("Object count after spawning hand rigs: {}", runtime.scene().objects.len());

    let mut mesh_cache: HashMap<String, (GltfMesh, space_soup::renderer::mesh_pipeline::ModelUniform)> =
        HashMap::new();

    info!("Preloading meshes...");
    let mesh_paths: Vec<String> = runtime.scene().objects.iter()
        .filter_map(|o| o.mesh.as_ref().map(|m| m.path.clone()))
        .collect();

    for path in mesh_paths {
        if mesh_cache.contains_key(&path) { continue; }
        let full_path = runtime.game_dir().join(&path);
        match GltfMesh::load(
            renderer.device(),
            renderer.queue(),
            renderer.mesh_texture_layout(),
            &full_path,
        ) {
            Ok(mesh) => {
                let model_uniform = renderer.create_model_uniform();
                info!("Preloaded mesh '{path}' ({} primitives)", mesh.primitives.len());
                mesh_cache.insert(path, (mesh, model_uniform));
            }
            Err(e) => {
                warn!("Failed to preload mesh '{path}': {e}");
            }
        }
    }
    info!("Mesh preload complete — {} meshes cached", mesh_cache.len());

    info!("All resources ready — entering event loop");

    let mut exit             = false;
    let mut frame_count:     u64 = 0;
    let mut input_log_timer: u64 = 0;
    let mut debug_reconnect_timer: u64 = 0;
    let mut last_time:       Option<std::time::Instant> = None;

    let mut prev_r_trigger = false;
    let mut prev_l_trigger = false;
    let mut prev_r_squeeze = false;
    let mut prev_l_squeeze = false;

    const JOINT_NAMES: [&str; 26] = [
        "palm", "wrist",
        "thumb_meta", "thumb_prox", "thumb_dist", "thumb_tip",
        "index_meta", "index_prox", "index_inter", "index_dist", "index_tip",
        "middle_meta", "middle_prox", "middle_inter", "middle_dist", "middle_tip",
        "ring_meta", "ring_prox", "ring_inter", "ring_dist", "ring_tip",
        "little_meta", "little_prox", "little_inter", "little_dist", "little_tip",
    ];

    'main: loop {
        // [input-anr-fix] Drain the Android NativeActivity input queue every
        // iteration. We don't act on these events, but we must finish each one
        // so Android's input dispatcher doesn't time out (5s) and raise an ANR
        // ("app isn't responding" popup). Remove this block to revert.
        if let Some(input_queue) = ndk_glue::input_queue().as_ref() {
            while let Ok(Some(event)) = input_queue.get_event() {
                // pre_dispatch returns Some when the event still needs finishing
                // (None means the IME consumed it and will finish it itself).
                if let Some(event) = input_queue.pre_dispatch(event) {
                    input_queue.finish_event(event, false);
                }
            }
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
                    if headset.handle_state_change(e.state())? { exit = true; }
                }
                Some(openxr::Event::InstanceLossPending(_)) => exit = true,
                Some(_) => {}
                None    => break,
            }
        }

        if exit { break 'main; }
        if !headset.running {
            std::thread::sleep(std::time::Duration::from_millis(100));
            continue;
        }

        let frame_state = headset.frame_waiter.wait()?;
        headset.frame_stream.begin()?;

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
            headset.frame_stream.end(time, openxr::EnvironmentBlendMode::OPAQUE, &[])?;
            continue;
        }

        let (_, eye_views) = headset.session.locate_views(
            openxr::ViewConfigurationType::PRIMARY_STEREO,
            time,
            &headset.stage,
        )?;

        let now = std::time::Instant::now();
        let dt = last_time.map(|t| now.duration_since(t).as_secs_f32()).unwrap_or(1.0 / 90.0);
        last_time = Some(now);

        let mut rig = PlayerRig::new();

        if let Some(ev) = eye_views.first() {
            let p = ev.pose.position;
            let o = ev.pose.orientation;
            rig.set_head(
                Vec3::new(p.x, p.y, p.z),
                Quat::from_xyzw(o.x, o.y, o.z, o.w),
            );
        }

        let cs = &controllers.state;

        if let Some(p) = cs.r_grip_pose {
            rig.set_hand_grip(Hand::Right, xr_vec3(p.position), xr_quat(p.orientation));
        }
        if let Some(p) = cs.l_grip_pose {
            rig.set_hand_grip(Hand::Left, xr_vec3(p.position), xr_quat(p.orientation));
        }
        if let Some(p) = cs.r_aim_pose {
            rig.set_hand_aim(Hand::Right, xr_vec3(p.position), xr_quat(p.orientation));
        }
        if let Some(p) = cs.l_aim_pose {
            rig.set_hand_aim(Hand::Left, xr_vec3(p.position), xr_quat(p.orientation));
        }

        if !hands.right_joints.is_empty() {
            let joints: Vec<(Vec3, Quat, bool)> = hands.right_joints.iter()
                .map(|j| (xr_vec3(j.pose.position), xr_quat(j.pose.orientation), j.valid))
                .collect();
            rig.set_hand_joints(Hand::Right, &joints);
        } else {
            rig.clear_hand_tracking(Hand::Right);
        }
        if !hands.left_joints.is_empty() {
            let joints: Vec<(Vec3, Quat, bool)> = hands.left_joints.iter()
                .map(|j| (xr_vec3(j.pose.position), xr_quat(j.pose.orientation), j.valid))
                .collect();
            rig.set_hand_joints(Hand::Left, &joints);
        } else {
            rig.clear_hand_tracking(Hand::Left);
        }

        let mut input = InputFrame::default();

        let r_trigger_down = cs.r_trigger > 0.5 || cs.r_squeeze > 0.5;
        let l_trigger_down = cs.l_trigger > 0.5 || cs.l_squeeze > 0.5;

        if r_trigger_down && !(prev_r_trigger || prev_r_squeeze) {
            if let Some(id) = nearest_object_to(&runtime, rig.hand_grip(Hand::Right).position) {
                input.grabbed.push((id, Hand::Right));
            }
        }
        if !r_trigger_down && (prev_r_trigger || prev_r_squeeze) {
            if let Some(id) = nearest_object_to(&runtime, rig.hand_grip(Hand::Right).position) {
                input.released.push((id, Hand::Right));
            }
        }
        if l_trigger_down && !(prev_l_trigger || prev_l_squeeze) {
            if let Some(id) = nearest_object_to(&runtime, rig.hand_grip(Hand::Left).position) {
                input.grabbed.push((id, Hand::Left));
            }
        }
        if !l_trigger_down && (prev_l_trigger || prev_l_squeeze) {
            if let Some(id) = nearest_object_to(&runtime, rig.hand_grip(Hand::Left).position) {
                input.released.push((id, Hand::Left));
            }
        }

        prev_r_trigger = cs.r_trigger > 0.5;
        prev_l_trigger = cs.l_trigger > 0.5;
        prev_r_squeeze = cs.r_squeeze > 0.5;
        prev_l_squeeze = cs.l_squeeze > 0.5;

        let locomotion_input = LocomotionInput {
            move_stick:        (cs.l_stick.x, cs.l_stick.y),
            turn_stick_x:      cs.r_stick.x,
            teleport_pressed:  cs.r_trigger > 0.5 && !prev_r_trigger,
            teleport_released: !(cs.r_trigger > 0.5) && prev_r_trigger,
            teleport_hand:     Hand::Right,
        };

        let teleport_target: Option<TeleportTarget> = None;

        let (render_cuboids, render_meshes, scene_change) = runtime.update(
            dt, &input, rig, &locomotion_input, teleport_target,
        );

        if frame_count % 90 == 0 {
            info!(
                "Frame {frame_count}: render_cuboids={} render_meshes={} scene.objects={} dt={:.4}",
                render_cuboids.len(),
                render_meshes.len(),
                runtime.scene().objects.len(),
                dt,
            );
        }

        if let Some(next_scene) = scene_change {
            if let Err(e) = runtime.load_scene(&next_scene) {
                warn!("Failed to switch scene to '{next_scene}': {e}");
            } else {
                spawn_both_hand_rigs(&mut runtime);
                info!("After scene reload + hand respawn: {} objects", runtime.scene().objects.len());

                let new_paths: Vec<String> = runtime.scene().objects.iter()
                    .filter_map(|o| o.mesh.as_ref().map(|m| m.path.clone()))
                    .collect();
                for path in new_paths {
                    if mesh_cache.contains_key(&path) { continue; }
                    let full_path = runtime.game_dir().join(&path);
                    match GltfMesh::load(
                        renderer.device(),
                        renderer.queue(),
                        renderer.mesh_texture_layout(),
                        &full_path,
                    ) {
                        Ok(mesh) => {
                            let model_uniform = renderer.create_model_uniform();
                            info!("Preloaded mesh '{path}' for new scene");
                            mesh_cache.insert(path, (mesh, model_uniform));
                        }
                        Err(e) => warn!("Failed to preload mesh '{path}': {e}"),
                    }
                }
            }
        }

        if let Some(ref mut stream) = debug_stream {
            let to_joint_samples = |joints: &[space_soup::HandJoint]| -> Vec<JointSample> {
                joints.iter().enumerate().map(|(i, j)| JointSample {
                    name: JOINT_NAMES.get(i).unwrap_or(&"unknown").to_string(),
                    pose: Pose::new(xr_vec3(j.pose.position), xr_quat(j.pose.orientation)),
                    valid: j.valid,
                }).collect()
            };

            let left_joints  = to_joint_samples(&hands.left_joints);
            let right_joints = to_joint_samples(&hands.right_joints);

            let packet = DebugPacket {
                head: eye_views.first().map(|ev| Pose::new(
                    xr_vec3(ev.pose.position), xr_quat(ev.pose.orientation),
                )).unwrap_or_default(),

                left_hand: HandSample {
                    tracking_active: !hands.left_joints.is_empty(),
                    grip: cs.l_grip_pose.map(|p| Pose::new(xr_vec3(p.position), xr_quat(p.orientation))),
                    aim:  cs.l_aim_pose.map(|p| Pose::new(xr_vec3(p.position), xr_quat(p.orientation))),
                    joints: left_joints,
                    trigger: cs.l_trigger,
                    squeeze: cs.l_squeeze,
                    stick: [cs.l_stick.x, cs.l_stick.y],
                    stick_click: cs.l_stick_click,
                    btn_a: false, btn_b: false,
                    btn_x: cs.btn_x, btn_y: cs.btn_y,
                },

                right_hand: HandSample {
                    tracking_active: !hands.right_joints.is_empty(),
                    grip: cs.r_grip_pose.map(|p| Pose::new(xr_vec3(p.position), xr_quat(p.orientation))),
                    aim:  cs.r_aim_pose.map(|p| Pose::new(xr_vec3(p.position), xr_quat(p.orientation))),
                    joints: right_joints,
                    trigger: cs.r_trigger,
                    squeeze: cs.r_squeeze,
                    stick: [cs.r_stick.x, cs.r_stick.y],
                    stick_click: cs.r_stick_click,
                    btn_a: cs.btn_a, btn_b: cs.btn_b,
                    btn_x: false, btn_y: false,
                },

                locomotion: LocomotionSample {
                    mode: format!("{:?}", runtime.locomotion.mode),
                    player_offset: runtime.locomotion.player_offset.into(),
                    player_yaw_deg: runtime.locomotion.player_yaw.to_degrees(),
                    teleport_aiming: runtime.locomotion.is_teleport_aiming(),
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

        let offset = runtime.locomotion.player_offset;
        let yaw_inv = Quat::from_rotation_y(-runtime.locomotion.player_yaw);

        let cuboids: Vec<Cuboid> = render_cuboids.iter()
            .map(|rc| to_space_soup_cuboid(rc, offset, yaw_inv))
            .collect();

        for rm in &render_meshes {
            if let Some((mesh, _)) = mesh_cache.get_mut(&rm.path) {
                mesh.position = yaw_inv * (rm.position - offset);
                mesh.rotation = yaw_inv * rm.rotation;
                mesh.scale    = rm.scale;
            }
        }

        let mesh_instances: Vec<MeshInstance> = render_meshes.iter()
            .filter_map(|rm| {
                let (mesh, model) = mesh_cache.get(&rm.path)?;
                Some(MeshInstance { mesh, model })
            })
            .collect();

        let proj_views = renderer.render_frame_with_meshes(
            &headset.session, &headset.stage, time, &cuboids, &mesh_instances,
        )?;
        let proj_layer = openxr::CompositionLayerProjection::new()
            .space(&headset.stage)
            .views(&proj_views);
        headset.frame_stream.end(
            time, openxr::EnvironmentBlendMode::OPAQUE, &[&proj_layer],
        )?;

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
fn nearest_object_to(runtime: &GameRuntime, point: Vec3) -> Option<String> {
    const GRAB_RANGE: f32 = 0.15;
    runtime.scene().objects.iter()
        .map(|o| (o.id.clone(), o.cuboid.position.distance(point)))
        .filter(|(_, d)| *d <= GRAB_RANGE)
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(id, _)| id)
}

#[cfg(target_os = "android")]
fn to_space_soup_cuboid(rc: &RenderCuboid, offset: Vec3, yaw_inv: Quat) -> Cuboid {
    let position = yaw_inv * (rc.position - offset);
    let rotation = yaw_inv * rc.rotation;

    let style = match rc.style {
        EngineCuboidStyle::Solid        => SsCuboidStyle::Solid,
        EngineCuboidStyle::Wireframe    => SsCuboidStyle::Wireframe,
        EngineCuboidStyle::SolidAndWire => SsCuboidStyle::SolidAndWire,
    };

    let mut c = match style {
        SsCuboidStyle::Solid     => Cuboid::solid(position, rc.half_size, ss_color(rc.color)),
        SsCuboidStyle::Wireframe => Cuboid::wireframe(position, rc.half_size, ss_color(rc.wire_color)),
        SsCuboidStyle::SolidAndWire => Cuboid::solid_and_wire(
            position, rc.half_size, ss_color(rc.color), ss_color(rc.wire_color),
        ),
    };
    c.rotation = rotation;
    c
}

#[cfg(target_os = "android")]
fn ss_color(c: space_soup_engine::Color3) -> Color3 {
    Color3(c.0, c.1, c.2, c.3)
}