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
    GameRuntime, InputFrame, PlayerRig, Hand, FingerJoint, JointId, GripPoseDef,
    LocomotionInput, LocomotionMode, TeleportTarget,
    RenderCuboid, RenderMesh, CuboidStyle as EngineCuboidStyle,
    DebugPacket, Pose, HandSample, JointSample, LocomotionSample, SceneSample, TimingSample,
    debug_sender,
};
#[cfg(target_os = "android")]
use std::path::PathBuf;
#[cfg(target_os = "android")]
use std::collections::HashMap;

#[cfg(target_os = "android")]
const ANDROID_LOOPER_ID_MAIN: u32 = 0;
#[cfg(target_os = "android")]
const ANDROID_LOOPER_ID_INPUT: u32 = 1;

#[cfg(target_os = "android")]
fn pump_android_events(exit: &mut bool) {
    use ndk::looper::{Poll, ThreadLooper};
    let Some(looper) = ThreadLooper::for_thread() else { return };
    loop {
        let Ok(Poll::Event { ident, .. }) = looper.poll_all_timeout(std::time::Duration::ZERO) else { break };
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
                let Some(queue) = ndk_glue::input_queue() else { break };
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
fn hand_for_mesh_id(id: &str) -> Option<Hand> {
    if id.starts_with("lhand") {
        Some(Hand::Left)
    } else if id.starts_with("rhand") {
        Some(Hand::Right)
    } else {
        None
    }
}

#[cfg(target_os = "android")]
fn hand_skin_matrices(
    skin: &space_soup::renderer::mesh::GltfSkin,
    root_pos: Vec3,
    root_rot: Quat,
    curl: f32,
    grip: Option<&GripPoseDef>,
) -> Vec<glam::Mat4> {
    let local_pose = skin.blended_local_pose(0, 1, |ji| {
        grip.and_then(|g| {
            let name = space_soup::renderer::mesh::GltfSkin::generic_joint_name(&skin.joint_names[ji]);
            g.finger_curl.get(name).copied()
        }).unwrap_or(curl)
    });
    let global_in_mesh_space = skin.hierarchical_transforms(&local_pose);
    let root = glam::Mat4::from_rotation_translation(root_rot, root_pos);

    skin.inv_bind_mats.iter().enumerate()
        .map(|(ji, inv_bind)| root * global_in_mesh_space[ji] * *inv_bind)
        .collect()
}

#[cfg(target_os = "android")]
fn held_grip_pose<'a>(runtime: &'a GameRuntime, hand: Hand) -> Option<(&'a space_soup_engine::GameObject, &'a GripPoseDef)> {
    let id = runtime.attachments.object_for_joint(JointId::HandGrip(hand))?;
    let obj = runtime.scene().find_object(id)?;
    let grip = obj.grip_pose.as_ref()?;
    Some((obj, grip))
}

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

    ndk_glue::init(activity as _, saved_state as _, saved_state_size, run);
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
    std::panic::set_hook(Box::new(|info| {
        error!("PANIC: {info}");
    }));

    info!("init: waiting for activity resume");
    'wait_resume: loop {
        while let Some(event) = ndk_glue::poll_events() {
            match event {
                ndk_glue::Event::Resume  => { info!("init: activity resumed"); break 'wait_resume; }
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
                    warn!("xr: limit reached — previous session still cleaning up \
                           (attempt {}/25), retrying in 200ms", attempts + 1);
                    std::thread::sleep(std::time::Duration::from_millis(200));
                    attempts += 1;
                }
                Err(e) => return Err(e),
            }
        }
    };
    info!("init: creating Vulkan context");
    let vk              = VkContext::new(&xr)?;
    info!("init: creating headset session");
    let mut headset     = Headset::new(&xr, &vk)?;
    info!("init: creating controllers");
    let mut controllers = Controllers::new(&xr.instance, &headset.session)?;
    info!("init: creating hand trackers");
    let mut hands       = HandTrackers::new(&xr, &headset.session)?;
    info!("init: creating XR renderer");
    let mut renderer    = XrRenderer::new(&vk, &xr, &headset.session)?;
    info!("init: all subsystems ready");

    renderer.device().on_uncaptured_error(Box::new(|error| {
        error!("=== WGPU UNCAPTURED ERROR ===\n{error}\n=============================");
    }));

    let mut debug_stream: Option<std::net::TcpStream> = None;

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

    runtime.locomotion.set_mode(LocomotionMode::Smooth);

    info!("Object count after load (incl. hand bones from scene JSON): {}", runtime.scene().objects.len());

    let mut mesh_cache: HashMap<String, (GltfMesh, space_soup::renderer::mesh_pipeline::ModelUniform)> =
        HashMap::new();

    let (mesh_tx, mesh_rx) = std::sync::mpsc::channel::<(String, GltfMesh)>();
    {
        let device = renderer.device().clone();
        let queue  = renderer.queue().clone();
        let layout = renderer.mesh_texture_layout().clone();
        let gdir   = runtime.game_dir().to_path_buf();

        let mut path_to_ids: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
        for obj in runtime.scene().objects.iter() {
            if let Some(mesh_ref) = obj.mesh.as_ref() {
                path_to_ids.entry(mesh_ref.path.clone()).or_default().push(obj.id.clone());
            }
        }

        info!("Background mesh load starting: {} unique meshes", path_to_ids.len());
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
        pump_android_events(&mut exit);
        if exit { break 'main; }

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
            if frame_count % 50 == 0 {
                info!("idle: waiting for XR session READY ({}s elapsed)", frame_count / 10);
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
                info!("waiting: session running but should_render=false ({}s elapsed)", frame_count / 10);
            }
            frame_count += 1;
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

        let cs = &controllers.state;

        let mut rig = PlayerRig::new();

        if let Some(ev) = eye_views.first() {
            let p = ev.pose.position;
            let o = ev.pose.orientation;
            let (wp, wr) = runtime.locomotion.apply_to_head(
                Vec3::new(p.x, p.y, p.z), Quat::from_xyzw(o.x, o.y, o.z, o.w),
            );
            rig.set_head(wp, wr);
        }

        if let Some(p) = cs.r_grip_pose {
            let (wp, wr) = runtime.locomotion.apply_to_head(xr_vec3(p.position), xr_quat(p.orientation));
            rig.set_hand_grip(Hand::Right, wp, wr);
        }
        if let Some(p) = cs.l_grip_pose {
            let (wp, wr) = runtime.locomotion.apply_to_head(xr_vec3(p.position), xr_quat(p.orientation));
            rig.set_hand_grip(Hand::Left, wp, wr);
        }
        if let Some(p) = cs.r_aim_pose {
            let (wp, wr) = runtime.locomotion.apply_to_head(xr_vec3(p.position), xr_quat(p.orientation));
            rig.set_hand_aim(Hand::Right, wp, wr);
        }
        if let Some(p) = cs.l_aim_pose {
            let (wp, wr) = runtime.locomotion.apply_to_head(xr_vec3(p.position), xr_quat(p.orientation));
            rig.set_hand_aim(Hand::Left, wp, wr);
        }

        if hands.right_joints.iter().any(|j| j.valid) {
            let joints: Vec<(Vec3, Quat, bool)> = hands.right_joints.iter()
                .map(|j| {
                    let (wp, wr) = runtime.locomotion.apply_to_head(xr_vec3(j.pose.position), xr_quat(j.pose.orientation));
                    (wp, wr, j.valid)
                })
                .collect();
            rig.set_hand_joints(Hand::Right, &joints);
        } else if let Some(grip) = cs.r_grip_pose {
            let (gp, gr) = runtime.locomotion.apply_to_head(xr_vec3(grip.position), xr_quat(grip.orientation));
            rig.set_hand_joints(Hand::Right, &controller_hand_joints(gp, gr, Hand::Right, cs.r_squeeze));
        } else {
            rig.clear_hand_tracking(Hand::Right);
        }
        if hands.left_joints.iter().any(|j| j.valid) {
            let joints: Vec<(Vec3, Quat, bool)> = hands.left_joints.iter()
                .map(|j| {
                    let (wp, wr) = runtime.locomotion.apply_to_head(xr_vec3(j.pose.position), xr_quat(j.pose.orientation));
                    (wp, wr, j.valid)
                })
                .collect();
            rig.set_hand_joints(Hand::Left, &joints);
        } else if let Some(grip) = cs.l_grip_pose {
            let (gp, gr) = runtime.locomotion.apply_to_head(xr_vec3(grip.position), xr_quat(grip.orientation));
            rig.set_hand_joints(Hand::Left, &controller_hand_joints(gp, gr, Hand::Left, cs.l_squeeze));
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
                info!("After scene reload: {} objects", runtime.scene().objects.len());

                let new_objs: Vec<(String, String)> = runtime.scene().objects.iter()
                    .filter_map(|o| o.mesh.as_ref().map(|m| (o.id.clone(), m.path.clone())))
                    .collect();
                for (obj_id, path) in new_objs {
                    if mesh_cache.contains_key(&obj_id) { continue; }
                    let full_path = runtime.game_dir().join(&path);
                    info!("Attempting to load mesh (scene reload): '{path}' -> {}", full_path.display());
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

        let (head_pos, _) = runtime.world_head_transform();
        const MAX_RENDER_DIST: f32 = 40.0;

        let cuboids: Vec<Cuboid> = render_cuboids.iter()
            .filter(|rc| rc.position.distance(head_pos) < MAX_RENDER_DIST)
            .map(|rc| to_space_soup_cuboid(rc, offset, yaw_inv))
            .collect();

        for rm in &render_meshes {
            if let Some((mesh, _)) = mesh_cache.get_mut(&rm.id) {
                mesh.position = yaw_inv * (rm.position - offset);
                mesh.rotation = yaw_inv * rm.rotation;
                mesh.scale    = rm.scale;
            }
        }

        for rm in &render_meshes {
            if let Some((mesh, _)) = mesh_cache.get(&rm.id) {
                if let Some(skin) = &mesh.skin {
                    let Some(hand) = hand_for_mesh_id(&rm.id) else { continue };
                    let (has_grip, squeeze) = match hand {
                        Hand::Left  => (cs.l_grip_pose.is_some(), cs.l_squeeze),
                        Hand::Right => (cs.r_grip_pose.is_some(), cs.r_squeeze),
                    };
                    let (mut root_pos, mut root_rot, curl) = if has_grip {
                        let tf = runtime.rig.hand_grip(hand);
                        (tf.position, tf.rotation, squeeze)
                    } else {
                        let tf = runtime.rig.finger(hand, FingerJoint::Wrist);
                        (tf.position, tf.rotation, 0.0)
                    };

                    let held = held_grip_pose(&runtime, hand);
                    if let Some((obj, grip)) = held {
                        let obj_mat = glam::Mat4::from_rotation_translation(obj.cuboid.rotation, obj.cuboid.position);
                        let offset_mat = glam::Mat4::from_rotation_translation(
                            Quat::from_array(grip.hand_offset_rot),
                            Vec3::from(grip.hand_offset_pos),
                        );
                        let (_, rot, pos) = (obj_mat * offset_mat).to_scale_rotation_translation();
                        root_pos = pos;
                        root_rot = rot;
                    }

                    let skinned_mats = hand_skin_matrices(skin, root_pos, root_rot, curl, held.map(|(_, g)| g));
                    skin.update_joint_matrices(renderer.queue(), &skinned_mats);
                }
            }
        }

        let mesh_instances: Vec<MeshInstance> = render_meshes.iter()
            .filter(|rm| rm.position.distance(head_pos) < MAX_RENDER_DIST)
            .filter_map(|rm| {
                let (mesh, model) = mesh_cache.get(&rm.id)?;
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
        .map(|o| {
            let half = o.cuboid.half_size;
            let closest = point.clamp(o.cuboid.position - half, o.cuboid.position + half);
            (o.id.clone(), point.distance(closest))
        })
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

#[cfg(target_os = "android")]
fn walk_finger_chain(
    grip_pos: Vec3, grip_rot: Quat,
    base_local: Vec3,
    start_rot: Quat,
    lengths: &[f32],
    curl_max_deg: &[f32],
    squeeze: f32,
) -> Vec<(Vec3, Quat)> {
    let mut local_pos = base_local;
    let mut chain_rot = start_rot;
    let mut out = Vec::with_capacity(lengths.len());
    for (&len, &curl_deg) in lengths.iter().zip(curl_max_deg.iter()) {
        let dir = chain_rot * Vec3::new(0.0, 0.0, -1.0);
        local_pos += dir * len;
        out.push((grip_pos + grip_rot * local_pos, grip_rot * chain_rot));
        chain_rot *= Quat::from_rotation_x((squeeze * curl_deg).to_radians());
    }
    out
}

#[cfg(target_os = "android")]
fn controller_hand_joints(grip_pos: Vec3, grip_rot: Quat, hand: Hand, squeeze: f32) -> Vec<(Vec3, Quat, bool)> {
    let squeeze = squeeze.clamp(0.0, 1.0);
    let side = if hand == Hand::Left { -1.0 } else { 1.0 };

    let mut out = vec![(grip_pos, grip_rot, true); 26];
    out[FingerJoint::Wrist as usize] = (grip_pos + grip_rot * Vec3::new(0.0, 0.0, 0.04), grip_rot, true);
    out[FingerJoint::Palm  as usize] = (grip_pos + grip_rot * Vec3::new(0.0, 0.0, -0.02), grip_rot, true);

    struct Finger {
        joints:    &'static [FingerJoint],
        base:      Vec3,
        start_rot: Quat,
        lengths:   &'static [f32],
        curl_deg:  &'static [f32],
    }

    let fingers = [
        Finger {
            joints:    &[FingerJoint::ThumbMeta, FingerJoint::ThumbProx, FingerJoint::ThumbDist],
            base:      Vec3::new(side * 0.032, -0.010, 0.025),
            start_rot: Quat::from_rotation_y(side * -50.0_f32.to_radians())
                     * Quat::from_rotation_z(side * 20.0_f32.to_radians()),
            lengths:   &[0.030, 0.032, 0.024],
            curl_deg:  &[20.0, 55.0, 60.0],
        },
        Finger {
            joints:    &[FingerJoint::IndexMeta, FingerJoint::IndexProx, FingerJoint::IndexInter, FingerJoint::IndexDist],
            base:      Vec3::new(side * 0.018, -0.002, 0.0),
            start_rot: Quat::IDENTITY,
            lengths:   &[0.022, 0.036, 0.024, 0.018],
            curl_deg:  &[15.0, 90.0, 100.0, 80.0],
        },
        Finger {
            joints:    &[FingerJoint::MiddleMeta, FingerJoint::MiddleProx, FingerJoint::MiddleInter, FingerJoint::MiddleDist],
            base:      Vec3::new(side * -0.002, 0.0, 0.01),
            start_rot: Quat::IDENTITY,
            lengths:   &[0.024, 0.040, 0.026, 0.020],
            curl_deg:  &[15.0, 95.0, 100.0, 80.0],
        },
        Finger {
            joints:    &[FingerJoint::RingMeta, FingerJoint::RingProx, FingerJoint::RingInter, FingerJoint::RingDist],
            base:      Vec3::new(side * -0.020, -0.002, 0.0),
            start_rot: Quat::IDENTITY,
            lengths:   &[0.022, 0.035, 0.022, 0.018],
            curl_deg:  &[15.0, 95.0, 100.0, 80.0],
        },
        Finger {
            joints:    &[FingerJoint::LittleMeta, FingerJoint::LittleProx, FingerJoint::LittleInter, FingerJoint::LittleDist],
            base:      Vec3::new(side * -0.038, -0.005, -0.01),
            start_rot: Quat::IDENTITY,
            lengths:   &[0.020, 0.028, 0.018, 0.016],
            curl_deg:  &[15.0, 90.0, 100.0, 80.0],
        },
    ];

    for finger in &fingers {
        let chain = walk_finger_chain(
            grip_pos, grip_rot, finger.base, finger.start_rot,
            finger.lengths, finger.curl_deg, squeeze,
        );
        for (&joint, &(pos, rot)) in finger.joints.iter().zip(chain.iter()) {
            out[joint as usize] = (pos, rot, true);
        }
        if let (Some(tip_joint), Some(&(pos, rot))) = (tip_of(finger.joints[0]), chain.last()) {
            out[tip_joint as usize] = (pos, rot, true);
        }
    }

    let axis_fix = Quat::from_rotation_x(-90.0_f32.to_radians());
    for (_, rot, _) in out.iter_mut() {
        *rot *= axis_fix;
    }

    out
}

#[cfg(target_os = "android")]
fn tip_of(meta_joint: FingerJoint) -> Option<FingerJoint> {
    match meta_joint {
        FingerJoint::ThumbMeta  => Some(FingerJoint::ThumbTip),
        FingerJoint::IndexMeta  => Some(FingerJoint::IndexTip),
        FingerJoint::MiddleMeta => Some(FingerJoint::MiddleTip),
        FingerJoint::RingMeta   => Some(FingerJoint::RingTip),
        FingerJoint::LittleMeta => Some(FingerJoint::LittleTip),
        _ => None,
    }
}
