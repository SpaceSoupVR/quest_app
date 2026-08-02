use log::{error, info};

pub mod avatar;
#[cfg(target_os = "android")]
mod client_audio;
#[cfg(target_os = "android")]
mod grab_detect;
#[cfg(target_os = "android")]
mod lightmap_client;
#[cfg(target_os = "android")]
mod network;
#[cfg(target_os = "android")]
mod particles;
#[cfg(target_os = "android")]
mod soundmap_client;
#[cfg(target_os = "android")]
mod to_wire;

#[cfg(target_os = "android")]
use glam::{Mat4, Quat, Vec3};
#[cfg(target_os = "android")]
use openxr;
#[cfg(target_os = "android")]
use space_soup::renderer::xr_renderer::XrRenderer;
#[cfg(target_os = "android")]
use space_soup::renderer::{
    scope::{self, ScopeRender},
    Beam, Color3, Cuboid, CuboidShape as SsCuboidShape, CuboidStyle as SsCuboidStyle, GltfMesh,
    Light, LightKind as SsLightKind, MeshInstance, MirrorSurface,
};
#[cfg(target_os = "android")]
use space_soup::{Controllers, ControllerState, HandTrackers, Headset, VkContext, XrContext};
#[cfg(target_os = "android")]
use space_soup_engine::optic::{OpticController, OpticalPathPose};
#[cfg(target_os = "android")]
use space_soup_engine::scene::{OpticalPathDef, OpticalPathsDef};
#[cfg(target_os = "android")]
use space_soup_engine::{
    debug_sender, ButtonPress, DebugPacket, Hand, HandSample, InputFrame,
    JointSample, Locomotion, LocomotionInput, LocomotionMode, LocomotionSample, Manifest,
    PartDriver, Pose, SceneSample, TeleportTarget, TimingSample,
};
#[cfg(target_os = "android")]
use space_soup_hands::{build_player_rig, load_synthetic_hand_config};
#[cfg(target_os = "android")]
use space_soup_protocol::{
    PlayerId, WireColor3, WireCuboidShape, WireCuboidStyle, WireHeldGrip, WireLightKind,
    WireRenderCuboid, WireRenderLaser, WireRenderLight, WireRenderMesh,
    WireRenderParticleEmitter,
};
#[cfg(target_os = "android")]
use log::warn;
#[cfg(target_os = "android")]
use std::collections::{HashMap, HashSet};
#[cfg(target_os = "android")]
use std::path::PathBuf;

#[cfg(target_os = "android")]
const IPD_M: f32 = 0.064;
#[cfg(target_os = "android")]
const EYE_FOV_Y_RAD: f32 = 1.6;
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
fn part_animation_blend(driver: PartDriver, hand: Hand, cs: &ControllerState) -> f32 {
    match (driver, hand) {
        (PartDriver::HoldTrigger, Hand::Left) => cs.l_trigger,
        (PartDriver::HoldTrigger, Hand::Right) => cs.r_trigger,
        (PartDriver::HoldGrip, Hand::Left) => cs.l_squeeze,
        (PartDriver::HoldGrip, Hand::Right) => cs.r_squeeze,
        (PartDriver::HandPull | PartDriver::Manual, _) => 0.0,
    }
}

#[cfg(target_os = "android")]
const PART_PULL_GRAB_RANGE: f32 = 0.09;

#[cfg(target_os = "android")]
fn hand_idx(hand: Hand) -> usize {
    match hand {
        Hand::Left => 0,
        Hand::Right => 1,
    }
}

#[cfg(target_os = "android")]
struct PullSession {
    object_id: String,
    clip: String,
    grab_local: Vec3,
    axis_model: Vec3,
    travel: f32,
    b0: f32,
    blend: f32,
}

#[cfg(target_os = "android")]
fn try_start_pull(
    object_id: &str,
    gun_world: Mat4,
    pull_hand_pos: Vec3,
    static_scene: &grab_detect::StaticScene,
    mesh_cache: &HashMap<String, (GltfMesh, space_soup::renderer::mesh_pipeline::ModelUniform)>,
) -> Option<PullSession> {
    let parts = static_scene.part_animations.get(object_id)?;
    let (mesh, _) = mesh_cache.get(object_id)?;
    let skin = mesh.skin.as_ref()?;
    let hand_local = gun_world.inverse().transform_point3(pull_hand_pos);
    for pa in parts.iter().filter(|pa| pa.driver == PartDriver::HandPull) {
        let Some(clip_idx) = skin.animation_index(&pa.clip) else {
            warn!("HandPull '{object_id}': clip '{}' not found in skin", pa.clip);
            continue;
        };
        let Some((anchor_model, axis_model, travel)) = skin.pull_geometry(clip_idx) else {
            warn!("HandPull '{object_id}': clip '{}' has no pull_geometry", pa.clip);
            continue;
        };
        let anchor_world = gun_world.transform_point3(anchor_model);
        let dist = pull_hand_pos.distance(anchor_world);
        info!(
            "HandPull '{object_id}' clip '{}': dist={dist:.3}m (need <={PART_PULL_GRAB_RANGE:.2}) anchor_world=({:.2},{:.2},{:.2}) travel={travel:.3}",
            pa.clip, anchor_world.x, anchor_world.y, anchor_world.z
        );
        if dist <= PART_PULL_GRAB_RANGE {
            return Some(PullSession {
                object_id: object_id.to_string(),
                clip: pa.clip.clone(),
                grab_local: hand_local,
                axis_model,
                travel,
                b0: 0.0,
                blend: 0.0,
            });
        }
    }
    None
}

#[cfg(target_os = "android")]
fn queue_new_meshes(
    world_meshes: &[WireRenderMesh],
    mesh_cache: &HashMap<String, (GltfMesh, space_soup::renderer::mesh_pipeline::ModelUniform)>,
    requested: &mut HashSet<String>,
    req_tx: &std::sync::mpsc::Sender<(String, Vec<String>)>,
) {
    let mut by_path: HashMap<String, Vec<String>> = HashMap::new();
    for rm in world_meshes {
        if mesh_cache.contains_key(&rm.id) || requested.contains(&rm.id) {
            continue;
        }
        requested.insert(rm.id.clone());
        by_path.entry(rm.path.clone()).or_default().push(rm.id.clone());
    }
    for (path, ids) in by_path {
        let _ = req_tx.send((path, ids));
    }
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

    let dir = game_dir();

    info!("init: creating XR renderer");
    let mut renderer = XrRenderer::new(&vk, &xr, &headset.session)?;
    info!("init: all subsystems ready");

    renderer.device().on_uncaptured_error(Box::new(|error| {
        error!("=== WGPU UNCAPTURED ERROR ===\n{error}\n=============================");
    }));

    let mut debug_stream: Option<std::net::TcpStream> = None;

    let local_player = PlayerId::local();

    let entry_scene = match Manifest::load(&dir) {
        Ok(m) => m.entry_scene,
        Err(e) => {
            error!("Failed to load manifest from {}: {e}", dir.display());
            error!("adb push your game folder to that path and relaunch.");
            return Err(e.into());
        }
    };
    let mut static_scene = grab_detect::StaticScene::load(&dir, &entry_scene);
    let mut live_objects = grab_detect::LiveObjects::default();
    let mut optic_controllers: HashMap<String, OpticController> = HashMap::new();
    let mut client_audio = client_audio::ClientAudio::new();

    let mut server_player_offset: Option<Vec3> = None;
    let mut server_player_yaw: Option<f32> = None;

    let mut locomotion = Locomotion::new(LocomotionMode::Smooth);

    let net = network::spawn(network::server_url());
    let lightmap_rx = lightmap_client::spawn(entry_scene.clone());
    let soundmap_rx = soundmap_client::spawn(entry_scene.clone());
    let mut soundmap_grids: HashMap<String, soundmap_client::OcclusionGrid> = HashMap::new();

    let mut mesh_cache: HashMap<
        String,
        (GltfMesh, space_soup::renderer::mesh_pipeline::ModelUniform),
    > = HashMap::new();
    let mut requested_mesh_ids: HashSet<String> = HashSet::new();

    let mut avatar_mesh_cache: HashMap<
        PlayerId,
        (GltfMesh, space_soup::renderer::mesh_pipeline::ModelUniform),
    > = HashMap::new();
    let mut avatar_skeleton_cache: HashMap<PlayerId, avatar_ik::SkeletonData> = HashMap::new();
    let boy_glb_path = dir.join("models/boy/boy.glb");
    let rig_config = avatar::load_rig_config(&dir.join("avatar_rig.json"));
    let synthetic_hand_config = load_synthetic_hand_config(&dir.join("synthetic_hand.json"));

    let mut calibrated_heights: HashMap<PlayerId, avatar_ik::HeightCalibrator> = HashMap::new();

    let mut local_direct_mesh: Option<(
        GltfMesh,
        space_soup::renderer::mesh_pipeline::ModelUniform,
    )> = None;

    let (mesh_req_tx, mesh_rx) = {
        let (req_tx, req_rx) = std::sync::mpsc::channel::<(String, Vec<String>)>();
        let (mesh_tx, mesh_rx) = std::sync::mpsc::channel::<(String, GltfMesh)>();
        let device = renderer.device().clone();
        let queue = renderer.queue().clone();
        let layout = renderer.mesh_texture_layout().clone();
        let gdir = dir.clone();
        std::thread::Builder::new()
            .name("mesh_loader".into())
            .spawn(move || {
                for (path, ids) in req_rx {
                    let full_path = gdir.join(&path);
                    match GltfMesh::load(&device, &queue, &layout, &full_path) {
                        Ok(mesh) => {
                            info!("Mesh loaded: '{path}' ({} object(s))", ids.len());
                            for id in &ids {
                                let instance = mesh.clone_with_independent_skin(&device);
                                if mesh_tx.send((id.clone(), instance)).is_err() {
                                    return;
                                }
                            }
                        }
                        Err(e) => warn!("Failed to load mesh '{path}': {e}"),
                    }
                }
            })
            .expect("failed to spawn mesh_loader");
        (req_tx, mesh_rx)
    };

    let avatar_mesh_rx = {
        let (tx, rx) = std::sync::mpsc::channel::<GltfMesh>();
        let device = renderer.device().clone();
        let queue = renderer.queue().clone();
        let layout = renderer.skinned_mesh_texture_layout().clone();
        let path = boy_glb_path.clone();
        std::thread::Builder::new()
            .name("avatar_loader".into())
            .spawn(move || match GltfMesh::load(&device, &queue, &layout, &path) {
                Ok(mesh) => {
                    info!("Avatar mesh loaded: '{}'", path.display());
                    let _ = tx.send(mesh);
                }
                Err(e) => warn!("Failed to load avatar mesh '{}': {e}", path.display()),
            })
            .expect("failed to spawn avatar_loader");
        rx
    };
    let mut avatar_master_mesh: Option<GltfMesh> = None;

    info!("All resources ready — entering event loop");

    let mut exit = false;
    let mut frame_count: u64 = 0;
    let mut input_log_timer: u64 = 0;
    let mut debug_reconnect_timer: u64 = 0;
    let mut last_time: Option<std::time::Instant> = None;
    let mut sim_time: f32 = 0.0;

    let mut prev_r_trigger = false;
    let mut prev_l_trigger = false;
    let mut prev_r_squeeze = false;
    let mut prev_l_squeeze = false;
    let mut pull_sessions: [Option<PullSession>; 2] = [None, None];
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
                if let Some(bind) = mesh.skin.as_ref().map(|s| s.skin_matrices_blended_multi(&[])) {
                    mesh.update_joint_matrices(renderer.queue(), &bind);
                }
                let model_uniform = renderer.create_skinned_model_uniform();
                mesh_cache.insert(obj_id, (mesh, model_uniform));
            } else {
                let model_uniform = renderer.create_model_uniform();
                mesh_cache.insert(obj_id, (mesh, model_uniform));
            }
        }

        for update in lightmap_rx.try_iter() {
            renderer.set_cuboid_lightmap(&update.object_id, &update.rgba, update.width, update.height);
            renderer.set_mesh_lightmap(&update.object_id, &update.rgba, update.width, update.height);
        }
        for update in soundmap_rx.try_iter() {
            soundmap_grids.insert(update.object_id, update.grid);
        }

        if avatar_master_mesh.is_none() {
            if let Ok(mesh) = avatar_mesh_rx.try_recv() {
                avatar_master_mesh = Some(mesh);
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
        sim_time += dt;

        let cs = &controllers.state;

        let rig = build_player_rig(&eye_views, &locomotion, cs, &hands, &synthetic_hand_config);

        let world = net.latest_world.lock().unwrap().clone();

        let empty_cuboids: Vec<WireRenderCuboid> = Vec::new();
        let empty_meshes: Vec<WireRenderMesh> = Vec::new();
        let empty_lights: Vec<WireRenderLight> = Vec::new();
        let empty_bounds: Vec<space_soup_protocol::WireObjectBounds> = Vec::new();
        let empty_particle_emitters: Vec<WireRenderParticleEmitter> = Vec::new();
        let empty_lasers: Vec<WireRenderLaser> = Vec::new();
        let cuboids_src = world.as_ref().map(|w| &w.cuboids).unwrap_or(&empty_cuboids);
        let meshes_src = world.as_ref().map(|w| &w.meshes).unwrap_or(&empty_meshes);
        let lights_src = world.as_ref().map(|w| &w.lights).unwrap_or(&empty_lights);
        let object_bounds_src = world.as_ref().map(|w| &w.object_bounds).unwrap_or(&empty_bounds);
        let particle_emitters_src = world
            .as_ref()
            .map(|w| &w.particle_emitters)
            .unwrap_or(&empty_particle_emitters);
        let lasers_src = world.as_ref().map(|w| &w.lasers).unwrap_or(&empty_lasers);

        if let Some(w) = &world {
            server_player_offset = Some(Vec3::from(w.player_offset));
            server_player_yaw = Some(w.player_yaw);

            if w.scene_name != static_scene.scene_name {
                info!(
                    "Scene changed: '{}' -> '{}' — reloading local grip-point data",
                    static_scene.scene_name, w.scene_name
                );
                static_scene = grab_detect::StaticScene::load(&dir, &w.scene_name);
            }
        }

        live_objects.update(object_bounds_src);
        queue_new_meshes(meshes_src, &mesh_cache, &mut requested_mesh_ids, &mesh_req_tx);

        let mut input = InputFrame::default();

        let r_trigger_down = cs.r_trigger > 0.5 || cs.r_squeeze > 0.5;
        let l_trigger_down = cs.l_trigger > 0.5 || cs.l_squeeze > 0.5;
        let r_trigger_only = cs.r_trigger > 0.5;
        let l_trigger_only = cs.l_trigger > 0.5;

        let held_gun_world = |hand: Hand| -> Option<(String, Mat4)> {
            let held = world.as_ref().and_then(|w| match hand {
                Hand::Left => w.left_hand_held.as_ref(),
                Hand::Right => w.right_hand_held.as_ref(),
            })?;
            let hand_tf = rig.hand_grip(hand);
            let hand_mat = Mat4::from_rotation_translation(hand_tf.rotation, hand_tf.position);
            let offset_mat = Mat4::from_rotation_translation(
                Quat::from_array(held.point_local_rot),
                Vec3::from(held.point_local_pos),
            );
            let (_, rot, pos) = (hand_mat * offset_mat.inverse()).to_scale_rotation_translation();
            let scale = meshes_src
                .iter()
                .find(|m| m.id == held.object_id)
                .map(|m| Vec3::from(m.scale))
                .unwrap_or(Vec3::ONE);
            Some((held.object_id.clone(), Mat4::from_scale_rotation_translation(scale, rot, pos)))
        };

        if r_trigger_down && !(prev_r_trigger || prev_r_squeeze) {
            let p = rig.hand_grip(Hand::Right).position;
            let pull = held_gun_world(Hand::Left)
                .and_then(|(oid, gw)| try_start_pull(&oid, gw, p, &static_scene, &mesh_cache));
            if let Some(session) = pull {
                info!("GRAB R: started HandPull on '{}'", session.object_id);
                pull_sessions[hand_idx(Hand::Right)] = Some(session);
            } else if let Some((id, point)) =
                grab_detect::nearest_grip_point_to(&live_objects, &static_scene, p, r_trigger_only, Hand::Right)
            {
                info!("GRAB R: '{id}' via grip point '{point}'");
                input.grabbed.push((id, Hand::Right, point));
            } else if let Some(id) = grab_detect::nearest_object_to(&live_objects, p) {
                info!("GRAB R: '{id}' via proximity (no grip point)");
                input.grabbed.push((id, Hand::Right, String::new()));
            } else {
                info!(
                    "GRAB R: pressed, nothing in range (hand {:.2},{:.2},{:.2}; {} live objs)",
                    p.x, p.y, p.z, live_objects.by_id.len()
                );
            }
        }
        if !r_trigger_down && (prev_r_trigger || prev_r_squeeze) {
            if pull_sessions[hand_idx(Hand::Right)].is_none() {
                if let Some(id) =
                    grab_detect::nearest_object_to(&live_objects, rig.hand_grip(Hand::Right).position)
                {
                    input.released.push((id, Hand::Right));
                }
            }
        }
        if l_trigger_down && !(prev_l_trigger || prev_l_squeeze) {
            let p = rig.hand_grip(Hand::Left).position;
            let pull = held_gun_world(Hand::Right)
                .and_then(|(oid, gw)| try_start_pull(&oid, gw, p, &static_scene, &mesh_cache));
            if let Some(session) = pull {
                info!("GRAB L: started HandPull on '{}'", session.object_id);
                pull_sessions[hand_idx(Hand::Left)] = Some(session);
            } else if let Some((id, point)) =
                grab_detect::nearest_grip_point_to(&live_objects, &static_scene, p, l_trigger_only, Hand::Left)
            {
                info!("GRAB L: '{id}' via grip point '{point}'");
                input.grabbed.push((id, Hand::Left, point));
            } else if let Some(id) = grab_detect::nearest_object_to(&live_objects, p) {
                info!("GRAB L: '{id}' via proximity (no grip point)");
                input.grabbed.push((id, Hand::Left, String::new()));
            } else {
                info!(
                    "GRAB L: pressed, nothing in range (hand {:.2},{:.2},{:.2}; {} live objs)",
                    p.x, p.y, p.z, live_objects.by_id.len()
                );
            }
        }
        if !l_trigger_down && (prev_l_trigger || prev_l_squeeze) {
            if pull_sessions[hand_idx(Hand::Left)].is_none() {
                if let Some(id) =
                    grab_detect::nearest_object_to(&live_objects, rig.hand_grip(Hand::Left).position)
                {
                    input.released.push((id, Hand::Left));
                }
            }
        }

        for hand in [Hand::Left, Hand::Right] {
            let idx = hand_idx(hand);
            let Some(session) = pull_sessions[idx].as_mut() else {
                continue;
            };
            let still_pulling = match hand {
                Hand::Left => l_trigger_down,
                Hand::Right => r_trigger_down,
            };
            let gun = held_gun_world(Hand::Left)
                .filter(|(oid, _)| *oid == session.object_id)
                .or_else(|| held_gun_world(Hand::Right).filter(|(oid, _)| *oid == session.object_id));
            let Some((_, gun_world)) = gun else {
                pull_sessions[idx] = None;
                continue;
            };
            if !still_pulling {
                pull_sessions[idx] = None;
                continue;
            }
            let hand_local = gun_world.inverse().transform_point3(rig.hand_grip(hand).position);
            let pulled = (hand_local - session.grab_local).dot(session.axis_model) / session.travel;
            session.blend = (session.b0 + pulled).clamp(0.0, 1.0);
        }

        {
            let held_r = grab_detect::held_object_id(
                world.as_ref().and_then(|w| w.right_hand_held.as_ref()),
                &live_objects,
                rig.hand_grip(Hand::Right).position,
            );
            let held_l = grab_detect::held_object_id(
                world.as_ref().and_then(|w| w.left_hand_held.as_ref()),
                &live_objects,
                rig.hand_grip(Hand::Left).position,
            );
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

        locomotion.update(dt, &locomotion_input, &rig, teleport_target);

        if let Some(sp) = server_player_offset {
            const RECONCILE_RATE: f32 = 8.0;
            let t = (RECONCILE_RATE * dt).min(1.0);
            locomotion.player_offset = locomotion.player_offset.lerp(sp, t);
        }
        if let Some(sy) = server_player_yaw {
            const RECONCILE_RATE: f32 = 8.0;
            let t = (RECONCILE_RATE * dt).min(1.0);
            let diff = (sy - locomotion.player_yaw + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
                - std::f32::consts::PI;
            locomotion.player_yaw += diff * t;
        }

        if frame_count % 30 == 0
            && (cs.l_stick.x.abs() > 0.1 || cs.l_stick.y.abs() > 0.1 || cs.r_stick.x.abs() > 0.1)
        {
            let logged_offset = server_player_offset.unwrap_or_default();
            let logged_yaw = server_player_yaw.unwrap_or_default();
            info!(
                "LOCO: Lstick=({:.2},{:.2}) Rstick.x={:.2} -> server off=({:.2},{:.2},{:.2}) yaw={:.1}deg conn={}",
                cs.l_stick.x, cs.l_stick.y, cs.r_stick.x,
                logged_offset.x, logged_offset.y, logged_offset.z, logged_yaw.to_degrees(),
                world.is_some()
            );
        }

        let _ = net.input_tx.send(network::PendingInput {
            input: to_wire::input_frame_to_wire(&input),
            locomotion_input: to_wire::locomotion_input_to_wire(&locomotion_input),
            rig: to_wire::player_rig_to_wire(&rig),
            teleport_target: teleport_target.map(to_wire::teleport_target_to_wire),
        });

        {
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

        if frame_count % 90 == 0 {
            info!(
                "Frame {frame_count}: cuboids={} meshes={} lights={} bounds={} connected={}",
                cuboids_src.len(),
                meshes_src.len(),
                lights_src.len(),
                live_objects.by_id.len(),
                world.is_some(),
            );
        }

        if frame_count % 30 == 0 {
            for hand in [Hand::Left, Hand::Right] {
                let hp = rig.hand_grip(hand).position;
                let diag = grab_detect::grab_diagnostic(&live_objects, &static_scene, hp, hand);
                info!("{}", diag.summary(hand, hp));
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
                    mode: "client-predicted".to_string(),
                    player_offset: locomotion.player_offset.into(),
                    player_yaw_deg: locomotion.player_yaw.to_degrees(),
                    teleport_aiming: false,
                },

                scene: SceneSample {
                    scene_name: static_scene.scene_name.clone(),
                    object_count: cuboids_src.len() + meshes_src.len(),
                    render_cuboids: cuboids_src.len(),
                    render_meshes: meshes_src.len(),
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

        let offset = locomotion.player_offset;
        let yaw_inv = Quat::from_rotation_y(-locomotion.player_yaw);

        let head_pos = rig.head().position;
        const MAX_RENDER_DIST: f32 = 40.0;

        let remotes = net.remote_players.lock().unwrap().clone();

        let local_state = avatar::RemotePlayerState {
            head: avatar::Transform {
                position: rig.head().position,
                rotation: rig.head().rotation,
            },
            left_hand: Some(avatar::Transform {
                position: rig.hand_grip(Hand::Left).position,
                rotation: rig.hand_grip(Hand::Left).rotation,
            }),
            right_hand: Some(avatar::Transform {
                position: rig.hand_grip(Hand::Right).position,
                rotation: rig.hand_grip(Hand::Right).rotation,
            }),
        };
        let bodies: Vec<(PlayerId, avatar::RemotePlayerState)> =
            std::iter::once((local_player, local_state))
                .chain(remotes.iter().map(|(&id, &state)| (id, state)))
                .collect();

        avatar_mesh_cache.retain(|id, _| bodies.iter().any(|(bid, _)| *bid == *id));
        avatar_skeleton_cache.retain(|id, _| bodies.iter().any(|(bid, _)| *bid == *id));

        for (id, state) in bodies.iter().copied() {
            if !avatar_mesh_cache.contains_key(&id) {
                let Some(master) = &avatar_master_mesh else { continue };
                let mut mesh = master.clone_with_independent_skin(renderer.device());
                if mesh.is_skinned() {
                    mesh.create_skin_bind_group(renderer.device(), renderer.skin_joint_layout());
                    let model_uniform = renderer.create_skinned_model_uniform();
                    if let Some(skin) = &mesh.skin {
                        avatar_skeleton_cache.insert(id, avatar::skeleton_data_from_skin(skin));
                    }
                    avatar_mesh_cache.insert(id, (mesh, model_uniform));
                } else {
                    warn!("'models/boy/boy.glb' has no skin — avatar bodies need a rigged mesh");
                    let model_uniform = renderer.create_model_uniform();
                    avatar_mesh_cache.insert(id, (mesh, model_uniform));
                }
            }
            let (mesh, _) = avatar_mesh_cache.get_mut(&id).expect("just inserted above");
            mesh.position = Vec3::ZERO;
            mesh.rotation = Quat::IDENTITY;
            mesh.scale = Vec3::ONE;
            let Some(skin) = &mesh.skin else { continue };
            let Some(skeleton) = avatar_skeleton_cache.get(&id) else { continue };

            let mut rig_cfg = rig_config;
            let up = avatar_ik::detect_up_axis(skeleton);
            rig_cfg.up_axis = up.to_array();
            let raw_bind_head_height = avatar_ik::bind_head_height_along(skeleton, up);
            let calibrated_height = calibrated_heights
                .entry(id)
                .or_default()
                .observe(state.head.position.y);
            let root_scale = avatar::height_calibrated_scale(calibrated_height, raw_bind_head_height);

            let to_render = |p: Vec3| yaw_inv * (p - offset);
            let floor_drop = raw_bind_head_height * root_scale;
            let head_rot = yaw_inv * state.head.rotation;
            let root = avatar_ik::body_root_transform_basis(
                avatar::Transform {
                    position: to_render(state.head.position),
                    rotation: head_rot,
                },
                floor_drop,
                up,
                rig_cfg.forward(),
            );
            let left_hand = state.left_hand.map(|h| avatar::Transform {
                position: to_render(h.position),
                rotation: yaw_inv * h.rotation,
            });
            let right_hand = state.right_hand.map(|h| avatar::Transform {
                position: to_render(h.position),
                rotation: yaw_inv * h.rotation,
            });

            let (left_curl, right_curl) = if id == local_player {
                let held_l = world.as_ref().and_then(|w| w.left_hand_held.as_ref());
                let held_r = world.as_ref().and_then(|w| w.right_hand_held.as_ref());
                let l = match held_l {
                    Some(held) => avatar::HandCurl::from_finger_curl(&held.finger_curl, cs.l_squeeze),
                    None => avatar::HandCurl::free_hand(
                        cs.l_trigger,
                        cs.l_squeeze,
                        cs.l_stick_touch,
                        rig_config.thumb_touch_curl,
                    ),
                };
                let r = match held_r {
                    Some(held) => avatar::HandCurl::from_finger_curl(&held.finger_curl, cs.r_squeeze),
                    None => avatar::HandCurl::free_hand(
                        cs.r_trigger,
                        cs.r_squeeze,
                        cs.r_stick_touch,
                        rig_config.thumb_touch_curl,
                    ),
                };
                (Some(l), Some(r))
            } else {
                (None, None)
            };

            let skinned_mats = avatar_ik::body_skin_matrices(
                skeleton,
                &rig_cfg,
                root.position,
                root.rotation,
                head_rot,
                root_scale,
                left_hand,
                right_hand,
                left_curl,
                right_curl,
            );
            skin.update_joint_matrices(renderer.queue(), &skinned_mats);

            if id == local_player {
                let direct = local_direct_mesh.get_or_insert_with(|| {
                    let hidden_joints = avatar_ik::head_and_descendant_joints(skeleton);
                    let mut direct_mesh = mesh.clone_with_independent_skin_excluding_joints(
                        renderer.device(),
                        &hidden_joints,
                    );
                    direct_mesh.create_skin_bind_group(renderer.device(), renderer.skin_joint_layout());
                    (direct_mesh, renderer.create_skinned_model_uniform())
                });
                direct.0.position = Vec3::ZERO;
                direct.0.rotation = Quat::IDENTITY;
                direct.0.scale = Vec3::ONE;
                if let Some(direct_skin) = &direct.0.skin {
                    let direct_mats = avatar_ik::body_skin_matrices(
                        skeleton,
                        &rig_cfg,
                        root.position,
                        root.rotation,
                        head_rot,
                        root_scale,
                        left_hand,
                        right_hand,
                        left_curl,
                        right_curl,
                    );
                    direct_skin.update_joint_matrices(renderer.queue(), &direct_mats);
                }
            }
        }

        let cuboids: Vec<Cuboid> = cuboids_src
            .iter()
            .filter(|rc| Vec3::from(rc.position).distance(head_pos) < MAX_RENDER_DIST)
            .map(|rc| to_space_soup_cuboid(rc, offset, yaw_inv))
            .collect();

        let lights: Vec<Light> = lights_src
            .iter()
            .map(|rl| to_space_soup_light(rl, offset, yaw_inv))
            .collect();

        for rm in meshes_src {
            if let Some((mesh, _)) = mesh_cache.get_mut(&rm.id) {
                mesh.position = yaw_inv * (Vec3::from(rm.position) - offset);
                mesh.rotation = yaw_inv * Quat::from_array(rm.rotation);
                mesh.scale = Vec3::from(rm.scale);
            }
        }

        for hand in [Hand::Left, Hand::Right] {
            let held: Option<&WireHeldGrip> = world.as_ref().and_then(|w| match hand {
                Hand::Left => w.left_hand_held.as_ref(),
                Hand::Right => w.right_hand_held.as_ref(),
            });
            let Some(held) = held else { continue };
            let Some((mesh, _)) = mesh_cache.get_mut(&held.object_id) else {
                continue;
            };
            let hand_tf = rig.hand_grip(hand);
            let hand_mat = glam::Mat4::from_rotation_translation(hand_tf.rotation, hand_tf.position);
            let offset_mat = glam::Mat4::from_rotation_translation(
                Quat::from_array(held.point_local_rot),
                Vec3::from(held.point_local_pos),
            );
            let (_, rot, pos) = (hand_mat * offset_mat.inverse()).to_scale_rotation_translation();
            mesh.position = yaw_inv * (pos - offset);
            mesh.rotation = yaw_inv * rot;

            if let Some(parts) = static_scene.part_animations.get(&held.object_id) {
                if let Some(skin) = &mesh.skin {
                    let targets: Vec<(usize, f32)> = parts
                        .iter()
                        .filter_map(|pa| {
                            let clip_idx = skin.animation_index(&pa.clip)?;
                            let raw = if pa.driver == PartDriver::HandPull {
                                pull_sessions
                                    .iter()
                                    .flatten()
                                    .find(|s| s.object_id == held.object_id && s.clip == pa.clip)
                                    .map(|s| s.blend)
                                    .unwrap_or(0.0)
                            } else {
                                part_animation_blend(pa.driver, hand, cs)
                            };
                            Some((clip_idx, pa.easing.apply(raw)))
                        })
                        .collect();
                    if !targets.is_empty() {
                        let mats = skin.skin_matrices_blended_multi(&targets);
                        mesh.update_joint_matrices(renderer.queue(), &mats);
                    }
                }
            }
        }

        let mut scope_renders: Vec<ScopeRender> = Vec::new();
        for hand in [Hand::Left, Hand::Right] {
            let held: Option<&WireHeldGrip> = world.as_ref().and_then(|w| match hand {
                Hand::Left => w.left_hand_held.as_ref(),
                Hand::Right => w.right_hand_held.as_ref(),
            });
            let Some(held) = held else { continue };
            let Some((mesh, _)) = mesh_cache.get(&held.object_id) else {
                continue;
            };

            let Some(optic) = static_scene.optics.get(&held.object_id) else {
                continue;
            };
            let paths: Vec<&OpticalPathDef> = match &optic.paths {
                OpticalPathsDef::Monocular { path } => vec![path],
                OpticalPathsDef::Binocular { left, right, .. } => vec![left, right],
            };

            let ctrl = optic_controllers
                .entry(held.object_id.clone())
                .or_insert_with(|| OpticController::new(optic));
            ctrl.advance(dt);

            let head_tf = rig.head();
            let eye_l = yaw_inv * (head_tf.position - offset);
            let eye_r =
                yaw_inv * (head_tf.position + (head_tf.rotation * Vec3::X) * IPD_M - offset);

            let posed: Vec<OpticalPathPose> = paths
                .iter()
                .map(|pd| {
                    let obj_w = mesh.position + mesh.rotation * Vec3::from_array(pd.objective_offset);
                    let ocu_w = mesh.position + mesh.rotation * Vec3::from_array(pd.ocular_offset);
                    OpticalPathPose {
                        objective: obj_w,
                        ocular: ocu_w,
                        ocular_radius_m: pd.ocular_radius_m,
                    }
                })
                .collect();

            let view = ctrl.evaluate(optic, [eye_l, eye_r], &posed, EYE_FOV_Y_RAD);
            if !view.render_needed {
                continue;
            }

            for (i, pose) in posed.iter().enumerate() {
                let occ = [
                    view.eyes[0][i].eye_box.occupancy,
                    view.eyes[1][i].eye_box.occupancy,
                ];

                let dir = |e: Vec3| -> [f32; 2] {
                    let (right, up, _) = scope::scope_basis(pose.axis(), Vec3::Y);
                    let rel = e - pose.ocular;
                    [rel.dot(right), rel.dot(up)]
                };

                scope_renders.push(ScopeRender {
                    objective: pose.objective,
                    ocular: pose.ocular,
                    axis: pose.axis(),
                    ocular_radius_m: pose.ocular_radius_m,
                    magnification: view.magnification,
                    true_fov_y_rad: view.scope_fov_deg.to_radians(),
                    occupancy: occ,
                    offset_dir: [dir(eye_l), dir(eye_r)],
                });
            }
        }

        let mesh_instances: Vec<MeshInstance> = meshes_src
            .iter()
            .filter(|rm| Vec3::from(rm.position).distance(head_pos) < MAX_RENDER_DIST)
            .filter_map(|rm| {
                let (mesh, model) = mesh_cache.get(&rm.id)?;
                Some(MeshInstance { mesh, model, lightmap_key: Some(rm.id.as_str()) })
            })
            .chain(
                avatar_mesh_cache
                    .iter()
                    .filter(|(&id, _)| id != local_player)
                    .map(|(_, (mesh, model))| MeshInstance { mesh, model, lightmap_key: None }),
            )
            .chain(
                local_direct_mesh
                    .iter()
                    .map(|(mesh, model)| MeshInstance { mesh, model, lightmap_key: None }),
            )
            .collect();

        let mirror_only_mesh_instances: Vec<MeshInstance> = avatar_mesh_cache
            .get(&local_player)
            .map(|(mesh, model)| MeshInstance { mesh, model, lightmap_key: None })
            .into_iter()
            .collect();

        let sounds_src = world.as_ref().map(|w| w.sounds.as_slice()).unwrap_or(&[]);
        let occlusion: HashMap<String, f32> = sounds_src
            .iter()
            .filter_map(|s| {
                let grid = soundmap_grids.get(&s.object_id)?;
                let occ = grid.sample(Vec3::from(s.position), s.max_distance, rig.head().position);
                Some((s.object_id.clone(), occ))
            })
            .collect();
        client_audio.update(
            &dir,
            sounds_src,
            (rig.head().position, rig.head().rotation),
            &occlusion,
        );

        let mirror_surface = cuboids_src.iter().find(|rc| rc.id == "mirror").map(|rc| {
            let rotation = yaw_inv * Quat::from_array(rc.rotation);
            let half_size = Vec3::from(rc.half_size);
            let normal = rotation * Vec3::NEG_Z;
            let position =
                yaw_inv * (Vec3::from(rc.position) - offset) + normal * (half_size.z + 0.005);
            MirrorSurface {
                position,
                rotation,
                half_size,
            }
        });

        let particles = particles::simulate(particle_emitters_src, sim_time, offset, yaw_inv);
        let beams: Vec<Beam> = lasers_src
            .iter()
            .map(|rl| to_space_soup_beam(rl, offset, yaw_inv))
            .collect();
        let proj_views = renderer.render_frame_with_meshes(
            &headset.session,
            &headset.stage,
            time,
            &cuboids,
            &mesh_instances,
            &mirror_only_mesh_instances,
            &lights,
            &particles,
            &beams,
            mirror_surface,
            &scope_renders,
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
fn to_space_soup_cuboid(rc: &WireRenderCuboid, offset: Vec3, yaw_inv: Quat) -> Cuboid {
    let style = match rc.style {
        WireCuboidStyle::Solid => SsCuboidStyle::Solid,
        WireCuboidStyle::Wireframe => SsCuboidStyle::Wireframe,
        WireCuboidStyle::SolidAndWire => SsCuboidStyle::SolidAndWire,
    };

    let position = yaw_inv * (Vec3::from(rc.position) - offset);
    let half_size = Vec3::from(rc.half_size);
    let mut c = match style {
        SsCuboidStyle::Solid => Cuboid::solid(position, half_size, ss_color(rc.color)),
        SsCuboidStyle::Wireframe => {
            Cuboid::wireframe(position, half_size, ss_color(rc.wire_color))
        }
        SsCuboidStyle::SolidAndWire => Cuboid::solid_and_wire(
            position,
            half_size,
            ss_color(rc.color),
            ss_color(rc.wire_color),
        ),
    };
    c.rotation = yaw_inv * Quat::from_array(rc.rotation);
    c.lightmap_key = Some(rc.id.clone());
    c.reflectivity = rc.reflectivity.clamp(0.0, 1.0);
    c.shape = match rc.shape {
        WireCuboidShape::Box => SsCuboidShape::Box,
        WireCuboidShape::Cylinder => SsCuboidShape::Cylinder,
    };
    c
}

#[cfg(target_os = "android")]
fn to_space_soup_light(rl: &WireRenderLight, offset: Vec3, yaw_inv: Quat) -> Light {
    Light {
        position: yaw_inv * (Vec3::from(rl.position) - offset),
        direction: yaw_inv * Vec3::from(rl.direction),
        kind: match rl.kind {
            WireLightKind::Point => SsLightKind::Point,
            WireLightKind::Spot => SsLightKind::Spot,
        },
        color: ss_color(rl.color),
        intensity: rl.intensity,
        range: rl.range,
        cone_angle_deg: rl.cone_angle_deg,
    }
}

#[cfg(target_os = "android")]
fn to_space_soup_beam(rl: &WireRenderLaser, offset: Vec3, yaw_inv: Quat) -> Beam {
    Beam {
        start: yaw_inv * (Vec3::from(rl.origin) - offset),
        end: yaw_inv * (Vec3::from(rl.end) - offset),
        width: rl.beam_width,
        color: ss_color(rl.color),
    }
}

#[cfg(target_os = "android")]
fn ss_color(c: WireColor3) -> Color3 {
    Color3(c.0, c.1, c.2, c.3)
}

