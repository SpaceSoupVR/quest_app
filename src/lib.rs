use log::{error, info};

pub mod avatar;
#[cfg(target_os = "android")]
mod avatar_render;
#[cfg(target_os = "android")]
mod client_audio;
#[cfg(target_os = "android")]
mod convert;
#[cfg(target_os = "android")]
mod debug_packet;
#[cfg(target_os = "android")]
mod frame_log;
#[cfg(target_os = "android")]
mod grab_detect;
#[cfg(target_os = "android")]
mod lightmap_client;
#[cfg(target_os = "android")]
mod loaders;
#[cfg(target_os = "android")]
mod mesh_load;
#[cfg(target_os = "android")]
mod movement;
#[cfg(target_os = "android")]
mod network;
#[cfg(target_os = "android")]
mod part_pull;
#[cfg(target_os = "android")]
mod particles;
#[cfg(target_os = "android")]
mod platform;
#[cfg(target_os = "android")]
mod render_prep;
#[cfg(target_os = "android")]
mod soundmap_client;
// Deliberately NOT android-gated, unlike its neighbours. The geometry and
// shading maths here is pure -- glam plus the renderer's vertex struct -- and
// gating it would mean its tests never compile, let alone run, on any machine a
// developer actually types on.
mod brush_render;
mod terrain_render;
#[cfg(target_os = "android")]
mod to_wire;

#[cfg(target_os = "android")]
use glam::{Quat, Vec3};
#[cfg(target_os = "android")]
use openxr;
#[cfg(target_os = "android")]
use space_soup::renderer::{
    Beam, GltfMesh,
};
#[cfg(target_os = "android")]
use space_soup_engine::{
    Locomotion, LocomotionMode, Manifest,
};
#[cfg(target_os = "android")]
use space_soup_hands::{build_player_rig, load_synthetic_hand_config};
#[cfg(target_os = "android")]
use space_soup_protocol::{
    PlayerId,
    WireRenderCuboid, WireRenderLaser, WireRenderLight, WireRenderMesh,
    WireRenderParticleEmitter,
};
#[cfg(target_os = "android")]
use std::collections::{HashMap, HashSet};

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
use convert::to_space_soup_beam;
#[cfg(target_os = "android")]
use mesh_load::queue_new_meshes;
#[cfg(target_os = "android")]
use part_pull::PullSession;
#[cfg(target_os = "android")]
use platform::{game_dir, pump_android_events};

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

    let platform::XrSetup {
        xr,
        mut headset,
        mut controllers,
        mut hands,
        mut renderer,
    } = platform::init_xr()?;

    let dir = game_dir();

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
    let mut loaded_terrain = load_scene_terrain(&dir, &static_scene.scene_name);
    let mut brushes = load_scene_brushes(&dir, &static_scene.scene_name);
    // Layer textures are per PROJECT, not per scene -- every level shares the
    // same four materials -- so they load once here rather than on every scene
    // change. A missing file leaves that layer flat-coloured; see
    // game/textures/terrain/SOURCES.md.
    let texture_dir = dir.join("textures").join("terrain");
    renderer.set_terrain_layers(
        space_soup::renderer::terrain_pipeline::load_terrain_layers(&texture_dir),
        space_soup::renderer::terrain_pipeline::load_terrain_normals(&texture_dir),
    );
    // How those layers tile. Per project like the textures, and authored in the
    // scene editor -- without this the headset renders every terrain at the
    // engine's built-in tile sizes no matter what the editor previewed, which
    // is a difference nobody can see until they put the headset on.
    renderer.set_terrain_settings(
        space_soup::renderer::terrain_pipeline::load_terrain_settings(&texture_dir),
    );
    renderer.set_terrain_splat(loaded_terrain.as_ref().and_then(|(_, s)| s.as_ref()));
    let mut live_objects = grab_detect::LiveObjects::default();
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
    // Geometry variants with hidden parts removed, keyed by object id and tagged
    // with the hidden set they were built for. Rebuilding one costs new vertex and
    // index buffers, so it happens only when that set changes.
    let mut hidden_part_meshes: HashMap<
        String,
        (Vec<String>, GltfMesh, space_soup::renderer::mesh_pipeline::ModelUniform),
    > = HashMap::new();
    let mut requested_mesh_ids: HashSet<String> = HashSet::new();
    // Posed part transforms published by the previous frame's render pass, sent
    // up so the engine can resolve part-anchored sockets and spawn detached parts
    // where the part actually is.
    let mut part_transforms: HashMap<String, HashMap<String, ([f32; 3], [f32; 4])>> =
        HashMap::new();

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

    let (mesh_req_tx, mesh_rx) = loaders::spawn_mesh_loader(&dir, &renderer);
    let avatar_mesh_rx = loaders::spawn_avatar_loader(boy_glb_path.clone(), &renderer);
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
    // What each hand grabbed and has not released. Client-side because the
    // server cannot always tell us: a proximity grab records no grip point name,
    // so resolve_held_grip reports an empty hand for an object the player is
    // plainly carrying.
    let mut grabbed_ids: [Option<String>; 2] = [None, None];
    let mut prev_btn_a = false;
    let mut prev_btn_b = false;
    let mut prev_btn_x = false;
    let mut prev_btn_y = false;

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
        let empty_particle_bursts: Vec<space_soup_protocol::WireRenderParticleBurst> = Vec::new();
        let empty_lasers: Vec<WireRenderLaser> = Vec::new();
        let cuboids_src = world.as_ref().map(|w| &w.cuboids).unwrap_or(&empty_cuboids);
        let meshes_src = world.as_ref().map(|w| &w.meshes).unwrap_or(&empty_meshes);
        let lights_src = world.as_ref().map(|w| &w.lights).unwrap_or(&empty_lights);
        let object_bounds_src = world.as_ref().map(|w| &w.object_bounds).unwrap_or(&empty_bounds);
        let particle_emitters_src = world
            .as_ref()
            .map(|w| &w.particle_emitters)
            .unwrap_or(&empty_particle_emitters);
        let particle_bursts_src = world
            .as_ref()
            .map(|w| &w.particle_bursts)
            .unwrap_or(&empty_particle_bursts);
        let lasers_src = world.as_ref().map(|w| &w.lasers).unwrap_or(&empty_lasers);
        // Empty when there is no snapshot yet, which draws the level whole --
        // the right failure: a wall that has not been told it was destroyed is
        // a frame behind, and one that vanishes on a dropped packet is a hole
        // the level did not authorise.
        let empty_hidden: Vec<String> = Vec::new();
        let hidden_brushes = world
            .as_ref()
            .map(|w| &w.hidden_brushes)
            .unwrap_or(&empty_hidden);

        if let Some(w) = &world {
            server_player_offset = Some(Vec3::from(w.player_offset));
            server_player_yaw = Some(w.player_yaw);

            if w.scene_name != static_scene.scene_name {
                info!(
                    "Scene changed: '{}' -> '{}' — reloading local grip-point data",
                    static_scene.scene_name, w.scene_name
                );
                static_scene = grab_detect::StaticScene::load(&dir, &w.scene_name);
                loaded_terrain = load_scene_terrain(&dir, &w.scene_name);
                // Per scene, like the terrain: the previous level's walls would
                // otherwise still be standing in this one.
                brushes = load_scene_brushes(&dir, &w.scene_name);
                // Per scene, alongside the geometry: a splat map left over from
                // the previous level would paint this one with its materials.
                renderer.set_terrain_splat(
                    loaded_terrain.as_ref().and_then(|(_, s)| s.as_ref()),
                );
            }
        }

        live_objects.update(object_bounds_src);
        queue_new_meshes(meshes_src, &mesh_cache, &mut requested_mesh_ids, &mesh_req_tx);

        let input = part_pull::handle_input(
            cs,
            &rig,
            &world,
            meshes_src,
            &static_scene,
            &mesh_cache,
            &live_objects,
            &mut pull_sessions,
            &mut grabbed_ids,
            &mut prev_r_trigger,
            &mut prev_l_trigger,
            &mut prev_r_squeeze,
            &mut prev_l_squeeze,
            &mut prev_btn_a,
            &mut prev_btn_b,
            &mut prev_btn_x,
            &mut prev_btn_y,
            sim_time,
            &part_transforms,
        );

        movement::step_locomotion(
            cs,
            dt,
            &rig,
            frame_count,
            server_player_offset,
            server_player_yaw,
            world.is_some(),
            &mut locomotion,
            &static_scene.physics,
            prev_r_trigger,
            &input,
            &net,
        );

        frame_log::send_local_pose(&net, &rig);
        frame_log::log_frame_status(
            frame_count,
            cuboids_src.len(),
            meshes_src.len(),
            lights_src.len(),
            &live_objects,
            &static_scene,
            &rig,
            world.is_some(),
        );

        debug_packet::maybe_send(
            &mut debug_stream,
            &mut debug_reconnect_timer,
            &hands,
            cs,
            &eye_views,
            &locomotion,
            &static_scene,
            cuboids_src.len(),
            meshes_src.len(),
            dt,
            frame_count,
        );

        let offset = locomotion.player_offset;
        let yaw_inv = Quat::from_rotation_y(-locomotion.player_yaw);

        let head_pos = rig.head().position;

        let remotes = net.remote_players.lock().unwrap().clone();
        let bodies = avatar_render::build_bodies(local_player, &rig, &remotes);

        let pull_hands = part_pull::pull_hand_poses(
            &pull_sessions, &static_scene, &live_objects, &part_transforms,
        );
        let mut local_hand_world: [Option<avatar_ik::Transform>; 2] = [None, None];
        avatar_render::update_avatar_bodies(
            &mut renderer,
            &mut avatar_mesh_cache,
            &mut avatar_skeleton_cache,
            &avatar_master_mesh,
            &mut local_direct_mesh,
            local_player,
            &rig_config,
            &mut calibrated_heights,
            offset,
            yaw_inv,
            &world,
            cs,
            &bodies,
            &pull_hands,
            &mut local_hand_world,
        );

        let (cuboids, lights, mesh_instances, mirror_only_mesh_instances, mirror_surface) =
            render_prep::build_render_lists(
                cuboids_src,
                lights_src,
                meshes_src,
                &mut mesh_cache,
                &mut hidden_part_meshes,
                &avatar_mesh_cache,
                &local_direct_mesh,
                local_player,
                &world,
                &static_scene,
                &pull_sessions,
                cs,
                &renderer,
                offset,
                yaw_inv,
                head_pos,
                sim_time,
                local_hand_world,
                rig_config.held_grip_offset(),
                &mut part_transforms,
            );

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

        let mut particles = particles::simulate(particle_emitters_src, sim_time, offset, yaw_inv);
        particles.extend(particles::simulate_bursts(particle_bursts_src, offset, yaw_inv));
        let beams: Vec<Beam> = lasers_src
            .iter()
            .map(|rl| to_space_soup_beam(rl, offset, yaw_inv))
            .collect();
        let terrain_arg = loaded_terrain
            .as_ref()
            .map(|(g, _)| g)
            .filter(|t| !t.is_empty())
            .map(|t| (t.vertices.as_slice(), t.indices.as_slice()));
        // Transformed into the player's frame here rather than at load, and
        // only when something moved -- see BrushGeometry::assemble.
        let brush_arg = brushes.assemble(
            hidden_brushes,
            offset,
            yaw_inv,
            locomotion.player_yaw,
        );

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
            terrain_arg,
            brush_arg,
            mirror_surface,
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


/// Terrain geometry for a scene, or `None` when it has none.
///
/// Reads the scene document straight off the device rather than waiting for the
/// server to describe it. Terrain is static scene data that run.sh already
/// pushes with the rest of game/, so sending it per snapshot would spend
/// bandwidth on something both ends already have -- which matters when the
/// target is 64 players.
/// Mesh a scene's brushes for the renderer.
///
/// Beside `load_scene_terrain` and for the same reason: this is static level
/// data that run.sh already pushed with the rest of game/, so building it here
/// costs one scene load rather than a share of every snapshot.
///
/// An unreadable scene yields no brushes rather than failing: the server is
/// still authoritative for everything else, and a level missing its walls is
/// diagnosable from the log in a way that a client which will not start is not.
#[cfg(target_os = "android")]
fn load_scene_brushes(
    game_dir: &std::path::Path,
    scene_name: &str,
) -> brush_render::BrushGeometry {
    let path = game_dir.join("scenes").join(format!("{scene_name}.json"));
    match space_soup_engine::scene::Scene::load(&path) {
        Ok(scene) => brush_render::BrushGeometry::load(&scene),
        Err(e) => {
            log::warn!("brush_render: could not read {} for brushes: {e}", path.display());
            brush_render::BrushGeometry::default()
        }
    }
}

#[cfg(target_os = "android")]
fn load_scene_terrain(
    game_dir: &std::path::Path,
    scene_name: &str,
) -> Option<(
    terrain_render::TerrainGeometry,
    Option<space_soup::renderer::terrain_pipeline::TerrainImage>,
)> {
    let path = game_dir.join("scenes").join(format!("{scene_name}.json"));
    let scene = match space_soup_engine::scene::Scene::load(&path) {
        Ok(s) => s,
        Err(e) => {
            log::warn!("terrain_render: could not read {} for terrain: {e}", path.display());
            return None;
        }
    };
    let def = scene.terrain.as_ref()?;
    // Step 1 for now. The LOD knob exists on TerrainSource::patch and is where
    // distant chunks get cheaper once terrain is big enough to need it.
    let geometry = terrain_render::load(def, game_dir, 1)?;

    // The splat map is optional and a missing one is not an error: terrain
    // without authored weights falls back to the slope blend, which is what
    // every scene looked like before painting existed. A DECLARED map that
    // cannot be read is worth a warning, though -- that is a broken level
    // rather than an unpainted one.
    let splat = match space_soup_engine::terrain::load_splat(def, game_dir) {
        Ok(Some(map)) => {
            let [w, h] = map.resolution();
            log::info!("terrain_render: splat map {w}x{h}");
            Some(space_soup::renderer::terrain_pipeline::TerrainImage {
                width: w,
                height: h,
                rgba: map.as_bytes().to_vec(),
            })
        }
        Ok(None) => None,
        Err(e) => {
            log::warn!("terrain_render: {e}");
            None
        }
    };
    Some((geometry, splat))
}
