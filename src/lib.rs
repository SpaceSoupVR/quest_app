use log::{error, info};

pub mod avatar;
#[cfg(target_os = "android")]
mod avatar_render;
#[cfg(target_os = "android")]
mod selector_pads;
#[cfg(target_os = "android")]
mod vr_props;
#[cfg(target_os = "android")]
mod hand_menu;
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
mod local_sim;
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
    Hand, Locomotion, LocomotionMode, Manifest,
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
    // Read before anything physics- or network-related below, since both
    // depend on the persisted multiplayer toggle this panel owns.
    let rig_config = avatar::load_rig_config(&dir.join("avatar_rig.json"));
    let mut hand_menu = hand_menu::HandMenu::new(&renderer, &dir, &rig_config)
        .expect("hand menu init (is game/fonts/menu.ttf pushed?)");

    // Off by default (see hand_menu_core::MenuValues::multiplayer_enabled): a
    // fresh install, or one where the droplet server is unreachable, runs the
    // scene entirely on-device via `local_sim` instead of silently retrying a
    // server the player never asked to connect to.
    let multiplayer_enabled = hand_menu.values().multiplayer_enabled;

    // local_sim's GameRuntime owns the scene's one-per-process PhysX
    // Foundation in local mode, so static_scene must not stand up a second one
    // (see grab_detect::StaticScene::physics and local_sim::LocalSim::physics).
    let mut static_scene = if multiplayer_enabled {
        grab_detect::StaticScene::load(&dir, &entry_scene)
    } else {
        grab_detect::StaticScene::load_without_physics(&dir, &entry_scene)
    };
    // Selection pads in the scene (body / hands).
    let mut selector_pads = selector_pads::SelectorPads::load(&dir, &entry_scene);
    // Props linked in the scene (loaded once the renderer is up).
    let mut vr_props: Option<vr_props::VrProps> = None;
    // The local body as last solved (where the hip pouch is), and which
    // hands each prop held last frame (for the log).
    let mut local_body: Option<avatar_render::LocalBody> = None;
    let mut prop_held_log = [false; 2];
    let mut live_objects = grab_detect::LiveObjects::default();
    let mut client_audio = client_audio::ClientAudio::new();

    let mut server_player_offset: Option<Vec3> = None;
    let mut server_player_yaw: Option<f32> = None;

    let mut locomotion = Locomotion::new(LocomotionMode::Smooth);

    let net = if multiplayer_enabled {
        network::spawn(network::server_url())
    } else {
        info!("multiplayer disabled — running the scene locally, no server connection");
        network::spawn_offline()
    };
    let mut local_sim = if multiplayer_enabled {
        None
    } else {
        local_sim::LocalSim::load(&dir)
    };
    let mut local_world: Option<space_soup_protocol::WireWorld> =
        local_sim.as_ref().map(|s| s.initial_world(local_player));

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
    let mut avatar_solve_states: HashMap<PlayerId, avatar_render::AvatarSolveState> =
        HashMap::new();
    // The bodies and hands players can wear (`game/avatars.json`). Other
    // players are drawn with the first body listed (their own choices aren't
    // sent over the network yet).
    let avatar_catalog = hand_menu_core::AvatarCatalog::load(&dir);
    let default_body_path = dir.join(avatar_catalog.body("").map(|b| b.model.clone()).unwrap_or_default());
    let synthetic_hand_config = load_synthetic_hand_config(&dir.join("synthetic_hand.json"));

    let mut local_direct_mesh: Option<(
        GltfMesh,
        space_soup::renderer::mesh_pipeline::ModelUniform,
    )> = None;

    let (mesh_req_tx, mesh_rx) = loaders::spawn_mesh_loader(&dir, &renderer);
    let avatar_mesh_rx = loaders::spawn_avatar_loader(default_body_path.clone(), &renderer);
    let mut avatar_master_mesh: Option<GltfMesh> = None;
    // The local player's chosen body (the first body shares
    // `avatar_master_mesh`; any other is loaded here) and hands: the
    // chosen ids, and the model paths they were loaded for.
    let mut local_body_mesh: Option<GltfMesh> = None;
    let mut local_body_rx: Option<std::sync::mpsc::Receiver<GltfMesh>> = None;
    let mut local_body_path: Option<std::path::PathBuf> = None;
    let mut local_look = (hand_menu.values().body.clone(), hand_menu.values().hands.clone());
    // Separate hand models (gloves) while the chosen hands have them.
    let mut gloves: Option<avatar_render::Gloves> = None;
    let mut glove_rx: Option<[std::sync::mpsc::Receiver<GltfMesh>; 2]> = None;
    let mut glove_parts: [Option<GltfMesh>; 2] = [None, None];
    let mut gloves_for: Option<[String; 2]> = None;
    let mut gloves_failed: Option<[String; 2]> = None;

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
        // The local player's body and hands (from the catalog): load the
        // chosen body and the chosen hands' models, and rebuild the local
        // avatar whenever the choice changes.
        {
            let look = (hand_menu.values().body.clone(), hand_menu.values().hands.clone());
            let mut rebuild = false;
            if look != local_look {
                info!("player look: body '{}', hands '{}'", look.0, look.1);
                local_look = look.clone();
                rebuild = true;
            }
            // Body: the first body is the shared master; others load here.
            let want_body = avatar_catalog.body(&look.0).map(|b| dir.join(&b.model));
            let want_own = want_body.as_ref().filter(|p| **p != default_body_path).cloned();
            if want_own != local_body_path {
                local_body_path = want_own.clone();
                local_body_mesh = None;
                local_body_rx = want_own.map(|p| loaders::spawn_avatar_loader(p, &renderer));
                rebuild = true;
            }
            if let Some(rx) = local_body_rx.as_ref() {
                if let Ok(mesh) = rx.try_recv() {
                    local_body_mesh = Some(mesh);
                    local_body_rx = None;
                    rebuild = true;
                }
            }
            // Hands: separate models (gloves) if the chosen hands have them.
            let want_gloves = avatar_catalog
                .hands(&look.1)
                .and_then(|h| h.models())
                .map(|[l, r]| [l.to_string(), r.to_string()]);
            if want_gloves != gloves_for {
                gloves_for = want_gloves.clone();
                gloves = None;
                glove_parts = [None, None];
                glove_rx = want_gloves
                    .as_ref()
                    .filter(|w| gloves_failed.as_ref() != Some(*w))
                    .map(|[l, r]| [loaders::spawn_avatar_loader(dir.join(l), &renderer), loaders::spawn_avatar_loader(dir.join(r), &renderer)]);
                rebuild = true;
            }
            if let Some(rxs) = glove_rx.as_ref() {
                for (i, rx) in rxs.iter().enumerate() {
                    if let Ok(m) = rx.try_recv() {
                        glove_parts[i] = Some(m);
                    }
                }
            }
            if glove_rx.is_some() && glove_parts.iter().all(Option::is_some) {
                let mut built = Vec::new();
                for (i, part) in glove_parts.iter_mut().enumerate() {
                    let Some(mut m) = part.take() else { continue };
                    let Some(skin) = m.skin.as_ref() else { continue };
                    let skel = avatar_pose::Skeleton::from_skin(skin.joint_names.clone(), skin.joint_parents.clone(), skin.inv_bind_mats.clone());
                    let Some(rig) = avatar_pose::GloveRig::new(skel, i == 0) else { continue };
                    m.create_skin_bind_group(renderer.device(), renderer.skin_joint_layout());
                    built.push((m, renderer.create_skinned_model_uniform(), rig));
                }
                glove_rx = None;
                match <[_; 2]>::try_from(built) {
                    Ok([l, r]) => {
                        gloves = Some(avatar_render::Gloves { parts: [l, r], shown: false });
                        rebuild = true;
                    }
                    Err(_) => {
                        error!("hands '{}': couldn't rig its hand models -- showing the body's own hands", look.1);
                        gloves_failed = gloves_for.clone();
                    }
                }
            }
            if rebuild {
                avatar_mesh_cache.remove(&local_player);
                avatar_solve_states.remove(&local_player);
                local_direct_mesh = None;
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

        let effective_rig_config = rig_config.clone();

        let rig = build_player_rig(&eye_views, &locomotion, cs, &hands, &synthetic_hand_config);

        if frame_count % 90 == 0 {
            info!(
                "GRIPDIAG l_grip={:?} r_grip={:?} l_aim={:?} r_aim={:?}",
                cs.l_grip_pose.map(|p| (p.position.x, p.position.y, p.position.z)),
                cs.r_grip_pose.map(|p| (p.position.x, p.position.y, p.position.z)),
                cs.l_aim_pose.is_some(),
                cs.r_aim_pose.is_some(),
            );
        }

        let world = if multiplayer_enabled {
            net.latest_world.lock().unwrap().clone()
        } else {
            local_world.clone()
        };

        // Only `meshes_src`/`object_bounds_src` are needed before the local-sim
        // step below (mesh prefetch + this frame's grab detection, both reading
        // last frame's world same as the multiplayer path always has). The rest
        // are derived again, from the frame's final `world`, further down.
        let empty_cuboids: Vec<WireRenderCuboid> = Vec::new();
        let empty_meshes: Vec<WireRenderMesh> = Vec::new();
        let empty_lights: Vec<WireRenderLight> = Vec::new();
        let empty_bounds: Vec<space_soup_protocol::WireObjectBounds> = Vec::new();
        let empty_particle_emitters: Vec<WireRenderParticleEmitter> = Vec::new();
        let empty_particle_bursts: Vec<space_soup_protocol::WireRenderParticleBurst> = Vec::new();
        let empty_lasers: Vec<WireRenderLaser> = Vec::new();
        let meshes_src = world.as_ref().map(|w| &w.meshes).unwrap_or(&empty_meshes);
        let object_bounds_src = world.as_ref().map(|w| &w.object_bounds).unwrap_or(&empty_bounds);

        if let Some(w) = &world {
            server_player_offset = Some(Vec3::from(w.player_offset));
            server_player_yaw = Some(w.player_yaw);

            if w.scene_name != static_scene.scene_name {
                info!(
                    "Scene changed: '{}' -> '{}' — reloading local grip-point data",
                    static_scene.scene_name, w.scene_name
                );
                static_scene = if multiplayer_enabled {
                    grab_detect::StaticScene::load(&dir, &w.scene_name)
                } else {
                    grab_detect::StaticScene::load_without_physics(&dir, &w.scene_name)
                };
            }
        }

        live_objects.update(object_bounds_src);
        queue_new_meshes(meshes_src, &mesh_cache, &mut requested_mesh_ids, &mesh_req_tx);

        let mut input = part_pull::handle_input(
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
        // Props are picked up and worked through vr_props, never by the
        // scene's own grabbing -- nor does a hand holding one grab anything
        // else (pulling a held gun's trigger is not a grab).
        if let Some(vp) = vr_props.as_ref() {
            input.grabbed.retain(|(id, hand, _)| {
                let i = part_pull::hand_idx(*hand);
                let keep = !vp.is_prop(id) && !vp.holds(i);
                if !keep {
                    grabbed_ids[i] = None;
                }
                keep
            });
        }

        // In local mode, local_sim's GameRuntime owns the scene's one and only
        // PhysicsWorld (see its doc comment above); in multiplayer mode it's
        // this client's own copy, built alongside the scene's grip points.
        let physics_ref = match local_sim.as_ref() {
            Some(sim) => sim.physics(),
            None => static_scene
                .physics
                .as_ref()
                .expect("multiplayer static_scene always builds its own physics"),
        };
        let locomotion_input = movement::step_locomotion(
            cs,
            dt,
            &rig,
            frame_count,
            server_player_offset,
            server_player_yaw,
            world.is_some(),
            &mut locomotion,
            physics_ref,
            prev_r_trigger,
            &input,
            &net,
        );

        // Rebuilt from this frame's now-updated locomotion (offset/yaw), not
        // the pre-movement one above. `rig` positions bake `player_offset`/
        // `player_yaw` into world space (see Locomotion::apply_to_head), and
        // everything from here on converts back to render space with THIS
        // frame's offset/yaw_inv (just below). Rendering with the old rig
        // and the new yaw_inv disagreed by exactly one frame's turn, which
        // is invisible for smooth turning but a sharp, visible snap of the
        // hands/avatar/hand-menu on every snap-turn (the default turn mode).
        // part_pull::handle_input's grab detection above intentionally still
        // used the pre-movement rig -- that's the existing, correct
        // one-frame-stale timing the multiplayer path already has.
        let rig = build_player_rig(&eye_views, &locomotion, cs, &hands, &synthetic_hand_config);

        // Steps the scene itself when there is no server to do it: physics,
        // scripts, part animations and grabbing all run right here, right now,
        // using this frame's own input/pose rather than the previous frame's
        // (the `world` above, used for grab detection, is one frame behind --
        // the same lag a live server round-trip already has). The scene-change
        // check above picks up a script-triggered scene switch on the next
        // frame, same latency as the multiplayer path.
        let world = if let Some(sim) = local_sim.as_mut() {
            let (fresh, _scene_change) = sim.step(
                dt,
                local_player,
                rig.clone(),
                input.clone(),
                locomotion_input,
                locomotion.player_offset,
                locomotion.player_yaw,
            );
            local_world = Some(fresh.clone());
            // Scripts and part triggers queue vibrations on the runtime, which
            // has no way to reach OpenXR itself; hand them to the controllers
            // now, the same frame they were asked for. Only the local sim has
            // any -- in multiplayer the scripts run on the server, which does
            // not send haptics over the wire (yet).
            for h in sim.drain_haptics(local_player) {
                for hand in [Hand::Left, Hand::Right] {
                    if !h.hand.includes(hand) {
                        continue;
                    }
                    if let Err(e) = controllers.vibrate(
                        &headset.session,
                        hand == Hand::Left,
                        h.strength,
                        h.seconds,
                    ) {
                        error!("haptics: vibrate {} failed: {e}", hand.as_str());
                    }
                }
            }
            Some(fresh)
        } else {
            world
        };
        let cuboids_src = world.as_ref().map(|w| &w.cuboids).unwrap_or(&empty_cuboids);
        let meshes_src = world.as_ref().map(|w| &w.meshes).unwrap_or(&empty_meshes);
        let lights_src = world.as_ref().map(|w| &w.lights).unwrap_or(&empty_lights);
        let particle_emitters_src = world
            .as_ref()
            .map(|w| &w.particle_emitters)
            .unwrap_or(&empty_particle_emitters);
        let particle_bursts_src = world
            .as_ref()
            .map(|w| &w.particle_bursts)
            .unwrap_or(&empty_particle_bursts);
        let lasers_src = world.as_ref().map(|w| &w.lasers).unwrap_or(&empty_lasers);

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

        // Selection pads: a controller put into one chooses the body or
        // hands, remembered in the player's settings, with a buzz.
        {
            if let Some(w) = world.as_ref() {
                if w.scene_name != selector_pads.scene_name() {
                    selector_pads = selector_pads::SelectorPads::load(&dir, &w.scene_name);
                }
            }
            let grips = [
                cs.l_grip_pose.map(|_| rig.hand_grip(space_soup_engine::Hand::Left).position),
                cs.r_grip_pose.map(|_| rig.hand_grip(space_soup_engine::Hand::Right).position),
            ];
            for chosen in selector_pads.update(grips) {
                // Only ids the catalog actually has.
                let known = match chosen.set.as_str() {
                    "body" => avatar_catalog.bodies.iter().any(|b| b.id == chosen.value),
                    "hands" => avatar_catalog.hands.iter().any(|h| h.id == chosen.value),
                    _ => false,
                };
                if !known {
                    log::warn!("selector pad: {} = '{}' isn't in avatars.json", chosen.set, chosen.value);
                    continue;
                }
                let (body, hands_id) = if chosen.set == "body" { (Some(chosen.value.as_str()), None) } else { (None, Some(chosen.value.as_str())) };
                let changed = hand_menu.choose(body, hands_id, &dir);
                info!("selector pad: {} = {} ({})", chosen.set, chosen.value, if changed { "changed" } else { "already" });
                if let Err(e) = controllers.vibrate(&headset.session, chosen.hand == 0, if changed { 0.6 } else { 0.25 }, 0.08) {
                    error!("haptics: vibrate failed: {e}");
                }
            }
        }

        let offset = locomotion.player_offset;
        let yaw_inv = Quat::from_rotation_y(-locomotion.player_yaw);
        let space = vr_props::Space { offset, yaw_inv };

        // Scene props: (re)load for this scene, then work them with this
        // frame's hands -- before the body solve, which they steer (a hand
        // on the slide rides the slide, fingers take the grip's pose).
        let mut prop_hands = [avatar_render::HandOverride::default(); 2];
        {
            let scene = world.as_ref().map(|w| w.scene_name.as_str()).unwrap_or(entry_scene.as_str());
            if vr_props.as_ref().is_none_or(|p| p.scene_name() != scene) {
                vr_props = Some(vr_props::VrProps::load(&dir, scene, &renderer));
            }
            let values = hand_menu.values();
            let tunings = [values.hand(true).tuning(), values.hand(false).tuning()];
            let grips = [
                cs.l_grip_pose.map(|_| rig.hand_grip(Hand::Left)),
                cs.r_grip_pose.map(|_| rig.hand_grip(Hand::Right)),
            ];
            let mut hands_in = [prop_core::play::HandIn::default(); 2];
            for i in 0..2 {
                let t = &tunings[i];
                // Where the solver will put this wrist (the hand's tuning
                // applied to its controller), world space.
                hands_in[i].wrist = grips[i].as_ref().map(|g| {
                    let r = g.rotation * avatar_render::grip_to_hand(i == 0);
                    (g.position + r * t.pos_offset, r * t.rot_offset)
                });
            }
            hands_in[0].squeeze = cs.l_squeeze;
            hands_in[1].squeeze = cs.r_squeeze;
            hands_in[0].trigger = cs.l_trigger;
            // The right trigger works the hand menu while it's open.
            hands_in[1].trigger = if hand_menu.open { 0.0 } else { cs.r_trigger };
            hands_in[0].button = cs.btn_x;
            hands_in[1].button = cs.btn_a;
            let body = local_body.map(|b| {
                // The solver's body faces +Z at yaw 0; the pouch layout faces -Z.
                let (feet, facing) = space.to_world((b.root_pos, Quat::from_rotation_y(b.root_yaw + std::f32::consts::PI)));
                prop_core::play::Body { feet, facing, k: values.height_cm / 100.0 / 1.8288 }
            });
            // Which version of each hand pose: the worn hands', else the body's own.
            let hands_key = match avatar_catalog.hands(&values.hands) {
                Some(h) if h.models().is_some() => h.id.clone(),
                _ => avatar_catalog.body(&values.body).map(|b| b.id.clone()).unwrap_or_default(),
            };
            if let Some(vp) = vr_props.as_mut() {
                let out = vp.step(&renderer, hands_in, body, &hands_key, dt);
                for i in 0..2 {
                    let t = &tunings[i];
                    // Back through the tuning, to what the solver takes.
                    prop_hands[i].input = out[i].wrist.map(|w| {
                        let (p, r) = space.to_render(w);
                        let r_in = r * t.rot_offset.inverse();
                        (p - r_in * t.pos_offset, r_in)
                    });
                    prop_hands[i].pose = out[i].fingers.map(|f| avatar_pose::HandPose { curls: f.curls, joints: f.joints, ..Default::default() });
                    let held = vp.holds(i);
                    if held != prop_held_log[i] {
                        info!("prop: {} hand {} ({:?})", if i == 0 { "left" } else { "right" }, if held { "took hold" } else { "let go" }, vp.held_ids());
                        prop_held_log[i] = held;
                    }
                }
            }
        }

        let head_pos = rig.head().position;

        {
            let to_render = |p: glam::Vec3| yaw_inv * (p - offset);
            let head_t = rig.head();
            let aim_t = rig.hand_aim(space_soup_engine::Hand::Right);
            hand_menu.update(
                &renderer,
                cs,
                (to_render(head_t.position), yaw_inv * head_t.rotation),
                (to_render(aim_t.position), yaw_inv * aim_t.rotation),
                &dir,
            );
        }

        let remotes = net.remote_players.lock().unwrap().clone();
        let bodies = avatar_render::build_bodies(local_player, &rig, &remotes);

        let pull_hands = part_pull::pull_hand_poses(
            &pull_sessions, &static_scene, &live_objects, &part_transforms,
        );
        let mut local_hand_world: [Option<avatar_ik::Transform>; 2] = [None, None];
        avatar_render::update_avatar_bodies(
            &mut renderer,
            &mut avatar_mesh_cache,
            &mut avatar_solve_states,
            if local_body_path.is_none() { &avatar_master_mesh } else { &local_body_mesh },
            &avatar_master_mesh,
            &mut gloves,
            &mut local_direct_mesh,
            local_player,
            hand_menu.values(),
            offset,
            yaw_inv,
            &world,
            cs,
            &bodies,
            &pull_hands,
            &prop_hands,
            &mut local_hand_world,
            &mut local_body,
        );

        // Props after the solve: the held one in the hand as solved, what
        // came out of it, and the sounds and rumbles it asked for.
        if let Some(vp) = vr_props.as_mut() {
            let solved = local_body.map_or([None, None], |b| [Some(space.to_world(b.wrists[0])), Some(space.to_world(b.wrists[1]))]);
            let (sounds, rumbles) = vp.pose(&renderer, solved, space);
            for snd in sounds {
                client_audio.play_once(&snd.file, snd.volume, snd.at);
            }
            for r in rumbles {
                if let Err(e) = controllers.vibrate(&headset.session, r.left, r.strength, r.seconds) {
                    error!("haptics: prop vibrate failed: {e}");
                }
            }
        }

        // A prop's own box isn't drawn -- the prop is.
        let prop_free_cuboids: Vec<WireRenderCuboid>;
        let cuboids_for_render: &[WireRenderCuboid] = match vr_props.as_ref() {
            Some(vp) => {
                prop_free_cuboids = cuboids_src.iter().filter(|c| !vp.is_prop(&c.id)).cloned().collect();
                &prop_free_cuboids
            }
            None => cuboids_src,
        };
        let (mut cuboids, lights, mut mesh_instances, mirror_only_mesh_instances, mirror_surface) =
            render_prep::build_render_lists(
                cuboids_for_render,
                lights_src,
                meshes_src,
                &mut mesh_cache,
                &mut hidden_part_meshes,
                &avatar_mesh_cache,
                &local_direct_mesh,
                &gloves,
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
                effective_rig_config.held_grip_offset(),
                &mut part_transforms,
            );
        // Anchored to the left wrist -- toggled by the left menu button, driven
        // by the right hand's stick/trigger. Just cuboids appended onto the same
        // list the scene renders from; nothing else in the frame needs to know
        // the menu exists.
        cuboids.extend_from_slice(hand_menu.laser_cuboids());
        if let Some(p) = vr_props.as_ref() {
            mesh_instances.extend(p.instances());
        }
        if let Some(menu_instance) = hand_menu.mesh_instance() {
            mesh_instances.push(menu_instance);
        }

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
