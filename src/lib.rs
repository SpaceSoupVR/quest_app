use log::{error, info};

pub mod avatar;
#[cfg(target_os = "android")]
mod avatar_render;
#[cfg(target_os = "android")]
mod client_audio;
// Not gated on Android: see the module's own note. Its OpenXR helpers are.
mod convert;
mod npu_probe;
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
mod offline_frame;
mod probe_level;
mod glare_fixtures;
mod mesh_masks;
mod scene_lights;
mod scene_meshes;
mod terrain_render;
mod water_render;
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

/// The scene's display settings, in the form the renderer wants.
///
/// Set beside the sky and for the same reason: both are properties of the level
/// being shown, and a level change must carry them across or the new scene is
/// graded with the old one's exposure.
fn post_upload_for(post: &space_soup_engine::scene::PostDef)
    -> space_soup::renderer::uniforms::PostUpload
{
    use space_soup_engine::scene::ToneMapDef;
    space_soup::renderer::uniforms::PostUpload {
        exposure: post.exposure,
        tonemap: match post.tonemap {
            ToneMapDef::Aces => space_soup::renderer::tonemap::ToneMapping::Aces,
            ToneMapDef::None => space_soup::renderer::tonemap::ToneMapping::None,
        },
        // Set per frame by the renderer.
        terrain_detail_distance: 0.0,
        reflection_share: false,
    }
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
use convert::to_space_soup_beam;
#[cfg(target_os = "android")]
use mesh_load::queue_new_meshes;
#[cfg(target_os = "android")]
use part_pull::PullSession;
#[cfg(target_os = "android")]
use platform::{game_dir, pump_android_events};

/// WHICH IN-HEADSET CONTROLS ARE LIVE. `false` is the tester's build: the
/// shipped settings and ONE control, the left stick click toggling the
/// lighting-sources view (user, 2026-09-29). `true` brings back the
/// developer A/B toggles: SSR on the right stick, the whole debug-view cycle
/// on the left, multiview on the menu button, SpaceWarp and foveation on a
/// trigger + menu. The lever file works either way.
#[cfg(target_os = "android")]
const DEVELOPER_TOGGLES: bool = false;

#[cfg(target_os = "android")]
fn run_inner() -> Result<(), Box<dyn std::error::Error>> {
    std::panic::set_hook(Box::new(|info| {
        error!("PANIC: {info}");
    }));
    // Before anything slow: startup runs for seconds before the frame loop
    // pumps input, and Android calls the app "not responding" after five.
    platform::spawn_input_drain();
    // NPU Gate 0: does this app reach the Hexagon DSP? Logs only, on its own
    // thread. See `npu_probe`.
    #[cfg(target_os = "android")]
    npu_probe::spawn();

    // Wall clock from the first line this process runs, so every STARTUP line
    // below can be read as "how far into startup" as well as "how long".
    let t_start = std::time::Instant::now();
    info!("STARTUP begin: waiting for activity resume");
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
    // The standalone lighting path: a game with no multiplayer server still has
    // to light its levels. See scene_lights.
    let mut static_lights = scene_lights::load(&dir, &static_scene.scene_name);
    // The level's Baked lamps, for the characters and the ground. See
    // `scene_lights::load_baked`.
    let mut baked_lights = scene_lights::load_baked(&dir, &static_scene.scene_name);
    // Each stationary lamp's channel of the baked shadow masks. See
    // `scene_lights::stationary_channels`.
    let mut stationary_channels = scene_lights::stationary_channels(&dir, &static_scene.scene_name);
    // The other half of the standalone path: without this a game with no server
    // draws its terrain, its brushes and its lighting, and none of the objects
    // standing in it. See scene_meshes.
    let mut static_meshes = scene_meshes::load(&dir, &static_scene.scene_name);
    {
        // Per scene, from the ids its own brush faces reference.
        let maps = brush_render::load_materials(&dir, brushes.materials());
        renderer.set_brush_materials(&maps.colours, &maps.normals, &maps.roughs, &maps.aos);
    }
    // Layer textures are per PROJECT, not per scene -- every level shares the
    // same four materials -- so they load once here rather than on every scene
    // change. A missing file leaves that layer flat-coloured; see
    // game/textures/terrain/SOURCES.md.
    let texture_dir = dir.join("textures").join("terrain");
    renderer.set_terrain_layers(
        space_soup::renderer::terrain_pipeline::load_terrain_layers(&texture_dir),
        space_soup::renderer::terrain_pipeline::load_terrain_normals(&texture_dir),
        space_soup::renderer::terrain_pipeline::load_terrain_rough(&texture_dir),
        space_soup::renderer::terrain_pipeline::load_terrain_ao(&texture_dir),
    );
    // How those layers tile. Per project like the textures, and authored in the
    // scene editor -- without this the headset renders every terrain at the
    // engine's built-in tile sizes no matter what the editor previewed, which
    // is a difference nobody can see until they put the headset on.
    renderer.set_terrain_settings(
        space_soup::renderer::terrain_pipeline::load_terrain_settings(&texture_dir),
    );
    renderer.set_terrain_splat(loaded_terrain.as_ref().and_then(|(_, s)| s.as_ref()));
    // The ground reflections land on. See `space_soup::renderer::ground_map`.
    renderer.set_terrain_heights(
        loaded_terrain.as_ref().and_then(|(t, _)| t.height_grid(static_scene.terrain.as_ref(), &dir)),
    );

    // WATER. Built once here rather than per frame: the surface is static world
    // geometry whose depth comes from the terrain, so the only thing that
    // changes at runtime is the frame it is expressed in and the wave clock.
    //
    // Loaded through `terrain::load` again rather than reusing the render
    // geometry: that copy has been decimated to a render LOD and rebased into
    // the player's frame, and a shoreline measured against it would be wrong by
    // however much the LOD smoothed the ground.
    let mut water_bodies = {
        let source = static_scene
            .terrain
            .as_ref()
            .and_then(|def| space_soup_engine::terrain::load(def, &dir).ok());
        water_render::build(&static_scene.water, source.as_deref())
    };
    renderer.set_water(
        &water_bodies
            .iter()
            .map(|b| (b.world().to_vec(), b.indices.clone(), b.uniform))
            .collect::<Vec<_>>(),
    );
    // The sky: the background, and the ambient inside the level. Projecting its
    // irradiance walks every texel, so it happens here and on a scene change --
    // never per frame.
    {
        let sky = loaders::load_scene_sky(&dir, static_scene.sky.as_ref());
        renderer.set_sky(
            sky.as_ref().map(|(p, _, _)| p),
            sky.as_ref().map_or(0.0, |(_, r, _)| *r),
            sky.as_ref().map_or(1.0, |(_, _, i)| *i),
        );
        renderer.set_post(post_upload_for(&static_scene.post));
    }
    let mut live_objects = grab_detect::LiveObjects::default();
    let mut client_audio = client_audio::ClientAudio::new();

    let mut server_player_offset: Option<Vec3> = None;
    let mut server_player_yaw: Option<f32> = None;

    let mut locomotion = Locomotion::new(LocomotionMode::Smooth);

    // BAKED LIGHTING FROM DISK, BEFORE ANY NETWORKING.
    //
    // This is the path a shipped game uses, and until now it did not exist: the
    // only source of lightmaps was the WebSocket below, which reaches the
    // editor's server on localhost:8000. On a headset with no server that
    // silently falls back to a white 1x1 texture, so every light shines through
    // every wall and the level looks lit but wrong -- there is no error, because
    // the fallback is deliberate and correct for "not baked yet".
    //
    // Loaded first so the WebSocket, when there IS one, overrides it. That
    // ordering is what keeps lighting edits appearing live while authoring
    // without making authoring a requirement for shipping.
    // BAKED REFLECTION PROBES, before the lightmaps so a probe is bound by the
    // time the first frame shades anything with it.
    //
    // Each probe's parallax box is the OBJECT's own cuboid -- the same one the
    // baker captured from -- so the two cannot disagree about which room a
    // probe describes.
    //
    // The lamps' glare is measured from the same cards the reflections use,
    // so it waits for them. See `glare_fixtures`.
    let mut glare_measured = HashMap::new();
    {
        // STARTUP IS TIMED, phase by phase.
        //
        // The app has been throwing "not responding" a few seconds after
        // launch all day -- input dispatch times out after 5 s and the main
        // thread is busy loading. Frames once running are 15 ms, so this is
        // not a rendering cost, and guessing which phase it is has been wrong
        // twice already. Each phase says how long it took, so the next log
        // names the culprit instead of narrowing it down.
        info!("STARTUP at probes: {} ms since process start", t_start.elapsed().as_millis());
        let t_probe = std::time::Instant::now();
        // DESCRIBED FROM THE INDEX, PIXELS FROM DISK ON DEMAND. The box and
        // capture point come from the bake -- one authored volume subdivides
        // into cells no scene object describes -- and each probe's file is
        // read when the renderer's pool needs it, so a large level never holds
        // every probe at once. See `space_soup::renderer::probe_stream`.
        if let Some(level) = probe_level::ProbeLevel::load(&dir, &static_scene.scene_name) {
            info!(
                "STARTUP probes: indexed {} probe(s), {} doorway(s) in {} ms",
                level.descs.len(),
                level.portals.len(),
                t_probe.elapsed().as_millis(),
            );
            let t_upload = std::time::Instant::now();
            let source = level.source();
            let depth = level.depth_source();
            // What stands in the rooms, for the reflection trace; the rooms'
            // own boxes are its walls. See `probe_level::ProbeLevel::proxies`.
            let standing = level.scene_proxies(&dir, &static_scene.scene_name);
            info!(
                "STARTUP probes: {} reflection prox(ies), {} model field(s), {} model(s) on cards",
                standing.proxies.len(),
                standing.fields.len(),
                standing.cards.len(),
            );
            let closed_rooms = level.closed_rooms.clone();
            // The buildings' outsides, for reflections that leave a building.
            let buildings = level.buildings(&dir, &static_scene.scene_name);
            info!("STARTUP probes: {} building outside(s)", buildings.len());
            renderer.set_building_outsides(buildings);
            renderer.set_reflection_probes_with_depth(level.descs, level.resolution, level.portals, source, Some(depth));
            renderer.set_reflection_proxies(standing.proxies, standing.fields, standing.cards);
            glare_measured = standing.glare;
            // Which rooms are walled all round but their doorways, so the
            // terrain outside is drawn only where a doorway shows it. After
            // the probes, whose room boxes it uses. See `portal_cull`.
            renderer.set_closed_rooms(&closed_rooms);
            info!(
                "STARTUP probes: metered, built mip chains and uploaded in {} ms",
                t_upload.elapsed().as_millis(),
            );
        }
    }
    // WHICH LAMPS GLARE, from which sides. See `glare_fixtures`.
    let mut lamp_glare = glare_fixtures::load(&dir, &static_scene.scene_name, &glare_measured);

    // THE STATIONARY LAMPS' SHADOWS ON MESHES, by object, from disk now and
    // from the editor's stream later. See `mesh_masks`.
    let mut mesh_masks: mesh_masks::MeshMasks<lightmap_client::LightmapUpdate> = mesh_masks::MeshMasks::default();
    {
        info!("STARTUP at lightmaps: {} ms since process start", t_start.elapsed().as_millis());
        let t_lm = std::time::Instant::now();
        let maps = space_soup_engine::lightmaps::load_scene_lightmaps(&dir, &static_scene.scene_name);
        let masked: HashSet<String> = maps.iter().filter_map(|m| mesh_masks.take(&m.object_id, &m.rgba, m.width, m.height)).collect();
        if !masked.is_empty() {
            info!("lightmaps: stationary masks on {} mesh(es)", masked.len());
        }
        info!(
            "STARTUP lightmaps: loaded {} baked map(s) from disk in {} ms",
            maps.len(),
            t_lm.elapsed().as_millis(),
        );
        // The brush bounce DIRECTION rides under its own reserved id while
        // carrying the same `Brush` target as the atlas it accompanies, so it
        // has to be picked out by id BEFORE the loop -- matching on target
        // alone would hand a map of unit vectors to `set_brush_lightmap` and
        // let whichever arrived last win.
        let brush_dir = maps
            .iter()
            .find(|m| m.object_id == space_soup_engine::lightmaps::SCENE_BRUSH_DIRECTION_ID)
            .map(|m| (m.rgba.clone(), m.width, m.height));
        if brush_dir.is_some() {
            info!("lightmaps: brush bounce direction map present");
        }
        // The sky sun's visibility mask, picked out by id for the same reason.
        let brush_sun = maps
            .iter()
            .find(|m| m.object_id == space_soup_engine::lightmaps::SCENE_BRUSH_SUN_MASK_ID)
            .map(|m| (m.rgba.clone(), m.width, m.height));
        match &brush_sun {
            Some((_, w, h)) => info!("lightmaps: brush sun mask {w}x{h} present"),
            None => info!("lightmaps: no brush sun mask; brushes take the sun's static map"),
        }
        // The stationary lamps' shadow masks, one image per two lamps, in
        // layer order. See `space_soup_engine::stationary`.
        // And the same lamps' masks on the ground, in the same layers. See
        // `space_soup_engine::lightmaps::scene_terrain_stationary_id`.
        let terrain_stationary_ids: Vec<String> = (0..space_soup_engine::stationary::MAX_STATIONARY_LAYERS)
            .map(space_soup_engine::lightmaps::scene_terrain_stationary_id)
            .collect();
        let stationary_ids: Vec<String> =
            (0..space_soup_engine::stationary::MAX_STATIONARY_LAYERS).map(space_soup_engine::lightmaps::scene_brush_stationary_id).collect();
        let stationary_maps: Vec<&space_soup_engine::lightmaps::LoadedLightmap> = scene_lights::usable_stationary_masks(
            stationary_ids.iter().map_while(|id| maps.iter().find(|m| &m.object_id == id)).collect(),
            &stationary_channels,
        );
        if let Some(first) = stationary_maps.first() {
            info!("lightmaps: {} stationary mask layer(s) {}x{}", stationary_maps.len(), first.width, first.height);
        }
        for m in &maps {
            if m.object_id == space_soup_engine::lightmaps::SCENE_BRUSH_DIRECTION_ID
                || m.object_id == space_soup_engine::lightmaps::SCENE_BRUSH_SUN_MASK_ID
                || stationary_ids.contains(&m.object_id)
            {
                continue;
            }
            // The level's brushes share one atlas under a reserved id, because
            // they share one draw call. Everything else is per object.
            if m.target == space_soup_engine::lightmaps::LightmapTarget::Brush {
                renderer.set_brush_lightmap(
                    lightmap_light(m),
                    m.width,
                    m.height,
                    brush_dir.as_ref().map(|(d, w, h)| (d.as_slice(), *w, *h)),
                    brush_sun.as_ref().map(|(d, w, h)| (d.as_slice(), *w, *h)),
                    &stationary_maps.iter().map(|m| m.rgba.as_slice()).collect::<Vec<_>>(),
                    stationary_maps.first().map_or((1, 1), |m| (m.width, m.height)),
                );
                continue;
            }
            // The stationary lamps' masks on the ground travel as terrain maps
            // too; they are installed together below, after this loop.
            if terrain_stationary_ids.contains(&m.object_id) {
                continue;
            }
            // The ground's sky-visibility map: one image for the level, sampled
            // by footprint position rather than belonging to any object.
            if m.target == space_soup_engine::lightmaps::LightmapTarget::Terrain {
                renderer.set_terrain_sky_occlusion(Some(
                    &space_soup::renderer::terrain_pipeline::TerrainImage {
                        width: m.width,
                        height: m.height,
                        rgba: m.rgba.clone(),
                    },
                ));
                // The footprint the map spans, without which a world position
                // cannot be turned back into a texel. Set together with the
                // image because either alone is useless.
                if let Some((t, _)) = loaded_terrain.as_ref() {
                    let b = t.world_bounds();
                    renderer.set_terrain_footprint(b.0, b.1);
                }
                info!("terrain sky occlusion: {}x{} loaded", m.width, m.height);
                continue;
            }
            // A mesh's masks go in with its light, below; they are not maps
            // of their own.
            if space_soup_engine::lightmaps::mesh_stationary_of(&m.object_id).is_some() {
                continue;
            }
            renderer.set_cuboid_lightmap(&m.object_id, lightmap_light(m), m.width, m.height);
            let (masks, mask_size) = mesh_masks.layers_for(&m.object_id, &stationary_channels);
            renderer.set_mesh_lightmap(&m.object_id, lightmap_light(m), m.width, m.height, &masks, mask_size);
        }
        // THE STATIONARY LAMPS' SHADOWS ON THE GROUND, layer by layer as on
        // the brushes, and only a set baked for these lamps. Without them the
        // lamps light the grass straight through the walls.
        // A bake from before the ground had masks simply has none, which is
        // not the stale-bake warning `usable_stationary_masks` gives.
        let found: Vec<&space_soup_engine::lightmaps::LoadedLightmap> =
            terrain_stationary_ids.iter().map_while(|id| maps.iter().find(|m| &m.object_id == id)).collect();
        let ground_masks =
            if found.is_empty() { Vec::new() } else { scene_lights::usable_stationary_masks(found, &stationary_channels) };
        if !ground_masks.is_empty() {
            info!("lightmaps: {} stationary mask layer(s) on the ground", ground_masks.len());
            renderer.set_terrain_stationary_masks(
                ground_masks
                    .iter()
                    .map(|m| space_soup::renderer::terrain_pipeline::TerrainImage {
                        width: m.width,
                        height: m.height,
                        rgba: m.rgba.clone(),
                    })
                    .collect(),
            );
        }
    }

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
    // The avatar's mean colour, for its capsules' reflections.
    let avatar_colour = space_soup_engine::mesh_lightmap::model_albedo(&boy_glb_path)
        .map_or([0.46, 0.34, 0.27], |a| a.to_array());
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
    // For the SSR A/B toggle below. The stick clicks are bound to nothing
    // else; A/B/X/Y belong to part_pull.
    let mut prev_r_stick_click = false;
    // For the brush debug-view cycle below.
    let mut prev_l_stick_click = false;
    let mut prev_btn_menu = false;
    // What the compositor was last asked for, logged when it changes.
    let mut last_layer_state: Option<(bool, bool)> = None;

    // RUNTIME LEVERS: switch renderer features on the headset without a
    // build. Polled about once a second; see `space_soup::renderer::levers`.
    let mut lever_file = space_soup::renderer::levers::LeverFile::new(platform::levers_path());
    let mut lever_tick: u32 = 0;
    renderer.set_perf_log(platform::perf_log_path());
    // A BENCHMARK VIEWPOINT from the lever file: the rig is moved onto it and
    // the tracked head pinned, so the frame is measured from a named place
    // with nobody wearing the headset. See `space_soup::renderer::bench`.
    // `bench_return` is where the player stood before, to go back to.
    let mut bench: Option<space_soup::renderer::bench::BenchRig> = None;
    let mut bench_return: Option<(Vec3, f32)> = None;
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
                // The runtime saying compositing, rendering or heat crossed a
                // warning level (`XR_EXT_performance_settings`): the signal
                // for when the game needs its performance level raised.
                Some(openxr::Event::PerfSettingsEXT(e)) => info!(
                    "XR PERF SETTINGS: {:?} {:?} {:?} -> {:?}",
                    e.domain(),
                    e.sub_domain(),
                    e.from_level(),
                    e.to_level()
                ),
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
            // A mesh's mask layer: kept, and put in with its light if the
            // light came first. See `mesh_masks`.
            if let Some(object) = mesh_masks.take(&update.object_id, &update.rgba, update.width, update.height) {
                if let Some(light) = mesh_masks.streamed_light(&object) {
                    let (masks, mask_size) = mesh_masks.layers_for(&object, &stationary_channels);
                    renderer.set_mesh_lightmap(&object, streamed_light(light), light.width, light.height, &masks, mask_size);
                }
                continue;
            }
            renderer.set_cuboid_lightmap(&update.object_id, streamed_light(&update), update.width, update.height);
            let (masks, mask_size) = mesh_masks.layers_for(&update.object_id, &stationary_channels);
            renderer.set_mesh_lightmap(&update.object_id, streamed_light(&update), update.width, update.height, &masks, mask_size);
            let id = update.object_id.clone();
            mesh_masks.remember_light(&id, update);
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

        let (view_flags, mut eye_views) = headset.session.locate_views(
            openxr::ViewConfigurationType::PRIMARY_STEREO,
            time,
            &headset.stage,
        )?;
        // PINNED: the rig stands where the view says, and the head is the
        // pinned one for everything this frame -- the rig, the lights chosen,
        // the audio -- exactly as the renderer will pin it. Released, the
        // player goes back to where they stood. Taken once, here: the lever
        // file is read later in the frame, and a pin that changed halfway
        // would draw the new head against geometry placed for the old rig.
        let frame_bench = bench;
        match &frame_bench {
            Some(rig) => {
                let located = view_flags.contains(openxr::ViewStateFlags::ORIENTATION_VALID)
                    && view_flags.contains(openxr::ViewStateFlags::POSITION_VALID);
                space_soup::renderer::bench::pin_xr_views(&mut eye_views, located, rig);
                if bench_return.is_none() {
                    bench_return = Some((locomotion.player_offset, locomotion.player_yaw));
                }
                locomotion.player_offset = rig.offset;
                locomotion.player_yaw = rig.yaw;
            }
            None => {
                if let Some((offset, yaw)) = bench_return.take() {
                    locomotion.player_offset = offset;
                    locomotion.player_yaw = yaw;
                }
            }
        }

        let now = std::time::Instant::now();
        let dt = last_time
            .map(|t| now.duration_since(t).as_secs_f32())
            .unwrap_or(1.0 / 90.0);
        last_time = Some(now);
        sim_time += dt;
        // The wave clock. Driven from sim_time rather than the frame counter so
        // the water moves at the same speed whatever the frame rate -- a wave
        // that sped up when the scene got simpler would be very noticeable.
        if renderer.water_body_count() > 0 {
            renderer.set_water_time(sim_time);
        }

        let cs = &controllers.state;

        // DEBUG A/B: the right stick click CYCLES THREE STATES, on the press
        // rather than while held, so one click is one step.
        //
        //   off -> inline -> buffered -> off
        //
        // `inline` marches and blends in the forward pass, which is what has
        // always shipped. `buffered` marches into a reflection buffer, filters
        // it across neighbouring pixels and composites the result -- the only
        // place the hit/miss cliff can be removed, and therefore the comb.
        //
        // One button rather than two because the comparison that matters is
        // between the two reflection paths FROM ONE VIEWPOINT, and reaching for
        // a second control moves your head.
        if DEVELOPER_TOGGLES && cs.r_stick_click && !prev_r_stick_click {
            let (on, buffered) =
                match (renderer.screen_space_reflections(), renderer.buffered_reflections()) {
                    (false, _) => (true, false),
                    (true, false) => (true, true),
                    (true, true) => (false, false),
                };
            renderer.set_screen_space_reflections(on);
            renderer.set_buffered_reflections(buffered);
            info!(
                "SSR -> {} by right stick click",
                match (on, buffered) {
                    (false, _) => "OFF",
                    (true, false) => "ON (inline march)",
                    (true, true) => "ON (buffered: trace + resolve + composite)",
                },
            );
            // BACK TO THE NORMAL PICTURE. A debug view left on from the left
            // stick is invisible while SSR is off (the SSR view only draws in
            // the reflection pass), so switching SSR on appeared to switch the
            // false-colour view on instead of reflections (headset, 2026-09-17).
            if renderer.debug_view() != space_soup::renderer::brush_pipeline::DebugView::Off {
                renderer.set_debug_view(space_soup::renderer::brush_pipeline::DebugView::Off);
                info!("debug view -> Off (reset by the SSR toggle)");
            }
        }
        prev_r_stick_click = cs.r_stick_click;
        // DIAGNOSTIC: the left stick click cycles what brushes are drawn with --
        // off, lighting sources (`SOURCES_VIEW`: red baked, green probe, blue
        // direct), then SSR (blue too rough, red left frame, green out of
        // steps, magenta facing the viewer, grey a hit). The SSR view only
        // shows while SSR is on. The tester's build toggles off <-> sources.
        if cs.l_stick_click && !prev_l_stick_click {
            let now = renderer.debug_view();
            let view = if DEVELOPER_TOGGLES { now.next() } else { now.toggle_sources() };
            renderer.set_debug_view(view);
            info!("debug view -> {:?} by left stick click", view);
        }
        prev_l_stick_click = cs.l_stick_click;
        // MULTIVIEW A/B: the menu button flips the stereo scene pass, which
        // draws both eyes in one go instead of once each. Its own control
        // rather than a state in the SSR cycle, because the comparison that
        // matters is the SCENE pass and it must be switchable without touching
        // reflections.
        //
        // `set_multiview_scene` returns whether it TOOK: on a device without
        // MULTIVIEW or MULTISAMPLE_ARRAY the stereo pipelines were never built
        // and it stays off. Logging the request rather than the answer would
        // describe a frame that is still drawing one eye at a time.
        // IN-HEADSET A/B OF THE NEW FRAME-RATE FEATURES, on the same button
        // with a trigger held, so neither takes a gameplay control:
        //   left trigger + menu:  Application SpaceWarp on/off (lever
        //                         `space_warp`; 36 fps rendered, the rest
        //                         made by the compositor)
        //   right trigger + menu: foveation off -> low -> medium -> high
        // The lever file still wins when it changes.
        if DEVELOPER_TOGGLES && cs.btn_menu && !prev_btn_menu && (cs.l_trigger > 0.5 || cs.r_trigger > 0.5) {
            use space_soup::renderer::foveation::FoveationLevel;
            let mut levers = renderer.levers();
            if cs.l_trigger > 0.5 {
                levers.space_warp = !levers.space_warp;
                info!("space warp -> {} by left trigger + menu", if levers.space_warp { "ON" } else { "OFF" });
            } else {
                levers.foveation = match levers.foveation {
                    FoveationLevel::Off => FoveationLevel::Low,
                    FoveationLevel::Low => FoveationLevel::Medium,
                    FoveationLevel::Medium => FoveationLevel::High,
                    FoveationLevel::High => FoveationLevel::Off,
                };
                info!("foveation -> {} by right trigger + menu", levers.foveation.label());
            }
            renderer.set_levers(levers);
        } else if DEVELOPER_TOGGLES && cs.btn_menu && !prev_btn_menu {
            let want = !renderer.multiview_scene();
            let got = renderer.set_multiview_scene(want);
            info!(
                "multiview scene pass -> {} by menu button{}",
                if got { "ON (both eyes in one pass)" } else { "OFF (one pass per eye)" },
                if want && !got { " -- REFUSED: this device has no stereo pipelines" } else { "" },
            );
        }
        prev_btn_menu = cs.btn_menu;

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
        // Cloned for the same reason as the lights: a scene change reassigns
        // this later in the loop, and a borrow held across that will not build.
        let fallback_meshes = static_meshes.clone();
        let meshes_src = match world.as_ref() {
            Some(w) if !w.meshes.is_empty() => &w.meshes,
            _ => &fallback_meshes,
        };
        let _ = &empty_meshes;
        // A connected server runs the same collection over the same scene and
        // sends the same lights, so taking both would light the level twice.
        // The disk copy is the FALLBACK, not an addition.
        // Cloned rather than borrowed: a scene change reassigns `static_lights`
        // later in the same loop, and a borrow held across that is a compile
        // error. A handful of lights per frame is nothing next to the copy the
        // networked path already makes.
        let fallback_lights = static_lights.clone();
        let lights_src = match world.as_ref() {
            Some(w) if !w.lights.is_empty() => &w.lights,
            _ => &fallback_lights,
        };
        let _ = &empty_lights;
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
                static_lights = scene_lights::load(&dir, &w.scene_name);
                baked_lights = scene_lights::load_baked(&dir, &w.scene_name);
                stationary_channels = scene_lights::stationary_channels(&dir, &w.scene_name);
                // The authored faces only: the probes, and with them the cards
                // the glare is measured from, are the first scene's.
                lamp_glare = glare_fixtures::load(&dir, &w.scene_name, &HashMap::new());
                static_meshes = scene_meshes::load(&dir, &w.scene_name);
                {
                    let sky = loaders::load_scene_sky(&dir, static_scene.sky.as_ref());
                    renderer.set_sky(
                        sky.as_ref().map(|(p, _, _)| p),
                        sky.as_ref().map_or(0.0, |(_, r, _)| *r),
                        sky.as_ref().map_or(1.0, |(_, _, i)| *i),
                    );
                    renderer.set_post(post_upload_for(&static_scene.post));
                }
                // Alongside the geometry: the previous level's materials would
                // otherwise be bound against this one's layer numbering, which
                // paints every wall with an unrelated texture.
                let maps = brush_render::load_materials(&dir, brushes.materials());
                renderer.set_brush_materials(&maps.colours, &maps.normals, &maps.roughs, &maps.aos);
                // Per scene, alongside the geometry: a splat map left over from
                // the previous level would paint this one with its materials.
                renderer.set_terrain_splat(
                    loaded_terrain.as_ref().and_then(|(_, s)| s.as_ref()),
                );
                renderer.set_terrain_heights(
                    loaded_terrain.as_ref().and_then(|(t, _)| t.height_grid(static_scene.terrain.as_ref(), &dir)),
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

        // Held still while pinned: a stick knocked on the desk must not walk
        // the benchmark out of its viewpoint.
        if frame_bench.is_none() {
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
        }

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
        let mut capsule_groups: Vec<space_soup::renderer::uniforms::CapsuleGroup> = Vec::new();
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
            &mut capsule_groups,
            avatar_colour,
        );
        // The local player first, then everyone else nearest first: the
        // shaders take the first few. See `CapsuleUpload`.
        if capsule_groups.len() > 2 {
            let here = capsule_groups[0].capsules.first().map_or(Vec3::ZERO, |c| c.0);
            let dist = |g: &space_soup::renderer::uniforms::CapsuleGroup| {
                g.capsules.first().map_or(f32::MAX, |c| (c.0 - here).length_squared())
            };
            capsule_groups[1..].sort_by(|a, b| dist(a).total_cmp(&dist(b)));
        }
        renderer.set_capsules(&capsule_groups);

        lever_tick = lever_tick.wrapping_add(1);
        if lever_tick % 72 == 0 {
            match lever_file.poll() {
                Some(Ok(levers)) => {
                    info!("LEVERS: {} (from {})", levers.summary(), lever_file.path().display());
                    bench = levers.bench.as_ref().map(space_soup::renderer::bench::BenchRig::for_pose);
                    renderer.set_levers(levers);
                }
                Some(Err(e)) => log::warn!("LEVERS: {} ignored, previous levers kept: {e}", lever_file.path().display()),
                None => {}
            }
        }

        // In the player's frame, like the live lights `build_render_lists` makes.
        renderer.set_baked_lights(
            baked_lights.iter().map(|l| convert::to_space_soup_light(l, offset, yaw_inv)).collect(),
        );
        // Every lamp with a fixture, live or baked: a bulb glares however its
        // light is shaded.
        renderer.set_glare_sources(glare_fixtures::sources(
            lights_src.iter().chain(baked_lights.iter()),
            &lamp_glare,
            offset,
            yaw_inv,
        ));
        let (cuboids, lights, mesh_instances, mirror_only_mesh_instances, mirror_surface) =
            render_prep::build_render_lists(
                cuboids_src,
                lights_src,
                meshes_src,
                &mut mesh_cache,
                &mut hidden_part_meshes,
                &avatar_mesh_cache,
                // Not while pinned: the controllers on the desk would carry the
                // player's hands into the benchmark's view (and its cost).
                if frame_bench.is_some() { &None } else { &local_direct_mesh },
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
        // STATIONARY lamps take their shadows from their mask channel; the
        // list is `lights_src` converted in order, so each pairs with its id.
        let lights: Vec<space_soup::renderer::Light> = lights
            .into_iter()
            .zip(lights_src.iter())
            .map(|(mut l, src)| {
                l.mask_channel = stationary_channels.get(&src.id).copied();
                l
            })
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

        let mut particles = particles::simulate(particle_emitters_src, sim_time, offset, yaw_inv);
        particles.extend(particles::simulate_bursts(particle_bursts_src, offset, yaw_inv));
        let beams: Vec<Beam> = lasers_src
            .iter()
            .map(|rl| to_space_soup_beam(rl, offset, yaw_inv))
            .collect();
        // The renderer needs to know where the player is, so geometry that must
        // stay pinned to the world -- terrain's texture projection -- can undo
        // the player-frame transform. Without it the ground's texture travels
        // with the player and walking looks like standing still.
        renderer.set_player_frame(offset, locomotion.player_yaw);
        renderer.set_pinned_head(frame_bench.map(|r| (r.head_position, r.head_rotation)));

        // Into the player's frame, exactly like brushes below. Passing the raw
        // world-space vertices left the ground glued to the player: walking
        // moved every other object past you while the terrain came along, which
        // reads as "the room is sliding around" rather than as a terrain bug.
        // Water follows the player's frame exactly as brushes and terrain do,
        // and is re-uploaded ONLY when that frame actually changed.
        for (i, body) in water_bodies.iter_mut().enumerate() {
            if let Some(posed) = body.assemble(offset, yaw_inv, locomotion.player_yaw) {
                renderer.update_water_surface(i, posed);
            }
        }

        // Assembled once for its side effect, so the chunk bounds are rebuilt
        // into the player's frame, then read as an OWNED list before the slices
        // are taken -- `assemble` borrows mutably and `caster_chunks` borrows
        // shared, and the two cannot overlap. The second `assemble` is the
        // cached path and costs a comparison.
        if let Some((t, _)) = loaded_terrain.as_mut() {
            t.assemble(offset, yaw_inv, locomotion.player_yaw);
        }
        let terrain_chunks: Vec<space_soup::renderer::shadow::CasterChunk> = loaded_terrain
            .as_ref()
            .map(|(t, _)| t.caster_chunks())
            .unwrap_or_default();
        let terrain_arg = loaded_terrain
            .as_mut()
            .map(|(g, _)| g)
            .and_then(|t| t.assemble(offset, yaw_inv, locomotion.player_yaw));
        // Transformed into the player's frame here rather than at load, and
        // only when something moved -- see BrushGeometry::assemble.
        let brush_arg = brushes.assemble(
            hidden_brushes,
            offset,
            yaw_inv,
            locomotion.player_yaw,
        );

        // SHARPENING NEVER RIDES WITH SPACEWARP: together the compositor tore
        // 30-60 times a second. See `layer_settings::sharpening_for`. Bit 64
        // of `Levers::space_warp_debug` still drops it for diagnosis.
        let sharpening = space_soup::renderer::layer_settings::sharpening_for(renderer.space_warp_running());
        let layer_settings = xr.has_layer_settings
            && sharpening.wants_layer_settings()
            && renderer.levers().space_warp_debug & 64 == 0;
        if last_layer_state != Some((renderer.space_warp_running(), layer_settings)) {
            last_layer_state = Some((renderer.space_warp_running(), layer_settings));
            info!(
                "COMPOSITOR: space warp {}, sharpening {}",
                if renderer.space_warp_running() { "ON" } else { "off" },
                if layer_settings { format!("{sharpening:?}") } else { "off".to_string() },
            );
        }
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
            &terrain_chunks,
            brush_arg,
            mirror_surface,
        )?;
        // NO VIEWS MEANS THE FRAME WAS NOT LOCATED. The renderer skipped it
        // rather than drawing from poses that mean nothing; end the frame with
        // no layers, which OpenXR allows and the compositor covers with the
        // previous frame. Submitting an EMPTY projection layer instead is what
        // `XR_ERROR_POSE_INVALID` is, and it used to kill the app on a cold
        // start before tracking had settled.
        if proj_views.is_empty() {
            headset
                .frame_stream
                .end(time, openxr::EnvironmentBlendMode::OPAQUE, &[])?;
        } else {
            // MQSR -- compositor-side sharpening, asked for through
            // `XR_FB_composition_layer_settings`. See
            // `space_soup::renderer::layer_settings` for what it costs and why
            // it is worth having while RENDER_SCALE is below 1.0.
            //
            // The safe `CompositionLayerProjection` builder exposes no `next`,
            // so the settings struct is chained onto the RAW layer and wrapped
            // straight back up. `settings` is a local of this block and is
            // never moved, so it outlives the `end` call that follows the
            // pointer -- that, plus `proj_views` outliving the same call, is
            // the whole of what `from_raw` is unsafe about.
            use space_soup::renderer::layer_settings::Sharpening;

            let settings = openxr::sys::CompositionLayerSettingsFB {
                ty: openxr::sys::CompositionLayerSettingsFB::TYPE,
                next: std::ptr::null(),
                layer_flags: match sharpening {
                    Sharpening::Quality => {
                        openxr::sys::CompositionLayerSettingsFlagsFB::QUALITY_SHARPENING
                    }
                    Sharpening::Normal => {
                        openxr::sys::CompositionLayerSettingsFlagsFB::NORMAL_SHARPENING
                    }
                    // Never chained -- `layer_settings` above is false when
                    // the policy is Off, so this value is never read by anyone.
                    Sharpening::Off => openxr::sys::CompositionLayerSettingsFlagsFB::EMPTY,
                },
            };

            let mut raw = openxr::CompositionLayerProjection::new()
                .space(&headset.stage)
                .views(&proj_views)
                .into_raw();
            if layer_settings {
                raw.next = &settings as *const _ as *const std::ffi::c_void;
            }
            let proj_layer =
                unsafe { openxr::CompositionLayerProjection::<openxr::Vulkan>::from_raw(raw) };

            headset
                .frame_stream
                .end(time, openxr::EnvironmentBlendMode::OPAQUE, &[&proj_layer])?;
        }

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

// And the offline harness, which builds the ground map from it as the headset does.
#[cfg(any(target_os = "android", test))]
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
    // Every brush in the level, so terrain buried under a structure's floor is
    // hidden or lowered in the drawn copy. See `terrain_render::bury_under_structures`.
    let structures: Vec<&space_soup_engine::brush::BrushDef> =
        scene.objects.iter().filter_map(|o| o.brush.as_ref()).collect();
    let geometry = terrain_render::load(def, game_dir, 1, &structures)?;

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

/// A loaded lightmap's light in the form the renderer takes: the half floats
/// where the bake stored them, the 8-bit sRGB bytes for an older bake.
fn lightmap_light(m: &space_soup_engine::lightmaps::LoadedLightmap) -> space_soup::renderer::mesh::LightmapLight<'_> {
    match &m.linear {
        Some(l) => space_soup::renderer::mesh::LightmapLight::Linear(l),
        None => space_soup::renderer::mesh::LightmapLight::Srgb8(&m.rgba),
    }
}

/// The same for a map the editor's stream sent.
#[cfg(target_os = "android")]
fn streamed_light(u: &lightmap_client::LightmapUpdate) -> space_soup::renderer::mesh::LightmapLight<'_> {
    match &u.linear {
        Some(l) => space_soup::renderer::mesh::LightmapLight::Linear(l),
        None => space_soup::renderer::mesh::LightmapLight::Srgb8(&u.rgba),
    }
}
