//! A LEVEL, RENDERED OFF THE HEADSET THROUGH THE HEADSET'S OWN SHADERS.
//!
//! Every lighting defect in this project has been chased by building an APK,
//! walking to a spot, taking a screenshot and guessing. That loop is slow, and
//! worse, it cannot hold anything still: the question "does this pixel change
//! if the normal map is off?" needs two renders of the same pixel.
//!
//! This draws a scene's brushes with the real `BrushPipeline` -- 4x MSAA, the
//! real materials, lightmap, direction map, sun mask, reflection probes, sky
//! and lights, all loaded by the same functions the Quest app calls -- from any
//! eye, at the headset's eye-buffer resolution, into an image on disk. The
//! geometry goes through `BrushGeometry::assemble`, the headset's own player
//! frame transform, so its floating-point behaviour is the headset's too.
//!
//! Shadows are off (every light unshadowed, the static sun map absent): this
//! is for looking at surfaces, and spot shadows would need the whole caster
//! set. The sun on brushes still comes from its baked mask.
#![cfg(test)]

use glam::{Mat4, Quat, Vec3};
use space_soup::renderer::brush_pipeline::{BrushMaterials, BrushPipeline};
use space_soup::renderer::multiview::ViewMode;
use space_soup::renderer::lights::{Light, LightsUniform};
use space_soup::renderer::shadow::ShadowMap;
use space_soup::renderer::uniforms::{PlayerUpload, PostUpload, ShadowUpload, SkyUpload, UniformBuffer};

use crate::brush_render::{load_materials, BrushGeometry};

/// A STANDING FIGURE as capsules, for the characters' shadows, contact
/// darkening and reflections: `CAPSULE_AT=x,z` (or `x,y,z`) stands one 1.75 m
/// tall, arms down, feet on the floor there, facing +z. None without it. In the
/// harness's player frame: the world less `offset`, turned by `yaw_inv`.
/// After it, `carried` -- a torch in the reflections.
fn offline_capsules(
    offset: Vec3,
    yaw_inv: Quat,
    carried: Option<space_soup::renderer::uniforms::CapsuleGroup>,
) -> space_soup::renderer::uniforms::CapsuleUpload {
    use space_soup::renderer::uniforms::{CapsuleGroup, CapsuleUpload};
    let at = std::env::var("CAPSULE_AT").ok().and_then(|v| {
        let n: Vec<f32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        match n.as_slice() {
            [x, z] => Some(Vec3::new(*x, 0.0, *z)),
            [x, y, z] => Some(Vec3::new(*x, *y, *z)),
            _ => None,
        }
    });
    let Some(at) = at else { return CapsuleUpload::from_groups(&carried.into_iter().collect::<Vec<_>>()) };
    let p = |x: f32, y: f32, z: f32| yaw_inv * (at + Vec3::new(x, y, z) - offset);
    let capsules = vec![
        (p(0.0, 1.60, 0.0), p(0.0, 1.68, 0.0), 0.10),
        (p(0.0, 1.40, 0.0), p(0.0, 0.95, 0.0), 0.15),
        (p(-0.19, 1.42, 0.0), p(-0.21, 1.15, 0.0), 0.047),
        (p(0.19, 1.42, 0.0), p(0.21, 1.15, 0.0), 0.047),
        (p(-0.21, 1.15, 0.0), p(-0.22, 0.88, 0.0), 0.037),
        (p(0.21, 1.15, 0.0), p(0.22, 0.88, 0.0), 0.037),
        (p(-0.1, 0.92, 0.0), p(-0.1, 0.5, 0.0), 0.073),
        (p(0.1, 0.92, 0.0), p(0.1, 0.5, 0.0), 0.073),
        (p(-0.1, 0.5, 0.0), p(-0.1, 0.08, 0.0), 0.052),
        (p(0.1, 0.5, 0.0), p(0.1, 0.08, 0.0), 0.052),
        (p(-0.1, 0.04, 0.0), p(-0.1, 0.04, 0.16), 0.04),
        (p(0.1, 0.04, 0.0), p(0.1, 0.04, 0.16), 0.04),
    ];
    let figure = CapsuleGroup { capsules, colour: [0.46, 0.34, 0.27], surfaces: Vec::new() };
    CapsuleUpload::from_groups(&std::iter::once(figure).chain(carried).collect::<Vec<_>>())
}

/// Quest 3 eye buffer at the shipped render scale (2064 x 2208 x 0.7).
pub const EYE_W: u32 = 1445;
pub const EYE_H: u32 = 1546;

pub struct Shot {
    pub width: u32,
    pub height: u32,
    /// sRGB-encoded RGBA, as the swapchain would hold it.
    pub rgba: Vec<u8>,
}

#[derive(Clone, Copy)]
pub struct View {
    pub eye: Vec3,
    pub at: Vec3,
    /// Vertical field of view in degrees.
    pub fov_y: f32,
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    /// Draw the lighting-sources diagnostic instead of the shaded picture.
    pub sources: bool,
    /// Eye adaptation as the headset does it, settled: metered from the probe
    /// at `eye` looking at `at`. Off renders at exposure 1.
    pub adapt: bool,
    /// Switch the doorway handover off, to see what it changes.
    pub no_portals: bool,
    /// The light loop's culling of lamps that cannot reach a pixel -- the
    /// `light_culling` lever. On as shipped.
    pub light_culling: bool,
    /// Each lamp's terminator shaded over the pixel's footprint -- the
    /// `terminator_aa` lever. On as shipped.
    pub terminator_aa: bool,
    /// MEASUREMENT: one of the scene shader's register cuts applied to the
    /// picture (`BrushPipeline::new_multisampled_probe_reader_with_cut`), to
    /// see what that term contributes. Half-resolution reflections only.
    pub cut: Option<&'static str>,
    /// Reflections from the half-resolution probe pass rather than traced per
    /// pixel -- the `half_res_reflections` lever. Not for the sources view,
    /// which only the per-pixel shader paints.
    pub half_res_reflections: bool,
    /// The brushes' depth drawn first, the `depth_prepass` lever. On as
    /// shipped; it must change no pixel.
    pub depth_prepass: bool,
    /// Raw radiance -- no tone curve, exposure 1 -- to set beside a Cycles
    /// reference, whose PNG is the same sRGB-encoded radiance.
    pub linear: bool,
    /// The head's roll about the view direction, degrees. An edge the eye
    /// sees vertical runs down the pixel grid and cannot show a staircase;
    /// the headset's 22:04:15 screenshot was tilted about 15 degrees, and its
    /// reflected doorway edges stepped (2026-09-29).
    pub roll_deg: f32,
    /// The rig's turn, degrees: `Locomotion::player_yaw`, which a snap turn
    /// changes. The world drawn is the same at any turn; only the player's
    /// frame, the one geometry and lights reach the shaders in, rotates. A
    /// picture that changes with it is a frame bug (the headset's 45-degree
    /// snap turn at 00:59, 2026-10-01).
    pub yaw_deg: f32,
    /// The pixel grid shifted by this many pixels, x and y, with the camera
    /// held still: the crawl test's probe. A band-limited picture resamples
    /// smoothly under it; a hard per-pixel decision flips whole pixels.
    pub jitter_px: [f32; 2],
    /// The player's flashlight, its glass and the point it is aimed at, as a
    /// bench view's `flashlight` has them: its beam and the lights of its
    /// bounce (`flashlight_bounce`). The beam casts no shadow here.
    pub flashlight: Option<(Vec3, Vec3)>,
    /// With the flashlight, its torch in the reflections, as its capsules
    /// (`flashlight::torch_capsules`). The torch itself is not drawn here.
    pub torch: bool,
    /// The ground drawn too, as the headset draws it: into the probe pass
    /// after the brushes and in the scene pass after their depth. Only with
    /// the shipped half-resolution reflections and their deferred lookups.
    /// See `offline_terrain`.
    pub terrain: bool,
    /// The level's effects drawn too, as the headset draws them after
    /// everything opaque, at `EFFECTS_TIME` seconds (30 by default). Only
    /// with the half-resolution probe pass, whose depth they fade into the
    /// walls by. See `space_soup::renderer::effects`.
    pub effects: bool,
    /// The level's water and the sky drawn too, as the headset's scene pass
    /// draws them after everything opaque: the waves at `WATER_TIME` seconds
    /// (30 by default). Off, what nothing covers stays magenta, the hole
    /// finder's colour.
    pub water: bool,
    /// The level's weather drawn too (`weather_render`): the ground's weather
    /// twins over its areas and the falling rain and snow, as they stand
    /// `WEATHER_TIME` seconds after the level opened (600 by default).
    pub weather: bool,
}

impl View {
    pub fn headset(eye: Vec3, at: Vec3) -> Self {
        Self {
            eye,
            at,
            fov_y: 104.0,
            width: EYE_W,
            height: EYE_H,
            samples: 4,
            sources: false,
            adapt: false,
            no_portals: false,
            light_culling: true,
            terminator_aa: true,
            cut: None,
            half_res_reflections: true,
            depth_prepass: true,
            linear: false,
            roll_deg: 0.0,
            yaw_deg: 0.0,
            jitter_px: [0.0, 0.0],
            flashlight: None,
            torch: false,
            terrain: false,
            effects: false,
            water: false,
            weather: false,
        }
    }
}

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
    // `OFFLINE_F16=1`: render with half-precision arithmetic where the shaders
    // use it, as the headset does (see `space_soup::renderer::shader_precision`),
    // to compare against the default f32 renders. Apple GPUs have f16.
    let f16 = std::env::var("OFFLINE_F16").is_ok_and(|v| v == "1")
        && adapter.features().contains(wgpu::Features::SHADER_F16);
    // Dual-source blending, as the headset asks for it: the water's tint.
    let dual = adapter.features() & wgpu::Features::DUAL_SOURCE_BLENDING;
    let desc = wgpu::DeviceDescriptor {
        required_features: dual | if f16 { wgpu::Features::SHADER_F16 } else { wgpu::Features::empty() },
        // As many textures as the scene's shaders bind, as the headset asks;
        // and buffers as large as this machine's GPU takes, for the aliasing
        // test's supersampled renders (its probe pass fix-up list outgrows
        // the default 256 MB at three times the eye's size).
        required_limits: wgpu::Limits {
            max_buffer_size: adapter.limits().max_buffer_size,
            max_storage_buffer_binding_size: adapter.limits().max_storage_buffer_binding_size,
            ..space_soup::renderer::uniforms::scene_limits(wgpu::Limits::default())
        },
        ..Default::default()
    };
    pollster::block_on(adapter.request_device(&desc)).ok()
}

/// Render `scene_name` from `view`. `None` when no GPU is available.
pub fn render_brushes(scene_name: &str, view: View) -> Option<Shot> {
    let (device, queue) = device()?;
    let game = crate::offline_frame::offline_game_dir();
    let scene = space_soup_engine::scene::Scene::load(&space_soup_engine::Manifest::scene_path(&game, scene_name)).ok()?;
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;

    // THE PLAYER'S FRAME, as on the headset: the rig stands under the eye.
    let offset = Vec3::new(view.eye.x, 0.0, view.eye.z);
    let yaw = view.yaw_deg.to_radians();
    let yaw_inv = Quat::from_rotation_y(-yaw);
    let to_player = |p: Vec3| yaw_inv * (p - offset);

    // Lights: the scene's realtime ones through the wire conversion, plus the
    // sky's sun exactly as the frame adds it.
    // The panorama itself, kept: the reflections' sky is made from it too.
    let sky_pano = scene.sky.as_ref().and_then(|s| {
        let bytes = std::fs::read(game.join("skies").join(&s.id).join("sky.hdr")).ok()?;
        let p = space_soup::renderer::sky::decode_radiance(&bytes).ok()?;
        Some((p, s.rotation_deg, s.intensity))
    });
    let pano = sky_pano.as_ref().map(|(p, r, i)| space_soup::renderer::sky::sky_lighting(p, *r, *i));
    let (irradiance, sky_sun) = pano.unwrap_or((
        space_soup::renderer::sky::SkyIrradiance::flat(space_soup::renderer::sky::AMBIENT),
        None,
    ));
    // `TIME_OF_DAY=hours`: THE LEVEL AT THAT HOUR, lit as the headset's time
    // of day lights it (`XrRenderer::apply_time_of_day`): the sky, its ambient
    // and the ONE directional light (the sun, or the moon after it) from the
    // time-of-day sky; the brush atlas rebuilt from its daylight layers; the
    // probes relit from the lamps-only photographs; the sun's level shadow from
    // the static map (the baked masks hold the baked direction only).
    // `NIGHT_VISION=0` leaves the Purkinje shift out. See
    // `docs/time-of-day-2026-10-07.md`.
    let tod = crate::time_of_day::offline_hour()
        .map(|hour| crate::time_of_day::offline_snapshot(scene.sky.as_ref(), sky_sun.map(|s| s.direction), hour));
    let (baked_irradiance, baked_sun) = (irradiance, sky_sun);
    let (irradiance, sky_sun) = match &tod {
        Some((_, snap, _)) => (snap.irradiance, snap.light),
        None => (irradiance, sky_sun),
    };
    let tod_daylight = tod
        .as_ref()
        .map(|(_, snap, _)| space_soup::renderer::time_of_day::daylight_share(&baked_irradiance, baked_sun.as_ref(), snap));
    let tod_weights = tod.as_ref().map(|(_, snap, _)| {
        space_soup::renderer::sky::LayerWeights::between(&baked_irradiance, baked_sun.as_ref(), &snap.irradiance, snap.light.as_ref())
    });
    if let (Some((_, snap, _)), Some(k), Some(w)) = (&tod, tod_daylight, tod_weights) {
        let elevation = |d: [f32; 3]| d[1].clamp(-1.0, 1.0).asin().to_degrees();
        eprintln!(
            "offline frame: time of day {:.2} h: sun {:.1} deg {:?}, moon {:.1} deg {:?}, light {} {:?}, daylight x{k:.3e}, layers sky {:?} sun {:?}, {:.0} cd/m2 per unit",
            snap.hour,
            elevation(snap.sun.direction),
            snap.sun.direction,
            elevation(snap.moon.direction),
            snap.moon.direction,
            if snap.light.is_none() { "none" } else if snap.light_is_moon { "moon" } else { "sun" },
            snap.light_rgb(),
            w.sky,
            w.sun,
            snap.candelas_per_engine,
        );
    }
    // Stationary lamps with their mask channel, exactly as the headset pairs
    // them. See `scene_lights::stationary_channels`.
    let channels = crate::scene_lights::stationary_channels(&game, scene_name);
    let mut lights: Vec<Light> = crate::scene_lights::load(&game, scene_name)
        .iter()
        .map(|l| {
            let mut light = crate::convert::to_space_soup_light(l, offset, yaw_inv);
            light.mask_channel = channels.get(&l.id).copied();
            light
        })
        .collect();
    // World to player, as the frame passes it: the inverse turned the sun
    // the wrong way, and sunlit walls went dark at 90 degrees (2026-10-01).
    let sun = space_soup::renderer::lights::sky_sun_light(sky_sun.as_ref(), &lights, yaw_inv);
    lights.extend(sun);
    // THE FLASHLIGHT, when the view holds one: its beam, then its bounce off
    // what the beam lands on, cast against the brushes alone -- the frame
    // casts the physics scene too, which adds the ground and the models.
    let mut lit_surfaces: Vec<space_soup::renderer::lights::LitSurface> = Vec::new();
    if let Some((glass, aim)) = view.flashlight {
        let (lens, rotation) = crate::flashlight::lens_from_bench(glass, aim);
        lights.push(crate::flashlight::beam(lens, rotation * Vec3::NEG_Z, offset, yaw_inv));
        let brushes = BrushGeometry::load(&scene);
        let albedos: Vec<Vec3> =
            load_materials(&game, brushes.materials()).colours.iter().map(crate::brush_render::mean_albedo).collect();
        let land = |o: Vec3, d: Vec3| {
            brushes.cast(o, d, crate::flashlight::RANGE, &[]).map(|b| crate::flashlight_bounce::Landing {
                point: b.point,
                normal: b.normal,
                albedo: albedos.get(b.material as usize).copied().unwrap_or(Vec3::ONE) * b.tint,
            })
        };
        let rays = crate::flashlight_bounce::beam_rays(crate::flashlight::CONE_DEG, crate::flashlight::HOTSPOT_DEG);
        let landed = crate::flashlight_bounce::landings(lens, rotation, &rays, land);
        let patches = landed.patches();
        // The surfaces its pool shows on in reflections, as the frame names
        // them (`flashlight_bounce::HeldSurfaces`, freshly found).
        // `NO_RELIGHT=1`: none, to see what the pool in reflections adds.
        for s in landed.surfaces() {
            eprintln!("LIT SURFACE {s:?}");
            if std::env::var("NO_RELIGHT").as_deref() != Ok("1") {
                lit_surfaces.push(crate::flashlight_bounce::lit_surface(&s, offset, yaw_inv));
            }
        }
        // `NO_BOUNCE=1`: the beam alone, to see what the bounce adds.
        if std::env::var("NO_BOUNCE").as_deref() != Ok("1") {
            for p in &patches {
                eprintln!("BOUNCE patch {p:?}");
            }
            lights.extend(patches.iter().filter_map(|p| crate::flashlight_bounce::light(p, offset, yaw_inv)));
        }
    }
    // THE FIRES' LIGHT, with the effects, as the frame adds it.
    if view.effects {
        let time = std::env::var("EFFECTS_TIME").ok().and_then(|t| t.parse().ok()).unwrap_or(30.0);
        lights.extend(
            crate::scene_effects::load(&game, scene_name)
                .iter()
                .filter_map(|e| space_soup::renderer::effects::fire_light(e, time, offset, yaw_inv)),
        );
    }
    let lights = space_soup::renderer::lights::rank_for_budget(&lights, space_soup::renderer::lights::MAX_LIGHTS);
    let lights_uniform = LightsUniform::new(&device);
    lights_uniform.set_culling(view.light_culling);
    lights_uniform.set_terminator_aa(view.terminator_aa);
    lights_uniform.set_lit_surfaces(&lit_surfaces);
    lights_uniform.upload_frame(&queue, &lights, &[], sun.is_some());

    // Probes, as `set_reflection_probes` binds them.
    // The same description, rooms and doorways the headset builds.
    let level = crate::probe_level::ProbeLevel::load(&game, scene_name);
    let resolution = level.as_ref().map(|l| l.resolution).unwrap_or(1);
    // Under the time of day, each photograph relit: `lamps + (full - lamps) x
    // daylight`, the lamps from the lamps-only bake. A level without one has
    // its photographs scaled whole.
    let lamps_level = tod_daylight.and_then(|_| crate::time_of_day::lamps_probe_level(&game, scene_name));
    if tod_daylight.is_some() {
        eprintln!(
            "offline frame: probes relit {}",
            if lamps_level.is_some() { "from the lamps-only photographs" } else { "WHOLE (no lamps-only photographs)" }
        );
    }
    let relight = |full: Vec<u8>, lamps: Option<Vec<u8>>| -> Vec<u8> {
        match tod_daylight {
            Some(k) => crate::time_of_day::relight_faces(&full, &lamps.unwrap_or_else(|| vec![0u8; full.len()]), k),
            None => full,
        }
    };
    // The photographs as baked and as the lamps alone light them: the meter
    // bins both and relights in f32, as the headset's does -- relit in half
    // floats first, a moonlit photograph (~1e-7) is below their normal range.
    let photographs: Vec<(Vec<u8>, Option<Vec<u8>>)> = level
        .as_ref()
        .map(|l| {
            (0..l.descs.len())
                .filter_map(|i| Some(((l.source())(i)?, lamps_level.as_ref().and_then(|z| (z.source())(i)))))
                .collect()
        })
        .unwrap_or_default();
    let owned: Vec<Vec<u8>> = photographs.iter().map(|(full, lamps)| relight(full.clone(), lamps.clone())).collect();
    // THE SKY REFLECTIONS SEE, as the cube after the probes' -- as the headset
    // builds it (`ProbeStream::new_with_depth`).
    let reflection_sky = if owned.is_empty() {
        None
    } else {
        match &tod {
            Some((_, snap, _)) => Some(
                space_soup::renderer::sky::ReflectionSky { pano: snap.panorama_engine(), rotation_deg: 0.0, intensity: 1.0 }
                    .cube_faces(resolution),
            ),
            None => sky_pano
                .as_ref()
                .map(|(p, r, i)| space_soup::renderer::sky::ReflectionSky::new(p, *r, *i).cube_faces(resolution)),
        }
    };
    let sky_layer = reflection_sky.as_ref().map(|_| owned.len() as u32);
    let mut faces: Vec<&[u8]> = owned.iter().map(|f| f.as_slice()).collect();
    if let Some(sky) = reflection_sky.as_ref() {
        faces.push(sky.as_slice());
    }
    // THE BUILDINGS' OUTSIDES after the sky, as the headset's stream puts them.
    // `NO_BUILDINGS=1` leaves them out, for a before/after.
    let buildings = match (&level, std::env::var("NO_BUILDINGS").as_deref()) {
        (Some(l), Err(_)) | (Some(l), Ok("0")) => l.buildings(&game, scene_name),
        _ => Vec::new(),
    };
    let buildings: Vec<(Vec3, Vec3, Vec<u8>)> = match (tod_daylight, &lamps_level) {
        (Some(_), lamps) => {
            let lamps_daylight = space_soup_engine::daylight::daylight_dir(&game, scene_name);
            let lamps_b = lamps.as_ref().map(|z| z.buildings(&lamps_daylight, scene_name)).unwrap_or_default();
            buildings
                .into_iter()
                .enumerate()
                .map(|(i, (lo, hi, f))| (lo, hi, relight(f, lamps_b.get(i).map(|b| b.2.clone()))))
                .collect()
        }
        _ => buildings,
    };
    let building_layer = (!buildings.is_empty()).then_some(faces.len() as u32);
    for (_, _, f) in &buildings {
        faces.push(f.as_slice());
    }
    let building_boxes: Vec<(Vec3, Vec3)> = buildings.iter().map(|(lo, hi, _)| (*lo, *hi)).collect();
    let probe_view = space_soup::renderer::uniforms::upload_probe_cubes(&device, &queue, resolution, &faces);
    let probe_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });
    // Every probe is in the array here, so a probe's index IS its layer.
    let descs = level.as_ref().map(|l| l.descs.clone()).unwrap_or_default();
    let volumes: Vec<(u32, Vec3, Vec3, Vec3)> =
        descs.iter().enumerate().map(|(i, d)| (i as u32, d.centre, d.min, d.max)).collect();
    let rooms: Vec<u32> = descs.iter().map(|d| d.volume).collect();
    let portals = level.as_ref().map(|l| l.portals.clone()).unwrap_or_default();
    let crate::probe_level::ReflectionProxies { mut proxies, fields, cards, .. } =
        level.as_ref().map(|l| l.scene_proxies(&game, scene_name)).unwrap_or_default();
    // `NO_FIELDS=1`: the models traced by their bounds and the photographs,
    // as before their distance fields -- for a before/after.
    // `NO_PROXIES=1`: no reflection proxies at all -- rooms and photographs only.
    if std::env::var("NO_PROXIES").as_deref() == Ok("1") {
        proxies.clear();
    }
    let fields = if std::env::var("NO_FIELDS").as_deref() == Ok("1") {
        proxies.iter_mut().for_each(|p| p.field = None);
        Vec::new()
    } else {
        fields
    };
    let brightness: Vec<f32> = owned
        .iter()
        .map(|f| space_soup::renderer::uniforms::probe_mean_radiance(f, resolution))
        .collect();

    // THE SUN'S SHADOW FROM THE CAVES, as the headset's static sun map has
    // it: the level's own shadows are baked masks, but a cave is lit live and
    // its roof must shade its floor. `NO_CAVE_SHADOWS=1` leaves it out.
    // Under the time of day the level's own sun shadow comes from the static
    // sun map as well (brushes, ground and caves), at the headset's size.
    let tod_sun = tod.as_ref().and_then(|_| tod_static_sun_matrix(&scene, &lights, offset, yaw_inv));
    let cave_sun = if tod.is_some() { None } else { cave_sun_matrix(&scene, &lights, offset, yaw_inv) };
    let sun_map = tod_sun.or(cave_sun);
    let shadows = ShadowMap::with_dimension(
        &device,
        if tod_sun.is_some() {
            TOD_STATIC_SUN_SIZE
        } else if cave_sun.is_some() {
            2048
        } else {
            64
        },
    );
    let mut uniforms = UniformBuffer::new_with_probes(
        &device,
        &lights_uniform,
        shadows.sun_depth_view(),
        shadows.sun_dynamic_depth_view(),
        shadows.spot_depth_view(),
        shadows.sampler(),
        &probe_view,
        &probe_sampler,
    );

    // The models' distance fields, packed and bound as the headset binds them.
    let field_slots = match space_soup::renderer::proxy_field::atlas(&device, &queue, &fields) {
        Some((atlas, slots)) => {
            uniforms.set_proxy_field_atlas(atlas);
            slots
        }
        None => Vec::new(),
    };
    // And their cards, each proxy naming its row as the headset's
    // `set_reflection_proxies` names it, with the player's rows after them,
    // under whose cards the torch's pool maps are made (`pool_cards`).
    // `NO_CARDS=1`: no model's, for a before/after -- the models then take
    // their mean colour.
    let card_atlas = space_soup::renderer::proxy_cards::atlas_with_characters(
        &device,
        &queue,
        &cards,
        space_soup::renderer::proxy_cards::CHARACTER_CARD_SETS,
    );
    uniforms.set_proxy_card_atlas(card_atlas.view.clone());
    if std::env::var("NO_CARDS").as_deref() != Ok("1") {
        for p in &mut proxies {
            p.cards = p.cards.and_then(|i| card_atlas.rows.get(i as usize).copied().flatten());
        }
    } else {
        proxies.iter_mut().for_each(|p| p.cards = None);
    }
    // The torch's pool maps, as the headset makes them each frame, and the
    // lights sent again to name where. See `pool_cards`.
    let pools = space_soup::renderer::pool_cards::PoolCards::new(&device, &uniforms.layout, card_atlas.resolution);
    let pool_row = Some(pools.first_row(card_atlas.pool_row));
    lights_uniform.set_pool_row(pool_row);
    // THE DOORS, and the lamps whose tiles hold them. See `OfflineDoors`.
    let doors = OfflineDoors::new(
        &device,
        &queue,
        &game,
        &scene,
        (format, view.samples),
        &uniforms.layout,
        (offset, yaw),
        &lights,
        to_player(view.eye),
        &descs,
    );
    lights_uniform.set_tile_lamps(&doors.as_ref().map(OfflineDoors::tile_lamps).unwrap_or_default());
    lights_uniform.upload_frame(&queue, &lights, &[], sun.is_some());
    let pool_mips = space_soup::renderer::brush_pipeline::probe_pass::MirrorMips::new(&device);

    let mut ground_placement = None;
    // THE PROBES' DISTANCES, bound as the headset binds them, so the trace
    // this harness renders is the one the headset runs. Absent from an older
    // bake, and then zero, which the shader answers with the box projection.
    if let Some(l) = level.as_ref() {
        let depth_tex = device.create_texture(&space_soup::renderer::uniforms::probe_depth_descriptor(
            resolution,
            descs.len().max(1) as u32,
        ));
        let depth_of = l.depth_source();
        for i in 0..descs.len() {
            if let Some(d) = depth_of(i) {
                space_soup::renderer::uniforms::write_probe_depth_layer(&queue, &depth_tex, i as u32, resolution, &d);
            }
        }
        uniforms.set_probe_depth(depth_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::CubeArray),
            ..Default::default()
        }));
        if let Some((view, placement)) = ground_map(&device, &queue, &game, scene_name, &scene, &irradiance, sky_sun.as_ref()) {
            uniforms.set_ground_map(view);
            ground_placement = Some(placement);
        }
        uniforms.rebind_probes(
            &device,
            &lights_uniform,
            shadows.sun_depth_view(),
            shadows.sun_dynamic_depth_view(),
            shadows.spot_depth_view(),
            shadows.sampler(),
            &probe_view,
            &probe_sampler,
            Default::default(),
        );
    }

    let eye_p = to_player(view.eye);
    let at_p = to_player(view.at);
    let proj = Mat4::perspective_rh(view.fov_y.to_radians(), view.width as f32 / view.height as f32, 0.03, 1000.0);
    let up = glam::Quat::from_axis_angle((at_p - eye_p).normalize(), view.roll_deg.to_radians()) * Vec3::Y;
    let shift = Mat4::from_translation(Vec3::new(
        2.0 * view.jitter_px[0] / view.width as f32,
        -2.0 * view.jitter_px[1] / view.height as f32,
        0.0,
    ));
    let view_proj = shift * proj * Mat4::look_at_rh(eye_p, at_p, up);
    let mut probes = space_soup::renderer::uniforms::select_resident_probes(&volumes, view.eye, |_, _| true);
    probes.fill_brightness(&brightness);
    probes.fill_volumes(&rooms);
    let resident_rooms = probes.volumes();
    probes.set_portals(&portals, view.eye, &resident_rooms);
    // The doors' proxies turned with their leaves, as the headset turns them.
    let posed = space_soup::renderer::doors::posed_proxies(&proxies, doors.as_ref().map_or(&[][..], |d| &d.views));
    probes.set_proxies(&posed, view.eye, &resident_rooms);
    probes.set_proxy_fields(&field_slots);
    // The outdoors, as the headset carries it every frame.
    probes.set_outdoors(
        space_soup::renderer::probe_stream::outdoor_volume(&descs),
        sky_layer,
        ground_placement,
    );
    probes.set_buildings(&building_boxes, building_layer);
    if view.no_portals {
        probes.portal_count = 0;
    }
    // `NO_TRACE=1`: the reflection's trace switched off, as the `probe_trace`
    // lever does on the headset -- every pixel takes the untraced path, the
    // photographs chosen at the surface and blended through doorways. To put
    // that path on screen everywhere.
    if std::env::var("NO_TRACE").as_deref() == Ok("1") {
        probes.no_trace = true;
    }
    // THE EYE IN THE WATER: which body, if any, and how -- against the moving
    // surface at `WATER_TIME`, as the headset tests it each frame. See
    // `space_soup::renderer::underwater`.
    let water_time: f64 = std::env::var("WATER_TIME").ok().and_then(|t| t.parse().ok()).unwrap_or(30.0);
    let in_water = if view.water && !scene.water.is_empty() {
        use space_soup::renderer::underwater::{eye_water, wave_reach, EyeWater, StillDepth};
        let source = scene.terrain.as_ref().and_then(|def| space_soup_engine::terrain::load(def, &game).ok());
        crate::water_render::build(&scene.water, source.as_deref()).into_iter().enumerate().find_map(|(i, mut b)| {
            b.uniform.set_time(water_time, b.waves.loop_seconds);
            let depth = StillDepth::new(&b.world).at(glam::Vec2::new(view.eye.x, view.eye.z))?;
            let state = eye_water(&b.uniform, depth, wave_reach(b.waves.significant_height(), depth), view.eye);
            (state != EyeWater::Above).then_some((i, state, b.uniform))
        })
    } else {
        None
    };
    // `FILM=<seconds>`: the wet film, that long after surfacing.
    let film_age: Option<f32> = std::env::var("FILM").ok().and_then(|t| t.parse().ok());
    let under_uniform = space_soup::renderer::underwater::UnderUniform::new(&space_soup::renderer::underwater::UnderFrame {
        size: (view.width, view.height),
        fov_y: view.fov_y.to_radians(),
        sun: space_soup::renderer::underwater::sun_from_lights(&lights, Quat::from_rotation_y(yaw)),
        sky_down: Vec3::from(irradiance.evaluate([0.0, 1.0, 0.0])),
        waterline: in_water.is_some_and(|(_, s, _)| s == space_soup::renderer::underwater::EyeWater::Waterline),
        film_age,
        spheres: &[],
    });
    if let Some((i, state, _)) = &in_water {
        eprintln!("offline frame: the eye is {state:?} in body {i}");
    }
    let mut metered: Option<f32> = None;
    let exposure = if view.adapt {
        let (irr, _) = (irradiance, ());
        let loaded: Vec<(&[u8], u32, space_soup::renderer::probe_stream::ProbeDesc)> =
            owned.iter().zip(&descs).map(|(f, d)| (f.as_slice(), resolution, *d)).collect();
        let mut eye = match tod_daylight {
            None => space_soup::renderer::exposure::EyeAdaptation::from_probes(&loaded, irr),
            Some(k) => {
                let mut eye = space_soup::renderer::exposure::EyeAdaptation::sky_only(baked_irradiance);
                for ((full, lamps), d) in photographs.iter().zip(&descs) {
                    let unlit = vec![0u8; if lamps.is_some() { 0 } else { full.len() }];
                    eye.add_probe_layered(full, Some(lamps.as_deref().unwrap_or(&unlit)), resolution, d);
                }
                eye.relight(irr, k);
                eye.set_max_exposure(space_soup::renderer::time_of_day::NIGHT_MAX_EXPOSURE);
                eye
            }
        };
        eye.set_portals(&portals);
        // A burning fire, as the frame meters it.
        let fires = if view.effects {
            space_soup::renderer::effects::meter_samples(&crate::scene_effects::load(&game, scene_name), view.eye, &descs)
        } else {
            Vec::new()
        };
        let mut m = eye.meter_with(view.eye, view.at - view.eye, &fires);
        // Under the water the eye adapts to the water's own light.
        if let Some((_, state, u)) = &in_water {
            use space_soup::renderer::underwater::{eye_share, in_water_luminance, meter_in_water};
            let water = in_water_luminance(u, &under_uniform, u.extinction[3] - view.eye.y);
            m = meter_in_water(m, water, eye_share(*state));
        }
        // The night lets the eye open past the day's ceiling, as the time of
        // day lets it on the headset (`EyeAdaptation::set_max_exposure`).
        let e = match &tod {
            Some(_) => space_soup::renderer::exposure::exposure_for_up_to(m, space_soup::renderer::time_of_day::NIGHT_MAX_EXPOSURE),
            None => space_soup::renderer::exposure::exposure_for(m),
        };
        metered = Some(m);
        eprintln!("offline frame: metered {m:.4}, exposure x{e:.2}");
        e
    } else {
        1.0
    };
    let tod_night_vision = match (&tod, metered) {
        (Some((_, snap, _)), Some(m)) if std::env::var("NIGHT_VISION").as_deref() != Ok("0") => {
            let v = space_soup::renderer::time_of_day::night_vision(m, snap.candelas_per_engine);
            eprintln!("offline frame: adapted to {:.3e} cd/m2, night vision {v:.2}", m * snap.candelas_per_engine);
            v
        }
        _ => 0.0,
    };
    uniforms.upload_scene_with_probes(
        &queue,
        view_proj,
        eye_p,
        &{
            let upload = sun_map.map_or_else(ShadowUpload::disabled, |m| ShadowUpload {
                sun_view_proj: m,
                sun_enabled: true,
                ..ShadowUpload::disabled()
            });
            match &doors {
                Some(d) => d.shadow_upload(upload),
                None => upload,
            }
        },
        &SkyUpload::from(&irradiance),
        &PostUpload {
            exposure,
            tonemap: if view.linear {
                space_soup::renderer::tonemap::ToneMapping::None
            } else {
                Default::default()
            },
            // `TERRAIN_DETAIL=metres` renders with the terrain's normal maps
            // faded out past that distance, as the lever does on the headset.
            terrain_detail_distance: std::env::var("TERRAIN_DETAIL").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0),
            reflection_share: false,
            night_vision: tod_night_vision,
        },
        &PlayerUpload {
            offset,
            yaw,
            capsules: offline_capsules(
                offset,
                yaw_inv,
                view.flashlight.filter(|_| view.torch).map(|(glass, aim)| {
                    let (lens, rotation) = crate::flashlight::lens_from_bench(glass, aim);
                    crate::flashlight::torch_capsules(lens, rotation, offset, yaw_inv)
                }),
            ),
        },
        Some(&probes),
    );

    // THE HALF-RESOLUTION PROBE PASS, as the headset runs it. See
    // `space_soup::renderer::brush_pipeline::probe_pass`.
    let probe_layout = space_soup::renderer::brush_pipeline::probe_pass::bind_group_layout(&device);
    let half_res = view.half_res_reflections && !view.sources;
    let pipeline = if view.sources {
        BrushPipeline::new_multisampled_sources(&device, format, &uniforms.layout, view.samples)
    } else if let (true, Some(cut)) = (half_res, view.cut) {
        BrushPipeline::new_multisampled_probe_reader_with_cut(&device, format, &uniforms.layout, view.samples, &probe_layout, cut)
            .unwrap_or_else(|| panic!("no scene register cut {cut}, or it no longer matches the shader"))
    } else if half_res {
        BrushPipeline::new_multisampled_probe_reader(&device, format, &uniforms.layout, view.samples, &probe_layout, ViewMode::Mono)
    } else {
        BrushPipeline::new_multisampled(&device, format, &uniforms.layout, view.samples)
    };
    let prepass = view
        .depth_prepass
        .then(|| BrushPipeline::new_depth_prepass(&device, format, &uniforms.layout, view.samples, ViewMode::Mono));
    // As it ships: the secondary lookups deferred to `probe_fixup`.
    // `PROBE_INLINE=1` renders with them made in the pass, to compare.
    let inline_lookups = std::env::var("PROBE_INLINE").is_ok_and(|v| v == "1");
    let probe_pass = half_res.then(|| {
        let target =
            space_soup::renderer::brush_pipeline::probe_pass::Target::new(&device, &probe_layout, view.width, view.height, 1);
        if inline_lookups {
            (BrushPipeline::new_probe_pass(&device, &uniforms.layout, ViewMode::Mono), target, None)
        } else {
            let mut fixups = space_soup::renderer::probe_fixup::ProbeFixups::new(&device, &uniforms.layout, target.width * target.height);
            // `FIXUP_CUT=<cut>`: the fix-up run with one of its measurement cuts
            // (`probe_fixup::FIXUP_CUTS`, or `fixup_inlined`), as the `fixup_cut`
            // lever runs it on the headset.
            if let Ok(cut) = std::env::var("FIXUP_CUT") {
                assert!(fixups.set_cut(&device, Some(&cut)), "no fix-up cut {cut}, or it no longer matches the shader");
            }
            // `PASS_CUT=<cut>`: the pass drawn with one of its measurement cuts
            // (`brush_pipeline::DEFERRED_REGISTER_CUTS`), as the `pass_cut`
            // lever draws it on the headset.
            // With no lit surface, the poolless twin, as the headset draws it
            // (`lights::without_pool_maps`); `POOLLESS=0` draws the full pass.
            let poolless = std::env::var("POOLLESS").as_deref() != Ok("0") && !lights_uniform.reads_pool_maps();
            let pipeline = match std::env::var("PASS_CUT") {
                Ok(cut) => BrushPipeline::new_probe_pass_deferred_with_cut(&device, &uniforms.layout, &fixups, &cut)
                    .unwrap_or_else(|| panic!("no probe pass cut {cut}, or it no longer matches the shader")),
                Err(_) if poolless => BrushPipeline::new_probe_pass_deferred_poolless(&device, &uniforms.layout, &fixups),
                Err(_) => BrushPipeline::new_probe_pass_deferred(&device, &uniforms.layout, &fixups),
            };
            let target_bg = fixups.target_bind_group(&device, &target);
            let pass_bg = fixups.pass_bind_group_for(&device, &target);
            (pipeline, target, Some((fixups, (target_bg, pass_bg))))
        }
    });

    let mut geometry = BrushGeometry::load_with(&scene, true);
    let maps = load_materials(&game, geometry.materials());
    let colours: Vec<Option<_>> = maps.colours.into_iter().map(Some).collect();
    let materials = BrushMaterials::new(
        &device, &queue, &pipeline.material_layout, &colours, &maps.normals, &maps.roughs, &maps.aos,
    );

    let lm = space_soup_engine::lightmaps::load_scene_lightmaps(&game, scene_name);
    let pick = |id: &str| lm.iter().find(|m| m.object_id == id);
    let stationary_ids: Vec<String> = (0..space_soup_engine::stationary::MAX_STATIONARY_LAYERS)
        .map(space_soup_engine::lightmaps::scene_brush_stationary_id)
        .collect();
    let base = lm.iter().find(|m| {
        m.target == space_soup_engine::lightmaps::LightmapTarget::Brush
            && m.object_id != space_soup_engine::lightmaps::SCENE_BRUSH_DIRECTION_ID
            && m.object_id != space_soup_engine::lightmaps::SCENE_BRUSH_SUN_MASK_ID
            && !stationary_ids.contains(&m.object_id)
    })?;
    let stationary: Vec<&space_soup_engine::lightmaps::LoadedLightmap> = crate::scene_lights::usable_stationary_masks(
        stationary_ids.iter().map_while(|id| pick(id)).collect(),
        &crate::scene_lights::stationary_channels(&game, scene_name),
    );
    let dir = pick(space_soup_engine::lightmaps::SCENE_BRUSH_DIRECTION_ID);
    // Filled round its charts as the headset fills it. See
    // `brush_pipeline::dilate_sun_mask`.
    let sun_mask = pick(space_soup_engine::lightmaps::SCENE_BRUSH_SUN_MASK_ID)
        .map(|d| (space_soup::renderer::brush_pipeline::dilate_sun_mask(&d.rgba, d.width, d.height), d.width, d.height));
    let stationary_layers: Vec<&[u8]> = stationary.iter().map(|m| m.rgba.as_slice()).collect();
    // Under the time of day, the atlas the hour's light makes of its layers
    // (`TimeOfDayRuntime`'s worker builds the same), and no baked sun mask.
    let tod_atlas: Option<Vec<f32>> = tod_weights.and_then(|w| {
        let layers =
            crate::time_of_day::daylight_layers(&game, scene_name, base.linear.as_deref().map(|l| (l, base.width, base.height)));
        eprintln!(
            "offline frame: brush atlas {}",
            if layers.is_some() { "from its daylight layers" } else { "AS SHIPPED (no daylight layers that add back to it)" }
        );
        layers.map(|l| l.combine(w.sky, w.sun))
    });
    let lightmap = space_soup::renderer::mesh::create_lightmap_texture_full(
        &device,
        &queue,
        &pipeline.lightmap_layout,
        match (&tod_atlas, &base.linear) {
            (Some(l), _) | (None, Some(l)) => space_soup::renderer::mesh::LightmapLight::Linear(l),
            (None, None) => space_soup::renderer::mesh::LightmapLight::Srgb8(&base.rgba),
        },
        base.width,
        base.height,
        dir.map(|d| (d.rgba.as_slice(), d.width, d.height)),
        sun_mask.as_ref().filter(|_| tod.is_none()).map(|(rgba, w, h)| (rgba.as_slice(), *w, *h)),
        stationary.first().map(|m| (stationary_layers.as_slice(), m.width, m.height)),
    );

    let (verts, idx) = geometry.assemble(&[], offset, yaw_inv, 0.0)?;
    use wgpu::util::DeviceExt;
    let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("offline_brush_vb"),
        contents: bytemuck_cast(verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("offline_brush_ib"),
        contents: bytemuck_cast(idx),
        usage: wgpu::BufferUsages::INDEX,
    });
    // THE WEATHER, when the view asks for it and the level has some: its
    // areas' ground through the physics scene, as the headset finds it, at
    // `WEATHER_TIME`.
    let weather_layout = space_soup::renderer::weather::bind_group_layout(&device);
    let mut weather = match (view.weather, scene.weather.is_empty()) {
        (true, false) => {
            let mut physics = space_soup_engine::rigid_physics::PhysicsWorld::new();
            physics.rebuild(&scene, &game);
            let mut field = crate::weather_render::field(&scene, &physics);
            let time: f64 = std::env::var("WEATHER_TIME").ok().and_then(|t| t.parse().ok()).unwrap_or(600.0);
            field.set_time(time);
            for (d, _, st) in &field.areas {
                eprintln!("offline frame: weather {:?} at {time} s: {st:?}", d.name);
            }
            let mut maps = crate::weather_render::maps(&device, &weather_layout, &field);
            crate::weather_render::upload(&queue, &mut maps, &field);
            Some((field, maps, time))
        }
        _ => None,
    };
    // THE GROUND, when the view asks for it. Its reader's twin as the frame
    // would choose it: no spot casts here, so spotless unless a lit surface's
    // light is in the list.
    let terrain = match (&probe_pass, view.terrain) {
        (Some((_, _, Some((fixups, _)))), true) => offline_terrain(
            &device,
            &queue,
            &game,
            scene_name,
            &lm,
            (offset, yaw_inv, yaw),
            (format, view.samples),
            (&uniforms.layout, &probe_layout, fixups),
            (
                !lights.iter().any(Light::is_surface_light),
                std::env::var("POOLLESS").as_deref() != Ok("0") && !lights_uniform.reads_pool_maps(),
            ),
            weather.as_ref().map(|_| &weather_layout),
        ),
        (_, true) => {
            eprintln!("offline frame: the ground is drawn only with half-resolution reflections and deferred lookups");
            None
        }
        _ => None,
    };
    // THE CAVES: models baked with layered shading (the editor's caves),
    // drawn with the ground's materials as the headset draws them. Without
    // them a cave's mouth showed the sky through the terrain's holes, and
    // nothing inside a cave could be looked at offline (2026-10-07).
    // `NO_CAVES=1` leaves them out.
    let caves = match (&terrain, std::env::var("NO_CAVES").as_deref()) {
        (Some(t), Err(_)) | (Some(t), Ok("0")) => offline_caves(
            &device,
            &queue,
            &game,
            scene_name,
            (format, view.samples),
            (&uniforms.layout, &t.reader.material_layout),
            (offset, yaw_inv),
        ),
        _ => None,
    };

    // THE FALLING RAIN AND SNOW, lit and placed as the frame does it.
    let weather_particles = match (&mut weather, &terrain) {
        (Some((field, maps, time)), Some(_)) => {
            let areas = crate::weather_render::area_params(field);
            let counts = space_soup::renderer::weather::particle_counts(&areas, view.eye.to_array());
            let (sky, sun) = crate::weather_render::particle_light(&irradiance, sky_sun.as_ref());
            let forward = (at_p - eye_p).normalize();
            let right = forward.cross(up).normalize();
            let ground = crate::load_scene_terrain(&game, scene_name).map(|(g, _)| {
                let (lo, hi) = g.world_bounds();
                ([lo.x, lo.z], [hi.x, hi.z])
            });
            let torch = view.flashlight.map(|(glass, aim)| {
                let (lens, rotation) = crate::flashlight::lens_from_bench(glass, aim);
                let beam = crate::flashlight::beam(lens, rotation * Vec3::NEG_Z, offset, yaw_inv);
                let (outer, inner) = beam.cone_cosines();
                let c = beam.color.to_linear();
                let i = beam.intensity;
                (beam.position.to_array(), beam.direction.to_array(), beam.range, outer, inner, [c[0] * i, c[1] * i, c[2] * i])
            });
            maps.set_view(
                &space_soup::renderer::weather::ParticleView {
                    head_world: view.eye.to_array(),
                    right: right.to_array(),
                    up: right.cross(forward).to_array(),
                    frame: [offset.x, offset.y, offset.z, yaw],
                    sky,
                    sun,
                    torch,
                    exposure,
                    pixel: 2.0 * (0.5 * view.fov_y.to_radians()).tan() / view.height as f32,
                    time: *time,
                    ground,
                },
                counts,
                crate::weather_render::wind_at(field, view.eye),
            );
            maps.write(&queue);
            eprintln!("offline frame: weather particles {counts:?}");
            let pipeline = space_soup::renderer::weather::WeatherPipeline::new(
                &device,
                format,
                &uniforms.layout,
                &space_soup::renderer::terrain_pipeline::material_bind_group_layout(&device),
                &weather_layout,
                view.samples,
                space_soup::renderer::multiview::ViewMode::Mono,
            );
            Some((pipeline, counts))
        }
        _ => None,
    };
    // A SPLASH to look at: `SPLASH=x,y,z,speed,size,age[,depth]` (world;
    // seconds since it struck at the effects' and the water's clock; the
    // still water's depth there), in full sun, lifted by the swash as
    // `XrRenderer::add_splash` lifts it.
    let swash_uniforms: Vec<_> = if std::env::var("SPLASH").is_ok() {
        let source = scene.terrain.as_ref().and_then(|def| space_soup_engine::terrain::load(def, &game).ok());
        crate::water_render::build(&scene.water, source.as_deref()).into_iter().map(|b| (b.uniform, b.waves.loop_seconds)).collect()
    } else {
        Vec::new()
    };
    let splash_at = |time: f64| -> Vec<space_soup::renderer::effects::Splash> {
        let Ok(v) = std::env::var("SPLASH") else { return Vec::new() };
        let n: Vec<f32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        if n.len() < 6 {
            eprintln!("offline frame: SPLASH wants x,y,z,speed,size,age; got {v}");
            return Vec::new();
        }
        let depth = n.get(6).copied().unwrap_or(0.0);
        let mut position = Vec3::new(n[0], n[1], n[2]);
        let born = time - n[5] as f64;
        if let Some((mut u, loop_seconds)) = swash_uniforms.iter().copied().find(|(u, _)| (u.extinction[3] - position.y).abs() < 0.05) {
            u.set_time(born, loop_seconds);
            position.y += space_soup::renderer::water_pipeline::swash_lift(&u, depth, glam::Vec2::new(position.x, position.z));
        }
        vec![space_soup::renderer::effects::Splash { position, born, speed: n[3], size: n[4], sunlit: 1.0, depth, seed: 7 }]
    };
    // THE EFFECTS, simulated, lit and uploaded as the headset's frame does it.
    let effects = {
        use space_soup::renderer::effects;
        let mut emitters = if view.effects { crate::scene_effects::load(&game, scene_name) } else { Vec::new() };
        // Their ceilings by the same ray the headset casts through its physics.
        if !emitters.is_empty() {
            let mut physics = space_soup_engine::rigid_physics::PhysicsWorld::new();
            physics.rebuild(&scene, &game);
            crate::scene_effects::find_ceilings(&mut emitters, |p| physics.raycast(p, Vec3::Y, 40.0).map(|(hit, _)| hit.y));
        }
        let effects_time: f64 = std::env::var("EFFECTS_TIME").ok().and_then(|t| t.parse().ok()).unwrap_or(30.0);
        let splashes = splash_at(effects_time);
        match (&probe_pass, emitters.is_empty() && splashes.is_empty()) {
            (Some(_), false) => {
                let layout = effects::bind_group_layout(&device);
                let mut gpu = effects::EffectsGpu::new(&device, &queue, &layout);
                let pipeline =
                    effects::EffectsPipeline::new_multisampled(&device, format, &uniforms.layout, &probe_layout, &layout, view.samples);
                let time = std::env::var("EFFECTS_TIME").ok().and_then(|t| t.parse().ok()).unwrap_or(30.0);
                let turn = Quat::from_rotation_y(yaw);
                let reaches =
                    |e: &effects::EffectEmitter, i: usize| effects::lamp_in_room(&descs, e.position, turn * lights[i].position + offset);
                let ambient = |w: Vec3| effects::room_ambient(&descs, w);
                let at = effects::Surroundings {
                    head: eye_p,
                    offset,
                    yaw_inv,
                    lights: &lights,
                    reaches: &reaches,
                    ambient: &ambient,
                    exposure,
                };
                let frame = effects::simulate_with(&emitters, &splashes, time, &at);
                eprintln!("offline frame: effects at {time} s: {} over, {} screened", frame.over, frame.screened);
                gpu.upload(&device, &queue, &frame);
                let forward = (at_p - eye_p).normalize();
                let right = forward.cross(up).normalize();
                gpu.set_view(
                    &queue,
                    &effects::EffectsUniform {
                        right: right.extend(0.0).to_array(),
                        up: right.cross(forward).extend(0.0).to_array(),
                        head: eye_p.extend(1.0).to_array(),
                        depth: [0.03, 1000.0, 1.0, 2.0 * (0.5 * view.fov_y.to_radians()).tan() / view.height as f32],
                    },
                );
                Some((gpu, pipeline))
            }
            (None, false) => {
                eprintln!("offline frame: the effects are drawn only with the half-resolution probe pass");
                None
            }
            _ => None,
        }
    };

    // THE WATER AND THE SKY, when the view asks for them: built as the
    // headset builds them on load (`water_render`, `XrRenderer::set_water`).
    let water = match (view.water, scene.water.is_empty()) {
        (true, false) => {
            let source = scene.terrain.as_ref().and_then(|def| space_soup_engine::terrain::load(def, &game).ok());
            let pipeline = space_soup::renderer::water_pipeline::WaterPipeline::new(&device, format, &uniforms.layout, view.samples);
            let time: f64 = std::env::var("WATER_TIME").ok().and_then(|t| t.parse().ok()).unwrap_or(30.0);
            let bodies: Vec<_> = crate::water_render::build(&scene.water, source.as_deref())
                .into_iter()
                .map(|mut b| {
                    let waves = space_soup::renderer::water_waves::WaveField::new(&device, &queue, b.waves);
                    b.uniform.set_time(time, b.waves.loop_seconds);
                    b.uniform.rings = space_soup::renderer::effects::splash_rings(&splash_at(time), time, view.eye);
                    let buffer = |label, contents: &[u8], usage| {
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage })
                    };
                    let ub = buffer("offline_water_uniform", bytemuck_cast(std::slice::from_ref(&b.uniform)), wgpu::BufferUsages::UNIFORM);
                    let vb = buffer("offline_water_vb", bytemuck_cast(&b.world), wgpu::BufferUsages::VERTEX);
                    let ib = buffer("offline_water_ib", bytemuck_cast(&b.indices), wgpu::BufferUsages::INDEX);
                    let groups = pipeline.bind_groups(&device, &ub, &waves);
                    (waves, groups, vb, ib, b.indices.len() as u32, ub)
                })
                .collect();
            eprintln!(
                "offline frame: {} body/bodies of water at {time} s, {} blending",
                bodies.len(),
                if pipeline.dual_source { "dual-source" } else { "one-alpha" }
            );
            Some((pipeline, bodies, time))
        }
        _ => None,
    };
    // THE VIEW FROM UNDER THE WATER, when the eye is in it: the veil (with
    // the probe pass's depth), the underside, the film.
    let underwater = match (&in_water, &water, &probe_pass) {
        (Some((i, state, _)), Some((_, bodies, _)), Some(_)) => {
            let pipes = space_soup::renderer::underwater::UnderwaterPipelines::new(&device, format, &uniforms.layout, &probe_layout, view.samples);
            let (waves, _, _, _, _, ub) = &bodies[*i];
            let groups = pipes.water_groups(&device, ub, waves);
            let (buffer, group) = pipes.under_buffer(&device);
            queue.write_buffer(&buffer, 0, bytemuck_cast(std::slice::from_ref(&under_uniform)));
            Some((pipes, groups, buffer, group, *i, *state))
        }
        (Some(_), Some(_), None) => {
            eprintln!("offline frame: the view from under the water needs the half-resolution probe pass");
            None
        }
        _ => None,
    };
    let film = match (film_age, &water, &probe_pass) {
        (Some(_), Some((_, bodies, _)), Some(_)) if underwater.is_none() && !bodies.is_empty() => {
            let pipes = space_soup::renderer::underwater::UnderwaterPipelines::new(&device, format, &uniforms.layout, &probe_layout, view.samples);
            let (waves, _, _, _, _, ub) = &bodies[0];
            let groups = pipes.water_groups(&device, ub, waves);
            let (buffer, group) = pipes.under_buffer(&device);
            queue.write_buffer(&buffer, 0, bytemuck_cast(std::slice::from_ref(&under_uniform)));
            Some((pipes, groups, buffer, group))
        }
        _ => None,
    };
    let sky = view.water.then(|| {
        use space_soup::renderer::sky::{Sky, SkyPipeline, AMBIENT};
        let pipeline = SkyPipeline::new(&device, format, &uniforms.layout, view.samples);
        let mut sky = match &sky_pano {
            Some((p, r, i)) => Sky::new(&device, &queue, &pipeline.layout, p, *r, *i),
            None => Sky::none(&device, &queue, &pipeline.layout, AMBIENT),
        };
        if let Some((_, snap, zenith)) = &tod {
            sky.apply_snapshot(&device, &queue, &pipeline.layout, snap, 0.0, *zenith);
        }
        (pipeline, sky)
    });

    let size = wgpu::Extent3d { width: view.width, height: view.height, depth_or_array_layers: 1 };
    let tex = |label, samples, format, usage| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let msaa = tex("offline_msaa", view.samples, format, wgpu::TextureUsages::RENDER_ATTACHMENT);
    let resolved = tex("offline_resolved", 1, format, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
    let depth = tex("offline_depth", view.samples, wgpu::TextureFormat::Depth32Float, wgpu::TextureUsages::RENDER_ATTACHMENT);
    let (msaa_v, resolved_v, depth_v) = (
        msaa.create_view(&Default::default()),
        resolved.create_view(&Default::default()),
        depth.create_view(&Default::default()),
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    if let Some(d) = &doors {
        d.record_shadows(&queue, &shadows, &mut encoder);
    }
    if let Some(m) = sun_map {
        use space_soup::renderer::shadow::ShadowKind;
        let casters: Vec<space_soup::renderer::shadow::ShadowMeshDraw> = caves
            .as_ref()
            .map(|(_, draws)| draws.iter().map(|(_, _, ib, count, (vb, model))| (vb, ib, *count, &model.bind_group)).collect())
            .unwrap_or_default();
        // The level's own casters only for the time of day's map: the caves'
        // map is theirs alone, as before.
        let level = tod_sun.is_some();
        let solid = terrain.as_ref().filter(|_| level).map(|t| (&t.vb, &t.ib, t.count));
        let brushes = level.then_some((&vb, &ib, idx.len() as u32));
        shadows.upload_light(&queue, ShadowKind::Sun, m);
        let drawn = shadows.record(&mut encoder, ShadowKind::Sun, solid, brushes, &casters, &[], &[], m);
        if level {
            eprintln!("offline frame: static sun map {TOD_STATIC_SUN_SIZE}^2 recorded ({drawn} indices)");
        }
    }
    if let Some((_, bodies, time)) = &water {
        for (waves, ..) in bodies {
            waves.update(&queue, &mut encoder, *time);
        }
    }
    // See `FIXUP_STATS` at the fix-up's dispatch.
    let mut fixup_census: Option<(wgpu::Buffer, (u32, u32))> = None;
    if let (Some(_), Some(row)) = (&probe_pass, pool_row) {
        if lights_uniform.reads_pool_maps() {
            pools.record(&device, &mut encoder, &uniforms.bind_group, &pool_mips, &card_atlas.texture, row, None);
        }
    }
    if let Some((probe_pipeline, target, fixups)) = &probe_pass {
        if let Some((fixups, _)) = fixups {
            fixups.clear(&mut encoder);
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("offline_probe_pass"),
            // The reflection, and how far it reached (for SpaceWarp), as the
            // headset's pass has them. See `probe_pass::targets`.
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &target.color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &target.reach_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &target.depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_pipeline(&probe_pipeline.pipeline);
        pass.set_bind_group(0, &uniforms.bind_group, &[]);
        pass.set_bind_group(1, &materials.bind_group, &[]);
        pass.set_bind_group(2, &lightmap.bind_group, &[]);
        if let Some((_, (_, pass_bg))) = fixups {
            pass.set_bind_group(3, pass_bg, &[]);
        }
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..idx.len() as u32, 0, 0..1);
        // The ground's reflection after the brushes', as the headset's pass
        // draws it.
        if let (Some(t), Some((_, (_, pass_bg)))) = (&terrain, fixups) {
            pass.set_pipeline(&t.pass.pipeline);
            pass.set_bind_group(0, &uniforms.bind_group, &[]);
            pass.set_bind_group(1, &t.material.bind_group, &[]);
            if let Some((_, maps, _)) = &weather {
                pass.set_bind_group(2, &maps.bind_group, &[]);
            }
            pass.set_bind_group(3, pass_bg, &[]);
            pass.set_vertex_buffer(0, t.vb.slice(..));
            pass.set_index_buffer(t.ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..t.count, 0, 0..1);
        }
        drop(pass);
        if let Some((fixups, (target_bg, _))) = fixups {
            fixups.dispatch(&mut encoder, &uniforms.bind_group, target_bg, None);
            // `FIXUP_STATS=1`: a census of the records the pass made, read
            // back below -- what the fix-up is asked to do in this view.
            if std::env::var("FIXUP_STATS").as_deref() == Ok("1") {
                let list = fixups.list_buffer();
                let copy = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("offline_fixup_census"),
                    size: list.size(),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                encoder.copy_buffer_to_buffer(list, 0, &copy, 0, list.size());
                fixup_census = Some((copy, (target.width, target.height)));
            }
        }
    }
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("offline_brush_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: if view.samples > 1 { &msaa_v } else { &resolved_v },
                depth_slice: None,
                resolve_target: if view.samples > 1 { Some(&resolved_v) } else { None },
                ops: wgpu::Operations {
                    // Magenta: anything not covered by a brush is a hole.
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 1.0, g: 0.0, b: 1.0, a: 1.0 }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_v,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        // The brushes' depth first, as the headset draws it.
        if let Some(prepass) = &prepass {
            pass.set_pipeline(&prepass.pipeline);
            pass.set_bind_group(0, &uniforms.bind_group, &[]);
            pass.set_bind_group(1, &materials.bind_group, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..idx.len() as u32, 0, 0..1);
        }
        // The ground between the brushes' depth and their shading, as the
        // headset's scene pass draws it.
        if let (Some(t), Some((_, target, _))) = (&terrain, &probe_pass) {
            pass.set_pipeline(&t.reader.pipeline);
            pass.set_bind_group(0, &uniforms.bind_group, &[]);
            pass.set_bind_group(1, &t.material.bind_group, &[]);
            if let Some((_, maps, _)) = &weather {
                pass.set_bind_group(2, &maps.bind_group, &[]);
            }
            pass.set_bind_group(3, &target.bind_group, &[]);
            pass.set_vertex_buffer(0, t.vb.slice(..));
            pass.set_index_buffer(t.ib.slice(..), wgpu::IndexFormat::Uint32);
            match &t.gentle {
                Some((gentle, g, steep)) => {
                    let st = steep.as_ref().map_or(0, |s| s.1);
                    pass.draw_indexed(*g..t.count - st, 0, 0..1);
                    pass.set_pipeline(&gentle.pipeline);
                    pass.draw_indexed(0..*g, 0, 0..1);
                    if let Some((steep, st)) = steep {
                        pass.set_pipeline(&steep.pipeline);
                        pass.draw_indexed(t.count - st..t.count, 0, 0..1);
                    }
                }
                None => pass.draw_indexed(0..t.count, 0, 0..1),
            }
        }
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, &uniforms.bind_group, &[]);
        pass.set_bind_group(1, &materials.bind_group, &[]);
        pass.set_bind_group(2, &lightmap.bind_group, &[]);
        if let Some((_, target, _)) = &probe_pass {
            pass.set_bind_group(3, &target.bind_group, &[]);
        }
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..idx.len() as u32, 0, 0..1);
        // The doors after the brushes, as the headset draws its models.
        if let Some(d) = &doors {
            d.draw(&mut pass, &uniforms.bind_group);
        }
        // The caves after the brushes, with the ground's materials, as the
        // headset's scene pass draws its layered batch.
        if let (Some((pipeline, draws)), Some(t)) = (&caves, &terrain) {
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &uniforms.bind_group, &[]);
            pass.set_bind_group(1, &t.material.bind_group, &[]);
            for (model, vb, ib, count, _) in draws {
                pass.set_bind_group(2, &model.bind_group, &[]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..*count, 0, 0..1);
            }
        }
        // UNDER THE WATER: the veil over everything opaque, then the body's
        // underside -- before its own surface, which keeps only what is seen
        // from above, along a waterline. See `space_soup::renderer::underwater`.
        let mut under_body = None;
        if let (Some((pipes, groups, _, group, i, state)), Some((_, bodies, _)), Some((_, target, _))) = (&underwater, &water, &probe_pass) {
            let (waves, _, vb, ib, count, _) = &bodies[*i];
            let water_group = &groups[waves.current()];
            // `UW_SKIP=veil|underside`: one of them left out, to see what each draws.
            let skip = std::env::var("UW_SKIP").unwrap_or_default();
            if skip != "veil" {
                pipes.draw_veil(&mut pass, *state, [&uniforms.bind_group, water_group, &target.bind_group, group]);
            }
            if skip != "underside" {
                pipes.draw_underside(&mut pass, [&uniforms.bind_group, water_group, group], vb, ib, *count);
            }
            under_body = Some((*i, *state));
        }
        // The water after everything opaque, then the sky where nothing was
        // drawn, as the headset's scene pass has them.
        if let Some((pipeline, bodies, _)) = &water {
            pass.set_pipeline(if std::env::var("SPLASH").is_ok() { &pipeline.pipeline } else { &pipeline.ringless });
            pass.set_bind_group(0, &uniforms.bind_group, &[]);
            for (b, (waves, groups, vb, ib, count, _)) in bodies.iter().enumerate() {
                if under_body == Some((b, space_soup::renderer::underwater::EyeWater::Under)) {
                    continue;
                }
                // Along a waterline, the twin that leaves the view under the
                // line to the underwater one.
                let line = under_body == Some((b, space_soup::renderer::underwater::EyeWater::Waterline));
                // `WATER_CUT=<cut>`: the ringless water with one of its
                // measurement cuts (`water_pipeline::WATER_CUTS`).
                let cut = std::env::var("WATER_CUT").ok().map(|c| {
                    pipeline.with_cut(&device, &c).unwrap_or_else(|| panic!("no water cut {c}, or it no longer matches"))
                });
                pass.set_pipeline(if line {
                    &pipeline.waterline
                } else if let Some(cut) = cut.as_ref() {
                    cut
                } else if std::env::var("SPLASH").is_ok() {
                    &pipeline.pipeline
                } else {
                    &pipeline.ringless
                });
                pass.set_bind_group(1, &groups[waves.current()], &[]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..*count, 0, 0..1);
            }
        }
        let wholly_under = under_body.is_some_and(|(_, s)| s == space_soup::renderer::underwater::EyeWater::Under);
        if let (Some((pipeline, sky)), false) = (&sky, wholly_under) {
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &uniforms.bind_group, &[]);
            pass.set_bind_group(1, &sky.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        // The effects after everything opaque, as the headset draws them.
        if let (Some((gpu, pipeline)), Some((_, target, _))) = (&effects, &probe_pass) {
            gpu.draw(&mut pass, pipeline, &uniforms.bind_group, &target.bind_group);
        }
        // The falling rain and snow last, lit by the ground's baked sky and sun.
        if let (Some((pipeline, counts)), Some((_, maps, _)), Some(t)) = (&weather_particles, &weather, &terrain) {
            pipeline.draw(&mut pass, &uniforms.bind_group, &t.material.bind_group, &maps.bind_group, *counts);
            // And the areas' columns seen from afar, ended by the probe pass's depth.
            if let Some((_, target, _)) = &probe_pass {
                pipeline.draw_veils(&mut pass, &uniforms.bind_group, &t.material.bind_group, &maps.bind_group, &target.bind_group);
            }
        }
        // The wet film after surfacing, over everything.
        if let (Some((pipes, groups, _, group)), Some((_, target, _)), Some((_, bodies, _))) = (&film, &probe_pass, &water) {
            pipes.draw_film(&mut pass, [&uniforms.bind_group, &groups[bodies[0].0.current()], &target.bind_group, group]);
        }
    }
    let row = (view.width * 4).div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("offline_readback"),
        size: (row * view.height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        resolved.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(view.height) },
        },
        size,
    );
    queue.submit(Some(encoder.finish()));
    if let Some((copy, (tw, th))) = &fixup_census {
        let slice = copy.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let data = slice.get_mapped_range().expect("map the fix-up census");
        let word = |o: usize| u32::from_le_bytes(data[o..o + 4].try_into().unwrap());
        let capacity = ((data.len() - 16) / 128) as u32;
        let n = word(0).min(capacity) as usize;
        let (mut subsample, mut retest, mut edge, mut rim, mut mirror, mut recolour) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        // The rims by kind -- through the opening (a point already known),
        // into the wall beside it (a second trace), at the opening's far end
        // -- and by the surface's roughness: under 0.2, 0.45, 0.7, and above.
        let (mut rim_through, mut rim_far) = (0usize, 0usize);
        let mut rim_rough = [0usize; 4];
        for k in 0..n {
            let b = 16 + k * 128;
            let f = |i: usize| f32::from_bits(word(b + 96 + i * 4));
            let (rim_at, edge_cover) = (f(0), f(2));
            let edge_code = word(b + 116) as i32;
            if rim_at >= 0.0 {
                let rim_code = word(b + 112) as i32;
                rim_through += (rim_code & 1 != 0) as usize;
                rim_far += (rim_code & 8192 != 0) as usize;
                let roughness = f32::from_bits(word(b + 60));
                rim_rough[[0.2f32, 0.45, 0.7].iter().filter(|&&r| roughness >= r).count()] += 1;
            }
            // `PROBE_RETEST` (-4 less the cover) or `PROBE_SUBSAMPLE` (-cover).
            if edge_code >= 0 && edge_cover < -2.0 {
                retest += 1;
            } else if edge_code >= 0 && edge_cover < 0.0 {
                subsample += 1;
            } else if edge_code >= 0 && edge_cover < 0.99 {
                edge += 1;
            }
            rim += (rim_at >= 0.0) as usize;
            mirror += (f(7) < 0.0) as usize;
            // On a model's cards and marked for nothing else: recorded only so
            // the fix-up colours it (`lights::probe_hit_carded`).
            recolour += (word(b + 12) != 0 && !(edge_code >= 0 && edge_cover < 0.0)) as usize;
        }
        let texels = (tw * th).max(1) as f32;
        eprintln!(
            "fixups: {n} records ({:.2}% of the {tw}x{th} pass): outline subsamples {subsample}, card retests {retest}, card recolours {recolour}, edges {edge}, rims {rim}, floor mirror {mirror}",
            100.0 * n as f32 / texels
        );
        eprintln!(
            "fixup rims: through {rim_through}, into the wall {}, far end {rim_far}; roughness <0.2 {}, <0.45 {}, <0.7 {}, >=0.7 {}",
            rim - rim_through,
            rim_rough[0],
            rim_rough[1],
            rim_rough[2],
            rim_rough[3]
        );
    }
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    let data = slice.get_mapped_range().expect("map the readback");
    let mut rgba = Vec::with_capacity((view.width * view.height * 4) as usize);
    for y in 0..view.height {
        let start = (y * row) as usize;
        rgba.extend_from_slice(&data[start..start + (view.width * 4) as usize]);
    }
    Some(Shot { width: view.width, height: view.height, rgba })
}

/// The ground as [`render_brushes`] draws it. See [`offline_terrain`].
/// The layered pipeline and one draw per layered primitive: its model
/// uniform, layered vertex and index buffers, index count, and the ordinary
/// vertex buffer and model uniform its shadow is drawn with.
type OfflineCaves = (
    space_soup::renderer::layered_mesh_pipeline::LayeredMeshPipeline,
    Vec<(
        space_soup::renderer::mesh_pipeline::ModelUniform,
        wgpu::Buffer,
        wgpu::Buffer,
        u32,
        (wgpu::Buffer, space_soup::renderer::mesh_pipeline::ModelUniform),
    )>,
);

/// The sun's matrix over the scene's caves (models that are part of the
/// ground: a mesh with a `terrain_collider`), in the player's frame, as the
/// headset fits its sun map round what it shades. `None` without a sun or a
/// cave, or with `NO_CAVE_SHADOWS=1`.
/// THE DOORS, as the headset draws them: each leaf a model at its angle, lit
/// by its room's baked light and the lamps, drawn into the moving casters'
/// tiles of the lamps reaching it (`shadow::moving_caster_tiles`), whose
/// lights name those tiles. `DOORS=a,b,..` sets the leaves' angles in
/// degrees, in the scene's order (the last repeats); `DOORS=none` draws no
/// door and casts nothing -- the level as it was before them. Unset, they
/// hang where the scene puts them. `NO_DOOR_SHADOWS=1` draws them without
/// their tiles.
///
/// The torch casts no shadow here (`View::flashlight`); a door's shadow from
/// it comes from a tile like a lamp's, where the headset gives the torch its
/// spot slot.
struct OfflineDoors {
    pipeline: space_soup::renderer::mesh_pipeline::MeshPipeline,
    leaves: Vec<(space_soup::renderer::mesh::GltfMesh, space_soup::renderer::mesh_pipeline::ModelUniform)>,
    lightmap: space_soup::renderer::mesh::LoadedTexture,
    tiles: Vec<(usize, Mat4)>,
    /// The leaves as the renderer takes them, for their reflection proxies.
    views: Vec<space_soup::renderer::doors::DoorView>,
}

impl OfflineDoors {
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        game: &std::path::Path,
        scene: &space_soup_engine::scene::Scene,
        (format, samples): (wgpu::TextureFormat, u32),
        uniform_layout: &wgpu::BindGroupLayout,
        (offset, yaw): (Vec3, f32),
        lights: &[Light],
        eye: Vec3,
        descs: &[space_soup::renderer::probe_stream::ProbeDesc],
    ) -> Option<Self> {
        let spec = std::env::var("DOORS").ok();
        if spec.as_deref() == Some("none") {
            return None;
        }
        let mut doors = crate::client_doors::ClientDoors::from_scene(scene);
        if doors.doors.is_empty() {
            return None;
        }
        if let Some(spec) = spec {
            let mut angles: Vec<f32> = spec.split(',').filter_map(|a| a.trim().parse().ok()).collect();
            if let Some(&last) = angles.last() {
                angles.resize(doors.doors.len(), last);
            }
            doors.set_angles_deg(&angles);
        }
        let pipeline = space_soup::renderer::mesh_pipeline::MeshPipeline::new_multisampled(device, format, uniform_layout, samples);
        let lightmap = space_soup::renderer::brush_pipeline::default_brush_lightmap(device, queue, &pipeline.lightmap_layout);
        let yaw_inv = Quat::from_rotation_y(-yaw);
        let mut leaves = Vec::new();
        for (id, position, rotation) in doors.poses() {
            let Some(mref) = scene.find_object(id).and_then(|o| o.mesh.clone()) else { continue };
            let mut mesh = match space_soup::renderer::mesh::GltfMesh::load(device, queue, &pipeline.texture_layout, &game.join(&mref.path)) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("offline frame: door {id}'s model did not load: {e:#}");
                    continue;
                }
            };
            mesh.position = yaw_inv * (position - offset);
            mesh.rotation = yaw_inv * rotation * mref.rotation_offset;
            mesh.scale = mref.scale;
            let model = pipeline.create_model_uniform(device);
            // Read in front of the face the eye sees, as the headset does.
            let light_at = space_soup::renderer::doors::light_point(&doors.views(), position, Quat::from_rotation_y(yaw) * eye + offset);
            let room = space_soup::renderer::room_light::turned_to_player(
                &space_soup::renderer::room_light::room_light_at(descs, light_at),
                yaw,
            );
            model.upload_lit_bulb(queue, mesh.model_matrix(), 0.0, 0.0, &room, None, 0.0);
            leaves.push((mesh, model));
        }
        let casters: Vec<space_soup::renderer::shadow::DoorCaster> = doors
            .views()
            .iter()
            .map(|v| space_soup::renderer::shadow::DoorCaster { corners: v.corners_in(offset, yaw) })
            .collect();
        let tiles = if std::env::var("NO_DOOR_SHADOWS").as_deref() == Ok("1") {
            Vec::new()
        } else {
            let lamps: Vec<space_soup::renderer::shadow::CharacterLamp> = lights
                .iter()
                .map(|l| space_soup::renderer::shadow::CharacterLamp {
                    position: l.position,
                    direction: l.direction,
                    cos_outer: if l.kind == space_soup::renderer::LightKind::Spot { (l.cone_angle_deg.to_radians() * 0.5).cos() } else { -1.0 },
                    range: l.range,
                    intensity: l.intensity,
                    eligible: l.kind != space_soup::renderer::LightKind::Directional && l.casts_shadow(),
                })
                .collect();
            space_soup::renderer::shadow::moving_caster_tiles(&lamps, None, &casters, eye, &[])
                .into_iter()
                .filter_map(|(i, spheres)| {
                    let l = &lights[i];
                    let spot = (l.kind == space_soup::renderer::LightKind::Spot).then(|| (l.direction, (l.cone_angle_deg.to_radians() * 0.5).cos()));
                    space_soup::renderer::shadow::character_light_matrix(l.position, spot, &spheres, l.range).map(|m| (i, m))
                })
                .collect()
        };
        for (d, v) in doors.doors.iter().zip(doors.views()) {
            eprintln!("offline frame: door {} at {:.1} deg{}", d.id, d.drawn.to_degrees(), if v.shut { ", shut" } else { "" });
        }
        eprintln!(
            "offline frame: door tiles for lamps {:?}",
            tiles.iter().map(|(i, _)| (*i, (Quat::from_rotation_y(yaw) * lights[*i].position + offset).to_array().map(|v| (v * 10.0).round() / 10.0))).collect::<Vec<_>>()
        );
        Some(Self { pipeline, leaves, lightmap, tiles, views: doors.views() })
    }

    fn tile_lamps(&self) -> Vec<usize> {
        self.tiles.iter().map(|(i, _)| *i).collect()
    }

    /// The frame's shadow matrices with the doors' tiles in the moving
    /// casters' places.
    fn shadow_upload(&self, mut upload: ShadowUpload) -> ShadowUpload {
        for (k, (_, m)) in self.tiles.iter().enumerate() {
            upload.spot_view_proj[space_soup::renderer::shadow::MAX_SPOT_SHADOWS + k] = *m;
        }
        upload
    }

    /// The doors into their tiles of the moving-objects map, as the frame's
    /// moving pass draws them.
    fn record_shadows(&self, queue: &wgpu::Queue, shadows: &ShadowMap, encoder: &mut wgpu::CommandEncoder) {
        if self.tiles.is_empty() {
            return;
        }
        use space_soup::renderer::shadow::ShadowKind;
        for (k, (_, m)) in self.tiles.iter().enumerate() {
            shadows.upload_light(queue, ShadowKind::Character(k), *m);
        }
        let draws: Vec<space_soup::renderer::shadow::ShadowMeshDraw> = self
            .leaves
            .iter()
            .flat_map(|(mesh, model)| {
                mesh.primitives
                    .iter()
                    .filter(|p| p.casts_shadow)
                    .map(move |p| (&p.vertex_buffer, &p.index_buffer, p.indices.len() as u32, &model.bind_group))
            })
            .collect();
        let all: Vec<usize> = (0..draws.len()).collect();
        let mats: Vec<Mat4> = self.tiles.iter().map(|(_, m)| *m).collect();
        shadows.record_moving(encoder, false, Mat4::IDENTITY, &mats, &draws, &[], &[], &[], &all);
    }

    /// The leaves in the scene pass, after the brushes, as the frame draws
    /// its models.
    fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, uniforms: &'a wgpu::BindGroup) {
        pass.set_pipeline(&self.pipeline.pipeline);
        pass.set_bind_group(0, uniforms, &[]);
        for (mesh, model) in &self.leaves {
            for prim in mesh.primitives.iter().filter(|p| !p.blended) {
                pass.set_bind_group(1, &model.bind_group, &[]);
                pass.set_bind_group(2, &prim.texture.bind_group, &[]);
                pass.set_bind_group(3, &self.lightmap.bind_group, &[]);
                pass.set_vertex_buffer(0, prim.vertex_buffer.slice(..));
                pass.set_index_buffer(prim.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..prim.indices.len() as u32, 0, 0..1);
            }
        }
    }
}

/// A live sun map over the caves, OPT-IN (`CAVE_SHADOWS=1`). The hill's and
/// the brow's shadow on a cave now arrive baked in its `COLOR_1` alpha
/// (caveSky.sunVisibility, the same static sun the terrain bake casts), so the
/// default frame shades a cave the way that data does; this map, fitted over
/// the whole cave box at 2048, over-reached at the mouth (2026-10-08).
fn cave_sun_matrix(scene: &space_soup_engine::scene::Scene, lights: &[Light], offset: Vec3, yaw_inv: Quat) -> Option<Mat4> {
    if std::env::var("CAVE_SHADOWS").as_deref() != Ok("1") || std::env::var("NO_CAVE_SHADOWS").as_deref() == Ok("1") {
        return None;
    }
    let sun = lights.iter().find(|l| matches!(l.kind, space_soup::renderer::lights::LightKind::Directional))?;
    let caves: Vec<_> = scene.objects.iter().filter(|o| !o.hidden && o.mesh.is_some() && o.terrain_collider.is_some()).collect();
    let lo = caves.iter().map(|o| o.cuboid.position - o.cuboid.half_size).reduce(Vec3::min)?;
    let hi = caves.iter().map(|o| o.cuboid.position + o.cuboid.half_size).reduce(Vec3::max)?;
    let centre = yaw_inv * ((lo + hi) * 0.5 - offset);
    eprintln!("offline frame: the caves cast the sun's shadow (sun travelling along {:.2})", sun.direction);
    Some(space_soup::renderer::shadow::directional_light_matrix(sun.direction, centre, (hi - lo).length() * 0.5 + 1.0))
}

/// The static sun map's size on the headset.
const TOD_STATIC_SUN_SIZE: u32 = 1024;

/// THE STATIC SUN MAP UNDER THE TIME OF DAY: fitted to the brushes as
/// `lights::static_sun_matrix` fits it on the headset, margin included, here
/// in the player's frame, for the ONE directional light (the sun or the moon).
fn tod_static_sun_matrix(scene: &space_soup_engine::scene::Scene, lights: &[Light], offset: Vec3, yaw_inv: Quat) -> Option<Mat4> {
    let sun = lights.iter().find(|l| matches!(l.kind, space_soup::renderer::lights::LightKind::Directional))?;
    let mut geometry = BrushGeometry::load_with(scene, true);
    let (verts, _) = geometry.assemble(&[], offset, yaw_inv, 0.0)?;
    Some(space_soup::renderer::lights::static_sun_matrix(
        sun.direction,
        Some(verts.iter().map(|v| v.position)),
        None::<std::iter::Empty<[f32; 3]>>,
        Mat4::IDENTITY,
    ))
}

/// THE CAVES, as the headset draws them: every drawn scene model
/// (`scene_meshes::load`, the standalone path's list) whose file asked for
/// layered shading, placed in the player's frame as `render_prep` places a
/// mesh. Other models are not drawn here. `None` when there are none.
fn offline_caves(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    game: &std::path::Path,
    scene_name: &str,
    (format, samples): (wgpu::TextureFormat, u32),
    (uniform_layout, material_layout): (&wgpu::BindGroupLayout, &wgpu::BindGroupLayout),
    (offset, yaw_inv): (Vec3, Quat),
) -> Option<OfflineCaves> {
    use space_soup::renderer::layered_mesh_pipeline::LayeredMeshPipeline;
    let pipeline = LayeredMeshPipeline::new_multisampled(device, format, uniform_layout, material_layout, samples);
    // Only for its texture layout, which the glTF loader binds textures to.
    let textures = space_soup::renderer::mesh_pipeline::MeshPipeline::new(device, format, uniform_layout);
    let mut draws = Vec::new();
    for m in crate::scene_meshes::load(game, scene_name) {
        let mesh = match space_soup::renderer::mesh::GltfMesh::load(device, queue, &textures.texture_layout, &game.join(&m.path)) {
            Ok(mesh) => mesh,
            Err(e) => {
                eprintln!("offline frame: model {} did not load: {e:#}", m.path);
                continue;
            }
        };
        let model = Mat4::from_scale_rotation_translation(
            Vec3::from(m.scale),
            yaw_inv * Quat::from_array(m.rotation),
            yaw_inv * (Vec3::from(m.position) - offset),
        );
        for prim in &mesh.primitives {
            let Some(layered) = &prim.layered else { continue };
            // The game draws a cave with a MESH pipeline ModelUniform, so this
            // does too: binding the layered pipeline's own uniform here hid a
            // layout mismatch that broke every headset frame (2026-10-08).
            let uniform = textures.create_model_uniform(device);
            uniform.upload(queue, model);
            let shadow = textures.create_model_uniform(device);
            shadow.upload(queue, model);
            draws.push((
                uniform,
                layered.vertex_buffer.clone(),
                prim.index_buffer.clone(),
                prim.indices.len() as u32,
                (prim.vertex_buffer.clone(), shadow),
            ));
        }
    }
    eprintln!("offline frame: {} cave draw(s)", draws.len());
    (!draws.is_empty()).then_some((pipeline, draws))
}

struct OfflineTerrain {
    vb: wgpu::Buffer,
    ib: wgpu::Buffer,
    count: u32,
    material: space_soup::renderer::terrain_pipeline::TerrainMaterial,
    /// Its probe pass, and its scene reader.
    pass: space_soup::renderer::terrain_pipeline::TerrainPipeline,
    reader: space_soup::renderer::terrain_pipeline::TerrainPipeline,
    /// `TERRAIN_GENTLE=1`: the gentle reader, and how many indices from the
    /// first it draws.
    gentle: Option<(
        space_soup::renderer::terrain_pipeline::TerrainPipeline,
        u32,
        Option<(space_soup::renderer::terrain_pipeline::TerrainPipeline, u32)>,
    )>,
}

/// THE GROUND, as the headset draws it in the shipped single-eye path: its
/// geometry in the player's frame, its material with the level's ground map
/// and the lamps' masks after it (as `run_inner` hands them to the renderer),
/// its probe pass, poolless when `poolless`, and its reader's twin as the
/// frame chooses it -- spotless when `spotless`, baked where the ground map is
/// (`XrRenderer::terrain_reader`). `TERRAIN_READER=full|spotless|baked|
/// baked_spotless` draws that reader instead, `inlined` the
/// `terrain_reader` lever's inlined pass and reader. `None` for a level
/// without terrain.
#[allow(clippy::too_many_arguments)]
fn offline_terrain(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    game: &std::path::Path,
    scene_name: &str,
    lm: &[space_soup_engine::lightmaps::LoadedLightmap],
    (offset, yaw_inv, yaw): (Vec3, Quat, f32),
    (format, samples): (wgpu::TextureFormat, u32),
    (uniform_layout, probe_layout, fixups): (
        &wgpu::BindGroupLayout,
        &wgpu::BindGroupLayout,
        &space_soup::renderer::probe_fixup::ProbeFixups,
    ),
    (spotless, poolless): (bool, bool),
    weather: Option<&wgpu::BindGroupLayout>,
) -> Option<OfflineTerrain> {
    use space_soup::renderer::terrain_pipeline::{self as tp, TerrainImage, TerrainPipeline};
    use wgpu::util::DeviceExt;
    let (mut geometry, splat) = crate::load_scene_terrain(game, scene_name)?;
    let image = |m: &space_soup_engine::lightmaps::LoadedLightmap| TerrainImage {
        width: m.width,
        height: m.height,
        rgba: m.rgba.clone(),
    };
    let sky = lm.iter().find(|m| m.object_id == space_soup_engine::lightmaps::SCENE_TERRAIN_SKY_ID).map(image);
    let found: Vec<&space_soup_engine::lightmaps::LoadedLightmap> = (0..space_soup_engine::stationary::MAX_STATIONARY_LAYERS)
        .map(space_soup_engine::lightmaps::scene_terrain_stationary_id)
        .map_while(|id| lm.iter().find(|m| m.object_id == id))
        .collect();
    let masks: Vec<TerrainImage> = if found.is_empty() {
        Vec::new()
    } else {
        crate::scene_lights::usable_stationary_masks(found, &crate::scene_lights::stationary_channels(game, scene_name))
            .into_iter()
            .map(image)
            .collect()
    };
    let ground = sky.as_ref().map(|map| {
        map.with_stationary_masks(&masks)
            .unwrap_or_else(|| TerrainImage { width: map.width, height: map.height, rgba: map.rgba.clone() })
    });
    let sun_baked = sky.as_ref().is_some_and(TerrainImage::sun_baked_everywhere);
    // Under `TIME_OF_DAY` the sun is off its baked direction: the map's sun
    // marked unbaked in its own layer, as `XrRenderer::rebuild_terrain_material`
    // marks it, so the static sun map shades the ground.
    let tod = crate::time_of_day::offline_hour().is_some();
    let ground = match (ground, tod) {
        (Some(mut g), true) => {
            let first = sky.as_ref().map_or(0, |m| m.rgba.len()).min(g.rgba.len());
            for texel in g.rgba[..first].chunks_exact_mut(4) {
                texel[3] = 255;
            }
            Some(g)
        }
        (g, _) => g,
    };
    let sun_baked = sun_baked && !tod;
    let choice = std::env::var("TERRAIN_READER").ok();
    eprintln!(
        "offline frame: the ground, its map {} everywhere, {} mask layer(s), reader {}",
        if sun_baked { "baked" } else { "not baked" },
        masks.len(),
        choice.as_deref().unwrap_or("as shipped"),
    );
    let (reader, pass) = match choice.as_deref() {
        Some("inlined") => {
            let [read, pass, poolless_pass] =
                TerrainPipeline::new_inlined(device, format, uniform_layout, samples, probe_layout, fixups);
            (read, if poolless { poolless_pass } else { pass })
        }
        other => {
            let (spotless, sun_baked) = match other {
                None => (spotless, sun_baked),
                Some("full") => (false, false),
                Some("spotless") => (true, false),
                Some("baked") => (false, true),
                Some("baked_spotless") => (true, true),
                Some(x) => panic!("TERRAIN_READER={x}: not full, spotless, baked, baked_spotless or inlined"),
            };
            let reader =
                TerrainPipeline::new_probe_reader_twin(device, format, uniform_layout, samples, probe_layout, spotless, sun_baked);
            // `TERRAIN_PASS_CUT=<cut>`: the pass drawn with one of its
            // measurement cuts (`terrain_pipeline::PROBE_PASS_REGISTER_CUTS`),
            // as the `pass_cut` lever draws it on the headset.
            let pass = match std::env::var("TERRAIN_PASS_CUT") {
                Ok(cut) => TerrainPipeline::new_probe_pass_with_cut(device, uniform_layout, fixups, &cut)
                    .unwrap_or_else(|| panic!("no ground probe pass cut {cut}, or it no longer matches the shader")),
                Err(_) if poolless => TerrainPipeline::new_probe_pass_poolless(device, uniform_layout, fixups),
                Err(_) => TerrainPipeline::new_probe_pass(device, uniform_layout, fixups),
            };
            (reader, pass)
        }
    };
    // WITH WEATHER, the whole ground drawn with its weather twins: the map is
    // empty outside the areas, so outside them it is the same picture.
    let (reader, pass) = match weather {
        Some(layout) => {
            let twins = TerrainPipeline::new_weather_twins(device, format, uniform_layout, samples, probe_layout, fixups, layout);
            let [full, spotless_r, baked, baked_spotless] = twins.readers;
            let reader = match (spotless, sun_baked) {
                (false, false) => full,
                (true, false) => spotless_r,
                (false, true) => baked,
                (true, true) => baked_spotless,
            };
            (reader, if poolless { twins.pass_poolless } else { twins.pass })
        }
        None => (reader, pass),
    };
    let dir = game.join("textures").join("terrain");
    let material = tp::TerrainMaterial::from_layers_with(
        device,
        queue,
        &reader.material_layout,
        &tp::load_terrain_layers(&dir),
        &tp::load_terrain_normals(&dir),
        &tp::load_terrain_rough(&dir),
        &tp::load_terrain_ao(&dir),
        splat.as_ref(),
        ground.as_ref(),
        {
            let mut settings = tp::load_terrain_settings(&dir);
            let water = space_soup_engine::scene::Scene::load(&space_soup_engine::Manifest::scene_path(game, scene_name)).map(|s| s.water).unwrap_or_default();
            if let Some((line, band)) = crate::water_render::wet_shore(&water) {
                settings.wet_line = line;
                settings.wet_band = band;
            }
            settings
        },
    );
    let (verts, idx) = geometry.assemble(offset, yaw_inv, yaw)?;
    // `TERRAIN_GENTLE=1`: the gentle triangles drawn by the ground's slope
    // twin (`ground_twins`), as the headset draws them -- to prove the twin
    // draws the same picture.
    let threshold = tp::load_terrain_settings(&dir).biplanar_start_deg;
    let (idx, gentle) = if std::env::var("TERRAIN_GENTLE").is_ok_and(|v| v == "1") {
        let split = space_soup::renderer::ground_twins::SlopeSplit::new(&verts, &idx, &[], threshold);
        let kinds = weather.map(|l| (l, space_soup::renderer::weather::WeatherKinds::Both));
        let readers = space_soup::renderer::ground_twins::gentle_readers(device, format, uniform_layout, samples, probe_layout, kinds)
            .expect("the ground's gentle twins");
        let [full, spotless_r, baked, baked_spotless] = readers;
        let reader = match (spotless, sun_baked) {
            (false, false) => full,
            (true, false) => spotless_r,
            (false, true) => baked,
            (true, true) => baked_spotless,
        };
        // The steep ones by the steep twin, dry ground only, as the headset
        // draws them.
        let steep = match weather {
            None => {
                let [full, spotless_r, baked, baked_spotless] = space_soup::renderer::ground_twins::steep_readers(device, format, uniform_layout, samples, probe_layout)
                    .expect("the ground's steep twins");
                Some((
                    match (spotless, sun_baked) {
                        (false, false) => full,
                        (true, false) => spotless_r,
                        (false, true) => baked,
                        (true, true) => baked_spotless,
                    },
                    split.steep_total as u32,
                ))
            }
            Some(_) => None,
        };
        eprintln!(
            "offline frame: {} of {} ground indices drawn by the gentle twin, {} by the steep twin",
            split.gentle_total,
            split.indices.len(),
            steep.as_ref().map_or(0, |s| s.1)
        );
        (split.indices, Some((reader, split.gentle_total as u32, steep)))
    } else {
        (idx.to_vec(), None)
    };
    Some(OfflineTerrain {
        vb: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("offline_terrain_vb"),
            contents: bytemuck_cast(verts),
            usage: wgpu::BufferUsages::VERTEX,
        }),
        ib: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("offline_terrain_ib"),
            contents: bytemuck_cast(&idx),
            usage: wgpu::BufferUsages::INDEX,
        }),
        count: idx.len() as u32,
        material,
        pass,
        reader,
        gentle,
    })
}

/// THE GROUND MAP, built as `XrRenderer::ensure_ground_map` builds it on the
/// headset, from the same inputs the app hands the renderer: the terrain's
/// heights over its footprint, its baked map, its layers and settings, and the
/// sky and its sun. `None` for a level without terrain.
#[allow(clippy::too_many_arguments)]
fn ground_map(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    game: &std::path::Path,
    scene_name: &str,
    scene: &space_soup_engine::scene::Scene,
    sky: &space_soup::renderer::sky::SkyIrradiance,
    sun: Option<&space_soup::renderer::sky::SkySun>,
) -> Option<(wgpu::TextureView, ([f32; 4], f32))> {
    let map = ground_map_cpu(game, scene_name, scene, sky, sun)?;
    let extent = map.max - map.min;
    let placement = ([map.min.x, map.min.y, 1.0 / extent.x, 1.0 / extent.y], map.top);
    Some((space_soup::renderer::ground_map::upload(device, queue, &map), placement))
}

/// [`ground_map`]'s picture, before it goes to the GPU.
pub(crate) fn ground_map_cpu(
    game: &std::path::Path,
    scene_name: &str,
    scene: &space_soup_engine::scene::Scene,
    sky: &space_soup::renderer::sky::SkyIrradiance,
    sun: Option<&space_soup::renderer::sky::SkySun>,
) -> Option<space_soup::renderer::ground_map::GroundMap> {
    use space_soup::renderer::{ground_map, terrain_pipeline};
    let (geometry, splat) = crate::load_scene_terrain(game, scene_name)?;
    let heights = geometry.height_grid(scene.terrain.as_ref(), game)?;
    let occlusion = space_soup_engine::lightmaps::load_scene_lightmaps(game, scene_name)
        .into_iter()
        .find(|m| m.target == space_soup_engine::lightmaps::LightmapTarget::Terrain)
        .map(|m| terrain_pipeline::TerrainImage { width: m.width, height: m.height, rgba: m.rgba });
    let dir = game.join("textures").join("terrain");
    let layers = terrain_pipeline::load_terrain_layers(&dir);
    let mut settings = terrain_pipeline::load_terrain_settings(&dir);
    if let Some((line, band)) = crate::water_render::wet_shore(&scene.water) {
        settings.wet_line = line;
        settings.wet_band = band;
    }
    let map = ground_map::build(
        &ground_map::GroundInputs {
            heights: &heights,
            sky,
            sun,
            sky_occlusion: occlusion.as_ref(),
            layers: &layers,
            splat: splat.as_ref(),
            settings: &settings,
        },
        ground_map::GROUND_MAP_SIZE,
    );
    Some(map)
}

fn bytemuck_cast<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: BrushVertex and u32 are plain-old-data GPU layouts with no
    // padding the shader reads; this is exactly what bytemuck::cast_slice does
    // for them in the renderer.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

impl Shot {
    pub fn save(&self, path: &std::path::Path) {
        image::save_buffer(path, &self.rgba, self.width, self.height, image::ExtendedColorType::Rgba8)
            .expect("write the frame");
    }

    pub fn px(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * self.width + x) * 4) as usize;
        [self.rgba[i], self.rgba[i + 1], self.rgba[i + 2], self.rgba[i + 3]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The view the seam screenshots were taken from: the back of the hall,
    /// looking at the front wall and the ceiling junctions.
    fn back_to_front() -> View {
        View::headset(Vec3::new(-1.3, 1.6, -14.5), Vec3::new(0.4, 2.4, 3.7))
    }

    /// Pixels whose probe term (green, in the sources view) stands above the
    /// MEDIAN of its eight neighbours -- a one-pixel line or dot, which no real
    /// surface produces. Pixels touching an uncovered (magenta) one are left
    /// out: the harness draws no sky, so the edge of every opening is a real
    /// edge against nothing.
    fn probe_spikes(shot: &Shot) -> Vec<(u32, u32)> {
        let magenta = |p: [u8; 4]| p[0] > 250 && p[1] < 5 && p[2] > 250;
        let mut out = Vec::new();
        for y in 1..shot.height - 1 {
            for x in 1..shot.width - 1 {
                let mut nb = Vec::with_capacity(8);
                let mut near_hole = magenta(shot.px(x, y));
                for (dx, dy) in [(-1i32, -1i32), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
                    let p = shot.px((x as i32 + dx) as u32, (y as i32 + dy) as u32);
                    near_hole |= magenta(p);
                    nb.push(p[1]);
                }
                if near_hole {
                    continue;
                }
                nb.sort_unstable();
                if shot.px(x, y)[1] as i32 - nb[4] as i32 > 20 {
                    out.push((x, y));
                }
            }
        }
        out
    }

    /// THE FRONT-OF-HALL SEAM, held down.
    ///
    /// Probe box projection rejected a zero exit distance, so every MSAA edge
    /// pixel clamped onto a wall it reflected out through took the raw
    /// reflection direction: a one-pixel line of a different part of the probe
    /// photograph along each junction, visible from the back of the hall and
    /// never up close. 139 such pixels in the front junctions of this view
    /// before the fix, none after (2026-09-23).
    /// THE LIGHT LOOP'S CULLING CHANGES NO PIXEL. It skips a lamp past its
    /// range, outside its cone, or one its baked mask hides from the pixel,
    /// and each of those multiplies the lamp's whole contribution by exactly
    /// zero -- so the frame with culling must be the frame without it, byte
    /// for byte, in views that see lamps in other rooms through doorways and
    /// across the hall, and with the flashlight's bounce, whose half-space
    /// cone is culled behind its patch.
    #[test]
    fn light_culling_changes_no_pixel() {
        for (eye, at, flashlight) in [
            // Across the hall toward the hallway door: lamps in all three rooms.
            (Vec3::new(-2.45, 1.6, 2.0), Vec3::new(0.0, 0.0, -5.9), None),
            // Down the hallway, its sconces and the brick room beyond.
            (Vec3::new(3.4, 1.6, -3.0), Vec3::new(9.0, 1.4, -3.0), None),
            // The bench's torch_wall and torch_pillar: the beam on the west
            // wall, and past the pillar's edge into the room behind it.
            (
                Vec3::new(0.2, 1.6, -9.6),
                Vec3::new(-2.7, 1.3, -10.6),
                Some((Vec3::new(0.0, 1.25, -9.85), Vec3::new(-2.7, 1.1, -10.4))),
            ),
            (
                Vec3::new(0.3, 1.6, -3.0),
                Vec3::new(0.0, 0.9, -7.0),
                Some((Vec3::new(0.55, 1.25, -3.2), Vec3::new(0.0, 1.0, -6.4))),
            ),
        ] {
            let v = View { flashlight, ..View::headset(eye, at) };
            let Some(culled) = render_brushes("test_room", v) else {
                eprintln!("skipping: no GPU or no test_room");
                return;
            };
            let every = render_brushes("test_room", View { light_culling: false, ..v }).unwrap();
            let differ = culled.rgba.chunks(4).zip(every.rgba.chunks(4)).filter(|(a, b)| a != b).count();
            assert_eq!(differ, 0, "culling changed {differ} pixel(s) looking from {eye} at {at}");
        }
    }

    /// REFLECTIONS FROM THE HALF-RESOLUTION PASS ARE THE PER-PIXEL ONES. A
    /// probe texel spans several screen pixels, so computing the reflection at
    /// a quarter of the pixels -- and reading it back four texels at a time,
    /// by depth -- should change almost nothing. Measured when it was built
    /// (2026-09-27): a mean difference of 0.02-0.13 levels in four views, with
    /// only single-pixel edges inside the reflected image moving. This keeps
    /// it there: a broken upsample bleeds across every silhouette, and a
    /// broken normalisation shifts whole surfaces.
    /// THE DEPTH PREPASS CHANGES NO PIXEL: the brushes drawn depth-only first
    /// and shaded after at LessEqual give byte for byte the picture of
    /// shading them straight away. Anything else -- a surface lost to
    /// LessEqual failing by a rounding difference between the two draws, a
    /// face flickering -- shows here. From the back of the hall, where the
    /// most brushes overlap.
    #[test]
    fn the_depth_prepass_changes_no_pixel() {
        for (eye, at) in [
            (Vec3::new(-1.3, 1.6, -14.5), Vec3::new(0.4, 2.4, 3.7)),
            (Vec3::new(3.4, 1.6, -3.0), Vec3::new(9.0, 1.4, -3.0)),
        ] {
            let v = View::headset(eye, at);
            let Some(with) = render_brushes("test_room", v) else {
                eprintln!("skipping: no GPU or no test_room");
                return;
            };
            let without = render_brushes("test_room", View { depth_prepass: false, ..v }).unwrap();
            let differ = with.rgba.chunks(4).zip(without.rgba.chunks(4)).filter(|(a, b)| a != b).count();
            assert_eq!(differ, 0, "the depth prepass changed {differ} pixel(s) looking from {eye} at {at}");
        }
    }

    #[test]
    fn half_res_reflections_match_per_pixel_ones() {
        for (eye, at) in [
            (Vec3::new(-0.6, 1.6, -13.5), Vec3::new(0.0, 0.2, -7.0)),
            (Vec3::new(0.2, 1.6, -2.0), Vec3::new(0.0, 0.2, 3.7)),
        ] {
            let v = View::headset(eye, at);
            let Some(half) = render_brushes("test_room", v) else {
                eprintln!("skipping: no GPU or no test_room");
                return;
            };
            let full = render_brushes("test_room", View { half_res_reflections: false, ..v }).unwrap();
            let (mut total, mut large) = (0u64, 0usize);
            for (a, b) in half.rgba.chunks(4).zip(full.rgba.chunks(4)) {
                let d = (0..3).map(|c| (a[c] as i32 - b[c] as i32).unsigned_abs()).max().unwrap();
                total += d as u64;
                large += usize::from(d > 16);
            }
            let n = (half.width * half.height) as f64;
            let mean = total as f64 / n;
            let share = large as f64 / n;
            assert!(mean < 0.5, "half-res reflections moved the picture by {mean:.2} levels on average from {eye}");
            assert!(share < 0.005, "{:.3}% of pixels differ by more than 16 levels from {eye}", share * 100.0);
        }
    }

    #[test]
    fn no_single_pixel_probe_lines_along_the_room_seams() {
        let Some(msaa) = render_brushes("test_room", View { sources: true, ..back_to_front() }) else {
            eprintln!("skipping: no GPU or no test_room");
            return;
        };
        // WHAT MSAA ADDS. A thin bright line can be real -- the marble beside
        // the door reflects the sunlit strip of floor as one -- and it is there
        // with or without multisampling. The seam was an EDGE-PIXEL defect:
        // present at 4x, absent at 1x. So only a spike with no counterpart in
        // the single-sampled render, within a pixel, counts.
        let single = render_brushes("test_room", View { sources: true, samples: 1, ..back_to_front() }).unwrap();
        // A RESOLVED PIXEL CANNOT OUTSHINE WHAT IT AVERAGES. An MSAA pixel is the
        // mean of samples of the surfaces within about a pixel of it, so its
        // probe term can be no brighter than the brightest single-sampled pixel
        // in its 3x3 neighbourhood. The seam broke exactly that -- an edge pixel
        // showed a part of the photograph no neighbour showed. A thin REAL
        // feature does not: the doorway lintel added 2026-09-24, seen edge-on,
        // is a one-pixel line in both renders, and the old test (which only
        // excused spikes near a 1x SPIKE) failed on it.
        let brightest_single = |x: u32, y: u32| {
            (-1i32..=1)
                .flat_map(|dx| (-1i32..=1).map(move |dy| (dx, dy)))
                .map(|(dx, dy)| single.px((x as i32 + dx) as u32, (y as i32 + dy) as u32)[1])
                .max()
                .unwrap_or(0)
        };
        let added: Vec<(u32, u32)> = probe_spikes(&msaa)
            .into_iter()
            .filter(|&(x, y)| msaa.px(x, y)[1] as i32 > brightest_single(x, y) as i32 + 8)
            .collect();
        eprintln!("{} probe spikes only under MSAA: {:?}", added.len(), &added[..added.len().min(20)]);
        assert!(added.len() <= 4, "{} one-pixel probe spikes appear only with MSAA -- the seam is back", added.len());
    }

    /// THE DOORWAYS. From the hall into the hallway, along the hallway into
    /// the brick room, and down at the hall-to-hallway threshold -- each with
    /// the doorway handover on and off, as the sources view and as shaded, to
    /// $OUT. Run with `--ignored`.
    #[test]
    #[ignore]
    fn render_the_doorway_views() {
        let out = std::path::PathBuf::from(std::env::var("OUT").unwrap_or_else(|_| "/tmp".into()));
        let views = [
            ("hall_to_hallway", View::headset(Vec3::new(-1.5, 1.6, -3.0), Vec3::new(6.0, 1.3, -3.0))),
            ("hallway_to_brick", View::headset(Vec3::new(4.0, 1.6, -3.0), Vec3::new(12.0, 1.3, -3.0))),
            ("threshold", View::headset(Vec3::new(1.2, 1.6, -1.2), Vec3::new(3.0, 0.0, -3.0))),
        ];
        for (name, v) in views {
            for (tag, no_portals) in [("", false), ("_noportals", true)] {
                let Some(src) = render_brushes("test_room", View { sources: true, no_portals, ..v }) else {
                    eprintln!("skipping: no GPU or no test_room");
                    return;
                };
                src.save(&out.join(format!("door_{name}{tag}_sources.png")));
                render_brushes("test_room", View { adapt: true, no_portals, ..v })
                    .unwrap()
                    .save(&out.join(format!("door_{name}{tag}_adapted.png")));
            }
        }
    }

    /// THE WALL SPOT'S CORNER from the middle of the back of the hall, looking
    /// up at the ceiling where its pool reflects -- the "too square" reflection
    /// on the headset (2026-09-24). Sources and adapted, to $OUT.
    #[test]
    #[ignore]
    fn render_the_wall_spot_corner() {
        let out = std::path::PathBuf::from(std::env::var("OUT").unwrap_or_else(|_| "/tmp".into()));
        let v = View::headset(Vec3::new(-0.8, 1.6, -10.5), Vec3::new(2.2, 3.0, -15.2));
        let Some(src) = render_brushes("test_room", View { sources: true, ..v }) else {
            eprintln!("skipping: no GPU or no test_room");
            return;
        };
        src.save(&out.join("corner_sources.png"));
        render_brushes("test_room", View { adapt: true, ..v }).unwrap().save(&out.join("corner_adapted.png"));
    }

    /// THE MARBLE FLOOR WITHOUT SSR: the front doorway and the grass pillar are
    /// reflected in it on the headset only with SSR on (2026-09-25). This
    /// harness has no SSR, so what it shows is the probe's share alone.
    /// Sources and adapted, to $OUT.
    #[test]
    #[ignore]
    fn render_the_floor_reflections() {
        let out = std::path::PathBuf::from(std::env::var("OUT").unwrap_or_else(|_| "/tmp".into()));
        let views = [
            ("to_door", View::headset(Vec3::new(0.5, 1.6, -4.0), Vec3::new(0.0, 0.0, 2.5))),
            ("to_pillar", View::headset(Vec3::new(-1.0, 1.6, -1.5), Vec3::new(0.0, 0.0, -5.5))),
        ];
        for (name, v) in views {
            let Some(src) = render_brushes("test_room", View { sources: true, ..v }) else {
                eprintln!("skipping: no GPU or no test_room");
                return;
            };
            src.save(&out.join(format!("floor_{name}_sources.png")));
            // At exposure 1 as well: adaptation meters from the probes, so a
            // re-bake moves the whole frame and hides what the reflection did.
            render_brushes("test_room", v).unwrap().save(&out.join(format!("floor_{name}_raw.png")));
            render_brushes("test_room", View { adapt: true, ..v })
                .unwrap()
                .save(&out.join(format!("floor_{name}_adapted.png")));
        }
    }

    /// DIAGNOSTIC: the ground map around test_room's building, radiance and
    /// height, to $OUT/ground_rgb.png and ground_height.png.
    #[test]
    #[ignore]
    fn save_the_ground_map() {
        let out = std::path::PathBuf::from(std::env::var("OUT").unwrap_or_else(|_| "/tmp".into()));
        let game = crate::offline_frame::offline_game_dir();
        let scene = space_soup_engine::scene::Scene::load(&space_soup_engine::Manifest::scene_path(&game, "test_room")).unwrap();
        let s = scene.sky.as_ref().unwrap();
        let bytes = std::fs::read(game.join("skies").join(&s.id).join("sky.hdr")).unwrap();
        let p = space_soup::renderer::sky::decode_radiance(&bytes).unwrap();
        let (sky, sun) = space_soup::renderer::sky::sky_lighting(&p, s.rotation_deg, s.intensity);
        let map = super::ground_map_cpu(&game, "test_room", &scene, &sky, sun.as_ref()).unwrap();
        let extent = map.max - map.min;
        eprintln!("ground map {}x{} min {:?} max {:?} top {}", map.width, map.height, map.min, map.max, map.top);
        let at = |x: f32, z: f32| {
            let i = (((x - map.min.x) / extent.x) * map.width as f32) as u32;
            let j = (((z - map.min.y) / extent.y) * map.height as f32) as u32;
            (i.min(map.width - 1), j.min(map.height - 1))
        };
        let (i0, j0) = at(-8.0, -20.0);
        let (i1, j1) = at(22.0, 10.0);
        let (w, h) = (i1 - i0, j1 - j0);
        let mut rgb = image::RgbImage::new(w, h);
        let mut height = image::GrayImage::new(w, h);
        for j in 0..h {
            for i in 0..w {
                let t = map.texels[((j0 + j) * map.width + i0 + i) as usize];
                let enc = |v: f32| ((v / (v + 0.5)).powf(1.0 / 2.2) * 255.0) as u8;
                rgb.put_pixel(i, j, image::Rgb([enc(t[0]), enc(t[1]), enc(t[2])]));
                height.put_pixel(i, j, image::Luma([((t[3] + 1.0) * 40.0).clamp(0.0, 255.0) as u8]));
            }
        }
        rgb.save(out.join("ground_rgb.png")).unwrap();
        height.save(out.join("ground_height.png")).unwrap();
        for (x, z) in [(0.0, 4.05), (0.0, 4.3), (0.0, 5.0), (0.0, 8.0), (-3.05, -8.0), (-3.5, -8.0), (-5.0, -8.0), (-8.0, -8.0)] {
            let (i, j) = at(x, z);
            eprintln!("({x}, {z}): {:?}", map.texels[(j * map.width + i) as usize]);
        }
    }

    /// THE SAME PICTURE AT ANY TURN. A snap or stick turn changes only the
    /// rig's yaw, which turns the frame geometry and lights reach the shaders
    /// in -- the world drawn is the same, so the picture must be. Players turn
    /// by stick, by snapping and by turning round (user, 2026-10-01). The
    /// bounce direction a lightmap stores was baked in the world and read
    /// against the player's normals: the hallway's rock read 1% darker at 180
    /// degrees on the headset and exactly the same here once fixed. From the
    /// hall through the hallway door, where rock, marble and both rooms show;
    /// and outside, at the building's sunlit walls. This harness draws no
    /// ground: the grass, whose normal map was bent along the player's axes
    /// (headset `outdoors_front-y90`, 2026-10-01 12:30), is
    /// `terrain_pipeline`'s `a_normal_maps_bumps_do_not_turn_with_the_player`.
    #[test]
    fn a_view_is_the_same_picture_at_any_turn() {
        for (eye, at) in [
            (Vec3::new(1.5, 1.6, -3.6), Vec3::new(2.8, 1.75, -2.35)),
            (Vec3::new(0.0, 1.6, 10.0), Vec3::new(0.5, 0.0, 7.0)),
        ] {
            let view = View { width: 192, height: 192, linear: true, ..View::headset(eye, at) };
            let Some(still) = render_brushes("test_room", view) else {
                eprintln!("skipping: no GPU or no test_room");
                return;
            };
            for yaw_deg in [90.0, 180.0, -45.0] {
                let turned = render_brushes("test_room", View { yaw_deg, ..view }).unwrap();
                let off: Vec<u8> = still.rgba.iter().zip(&turned.rgba).map(|(a, b)| a.abs_diff(*b)).collect();
                let share = off.iter().filter(|&&d| d > 2).count() as f32 / off.len() as f32;
                let mean = off.iter().map(|&d| d as f32).sum::<f32>() / off.len() as f32;
                assert!(
                    share < 0.001 && mean < 0.05,
                    "from {eye} turned {yaw_deg} degrees: {:.3}% of channels off by more than 2, mean {mean:.3}",
                    share * 100.0
                );
            }
        }
    }

    /// THE PILLAR FROM WHERE IT WAS LOOKED AT on the headset (2026-09-25,
    /// screenshot 23:06:10): the floor in front of it should mirror it and,
    /// with SSR off, did not. Adapted, to $OUT/pillar_$TAG.png.
    #[test]
    #[ignore]
    fn render_the_pillar_reflection() {
        let out = std::path::PathBuf::from(std::env::var("OUT").unwrap_or_else(|_| "/tmp".into()));
        let tag = std::env::var("TAG").unwrap_or_else(|_| "current".into());
        let v = match std::env::var("FAR").as_deref() {
            Ok("1") => View::headset(Vec3::new(0.3, 1.6, 1.0), Vec3::new(0.0, 0.5, -7.0)),
            // From the back of the hall toward the pillar, and from the hall
            // toward the hallway door across the marble floor.
            Ok("back") => View::headset(Vec3::new(-0.6, 1.6, -13.5), Vec3::new(0.0, 0.2, -7.0)),
            Ok("hallway") => View::headset(Vec3::new(-1.8, 1.6, -0.5), Vec3::new(2.7, 0.3, -3.0)),
            // Up at the ceiling where the pillar meets it (headset 19:50:16).
            Ok("ceiling") => View::headset(Vec3::new(0.9, 1.6, -3.6), Vec3::new(-0.1, 3.1, -7.6)),
            // Standing where the pillar HIDES the corner spot's pool on the
            // right wall at the back, looking at the floor in front of the
            // pillar: the pool must not appear in that floor (headset,
            // 2026-09-26). From here the eye-to-pool line crosses the pillar.
            Ok("hidden_pool") => View::headset(Vec3::new(-1.6, 1.6, -1.0), Vec3::new(0.3, 0.3, -6.3)),
            // From the front-left, where the pillar hides the corner pool AND
            // the pool's mirror point on the floor lies in front of the pillar
            // (eye z > 0.65 and x < -2.36 for this pool).
            Ok("hidden_pool_front") => View::headset(Vec3::new(-2.45, 1.6, 2.0), Vec3::new(0.0, 0.0, -5.9)),
            // From the front door, straight down the hall at the pillar's base.
            Ok("front_door") => View::headset(Vec3::new(0.0, 1.6, 2.5), Vec3::new(0.2, 0.0, -6.0)),
            // Close to the hallway door, looking down at the floor in front of
            // it where the hallway reflects (headset 21:35:38).
            Ok("hallway_near") => View::headset(Vec3::new(0.3, 1.6, -3.0), Vec3::new(2.2, 0.0, -3.0)),
            // Facing the front door from inside the hall, down at the floor
            // that reflects the outdoors through it (headset 01:48:36): the
            // reflection must carry the terrain and the sky, not go black.
            Ok("toward_front") => View::headset(Vec3::new(0.2, 1.6, -2.0), Vec3::new(0.0, 0.2, 3.7)),
            // The back corner spot's pool on the right wall (headset 01:48,
            // "reading like crosses, very blocky"), and the floor reflecting it.
            Ok("corner_pool") => View::headset(Vec3::new(-1.2, 1.6, -11.8), Vec3::new(2.7, 1.7, -14.2)),
            Ok("corner_pool_floor") => View::headset(Vec3::new(-1.0, 1.6, -10.0), Vec3::new(2.0, 0.0, -13.2)),
            // The hallway's wall sconces, where the housing must shadow its
            // own lamp (tracker 2.5): the south one, and down the hallway.
            Ok("sconce") => View::headset(Vec3::new(5.4, 1.6, -2.4), Vec3::new(5.0, 1.9, -4.2)),
            Ok("sconce_far") => View::headset(Vec3::new(3.4, 1.6, -3.0), Vec3::new(9.0, 1.4, -3.0)),
            // The pillar from the front-left, where its floor reflection
            // seemed to meet only half its width (headset 2026-09-27 19:10).
            Ok("pillar_oblique") => View::headset(Vec3::new(-1.2, 1.6, -2.5), Vec3::new(0.2, 0.2, -7.0)),
            // Any other viewpoint: `FAR=ex,ey,ez,ax,ay,az`.
            Ok(s) if s.split(',').count() == 6 => {
                let n: Vec<f32> = s.split(',').map(|x| x.trim().parse().unwrap()).collect();
                View::headset(Vec3::new(n[0], n[1], n[2]), Vec3::new(n[3], n[4], n[5]))
            }
            _ => View::headset(Vec3::new(0.3, 1.6, -3.0), Vec3::new(0.0, 0.9, -7.0)),
        };
        let sources = std::env::var("SOURCES").as_deref() == Ok("1");
        let half_res_reflections = std::env::var("FULL_RES").as_deref() != Ok("1");
        // Framed as a Cycles reference is (`tools/reference`): `W`, `H` and the
        // vertical field of view `FOVY`, and `LINEAR=1` for raw radiance.
        let num = |k: &str| std::env::var(k).ok().and_then(|x| x.parse::<f32>().ok());
        let linear = std::env::var("LINEAR").as_deref() == Ok("1");
        let v = View {
            width: num("W").map_or(v.width, |x| x as u32),
            height: num("H").map_or(v.height, |x| x as u32),
            fov_y: num("FOVY").unwrap_or(v.fov_y),
            roll_deg: num("ROLL").unwrap_or(0.0),
            yaw_deg: num("YAW").unwrap_or(0.0),
            jitter_px: [num("JX").unwrap_or(0.0), num("JY").unwrap_or(0.0)],
            linear,
            cut: std::env::var("CUT").ok().map(|c| &*Box::leak(c.into_boxed_str())),
            ..v
        };
        let Some(shot) = render_brushes("test_room", View { adapt: !sources && !linear, sources, half_res_reflections, ..v }) else {
            eprintln!("skipping: no GPU or no test_room");
            return;
        };
        shot.save(&out.join(format!("pillar_{tag}.png")));
    }

    /// THE DOORWAY'S SUN PATCH from inside the hall (headset, 2026-09-25,
    /// screenshot 23:06:15): its diagonal edge read as a comb of one-pixel
    /// steps. Adapted, to $OUT/sunedge_$TAG.png.
    #[test]
    #[ignore]
    fn render_the_doorway_sun_edge() {
        let out = std::path::PathBuf::from(std::env::var("OUT").unwrap_or_else(|_| "/tmp".into()));
        let tag = std::env::var("TAG").unwrap_or_else(|_| "current".into());
        let v = View::headset(Vec3::new(0.6, 1.6, 0.6), Vec3::new(-0.1, 0.0, 3.2));
        let Some(shot) = render_brushes("test_room", View { adapt: true, ..v }) else {
            eprintln!("skipping: no GPU or no test_room");
            return;
        };
        shot.save(&out.join(format!("sunedge_{tag}.png")));
    }

    /// Writes the shaded frame and the sources diagnostic to $OUT (default
    /// /tmp). Run by hand: `cargo test --lib offline_frame -- --ignored`.
    #[test]
    #[ignore]
    fn render_the_back_to_front_view() {
        let out = std::path::PathBuf::from(std::env::var("OUT").unwrap_or("/tmp".into()));
        let Some(shot) = render_brushes("test_room", back_to_front()) else {
            eprintln!("skipping: no GPU or no test_room");
            return;
        };
        shot.save(&out.join("offline_back_to_front.png"));
        let sources = render_brushes("test_room", View { sources: true, ..back_to_front() }).unwrap();
        sources.save(&out.join("offline_back_to_front_sources.png"));
        let single = render_brushes("test_room", View { sources: true, samples: 1, ..back_to_front() }).unwrap();
        single.save(&out.join("offline_back_to_front_sources_1x.png"));
        let adapted = render_brushes("test_room", View { adapt: true, ..back_to_front() }).unwrap();
        adapted.save(&out.join("offline_back_to_front_adapted.png"));
        let mid = View { adapt: true, ..View::headset(Vec3::new(-1.3, 1.6, -6.0), Vec3::new(-1.0, 1.4, -15.7)) };
        render_brushes("test_room", mid).unwrap().save(&out.join("offline_mid_looking_back_adapted.png"));
        let holes = shot.rgba.chunks(4).filter(|p| p[0] > 250 && p[1] < 5 && p[2] > 250).count();
        eprintln!("wrote {}x{} frames to {}; {holes} magenta (uncovered) pixels", shot.width, shot.height, out.display());
    }
}

#[cfg(test)]
mod probe_brightness_diag {
    /// DIAGNOSTIC: the mean radiance of each baked probe, which is what the
    /// headset logs at load and what eye adaptation meters from.
    #[test]
    #[ignore]
    fn print_probe_brightness() {
        let game = crate::offline_frame::offline_game_dir();
        for p in space_soup_engine::reflection_probe::load_scene_probes(&game, "test_room") {
            let m = space_soup::renderer::uniforms::probe_mean_radiance(&p.faces, p.resolution);
            eprintln!("probe centre {:?} box {:?}..{:?}: mean radiance {m:.4}", p.centre, p.min, p.max);
        }
    }
}

#[cfg(test)]
mod exposure_calibration {
    use glam::Vec3;
    use space_soup::renderer::exposure::{exposure_for, EyeAdaptation};

    pub(crate) fn test_room_eye() -> Option<EyeAdaptation> {
        let game = crate::offline_frame::offline_game_dir();
        let scene = space_soup_engine::scene::Scene::load(&space_soup_engine::Manifest::scene_path(&game, "test_room")).ok()?;
        let s = scene.sky.as_ref()?;
        let bytes = std::fs::read(game.join("skies").join(&s.id).join("sky.hdr")).ok()?;
        let pano = space_soup::renderer::sky::decode_radiance(&bytes).ok()?;
        let (sky, _) = space_soup::renderer::sky::sky_lighting(&pano, s.rotation_deg, s.intensity);
        let level = crate::probe_level::ProbeLevel::load(&game, "test_room")?;
        let source = level.source();
        let faces: Vec<(Vec<u8>, space_soup::renderer::probe_stream::ProbeDesc)> =
            level.descs.iter().enumerate().filter_map(|(i, d)| Some((source(i)?, *d))).collect();
        let probes: Vec<(&[u8], u32, space_soup::renderer::probe_stream::ProbeDesc)> =
            faces.iter().map(|(f, d)| (f.as_slice(), level.resolution, *d)).collect();
        let mut eye = EyeAdaptation::from_probes(&probes, sky);
        eye.set_portals(&level.portals);
        Some(eye)
    }

    #[test]
    #[ignore]
    fn print_test_room_metering() {
        let Some(eye) = test_room_eye() else { return };
        for (name, at, look) in [
            ("outside, horizon", Vec3::new(0.0, 1.6, 12.0), Vec3::new(0.0, 0.0, -1.0)),
            ("outside, toward hall", Vec3::new(0.0, 1.6, 8.0), Vec3::new(0.0, -0.1, -1.0)),
            ("doorway, looking in", Vec3::new(0.0, 1.6, 3.0), Vec3::new(0.0, 0.0, -1.0)),
            ("hall front, looking back", Vec3::new(-0.5, 1.6, 1.0), Vec3::new(0.0, 0.0, -1.0)),
            ("hall mid, looking back", Vec3::new(-1.3, 1.6, -6.0), Vec3::new(0.0, 0.0, -1.0)),
            ("hall back, looking front", Vec3::new(-1.3, 1.6, -14.5), Vec3::new(0.1, 0.05, 1.0)),
            ("hall back, looking at wall", Vec3::new(-1.3, 1.6, -14.5), Vec3::new(0.0, 0.0, -1.0)),
            // The hallway between the hall and the brick hall, and the views
            // into it from either side.
            ("hall, looking into hallway", Vec3::new(1.5, 1.6, -3.0), Vec3::new(1.0, 0.0, 0.0)),
            ("hallway west, looking east", Vec3::new(4.0, 1.6, -3.0), Vec3::new(1.0, 0.0, 0.0)),
            ("hallway mid, looking east", Vec3::new(6.5, 1.6, -3.0), Vec3::new(1.0, 0.0, 0.0)),
            ("hallway mid, looking west", Vec3::new(6.5, 1.6, -3.0), Vec3::new(-1.0, 0.0, 0.0)),
            ("hallway east, looking west", Vec3::new(9.0, 1.6, -3.0), Vec3::new(-1.0, 0.0, 0.0)),
            ("brick, looking into hallway", Vec3::new(11.5, 1.6, -3.0), Vec3::new(-1.0, 0.0, 0.0)),
            ("brick mid, looking west", Vec3::new(14.0, 1.6, -2.0), Vec3::new(-1.0, 0.0, 0.0)),
        ] {
            let m = eye.meter(at, look);
            eprintln!("{name:>28}: meter {m:.4}  exposure {:.2}", exposure_for(m));
        }
        // Walking into the hallway from either side, looking down it: the
        // meter should hand over through each doorway without a jump.
        for (name, from, to, look) in [
            ("hall -> hallway", Vec3::new(0.5, 1.6, -3.0), Vec3::new(5.0, 1.6, -3.0), Vec3::X),
            ("brick -> hallway", Vec3::new(13.0, 1.6, -3.0), Vec3::new(8.0, 1.6, -3.0), Vec3::NEG_X),
        ] {
            let steps = 18;
            let line: Vec<String> = (0..=steps)
                .map(|i| {
                    let at = from.lerp(to, i as f32 / steps as f32);
                    format!("{:.2}:{:.2}", at.x, exposure_for(eye.meter(at, look)))
                })
                .collect();
            eprintln!("{name}, x:exposure: {}", line.join(" "));
        }
    }

}

/// The benchmark's viewpoints, checked and rendered here. See `bench.py`.
#[cfg(test)]
mod bench_views {
    use super::*;

    /// The benchmark's viewpoints (`bench_views.json`), as `bench.py` pins
    /// the headset's camera to them.
    pub(super) fn bench_views() -> Vec<(String, Vec3, Vec3)> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("bench_views.json");
        let doc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        doc["views"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                let at = |k: &str| Vec3::from(<[f32; 3]>::try_from(serde_json::from_value::<Vec<f32>>(v[k].clone()).unwrap()).unwrap());
                (v["name"].as_str().unwrap().to_string(), at("eye"), at("at"))
            })
            .collect()
    }

    /// WHAT `bench.py` WRITES INTO THE LEVER FILE IS WHAT THE RENDERER READS.
    /// The script's own output, not a copy of its format: a field it spells
    /// differently would otherwise be refused on the headset, mid-run, as an
    /// unknown lever.
    #[test]
    fn the_bench_scripts_lever_file_is_read_by_the_renderer() {
        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("bench.py");
        for (name, eye, at) in bench_views() {
            let run = std::process::Command::new("python3").arg(&script).args(["--print-levers", &name, "--ab"]).output();
            let Ok(run) = run else {
                eprintln!("skipping: no python3");
                return;
            };
            assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
            let text = String::from_utf8(run.stdout).unwrap();
            let levers = space_soup::renderer::levers::Levers::parse(&text).unwrap_or_else(|e| panic!("{name}: {e}\n{text}"));
            let bench = levers.bench.as_ref().unwrap();
            assert_eq!((bench.name.as_str(), Vec3::from(bench.eye), Vec3::from(bench.at)), (name.as_str(), eye, at));
            assert!(levers.ab_cycle);
        }
    }

    /// Each viewpoint's flashlight, if it carries one: its glass, where it is
    /// aimed, and whether its torch is drawn.
    fn bench_flashlights() -> std::collections::HashMap<String, (Vec3, Vec3, bool)> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("bench_views.json");
        let doc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let v3 = |v: &serde_json::Value| Vec3::from(<[f32; 3]>::try_from(serde_json::from_value::<Vec<f32>>(v.clone()).unwrap()).unwrap());
        doc["views"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|v| v["flashlight"].is_object())
            .map(|v| {
                let f = &v["flashlight"];
                (v["name"].as_str().unwrap().to_string(), (v3(&f["at"]), v3(&f["aim"]), f["torch"].as_bool().unwrap_or(false)))
            })
            .collect()
    }

    /// THE BENCHMARK'S VIEWPOINTS, rendered here, so the frame the headset
    /// measures from each can be looked at: `$OUT/bench_<name>.png`, each with
    /// its flashlight and its torch's reflection (not the torch itself, nor
    /// any glare), the ground (`TERRAIN=0` leaves it out), the effects
/// (`EFFECTS=0` leaves them out, `EFFECTS_TIME` sets their clock) and the
/// water and sky (`WATER=0` leaves them out, `WATER_TIME`). `ONLY=a,b` for
    /// some of them, `CUT=<scene register cut>` for the brushes' scene shader
    /// with one of its measurement cuts.
    #[test]
    #[ignore]
    fn render_the_bench_views() {
        let out = std::path::PathBuf::from(std::env::var("OUT").unwrap_or_else(|_| "/tmp".into()));
        let only = std::env::var("ONLY").ok();
        let terrain = std::env::var("TERRAIN").as_deref() != Ok("0");
        let cut = std::env::var("CUT").ok().map(|c| &*Box::leak(c.into_boxed_str()));
        let lights = bench_flashlights();
        for (name, eye, at) in bench_views() {
            if only.as_ref().is_some_and(|o| !o.split(',').any(|n| n == name)) {
                continue;
            }
            let (flashlight, torch) = match lights.get(&name) {
                Some(&(glass, aim, torch)) => (Some((glass, aim)), torch),
                None => (None, false),
            };
            let effects = std::env::var("EFFECTS").as_deref() != Ok("0");
            let water = std::env::var("WATER").as_deref() != Ok("0");
            let weather = std::env::var("WEATHER").as_deref() != Ok("0");
            let Some(shot) = render_brushes("test_room", View { adapt: true, flashlight, torch, terrain, cut, effects, water, weather, ..View::headset(eye, at) }) else {
                eprintln!("skipping: no GPU or no test_room");
                return;
            };
            shot.save(&out.join(format!("bench_{name}.png")));
        }
    }

    /// ONE VIEWPOINT from the environment, to look again at something seen on
    /// the headset: `EYE=x,y,z` looking at `AT=x,y,z`, a flashlight held at
    /// `TORCH_AT` and aimed at `TORCH_AIM` if both are set, `CUT` and
    /// `TERRAIN=0` as for the bench views. `$OUT/$NAME.png` (`view`).
    #[test]
    #[ignore]
    fn render_one_view() {
        let v3 = |key: &str| {
            std::env::var(key).ok().map(|s| {
                let v: Vec<f32> = s.split(',').map(|x| x.trim().parse().unwrap()).collect();
                Vec3::new(v[0], v[1], v[2])
            })
        };
        let (Some(eye), Some(at)) = (v3("EYE"), v3("AT")) else { panic!("set EYE and AT") };
        let out = std::path::PathBuf::from(std::env::var("OUT").unwrap_or_else(|_| "/tmp".into()));
        let name = std::env::var("NAME").unwrap_or_else(|_| "view".into());
        let terrain = std::env::var("TERRAIN").as_deref() != Ok("0");
        let cut = std::env::var("CUT").ok().map(|c| &*Box::leak(c.into_boxed_str()));
        let flashlight = v3("TORCH_AT").zip(v3("TORCH_AIM"));
        let effects = std::env::var("EFFECTS").as_deref() != Ok("0");
        let water = std::env::var("WATER").as_deref() != Ok("0");
        let weather = std::env::var("WEATHER").as_deref() != Ok("0");
        let Some(shot) = render_brushes("test_room", View { adapt: true, flashlight, terrain, cut, effects, water, weather, ..View::headset(eye, at) }) else {
            eprintln!("skipping: no GPU or no test_room");
            return;
        };
        shot.save(&out.join(format!("{name}.png")));
    }
}

/// ALIASING, MEASURED AGAINST A SUPERSAMPLED REFERENCE (2026-09-30).
///
/// The crawl probe (`jitter_px`) slides the pixel grid under a still camera
/// and watches each pixel change -- but a sharp texture that is filtered
/// correctly changes too, so change alone cannot tell detail from aliasing
/// (the hallway's crawl was "mostly texture and normal detail", which is
/// either). Here every shift is rendered twice: at the headset's resolution,
/// and at `K` times it with each K x K block averaged in linear light -- the
/// picture the frame would be with K x K times the shading samples. A pixel
/// that follows its reference differs from it by about the same amount at
/// every shift, however sharp it is; an aliased one differs by a different
/// amount at every shift. The score is that variation: the standard deviation,
/// across the shifts, of the frame's luma minus the reference's, in sRGB
/// levels (0-255).
///
/// What it cannot see: anything the reference aliases the same way -- shadow
/// and lightmap texels, which are fixed in the world, not on the screen.
#[cfg(test)]
mod aliasing {
    use super::*;

    fn srgb_to_linear(c: u8) -> f32 {
        let c = c as f32 / 255.0;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    }

    /// Linear light to sRGB-encoded levels, 0-255, unrounded.
    fn linear_to_levels(l: f32) -> f32 {
        let l = l.clamp(0.0, 1.0);
        255.0 * if l <= 0.0031308 { l * 12.92 } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 }
    }

    /// `shot` averaged over `k` x `k` blocks in linear light: the per-pixel
    /// luminance, sRGB-encoded (levels), and the averaged colour for saving.
    fn downsampled(shot: &Shot, k: u32) -> (Vec<f32>, Shot) {
        let lut: Vec<f32> = (0..=255u8).map(srgb_to_linear).collect();
        let (w, h) = (shot.width / k, shot.height / k);
        let mut luma = Vec::with_capacity((w * h) as usize);
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        let n = (k * k) as f32;
        for y in 0..h {
            for x in 0..w {
                let mut sum = [0.0f32; 3];
                for dy in 0..k {
                    for dx in 0..k {
                        let p = shot.px(x * k + dx, y * k + dy);
                        for c in 0..3 {
                            sum[c] += lut[p[c] as usize];
                        }
                    }
                }
                let rgb = sum.map(|s| s / n);
                luma.push(linear_to_levels(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]));
                rgba.extend(rgb.map(|c| linear_to_levels(c).round() as u8));
                rgba.push(255);
            }
        }
        (luma, Shot { width: w, height: h, rgba })
    }

    /// Per pixel: the standard deviation across the shifts of `frames[i] -
    /// refs[i]` -- the part of the frame's error that moves with the grid.
    fn aliasing_score(frames: &[Vec<f32>], refs: &[Vec<f32>]) -> Vec<f32> {
        let n = frames.len() as f32;
        (0..frames[0].len())
            .map(|p| {
                let d: Vec<f32> = frames.iter().zip(refs).map(|(f, r)| f[p] - r[p]).collect();
                let mean = d.iter().sum::<f32>() / n;
                (d.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n).sqrt()
            })
            .collect()
    }

    /// Per pixel: the standard deviation across the shifts of the value
    /// itself -- the crawl probe's measure, detail and aliasing together.
    fn change(frames: &[Vec<f32>]) -> Vec<f32> {
        let n = frames.len() as f32;
        (0..frames[0].len())
            .map(|p| {
                let mean = frames.iter().map(|f| f[p]).sum::<f32>() / n;
                (frames.iter().map(|f| (f[p] - mean) * (f[p] - mean)).sum::<f32>() / n).sqrt()
            })
            .collect()
    }

    /// A still frame that follows its reference scores nothing however much
    /// both change; one that flips a pixel the reference only shades scores.
    #[test]
    fn a_frame_that_follows_its_reference_scores_nothing_and_a_flipping_one_scores() {
        let refs: Vec<Vec<f32>> = (0..6).map(|i| vec![100.0 + 20.0 * i as f32, 50.0]).collect();
        let following: Vec<Vec<f32>> = refs.iter().map(|r| r.iter().map(|v| v - 7.0).collect()).collect();
        assert!(aliasing_score(&following, &refs).iter().all(|&s| s < 1e-4));
        assert!(change(&following)[0] > 30.0, "the detail itself changes");
        let flipping: Vec<Vec<f32>> = (0..6).map(|i| vec![100.0 + 20.0 * i as f32, if i % 2 == 0 { 0.0 } else { 100.0 }]).collect();
        let s = aliasing_score(&flipping, &refs);
        assert!(s[0] < 1e-4 && (s[1] - 50.0).abs() < 1e-3, "{s:?}");
    }

    #[test]
    fn a_block_average_is_taken_in_linear_light() {
        // Black and white average to half the light: level 188, not 128.
        let shot = Shot { width: 2, height: 2, rgba: [[0, 0, 0, 255], [255; 4], [255; 4], [0, 0, 0, 255]].concat() };
        let (luma, small) = downsampled(&shot, 2);
        assert_eq!((small.width, small.height), (1, 1));
        assert!((luma[0] - 187.5).abs() < 0.5, "{}", luma[0]);
        assert_eq!(small.px(0, 0)[0], 188);
    }

    /// SHIMMER UNDER A HEAD MOVE (2026-10-01): `VIEW` (or `FAR`) rendered
    /// with the head moved right `STEP_MM` (default 1) at a time, `STEPS`
    /// (default 8) times -- `bench.py`'s `-mQ` views, offline. A steady
    /// picture slides nearly linearly over a millimetre, so per pixel the RMS
    /// of the series' second difference (display levels, 0-255) is what
    /// changes incoherently: on the headset, the bright bits of the lamps'
    /// reflections in the polished walls. `REGION=x0,y0,x1,y1` scores one area
    /// alone; `NO_CARDS=1` etc. as for `render_brushes`. Writes
    /// $OUT/move_$TAG.png, the score over the first frame.
    #[test]
    #[ignore]
    fn measure_the_move_shimmer() {
        let out = std::path::PathBuf::from(std::env::var("OUT").unwrap_or_else(|_| "/tmp".into()));
        let tag = std::env::var("TAG").unwrap_or_else(|_| "current".into());
        let num = |k: &str| std::env::var(k).ok().and_then(|x| x.parse::<f32>().ok());
        let (eye, at) = match (std::env::var("VIEW"), std::env::var("FAR")) {
            (Ok(name), _) => {
                let views = super::bench_views::bench_views();
                let v = views.iter().find(|v| v.0 == name).unwrap_or_else(|| panic!("no bench view {name}"));
                (v.1, v.2)
            }
            (_, Ok(s)) => {
                let n: Vec<f32> = s.split(',').map(|x| x.trim().parse().unwrap()).collect();
                (Vec3::new(n[0], n[1], n[2]), Vec3::new(n[3], n[4], n[5]))
            }
            _ => panic!("set VIEW=<bench view> or FAR=ex,ey,ez,ax,ay,az"),
        };
        let steps = num("STEPS").map_or(8, |x| x as usize).max(3);
        let step = num("STEP_MM").unwrap_or(1.0) / 1000.0;
        let f = (at - eye).normalize();
        let right = Vec3::new(-f.z, 0.0, f.x).normalize();
        let base = View {
            adapt: true,
            half_res_reflections: std::env::var("FULL_RES").as_deref() != Ok("1"),
            cut: std::env::var("CUT").ok().map(|c| &*Box::leak(c.into_boxed_str())),
            ..View::headset(eye, at)
        };
        let (w, h) = (base.width, base.height);
        let mut frames = Vec::new();
        let mut first = None;
        for k in 0..steps {
            let d = right * (k as f32 * step);
            let Some(frame) = render_brushes("test_room", View { eye: eye + d, at: at + d, ..base }) else {
                eprintln!("skipping: no GPU or no test_room");
                return;
            };
            frames.push(downsampled(&frame, 1).0);
            if std::env::var("SAVE_FRAMES").as_deref() == Ok("1") {
                frame.save(&out.join(format!("move_{tag}_{k}.png")));
            }
            first.get_or_insert(frame);
        }
        let n = (steps - 2) as f32;
        let score: Vec<f32> = (0..frames[0].len())
            .map(|p| {
                let s: f32 = (1..steps - 1).map(|k| (frames[k + 1][p] - 2.0 * frames[k][p] + frames[k - 1][p]).powi(2)).sum();
                (s / n).sqrt()
            })
            .collect();
        let stats = |x0: u32, y0: u32, x1: u32, y1: u32| {
            let (mut sum, mut over8, mut count) = (0.0f64, 0usize, 0usize);
            for y in y0..y1 {
                for x in x0..x1 {
                    let v = score[(y * w + x) as usize];
                    sum += v as f64;
                    over8 += (v > 8.0) as usize;
                    count += 1;
                }
            }
            (sum / count.max(1) as f64, 100.0 * over8 as f64 / count.max(1) as f64)
        };
        let (a, o8) = stats(0, 0, w, h);
        eprintln!("{tag}: move shimmer {a:.3} levels (pixels over 8: {o8:.3}%), {steps} steps of {:.1} mm", step * 1000.0);
        if let Some(region) = std::env::var("REGION").ok() {
            let c: Vec<u32> = region.split(',').map(|x| x.trim().parse().unwrap()).collect();
            let (a, o8) = stats(c[0], c[1], c[2].min(w), c[3].min(h));
            eprintln!("{tag} region {region}: move shimmer {a:.3} (over 8: {o8:.2}%)");
        }
        let tile = 32;
        let mut tiles: Vec<(f64, u32, u32)> = Vec::new();
        for ty in 0..h / tile {
            for tx in 0..w / tile {
                tiles.push((stats(tx * tile, ty * tile, (tx + 1) * tile, (ty + 1) * tile).0, tx * tile, ty * tile));
            }
        }
        tiles.sort_by(|p, q| q.0.total_cmp(&p.0));
        for (a, x, y) in tiles.iter().take(10) {
            eprintln!("  tile {x},{y}..{},{}: move shimmer {a:.2}", x + tile, y + tile);
        }
        let mut heat = Vec::with_capacity((w * h * 4) as usize);
        for p in 0..(w * h) as usize {
            let dim = (frames[0][p] * 0.3) as u8;
            let hot = (score[p] * 16.0).min(255.0) as u8;
            heat.extend([dim.saturating_add(hot), dim, dim, 255]);
        }
        Shot { width: w, height: h, rgba: heat }.save(&out.join(format!("move_{tag}.png")));
        first.unwrap().save(&out.join(format!("move_{tag}_frame.png")));
    }

    /// `VIEW=<bench view>` or `FAR=ex,ey,ez,ax,ay,az`; `K` (default 3) and
    /// `SHIFTS` (default 6, along the diagonal, a pixel in all); `TILE` (32)
    /// for the table of worst tiles; `REGION=x0,y0,x1,y1` to score one area
    /// alone; `FULL_RES=1` for per-pixel reflections, `TERMINATOR_AA=0` for
    /// the lamps' hard terminator, `CUT=<scene register cut>` to take one
    /// term out of the picture (`scene_cut_lamp_spec`, `scene_cut_bounce`,
    /// ...), `ROLL` in degrees.
    /// Writes $OUT/alias_$TAG.png -- the score over the frame, dimmed -- and
    /// the frame and its reference at the first shift beside it.
    #[test]
    #[ignore]
    fn measure_the_aliasing() {
        let out = std::path::PathBuf::from(std::env::var("OUT").unwrap_or_else(|_| "/tmp".into()));
        let tag = std::env::var("TAG").unwrap_or_else(|_| "current".into());
        let num = |k: &str| std::env::var(k).ok().and_then(|x| x.parse::<f32>().ok());
        let (eye, at) = match (std::env::var("VIEW"), std::env::var("FAR")) {
            (Ok(name), _) => {
                let views = super::bench_views::bench_views();
                let v = views.iter().find(|v| v.0 == name).unwrap_or_else(|| panic!("no bench view {name}"));
                (v.1, v.2)
            }
            (_, Ok(s)) => {
                let n: Vec<f32> = s.split(',').map(|x| x.trim().parse().unwrap()).collect();
                (Vec3::new(n[0], n[1], n[2]), Vec3::new(n[3], n[4], n[5]))
            }
            _ => panic!("set VIEW=<bench view> or FAR=ex,ey,ez,ax,ay,az"),
        };
        let k = num("K").map_or(3, |x| x as u32);
        let shifts = num("SHIFTS").map_or(6, |x| x as usize);
        let tile = num("TILE").map_or(32, |x| x as u32);
        let base = View {
            adapt: true,
            half_res_reflections: std::env::var("FULL_RES").as_deref() != Ok("1"),
            terminator_aa: std::env::var("TERMINATOR_AA").as_deref() != Ok("0"),
            cut: std::env::var("CUT").ok().map(|c| &*Box::leak(c.into_boxed_str())),
            roll_deg: num("ROLL").unwrap_or(0.0),
            ..View::headset(eye, at)
        };
        let (w, h) = (base.width, base.height);

        let mut frames = Vec::new();
        let mut refs = Vec::new();
        let mut first: Option<(Shot, Shot)> = None;
        for i in 0..shifts {
            let s = i as f32 / shifts as f32;
            let Some(frame) = render_brushes("test_room", View { jitter_px: [s, s], ..base }) else {
                eprintln!("skipping: no GPU or no test_room");
                return;
            };
            let big = render_brushes(
                "test_room",
                View { width: w * k, height: h * k, jitter_px: [s * k as f32, s * k as f32], ..base },
            )
            .expect("the reference renders where the frame did");
            let (frame_luma, _) = downsampled(&frame, 1);
            let (ref_luma, reference) = downsampled(&big, k);
            frames.push(frame_luma);
            refs.push(ref_luma);
            if first.is_none() {
                first = Some((frame, reference));
            }
            eprintln!("shift {}/{shifts} rendered", i + 1);
        }
        let score = aliasing_score(&frames, &refs);
        let moved = change(&frames);
        let moved_ref = change(&refs);

        let stats = |x0: u32, y0: u32, x1: u32, y1: u32| {
            let mut n = 0usize;
            let (mut sum, mut sum_moved, mut sum_ref, mut over4, mut over8) = (0.0f64, 0.0f64, 0.0f64, 0usize, 0usize);
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = (y * w + x) as usize;
                    n += 1;
                    sum += score[p] as f64;
                    sum_moved += moved[p] as f64;
                    sum_ref += moved_ref[p] as f64;
                    over4 += (score[p] > 4.0) as usize;
                    over8 += (score[p] > 8.0) as usize;
                }
            }
            let n = n.max(1) as f64;
            (sum / n, sum_moved / n, sum_ref / n, 100.0 * over4 as f64 / n, 100.0 * over8 as f64 / n)
        };
        let (a, m, r, o4, o8) = stats(0, 0, w, h);
        eprintln!(
            "{tag}: aliasing {a:.3} levels (pixels over 4: {o4:.2}%, over 8: {o8:.2}%); change {m:.3}, reference's own change {r:.3}; K={k}, {shifts} shifts"
        );
        if let Some(region) = std::env::var("REGION").ok() {
            let c: Vec<u32> = region.split(',').map(|x| x.trim().parse().unwrap()).collect();
            let (a, m, r, o4, o8) = stats(c[0], c[1], c[2].min(w), c[3].min(h));
            eprintln!("{tag} region {region}: aliasing {a:.3} (over 4: {o4:.2}%, over 8: {o8:.2}%); change {m:.3}, reference {r:.3}");
        }
        let mut tiles: Vec<(f64, u32, u32)> = Vec::new();
        for ty in 0..h / tile {
            for tx in 0..w / tile {
                let (a, ..) = stats(tx * tile, ty * tile, (tx + 1) * tile, (ty + 1) * tile);
                tiles.push((a, tx * tile, ty * tile));
            }
        }
        tiles.sort_by(|p, q| q.0.total_cmp(&p.0));
        for (a, x, y) in tiles.iter().take(12) {
            let (_, m, r, o4, _) = stats(*x, *y, x + tile, y + tile);
            eprintln!("  tile {x},{y}..{},{}: aliasing {a:.2} (over 4: {o4:.1}%), change {m:.2}, reference {r:.2}", x + tile, y + tile);
        }

        let (frame, reference) = first.unwrap();
        let mut heat = Vec::with_capacity((w * h * 4) as usize);
        for p in 0..(w * h) as usize {
            let dim = (frames[0][p] * 0.3) as u8;
            let hot = (score[p] * 16.0).min(255.0) as u8;
            heat.extend([dim.saturating_add(hot), dim, dim, 255]);
        }
        Shot { width: w, height: h, rgba: heat }.save(&out.join(format!("alias_{tag}.png")));
        frame.save(&out.join(format!("alias_{tag}_frame.png")));
        reference.save(&out.join(format!("alias_{tag}_reference.png")));
    }
}

/// The game folder the offline frames read: `GAME_DIR` when set -- a frozen
/// copy, so two renders compare while the level's bakes change -- else the
/// workspace's.
pub(crate) fn offline_game_dir() -> std::path::PathBuf {
    std::env::var("GAME_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../game"))
}
