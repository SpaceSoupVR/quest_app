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
use space_soup::renderer::lights::{Light, LightsUniform};
use space_soup::renderer::shadow::ShadowMap;
use space_soup::renderer::uniforms::{PlayerUpload, PostUpload, ShadowUpload, SkyUpload, UniformBuffer};

use crate::brush_render::{load_materials, BrushGeometry};

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
}

impl View {
    pub fn headset(eye: Vec3, at: Vec3) -> Self {
        Self { eye, at, fov_y: 104.0, width: EYE_W, height: EYE_H, samples: 4, sources: false, adapt: false, no_portals: false }
    }
}

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
}

/// Render `scene_name` from `view`. `None` when no GPU is available.
pub fn render_brushes(scene_name: &str, view: View) -> Option<Shot> {
    let (device, queue) = device()?;
    let game = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../game");
    let scene = space_soup_engine::scene::Scene::load(&space_soup_engine::Manifest::scene_path(&game, scene_name)).ok()?;
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;

    // THE PLAYER'S FRAME, as on the headset: the rig stands under the eye.
    let offset = Vec3::new(view.eye.x, 0.0, view.eye.z);
    let yaw_inv = Quat::IDENTITY;
    let to_player = |p: Vec3| p - offset;

    // Lights: the scene's realtime ones through the wire conversion, plus the
    // sky's sun exactly as the frame adds it.
    let pano = scene.sky.as_ref().and_then(|s| {
        let bytes = std::fs::read(game.join("skies").join(&s.id).join("sky.hdr")).ok()?;
        let p = space_soup::renderer::sky::decode_radiance(&bytes).ok()?;
        Some(space_soup::renderer::sky::sky_lighting(&p, s.rotation_deg, s.intensity))
    });
    let (irradiance, sky_sun) = pano.unwrap_or((
        space_soup::renderer::sky::SkyIrradiance::flat(space_soup::renderer::sky::AMBIENT),
        None,
    ));
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
    let sun = space_soup::renderer::lights::sky_sun_light(sky_sun.as_ref(), &lights, yaw_inv.inverse());
    lights.extend(sun);
    let lights = space_soup::renderer::lights::rank_for_budget(&lights, space_soup::renderer::lights::MAX_LIGHTS);
    let lights_uniform = LightsUniform::new(&device);
    lights_uniform.upload_frame(&queue, &lights, &[], sun.is_some());

    // Probes, as `set_reflection_probes` binds them.
    // The same description, rooms and doorways the headset builds.
    let level = crate::probe_level::ProbeLevel::load(&game, scene_name);
    let resolution = level.as_ref().map(|l| l.resolution).unwrap_or(1);
    let owned: Vec<Vec<u8>> = level
        .as_ref()
        .map(|l| (0..l.descs.len()).filter_map(|i| (l.source())(i)).collect())
        .unwrap_or_default();
    let faces: Vec<&[u8]> = owned.iter().map(|f| f.as_slice()).collect();
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
    let proxies = level.as_ref().map(|l| l.scene_proxies(&game, scene_name)).unwrap_or_default();
    let brightness: Vec<f32> = owned
        .iter()
        .map(|f| space_soup::renderer::uniforms::probe_mean_radiance(f, resolution))
        .collect();

    let shadows = ShadowMap::with_dimension(&device, 64);
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
    let view_proj = proj * Mat4::look_at_rh(eye_p, at_p, Vec3::Y);
    let mut probes = space_soup::renderer::uniforms::select_resident_probes(&volumes, view.eye, |_, _| true);
    probes.fill_brightness(&brightness);
    probes.fill_volumes(&rooms);
    let resident_rooms = probes.volumes();
    probes.set_portals(&portals, view.eye, &resident_rooms);
    probes.set_proxies(&proxies, view.eye, &resident_rooms);
    if view.no_portals {
        probes.portal_count = 0;
    }
    uniforms.upload_scene_with_probes(
        &queue,
        view_proj,
        eye_p,
        &ShadowUpload::disabled(),
        &SkyUpload::from(&irradiance),
        &PostUpload {
            exposure: if view.adapt {
                let (irr, _) = (irradiance, ());
                let loaded: Vec<(&[u8], u32, Vec3, Vec3, Vec3)> = owned
                    .iter()
                    .zip(&descs)
                    .map(|(f, d)| (f.as_slice(), resolution, d.centre, d.min, d.max))
                    .collect();
                let eye = space_soup::renderer::exposure::EyeAdaptation::from_probes(&loaded, irr);
                let m = eye.meter(view.eye, view.at - view.eye);
                let e = space_soup::renderer::exposure::exposure_for(m);
                eprintln!("offline frame: metered {m:.4}, exposure x{e:.2}");
                e
            } else {
                1.0
            },
            ..PostUpload::default()
        },
        &PlayerUpload { offset, yaw: 0.0 },
        Some(&probes),
    );

    let pipeline = if view.sources {
        BrushPipeline::new_multisampled_sources(&device, format, &uniforms.layout, view.samples)
    } else {
        BrushPipeline::new_multisampled(&device, format, &uniforms.layout, view.samples)
    };

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
    let sun_mask = pick(space_soup_engine::lightmaps::SCENE_BRUSH_SUN_MASK_ID);
    let stationary_layers: Vec<&[u8]> = stationary.iter().map(|m| m.rgba.as_slice()).collect();
    let lightmap = space_soup::renderer::mesh::create_lightmap_texture_full(
        &device,
        &queue,
        &pipeline.lightmap_layout,
        match &base.linear {
            Some(l) => space_soup::renderer::mesh::LightmapLight::Linear(l),
            None => space_soup::renderer::mesh::LightmapLight::Srgb8(&base.rgba),
        },
        base.width,
        base.height,
        dir.map(|d| (d.rgba.as_slice(), d.width, d.height)),
        sun_mask.map(|d| (d.rgba.as_slice(), d.width, d.height)),
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
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, &uniforms.bind_group, &[]);
        pass.set_bind_group(1, &materials.bind_group, &[]);
        pass.set_bind_group(2, &lightmap.bind_group, &[]);
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..idx.len() as u32, 0, 0..1);
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
            _ => View::headset(Vec3::new(0.3, 1.6, -3.0), Vec3::new(0.0, 0.9, -7.0)),
        };
        let Some(shot) = render_brushes("test_room", View { adapt: true, ..v }) else {
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
        let game = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../game");
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
        let game = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../game");
        let scene = space_soup_engine::scene::Scene::load(&space_soup_engine::Manifest::scene_path(&game, "test_room")).ok()?;
        let s = scene.sky.as_ref()?;
        let bytes = std::fs::read(game.join("skies").join(&s.id).join("sky.hdr")).ok()?;
        let pano = space_soup::renderer::sky::decode_radiance(&bytes).ok()?;
        let (sky, _) = space_soup::renderer::sky::sky_lighting(&pano, s.rotation_deg, s.intensity);
        let loaded = space_soup_engine::reflection_probe::load_scene_probes(&game, "test_room");
        let probes: Vec<(&[u8], u32, Vec3, Vec3, Vec3)> = loaded
            .iter()
            .map(|p| (p.faces.as_slice(), p.resolution, Vec3::from(p.centre), Vec3::from(p.min), Vec3::from(p.max)))
            .collect();
        Some(EyeAdaptation::from_probes(&probes, sky))
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
        ] {
            let m = eye.meter(at, look);
            eprintln!("{name:>28}: meter {m:.4}  exposure {:.2}", exposure_for(m));
        }
    }
}
