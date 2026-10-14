//! A scene's WEATHER for the renderer: its areas' ground found through the
//! physics scene (what the sky meets first at each point, by a ray straight
//! down), stepped with the clock, and handed to `space_soup::renderer::weather`
//! as maps and per-area numbers. See `space_soup_engine::weather` for the
//! physics and the renderer's module for how it is drawn and lit.

use glam::Vec3;
use space_soup::renderer::weather::AreaParams;
#[cfg(test)]
use space_soup::renderer::weather::WeatherMaps;
use space_soup::renderer::water_pipeline::{shore_wet_line, shore_wet_line_most};
use space_soup::renderer::weather::CHUNK_POOLS;
use space_soup_engine::water::WaterDef;
use space_soup_engine::weather::{PrecipitationKind, WeatherField};

/// Where a ray from the sky starts, and how far it may fall.
const SKY_Y: f32 = 400.0;
const SKY_DROP: f32 = 800.0;

/// The scene's areas, their ground from `physics` -- the terrain, the
/// brushes, the caves -- as everything that stops rain.
pub fn field(scene: &space_soup_engine::scene::Scene, physics: &space_soup_engine::rigid_physics::PhysicsWorld) -> WeatherField {
    WeatherField::new(&scene.weather, |x, z| physics.raycast_down(Vec3::new(x, SKY_Y, z), SKY_DROP).map(|(hit, _)| hit.y))
}

/// A LEVEL'S LAYERS OF THE WEATHER'S MAPS, which hold
/// `space_soup_engine::weather::MAX_AREAS` of them: THE SEA'S WASHED SHORE
/// FIRST -- it is one body's and the whole coast's -- then the areas of
/// weather nearest where the player starts, and which areas (by index in the
/// scene) found no layer.
///
/// The areas took the layers first, in the file's order, and the shore what
/// was left: on any level with four areas the sea's wet sand was gone, with
/// nothing said (`docs/scale-audit-2026-10-13.md`, F7: every stress level
/// above test_room). And an area's ground was found -- a ray a texel -- for
/// areas there was no layer for.
pub fn layers(
    scene: &space_soup_engine::scene::Scene,
    physics: &space_soup_engine::rigid_physics::PhysicsWorld,
) -> (WeatherField, Vec<Shore>, Vec<usize>) {
    let from = scene.objects.iter().find(|o| o.spawn_point.is_some()).map_or([0.0, 0.0], |o| [o.cuboid.position.x, o.cuboid.position.z]);
    layers_over(
        &scene.water,
        &scene.weather,
        from,
        |x, z| physics.raycast_down(Vec3::new(x, SKY_Y, z), SKY_DROP).map(|(hit, _)| hit.y),
        |x, z, top, drop| physics.raycast_down(Vec3::new(x, top, z), drop).is_some(),
    )
}

/// [`layers`] over any ground: `cover` the highest thing over a point,
/// `sand` whether there is ground under (`x`, `z`) between `top` and `drop`
/// below it (`Shore::find`'s question).
pub fn layers_over(
    water: &[WaterDef],
    weather: &[space_soup_engine::weather::WeatherDef],
    from: [f32; 2],
    cover: impl Fn(f32, f32) -> Option<f32>,
    sand: impl Fn(f32, f32, f32, f32) -> bool,
) -> (WeatherField, Vec<Shore>, Vec<usize>) {
    let shores: Vec<Shore> = match crate::water_render::sea_line(water) {
        Some(sea) => water
            .iter()
            .filter(|d| d.swash > 0.0 && d.height == sea)
            .filter_map(|d| Shore::find(d, &sand))
            .take(space_soup_engine::weather::MAX_AREAS)
            .collect(),
        None => Vec::new(),
    };
    let (field, dropped) = WeatherField::nearest(weather, cover, from, space_soup_engine::weather::MAX_AREAS.saturating_sub(shores.len()));
    (field, shores, dropped)
}

/// Each area as the GPU takes it.
pub fn area_params(field: &WeatherField) -> Vec<AreaParams> {
    field
        .areas
        .iter()
        .map(|(def, ground, state)| AreaParams {
            min: ground.min,
            extent: [ground.cell * ground.nx as f32, ground.cell * ground.nz as f32],
            snow: def.kind == PrecipitationKind::Snow,
            falling: state.falling,
            wind: def.wind,
            edge: def.edge,
            shore: false,
        })
        .collect()
}

/// A SHORE THE SEA WASHES, as the weather's maps carry it: a coarse layer
/// over the body's footprint holding, at each point, how far above the still
/// surface the wash has the sand wet now -- the WET LINE
/// (`water_pipeline::shore_wet_line`: it jumps up the beach with a wave and
/// creeps back after it, higher where the bigger sections break). The ground's
/// weather twin compares each pixel's own height with it, so the wet edge
/// follows the beach's shape to the pixel while the layer stays a few
/// thousand texels, of which only those with sand the wash can reach are
/// worked out again (`AreaParams::shore`).
#[derive(Debug, Clone)]
pub struct Shore {
    /// The still surface, world y, and the wash's rise, metres.
    pub sea: f32,
    pub swash: f32,
    pub min: [f32; 2],
    /// A texel's size along x and along z.
    pub cell: [f32; 2],
    pub nx: usize,
    pub nz: usize,
    /// The texels with ground the wash can reach in or beside them, each
    /// with its centre.
    active: Vec<(usize, glam::Vec2)>,
}

/// A shore layer's texel, metres, and the most along a side: the line varies
/// along a shore over tens of metres (`water_pipeline::SWASH_SECTION`).
pub const SHORE_CELL: f32 = 2.5;
const SHORE_MAX_CELLS: usize = 64;
/// How often the wet lines are worked out again, seconds: a wave lifts the
/// line a hand's height in a second, and four times a second showed as steps
/// at the water's edge.
pub const SHORE_REFRESH: f64 = 1.0 / 12.0;

impl Shore {
    /// `def`'s shore, if its waves wash one. `ground_within(x, z, top, drop)`
    /// is whether ground lies within `drop` metres under world `(x, top, z)`.
    pub fn find(def: &WaterDef, ground_within: impl Fn(f32, f32, f32, f32) -> bool) -> Option<Self> {
        if def.swash <= 0.0 {
            return None;
        }
        let [a, b, c, d] = def.bounds;
        let (x0, z0, x1, z1) = (a.min(c), b.min(d), a.max(c), b.max(d));
        let cell = SHORE_CELL.max((x1 - x0).max(z1 - z0) / SHORE_MAX_CELLS as f32);
        let nx = (((x1 - x0) / cell).ceil() as usize).clamp(2, SHORE_MAX_CELLS);
        let nz = (((z1 - z0) / cell).ceil() as usize).clamp(2, SHORE_MAX_CELLS);
        let (cx, cz) = ((x1 - x0) / nx as f32, (z1 - z0) / nz as f32);
        // Ground from a hand under the still surface to a hand over the
        // highest the line gets.
        let most = shore_wet_line_most(def.swash);
        let (top, drop) = (def.height + most + 0.3, most + 0.6);
        let mut reached = vec![false; nx * nz];
        for z in 0..nz {
            for x in 0..nx {
                reached[z * nx + x] = (0..25).any(|k| {
                    let (u, v) = ((k % 5) as f32 * 0.25, (k / 5) as f32 * 0.25);
                    ground_within(x0 + (x as f32 + u) * cx, z0 + (z as f32 + v) * cz, top, drop)
                });
            }
        }
        // And the texels beside them: the map is read between its texels.
        let active: Vec<(usize, glam::Vec2)> = (0..nx * nz)
            .filter(|i| {
                let (x, z) = (i % nx, i / nx);
                (z.saturating_sub(1)..=(z + 1).min(nz - 1)).any(|j| (x.saturating_sub(1)..=(x + 1).min(nx - 1)).any(|i| reached[j * nx + i]))
            })
            .map(|i| (i, glam::Vec2::new(x0 + ((i % nx) as f32 + 0.5) * cx, z0 + ((i / nx) as f32 + 0.5) * cz)))
            .collect();
        (!active.is_empty()).then_some(Self { sea: def.height, swash: def.swash, min: [x0, z0], cell: [cx, cz], nx, nz, active })
    }

    pub fn params(&self) -> AreaParams {
        AreaParams {
            min: self.min,
            extent: [self.cell[0] * self.nx as f32, self.cell[1] * self.nz as f32],
            edge: 1e-3,
            shore: true,
            ..Default::default()
        }
    }

    /// The layer at the waves' clock `t`, seconds: each texel the wet line's
    /// height over the still surface, no water, no snow, open sky (the wash
    /// comes in under a roof). Where no sand is in reach, the line at rest.
    pub fn texels(&self, t: f32) -> Vec<[f32; 4]> {
        let mut out = vec![[space_soup::renderer::water_pipeline::SHORE_BAND, 0.0, 0.0, space_soup_engine::weather::NO_COVER]; self.nx * self.nz];
        for (i, xz) in &self.active {
            out[*i][0] = shore_wet_line(t, self.swash, *xz);
        }
        out
    }

    /// Whether ground in the box `lo`..`hi` (world) can be in the wash: what
    /// decides whether a terrain chunk draws the weather's twin for it.
    /// Ground wholly under the still surface, or wholly above the line's
    /// highest, is the same in the dry shader (`terrain_pipeline::SEA_LINE_WGSL`).
    pub fn touches(&self, lo: Vec3, hi: Vec3) -> bool {
        let (x1, z1) = (self.min[0] + self.cell[0] * self.nx as f32, self.min[1] + self.cell[1] * self.nz as f32);
        lo.x <= x1 && hi.x >= self.min[0] && lo.z <= z1 && hi.z >= self.min[1] && hi.y >= self.sea - 0.1 && lo.y <= self.sea + shore_wet_line_most(self.swash) + 0.05
    }

    pub fn active(&self) -> usize {
        self.active.len()
    }
}

/// The scene's washed shores, their ground through `physics`: at most what
/// the maps have room for beside `areas` areas of weather, and only those of
/// the sea the ground's shader knows as the sea ([`crate::water_render::sea_line`]).
pub fn shores(water: &[WaterDef], physics: &space_soup_engine::rigid_physics::PhysicsWorld, areas: usize) -> Vec<Shore> {
    let Some(sea) = crate::water_render::sea_line(water) else { return Vec::new() };
    water
        .iter()
        .filter(|d| d.swash > 0.0 && d.height == sea)
        .filter_map(|d| Shore::find(d, |x, z, top, drop| physics.raycast_down(Vec3::new(x, top, z), drop).is_some()))
        .take(space_soup_engine::weather::MAX_AREAS.saturating_sub(areas))
        .collect()
}

/// Every layer of the weather's maps as the GPU takes it: the shores first,
/// so an area of weather over the same ground is the one read there, then the
/// areas ([`area_params`]).
pub fn layer_params(field: &WeatherField, shores: &[Shore]) -> Vec<AreaParams> {
    shores.iter().map(Shore::params).chain(area_params(field)).collect()
}

/// Each layer's size in texels, in [`layer_params`]' order.
pub fn layer_texels(field: &WeatherField, shores: &[Shore]) -> Vec<(u32, u32)> {
    shores.iter().map(|s| (s.nx as u32, s.nz as u32)).chain(field.areas.iter().map(|(_, g, _)| (g.nx as u32, g.nz as u32))).collect()
}

/// An area's map as it stands (`space_soup_engine::weather::surface`), with
/// what lies under the sea's still surface soaked and holding no puddle: the
/// ground's weather twin has the map where its dry shader has the sea's line
/// (`terrain_pipeline::SEA_LINE_WGSL`).
pub fn area_map(def: &space_soup_engine::weather::WeatherDef, ground: &space_soup_engine::weather::WeatherGround, state: &space_soup_engine::weather::WeatherState, sea: Option<f32>) -> Vec<[f32; 4]> {
    let mut texels = space_soup_engine::weather::surface(def, ground, state);
    if let Some(sea) = sea {
        for (t, cover) in texels.iter_mut().zip(&ground.cover) {
            if *cover < sea {
                t[0] = 1.0;
                t[1] = 0.0;
            }
        }
    }
    texels
}

/// The maps' texture, sized for `field`'s areas and `shores`. (The headset's
/// are the renderer's own: `XrRenderer::set_weather`.)
#[cfg(test)]
pub fn maps(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, field: &WeatherField, shores: &[Shore]) -> WeatherMaps {
    WeatherMaps::new(device, layout, &layer_texels(field, shores))
}

/// Every layer's map and numbers as `field` stands now and the shores at the
/// waves' clock `water_time`, uploaded.
#[cfg(test)]
pub fn upload(queue: &wgpu::Queue, maps: &mut WeatherMaps, field: &WeatherField, shores: &[Shore], water_time: f32) {
    let sea = shores.first().map(|s| s.sea);
    let texels = shores.iter().map(|s| s.texels(water_time)).chain(field.areas.iter().map(|(d, g, st)| area_map(d, g, st, sea)));
    for (k, ((texels, params), size)) in texels.zip(layer_params(field, shores)).zip(layer_texels(field, shores)).enumerate() {
        maps.upload_map(queue, k as u32, size.0, size.1, &texels);
        maps.set_area(k, &params, size);
    }
}

/// The wind where the head is: its area's, or none. (The headset's renderer
/// finds its own, from the areas it is handed.)
#[cfg(test)]
pub fn wind_at(field: &WeatherField, head: Vec3) -> [f32; 2] {
    field
        .areas
        .iter()
        .find(|(d, _, _)| d.contains(head.x, head.z))
        .map(|(d, _, _)| d.wind)
        .unwrap_or([0.0, 0.0])
}

/// Which terrain chunks a weather area touches, by their world bounds: the
/// ones drawn with the ground's weather twins. A chunk that holds any vertex
/// inside an area is one, so the snow's lift never tears a seam between a
/// lifted chunk and its neighbour (the snow is zero at an area's edge).
pub fn chunk_flags(field: &WeatherField, shores: &[Shore], bounds: &[(Vec3, Vec3)]) -> Vec<bool> {
    chunk_areas(field, shores, bounds).iter().map(|m| *m != 0).collect()
}

/// Which layers each chunk touches, as a bit mask in [`layer_params`]' order
/// -- a shore's where the chunk has ground the wash can reach
/// ([`Shore::touches`]), an area's where their footprints meet -- and, from
/// bit `CHUNK_POOLS`, which of those areas can hold standing water in the
/// chunk's own ground (`WeatherGround::can_pool`): what the renderer's
/// weather twin for the chunk must draw (`WeatherScene::chunks`,
/// `WeatherKinds`).
pub fn chunk_areas(field: &WeatherField, shores: &[Shore], bounds: &[(Vec3, Vec3)]) -> Vec<u32> {
    bounds
        .iter()
        .map(|(lo, hi)| {
            let washed = shores.iter().enumerate().fold(0u32, |mask, (k, s)| if s.touches(*lo, *hi) { mask | (1 << k) } else { mask });
            field.areas.iter().enumerate().take(CHUNK_POOLS as usize - shores.len()).fold(washed, |mask, (k, (d, g, _))| {
                let (x0, z0, x1, z1) = d.extent();
                let k = k + shores.len();
                if lo.x <= x1 && hi.x >= x0 && lo.z <= z1 && hi.z >= z0 {
                    mask | (1 << k) | if g.can_pool(d, [lo.x, lo.z], [hi.x, hi.z]) { 1 << (k + CHUNK_POOLS as usize) } else { 0 }
                } else {
                    mask
                }
            })
        })
        .collect()
}

/// The sky's and the sun's light on a drop or a flake, for the particles'
/// `ParticleView`: the sky as a white diffuser takes it, averaged over a
/// tumbling one's faces (half from above, half from round about), and the
/// sun's colour and direction (world). The offline frame's; the headset's
/// renderer has its own (`weather::particle_light`).
#[cfg(test)]
pub fn particle_light(
    sky: &space_soup::renderer::sky::SkyIrradiance,
    sun: Option<&space_soup::renderer::sky::SkySun>,
) -> ([f32; 3], Option<([f32; 3], [f32; 3])>) {
    let up = sky.evaluate([0.0, 1.0, 0.0]);
    let round: Vec<[f32; 3]> = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
        .iter()
        .map(|d| sky.evaluate(*d))
        .collect();
    let mean = |c: usize| 0.5 * up[c] + 0.125 * round.iter().map(|r| r[c]).sum::<f32>();
    (
        [mean(0), mean(1), mean(2)],
        sun.map(|s| (s.light_rgb, s.direction)),
    )
}

/// How often the maps are worked out again and uploaded, seconds. What lies
/// on the ground moves over minutes; four times a second shows it moving and
/// costs the render thread about a tenth of a millisecond each time.
pub const MAP_REFRESH: f64 = 0.25;

/// A level's weather as the headset keeps it: its areas, stepped with the
/// game's clock; the maps worked out again every [`MAP_REFRESH`]; and how wet
/// the player's own body is.
#[derive(Default)]
pub struct WeatherClient {
    pub field: WeatherField,
    /// The shores the sea washes, the maps' first layers.
    pub shores: Vec<Shore>,
    refreshed: Option<f64>,
    shores_refreshed: Option<f64>,
    /// The player's avatar and hands: 0 dry, 1 soaked. See
    /// `space_soup_engine::weather::body_wetness_step`.
    pub body_wet: f32,
}

impl WeatherClient {
    /// `scene_name`'s weather, its ground through `physics`. Empty for a scene
    /// without any (or one that will not load).
    pub fn load(dir: &std::path::Path, scene_name: &str, physics: &space_soup_engine::rigid_physics::PhysicsWorld) -> Self {
        let Ok(scene) = space_soup_engine::scene::Scene::load(&space_soup_engine::Manifest::scene_path(dir, scene_name)) else {
            return Self::default();
        };
        let (field, shores, dropped) = layers(&scene, physics);
        for (d, g, _) in &field.areas {
            log::info!("WEATHER area {:?}: {:?}, {}x{} texels of {:.2} m", d.name, d.kind, g.nx, g.nz, g.cell);
        }
        for s in &shores {
            log::info!("WEATHER shore at y={:.2}: {}x{} texels of {:.2} m, {} with sand in the wash", s.sea, s.nx, s.nz, s.cell[0], s.active());
        }
        if !dropped.is_empty() {
            // LOUD: these areas have no rain, no snow and no wet ground in
            // this build, wherever the player goes.
            log::error!(
                "WEATHER: THE LEVEL HAS {} AREAS OF WEATHER AND {} WASHED SHORE(S); THE MAPS HOLD {} LAYERS. {} area(s) are NOT SHOWN (the ones furthest from where the player starts): {}",
                scene.weather.len(),
                shores.len(),
                space_soup_engine::weather::MAX_AREAS,
                dropped.len(),
                dropped.iter().map(|&i| format!("{:?}", scene.weather[i].name)).collect::<Vec<_>>().join(", "),
            );
        }
        Self { field, shores, refreshed: None, shores_refreshed: None, body_wet: 0.0 }
    }

    /// Each layer's map size, for `XrRenderer::set_weather`.
    pub fn texels(&self) -> Vec<(u32, u32)> {
        layer_texels(&self.field, &self.shores)
    }

    /// The terrain chunks an area touches, by their first index, each with
    /// the areas it touches as a bit mask (`WeatherScene::chunks`).
    pub fn chunk_firsts(&self, terrain: Option<&crate::terrain_render::TerrainGeometry>) -> Vec<(u32, u32)> {
        let Some(t) = terrain else { return Vec::new() };
        let chunks = t.chunk_world_bounds();
        let bounds: Vec<(Vec3, Vec3)> = chunks.iter().map(|(_, b)| *b).collect();
        chunk_areas(&self.field, &self.shores, &bounds).iter().zip(&chunks).filter(|(m, _)| **m != 0).map(|(m, (first, _))| (*first, *m)).collect()
    }

    /// `dt` seconds on, at game time `time`, with the head at `head` (world):
    /// each layer's numbers ([`layer_params`]), and the maps: a layer's when
    /// it is due -- the areas' every [`MAP_REFRESH`], the shores' every
    /// [`SHORE_REFRESH`], on the waves' clock, which is `time` too
    /// (`XrRenderer::set_water_time`) -- and empty for one that is not; no
    /// maps at all when none is.
    pub fn step(&mut self, time: f64, dt: f32, head: Vec3) -> (Vec<AreaParams>, Vec<Vec<[f32; 4]>>) {
        if self.field.is_empty() && self.shores.is_empty() {
            return (Vec::new(), Vec::new());
        }
        self.field.step(time, dt.min(0.25));
        let (rain, snow) = self.field.falling_on(head.to_array());
        self.body_wet = space_soup_engine::weather::body_wetness_step(self.body_wet, rain, snow, dt.min(0.25));
        let due = self.refreshed.is_none_or(|t| time - t >= MAP_REFRESH || time < t);
        let shores_due = !self.shores.is_empty() && self.shores_refreshed.is_none_or(|t| time - t >= SHORE_REFRESH || time < t);
        let maps = if due || shores_due {
            // The waves' clock repeats every loop, and so does the line.
            let loop_s = space_soup::renderer::water_waves::WaveParams::default().loop_seconds as f64;
            let sea = self.shores.first().map(|s| s.sea);
            let shores = self.shores.iter().map(|s| if shores_due { s.texels(time.rem_euclid(loop_s) as f32) } else { Vec::new() });
            let areas = self.field.areas.iter().map(|(d, g, st)| if due { area_map(d, g, st, sea) } else { Vec::new() });
            let maps = shores.chain(areas).collect();
            if due {
                self.refreshed = Some(time);
            }
            if shores_due {
                self.shores_refreshed = Some(time);
            }
            maps
        } else {
            Vec::new()
        };
        (layer_params(&self.field, &self.shores), maps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sea(swash: f32) -> WaterDef {
        serde_json::from_value(serde_json::json!({ "height": 0.0, "bounds": [-20.0, -20.0, 20.0, 20.0], "swash": swash })).unwrap()
    }

    /// A LEVEL WITH MORE WEATHER THAN THE MAPS HOLD KEEPS ITS WASHED SHORE:
    /// seven areas of rain and snow across a level and a sea that washes its
    /// beach. The shore has its layer, the three areas nearest the player's
    /// start have the rest, in the scene's order, and the four that found
    /// none are NAMED, not dropped in silence. (As it was: the first four
    /// areas in the file took every layer, wherever they were, and the sea's
    /// wet sand was gone on every level with four areas or more.)
    #[test]
    fn a_level_with_more_weather_than_the_maps_hold_keeps_its_washed_shore() {
        let area = |k: usize| -> space_soup_engine::weather::WeatherDef {
            let x = k as f32 * 120.0 - 360.0;
            serde_json::from_value(serde_json::json!({ "name": format!("area_{k}"), "bounds": [x, -10.0, x + 30.0, 20.0] })).unwrap()
        };
        let weather: Vec<_> = (0..7).map(area).collect();
        let cover = |_: f32, _: f32| Some(0.0f32);
        // The player starts beside area 5 (x = 240..270), far from the first four.
        let (field, shores, dropped) = layers_over(&[sea(0.3)], &weather, [250.0, 0.0], cover, beach);
        assert_eq!(shores.len(), 1, "the sea's washed shore has its layer");
        let kept: Vec<&str> = field.areas.iter().map(|(d, _, _)| d.name.as_deref().unwrap_or("")).collect();
        assert_eq!(kept, ["area_4", "area_5", "area_6"], "the nearest three, in the scene's order");
        assert_eq!(dropped, [0, 1, 2, 3]);
        assert_eq!(layer_params(&field, &shores).len(), space_soup_engine::weather::MAX_AREAS);
        // The old rule's answer on the same level: every layer to the first
        // four areas, none left for the shore.
        let first_four = WeatherField::new(&weather, cover);
        assert_eq!(first_four.areas.len(), space_soup_engine::weather::MAX_AREAS);
        assert_eq!(space_soup_engine::weather::MAX_AREAS.saturating_sub(first_four.areas.len()), 0);
        // A level that fits loses nothing and names nothing: test_room's two
        // areas and its shore.
        let (field, shores, dropped) = layers_over(&[sea(0.3)], &weather[..2], [250.0, 0.0], cover, beach);
        assert_eq!((field.areas.len(), shores.len(), dropped.len()), (2, 1, 0));
        // No sea: all four layers are the weather's.
        let (field, shores, dropped) = layers_over(&[], &weather, [-350.0, 0.0], cover, beach);
        assert_eq!((field.areas.len(), shores.len(), dropped), (4, 0, vec![4, 5, 6]));
    }

    /// A beach rising a metre in ten along x, the still waterline at x = 0.
    fn beach(x: f32, _z: f32, top: f32, drop: f32) -> bool {
        let y = 0.1 * x;
        y <= top && y > top - drop
    }

    /// A shore's layer: found only for a body whose waves wash a beach; the
    /// texels with sand in the wash's reach carry the wet line, which moves
    /// with the waves between its rest and its most; the rest hold the line
    /// at rest; and only ground the wash can reach makes a chunk the
    /// weather's.
    #[test]
    fn a_shore_carries_the_wet_line_where_the_wash_reaches() {
        use space_soup::renderer::water_pipeline::SHORE_BAND;
        assert!(Shore::find(&sea(0.0), beach).is_none(), "a still lake has no wash");
        let shore = Shore::find(&sea(0.22), beach).expect("a washed beach");
        assert_eq!((shore.nx, shore.nz, shore.sea), (16, 16, 0.0));
        let most = shore_wet_line_most(0.22);
        // The texel columns round the waterline are in reach, far inland
        // (x = 18: the sand 1.8 m up) and far out (x = -18: 1.8 m down) not.
        let column = |x: f32| ((x + 20.0) / shore.cell[0]) as usize;
        let reached: Vec<usize> = (0..shore.nx).filter(|c| shore.active.iter().any(|(i, _)| i % shore.nx == *c)).collect();
        assert!(reached.contains(&column(0.5)) && reached.contains(&column(3.0)), "{reached:?}");
        assert!(!reached.contains(&column(18.0)) && !reached.contains(&column(-18.0)), "{reached:?}");
        let (a, b) = (shore.texels(10.0), shore.texels(13.0));
        let at = |x: f32, z: f32| column(z) * shore.nx + column(x);
        for t in [&a, &b] {
            assert_eq!(t.len(), shore.nx * shore.nz);
            assert!(t.iter().all(|x| x[0] >= SHORE_BAND && x[0] <= most + 1e-5 && x[1] == 0.0 && x[2] == 0.0 && x[3] == space_soup_engine::weather::NO_COVER));
            assert_eq!(t[at(18.0, 0.0)][0], SHORE_BAND, "no sand in reach: the line at rest");
        }
        assert!(reached.iter().any(|c| (0..shore.nz).any(|r| (a[r * shore.nx + c][0] - b[r * shore.nx + c][0]).abs() > 0.01)), "the line did not move in three seconds");
        // Chunks: the beach's is washed; the bed's and the dunes' are not.
        assert!(shore.touches(Vec3::new(-4.0, -0.4, -8.0), Vec3::new(4.0, 0.4, 8.0)));
        assert!(!shore.touches(Vec3::new(-20.0, -2.0, -8.0), Vec3::new(-12.0, -1.2, 8.0)), "wholly under the still water");
        assert!(!shore.touches(Vec3::new(12.0, 1.2, -8.0), Vec3::new(20.0, 2.0, 8.0)), "wholly above the wash");
        assert!(!shore.touches(Vec3::new(40.0, -0.4, 0.0), Vec3::new(48.0, 0.4, 8.0)), "beyond the body");
        let p = shore.params();
        assert!(p.shore && !p.snow && p.falling == 0.0 && p.min == [-20.0, -20.0] && p.extent == [40.0, 40.0]);
    }

    /// The layers' order and the chunks' masks: a shore first, then the areas;
    /// a chunk's bit for each it touches, and its pooling bit only where
    /// water can stand in its own ground; and an area's map soaked, with no
    /// puddle, where its ground lies under the sea.
    #[test]
    fn a_chunk_is_told_its_layers_and_where_water_can_stand() {
        let rain: space_soup_engine::weather::WeatherDef =
            serde_json::from_str(r#"{"bounds": [4, -8, 20, 8], "ground_relief": 0.0, "edge": 0.01, "start": {"wetness": 0.5, "puddles": 1.0}}"#).unwrap();
        // The beach, with a bowl a metre across at (14, 4) dug into it, deep
        // enough to be closed on its downhill side.
        let ground = |x: f32, z: f32| Some(0.1 * x - if (x - 14.0).hypot(z - 4.0) < 0.6 { 0.2 } else { 0.0 });
        let field = WeatherField::new(&[rain], ground);
        let shores = vec![Shore::find(&sea(0.22), beach).unwrap()];
        let params = layer_params(&field, &shores);
        assert_eq!(params.len(), 2);
        assert!(params[0].shore && !params[1].shore);
        assert_eq!(layer_texels(&field, &shores), vec![(16, 16), (64, 64)]);
        let masks = chunk_areas(
            &field,
            &shores,
            &[
                (Vec3::new(-4.0, -0.4, -8.0), Vec3::new(4.0, 0.4, 8.0)),   // the waterline, and the rain's edge
                (Vec3::new(12.0, 1.1, 0.0), Vec3::new(20.0, 2.0, 8.0)),    // the rain, with the bowl
                (Vec3::new(12.0, 1.2, -8.0), Vec3::new(20.0, 2.0, 0.0)),   // the rain, no hollow
                (Vec3::new(-20.0, -2.0, -8.0), Vec3::new(-12.0, -1.2, 8.0)), // the sea bed
            ],
        );
        let pools = 1u32 << (1 + CHUNK_POOLS);
        assert_eq!(masks, vec![0b11, 0b10 | pools, 0b10, 0]);
        assert_eq!(chunk_flags(&field, &shores, &[(Vec3::splat(100.0), Vec3::splat(101.0))]), vec![false]);
        // An area whose ground dips under a sea at y = 0.6 (x < 6): soaked
        // there, no puddle; as the weather has it above.
        let (def, g, state) = &field.areas[0];
        let (dry, wet) = (area_map(def, g, state, None), area_map(def, g, state, Some(0.6)));
        let texel = |x: f32, z: f32| ((z + 8.0) / g.cell) as usize * g.nx + ((x - 4.0) / g.cell) as usize;
        assert!(dry[texel(4.6, 0.0)][0] < 0.6 && wet[texel(4.6, 0.0)][0] == 1.0 && wet[texel(4.6, 0.0)][1] == 0.0);
        assert_eq!(dry[texel(10.0, 0.0)], wet[texel(10.0, 0.0)]);
        // The bowl holds water at its downhill end, and nowhere else does.
        assert!(dry[texel(13.6, 4.0)][1] > 0.0 && dry.iter().filter(|t| t[1] > 0.0).count() < 30, "{}", dry[texel(13.6, 4.0)][1]);
    }

    /// MEASUREMENT: each demo area's map at `WEATHER_TIME` (600 s), as a
    /// coarse picture and its coverage -- puddles (p), snow (s).
    #[test]
    #[ignore]
    fn print_the_demo_maps() {
        let game = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../game");
        let scene = space_soup_engine::scene::Scene::load(&space_soup_engine::Manifest::scene_path(&game, "test_room")).unwrap();
        let mut physics = space_soup_engine::rigid_physics::PhysicsWorld::new();
        physics.rebuild(&scene, &game);
        let mut f = field(&scene, &physics);
        let t: f64 = std::env::var("WEATHER_TIME").ok().and_then(|v| v.parse().ok()).unwrap_or(600.0);
        f.set_time(t);
        for (d, g, st) in &f.areas {
            let m = space_soup_engine::weather::surface(d, g, st);
            let wet = m.iter().filter(|t| t[1] > 0.002).count() as f32 / m.len() as f32;
            let snow = m.iter().filter(|t| t[2] > 0.02).count() as f32 / m.len() as f32;
            let deepest = m.iter().map(|t| t[1]).fold(0.0f32, f32::max);
            eprintln!("{:?} at {t} s {st:?}: puddles over {:.1}%, deepest {deepest:.3} m; snow over 2 cm on {:.1}%", d.name, 100.0 * wet, 100.0 * snow);
            for z in (0..g.nz).step_by(4) {
                let row: String = (0..g.nx)
                    .step_by(2)
                    .map(|x| {
                        let t = m[z * g.nx + x];
                        if t[1] > 0.01 { 'P' } else if t[1] > 0.002 { 'p' } else if t[2] > 0.08 { 'S' } else if t[2] > 0.02 { 's' } else if t[0] > 0.5 { ':' } else { '.' }
                    })
                    .collect();
                eprintln!("{row}");
            }
        }
    }

    /// THE SHIPPED DEMO AREAS, through the shipped level's physics: the cover
    /// is the ground's height in the open, and the scene's areas load.
    #[test]
    fn test_rooms_weather_areas_find_their_ground() {
        let game = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../game");
        let Ok(scene) = space_soup_engine::scene::Scene::load(&space_soup_engine::Manifest::scene_path(&game, "test_room")) else {
            eprintln!("skipping: no test_room");
            return;
        };
        if scene.weather.is_empty() {
            eprintln!("skipping: test_room has no weather");
            return;
        }
        let mut physics = space_soup_engine::rigid_physics::PhysicsWorld::new();
        physics.rebuild(&scene, &game);
        let f = field(&scene, &physics);
        assert_eq!(f.areas.len(), scene.weather.len().min(space_soup_engine::weather::MAX_AREAS));
        for (def, ground, _) in &f.areas {
            let (x0, z0, x1, z1) = def.extent();
            let mid = ground.cover_at(0.5 * (x0 + x1), 0.5 * (z0 + z1)).expect("inside");
            assert!(mid > -10.0 && mid < 15.0, "{:?}: the ground under its middle at {mid}", def.name);
        }
        // The brick hall's roof stops rain: a ray from the sky over the hall
        // meets the roof, 3.4 m up.
        let roof = physics.raycast_down(Vec3::new(14.0, SKY_Y, -2.0), SKY_DROP).map(|(h, _)| h.y);
        eprintln!("cover over the brick hall: {roof:?}");
    }
}
