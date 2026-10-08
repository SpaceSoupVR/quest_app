//! A scene's WEATHER for the renderer: its areas' ground found through the
//! physics scene (what the sky meets first at each point, by a ray straight
//! down), stepped with the clock, and handed to `space_soup::renderer::weather`
//! as maps and per-area numbers. See `space_soup_engine::weather` for the
//! physics and the renderer's module for how it is drawn and lit.

use glam::Vec3;
use space_soup::renderer::weather::AreaParams;
#[cfg(test)]
use space_soup::renderer::weather::WeatherMaps;
use space_soup_engine::weather::{PrecipitationKind, WeatherField};

/// Where a ray from the sky starts, and how far it may fall.
const SKY_Y: f32 = 400.0;
const SKY_DROP: f32 = 800.0;

/// The scene's areas, their ground from `physics` -- the terrain, the
/// brushes, the caves -- as everything that stops rain.
pub fn field(scene: &space_soup_engine::scene::Scene, physics: &space_soup_engine::rigid_physics::PhysicsWorld) -> WeatherField {
    WeatherField::new(&scene.weather, |x, z| physics.raycast_down(Vec3::new(x, SKY_Y, z), SKY_DROP).map(|(hit, _)| hit.y))
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
        })
        .collect()
}

/// The maps' texture, sized for `field`'s areas. (The headset's are the
/// renderer's own: `XrRenderer::set_weather`.)
#[cfg(test)]
pub fn maps(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, field: &WeatherField) -> WeatherMaps {
    let sizes: Vec<(u32, u32)> = field.areas.iter().map(|(_, g, _)| (g.nx as u32, g.nz as u32)).collect();
    WeatherMaps::new(device, layout, &sizes)
}

/// Every area's map and numbers as `field` stands now, uploaded.
#[cfg(test)]
pub fn upload(queue: &wgpu::Queue, maps: &mut WeatherMaps, field: &WeatherField) {
    for (k, ((def, ground, state), params)) in field.areas.iter().zip(area_params(field)).enumerate() {
        let texels = space_soup_engine::weather::surface(def, ground, state);
        maps.upload_map(queue, k as u32, ground.nx as u32, ground.nz as u32, &texels);
        maps.set_area(k, &params, (ground.nx as u32, ground.nz as u32));
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
pub fn chunk_flags(field: &WeatherField, bounds: &[(Vec3, Vec3)]) -> Vec<bool> {
    chunk_areas(field, bounds).iter().map(|m| *m != 0).collect()
}

/// Which areas each chunk touches, as a bit mask by the areas' order: what
/// the renderer's weather twin for the chunk must draw (`WeatherKinds`).
pub fn chunk_areas(field: &WeatherField, bounds: &[(Vec3, Vec3)]) -> Vec<u32> {
    bounds
        .iter()
        .map(|(lo, hi)| {
            field.areas.iter().enumerate().take(32).fold(0u32, |mask, (k, (d, _, _))| {
                let (x0, z0, x1, z1) = d.extent();
                if lo.x <= x1 && hi.x >= x0 && lo.z <= z1 && hi.z >= z0 { mask | (1 << k) } else { mask }
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
    refreshed: Option<f64>,
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
        let field = field(&scene, physics);
        for (d, g, _) in &field.areas {
            log::info!("WEATHER area {:?}: {:?}, {}x{} texels of {:.2} m", d.name, d.kind, g.nx, g.nz, g.cell);
        }
        Self { field, refreshed: None, body_wet: 0.0 }
    }

    /// Each area's map size, for `XrRenderer::set_weather`.
    pub fn texels(&self) -> Vec<(u32, u32)> {
        self.field.areas.iter().map(|(_, g, _)| (g.nx as u32, g.nz as u32)).collect()
    }

    /// The terrain chunks an area touches, by their first index, each with
    /// the areas it touches as a bit mask (`WeatherScene::chunks`).
    pub fn chunk_firsts(&self, terrain: Option<&crate::terrain_render::TerrainGeometry>) -> Vec<(u32, u32)> {
        let Some(t) = terrain else { return Vec::new() };
        let chunks = t.chunk_world_bounds();
        let bounds: Vec<(Vec3, Vec3)> = chunks.iter().map(|(_, b)| *b).collect();
        chunk_areas(&self.field, &bounds).iter().zip(&chunks).filter(|(m, _)| **m != 0).map(|(m, (first, _))| (*first, *m)).collect()
    }

    /// `dt` seconds on, at game time `time`, with the head at `head` (world):
    /// each area's numbers, and the maps when they are due (else empty).
    pub fn step(&mut self, time: f64, dt: f32, head: Vec3) -> (Vec<AreaParams>, Vec<Vec<[f32; 4]>>) {
        if self.field.is_empty() {
            return (Vec::new(), Vec::new());
        }
        self.field.step(time, dt.min(0.25));
        let (rain, snow) = self.field.falling_on(head.to_array());
        self.body_wet = space_soup_engine::weather::body_wetness_step(self.body_wet, rain, snow, dt.min(0.25));
        let due = self.refreshed.is_none_or(|t| time - t >= MAP_REFRESH || time < t);
        let maps = if due {
            self.refreshed = Some(time);
            self.field.areas.iter().map(|(d, g, st)| space_soup_engine::weather::surface(d, g, st)).collect()
        } else {
            Vec::new()
        };
        (area_params(&self.field), maps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
