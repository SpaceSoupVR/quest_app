//! Terrain geometry for the on-device renderer.
//!
//! The samples are static scene data that is ALREADY on the headset -- run.sh
//! pushes game/ wholesale -- so the client loads them itself rather than
//! receiving them over the wire. Terrain therefore costs nothing per snapshot,
//! which matters when the target is 64 players and the snapshot budget is the
//! binding constraint.
//!
//! Built once per scene load. A heightfield does not change at runtime yet, and
//! when it does (craters), the right shape is to re-cook the affected patch
//! rather than to stream vertices every frame.

use glam::{Quat, Vec3};
use space_soup::renderer::cuboid::SolidVertex;
use space_soup::renderer::ground_lod::{self, GroundLod};
use space_soup_engine::brush::{contains_point, BrushDef, BrushSolid};
use space_soup_engine::terrain::TerrainDef;

/// How far below the terrain surface to look for a structure that buries it.
///
/// BELOW, not above. `BrushDef::evaluate` yields a building's SOLID pieces --
/// slab, walls, ceiling -- and not the carved room, so a point just above the
/// ground inside a room is open air and would find nothing. Five centimetres
/// down lands inside a floor slab resting on the ground, and inside nothing at
/// all under a structure raised off it, like a bridge.
pub const BURIED_PROBE_DEPTH: f32 = 0.05;

/// How far a buried terrain vertex is lowered, in the RENDERED copy only.
///
/// A floor laid on flattened ground is coplanar with it: `test_room` measures
/// the terrain 0.1 mm under the room floor, well inside depth precision, so the
/// two z-fight and grass flickers through the floor. It only showed with SSR
/// off, because the SSR eye pass redraws every brush over the finished scene
/// and paints the fight over.
///
/// Physics is untouched: the terrain stays the ground you collide with.
pub const BURIED_DROP: f32 = 0.02;

/// How finely a candidate triangle is sampled for coverage: a barycentric grid
/// of `(n + 1)(n + 2) / 2` points (28 at 6). At the 0.86 m terrain spacing that
/// is a sample every ~14 cm, finer than any wall.
const COVERAGE_SAMPLES: u32 = 6;

/// What `bury_under_structures` changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Burial {
    pub hidden_triangles: usize,
    pub lowered_vertices: usize,
}

/// Render-only treatment of terrain covered by structure floors.
///
/// A triangle whose three vertices are ALL buried, and which is covered across
/// its whole area, is dropped: drawing it is pure wasted fill and the source of
/// the z-fight. (Three buried corners alone are not enough at an inside corner;
/// see the note in the body.)
/// A triangle that crosses a wall line cannot be dropped without opening a gap
/// outside the wall, so it is kept and only its buried vertices are lowered by
/// `BURIED_DROP` -- enough to put the part under the room well below the floor,
/// while the dip outside is hidden by the wall's foot below ground.
pub fn bury_under_structures(
    positions: &mut [Vec3],
    indices: &[u32],
    solids: &[BrushSolid],
) -> (Vec<u32>, Burial) {
    let (kept, burial, _) = bury_indexed(positions, indices, &SolidIndex::new(solids));
    (kept, burial)
}

/// THE LEVEL'S BRUSH SOLIDS BY WHERE THEY STAND, so "is this point inside a
/// solid" asks the few solids over that spot and not every solid of the
/// level.
///
/// `bury_under_structures` asked every ground vertex against every solid:
/// samples x solids, both of which grow with the level's area -- 13 ms on
/// test_room, 210 ms at 625 samples a side, 2.5 s at 1,025, 39 s at 2,049
/// (`docs/scale-audit-2026-10-13.md`, L1). Nearly all of the ground is under
/// no building at all.
///
/// A grid over the solids' footprints (x, z), each cell listing the solids
/// whose box reaches it. A point outside the grid, or in an empty cell, is
/// answered without a plane test; in a cell, by the solids listed there whose
/// box holds it. THE SAME ANSWER as asking them all (a solid can only hold a
/// point its own box holds; the boxes are widened by more than the test's
/// tolerance; a solid with no box -- unbounded, degenerate -- is asked every
/// time, as before): `the_indexed_burial_is_the_burial_asked_of_every_solid`.
///
/// WHAT IT COSTS NOW: the ground's vertices, once each, plus the plane tests
/// under buildings -- linear in the level's samples.
pub struct SolidIndex<'a> {
    solids: &'a [BrushSolid],
    /// Each solid's box, widened; `None` for one without.
    boxes: Vec<Option<([f64; 3], [f64; 3])>>,
    /// Solids with no box: asked for every point.
    boxless: Vec<u32>,
    origin: [f64; 2],
    cell: f64,
    cells_x: usize,
    cells_z: usize,
    cells: Vec<Vec<u32>>,
    /// Plane-set tests made (`contains_point` calls): what the index is for.
    tests: std::cell::Cell<u64>,
}

impl<'a> SolidIndex<'a> {
    /// Wider than `contains_point`'s tolerance here (1e-4), so no solid is
    /// missed at its own surface.
    const SLACK: f64 = 1e-3;
    /// Past this a face's extent is the CSG's seed polygon showing through
    /// (`brush::HUGE`, 1e5, over root three for a tilted face): the solid is
    /// open that way and is asked for every point instead.
    const BOUNDED: f64 = 2.5e4;

    pub fn new(solids: &'a [BrushSolid]) -> Self {
        let boxes: Vec<Option<([f64; 3], [f64; 3])>> = solids
            .iter()
            .map(|s| {
                // A solid open on some side has faces that run to the size
                // the CSG seeds a face at (100 km): its "box" is not one.
                space_soup_engine::brush::solid_bounds(s)
                    .filter(|(lo, hi)| lo.iter().chain(hi.iter()).all(|c| c.is_finite() && c.abs() < Self::BOUNDED))
                    .map(|(lo, hi)| (lo.map(|c| c - Self::SLACK), hi.map(|c| c + Self::SLACK)))
            })
            .collect();
        let boxless: Vec<u32> = boxes.iter().enumerate().filter(|(_, b)| b.is_none()).map(|(i, _)| i as u32).collect();
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for (l, h) in boxes.iter().flatten() {
            lo = [lo[0].min(l[0]), lo[1].min(l[2])];
            hi = [hi[0].max(h[0]), hi[1].max(h[2])];
        }
        let boxed = boxes.iter().flatten().count();
        if boxed == 0 {
            return Self { solids, boxes, boxless, origin: [0.0; 2], cell: 1.0, cells_x: 0, cells_z: 0, cells: Vec::new(), tests: Default::default() };
        }
        // About as many cells as solids, and no cell under 4 m: a building's
        // solids share a few cells, an open level's cells are mostly empty.
        let side = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1.0);
        let per_axis = ((boxed as f64).sqrt().ceil() as usize * 2).clamp(1, 512);
        let cell = (side / per_axis as f64).max(4.0);
        let cells_x = (((hi[0] - lo[0]) / cell).floor() as usize + 1).max(1);
        let cells_z = (((hi[1] - lo[1]) / cell).floor() as usize + 1).max(1);
        let mut cells = vec![Vec::new(); cells_x * cells_z];
        for (i, b) in boxes.iter().enumerate() {
            let Some((l, h)) = b else { continue };
            let at = |v: f64, o: f64, n: usize| (((v - o) / cell).floor().max(0.0) as usize).min(n - 1);
            for cz in at(l[2], lo[1], cells_z)..=at(h[2], lo[1], cells_z) {
                for cx in at(l[0], lo[0], cells_x)..=at(h[0], lo[0], cells_x) {
                    cells[cz * cells_x + cx].push(i as u32);
                }
            }
        }
        Self { solids, boxes, boxless, origin: lo, cell, cells_x, cells_z, cells, tests: Default::default() }
    }

    /// Whether any solid holds `p`, within `tolerance`.
    pub fn holds(&self, p: [f64; 3], tolerance: f64) -> bool {
        let inside = |i: u32| {
            self.tests.set(self.tests.get() + 1);
            contains_point(&self.solids[i as usize], p, tolerance)
        };
        if self.boxless.iter().any(|&i| inside(i)) {
            return true;
        }
        let (fx, fz) = ((p[0] - self.origin[0]) / self.cell, (p[2] - self.origin[1]) / self.cell);
        if !(fx >= 0.0 && fz >= 0.0) || fx as usize >= self.cells_x || fz as usize >= self.cells_z {
            return false;
        }
        self.cells[fz as usize * self.cells_x + fx as usize].iter().any(|&i| {
            let (lo, hi) = self.boxes[i as usize].expect("only boxed solids are listed");
            (0..3).all(|k| p[k] >= lo[k] && p[k] <= hi[k]) && inside(i)
        })
    }

    /// Plane-set tests made so far.
    pub fn tests(&self) -> u64 {
        self.tests.get()
    }
}

/// [`bury_under_structures`] through an index of the solids; also how many
/// plane-set tests it took.
fn bury_indexed(positions: &mut [Vec3], indices: &[u32], solids: &SolidIndex) -> (Vec<u32>, Burial, u64) {
    let before = solids.tests();
    let covered = |q: Vec3| solids.holds([q.x as f64, (q.y - BURIED_PROBE_DEPTH) as f64, q.z as f64], 1e-4);
    let buried: Vec<bool> = positions.iter().map(|p| covered(*p)).collect();
    // COVERED ACROSS ITS WHOLE AREA, not just at its corners. Three buried
    // corners prove a triangle covered only under a CONVEX footprint. Where the
    // hallway meets a hall the footprint is an L, and a triangle with a corner
    // under each building spans the open ground in the inside corner between
    // them: hiding it opened a hole in the grass (headset, 2026-09-24). So a
    // triangle whose corners are all buried is also sampled across its area,
    // and kept -- lowered, like any triangle crossing a wall -- if any sample
    // is open ground. Only triangles already buried at every corner pay for it.
    let fully_covered = |a: Vec3, b: Vec3, c: Vec3| {
        (0..=COVERAGE_SAMPLES).all(|i| {
            (0..=(COVERAGE_SAMPLES - i)).all(|j| {
                let (u, v) = (i as f32 / COVERAGE_SAMPLES as f32, j as f32 / COVERAGE_SAMPLES as f32);
                covered(a + (b - a) * u + (c - a) * v)
            })
        })
    };
    let mut kept = Vec::with_capacity(indices.len());
    let mut burial = Burial::default();
    for tri in indices.chunks_exact(3) {
        let [a, b, c] = [tri[0], tri[1], tri[2]].map(|i| positions[i as usize]);
        if tri.iter().all(|&i| buried[i as usize]) && fully_covered(a, b, c) {
            burial.hidden_triangles += 1;
        } else {
            kept.extend_from_slice(tri);
        }
    }
    for (p, b) in positions.iter_mut().zip(&buried) {
        if *b {
            p.y -= BURIED_DROP;
            burial.lowered_vertices += 1;
        }
    }
    (kept, burial, solids.tests() - before)
}

/// Ground vertices and indices in the renderer's own format.
///
/// `vertices` are WORLD space, as loaded. `assemble` produces the player-frame
/// copy for a renderer that does not pose the ground itself.
pub struct TerrainGeometry {
    pub vertices: Vec<SolidVertex>,
    /// THE GROUND'S INDEX BUFFER AS `ground_lod` LAYS IT: every chunk's
    /// step-1 triangles, the chunks 32 cells a side and in Morton order of
    /// their grid (the first [`Self::step1_len`] indices -- all a frame draws
    /// with `Levers::ground_lod` off), then the coarser lists where a level's
    /// load built them (`ground_lod::LISTS`).
    ///
    /// The order is what makes culling possible at all: a chunk has to be
    /// drawable as a single `draw_indexed` range, and a row-major grid
    /// scatters any square patch across the whole buffer.
    pub indices: Vec<u32>,
    /// Player-frame copy, rebuilt only when the player has actually moved.
    posed: Vec<SolidVertex>,
    built_for: Option<(Vec3, f32)>,
    /// The chunks, the quadtree over them and the lists (its indices are
    /// `indices` above): what a frame asks which chunks it draws, and at
    /// which step (`GroundLod::frame`). The renderer is handed a copy
    /// (`XrRenderer::set_ground_lod`).
    pub lod: GroundLod,
    /// Every run of `indices` the slope split sorts, its box in the WORLD:
    /// the same list every frame ([`Self::caster_chunks`]).
    units: Vec<space_soup::renderer::shadow::CasterChunk>,
    /// THE GROUND'S STEP-1 SHAPE, a texel a sample (`ground_shape`): what
    /// its readers take the normal and the face from, so a coarser step is
    /// lit as step 1 is. `None` for a ground that is not a lattice of samples
    /// at step 1.
    pub shape: Option<space_soup::renderer::ground_shape::GroundShape>,
}

impl TerrainGeometry {
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() || self.indices.is_empty()
    }

    /// How many of `indices`, from the first, are the ground at step 1 --
    /// "the ground's triangles" for anything that is not a draw (the
    /// horizon's occluder, a census): past them lie other lists of the same
    /// ground.
    pub fn step1_len(&self) -> usize {
        (self.lod.level0_len() as usize).min(self.indices.len())
    }

    /// The ground's triangles at step 1.
    pub fn step1(&self) -> &[u32] {
        &self.indices[..self.step1_len()]
    }

    /// Ground in the player's frame, the way every other kind of geometry is
    /// handed to the renderer.
    ///
    /// This was missing, and the symptom pointed at the wrong thing entirely.
    /// Brushes were transformed and terrain was passed through in world space,
    /// so walking left the ground glued to the player while the room slid past
    /// -- which reads as "the room brush is moving", not as "the ground is
    /// stuck". Terrain was the only geometry in the scene without a frame.
    ///
    /// Cached on (offset, yaw) exactly like `BrushGeometry::assemble`: standing
    /// still costs nothing, and a heightfield is static so there is never a
    /// re-mesh, only a rotate and a subtract per vertex.
    ///
    /// (The chunks' boxes are not made here any more: the renderer poses the
    /// boxes of the chunks a frame draws, from their boxes in the world --
    /// `GroundLod::posed`.)
    pub fn assemble(
        &mut self,
        offset: Vec3,
        yaw_inv: Quat,
        player_yaw: f32,
    ) -> Option<(&[SolidVertex], &[u32])> {
        if self.is_empty() {
            return None;
        }
        let key = (offset, player_yaw);
        if self.built_for != Some(key) {
            self.posed.clear();
            self.posed.reserve(self.vertices.len());
            for v in &self.vertices {
                let p = yaw_inv * (Vec3::from(v.position) - offset);
                // The normal is rotated but NOT translated -- it is a
                // direction. Subtracting the offset from it too would tilt the
                // ground's lighting by however far the player had walked.
                let n = yaw_inv * Vec3::from(v.normal);
                let mut out = *v;
                out.position = p.to_array();
                out.normal = n.to_array();
                self.posed.push(out);
            }
            self.built_for = Some(key);
        }
        Some((&self.posed, &self.indices))
    }

    /// THE GROUND AS LOADED, IN THE WORLD, for a renderer that poses it
    /// itself (`XrRenderer::ground_in_world`): nothing is touched and
    /// nothing follows the rig here -- not a vertex, and (since 2026-10-13)
    /// not a chunk's box either: every chunk's was posed on each frame the
    /// rig moved, 64 of them on any level then, 4,096 at 2,049 samples with
    /// chunks of a fixed size. The frame poses the boxes of the chunks it
    /// draws (`GroundLod::posed`).
    pub fn assemble_world(&mut self, _offset: Vec3, _yaw_inv: Quat, _player_yaw: f32) -> Option<(&[SolidVertex], &[u32])> {
        if self.is_empty() {
            return None;
        }
        // The posed copy is not kept beside the world's.
        if !self.posed.is_empty() {
            self.posed = Vec::new();
            self.built_for = None;
        }
        Some((&self.vertices, &self.indices))
    }

    /// The chunks a shadow pass may cull against a light's frustum.
    ///
    /// Empty until `assemble` has run, because the bounds are in the player's
    /// frame and there is no frame before then.
    /// World-space bounds of the terrain, from the unposed vertices.
    ///
    /// The UNPOSED ones deliberately: this is the footprint the baked occlusion
    /// map was generated over, which is fixed in the world and does not move
    /// with the player.
    /// THE TERRAIN'S HEIGHTS for the renderer's ground map
    /// (`space_soup::renderer::ground_map`): the heightfield itself -- not this
    /// render copy, which is decimated and has the ground under floors lowered
    /// -- sampled over this copy's footprint, which is where the terrain's own
    /// baked map is placed (`XrRenderer::set_terrain_footprint`).
    pub fn height_grid(&self, def: Option<&TerrainDef>, game_dir: &std::path::Path) -> Option<space_soup::renderer::ground_map::HeightGrid> {
        let source = space_soup_engine::terrain::load(def?, game_dir).ok()?;
        let (lo, hi) = self.world_bounds();
        if !(hi.x > lo.x && hi.z > lo.z) {
            return None;
        }
        Some(space_soup::renderer::ground_map::HeightGrid::sample(
            glam::Vec2::new(lo.x, lo.z),
            glam::Vec2::new(hi.x, hi.z),
            HEIGHT_GRID_SAMPLES,
            HEIGHT_GRID_SAMPLES,
            // A HOLE in the ground (a cave mouth) is no ground: far below
            // anything, so a reflected ray passes on through it.
            |x, z| source.height_at(x, z).unwrap_or(-1000.0),
        ))
    }

    pub fn world_bounds(&self) -> (Vec3, Vec3) {
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for v in &self.vertices {
            let p = Vec3::from(v.position);
            lo = lo.min(p);
            hi = hi.max(p);
        }
        (lo, hi)
    }

    /// Each run's first index and its WORLD bounds (its chunk's, or its
    /// node's for a coarser list): what a weather area is tested against
    /// (`weather_render::chunk_flags`) and the horizon is asked with, entry
    /// for entry with [`Self::caster_chunks`].
    pub fn chunk_world_bounds(&self) -> Vec<(u32, (Vec3, Vec3))> {
        self.units.iter().map(|c| (c.first_index, (c.min, c.max))).collect()
    }

    /// THE DRAWN GROUND AS WHAT IT HIDES, for the renderer's horizon test
    /// (`space_soup::renderer::ground_horizon`): from these triangles
    /// themselves, in the world -- the ground under floors left out or
    /// lowered, the caves' mouths open, the patch round them in -- over the
    /// heightfield's own cells, so every whole cell is two of its triangles.
    pub fn ground_occluder(&self, def: Option<&TerrainDef>) -> Option<space_soup::renderer::ground_horizon::GroundOccluder> {
        let cells = match def.map(|d| &d.kind) {
            Some(space_soup_engine::terrain::TerrainKind::Heightfield { resolution, .. }) => {
                [resolution[0].saturating_sub(1).max(1), resolution[1].saturating_sub(1).max(1)]
            }
            None => [256, 256],
        };
        let positions: Vec<[f32; 3]> = self.vertices.iter().map(|v| v.position).collect();
        // (Step 1's triangles: the coarser lists are the same ground again.)
        space_soup::renderer::ground_horizon::GroundOccluder::from_triangles(&positions, self.step1(), cells)
    }

    /// The index runs of the chunks the ground hides from an eye at `eye` in
    /// the WORLD, and the horizon itself for asking after anything else: what
    /// the headset leaves out under `Levers::ground_horizon`, for the offline
    /// frame to leave out too.
    /// `caves`: the baked caves that plug the ground's holes
    /// ([`plugging_caves`]), or none for the first rule alone.
    pub fn behind_the_ground(
        &self,
        def: Option<&TerrainDef>,
        eye: Vec3,
        caves: &[(Vec3, Vec3)],
    ) -> Option<(Vec<(u32, u32)>, space_soup::renderer::ground_horizon::GroundOccluder, space_soup::renderer::ground_horizon::Horizon)> {
        use space_soup::renderer::ground_horizon::{Horizon, REACH};
        let ground = self.ground_occluder(def)?;
        let mut horizon = Horizon::new(&ground, eye, REACH, caves);
        // The chunks at step 1: the first runs of the buffer.
        let step1 = self.step1_len() as u32;
        let runs = self
            .units
            .iter()
            .take_while(|c| c.first_index < step1)
            .filter(|c| horizon.hides_ground(&ground, c.min, c.max))
            .map(|c| (c.first_index, c.index_count))
            .collect();
        Some((runs, ground, horizon))
    }

    /// EVERY run of the index buffer as a chunk -- each chunk at step 1,
    /// then the coarser lists' -- with its box in the WORLD: what the slope
    /// split is built over (`XrRenderer::prepare_ground`) and a frame hands
    /// the renderer. The same slice every frame; which of them a frame draws,
    /// and their boxes in the player's frame, come from [`Self::lod`].
    pub fn caster_chunks(&self) -> &[space_soup::renderer::shadow::CasterChunk] {
        &self.units
    }

    /// WHAT A FRAME HANDS THE RENDERER: the vertices (the world's own where
    /// `world`, posed for the rig otherwise), the index buffer and every run
    /// of it as a chunk -- three borrows of one ground, so the list of runs
    /// is not copied each frame (tens of thousands on a large level with the
    /// coarser lists built).
    pub fn frame(
        &mut self,
        world: bool,
        offset: Vec3,
        yaw_inv: Quat,
        player_yaw: f32,
    ) -> Option<(&[SolidVertex], &[u32], &[space_soup::renderer::shadow::CasterChunk])> {
        if world {
            self.assemble_world(offset, yaw_inv, player_yaw)?;
            Some((&self.vertices, &self.indices, &self.units))
        } else {
            self.assemble(offset, yaw_inv, player_yaw)?;
            Some((&self.posed, &self.indices, &self.units))
        }
    }

    /// A ground from its parts, cut into chunks as [`load`] cuts it.
    fn from_parts(vertices: Vec<SolidVertex>, mut lod: GroundLod) -> Self {
        let units = lod.all_units();
        let indices = std::mem::take(&mut lod.indices);
        Self { vertices, indices, posed: Vec::new(), built_for: None, lod, units, shape: None }
    }
}

/// THE CAVES THAT PLUG THE GROUND'S HOLES, as WORLD boxes, for the horizon's
/// second rule (`space_soup::renderer::ground_horizon`): every model of the
/// scene that stands in for ground, when each is a baked cave -- a closed
/// sheet joined to the ground along its mouths. Empty when any is of the
/// older kind, whose edges were left open: then nothing is said of what lies
/// under the ground. What the headset reads off the frame's own draws; here
/// for the offline frame and the tests.
pub fn plugging_caves(scene: &space_soup_engine::scene::Scene, game_dir: &std::path::Path) -> Vec<(Vec3, Vec3)> {
    let mut boxes = Vec::new();
    for obj in scene.objects.iter().filter(|o| !o.hidden && o.terrain_collider.is_some()) {
        let Some(mesh) = &obj.mesh else { continue };
        let Some(parts) = space_soup_engine::mesh_lightmap::load_mesh_parts(&game_dir.join(&mesh.path)) else { return Vec::new() };
        if parts.iter().any(|p| p.authored.is_none()) {
            return Vec::new();
        }
        let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for p in parts.iter().flat_map(|p| p.positions.iter()) {
            let w = obj.cuboid.rotation * (Vec3::from(*p) * mesh.scale) + obj.cuboid.position;
            lo = lo.min(w);
            hi = hi.max(w);
        }
        boxes.push((lo, hi));
    }
    boxes
}

/// Load a scene's terrain and convert it for the solid pipeline.
///
/// Returns `None` rather than failing the frame when the asset is missing: a
/// level that renders without ground is diagnosable from the log, and a client
/// that refuses to start is not.
pub fn load(
    def: &TerrainDef,
    game_dir: &std::path::Path,
    step: u32,
    // Every brush in the level. Terrain buried under their floors is hidden or
    // lowered in the rendered copy; see `bury_under_structures`.
    structures: &[&BrushDef],
) -> Option<TerrainGeometry> {
    // The coarser lists when the launch asked for them (`ground_lod::LISTS`).
    load_with(def, game_dir, step, structures, ground_lod::LISTS.load(std::sync::atomic::Ordering::Relaxed))
}

/// [`load`], with the ground's coarser lists built or not as `lists` says.
pub fn load_with(def: &TerrainDef, game_dir: &std::path::Path, step: u32, structures: &[&BrushDef], lists: bool) -> Option<TerrainGeometry> {
    let source = match space_soup_engine::terrain::load(def, game_dir) {
        Ok(s) => s,
        Err(e) => {
            log::warn!("terrain_render: {e}");
            return None;
        }
    };

    let bounds = source.bounds();
    let patch = source.patch(bounds, step.max(1));
    if patch.positions.is_empty() || patch.indices.is_empty() {
        log::warn!("terrain_render: terrain produced no geometry");
        return None;
    }

    let normals = ground_normals(&patch);

    // Normalised position over the footprint, for sampling the splat map. Taken
    // from `bounds` rather than from the TerrainDef's declared size so it stays
    // correct for any TerrainSource -- the trait is what a future voxel or
    // chunked representation implements, and the def's shape is not.
    //
    // Guarded against a zero-extent axis: a degenerate terrain would otherwise
    // produce NaN uvs, and a NaN texture coordinate does not error, it just
    // samples something arbitrary and paints the ground with it.
    let extent = bounds.max - bounds.min;
    let inv_x = if extent.x.abs() > 1e-6 { 1.0 / extent.x } else { 0.0 };
    let inv_z = if extent.z.abs() > 1e-6 { 1.0 / extent.z } else { 0.0 };
    // Normals come from the ground as authored, before any vertex is lowered:
    // two centimetres under a floor must not tilt the lighting of the ground
    // beside the wall.
    let solids: Vec<BrushSolid> = structures.iter().flat_map(|b| b.evaluate()).collect();
    let mut positions = patch.positions.clone();
    let (kept_indices, burial) = bury_under_structures(&mut positions, &patch.indices, &solids);
    if burial.hidden_triangles > 0 || burial.lowered_vertices > 0 {
        log::info!(
            "terrain_render: {} triangle(s) under structure floors hidden, {} vertex(es) lowered {} m (render only)",
            burial.hidden_triangles,
            burial.lowered_vertices,
            BURIED_DROP,
        );
    }
    let vertices: Vec<SolidVertex> = positions
        .iter()
        .zip(normals.iter())
        .map(|(p, n)| SolidVertex {
            position: [p.x, p.y, p.z],
            normal: [n.x, n.y, n.z],
            color: GROUND_TINT,
            // Terrain is lit dynamically and has no lightmap, so the slot the
            // cuboid path uses for atlas coordinates is free. It carries the
            // splat uv instead -- one less vertex attribute, and the two can
            // never both be wanted on the same draw.
            uv2: [
                (p.x - bounds.min.x) * inv_x,
                (p.z - bounds.min.z) * inv_z,
            ],
            reflectivity: 0.0,
        })
        .collect();

    log::info!(
        "terrain_render: {} vertices, {} triangles (step {step})",
        patch.positions.len(),
        patch.indices.len() / 3
    );
    // CHUNKS OF A FIXED SIZE, 32 cells a side (13.75 m on test_room's cells,
    // 8 a side there as before; 20 a side at 625 samples, 64 at 2,049, where
    // "8 a side whatever the size" made each chunk 110 m of ground): small
    // enough that a 4 m lamp touches one or two, and that a frustum or the
    // ground's own horizon leaves most of a large level out. In Morton order
    // of the chunk grid, with a quadtree over them, and -- where the load
    // was asked for them -- the coarser lists. See `ground_lod`.
    // (MEASUREMENT: another count a side from `debug.spacesoup.chunks`.)
    let asked = space_soup::renderer::ground_horizon::CHUNKS_A_SIDE.load(std::sync::atomic::Ordering::Relaxed);
    let lod = match source.as_heightfield().filter(|_| step.max(1) == 1) {
        Some(field) => {
            let cells = field.meshed_cells();
            let chunk_cells = if asked > 0 { cells[0].max(cells[1]).div_ceil(asked).max(1) } else { ground_lod::CHUNK_CELLS };
            let open = |ix: u32, iz: u32| field.cell_is_open(ix, iz);
            GroundLod::build(&ground_lod::GridInput {
                cells,
                positions: &positions,
                triangles: &kept_indices,
                open: field.holes().map(|_| &open as &dyn Fn(u32, u32) -> bool),
                chunk_cells,
                lists: asked == 0 && lists,
            })
        }
        // Not a grid of samples at step 1: the whole of it as one chunk.
        None => GroundLod::build(&ground_lod::GridInput { cells: [0, 0], positions: &positions, triangles: &kept_indices, open: None, chunk_cells: 1, lists: false }),
    };
    let lod = if lod.units.is_empty() { GroundLod::single(&positions, &kept_indices) } else { lod };
    // THE STEP-1 SHAPE the readers light every step by: the lattice's
    // normals (as authored, before burial) and drawn heights, and each
    // cell's diagonal as drawn. A cave's ground patch (the vertices after the
    // lattice) is marked by its alpha, 0: its triangles keep their own.
    let shape = source.as_heightfield().filter(|_| step.max(1) == 1).and_then(|field| {
        let cells = field.meshed_cells();
        let uv_cells = [extent.x, extent.z];
        let cell_x = cells[0].max(1) as f32;
        let cell_z = cells[1].max(1) as f32;
        let first = positions.first()?;
        let span_x = positions.get(cells[0] as usize)?.x - first.x;
        let span_z = positions.get((cells[1] * (cells[0] + 1)) as usize)?.z - first.z;
        let (cx, cz) = (span_x / cell_x, span_z / cell_z);
        if cx <= 0.0 || cz <= 0.0 {
            return None;
        }
        space_soup::renderer::ground_shape::GroundShape::build(cells, [uv_cells[0] / cx, uv_cells[1] / cz], &positions, &normals, &kept_indices)
    });
    let mut vertices = vertices;
    if let Some(s) = &shape {
        for v in vertices.iter_mut().skip((s.size[0] * s.size[1]) as usize) {
            v.color[3] = 0.0;
        }
        log::info!("terrain_render: the ground's shape, {} x {} samples ({:.1} MB), lights every step as step 1", s.size[0], s.size[1], s.bytes().len() as f64 / 1e6);
    }
    log::info!(
        "terrain_render: {} chunk(s) of {} cells a side ({} x {}) over {} triangles{}",
        lod.units.iter().filter(|u| u.first < lod.level0_len()).count(),
        if asked > 0 { "a count asked for".to_string() } else { ground_lod::CHUNK_CELLS.to_string() },
        lod.chunks()[0],
        lod.chunks()[1],
        lod.level0_len() / 3,
        if lod.has_lists() { format!("; with the coarser lists {} indices in {} runs", lod.indices.len(), lod.units.len()) } else { String::new() },
    );
    let mut geometry = TerrainGeometry::from_parts(vertices, lod);
    geometry.shape = shape;
    Some(geometry)
}

/// Area-weighted vertex normals from the triangles.
///
/// Computed here rather than taken from the heightfield gradient because this
/// runs on whatever `patch` produced: at a coarse LOD step the triangles are not
/// the sample grid any more, and shading them by the fine-grained gradient would
/// light a surface that is not the one being drawn.
/// The ground's normal at every vertex of a patch: computed from its
/// triangles, except WHERE THE GROUND WAS AUTHORED WITH ONE -- the ground a
/// cave bake put back in its holes (`terrain::GroundPatch`). Along the carve
/// that is the normal the cave's own rock has at the same vertex, so the light
/// does not step from one draw to the other. A vertex the patch shares with
/// the grid has none, and keeps the one computed from the triangles of both.
fn ground_normals(patch: &space_soup_engine::terrain::TerrainPatch) -> Vec<Vec3> {
    let mut normals = vertex_normals(&patch.positions, &patch.indices);
    for (n, authored) in normals.iter_mut().zip(&patch.normals) {
        if let Some(a) = authored.try_normalize() {
            *n = a;
        }
    }
    normals
}

fn vertex_normals(positions: &[Vec3], indices: &[u32]) -> Vec<Vec3> {
    let mut normals = vec![Vec3::ZERO; positions.len()];
    for tri in indices.chunks_exact(3) {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        // Not normalised: the cross product's magnitude is twice the triangle
        // area, which is exactly the weighting a shared vertex should get.
        let face = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
        normals[a] += face;
        normals[b] += face;
        normals[c] += face;
    }
    for n in &mut normals {
        *n = n.normalize_or_zero();
        if *n == Vec3::ZERO {
            *n = Vec3::Y;
        }
    }
    normals
}

/// THE VERTEX TINT: none. The ground's colour is its layers' textures, blended
/// by slope, height or the painted splat map in the shader.
///
/// This was a grass-green (0.30, 0.38, 0.22) on flat ground turning grey-brown
/// on slopes -- the ground's whole colour when terrain was first drawn, before
/// it had textures. The shader multiplies its lit result by the vertex colour,
/// so once textured layers arrived the tint stayed on top of them: the grass
/// texture's albedo of about 0.2 came out near 0.07, three times darker than
/// the textures, the lightmap bake, the probes, eye adaptation and the Cycles
/// reference all took it to be. The probes photographed the ground three times
/// brighter than it was drawn, and every exterior wall reflected that brighter
/// ground (headset, 2026-09-28).
const GROUND_TINT: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

/// Samples a side of the height grid the ground map is built from: finer than
/// test_room's 129-sample heightfield, which it interpolates, so the grid adds
/// no error of its own.
const HEIGHT_GRID_SAMPLES: u32 = 257;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_ground_normals_point_up() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
        ];
        // Counter-clockwise seen from above, matching patch().
        let indices = vec![0, 2, 1, 1, 2, 3];
        for n in vertex_normals(&positions, &indices) {
            assert!((n.y - 1.0).abs() < 1e-5, "expected up, got {n:?}");
        }
    }

    #[test]
    fn a_vertex_with_no_triangles_still_gets_a_usable_normal() {
        // normalize_or_zero would otherwise hand the shader a zero vector.
        let positions = vec![Vec3::ZERO, Vec3::X, Vec3::Z, Vec3::new(9.0, 0.0, 9.0)];
        let normals = vertex_normals(&positions, &[0, 2, 1]);
        assert_eq!(normals[3], Vec3::Y);
    }

    /// The ground is the colour its textures say, not darker: the shader
    /// multiplies by the vertex colour, so anything but white here scales every
    /// layer's albedo. See `GROUND_TINT`.
    #[test]
    fn the_ground_carries_no_tint_over_its_textures() {
        assert_eq!(GROUND_TINT, [1.0; 4]);
    }

    /// The splat uv must span the whole footprint, corner to corner. A uv that
    /// only ever reaches 0.5 samples a quarter of the map over the whole level
    /// and looks like a painting mistake rather than a coordinate bug.
    #[test]
    fn splat_uvs_span_the_footprint() {
        use space_soup_engine::terrain::{Heightfield, TerrainSource};

        // 5x5 samples over 40x40 metres, offset so a bug that forgets the
        // origin cannot pass by accident.
        let field = Heightfield::new(
            vec![0u16; 25],
            [5, 5],
            [40.0, 40.0],
            [0.0, 10.0],
            Vec3::new(100.0, 0.0, -60.0),
        )
        .expect("build test heightfield");

        let bounds = field.bounds();
        let patch = field.patch(bounds, 1);
        let extent = bounds.max - bounds.min;

        let uvs: Vec<[f32; 2]> = patch
            .positions
            .iter()
            .map(|p| [
                (p.x - bounds.min.x) / extent.x,
                (p.z - bounds.min.z) / extent.z,
            ])
            .collect();

        let min_u = uvs.iter().map(|c| c[0]).fold(f32::INFINITY, f32::min);
        let max_u = uvs.iter().map(|c| c[0]).fold(f32::NEG_INFINITY, f32::max);
        let min_v = uvs.iter().map(|c| c[1]).fold(f32::INFINITY, f32::min);
        let max_v = uvs.iter().map(|c| c[1]).fold(f32::NEG_INFINITY, f32::max);

        assert!(min_u.abs() < 1e-5, "u should reach 0, got {min_u}");
        assert!((max_u - 1.0).abs() < 1e-5, "u should reach 1, got {max_u}");
        assert!(min_v.abs() < 1e-5, "v should reach 0, got {min_v}");
        assert!((max_v - 1.0).abs() < 1e-5, "v should reach 1, got {max_v}");

        for uv in &uvs {
            assert!(uv[0].is_finite() && uv[1].is_finite(), "uv must be finite: {uv:?}");
        }
    }
}

#[cfg(test)]
mod player_frame_tests {
    use super::*;

    fn geometry(positions: &[[f32; 3]]) -> TerrainGeometry {
        TerrainGeometry {
            vertices: positions
                .iter()
                .map(|p| SolidVertex {
                    position: *p,
                    normal: [0.0, 1.0, 0.0],
                    color: [1.0, 1.0, 1.0, 1.0],
                    uv2: [0.0, 0.0],
                    reflectivity: 0.0,
                })
                .collect(),
            indices: vec![0, 1, 2],
            posed: Vec::new(),
            built_for: None,
            lod: GroundLod::default(),
            units: Vec::new(),
            shape: None,
        }
    }

    #[test]
    fn walking_moves_the_ground_past_the_player() {
        // The bug this fixes. Terrain was handed to the renderer in world space
        // while everything else was transformed, so the ground travelled with
        // the player and the room appeared to slide instead.
        let mut g = geometry(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
        let at_origin = g
            .assemble(Vec3::ZERO, Quat::IDENTITY, 0.0)
            .map(|(v, _)| v[0].position)
            .unwrap();
        let walked = g
            .assemble(Vec3::new(0.0, 0.0, 5.0), Quat::IDENTITY, 0.0)
            .map(|(v, _)| v[0].position)
            .unwrap();
        assert_eq!(at_origin, [0.0, 0.0, 0.0]);
        assert_eq!(walked, [0.0, 0.0, -5.0], "ground must fall behind the player");
    }

    #[test]
    fn turning_rotates_the_ground_about_the_player() {
        let mut g = geometry(&[[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
        let turned = g
            .assemble(
                Vec3::ZERO,
                Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
                std::f32::consts::FRAC_PI_2,
            )
            .map(|(v, _)| v[0].position)
            .unwrap();
        assert!((turned[0] - -1.0).abs() < 1e-5, "got {turned:?}");
        assert!(turned[2].abs() < 1e-5, "got {turned:?}");
    }

    #[test]
    fn normals_are_rotated_but_never_translated() {
        // A normal is a direction. Subtracting the player's position from it
        // would tilt the ground's lighting by however far they had walked --
        // the ground would get darker the further you went.
        let mut g = geometry(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
        let n = g
            .assemble(Vec3::new(12.0, 3.0, -7.0), Quat::IDENTITY, 0.0)
            .map(|(v, _)| v[0].normal)
            .unwrap();
        assert_eq!(n, [0.0, 1.0, 0.0]);
    }

    /// Handed over in the world, no vertex is posed and nothing is kept
    /// beside the world's copy; and a chunk's box as the frame poses it
    /// (`GroundLod::posed`, from its box in the world) holds all its
    /// vertices posed -- at any turn and any place of the rig.
    #[test]
    fn handed_over_in_the_world_nothing_is_posed_and_a_drawn_chunks_box_holds_it_at_any_turn() {
        // A bumpy 72 x 72 cell patch: nine chunks of 32 cells (the last a part).
        let cells = 72u32;
        let n = cells + 1;
        let mut positions = Vec::new();
        for z in 0..n {
            for x in 0..n {
                let h = ((x * 7 + z * 13) % 11) as f32 * 0.37 + (x as f32 * 0.9).sin() * 2.0;
                positions.push(Vec3::new(x as f32 * 1.5 - 9.0, h, z as f32 * 1.5 + 4.0));
            }
        }
        let mut indices = Vec::new();
        for z in 0..cells {
            for x in 0..cells {
                let a = z * n + x;
                indices.extend_from_slice(&[a, a + n, a + 1, a + 1, a + n, a + n + 1]);
            }
        }
        let lod = GroundLod::build(&ground_lod::GridInput { cells: [cells, cells], positions: &positions, triangles: &indices, open: None, chunk_cells: ground_lod::CHUNK_CELLS, lists: false });
        let vertices: Vec<SolidVertex> = positions.iter().map(|p| SolidVertex { position: p.to_array(), normal: [0.0, 1.0, 0.0], color: [1.0; 4], uv2: [0.0; 2], reflectivity: 0.0 }).collect();
        let mut world = TerrainGeometry::from_parts(vertices, lod);
        assert_eq!(world.caster_chunks().len(), 9);
        assert_eq!(world.step1_len(), indices.len());
        for k in 0..200 {
            let yaw = k as f32 * 0.173;
            let offset = Vec3::new(k as f32 * 0.31 - 20.0, (k % 7) as f32 * 0.4, 3.0 - k as f32 * 0.11);
            let yaw_inv = Quat::from_rotation_y(yaw).inverse();
            let (v, i) = world.assemble_world(offset, yaw_inv, yaw).unwrap();
            assert_eq!(v.len(), positions.len());
            assert!(v.iter().zip(&positions).all(|(v, p)| Vec3::from(v.position) == *p), "a vertex was touched");
            assert_eq!(i.len(), indices.len());
            let frame = world.lod.frame(Vec3::ZERO, &[], false, offset, yaw, 0);
            assert_eq!(frame.chunks.len(), 9);
            for c in &frame.chunks {
                for v in &world.indices[c.first_index as usize..(c.first_index + c.index_count) as usize] {
                    let p = yaw_inv * (positions[*v as usize] - offset);
                    assert!(p.cmpge(c.min).all() && p.cmple(c.max).all(), "turn {k}: a vertex outside its chunk's box");
                }
            }
        }
        assert!(world.posed.is_empty());
        // The posed way still poses every vertex, and going back drops the copy.
        world.assemble(Vec3::ONE, Quat::IDENTITY, 0.0);
        assert_eq!(world.posed.len(), positions.len());
        world.assemble_world(Vec3::ONE, Quat::IDENTITY, 0.0);
        assert!(world.posed.is_empty());
    }

    #[test]
    fn standing_still_reuses_the_previous_build() {
        // 16k vertices rebuilt every frame would be real cost for no change.
        let mut g = geometry(&[[1.0, 2.0, 3.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
        g.assemble(Vec3::ONE, Quat::IDENTITY, 0.0);
        let first = g.built_for;
        g.assemble(Vec3::ONE, Quat::IDENTITY, 0.0);
        assert_eq!(first, g.built_for);
        assert_eq!(g.posed.len(), 3);
    }

    #[test]
    fn empty_terrain_asks_for_no_draw() {
        let mut g = TerrainGeometry {
            vertices: Vec::new(),
            indices: Vec::new(),
            posed: Vec::new(),
            built_for: None,
            lod: GroundLod::default(),
            units: Vec::new(),
            shape: None,
        };
        assert!(g.assemble(Vec3::ZERO, Quat::IDENTITY, 0.0).is_none());
    }
}


#[cfg(test)]
mod burial_tests {
    use super::*;
    use space_soup_engine::brush::block_solid;

    /// An n x n grid of vertices at height `y`, spacing 1 m, centred on the origin.
    fn flat(n: u32, y: f32) -> (Vec<Vec3>, Vec<u32>) {
        let half = (n - 1) as f32 * 0.5;
        let mut p = Vec::new();
        for iz in 0..n {
            for ix in 0..n {
                p.push(Vec3::new(ix as f32 - half, y, iz as f32 - half));
            }
        }
        let mut idx = Vec::new();
        for iz in 0..n - 1 {
            for ix in 0..n - 1 {
                let a = iz * n + ix;
                let (b, c, d) = (a + 1, a + n, a + n + 1);
                idx.extend([a, c, b, b, c, d]);
            }
        }
        (p, idx)
    }

    /// The burial as it was: every vertex and every sample against every
    /// solid of the level.
    fn bury_asking_every_solid(positions: &mut [Vec3], indices: &[u32], solids: &[BrushSolid]) -> (Vec<u32>, Burial, u64) {
        let tests = std::cell::Cell::new(0u64);
        let covered = |q: Vec3| {
            let probe = [q.x as f64, (q.y - BURIED_PROBE_DEPTH) as f64, q.z as f64];
            solids.iter().any(|s| {
                tests.set(tests.get() + 1);
                contains_point(s, probe, 1e-4)
            })
        };
        let buried: Vec<bool> = positions.iter().map(|p| covered(*p)).collect();
        let mut kept = Vec::new();
        let mut burial = Burial::default();
        for tri in indices.chunks_exact(3) {
            let [a, b, c] = [tri[0], tri[1], tri[2]].map(|i| positions[i as usize]);
            let whole = (0..=COVERAGE_SAMPLES).all(|i| {
                (0..=(COVERAGE_SAMPLES - i)).all(|j| covered(a + (b - a) * (i as f32 / COVERAGE_SAMPLES as f32) + (c - a) * (j as f32 / COVERAGE_SAMPLES as f32)))
            });
            if tri.iter().all(|&i| buried[i as usize]) && whole {
                burial.hidden_triangles += 1;
            } else {
                kept.extend_from_slice(tri);
            }
        }
        for (p, b) in positions.iter_mut().zip(&buried) {
            if *b {
                p.y -= BURIED_DROP;
                burial.lowered_vertices += 1;
            }
        }
        (kept, burial, tests.get())
    }

    /// A LEVEL LARGER THAN test_room: 121 x 121 samples of rolling ground and
    /// 144 buildings on it -- floors of every size, some overlapping, some
    /// L-shaped by two solids, some raised off the ground, some sunk into a
    /// hillside, one tilted. THE INDEXED BURIAL IS THE BURIAL ASKED OF EVERY
    /// SOLID: the same triangles kept, the same vertices lowered, bit for
    /// bit. And it does not ask every solid: under a tenth of the plane
    /// tests here, where the old way's count is vertices x solids.
    #[test]
    fn the_indexed_burial_is_the_burial_asked_of_every_solid() {
        let n = 121u32;
        let (mut positions, indices) = flat(n, 0.0);
        for p in positions.iter_mut() {
            p.x *= 3.0;
            p.z *= 3.0;
            p.y = 0.6 * ((p.x * 0.13).sin() + (p.z * 0.11).cos());
        }
        let (lo, hi) = positions.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(l, h), p| (l.min(*p), h.max(*p)));
        let mut solids: Vec<BrushSolid> = Vec::new();
        for k in 0..144u32 {
            let (i, j) = ((k % 12) as f64, (k / 12) as f64);
            let x = lo.x as f64 + (hi.x - lo.x) as f64 * (i + 0.5) / 12.0 + (k as f64 * 0.37).sin() * 3.0;
            let z = lo.z as f64 + (hi.z - lo.z) as f64 * (j + 0.5) / 12.0 + (k as f64 * 0.53).cos() * 3.0;
            let (w, d) = (2.0 + (k % 5) as f64 * 1.7, 1.5 + (k % 7) as f64 * 1.3);
            let lift = if k % 9 == 4 { 2.0 } else { -0.4 - (k % 3) as f64 * 0.5 };
            solids.push(block_solid([x - w, lift, z - d], [x + w, lift + 1.2, z + d], "m"));
            if k % 4 == 1 {
                // An L: a second solid off one corner.
                solids.push(block_solid([x + w - 0.5, lift, z + d - 0.5], [x + w + 4.0, lift + 1.2, z + d + 6.0], "m"));
            }
        }
        // One whose faces are not axis planes: a wedge cut off a block.
        let wedge = block_solid([-6.0, -1.0, -6.0], [6.0, 1.5, 6.0], "m");
        let (cut, _) = space_soup_engine::brush::split_solid(&wedge, space_soup_engine::brush::Plane::new([0.6, 0.0, 0.8], 1.0));
        solids.extend(cut);

        let mut asked = positions.clone();
        let (kept_all, burial_all, tests_all) = bury_asking_every_solid(&mut asked, &indices, &solids);
        let mut indexed = positions.clone();
        let (kept, burial, tests) = bury_indexed(&mut indexed, &indices, &SolidIndex::new(&solids));
        assert!(burial_all.hidden_triangles > 500 && burial_all.lowered_vertices > 500, "the level buries ground: {burial_all:?}");
        assert_eq!(burial, burial_all);
        assert_eq!(kept, kept_all);
        assert!(asked.iter().zip(&indexed).all(|(a, b)| a.to_array() == b.to_array()));
        // What the index is for. (As it was: every vertex against every solid.)
        assert!(tests_all >= positions.len() as u64 * solids.len() as u64 / 2, "{tests_all}");
        assert!(tests * 10 < tests_all, "{tests} plane-set tests through the index, {tests_all} asking every solid");
        // And the public entry is the indexed one.
        let mut again = positions.clone();
        assert_eq!(bury_under_structures(&mut again, &indices, &solids), (kept, burial));
    }

    /// A solid with no box of its own -- too few faces to close -- is still
    /// asked for every point, as before; and no solid at all buries nothing.
    #[test]
    fn a_solid_without_a_box_is_still_asked_and_no_solids_bury_nothing() {
        let (positions, indices) = flat(9, 0.0);
        let mut none = positions.clone();
        let (kept, burial) = bury_under_structures(&mut none, &indices, &[]);
        assert_eq!((kept.len(), burial), (indices.len(), Burial::default()));
        // A half space: everything under y = 1 is inside it.
        let block = block_solid([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], "m");
        let half = BrushSolid { faces: block.faces.iter().filter(|f| f.plane().n[1] > 0.5).cloned().collect() };
        // (Its one face's polygon is the CSG's seed quad, flat in y: a box
        // that holds none of the ground under it. Taken as a box, it would
        // bury nothing.)
        let index = SolidIndex::new(std::slice::from_ref(&half));
        assert_eq!((index.boxless.len(), index.cells.iter().map(Vec::len).sum::<usize>()), (1, 0));
        let mut asked = positions.clone();
        let (kept_all, burial_all, _) = bury_asking_every_solid(&mut asked, &indices, std::slice::from_ref(&half));
        let mut indexed = positions.clone();
        let (kept, burial) = bury_under_structures(&mut indexed, &indices, std::slice::from_ref(&half));
        assert_eq!((kept, burial), (kept_all, burial_all));
        assert_eq!(burial.lowered_vertices, positions.len());
    }

    #[test]
    fn terrain_under_a_floor_is_hidden_and_the_edge_is_lowered() {
        let (mut p, idx) = flat(9, -0.0001);
        let before = p.clone();
        let slab = block_solid([-2.0, -0.3, -2.0], [2.0, 0.0, 2.0], "m");
        let (kept, burial) = bury_under_structures(&mut p, &idx, &[slab]);
        assert!(burial.hidden_triangles > 0, "nothing under the floor was hidden");
        assert!(kept.len() < idx.len());
        for (i, (a, b)) in before.iter().zip(&p).enumerate() {
            let inside = a.x.abs() <= 2.0 && a.z.abs() <= 2.0;
            let dy = a.y - b.y;
            if inside {
                assert!((dy - BURIED_DROP).abs() < 1e-6, "buried vertex {i} at {a:?} was not lowered");
            } else {
                assert_eq!(dy, 0.0, "vertex {i} outside the structure moved");
            }
        }
    }

    #[test]
    fn no_hidden_triangle_reaches_outside_the_structure() {
        let (mut p, idx) = flat(9, -0.0001);
        let original = p.clone();
        let slab = block_solid([-2.0, -0.3, -2.0], [2.0, 0.0, 2.0], "m");
        let (kept, _) = bury_under_structures(&mut p, &idx, &[slab]);
        let kept_set: std::collections::HashSet<[u32; 3]> =
            kept.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
        for t in idx.chunks_exact(3) {
            if kept_set.contains(&[t[0], t[1], t[2]]) {
                continue;
            }
            for &i in t {
                let v = original[i as usize];
                assert!(v.x.abs() <= 2.0 + 1e-4 && v.z.abs() <= 2.0 + 1e-4,
                    "a hidden triangle has a vertex outside the floor at {v:?}: that is a gap");
            }
        }
    }

    #[test]
    fn a_structure_raised_off_the_ground_hides_nothing() {
        let (mut p, idx) = flat(9, 0.0);
        let before = p.clone();
        let bridge = block_solid([-2.0, 2.0, -2.0], [2.0, 2.3, 2.0], "m");
        let (kept, burial) = bury_under_structures(&mut p, &idx, &[bridge]);
        assert_eq!(burial, Burial::default());
        assert_eq!(kept, idx);
        assert_eq!(p, before);
    }

    /// THE REAL LEVEL: nowhere inside either room may drawn terrain come within
    /// 2 mm of the floor at y = 0 -- about twice the depth precision at 20 m with
    /// this near plane, so a fight there is impossible. Before this existed the
    /// terrain sat 0.1 mm under the floor.
    /// NO HOLES: a triangle is hidden only if structure covers ALL of it, not
    /// just its three corners. At an inside corner -- the hallway meeting a hall
    /// -- a triangle can have every corner under a building and still reach
    /// out over open ground, and hiding it opens a hole in the grass there
    /// (headset, 2026-09-24).
    /// THE JOIN BETWEEN A BAKED CAVE AND THE GROUND, MEASURED on the level's
    /// own files (`GAME_DIR`, or the shipped game): the ground -- the
    /// heightfield and the patch the cave's bake put back in its cells -- and
    /// the cave's mesh are ONE closed surface. Every edge the ground leaves
    /// open inside the field is an edge of the cave, at the same two points,
    /// with the same normal on both sides. The shipped cave before the rework
    /// left 436 edges open here.
    ///
    /// Prints the step it measures. Skips on a level with no baked cave.
    #[test]
    fn a_baked_cave_and_the_ground_round_it_are_one_surface() {
        use std::collections::HashMap;
        let game = std::env::var("GAME_DIR").unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../game").to_string());
        let game = std::path::Path::new(&game);
        let Ok(scene) = space_soup_engine::scene::Scene::load(&game.join("scenes/test_room.json")) else {
            eprintln!("skipped: no test_room");
            return;
        };
        let def = scene.terrain.as_ref().expect("test_room has terrain");
        let source = space_soup_engine::terrain::load(def, game).expect("terrain loads");
        let ground = source.patch(source.bounds(), 1);
        let ground_n = ground_normals(&ground);
        let bounds = source.bounds();

        // The caves' triangles in the world, the lip under the ground's edge
        // left out.
        let mut cave_p: Vec<Vec3> = Vec::new();
        let mut cave_n: Vec<Vec3> = Vec::new();
        let mut cave_i: Vec<u32> = Vec::new();
        for obj in scene.objects.iter().filter(|o| !o.hidden && o.terrain_collider.is_some()) {
            let Some(mesh) = &obj.mesh else { continue };
            let Some(parts) = space_soup_engine::mesh_lightmap::load_mesh_parts(&game.join(&mesh.path)) else { continue };
            // The file's chunk table says which of each run's indices are the
            // lip: `[first, count, box..., lip]` (`sdfBakeGlb.js`).
            let bytes = std::fs::read(game.join(&mesh.path)).expect("the cave's file");
            let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
            let json: serde_json::Value = serde_json::from_slice(&bytes[20..20 + json_len]).expect("the glb's json");
            let surface: Vec<std::ops::Range<usize>> = json["meshes"][0]["extras"]["space_soup"]["cave"]["chunks"]
                .as_array()
                .map(|chunks| {
                    chunks
                        .iter()
                        .map(|c| {
                            let n = |k: usize| c[k].as_u64().unwrap_or(0) as usize;
                            n(0)..n(0) + n(1) - n(8).min(n(1))
                        })
                        .collect()
                })
                .unwrap_or_default();
            for part in parts.iter().filter(|p| p.authored.is_some()) {
                assert!(obj.cuboid.rotation.angle_between(glam::Quat::IDENTITY) < 1e-6 && mesh.scale == Vec3::ONE, "a baked cave is placed, not turned");
                let base = cave_p.len() as u32;
                cave_p.extend(part.positions.iter().map(|p| Vec3::from(*p) + obj.cuboid.position));
                cave_n.extend(part.normals.iter().map(|n| Vec3::from(*n)));
                for run in &surface {
                    cave_i.extend(part.indices[run.clone()].iter().map(|i| i + base));
                }
            }
        }
        if cave_i.is_empty() || ground.normals.is_empty() {
            eprintln!("skipped: this level has no baked cave with its ground patch");
            return;
        }

        // One id a place: every vertex within a tenth of a millimetre of
        // another is the same point (the cave's are stored round its object
        // and split at its chart seams).
        const SAME: f32 = 1e-4;
        let cell = |p: Vec3| ((p.x / 1e-3).floor() as i64, (p.y / 1e-3).floor() as i64, (p.z / 1e-3).floor() as i64);
        let mut places: Vec<Vec3> = Vec::new();
        let mut grid: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
        let mut place = |p: Vec3| -> u32 {
            let (cx, cy, cz) = cell(p);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(ids) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                            if let Some(id) = ids.iter().find(|id| places[**id as usize].distance(p) < SAME) {
                                return *id;
                            }
                        }
                    }
                }
            }
            places.push(p);
            grid.entry((cx, cy, cz)).or_default().push((places.len() - 1) as u32);
            (places.len() - 1) as u32
        };
        let ground_at: Vec<u32> = ground.positions.iter().map(|p| place(*p)).collect();
        let cave_at: Vec<u32> = cave_p.iter().map(|p| place(*p)).collect();
        let open = |indices: &[u32], at: &[u32]| -> HashMap<(u32, u32), (u32, u32)> {
            let mut count: HashMap<(u32, u32), (u32, (u32, u32))> = HashMap::new();
            for t in indices.chunks_exact(3) {
                for k in 0..3 {
                    let (a, b) = (t[k], t[(k + 1) % 3]);
                    let (pa, pb) = (at[a as usize], at[b as usize]);
                    if pa == pb {
                        continue;
                    }
                    let e = count.entry((pa.min(pb), pa.max(pb))).or_insert((0, (a, b)));
                    e.0 += 1;
                }
            }
            count.into_iter().filter(|(_, (n, _))| *n == 1).map(|(k, (_, v))| (k, v)).collect()
        };
        let on_rim = |p: Vec3| {
            (p.x - bounds.min.x).abs() < 1e-3 || (p.x - bounds.max.x).abs() < 1e-3 || (p.z - bounds.min.z).abs() < 1e-3 || (p.z - bounds.max.z).abs() < 1e-3
        };
        let ground_open = open(&ground.indices, &ground_at);
        let cave_open = open(&cave_i, &cave_at);
        let inner: Vec<_> = ground_open
            .iter()
            .filter(|(_, (a, b))| !(on_rim(ground.positions[*a as usize]) && on_rim(ground.positions[*b as usize])))
            .collect();
        let unmatched_ground = inner.iter().filter(|(k, _)| !cave_open.contains_key(k)).count();
        let unmatched_cave = cave_open.keys().filter(|k| !ground_open.contains_key(k)).count();

        // The step across each shared edge: in position (the two draws' own
        // vertices) and in normal.
        let mut cave_normals_at: HashMap<u32, Vec<(Vec3, Vec3)>> = HashMap::new();
        for t in cave_i.chunks_exact(3) {
            for &i in t {
                cave_normals_at.entry(cave_at[i as usize]).or_default().push((cave_p[i as usize], cave_n[i as usize]));
            }
        }
        let (mut worst_gap, mut angles) = (0f32, Vec::new());
        let mut seen = std::collections::HashSet::new();
        for (key, (a, b)) in &inner {
            if !cave_open.contains_key(key) {
                continue;
            }
            for g in [*a, *b] {
                let id = ground_at[g as usize];
                if !seen.insert(id) {
                    continue;
                }
                let n = ground_n[g as usize];
                // The cave's vertex there whose normal is nearest the ground's:
                // rock a hand thick has a second, turned from it, for its other face.
                let mut best = 180f32;
                for (p, cn) in &cave_normals_at[&id] {
                    worst_gap = worst_gap.max(p.distance(ground.positions[g as usize]));
                    best = best.min(n.angle_between(*cn).to_degrees());
                }
                angles.push(best);
            }
        }
        angles.sort_by(|a, b| a.total_cmp(b));
        let pct = |f: f32| angles.get(((angles.len() as f32 * f) as usize).min(angles.len().saturating_sub(1))).copied().unwrap_or(0.0);
        eprintln!(
            "cave join: {} edges of the ground open onto the cave, {} unmatched; {} of the cave's open edges unmatched; \
             {} shared vertices; position step worst {:.4} mm; normal step median {:.3}, 95th {:.3}, worst {:.3} degrees",
            inner.len(), unmatched_ground, unmatched_cave, angles.len(), worst_gap * 1000.0, pct(0.5), pct(0.95), pct(1.0),
        );
        assert!(inner.len() > 100, "the cave meets the ground along its mouths");
        assert_eq!(unmatched_ground, 0, "an edge of the ground inside the field with no cave beyond it is a hole");
        assert_eq!(unmatched_cave, 0, "an edge of the cave with no ground beyond it is a gap into the hill");
        assert!(worst_gap < 0.05e-3, "the two draws' vertices are the same points: {worst_gap} m");
        assert!(pct(0.95) < 1.0, "and carry the same normal: 95th percentile {} degrees", pct(0.95));
    }

    #[test]
    fn no_hidden_terrain_triangle_leaves_a_hole_in_the_shipped_level() {
        let game = concat!(env!("CARGO_MANIFEST_DIR"), "/../game");
        let scene = space_soup_engine::scene::Scene::load(std::path::Path::new(&format!("{game}/scenes/test_room.json")))
            .expect("test_room.json should load");
        let def = scene.terrain.as_ref().expect("test_room has terrain");
        let source = space_soup_engine::terrain::load(def, std::path::Path::new(game)).expect("terrain loads");
        let patch = source.patch(source.bounds(), 1);
        let solids: Vec<BrushSolid> = scene.objects.iter().filter_map(|o| o.brush.as_ref()).flat_map(|b| b.evaluate()).collect();
        let mut p = patch.positions.clone();
        let original = p.clone();
        let (kept, _) = bury_under_structures(&mut p, &patch.indices, &solids);
        let kept_set: std::collections::HashSet<[u32; 3]> = kept.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
        let covered = |q: Vec3| {
            let probe = [q.x as f64, (q.y - BURIED_PROBE_DEPTH) as f64, q.z as f64];
            solids.iter().any(|s| contains_point(s, probe, 1e-4))
        };
        let mut holes = Vec::new();
        for t in patch.indices.chunks_exact(3) {
            if kept_set.contains(&[t[0], t[1], t[2]]) {
                continue;
            }
            let (a, b, c) = (original[t[0] as usize], original[t[1] as usize], original[t[2] as usize]);
            'samples: for i in 0..=6 {
                for j in 0..=(6 - i) {
                    let q = a + (b - a) * (i as f32 / 6.0) + (c - a) * (j as f32 / 6.0);
                    if !covered(q) {
                        holes.push((q.x, q.z));
                        break 'samples;
                    }
                }
            }
        }
        assert!(holes.is_empty(), "{} hidden triangle(s) reach open ground, first at {:?}", holes.len(), &holes[..holes.len().min(6)]);
    }

    #[test]
    fn inside_the_shipped_rooms_no_drawn_terrain_is_near_the_floor() {
        let game = concat!(env!("CARGO_MANIFEST_DIR"), "/../game");
        let scene = space_soup_engine::scene::Scene::load(std::path::Path::new(&format!("{game}/scenes/test_room.json")))
            .expect("test_room.json should load");
        let def = scene.terrain.as_ref().expect("test_room has terrain");
        let source = space_soup_engine::terrain::load(def, std::path::Path::new(game)).expect("terrain loads");
        let patch = source.patch(source.bounds(), 1);
        let structures: Vec<&BrushDef> = scene.objects.iter().filter_map(|o| o.brush.as_ref()).collect();
        let solids: Vec<BrushSolid> = structures.iter().flat_map(|b| b.evaluate()).collect();
        let mut p = patch.positions.clone();
        let (kept, burial) = bury_under_structures(&mut p, &patch.indices, &solids);
        assert!(burial.hidden_triangles > 0, "no terrain under the buildings was hidden");

        // THE ROOMS AS THE LEVEL'S OWN CARVES DEFINE THEM, not as coordinates
        // written here: this listed the brick room's interior by number, and
        // when the room moved east the test went on checking the open lawn
        // where it used to stand.
        let named: Vec<(&str, &BrushDef)> =
            scene.objects.iter().filter_map(|o| o.brush.as_ref().map(|b| (o.id.as_str(), b))).collect();
        let (rooms, _) = space_soup_engine::room_graph::rooms_from_scene(&named);
        assert!(rooms.len() >= 3, "expected the hall, the hallway and the brick room, got {}", rooms.len());
        let in_room = |x: f32, z: f32| {
            rooms.iter().any(|r| x > r.min.x && x < r.max.x && z > r.min.z && z < r.max.z)
        };
        let mut worst = f32::NEG_INFINITY;
        for t in kept.chunks_exact(3) {
            let (a, b, c) = (p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]);
            // Sample the triangle's interior, not just its corners: the height
            // that matters is wherever the triangle passes under the room.
            for i in 0..=8 {
                for j in 0..=(8 - i) {
                    let (u, v) = (i as f32 / 8.0, j as f32 / 8.0);
                    let q = a + (b - a) * u + (c - a) * v;
                    if in_room(q.x, q.z) {
                        worst = worst.max(q.y);
                    }
                }
            }
        }
        assert!(worst <= -0.002, "drawn terrain reaches {worst} m inside a room -- within 2 mm of the floor, so it can z-fight");
    }
}

/// THE HORIZON TEST ON THE SHIPPED LEVEL: whatever it calls hidden is hidden,
/// asked of the drawn triangles ray by ray. See
/// `space_soup::renderer::ground_horizon`.
#[cfg(test)]
pub(crate) mod horizon_tests {
    use super::*;
    use space_soup::renderer::ground_horizon::{GroundOccluder, Horizon, REACH};

    /// The shipped level's ground as the headset draws it and its baked caves'
    /// rock, each by the heightfield cell its triangles lie over, for asking
    /// one ray at a time.
    pub struct Level {
        pub geometry: TerrainGeometry,
        pub occluder: GroundOccluder,
        pub chunks: Vec<(u32, u32, (Vec3, Vec3))>,
        /// The caves that plug the ground's holes, and their triangles.
        pub caves: Vec<(Vec3, Vec3)>,
        rock: Vec<[Vec3; 3]>,
        lo: glam::Vec2,
        cell: glam::Vec2,
        n: [u32; 2],
        cells: Vec<Vec<u32>>,
        rock_cells: Vec<Vec<u32>>,
    }

    pub fn level() -> Option<Level> {
        let game = crate::offline_frame::offline_game_dir();
        let scene = space_soup_engine::scene::Scene::load(&game.join("scenes/test_room.json")).ok()?;
        let def = scene.terrain.as_ref()?;
        let structures: Vec<&BrushDef> = scene.objects.iter().filter_map(|o| o.brush.as_ref()).collect();
        let geometry = load(def, &game, 1, &structures)?;
        let occluder = geometry.ground_occluder(Some(def))?;
        let chunks = geometry.caster_chunks().iter().map(|c| (c.first_index, c.index_count, (c.min, c.max))).collect();
        let caves = plugging_caves(&scene, &game);
        let mut rock = Vec::new();
        for obj in scene.objects.iter().filter(|o| !o.hidden && o.terrain_collider.is_some()) {
            let Some(mesh) = &obj.mesh else { continue };
            for part in space_soup_engine::mesh_lightmap::load_mesh_parts(&game.join(&mesh.path)).unwrap_or_default() {
                let at = |i: u32| obj.cuboid.rotation * (Vec3::from(part.positions[i as usize]) * mesh.scale) + obj.cuboid.position;
                rock.extend(part.indices.chunks_exact(3).map(|t| [at(t[0]), at(t[1]), at(t[2])]));
            }
        }
        let (lo3, hi3) = geometry.world_bounds();
        let n = [256u32, 256];
        let lo = glam::Vec2::new(lo3.x, lo3.z);
        let cell = (glam::Vec2::new(hi3.x, hi3.z) - lo) / glam::Vec2::new(n[0] as f32, n[1] as f32);
        let bucket = |tris: &mut dyn Iterator<Item = [Vec3; 3]>| {
            let mut cells = vec![Vec::new(); (n[0] * n[1]) as usize];
            for (t, p) in tris.enumerate() {
                let (a, b) = (p[0].min(p[1]).min(p[2]), p[0].max(p[1]).max(p[2]));
                let at = |v: f32, o: f32, c: f32, n: u32, pad: f32| (((v - o) / c + pad).floor().max(0.0) as u32).min(n - 1);
                for iz in at(a.z, lo.y, cell.y, n[1], -1e-3)..=at(b.z, lo.y, cell.y, n[1], 1e-3) {
                    for ix in at(a.x, lo.x, cell.x, n[0], -1e-3)..=at(b.x, lo.x, cell.x, n[0], 1e-3) {
                        cells[(iz * n[0] + ix) as usize].push(t as u32);
                    }
                }
            }
            cells
        };
        let cells = bucket(&mut geometry.indices.chunks_exact(3).map(|tri| [0, 1, 2].map(|k| Vec3::from(geometry.vertices[tri[k] as usize].position))));
        let rock_cells = bucket(&mut rock.iter().copied());
        Some(Level { geometry, occluder, chunks, caves, rock, lo, cell, n, cells, rock_cells })
    }

    impl Level {
        fn triangle(&self, t: u32) -> [Vec3; 3] {
            let tri = &self.geometry.indices[t as usize * 3..t as usize * 3 + 3];
            [0, 1, 2].map(|k| Vec3::from(self.geometry.vertices[tri[k] as usize].position))
        }

        /// Whether the sight line from `eye` to `to` goes through the upper
        /// side of a drawn triangle of the ground before it arrives -- or,
        /// with `rock`, through the caves' rock from either side.
        pub fn blocked(&self, eye: Vec3, to: Vec3, rock: bool) -> bool {
            let dir = to - eye;
            let hit = |[a, b, c]: [Vec3; 3], both_sides: bool| {
                let n = (b - a).cross(c - a);
                let facing = n.dot(dir);
                if if both_sides { facing.abs() < 1e-12 } else { facing >= 0.0 } {
                    return false;
                }
                let s = n.dot(a - eye) / facing;
                if !(s > 1e-4 && s < 1.0 - 1e-4) {
                    return false;
                }
                let p = eye + dir * s;
                // A tenth of a millimetre over each edge: a line that meets
                // the shared edge of two triangles must not slip between
                // them, as it cannot between their pixels.
                let inside = |u: Vec3, v: Vec3| (v - u).cross(p - u).dot(n) >= -1e-4 * (v - u).length() * n.length();
                inside(a, b) && inside(b, c) && inside(c, a)
            };
            // Every cell the line passes over, a cell at a time.
            let g = |p: Vec3| (glam::Vec2::new(p.x, p.z) - self.lo) / self.cell;
            let (a, b) = (g(eye), g(to));
            let steps = ((b - a).abs().max_element() * 2.0).ceil().max(1.0) as u32;
            let mut last = (u32::MAX, u32::MAX);
            for i in 0..=steps {
                let q = a + (b - a) * (i as f32 / steps as f32);
                for (dx, dz) in [(0.0, 0.0), (0.51, 0.0), (-0.51, 0.0), (0.0, 0.51), (0.0, -0.51)] {
                    let (x, z) = ((q.x + dx).floor(), (q.y + dz).floor());
                    if x < 0.0 || z < 0.0 || x >= self.n[0] as f32 || z >= self.n[1] as f32 {
                        continue;
                    }
                    let c = (x as u32, z as u32);
                    if (dx, dz) == (0.0, 0.0) {
                        if c == last {
                            continue;
                        }
                        last = c;
                    }
                    let k = (c.1 * self.n[0] + c.0) as usize;
                    if self.cells[k].iter().any(|&t| hit(self.triangle(t), false)) {
                        return true;
                    }
                    if rock && self.rock_cells[k].iter().any(|&t| hit(self.rock[t as usize], true)) {
                        return true;
                    }
                }
            }
            false
        }

        /// The chunks the horizon at `eye` hides, and the first place on one
        /// that an eye within its reach sees after all: of each chunk called
        /// hidden, one triangle in `every` that faces the eye asked from, at
        /// its middle. `caves`: the second rule too.
        pub fn hidden_from(&self, eye: Vec3, every: usize, caves: bool) -> (Vec<usize>, Option<(Vec3, Vec3)>, bool) {
            let boxes: &[(Vec3, Vec3)] = if caves { &self.caves } else { &[] };
            let mut h = Horizon::new(&self.occluder, eye, REACH, boxes);
            let mut hidden = Vec::new();
            let mut seen = None;
            let eyes = [eye, eye + Vec3::X * REACH, eye - Vec3::X * REACH, eye + Vec3::Y * REACH, eye - Vec3::Y * REACH, eye + Vec3::Z * REACH, eye - Vec3::Z * REACH];
            for (k, &(first, count, (lo, hi))) in self.chunks.iter().enumerate() {
                if !h.hides_ground(&self.occluder, lo, hi) {
                    continue;
                }
                hidden.push(k);
                for t in (first / 3..(first + count) / 3).skip(k % every).step_by(every) {
                    let c = self.triangle(t);
                    let mid = (c[0] + c[1] + c[2]) / 3.0;
                    let from = eyes[t as usize % eyes.len()];
                    if (c[1] - c[0]).cross(c[2] - c[0]).dot(from - mid) <= 0.0 {
                        continue;
                    }
                    if seen.is_none() && !self.blocked(from, mid, h.plugged()) {
                        seen = Some((from, mid));
                    }
                }
            }
            (hidden, seen, h.plugged())
        }
    }

    fn bench_eyes() -> Vec<(String, Vec3, Vec3)> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("bench_views.json");
        let doc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let v3 = |v: &serde_json::Value| Vec3::from(<[f32; 3]>::try_from(serde_json::from_value::<Vec<f32>>(v.clone()).unwrap()).unwrap());
        doc["views"].as_array().unwrap().iter().map(|v| (v["name"].as_str().unwrap().to_string(), v3(&v["eye"]), v3(&v["at"]))).collect()
    }

    /// The walk through the cave, mouth to mouth and out both ways: the bench's
    /// four cave views joined up.
    fn cave_walk() -> Vec<Vec3> {
        let stops = [Vec3::new(6.0, 1.2, 36.0), Vec3::new(12.5, 1.0, 35.0), Vec3::new(19.6, 1.0, 33.3), Vec3::new(26.0, 1.2, 33.5), Vec3::new(35.5, 2.0, 35.5), Vec3::new(42.0, 2.5, 37.0)];
        stops.windows(2).flat_map(|w| (0..10).map(move |i| w[0].lerp(w[1], i as f32 / 10.0))).collect()
    }

    #[test]
    fn the_shipped_ground_is_a_sheet_with_the_caves_mouths_open() {
        let Some(level) = level() else { return };
        // Nearly every cell is ground; the caves' cells and the ground under
        // the buildings' floors are not.
        let cells = 256 * 256;
        assert!(level.occluder.solid_cells > cells * 9 / 10 && level.occluder.solid_cells < cells, "{} of {cells} cells are ground", level.occluder.solid_cells);
        // The fixture: the cave is open from end to end, to the ground's own
        // triangles and to its rock's; its hill is not; and the level's cave
        // is a baked one, which plugs the ground's holes.
        let (chamber, back) = (Vec3::new(19.6, 1.0, 33.3), Vec3::new(26.0, 1.2, 33.5));
        assert!(!level.blocked(chamber, back, true), "the cave's chamber sees its back passage");
        assert!(level.blocked(chamber, Vec3::new(19.6, 30.0, 33.3), true) && !level.blocked(chamber, Vec3::new(19.6, 30.0, 33.3), false), "the cave's roof is its rock, not the ground's upper side");
        assert!(level.blocked(Vec3::new(35.5, 2.0, 35.5), Vec3::new(-17.0, 0.5, 0.0), false) || level.blocked(Vec3::new(0.0, 1.7, 24.0), Vec3::new(40.0, 1.0, 60.0), false), "no hill stands between the bench's views");
        assert_eq!(level.caves.len(), 1, "test_room's cave is a baked one");
    }

    /// NO GROUND THE HORIZON HIDES CAN BE SEEN, from the bench's viewpoints,
    /// from a walk through the cave and out of both its mouths, and from a
    /// grid of places over the level, standing and crouched -- under the
    /// first rule (the ground's own upper side) asked of the ground's
    /// triangles alone, and under both (the caves' rock too) asked of the
    /// ground and the rock: of every chunk called hidden, the triangles that
    /// face an eye within the horizon's reach are asked ray by ray. And the
    /// rules do hide ground in the views they were built for, and never all
    /// of it from inside the cave, whose mouths show some.
    #[test]
    fn nothing_the_horizon_hides_in_the_shipped_level_can_be_seen() {
        let Some(level) = level() else { return };
        let source = {
            let game = crate::offline_frame::offline_game_dir();
            let scene = space_soup_engine::scene::Scene::load(&game.join("scenes/test_room.json")).unwrap();
            space_soup_engine::terrain::load(scene.terrain.as_ref().unwrap(), &game).unwrap()
        };
        let mut eyes: Vec<(String, Vec3, usize)> = bench_eyes().into_iter().map(|(n, e, _)| (n, e, 5)).collect();
        eyes.extend(cave_walk().into_iter().enumerate().map(|(i, e)| (format!("cave walk {i}"), e, 3)));
        for iz in 0..12 {
            for ix in 0..12 {
                let (x, z) = (-50.0 + ix as f32 * 9.1, -50.0 + iz as f32 * 9.1);
                if let Some(y) = source.height_at(x, z) {
                    eyes.push((format!("grid {ix},{iz}"), Vec3::new(x, y + 1.6, z), 13));
                    eyes.push((format!("grid {ix},{iz} crouched"), Vec3::new(x, y + 0.6, z), 17));
                }
            }
        }
        let mut hidden_by: std::collections::HashMap<String, (usize, usize)> = Default::default();
        for (name, eye, every) in &eyes {
            let (first, seen, _) = level.hidden_from(*eye, *every, false);
            assert!(seen.is_none(), "{name}, the first rule: from {:?} the ground at {:?}, of a chunk called hidden, is seen", seen.unwrap().0, seen.unwrap().1);
            let (both, seen, plugged) = level.hidden_from(*eye, *every, true);
            assert!(seen.is_none(), "{name}, both rules: from {:?} the ground at {:?}, of a chunk called hidden, is seen", seen.unwrap().0, seen.unwrap().1);
            assert!(first.iter().all(|k| both.contains(k)), "{name}: the second rule shows what the first hides");
            assert!(plugged || both == first, "{name}: the second rule spoke for an eye it does not cover");
            assert!(both.len() < level.chunks.len(), "{name}: every chunk hidden");
            hidden_by.insert(name.clone(), (first.len(), both.len()));
        }
        let of = |name: &str| hidden_by[name];
        eprintln!("chunks hidden, first rule and both: cave_back_out {:?}, cave_mouth {:?}, cave_chamber {:?}, cave_back_exit {:?}, outdoors_field {:?}", of("cave_back_out"), of("cave_mouth"), of("cave_chamber"), of("cave_back_exit"), of("outdoors_field"));
        assert!(of("cave_back_out").0 >= 8 && of("cave_back_out").1 > of("cave_back_out").0, "cave_back_out hides {:?} chunks", of("cave_back_out"));
        assert!(of("cave_chamber").0 == 0 && of("cave_chamber").1 >= 15, "from inside the cave {:?} chunks are hidden", of("cave_chamber"));
        let total: usize = hidden_by.values().map(|h| h.1).sum();
        assert!(total > eyes.len() * 4, "{total} chunks hidden over {} eyes: the test proves little", eyes.len());
    }

    /// DIAGNOSTIC: what the sight lines through given pixels of a bench view's
    /// offline picture (1445 x 1546, 104 degrees) meet first, sample by
    /// sample: for a pixel that `HORIZON=1` changes, whether a triangle of a
    /// chunk called hidden is the first thing a sample's line meets.
    /// `VIEW=pond PIXELS=779,740;331,687 cargo test --release horizon_pixels -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn horizon_pixels() {
        let Some(level) = level() else { return };
        let name = std::env::var("VIEW").unwrap_or_else(|_| "pond".into());
        let (_, eye, at) = bench_eyes().into_iter().find(|(n, _, _)| *n == name).expect("a bench view");
        let (w, h) = (1445.0f32, 1546.0f32);
        let clip = glam::Mat4::perspective_rh(104f32.to_radians(), w / h, 0.03, 1000.0) * glam::Mat4::look_at_rh(eye, at, Vec3::Y);
        let inv = clip.inverse();
        let mut horizon = Horizon::new(&level.occluder, eye, REACH, &level.caves);
        let hidden: Vec<bool> = level.chunks.iter().map(|&(_, _, (lo, hi))| horizon.hides_ground(&level.occluder, lo, hi)).collect();
        println!("{name}: {} chunk(s) hidden: {:?}", hidden.iter().filter(|h| **h).count(), hidden.iter().enumerate().filter(|(_, h)| **h).map(|(k, _)| k).collect::<Vec<_>>());
        let chunk_of = |t: u32| level.chunks.iter().position(|&(first, count, _)| t * 3 >= first && t * 3 < first + count).unwrap();
        // `RECT=x0,y0,x1,y1`: every sample of every pixel of a rectangle whose
        // line meets the upper side of a hidden chunk's triangle before
        // anything else of the ground.
        if let Ok(rect) = std::env::var("RECT") {
            let r: Vec<f32> = rect.split(',').map(|n| n.trim().parse().unwrap()).collect();
            let meet = |dir: Vec3, t: u32| -> Option<(f32, bool)> {
                let [a, b, c] = level.triangle(t);
                let (e1, e2) = (b - a, c - a);
                let p = dir.cross(e2);
                let det = e1.dot(p);
                if det.abs() < 1e-12 {
                    return None;
                }
                let s = eye - a;
                let u = s.dot(p) / det;
                let q = s.cross(e1);
                let v = dir.dot(q) / det;
                let at_t = e2.dot(q) / det;
                (u >= -1e-5 && v >= -1e-5 && u + v <= 1.0 + 1e-5 && at_t > 0.03).then_some((at_t, e1.cross(e2).dot(dir) < 0.0))
            };
            let hidden_tris: Vec<u32> = level.chunks.iter().zip(&hidden).filter(|(_, h)| **h).flat_map(|(&(first, count, _), _)| first / 3..(first + count) / 3).collect();
            let all = level.geometry.indices.len() as u32 / 3;
            let (mut rays, mut seen) = (0u32, 0u32);
            for y in r[1] as u32..=r[3] as u32 {
                for x in r[0] as u32..=r[2] as u32 {
                    for (sx, sy) in [(0.5, 0.5), (0.375, 0.125), (0.875, 0.375), (0.125, 0.625), (0.625, 0.875)] {
                        let ndc = glam::Vec2::new((x as f32 + sx) / w * 2.0 - 1.0, 1.0 - (y as f32 + sy) / h * 2.0);
                        let dir = (inv.project_point3(Vec3::new(ndc.x, ndc.y, 1.0)) - eye).normalize();
                        rays += 1;
                        let Some((near, tri)) = hidden_tris.iter().filter_map(|&t| meet(dir, t).filter(|m| m.1).map(|m| (m.0, t))).min_by(|a, b| a.0.total_cmp(&b.0)) else { continue };
                        let nearer = (0..all).filter_map(|t| meet(dir, t)).any(|m| m.0 < near - 1e-3);
                        if !nearer {
                            seen += 1;
                            if seen <= 12 {
                                println!("  SEEN: pixel {x},{y} sample ({sx}, {sy}) meets hidden chunk {} at {near:.2} m first", chunk_of(tri));
                            }
                        }
                    }
                }
            }
            println!("{name}: {seen} of {rays} sample lines in {rect} meet a hidden chunk first");
        }
        let pixels = std::env::var("PIXELS").unwrap_or_default();
        for px in pixels.split(';').filter(|p| !p.is_empty()) {
            let xy: Vec<f32> = px.split(',').map(|n| n.trim().parse().unwrap()).collect();
            for (sx, sy) in [(0.5, 0.5), (0.375, 0.125), (0.875, 0.375), (0.125, 0.625), (0.625, 0.875)] {
                let ndc = glam::Vec2::new((xy[0] + sx) / w * 2.0 - 1.0, 1.0 - (xy[1] + sy) / h * 2.0);
                let far = inv.project_point3(Vec3::new(ndc.x, ndc.y, 1.0));
                let dir = (far - eye).normalize();
                // Every triangle the line meets, nearest first.
                let mut hits: Vec<(f32, u32, bool)> = Vec::new();
                for t in 0..level.geometry.indices.len() as u32 / 3 {
                    let [a, b, c] = level.triangle(t);
                    let (e1, e2) = (b - a, c - a);
                    let p = dir.cross(e2);
                    let det = e1.dot(p);
                    if det.abs() < 1e-12 {
                        continue;
                    }
                    let s = eye - a;
                    let u = s.dot(p) / det;
                    let q = s.cross(e1);
                    let v = dir.dot(q) / det;
                    let at_t = e2.dot(q) / det;
                    if u >= -1e-5 && v >= -1e-5 && u + v <= 1.0 + 1e-5 && at_t > 0.03 {
                        hits.push((at_t, t, e1.cross(e2).dot(dir) < 0.0));
                    }
                }
                hits.sort_by(|a, b| a.0.total_cmp(&b.0));
                let line: Vec<String> = hits
                    .iter()
                    .take(5)
                    .map(|&(t, tri, front)| format!("{t:.2} m chunk {}{}{}", chunk_of(tri), if hidden[chunk_of(tri)] { " HIDDEN" } else { "" }, if front { "" } else { " (back)" }))
                    .collect();
                println!("  pixel {px} sample ({sx}, {sy}): {}", if line.is_empty() { "nothing".to_string() } else { line.join(" | ") });
            }
        }
    }

    /// DIAGNOSTIC: what the horizon takes out of each bench view -- the share
    /// of the ground's indices in the view's frustum that it hides by whole
    /// chunk under the first rule and under both, and what hiding each
    /// chunk's quarters or sixteenths, or each triangle, would add to both.
    /// `cargo test --release horizon_census -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn horizon_census() {
        let Some(level) = level() else { return };
        let only = std::env::var("ONLY").ok();
        println!("{:<20} {:>9} {:>8} {:>8} {:>8} {:>8} {:>8} {:>7}", "view", "in view", "rule 1", "both", "+1/4", "+1/16", "+tri", "sectors");
        for (name, eye, at) in bench_eyes() {
            if only.as_ref().is_some_and(|o| !o.split(',').any(|n| n == name)) {
                continue;
            }
            let view = glam::Mat4::look_at_rh(eye, at, Vec3::Y);
            let proj = glam::Mat4::perspective_rh(105f32.to_radians(), 0.93, 0.03, 1000.0);
            let planes = space_soup::renderer::shadow::frustum_planes(proj * view);
            let mut first = Horizon::new(&level.occluder, eye, REACH, &[]);
            let mut h = Horizon::new(&level.occluder, eye, REACH, &level.caves);
            let (mut in_view, mut by_first, mut by_chunk) = (0u32, 0u32, 0u32);
            let mut by_part = [0u32; 3];
            for &(first_index, count, (lo, hi)) in &level.chunks {
                if !space_soup::renderer::shadow::aabb_in_frustum(&planes, lo, hi) {
                    continue;
                }
                in_view += count;
                by_first += if first.hides_ground(&level.occluder, lo, hi) { count } else { 0 };
                if h.hides_ground(&level.occluder, lo, hi) {
                    by_chunk += count;
                    continue;
                }
                // Its triangles by quarter, by sixteenth, and one by one.
                for (slot, parts) in [(0usize, 2u32), (1, 4), (2, 0)] {
                    let mut boxes: std::collections::HashMap<(u32, u32), (Vec3, Vec3, u32)> = Default::default();
                    for t in first_index / 3..(first_index + count) / 3 {
                        let p = level.triangle(t);
                        let c = (p[0] + p[1] + p[2]) / 3.0;
                        let key = if parts == 0 {
                            (t, 0)
                        } else {
                            let f = ((c - lo) / (hi - lo).max(Vec3::splat(1e-6)) * parts as f32).min(Vec3::splat(parts as f32 - 1.0));
                            (f.x as u32, f.z as u32)
                        };
                        let e = boxes.entry(key).or_insert((Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY), 0));
                        e.0 = e.0.min(p[0]).min(p[1]).min(p[2]);
                        e.1 = e.1.max(p[0]).max(p[1]).max(p[2]);
                        e.2 += 3;
                    }
                    for (blo, bhi, n) in boxes.into_values() {
                        // (Out of the view: a finer frustum test's, not the horizon's.)
                        if space_soup::renderer::shadow::aabb_in_frustum(&planes, blo, bhi) && h.hides_ground(&level.occluder, blo, bhi) {
                            by_part[slot] += n;
                        }
                    }
                }
            }
            println!("{name:<20} {in_view:>9} {by_first:>8} {by_chunk:>8} {:>8} {:>8} {:>8} {:>7} {} {}", by_part[0], by_part[1], by_part[2], h.walked, if h.plugged() { "plugged" } else { "" }, if level.caves.iter().any(|(lo, hi)| eye.cmpge(*lo).all() && eye.cmple(*hi).all()) { "in a cave's box" } else { "" });
        }
    }
}

/// THE SHIPPED LEVEL'S GROUND BY LEVEL (`ground_lod`): its real holes, its
/// cave's patch and the ground hidden under its buildings, through the load
/// the headset runs.
#[cfg(test)]
mod lod_tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    fn shipped(lists: bool) -> Option<(TerrainGeometry, space_soup_engine::scene::Scene)> {
        let game = crate::offline_frame::offline_game_dir();
        let scene = space_soup_engine::scene::Scene::load(&game.join("scenes/test_room.json")).ok()?;
        let def = scene.terrain.as_ref()?;
        let structures: Vec<&BrushDef> = scene.objects.iter().filter_map(|o| o.brush.as_ref()).collect();
        let geometry = load_with(def, &game, 1, &structures, lists)?;
        Some((geometry, scene))
    }

    /// The directed edges of a triangle list with no partner the other way;
    /// a sliver (both ways round in the buffer) counted the way that closes.
    fn open_edges(position: &dyn Fn(u32) -> Vec3, idx: &[u32]) -> Vec<(u32, u32)> {
        let mut edges: HashMap<(u32, u32), i32> = HashMap::new();
        let mut slivers: HashSet<[u32; 3]> = HashSet::new();
        for t in idx.chunks_exact(3) {
            let p = |k: usize| position(t[k]);
            let area = (p(1).x - p(0).x) * (p(2).z - p(0).z) - (p(1).z - p(0).z) * (p(2).x - p(0).x);
            let mut s = [t[0], t[1], t[2]];
            s.sort_unstable();
            if area == 0.0 && idx.chunks_exact(3).len() < usize::MAX {
                slivers.insert(s);
                continue;
            }
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *edges.entry((a, b)).or_default() += 1;
            }
        }
        for [a, b, c] in slivers {
            let closes = |t: [u32; 3]| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])].iter().filter(|(x, y)| edges.get(&(*y, *x)).copied().unwrap_or(0) > edges.get(&(*x, *y)).copied().unwrap_or(0)).count();
            let t = if closes([a, b, c]) >= closes([a, c, b]) { [a, b, c] } else { [a, c, b] };
            for (x, y) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *edges.entry((x, y)).or_default() += 1;
            }
        }
        edges.iter().filter(|((a, b), n)| edges.get(&(*b, *a)).copied().unwrap_or(0) != **n).map(|(e, _)| *e).collect()
    }

    /// STEP 1 ON THE SHIPPED LEVEL: 64 chunks of 32 cells, as before, every
    /// triangle once; and with the lists built the first of the buffer is
    /// the same indices, which is all the lever-off frame draws.
    #[test]
    fn the_shipped_ground_is_64_chunks_and_the_lists_leave_step_1_as_it_is() {
        let (Some((plain, _)), Some((mut with, _))) = (shipped(false), shipped(true)) else {
            eprintln!("skipping: no test_room");
            return;
        };
        assert_eq!(plain.lod.chunks(), [8, 8]);
        assert_eq!(plain.caster_chunks().len(), 64, "64 chunks, as before the chunks had a fixed size");
        assert_eq!(plain.step1_len(), plain.indices.len());
        assert!(with.lod.has_lists() && with.indices.len() > plain.indices.len());
        assert_eq!(with.step1(), &plain.indices[..]);
        assert_eq!(&with.caster_chunks()[..64].iter().map(|c| (c.first_index, c.index_count)).collect::<Vec<_>>(), &plain.caster_chunks().iter().map(|c| (c.first_index, c.index_count)).collect::<Vec<_>>());
        // The lever off draws the 64, whatever else is in the buffer.
        let off = with.lod.select(Vec3::new(0.0, 1.6, 6.0), &[], false);
        assert_eq!(off.units, (0..64).collect::<Vec<u32>>());
    }

    /// STEP 2 ON THE SHIPPED LEVEL: drawn by level from anywhere -- by the
    /// design's rule and by the strict one -- the ground has NO CRACK the
    /// step-1 ground has not: every open edge ends at vertices that are open
    /// at step 1 too (the level's outline, a cave mouth's rim, the rim of
    /// ground hidden under a floor) or that step 1 does not draw at all.
    /// Neighbouring chunks differ by one level at most, and a chunk with a
    /// hole or the cave's patch is at step 1 within 60 m.
    #[test]
    fn the_shipped_ground_by_level_has_no_crack_step_1_has_not() {
        let Some((mut g, _)) = shipped(true) else {
            eprintln!("skipping: no test_room");
            return;
        };
        let positions: Vec<Vec3> = g.vertices.iter().map(|v| Vec3::from(v.position)).collect();
        let position = |v: u32| positions[v as usize];
        let step1 = g.step1().to_vec();
        let mut may_be_open: HashSet<u32> = open_edges(&position, &step1).into_iter().flat_map(|(a, b)| [a, b]).collect();
        let drawn_at_step1: HashSet<u32> = step1.iter().copied().collect();
        may_be_open.extend((0..positions.len() as u32).filter(|v| !drawn_at_step1.contains(v)));
        let (lo, hi) = g.world_bounds();
        let holes: Vec<usize> = (0..64).filter(|n| g.lod.nodes[*n].holes).collect();
        assert!(!holes.is_empty(), "test_room has a cave: some chunk has a hole");
        let mut coarse_seen = false;
        for strict in [false, true] {
            g.lod.set_strict(strict);
            let mut heads = vec![
                Vec3::new(0.0, 1.6, 6.0),
                Vec3::new(lo.x + 1.0, hi.y + 2.0, lo.z + 1.0),
                Vec3::new(hi.x - 1.0, hi.y + 2.0, lo.z + 1.0),
                Vec3::new(lo.x - 150.0, 30.0, (lo.z + hi.z) * 0.5),
                Vec3::new((lo.x + hi.x) * 0.5, 80.0, hi.z + 400.0),
                Vec3::new(lo.x - 3000.0, 100.0, lo.z - 3000.0),
            ];
            // And a walk across the level, so the hysteresis is in play.
            heads.extend((0..40).map(|k| Vec3::new(lo.x + (hi.x - lo.x) * k as f32 / 39.0, 3.0, lo.z + (hi.z - lo.z) * (0.3 + 0.01 * k as f32))));
            for head in heads {
                let chosen = g.lod.select(head, &[], true);
                let idx: Vec<u32> = chosen.units.iter().flat_map(|u| {
                    let unit = g.lod.units[*u as usize];
                    g.indices[unit.first as usize..(unit.first + unit.count) as usize].to_vec()
                }).collect();
                let cracks: Vec<(u32, u32)> = open_edges(&position, &idx).into_iter().filter(|(a, b)| !(may_be_open.contains(a) && may_be_open.contains(b))).collect();
                assert!(cracks.is_empty(), "strict {strict}, head {head}: {} open edge(s) step 1 does not have, first {:?} at {}", cracks.len(), cracks[0], position(cracks[0].0));
                let mut level_of = [u8::MAX; 64];
                for (n, _) in &chosen.nodes {
                    let node = &g.lod.nodes[*n as usize];
                    coarse_seen |= node.level > 0;
                    let span = 1u32 << node.level;
                    for j in node.at[1] * span..((node.at[1] + 1) * span).min(8) {
                        for i in node.at[0] * span..((node.at[0] + 1) * span).min(8) {
                            assert_eq!(level_of[(j * 8 + i) as usize], u8::MAX, "a chunk drawn twice");
                            level_of[(j * 8 + i) as usize] = node.level;
                        }
                    }
                }
                for j in 0..8usize {
                    for i in 0..8usize {
                        let here = level_of[j * 8 + i] as i32;
                        assert_ne!(here, u8::MAX as i32, "a chunk not drawn");
                        if i + 1 < 8 {
                            assert!((here - level_of[j * 8 + i + 1] as i32).abs() <= 1, "head {head}: neighbours two levels apart");
                        }
                        if j + 1 < 8 {
                            assert!((here - level_of[(j + 1) * 8 + i] as i32).abs() <= 1, "head {head}: neighbours two levels apart");
                        }
                    }
                }
                for n in &holes {
                    let node = &g.lod.nodes[*n];
                    let d = (head.clamp(node.lo, node.hi) - head).length();
                    if d < ground_lod::HOLE_FLOOR {
                        assert_eq!(level_of[*n], 0, "head {head}: a chunk with a hole {d:.0} m off is not at step 1");
                    }
                }
            }
        }
        assert!(coarse_seen, "some view drew a coarser node");
    }
}
