#![cfg(target_os = "android")]

//! Interactive props placed in the scene: objects linking a `.prop.json`
//! (made in SSStudio's prop studio). Each is loaded in the background --
//! its definition and its model split into parts -- and played with the
//! player's own hands through `prop_core::play`: picked up by its grip,
//! its controls worked by the trigger and face button or by the other
//! hand, spares taken from the hip pouch. Drawn here: the prop itself, and
//! every piece out of it (rounds, casings, magazines) as a mesh of just that
//! piece.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};

use glam::{Mat4, Quat, Vec3};
use log::{info, warn};
use space_soup::renderer::mesh_pipeline::ModelUniform;
use space_soup::renderer::xr_renderer::XrRenderer;
use space_soup::renderer::{GltfMesh, MeshInstance};
use space_soup_engine::{Manifest, Scene};

use prop_core::play::{Body, HandIn, HandOut, Part, PropPlay};
use prop_core::{PropDef, VibeHand};

/// A scene object's prop link as a file path (relative = in the game folder).
fn resolve(game_dir: &Path, path: &str) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() { p.to_path_buf() } else { game_dir.join(p) }
}

/// What the loader thread hands back for one prop.
struct Loaded {
    object_id: String,
    pos: Vec3,
    rot: Quat,
    def: PropDef,
    dir: PathBuf,
    mesh: GltfMesh,
    parts: Vec<Part>,
}

/// Copies of a mesh holding just one piece, for drawing that piece on its
/// own (as many at once as there are of it out).
struct PiecePool {
    master: GltfMesh,
    copies: Vec<(GltfMesh, ModelUniform)>,
    used: usize,
}

/// How many copies of one piece can be drawn at once.
const MAX_COPIES: usize = 120;

struct VrProp {
    object_id: String,
    play: PropPlay,
    /// The prop file's folder (its sounds are relative to it).
    dir: PathBuf,
    mesh: GltfMesh,
    model: ModelUniform,
    pools: HashMap<String, PiecePool>,
}

/// A sound a prop asked for: file, volume, where (world).
pub(crate) struct PropSound {
    pub file: PathBuf,
    pub volume: f32,
    pub at: Vec3,
}

/// A rumble a prop asked for: which controller, strength, seconds.
pub(crate) struct PropRumble {
    pub left: bool,
    pub strength: f32,
    pub seconds: f32,
}

#[derive(Default)]
pub(crate) struct VrProps {
    scene_name: String,
    /// The scene objects that are props (drawn and held here, not by the
    /// scene's own grabbing or as boxes).
    ids: HashSet<String>,
    props: Vec<VrProp>,
    pending: Option<Receiver<Loaded>>,
}

/// World <-> render space (render = the player's offset and turn taken out).
#[derive(Clone, Copy)]
pub(crate) struct Space {
    pub offset: Vec3,
    pub yaw_inv: Quat,
}

impl Space {
    pub fn to_world(&self, (p, r): (Vec3, Quat)) -> (Vec3, Quat) {
        let yaw = self.yaw_inv.inverse();
        (yaw * p + self.offset, yaw * r)
    }
    pub fn to_render(&self, (p, r): (Vec3, Quat)) -> (Vec3, Quat) {
        (self.yaw_inv * (p - self.offset), self.yaw_inv * r)
    }
    fn render_mat(&self) -> Mat4 {
        Mat4::from_quat(self.yaw_inv) * Mat4::from_translation(-self.offset)
    }
}

impl VrProps {
    /// Start loading the props of `scene_name` (in the background).
    pub fn load(game_dir: &Path, scene_name: &str, renderer: &XrRenderer) -> Self {
        let objects: Vec<(String, Vec3, Quat, PathBuf)> = match Scene::load(&Manifest::scene_path(game_dir, scene_name)) {
            Ok(scene) => scene
                .objects
                .iter()
                .filter(|o| !o.hidden)
                .filter_map(|o| Some((o.id.clone(), o.cuboid.position, o.cuboid.rotation, resolve(game_dir, &o.prop.as_ref()?.path))))
                .collect(),
            Err(e) => {
                warn!("props: can't read scene '{scene_name}': {e}");
                Vec::new()
            }
        };
        info!("props: {} in '{scene_name}'", objects.len());
        let ids = objects.iter().map(|o| o.0.clone()).collect();
        let (tx, rx) = channel();
        let device = renderer.device().clone();
        let queue = renderer.queue().clone();
        let layout = renderer.skinned_mesh_texture_layout().clone();
        std::thread::Builder::new()
            .name("prop_loader".into())
            .spawn(move || {
                for (object_id, pos, rot, json) in objects {
                    let def = match std::fs::read_to_string(&json).map_err(|e| e.to_string()).and_then(|t| PropDef::from_json(&t)) {
                        Ok(d) => d,
                        Err(e) => {
                            warn!("prop '{object_id}': {}: {e}", json.display());
                            continue;
                        }
                    };
                    let Some(model) = prop_core::model_for_prop_file(&json, &def) else {
                        warn!("prop '{object_id}': no model next to {}", json.display());
                        continue;
                    };
                    let (mesh, binds) = match GltfMesh::load_parts(&device, &queue, &layout, &model) {
                        Ok(x) => x,
                        Err(e) => {
                            warn!("prop '{object_id}': can't load {}: {e}", model.display());
                            continue;
                        }
                    };
                    // Each part's bounds, from its vertices.
                    let mut parts: Vec<Part> = binds
                        .into_iter()
                        .map(|(name, bind)| Part { name, bind, min: Vec3::splat(f32::MAX), max: Vec3::splat(f32::MIN) })
                        .collect();
                    if let Some(skin) = mesh.skin.as_ref() {
                        for prim in &skin.primitives {
                            for v in &prim.vertices {
                                if let Some(part) = parts.get_mut(v.dominant_joint()) {
                                    let p = part.bind.transform_point3(Vec3::from(v.position));
                                    part.min = part.min.min(p);
                                    part.max = part.max.max(p);
                                }
                            }
                        }
                    }
                    info!("prop '{object_id}' loaded from {}", json.display());
                    let dir = json.parent().map(Path::to_path_buf).unwrap_or_default();
                    if tx.send(Loaded { object_id, pos, rot, def, dir, mesh, parts }).is_err() {
                        return;
                    }
                }
            })
            .expect("failed to spawn prop_loader");
        Self { scene_name: scene_name.to_string(), ids, props: Vec::new(), pending: Some(rx) }
    }

    pub fn scene_name(&self) -> &str {
        &self.scene_name
    }

    /// Is this scene object a prop?
    pub fn is_prop(&self, id: &str) -> bool {
        self.ids.contains(id)
    }

    /// Is hand `i` (0 = left) holding any prop?
    pub fn holds(&self, i: usize) -> bool {
        self.props.iter().any(|p| p.play.holds(i))
    }

    /// Before the body solve: take in what finished loading, work every prop
    /// with this frame's hands (world space), and say what each hand should
    /// look like (the first prop with a say wins).
    pub fn step(&mut self, renderer: &XrRenderer, input: [HandIn; 2], body: Option<Body>, hands_key: &str, dt: f32) -> [HandOut; 2] {
        if let Some(rx) = self.pending.as_ref() {
            for mut l in rx.try_iter() {
                l.mesh.create_skin_bind_group(renderer.device(), renderer.skin_joint_layout());
                l.mesh.position = Vec3::ZERO;
                l.mesh.rotation = Quat::IDENTITY;
                l.mesh.scale = Vec3::ONE;
                self.props.push(VrProp {
                    object_id: l.object_id,
                    play: PropPlay::new(l.def, l.parts, l.pos, l.rot),
                    dir: l.dir,
                    mesh: l.mesh,
                    model: renderer.create_skinned_model_uniform(),
                    pools: HashMap::new(),
                });
            }
        }
        let mut out = [HandOut::default(); 2];
        let mut busy = [false; 2];
        for p in &mut self.props {
            p.play.set_hands_key(hands_key);
            let o = p.play.step(input, busy, body, dt);
            for i in 0..2 {
                if out[i].wrist.is_none() && out[i].fingers.is_none() {
                    out[i] = o[i];
                }
                busy[i] |= p.play.holds(i);
            }
        }
        out
    }

    /// After the body solve (`solved`: the local wrists, world space): pose
    /// and draw every prop and what's out of it; hand back the sounds and
    /// rumbles they asked for.
    pub fn pose(&mut self, renderer: &XrRenderer, solved: [Option<(Vec3, Quat)>; 2], space: Space) -> (Vec<PropSound>, Vec<PropRumble>) {
        let to_render = space.render_mat();
        let mut sounds = Vec::new();
        let mut rumbles = Vec::new();
        for p in &mut self.props {
            let posed = p.play.pose(solved);
            let mats: Vec<Mat4> = posed.skin.iter().map(|m| to_render * *m).collect();
            p.mesh.update_joint_matrices(renderer.queue(), &mats);

            for pool in p.pools.values_mut() {
                pool.used = 0;
            }
            for (piece, m) in posed.extra {
                let mats: Vec<Mat4> = p.play.parts_of(&piece).map(|(bind, mine)| if mine { to_render * m * bind } else { Mat4::ZERO }).collect();
                let Some(pool) = Self::pool(p, renderer, &piece) else { continue };
                if pool.used >= MAX_COPIES {
                    continue;
                }
                if pool.used == pool.copies.len() {
                    let mut copy = pool.master.clone_with_independent_skin(renderer.device());
                    copy.create_skin_bind_group(renderer.device(), renderer.skin_joint_layout());
                    pool.copies.push((copy, renderer.create_skinned_model_uniform()));
                }
                let (copy, _) = &mut pool.copies[pool.used];
                copy.update_joint_matrices(renderer.queue(), &mats);
                pool.used += 1;
            }

            let at = p.play.center();
            for id in std::mem::take(&mut p.play.runtime.sounds) {
                if let Some(snd) = p.play.def.sounds.iter().find(|s| s.id == id) {
                    sounds.push(PropSound { file: p.dir.join(&snd.file), volume: snd.volume, at });
                }
            }
            let main_left = p.play.main_hand().map_or(false, |i| i == 0);
            for r in std::mem::take(&mut p.play.runtime.rumbles) {
                let hands: &[bool] = match r.hand {
                    VibeHand::Holding => &[main_left][..],
                    VibeHand::Left => &[true],
                    VibeHand::Right => &[false],
                    VibeHand::Both => &[true, false],
                };
                for &left in hands {
                    rumbles.push(PropRumble { left, strength: r.strength, seconds: r.seconds });
                }
            }
        }
        (sounds, rumbles)
    }

    /// The pool of copies of `piece` (made the first time it's needed: a
    /// copy of the model with just that piece's parts).
    fn pool<'a>(p: &'a mut VrProp, renderer: &XrRenderer, piece: &str) -> Option<&'a mut PiecePool> {
        if !p.pools.contains_key(piece) {
            let others: Vec<usize> = p.play.parts_of(piece).enumerate().filter(|(_, (_, mine))| !mine).map(|(j, _)| j).collect();
            if others.len() == p.play.parts_of(piece).count() {
                return None;
            }
            let mut master = p.mesh.clone_with_independent_skin_excluding_joints(renderer.device(), &others);
            master.create_skin_bind_group(renderer.device(), renderer.skin_joint_layout());
            master.position = Vec3::ZERO;
            master.rotation = Quat::IDENTITY;
            master.scale = Vec3::ONE;
            p.pools.insert(piece.to_string(), PiecePool { master, copies: Vec::new(), used: 0 });
        }
        p.pools.get_mut(piece)
    }

    pub fn instances(&self) -> impl Iterator<Item = MeshInstance<'_>> {
        self.props.iter().flat_map(|p| {
            std::iter::once(MeshInstance { mesh: &p.mesh, model: &p.model, lightmap_key: None }).chain(
                p.pools.values().flat_map(|pool| pool.copies.iter().take(pool.used).map(|(m, u)| MeshInstance { mesh: m, model: u, lightmap_key: None })),
            )
        })
    }

    /// Ids of the props being held, for the log.
    pub fn held_ids(&self) -> Vec<&str> {
        self.props.iter().filter(|p| p.play.holds(0) || p.play.holds(1)).map(|p| p.object_id.as_str()).collect()
    }
}
