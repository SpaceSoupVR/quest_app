#![cfg(target_os = "android")]

//! Selection pads: boxes in the scene (objects with a `selector`) that the
//! player chooses something with -- their body or their hands -- by putting
//! a controller inside. Read from the scene file (pads don't move), checked
//! against both controllers every frame; a choice fires when a controller
//! goes in, not while it stays there.

use std::path::Path;

use glam::{Quat, Vec3};
use space_soup_engine::{distance_to_oriented_box, Manifest, Scene};

struct Pad {
    center: Vec3,
    rotation: Quat,
    half: Vec3,
    set: String,
    value: String,
    /// Which controllers [left, right] were inside last frame.
    inside: [bool; 2],
}

/// A pad a controller just went into: which hand (0 = left), and what it
/// chooses ("body"/"hands", and the value).
pub(crate) struct Chosen {
    pub hand: usize,
    pub set: String,
    pub value: String,
}

#[derive(Default)]
pub(crate) struct SelectorPads {
    scene_name: String,
    pads: Vec<Pad>,
}

impl SelectorPads {
    /// The pads of `scene_name` (none if it can't be read).
    pub fn load(game_dir: &Path, scene_name: &str) -> Self {
        let pads = match Scene::load(&Manifest::scene_path(game_dir, scene_name)) {
            Ok(scene) => scene
                .objects
                .iter()
                .filter(|o| !o.hidden)
                .filter_map(|o| {
                    let s = o.selector.as_ref()?;
                    Some(Pad {
                        center: o.cuboid.position,
                        rotation: o.cuboid.rotation,
                        half: o.cuboid.half_size,
                        set: s.set.clone(),
                        value: s.value.clone(),
                        inside: [false; 2],
                    })
                })
                .collect(),
            Err(e) => {
                log::warn!("selector pads: can't read scene '{scene_name}': {e}");
                Vec::new()
            }
        };
        log::info!("selector pads: {} in '{scene_name}'", pads.len());
        Self { scene_name: scene_name.to_string(), pads }
    }

    pub fn scene_name(&self) -> &str {
        &self.scene_name
    }

    /// Check both controllers' world positions [left, right]; returns the
    /// pads a controller has just gone into.
    pub fn update(&mut self, controllers: [Option<Vec3>; 2]) -> Vec<Chosen> {
        let mut out = Vec::new();
        for pad in &mut self.pads {
            for (hand, at) in controllers.iter().enumerate() {
                let now = at.is_some_and(|p| distance_to_oriented_box(pad.center, pad.rotation, pad.half, p) == 0.0);
                if now && !pad.inside[hand] {
                    out.push(Chosen { hand, set: pad.set.clone(), value: pad.value.clone() });
                }
                pad.inside[hand] = now;
            }
        }
        out
    }
}
