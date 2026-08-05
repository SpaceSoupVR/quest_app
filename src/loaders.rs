#![cfg(target_os = "android")]

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};

use log::{info, warn};

use space_soup::renderer::xr_renderer::XrRenderer;
use space_soup::renderer::GltfMesh;

pub(crate) fn spawn_mesh_loader(
    dir: &Path,
    renderer: &XrRenderer,
) -> (Sender<(String, Vec<String>)>, Receiver<(String, GltfMesh)>) {
    let (req_tx, req_rx) = std::sync::mpsc::channel::<(String, Vec<String>)>();
    let (mesh_tx, mesh_rx) = std::sync::mpsc::channel::<(String, GltfMesh)>();
    let device = renderer.device().clone();
    let queue = renderer.queue().clone();
    let layout = renderer.mesh_texture_layout().clone();
    let gdir = dir.to_path_buf();
    std::thread::Builder::new()
        .name("mesh_loader".into())
        .spawn(move || {
            for (path, ids) in req_rx {
                let full_path = gdir.join(&path);
                match GltfMesh::load(&device, &queue, &layout, &full_path) {
                    Ok(mesh) => {
                        info!("Mesh loaded: '{path}' ({} object(s))", ids.len());
                        for id in &ids {
                            let instance = mesh.clone_with_independent_skin(&device);
                            if mesh_tx.send((id.clone(), instance)).is_err() {
                                return;
                            }
                        }
                    }
                    Err(e) => warn!("Failed to load mesh '{path}': {e}"),
                }
            }
        })
        .expect("failed to spawn mesh_loader");
    (req_tx, mesh_rx)
}

pub(crate) fn spawn_avatar_loader(path: PathBuf, renderer: &XrRenderer) -> Receiver<GltfMesh> {
    let (tx, rx) = std::sync::mpsc::channel::<GltfMesh>();
    let device = renderer.device().clone();
    let queue = renderer.queue().clone();
    let layout = renderer.skinned_mesh_texture_layout().clone();
    std::thread::Builder::new()
        .name("avatar_loader".into())
        .spawn(move || match GltfMesh::load(&device, &queue, &layout, &path) {
            Ok(mesh) => {
                info!("Avatar mesh loaded: '{}'", path.display());
                let _ = tx.send(mesh);
            }
            Err(e) => warn!("Failed to load avatar mesh '{}': {e}", path.display()),
        })
        .expect("failed to spawn avatar_loader");
    rx
}
