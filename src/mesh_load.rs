#![cfg(target_os = "android")]

use std::collections::{HashMap, HashSet};

use space_soup::renderer::GltfMesh;
use space_soup_protocol::WireRenderMesh;

pub(crate) fn queue_new_meshes(
    world_meshes: &[WireRenderMesh],
    mesh_cache: &HashMap<String, (GltfMesh, space_soup::renderer::mesh_pipeline::ModelUniform)>,
    requested: &mut HashSet<String>,
    req_tx: &std::sync::mpsc::Sender<(String, Vec<String>)>,
) {
    let mut by_path: HashMap<String, Vec<String>> = HashMap::new();
    for rm in world_meshes {
        if mesh_cache.contains_key(&rm.id) || requested.contains(&rm.id) {
            continue;
        }
        requested.insert(rm.id.clone());
        by_path.entry(rm.path.clone()).or_default().push(rm.id.clone());
    }
    for (path, ids) in by_path {
        let _ = req_tx.send((path, ids));
    }
}
