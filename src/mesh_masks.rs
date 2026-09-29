//! THE STATIONARY LAMPS' SHADOWS ON MESHES, kept by object so a mesh gets
//! them however its maps arrive: from disk at start, or from the editor's
//! stream after a re-bake, its light and its masks in either order. See
//! `space_soup_engine::lightmaps::mesh_stationary_id`.

use std::collections::{BTreeMap, HashMap};

/// `L` is how the stream's light arrives (`lightmap_client::LightmapUpdate`).
pub(crate) struct MeshMasks<L> {
    layers: HashMap<String, BTreeMap<usize, (Vec<u8>, u32, u32)>>,
    /// The last light the stream sent for an object, so masks arriving after
    /// it can still be applied.
    streamed_light: HashMap<String, L>,
}

impl<L> Default for MeshMasks<L> {
    fn default() -> Self {
        Self { layers: HashMap::new(), streamed_light: HashMap::new() }
    }
}

impl<L> MeshMasks<L> {
    /// Remember a map if it is one of a mesh's mask layers; `Some(object id)`
    /// when it was.
    pub(crate) fn take(&mut self, id: &str, rgba: &[u8], width: u32, height: u32) -> Option<String> {
        let (object, layer) = space_soup_engine::lightmaps::mesh_stationary_of(id)?;
        self.layers.entry(object.to_string()).or_default().insert(layer, (rgba.to_vec(), width, height));
        Some(object.to_string())
    }

    /// Remember the light the stream sent for `object`.
    #[cfg_attr(not(target_os = "android"), allow(dead_code))]
    pub(crate) fn remember_light(&mut self, object: &str, light: L) {
        self.streamed_light.insert(object.to_string(), light);
    }

    /// The light the stream last sent for `object`.
    #[cfg_attr(not(target_os = "android"), allow(dead_code))]
    pub(crate) fn streamed_light(&self, object: &str) -> Option<&L> {
        self.streamed_light.get(object)
    }

    /// `object`'s mask layers in order and their size, when they are whole --
    /// every layer from 0, one size -- and are the ones this scene's lamps
    /// read (`scene_lights::usable_stationary_masks`); else none, and its
    /// stationary lamps shade unshadowed.
    pub(crate) fn layers_for(&self, object: &str, channels: &HashMap<String, u8>) -> (Vec<&[u8]>, (u32, u32)) {
        let Some(layers) = self.layers.get(object) else { return (Vec::new(), (1, 1)) };
        let whole: Vec<&(Vec<u8>, u32, u32)> =
            layers.iter().enumerate().map_while(|(i, (&l, m))| (i == l).then_some(m)).collect();
        let size = whole.first().map_or((1, 1), |m| (m.1, m.2));
        if whole.len() != layers.len() || whole.iter().any(|m| (m.1, m.2) != size) {
            log::warn!("lightmaps: '{object}' has a broken set of stationary masks; its stationary lamps shade unshadowed");
            return (Vec::new(), (1, 1));
        }
        let usable = crate::scene_lights::usable_stationary_masks(whole, channels);
        (usable.iter().map(|m| m.0.as_slice()).collect(), size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn channels(n: u8) -> HashMap<String, u8> {
        (0..n).map(|c| (format!("lamp_{c}#0"), c)).collect()
    }

    #[test]
    fn a_meshs_masks_come_back_whole_and_in_order() {
        let mut masks = MeshMasks::<()>::default();
        // Three lamps need two layers; they arrive out of order.
        assert_eq!(masks.take("sconce#stationary_1", &[1; 16], 2, 2), Some("sconce".to_string()));
        assert_eq!(masks.take("sconce", &[9; 16], 2, 2), None, "the mesh's own light is not a mask");
        assert_eq!(masks.take("sconce#stationary_0", &[0; 16], 2, 2), Some("sconce".to_string()));
        let (layers, size) = masks.layers_for("sconce", &channels(3));
        assert_eq!(size, (2, 2));
        assert_eq!(layers, vec![&[0u8; 16][..], &[1u8; 16][..]]);
        assert!(masks.layers_for("lamp", &channels(3)).0.is_empty(), "a mesh with no masks");
    }

    #[test]
    fn a_broken_or_foreign_set_of_masks_is_not_used() {
        let mut masks = MeshMasks::<()>::default();
        masks.take("sconce#stationary_1", &[1; 16], 2, 2);
        assert!(masks.layers_for("sconce", &channels(3)).0.is_empty(), "layer 0 is missing");
        masks.take("sconce#stationary_0", &[0; 4], 1, 1);
        assert!(masks.layers_for("sconce", &channels(3)).0.is_empty(), "the layers differ in size");
        masks.take("sconce#stationary_0", &[0; 16], 2, 2);
        assert!(masks.layers_for("sconce", &channels(1)).0.is_empty(), "baked for more lamps than this scene has");
        assert_eq!(masks.layers_for("sconce", &channels(4)).0.len(), 2);
    }
}
