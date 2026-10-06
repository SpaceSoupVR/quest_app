//! A scene's static meshes, loaded from disk by the client.
//!
//! WHY THIS EXISTS
//!
//! The same gap `scene_lights` was written to close, in the other half of the
//! frame. The client took every mesh from the server snapshot, so a game with
//! no multiplayer server drew NO scene objects at all -- terrain and brushes
//! came from disk and rendered, and every prop, crate and light fixture was
//! simply absent.
//!
//! That is not a subtle failure but it looks like one, because the level is
//! still THERE: walls, ground and lighting all present, and only the objects
//! standing in it missing. A hanging lamp lighting a room while being invisible
//! itself reads as a broken model or a bad path, not as "meshes are not being
//! collected on this path at all".
//!
//! WHY ONLY WHEN THE SERVER IS SILENT
//!
//! A connected server runs `collect_render_meshes` over the same scene file and
//! sends the same meshes. Taking both would draw every prop twice, z-fighting
//! with itself.
use space_soup_engine::{scene::Scene, Manifest};
use space_soup_protocol::WireRenderMesh;

/// Every static mesh in a scene, in the shape a server snapshot would carry.
///
/// The wire shape rather than the renderer's, so this feeds the identical
/// conversion the networked path uses. One place decides how a mesh becomes a
/// draw, and the standalone path cannot drift from the multiplayer one without
/// the drift showing up in both.
pub(crate) fn load(game_dir: &std::path::Path, scene_name: &str) -> Vec<WireRenderMesh> {
    let path = Manifest::scene_path(game_dir, scene_name);
    let scene = match Scene::load(&path) {
        Ok(s) => s,
        Err(e) => {
            log::warn!("scene meshes: {} did not load: {e:#}", path.display());
            return Vec::new();
        }
    };

    let mut out = Vec::new();
    for o in &scene.objects {
        if o.hidden {
            // A hidden object with BOTH a mesh and a light is almost always the
            // authoring mistake this warning exists to name, not an intent.
            //
            // `hidden` on a bare light means "do not draw my placeholder
            // cuboid" -- the editor sets it on every light it creates, because
            // the cuboid is a handle. Attach a fixture mesh to that same object
            // afterwards and the flag keeps its old name while acquiring a
            // second meaning: the lamp stops being drawn too. The beam still
            // works, so the level looks lit by nothing and the asset looks
            // broken. Two of these shipped in `test_room` and cost a headset
            // session to find.
            //
            // Skipping is still correct -- a script that hides an object must
            // be able to hide its mesh -- so this reports rather than repairs.
            if o.mesh.is_some() && !o.lights.is_empty() {
                log::warn!(
                    "scene meshes: '{}' has a mesh and a light but is hidden, so the \
                     fixture will not be drawn while its beam still lights the room. \
                     Clear `hidden` on it unless something means to hide it at runtime.",
                    o.id,
                );
            }
            continue;
        }
        let Some(mesh_ref) = o.mesh.as_ref() else {
            continue;
        };
        out.push(WireRenderMesh {
            id: o.id.clone(),
            path: mesh_ref.path.clone(),
            position: o.cuboid.position.to_array(),
            // The mesh's own orientation offset composed with the object's, the
            // same way `collect_render_meshes` does it -- a model authored
            // facing the wrong way is corrected once, in the asset reference,
            // and both paths have to apply it or a prop faces differently
            // depending on whether a server happens to be running.
            rotation: (o.cuboid.rotation * mesh_ref.rotation_offset).to_array(),
            scale: mesh_ref.scale.to_array(),
            manual_part_blends: Default::default(),
            // Damage and clip gating are RUNTIME state that lives on the
            // server. Standalone there is none, so nothing is hidden and no
            // clip is disabled -- which is the correct starting state, not a
            // placeholder.
            hidden_parts: Vec::new(),
            disabled_clips: Vec::new(),
            // A fixture lights its own bulb. Without this a lamp placed in the
            // editor renders dark standalone while its beam works, which is
            // exactly the sort of half-working that reads as a broken asset.
            emissive_drive: o
                .lights
                .iter()
                .map(space_soup_engine::scene_light::emissive_drive)
                .fold(0.0f32, f32::max),
        });
    }
    log::info!(
        "scene meshes: {} static mesh(es) loaded from disk for '{scene_name}'",
        out.len(),
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_scene(dir: &std::path::Path, body: &str) {
        let scenes = dir.join("scenes");
        std::fs::create_dir_all(&scenes).unwrap();
        std::fs::write(
            dir.join("manifest.json"),
            r#"{"name":"t","version":"0.1.0","entry_scene":"t","scenes":["t"]}"#,
        )
        .unwrap();
        std::fs::write(scenes.join("t.json"), body).unwrap();
    }

    /// A folder of the test's own. Not by the clock alone: the tests run in
    /// parallel and two can read the same time, and then share a folder and
    /// each other's scene -- the lit lamp read the switched-off one's and
    /// failed (2026-10-05). The count makes it the test's own.
    fn tmp() -> std::path::PathBuf {
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let d = std::env::temp_dir().join(format!(
            "ss_scene_meshes_{}_{}_{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_scene_mesh_is_loaded_standalone() {
        // The whole point: with no server, a prop must still be drawn.
        let d = tmp();
        write_scene(&d, r#"{"name":"t","objects":[{
            "id": "crate",
            "cuboid": { "position": [1,2,3], "half_size": [0.5,0.5,0.5] },
            "mesh": { "path": "models/crate.glb" }
        }]}"#);
        let out = load(&d, "t");
        std::fs::remove_dir_all(&d).ok();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].path, "models/crate.glb");
        assert_eq!(out[0].position, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn an_object_with_no_mesh_is_not_a_mesh() {
        let d = tmp();
        write_scene(&d, r#"{"name":"t","objects":[{
            "id": "trigger", "cuboid": { "position": [0,0,0], "half_size": [1,1,1] }
        }]}"#);
        let out = load(&d, "t");
        std::fs::remove_dir_all(&d).ok();
        assert!(out.is_empty());
    }

    #[test]
    fn a_hidden_object_stays_hidden() {
        let d = tmp();
        write_scene(&d, r#"{"name":"t","objects":[{
            "id": "ghost", "hidden": true,
            "cuboid": { "position": [0,0,0], "half_size": [1,1,1] },
            "mesh": { "path": "models/ghost.glb" }
        }]}"#);
        let out = load(&d, "t");
        std::fs::remove_dir_all(&d).ok();
        assert!(out.is_empty(), "an object marked hidden must not be drawn");
    }

    #[test]
    fn a_lit_fixture_carries_its_emissive_drive() {
        // A lamp has to light its own bulb standalone too, or it renders dark
        // while its beam works -- which reads as a broken asset rather than as
        // a missing field on one code path.
        let d = tmp();
        write_scene(&d, r#"{"name":"t","objects":[{
            "id": "lamp",
            "cuboid": { "position": [0,3,0], "half_size": [0.2,0.2,0.2] },
            "mesh": { "path": "models/lamp.glb" },
            "lights": [{ "kind": "Spot", "intensity": 4.0 }]
        }]}"#);
        let out = load(&d, "t");
        std::fs::remove_dir_all(&d).ok();
        assert_eq!(out.len(), 1);
        assert!(out[0].emissive_drive > 0.0, "a lit lamp must drive its bulb");
    }

    /// No scene we SHIP may hide a fixture that carries a light.
    ///
    /// The unit tests above pin the behaviour; this pins the DATA, and the data
    /// is where the bug actually was. `hall_spot_1` and `hall_spot_2` in
    /// `test_room` each had a lamp mesh attached to an object the editor had
    /// already marked hidden when it was a bare light, so both lamps lit the
    /// hall while being invisible. Nothing failed -- the level rendered, the
    /// beams worked, and the only symptom was "I saw no light fixtures".
    ///
    /// Reads the real scenes rather than a fixture string on purpose: a
    /// hand-authored file is exactly what has no other check on it.
    #[test]
    fn no_shipped_scene_hides_a_lit_fixture() {
        let scenes = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("game")
            .join("scenes");
        let Ok(dir) = std::fs::read_dir(&scenes) else {
            // Not a failure: the crate has to build outside this checkout.
            return;
        };
        let mut offenders = Vec::new();
        for entry in dir.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let Ok(scene) = Scene::load(&path) else { continue };
            for o in &scene.objects {
                if o.hidden && o.mesh.is_some() && !o.lights.is_empty() {
                    offenders.push(format!("{}:{}", scene.name, o.id));
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "these objects light a room with an invisible fixture: {offenders:?}",
        );
    }

    #[test]
    fn a_switched_off_fixture_does_not_glow() {
        let d = tmp();
        write_scene(&d, r#"{"name":"t","objects":[{
            "id": "lamp",
            "cuboid": { "position": [0,3,0], "half_size": [0.2,0.2,0.2] },
            "mesh": { "path": "models/lamp.glb" },
            "lights": [{ "kind": "Spot", "intensity": 4.0, "enabled": false }]
        }]}"#);
        let out = load(&d, "t");
        std::fs::remove_dir_all(&d).ok();
        assert_eq!(out[0].emissive_drive, 0.0);
    }
}
