#![cfg(target_os = "android")]

// Runs the scene entirely on-device, with no multiplayer server: the same
// `GameRuntime` the droplet server drives, stepped locally and repackaged as
// a `WireWorld` so every downstream system (render_prep, grab_detect,
// part_pull, client_audio, particles...) sees exactly the shape of data it
// already expects from the network path and needs no changes of its own.

use std::collections::HashMap;
use std::path::Path;

use glam::Vec3;
use log::error;

use space_soup_engine::{
    GameRuntime, Hand, HapticRequest, InputFrame, LocomotionInput, PlayerRig, RenderCuboid, RenderLaser,
    RenderLight, RenderMesh, RenderParticleBurst, RenderParticleEmitter, SoundState,
};
use space_soup_protocol::{
    PlayerId, WireColor3, WireCuboidShape, WireCuboidStyle, WireHeldGrip, WireLightKind,
    WireObjectBounds, WireRenderCuboid, WireRenderLaser, WireRenderLight, WireRenderMesh,
    WireRenderParticleBurst, WireRenderParticleEmitter, WireSoundState, WireWorld,
};

pub(crate) struct LocalSim {
    rt: GameRuntime,
}

impl LocalSim {
    /// The scene's one and only `PhysicsWorld` (see grab_detect::StaticScene's
    /// `physics` field doc) -- shared with movement::step_locomotion for the
    /// player's own wall/floor collision instead of standing up a second one.
    pub(crate) fn physics(&self) -> &space_soup_engine::rigid_physics::PhysicsWorld {
        self.rt.physics()
    }

    pub(crate) fn load(game_dir: &Path) -> Option<Self> {
        match GameRuntime::load(game_dir) {
            Ok(rt) => Some(Self { rt }),
            Err(e) => {
                error!("local sim: failed to load GameRuntime from {}: {e}", game_dir.display());
                None
            }
        }
    }

    /// The world before the first step -- scene geometry as authored, nobody
    /// holding anything yet.
    pub(crate) fn initial_world(&self, local_player: PlayerId) -> WireWorld {
        let (cuboids, meshes, lights) = self.rt.render_lists();
        self.wrap(
            local_player,
            cuboids,
            meshes,
            lights,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec3::ZERO,
            0.0,
        )
    }

    /// Advances the simulation one frame with this player's input and
    /// client-authoritative pose, exactly as the server would if it were
    /// reachable. Returns the fresh world plus a pending scene change, if a
    /// script just requested one.
    pub(crate) fn step(
        &mut self,
        dt: f32,
        local_player: PlayerId,
        rig: PlayerRig,
        input: InputFrame,
        locomotion_input: LocomotionInput,
        player_offset: Vec3,
        player_yaw: f32,
    ) -> (WireWorld, Option<String>) {
        let mut inputs = HashMap::new();
        inputs.insert(
            local_player,
            space_soup_engine::PlayerFrameInput {
                rig,
                input,
                locomotion_input,
                teleport_target: None,
                // The headset already simulated its own movement and collision
                // (see movement::step_locomotion) -- adopt it verbatim rather than
                // re-simulating, same as a live server does for this field.
                client_offset: Some(player_offset),
                client_yaw: Some(player_yaw),
            },
        );

        let (cuboids, meshes, lights, particle_emitters, particle_bursts, lasers, scene_change) =
            self.rt.update(dt, &inputs);

        let world = self.wrap(
            local_player,
            cuboids,
            meshes,
            lights,
            particle_emitters,
            particle_bursts,
            lasers,
            player_offset,
            player_yaw,
        );
        (world, scene_change)
    }

    /// Vibrations the last `step` asked for, for this player's own controllers.
    /// Only meaningful right after `step` -- the next step discards the rest.
    pub(crate) fn drain_haptics(&mut self, local_player: PlayerId) -> Vec<HapticRequest> {
        let mut haptics = self.rt.drain_haptics();
        haptics.retain(|h| h.player == local_player);
        haptics
    }

    #[allow(clippy::too_many_arguments)]
    fn wrap(
        &self,
        local_player: PlayerId,
        cuboids: Vec<RenderCuboid>,
        meshes: Vec<RenderMesh>,
        lights: Vec<RenderLight>,
        particle_emitters: Vec<RenderParticleEmitter>,
        particle_bursts: Vec<RenderParticleBurst>,
        lasers: Vec<RenderLaser>,
        player_offset: Vec3,
        player_yaw: f32,
    ) -> WireWorld {
        WireWorld {
            for_player: local_player,
            scene_name: self.rt.scene_name().to_string(),
            cuboids: cuboids.iter().map(cuboid_to_wire).collect(),
            meshes: meshes.iter().map(mesh_to_wire).collect(),
            object_bounds: self.object_bounds(),
            lights: lights.iter().map(light_to_wire).collect(),
            particle_emitters: particle_emitters.iter().map(particle_emitter_to_wire).collect(),
            particle_bursts: particle_bursts.iter().map(particle_burst_to_wire).collect(),
            lasers: lasers.iter().map(laser_to_wire).collect(),
            player_offset: player_offset.to_array(),
            player_yaw,
            left_hand_held: self.held_grip(local_player, Hand::Left),
            right_hand_held: self.held_grip(local_player, Hand::Right),
            sounds: self.rt.active_sounds().iter().map(sound_to_wire).collect(),
        }
    }

    fn object_bounds(&self) -> Vec<WireObjectBounds> {
        self.rt
            .scene()
            .objects
            .iter()
            .filter(|o| !o.hidden)
            .map(|o| WireObjectBounds {
                id: o.id.clone(),
                position: o.cuboid.position.to_array(),
                rotation: o.cuboid.rotation.to_array(),
                half_size: o.cuboid.half_size.to_array(),
                has_collider: true,
            })
            .collect()
    }

    // Reconstructs the one piece of render state that, on a live server,
    // never gets serialized to the wire until it crosses the network: which
    // object a hand is holding. Everything GameRuntime needs to know this
    // (held_grip_point) already lives in the shared engine crate, so this is
    // a local lookup rather than a guess at missing server logic.
    //
    // `drives_pose` is always true here: a second hand steadying an object
    // already held is an edge case this doesn't distinguish, same simplification
    // a naive reading of the wire protocol would make.
    fn held_grip(&self, player: PlayerId, hand: Hand) -> Option<WireHeldGrip> {
        let (obj, gp) = self.rt.held_grip_point(player, hand)?;
        Some(WireHeldGrip {
            object_id: obj.id.clone(),
            point_local_pos: gp.local_pos,
            point_local_rot: gp.local_rot,
            finger_curl: gp.finger_curl.clone(),
            hand_pose: Some(gp.hand_pose()),
            drives_pose: true,
        })
    }
}

fn color_to_wire(c: space_soup_engine::Color3) -> WireColor3 {
    WireColor3(c.0, c.1, c.2, c.3)
}

fn cuboid_style_to_wire(s: space_soup_engine::CuboidStyle) -> WireCuboidStyle {
    match s {
        space_soup_engine::CuboidStyle::Solid => WireCuboidStyle::Solid,
        space_soup_engine::CuboidStyle::Wireframe => WireCuboidStyle::Wireframe,
        space_soup_engine::CuboidStyle::SolidAndWire => WireCuboidStyle::SolidAndWire,
    }
}

fn cuboid_shape_to_wire(s: space_soup_engine::CuboidShape) -> WireCuboidShape {
    match s {
        space_soup_engine::CuboidShape::Box => WireCuboidShape::Box,
        space_soup_engine::CuboidShape::Cylinder => WireCuboidShape::Cylinder,
    }
}

fn light_kind_to_wire(k: space_soup_engine::LightKind) -> WireLightKind {
    match k {
        space_soup_engine::LightKind::Point => WireLightKind::Point,
        space_soup_engine::LightKind::Spot => WireLightKind::Spot,
    }
}

fn cuboid_to_wire(c: &RenderCuboid) -> WireRenderCuboid {
    WireRenderCuboid {
        id: c.id.clone(),
        position: c.position.to_array(),
        half_size: c.half_size.to_array(),
        rotation: c.rotation.to_array(),
        color: color_to_wire(c.color),
        wire_color: color_to_wire(c.wire_color),
        style: cuboid_style_to_wire(c.style),
        reflectivity: c.reflectivity,
        shape: cuboid_shape_to_wire(c.shape),
    }
}

fn mesh_to_wire(m: &RenderMesh) -> WireRenderMesh {
    WireRenderMesh {
        id: m.id.clone(),
        path: m.path.clone(),
        position: m.position.to_array(),
        rotation: m.rotation.to_array(),
        scale: m.scale.to_array(),
        manual_part_blends: m.manual_part_blends.clone(),
        hidden_parts: m.hidden_parts.clone(),
        disabled_clips: m.disabled_clips.clone(),
    }
}

fn light_to_wire(l: &RenderLight) -> WireRenderLight {
    WireRenderLight {
        id: l.id.clone(),
        position: l.position.to_array(),
        direction: l.direction.to_array(),
        kind: light_kind_to_wire(l.kind),
        color: color_to_wire(l.color),
        intensity: l.intensity,
        range: l.range,
        cone_angle_deg: l.cone_angle_deg,
    }
}

fn particle_emitter_to_wire(p: &RenderParticleEmitter) -> WireRenderParticleEmitter {
    WireRenderParticleEmitter {
        id: p.id.clone(),
        position: p.position.to_array(),
        direction: p.direction.to_array(),
        particle_size: p.particle_size,
        spawn_rate: p.spawn_rate,
        color: color_to_wire(p.color),
        lifetime: p.lifetime,
        speed: p.speed,
        spread_deg: p.spread_deg,
        size_growth: p.size_growth,
    }
}

fn particle_burst_to_wire(p: &RenderParticleBurst) -> WireRenderParticleBurst {
    WireRenderParticleBurst {
        id: p.id.clone(),
        position: p.position.to_array(),
        direction: p.direction.to_array(),
        color: color_to_wire(p.color),
        count: p.count,
        speed: p.speed,
        spread_deg: p.spread_deg,
        particle_size: p.particle_size,
        lifetime: p.lifetime,
        elapsed: p.elapsed,
    }
}

fn laser_to_wire(l: &RenderLaser) -> WireRenderLaser {
    WireRenderLaser {
        id: l.id.clone(),
        origin: l.origin.to_array(),
        direction: l.direction.to_array(),
        end: l.end.to_array(),
        color: color_to_wire(l.color),
        beam_width: l.beam_width,
    }
}

fn sound_to_wire(s: &SoundState) -> WireSoundState {
    WireSoundState {
        object_id: s.object_id.clone(),
        clip: s.clip.clone(),
        position: s.position.to_array(),
        volume: s.volume,
        pitch: s.pitch,
        looping: s.looping,
        min_distance: s.min_distance,
        max_distance: s.max_distance,
    }
}
