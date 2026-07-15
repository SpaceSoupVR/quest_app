//! Local playback of whatever sounds the server reports as active
//! (`WireWorld::sounds`). Physics/game logic moved server-side, but audio
//! can't follow it there: `kira`'s `AudioManager` is one local device with
//! one listener, and a shared server can never correctly be *the* listener
//! for several independent remote players at once (see
//! `space_soup_engine::audio::SoundEngine`'s doc comment for the long
//! version). So each client plays its own copy locally, spatialized against
//! its own head, triggered by the server's sound-state broadcast.
//!
//! Deliberately not a reuse of `SoundEngine`: that type owns per-object
//! authored data (directional cones, min/max distance, autoplay) that isn't
//! on the wire — `WireSoundState` only carries what's needed for basic
//! positional playback. Faking up `GameObject`s just to satisfy that API
//! would be more confusing than a small purpose-built player.

use std::collections::HashMap;
use std::path::Path;

use glam::{Quat, Vec3};
use kira::listener::ListenerHandle;
use kira::sound::static_sound::{StaticSoundData, StaticSoundHandle};
use kira::sound::PlaybackState;
use kira::track::{SpatialTrackBuilder, SpatialTrackHandle};
use kira::{AudioManager, AudioManagerSettings, Decibels, DefaultBackend, Tween};

use space_soup_protocol::WireSoundState;

struct ActiveClip {
    track: SpatialTrackHandle,
    handle: StaticSoundHandle,
}

pub struct ClientAudio {
    manager: Option<AudioManager>,
    listener: Option<ListenerHandle>,
    clips: HashMap<String, StaticSoundData>,
    playing: HashMap<String, ActiveClip>,
}

fn to_mint_vec3(v: Vec3) -> mint::Vector3<f32> {
    mint::Vector3 { x: v.x, y: v.y, z: v.z }
}
fn to_mint_quat(q: Quat) -> mint::Quaternion<f32> {
    mint::Quaternion {
        v: mint::Vector3 { x: q.x, y: q.y, z: q.z },
        s: q.w,
    }
}

fn linear_to_decibels(linear: f32) -> Decibels {
    let linear = linear.max(0.0);
    if linear <= 0.0001 {
        Decibels::SILENCE
    } else {
        Decibels(20.0 * linear.log10())
    }
}

impl ClientAudio {
    pub fn new() -> Self {
        let mut manager =
            match AudioManager::<DefaultBackend>::new(AudioManagerSettings::default()) {
                Ok(m) => Some(m),
                Err(e) => {
                    log::warn!("ClientAudio: no audio output available ({e}); sounds will be silent");
                    None
                }
            };
        let listener = manager.as_mut().and_then(|m| {
            match m.add_listener(to_mint_vec3(Vec3::ZERO), to_mint_quat(Quat::IDENTITY)) {
                Ok(l) => Some(l),
                Err(e) => {
                    log::warn!("ClientAudio: failed to create audio listener: {e}");
                    None
                }
            }
        });

        Self {
            manager,
            listener,
            clips: HashMap::new(),
            playing: HashMap::new(),
        }
    }

    fn load_clip(&mut self, game_dir: &Path, clip: &str) -> Option<StaticSoundData> {
        if let Some(data) = self.clips.get(clip) {
            return Some(data.clone());
        }
        match StaticSoundData::from_file(game_dir.join(clip)) {
            Ok(data) => {
                self.clips.insert(clip.to_string(), data.clone());
                Some(data)
            }
            Err(e) => {
                log::warn!("ClientAudio: failed to load clip '{clip}': {e}");
                None
            }
        }
    }

    fn start(&mut self, game_dir: &Path, sound: &WireSoundState) {
        let Some(clip) = self.load_clip(game_dir, &sound.clip) else {
            return;
        };
        let clip = clip
            .volume(linear_to_decibels(sound.volume))
            .playback_rate(sound.pitch as f64);
        let clip = if sound.looping { clip.loop_region(..) } else { clip };

        let (Some(manager), Some(listener)) = (self.manager.as_mut(), &self.listener) else {
            return;
        };

        let mut track = match manager.add_spatial_sub_track(
            listener.id(),
            to_mint_vec3(Vec3::from(sound.position)),
            SpatialTrackBuilder::new().distances((1.0, 10.0)),
        ) {
            Ok(t) => t,
            Err(e) => {
                log::warn!(
                    "ClientAudio: could not allocate spatial track for '{}': {e}",
                    sound.object_id
                );
                return;
            }
        };
        let handle = match track.play(clip) {
            Ok(h) => h,
            Err(e) => {
                log::warn!("ClientAudio: could not play '{}': {e}", sound.object_id);
                return;
            }
        };

        self.playing
            .insert(sound.object_id.clone(), ActiveClip { track, handle });
    }

    /// Reconciles currently-playing clips against the server's latest
    /// `sounds` list and updates the local listener pose.
    pub fn update(&mut self, game_dir: &Path, sounds: &[WireSoundState], head: (Vec3, Quat)) {
        if let Some(listener) = self.listener.as_mut() {
            listener.set_position(to_mint_vec3(head.0), Tween::default());
            listener.set_orientation(to_mint_quat(head.1), Tween::default());
        }

        let still_active: std::collections::HashSet<&str> =
            sounds.iter().map(|s| s.object_id.as_str()).collect();
        self.playing.retain(|id, active| {
            let keep = still_active.contains(id.as_str());
            if !keep {
                active.handle.stop(Tween::default());
            }
            keep
        });

        for sound in sounds {
            if !self.playing.contains_key(&sound.object_id) {
                self.start(game_dir, sound);
            }
            let Some(active) = self.playing.get_mut(&sound.object_id) else {
                continue;
            };
            if active.handle.state() == PlaybackState::Stopped {
                self.playing.remove(&sound.object_id);
                continue;
            }
            active
                .track
                .set_position(to_mint_vec3(Vec3::from(sound.position)), Tween::default());
            active.handle.set_volume(linear_to_decibels(sound.volume), Tween::default());
            active.handle.set_playback_rate(sound.pitch as f64, Tween::default());
        }
    }
}

impl Default for ClientAudio {
    fn default() -> Self {
        Self::new()
    }
}
