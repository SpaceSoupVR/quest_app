#![cfg(target_os = "android")]

use glam::Vec3;

use space_soup::renderer::{Color3, Cuboid};
use space_soup::ControllerState;

const ROW_COUNT: usize = 6;
// 0..=3 are live-adjusted numeric values; 4 and 5 are one-shot actions.
const ROW_RESET: usize = 4;
const ROW_SAVE: usize = 5;

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CalibrationValues {
    // Multiplies RigConfig's authored rest_curl_* fields. 1.0 = as authored,
    // 0.0 = old flat-bind-at-rest behavior, up to 2.0 = extra relaxed.
    pub rest_curl_scale: f32,
    // Multiplies thumb_touch_curl.
    pub thumb_sensitivity: f32,
    // Overrides RigConfig::hand_scale directly (same 0.5..1.8 range it clamps to).
    pub hand_scale: f32,
    // Additive nudge (metres) onto wrist_position_offset's depth axis.
    pub wrist_offset_z: f32,
}

impl Default for CalibrationValues {
    fn default() -> Self {
        Self { rest_curl_scale: 1.0, thumb_sensitivity: 1.0, hand_scale: 1.0, wrist_offset_z: 0.0 }
    }
}

impl CalibrationValues {
    /// Layers the player's adjustments onto the authored rig config. Called every
    /// frame rather than only on change -- cheap (a clone plus a handful of float
    /// ops), and keeps the menu live without needing a separate "apply" step.
    pub fn apply_to(&self, base: &avatar_ik::RigConfig) -> avatar_ik::RigConfig {
        let mut cfg = base.clone();
        let scale_rest = |v: f32| (v * self.rest_curl_scale).clamp(0.0, 1.0);
        cfg.rest_curl_thumb = scale_rest(cfg.rest_curl_thumb);
        cfg.rest_curl_index = scale_rest(cfg.rest_curl_index);
        cfg.rest_curl_middle = scale_rest(cfg.rest_curl_middle);
        cfg.rest_curl_ring = scale_rest(cfg.rest_curl_ring);
        cfg.rest_curl_little = scale_rest(cfg.rest_curl_little);
        cfg.thumb_touch_curl = (cfg.thumb_touch_curl * self.thumb_sensitivity).clamp(0.0, 1.0);
        cfg.hand_scale = self.hand_scale;
        cfg.wrist_position_offset[2] += self.wrist_offset_z;
        cfg
    }

    fn adjust_row(&mut self, row: usize, delta: f32) {
        match row {
            0 => self.rest_curl_scale = (self.rest_curl_scale + delta).clamp(0.0, 2.0),
            1 => self.thumb_sensitivity = (self.thumb_sensitivity + delta).clamp(0.0, 2.0),
            2 => self.hand_scale = (self.hand_scale + delta * 0.5).clamp(0.5, 1.8),
            3 => self.wrist_offset_z = (self.wrist_offset_z + delta * 0.03).clamp(-0.03, 0.03),
            _ => {}
        }
    }
}

const CALIBRATION_FILE: &str = "user_calibration.json";

pub fn load(game_dir: &std::path::Path) -> CalibrationValues {
    let path = game_dir.join(CALIBRATION_FILE);
    match std::fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str(&text).unwrap_or_else(|e| {
            log::warn!("{CALIBRATION_FILE} failed to parse ({e}) — using defaults");
            CalibrationValues::default()
        }),
        Err(_) => CalibrationValues::default(),
    }
}

fn save(game_dir: &std::path::Path, values: &CalibrationValues) {
    let path = game_dir.join(CALIBRATION_FILE);
    match serde_json::to_string_pretty(values) {
        Ok(text) => match std::fs::write(&path, text) {
            Ok(()) => log::info!("saved hand calibration to {}", path.display()),
            Err(e) => log::warn!("failed to save {CALIBRATION_FILE}: {e}"),
        },
        Err(e) => log::warn!("failed to serialize calibration: {e}"),
    }
}

/// A wrist-anchored settings panel for live-tuning hand feel on the headset,
/// no PC required. Toggled by the (left-hand) menu button; navigated and
/// adjusted with the right hand's thumbstick and trigger.
pub struct CalibrationMenu {
    pub open: bool,
    pub values: CalibrationValues,
    selected: usize,
    prev_menu_btn: bool,
    // Stick-driven row navigation and confirm are edge-triggered (armed only
    // once the stick/trigger returns near neutral) so holding it doesn't
    // repeat-fire every frame.
    stick_y_armed: bool,
    confirm_armed: bool,
}

impl CalibrationMenu {
    pub fn new(values: CalibrationValues) -> Self {
        Self {
            open: false,
            values,
            selected: 0,
            prev_menu_btn: false,
            stick_y_armed: true,
            confirm_armed: true,
        }
    }

    const ADJUST_RATE: f32 = 0.6; // units/sec at full stick deflection, before adjust_row's per-row scaling
    const STICK_DEADZONE: f32 = 0.25;
    const NAV_THRESHOLD: f32 = 0.6;
    const CONFIRM_THRESHOLD: f32 = 0.7;

    pub fn update(&mut self, cs: &ControllerState, dt: f32, game_dir: &std::path::Path) {
        if cs.btn_menu && !self.prev_menu_btn {
            self.open = !self.open;
        }
        self.prev_menu_btn = cs.btn_menu;
        if !self.open {
            return;
        }

        let y = cs.r_stick.y;
        if y.abs() < Self::STICK_DEADZONE {
            self.stick_y_armed = true;
        } else if self.stick_y_armed && y.abs() > Self::NAV_THRESHOLD {
            self.stick_y_armed = false;
            self.selected = if y > 0.0 {
                self.selected.saturating_sub(1)
            } else {
                (self.selected + 1).min(ROW_COUNT - 1)
            };
        }

        let x = cs.r_stick.x;
        if x.abs() > Self::STICK_DEADZONE && self.selected < ROW_RESET {
            self.values.adjust_row(self.selected, x * Self::ADJUST_RATE * dt);
        }

        if cs.r_trigger > Self::CONFIRM_THRESHOLD {
            if self.confirm_armed {
                self.confirm_armed = false;
                match self.selected {
                    ROW_RESET => self.values = CalibrationValues::default(),
                    ROW_SAVE => save(game_dir, &self.values),
                    _ => {}
                }
            }
        } else {
            self.confirm_armed = true;
        }
    }

    /// Builds the panel as plain colored cuboids anchored to the given wrist
    /// pose. There's no world-space text renderer in this engine (`agate` is a
    /// screen-space overlay only), so rows read as bar-graph fills plus a
    /// highlighted selection rather than labels -- fine for the fixed 6-row
    /// layout, and a natural thing to improve once this is validated on
    /// device. Empty when closed or the wrist isn't currently tracked.
    pub fn cuboids(&self, wrist: Option<avatar_ik::Transform>) -> Vec<Cuboid> {
        if !self.open {
            return Vec::new();
        }
        let Some(wrist) = wrist else { return Vec::new() };

        // Just above the wrist, toward the fingers -- a first guess at a
        // "look at your wrist" placement; wants tuning against the real rig
        // once it's visible on device.
        let local_origin = Vec3::new(0.0, 0.06, -0.02);
        let panel_rot = wrist.rotation;
        let to_world = |local: Vec3| wrist.position + panel_rot * local;

        let panel_w = 0.09;
        let panel_h = 0.06;
        let row_h = panel_h / ROW_COUNT as f32;

        let mut out = Vec::with_capacity(ROW_COUNT * 2 + 1);
        out.push(Cuboid {
            rotation: panel_rot,
            ..Cuboid::solid(
                to_world(local_origin),
                Vec3::new(panel_w / 2.0, panel_h / 2.0, 0.002),
                Color3(20, 20, 24, 230),
            )
        });

        for row in 0..ROW_COUNT {
            let row_local = local_origin
                + Vec3::new(0.0, panel_h / 2.0 - row_h * (row as f32 + 0.5), 0.003);

            if row == self.selected {
                out.push(Cuboid {
                    rotation: panel_rot,
                    ..Cuboid::solid(
                        to_world(row_local),
                        Vec3::new(panel_w / 2.0 * 0.95, row_h / 2.0 * 0.9, 0.001),
                        Color3(70, 130, 220, 160),
                    )
                });
            }

            // Value fill for the four numeric rows; Reset/Save show as a
            // full-width bar in their own color so they read as buttons.
            let (fill_t, color) = match row {
                0 => ((self.values.rest_curl_scale / 2.0).clamp(0.0, 1.0), Color3(200, 200, 80, 255)),
                1 => ((self.values.thumb_sensitivity / 2.0).clamp(0.0, 1.0), Color3(200, 140, 80, 255)),
                2 => (((self.values.hand_scale - 0.5) / 1.3).clamp(0.0, 1.0), Color3(120, 200, 120, 255)),
                3 => (
                    ((self.values.wrist_offset_z + 0.03) / 0.06).clamp(0.0, 1.0),
                    Color3(120, 160, 220, 255),
                ),
                4 => (1.0, Color3(200, 80, 80, 255)),
                _ => (1.0, Color3(80, 200, 120, 255)),
            };
            let fill_w = (panel_w * 0.85) * fill_t.max(0.04);
            let fill_x = -panel_w / 2.0 * 0.85 + fill_w / 2.0;
            out.push(Cuboid {
                rotation: panel_rot,
                ..Cuboid::solid(
                    to_world(row_local + Vec3::new(fill_x, 0.0, 0.004)),
                    Vec3::new(fill_w / 2.0, row_h / 2.0 * 0.6, 0.0015),
                    color,
                )
            });
        }

        out
    }
}
