#![cfg(target_os = "android")]

//! The in-VR hand settings menu: a big world-space panel the player points at
//! with the right controller's laser, adjusting every hand parameter per hand
//! (LEFT/RIGHT tabs) with trigger clicks and slider drags. SAVE persists to
//! `user_calibration.json` in the game dir. Layout, hit-testing and values
//! live in `hand_menu_core`, shared with ssstudio's desktop preview.

use std::sync::Arc;

use space_soup::wgpu;

use glam::{Quat, Vec3};

use hand_menu_core::{Action, Interaction, MenuValues, PANEL_M, PANEL_PX};
use space_soup::renderer::mesh::LoadedTexture;
use space_soup::renderer::xr_renderer::XrRenderer;
use space_soup::renderer::{mesh_pipeline::ModelUniform, Color3, Cuboid, GltfMesh, MeshInstance};
use space_soup::ui2d::{Font, Overlay};
use space_soup::ControllerState;

pub struct HandMenu {
    pub open: bool,
    /// Whether `pos`/`rot` have been set from a real head pose yet. Opening
    /// via the menu button always has a head pose in hand at that instant;
    /// starting open does not, so placement is deferred to the first
    /// `update()` call instead of guessing a position at construction time.
    placed: bool,
    values: MenuValues,
    /// The authored rig values; Reset returns here, not to zeroes.
    authored: MenuValues,
    interaction: Interaction,
    prev_menu_btn: bool,

    // Placement in render space.
    pos: Vec3,
    rot: Quat,

    // Rendering.
    overlay: Overlay,
    texture_view: wgpu::TextureView,
    quad: GltfMesh,
    model: ModelUniform,
    font: Arc<Font>,
    laser: Vec<Cuboid>,
}

const TRIGGER_ON: f32 = 0.6;
const LASER_MAX_M: f32 = 6.0;

impl HandMenu {
    pub fn new(
        renderer: &XrRenderer,
        game_dir: &std::path::Path,
        authored_cfg: &avatar_ik::RigConfig,
    ) -> Option<Self> {
        let font_bytes = std::fs::read(game_dir.join("fonts/menu.ttf"))
            .map_err(|e| log::error!("hand menu: fonts/menu.ttf: {e}"))
            .ok()?;
        let font = Arc::new(Font::new(&font_bytes));

        let (pw, ph) = PANEL_PX;
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let texture = renderer.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("hand_menu_texture"),
            size: wgpu::Extent3d {
                width: pw,
                height: ph,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let overlay = Overlay::new(renderer.device(), format, pw, ph, 1.0);
        let loaded = LoadedTexture::from_texture(
            renderer.device(),
            renderer.mesh_texture_layout(),
            texture,
        );
        let quad = GltfMesh::textured_quad(
            renderer.device(),
            Arc::new(loaded),
            PANEL_M.0,
            PANEL_M.1,
        );

        let authored = MenuValues::from_rig(authored_cfg);
        let values = hand_menu_core::load(game_dir, authored_cfg);
        Some(Self {
            // Starts open: with no scene content to orient by, the hand menu
            // is the only thing there is to see, so it shouldn't be hidden
            // behind a button press on first launch.
            open: true,
            placed: false,
            values,
            authored,
            interaction: Interaction::default(),
            prev_menu_btn: false,
            pos: Vec3::ZERO,
            rot: Quat::IDENTITY,
            overlay,
            texture_view,
            quad,
            model: renderer.create_model_uniform(),
            font,
            laser: Vec::new(),
        })
    }

    pub fn values(&self) -> &MenuValues {
        &self.values
    }

    /// Choose a body or hands (ids from `game/avatars.json`, e.g. from a
    /// selection pad) and remember it. True if it changed.
    pub fn choose(&mut self, body: Option<&str>, hands: Option<&str>, game_dir: &std::path::Path) -> bool {
        let before = (self.values.body.clone(), self.values.hands.clone());
        if let Some(b) = body {
            self.values.body = b.to_string();
        }
        if let Some(h) = hands {
            self.values.hands = h.to_string();
        }
        let changed = (self.values.body.clone(), self.values.hands.clone()) != before;
        if changed {
            hand_menu_core::save(game_dir, &self.values);
        }
        changed
    }

    /// 1.5 m ahead of the head, slightly below eye height, facing back.
    /// Further out than it used to be (1.15m) now that the panel itself is
    /// taller (see `hand_menu_core::PANEL_M`) -- keeps roughly the same
    /// angular size at a glance instead of needing more head movement to
    /// read top to bottom.
    fn place_in_front_of(&mut self, head: (Vec3, Quat)) {
        let fwd = head.1 * Vec3::NEG_Z;
        let flat = Vec3::new(fwd.x, 0.0, fwd.z).normalize_or_zero();
        self.pos = head.0 + flat * 1.5 - Vec3::Y * 0.05;
        // The quad's +Z normal must aim at the player.
        self.rot = Quat::from_rotation_y(flat.x.atan2(flat.z) + std::f32::consts::PI);
        self.placed = true;
    }

    /// One frame: toggle on the menu button, place in front of the head when
    /// opened, laser-pick from the right aim pose, apply trigger interaction,
    /// redraw the panel texture. `head`/`aim` are render-space transforms.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        renderer: &XrRenderer,
        cs: &ControllerState,
        head: (Vec3, Quat),
        aim: (Vec3, Quat),
        game_dir: &std::path::Path,
    ) {
        if cs.btn_menu && !self.prev_menu_btn {
            self.open = !self.open;
            if self.open {
                self.place_in_front_of(head);
            }
        } else if self.open && !self.placed {
            // Started open (see HandMenu::new) -- place it now that a real
            // head pose is finally available, rather than at construction
            // time against a placeholder identity transform.
            self.place_in_front_of(head);
        }
        self.prev_menu_btn = cs.btn_menu;
        self.laser.clear();
        if !self.open {
            return;
        }

        // --- laser vs panel plane ---
        let origin = aim.0;
        let dir = aim.1 * Vec3::NEG_Z;
        let normal = self.rot * Vec3::Z;
        let denom = dir.dot(normal);
        let mut cursor = None;
        let mut laser_end = origin + dir * LASER_MAX_M;
        if denom.abs() > 1e-4 {
            let t = (self.pos - origin).dot(normal) / denom;
            if t > 0.0 && t < LASER_MAX_M {
                let hit = origin + dir * t;
                let local = self.rot.inverse() * (hit - self.pos);
                let (w, h) = PANEL_M;
                if local.x.abs() <= w / 2.0 && local.y.abs() <= h / 2.0 {
                    laser_end = hit;
                    cursor = Some((
                        (local.x + w / 2.0) / w * PANEL_PX.0 as f32,
                        (h / 2.0 - local.y) / h * PANEL_PX.1 as f32,
                    ));
                }
            }
        }

        // --- interaction ---
        let trigger = cs.r_trigger > TRIGGER_ON;
        match self.interaction.update(cursor, trigger, &mut self.values) {
            Some(Action::Save) => hand_menu_core::save(game_dir, &self.values),
            Some(Action::Reset) => {
                // Reset the hand fit, keeping which hand is being edited and
                // what the player is wearing.
                let editing_left = self.values.editing_left;
                let (body, hands) = (std::mem::take(&mut self.values.body), std::mem::take(&mut self.values.hands));
                self.values = self.authored.clone();
                self.values.editing_left = editing_left;
                self.values.body = body;
                self.values.hands = hands;
            }
            Some(Action::Close) => {
                self.open = false;
                return;
            }
            None => {}
        }

        // --- laser visuals ---
        let seg = laser_end - origin;
        let len = seg.length().max(0.01);
        let mid = origin + seg * 0.5;
        let laser_rot = Quat::from_rotation_arc(Vec3::NEG_Z, seg / len);
        self.laser.push(Cuboid {
            rotation: laser_rot,
            ..Cuboid::solid(mid, Vec3::new(0.0025, 0.0025, len / 2.0), Color3(237, 106, 47, 235))
        });
        if cursor.is_some() {
            self.laser.push(Cuboid {
                rotation: self.rot,
                ..Cuboid::solid(laser_end, Vec3::splat(0.008), Color3(242, 236, 219, 255))
            });
        }

        // --- redraw the panel texture ---
        let items = hand_menu_core::build_items(&self.font, &self.values, &self.interaction);
        self.overlay.set_items(items);
        let mut encoder = renderer
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hand_menu_enc"),
            });
        {
            let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hand_menu_clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
        }
        self.overlay
            .render(renderer.device(), renderer.queue(), &mut encoder, &self.texture_view);
        renderer.queue().submit(Some(encoder.finish()));

        self.quad.position = self.pos;
        self.quad.rotation = self.rot;
    }

    pub fn laser_cuboids(&self) -> &[Cuboid] {
        &self.laser
    }

    pub fn mesh_instance(&self) -> Option<MeshInstance<'_>> {
        self.open.then_some(MeshInstance {
            mesh: &self.quad,
            model: &self.model,
            lightmap_key: None,
        })
    }
}
