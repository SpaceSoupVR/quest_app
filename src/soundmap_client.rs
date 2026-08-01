#![cfg(target_os = "android")]

use std::sync::mpsc;
use std::time::Duration;

use base64::Engine;
use futures_util::StreamExt;
use glam::Vec3;
use serde::Deserialize;
use tokio_tungstenite::tungstenite::Message;

pub struct SoundmapUpdate {
    pub object_id: String,
    pub grid: OcclusionGrid,
}

pub struct OcclusionGrid {
    width: u32,
    height: u32,
    gray: Vec<u8>,
}

impl OcclusionGrid {
    fn from_rgba(width: u32, height: u32, rgba: &[u8]) -> Self {
        let gray = rgba.chunks_exact(4).map(|px| px[0]).collect();
        Self { width, height, gray }
    }

    pub fn sample(&self, sound_pos: Vec3, max_distance: f32, listener_pos: Vec3) -> f32 {
        if self.width == 0 || self.height == 0 || max_distance <= 0.0 {
            return 0.0;
        }
        let u = ((listener_pos.x - sound_pos.x) / max_distance) * 0.5 + 0.5;
        let v = ((listener_pos.z - sound_pos.z) / max_distance) * 0.5 + 0.5;
        let clear = self.bilinear(u.clamp(0.0, 1.0), v.clamp(0.0, 1.0));
        1.0 - clear
    }

    fn bilinear(&self, u: f32, v: f32) -> f32 {
        let fx = u * (self.width as f32 - 1.0);
        let fy = v * (self.height as f32 - 1.0);
        let x0 = fx.floor() as u32;
        let y0 = fy.floor() as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;

        let at = |x: u32, y: u32| self.gray[(y * self.width + x) as usize] as f32 / 255.0;
        let top = at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx;
        let bottom = at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx;
        top * (1.0 - ty) + bottom * ty
    }
}

#[derive(Deserialize)]
struct WireSoundmapMessage {
    object_id: String,
    width: u32,
    height: u32,
    png_b64: String,
}

pub fn server_ws_url(scene_name: &str) -> String {
    format!("ws://127.0.0.1:8000/api/soundmap/{scene_name}")
}

pub fn spawn(scene_name: String) -> mpsc::Receiver<SoundmapUpdate> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build soundmap client runtime");
        rt.block_on(run_client(scene_name, tx));
    });
    rx
}

async fn run_client(scene_name: String, tx: mpsc::Sender<SoundmapUpdate>) {
    let url = server_ws_url(&scene_name);
    loop {
        match tokio_tungstenite::connect_async(&url).await {
            Ok((ws, _)) => {
                log::info!("soundmap: connected to {url}");
                let (_, mut stream) = ws.split();
                loop {
                    match stream.next().await {
                        Some(Ok(Message::Text(text))) => handle_message(&text, &tx),
                        Some(Ok(Message::Close(_))) | None => break,
                        Some(Err(e)) => {
                            log::warn!("soundmap: stream error: {e}");
                            break;
                        }
                        _ => {}
                    }
                }
            }
            Err(e) => {
                log::warn!("soundmap: failed to connect to {url}: {e}");
            }
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

fn handle_message(text: &str, tx: &mpsc::Sender<SoundmapUpdate>) {
    let Ok(msg) = serde_json::from_str::<WireSoundmapMessage>(text) else {
        return;
    };
    let Ok(png_bytes) = base64::engine::general_purpose::STANDARD.decode(&msg.png_b64) else {
        return;
    };
    let Ok(decoded) = image::load_from_memory(&png_bytes) else {
        return;
    };
    let rgba = decoded.to_rgba8();
    let grid = OcclusionGrid::from_rgba(msg.width, msg.height, rgba.as_raw());
    let _ = tx.send(SoundmapUpdate { object_id: msg.object_id, grid });
}
