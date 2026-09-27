#![cfg(target_os = "android")]

use std::sync::mpsc;
use std::time::Duration;

use base64::Engine;
use futures_util::StreamExt;
use serde::Deserialize;
use tokio_tungstenite::tungstenite::Message;

pub struct LightmapUpdate {
    pub object_id: String,
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    /// The light at full precision, when the bake sent it -- see
    /// `space_soup_engine::lightmaps::LightmapEncoding::F16`.
    pub linear: Option<Vec<f32>>,
}

#[derive(Deserialize)]
struct WireLightmapMessage {
    object_id: String,
    width: u32,
    height: u32,
    /// An 8-bit preview for a browser; the headset prefers `hdr_png_b64`.
    png_b64: String,
    #[serde(default)]
    hdr_png_b64: Option<String>,
}

pub fn server_ws_url(scene_name: &str) -> String {
    format!("ws://127.0.0.1:8000/api/lightmap/{scene_name}")
}

pub fn spawn(scene_name: String) -> mpsc::Receiver<LightmapUpdate> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build lightmap client runtime");
        rt.block_on(run_client(scene_name, tx));
    });
    rx
}

async fn run_client(scene_name: String, tx: mpsc::Sender<LightmapUpdate>) {
    let url = server_ws_url(&scene_name);
    loop {
        match tokio_tungstenite::connect_async(&url).await {
            Ok((ws, _)) => {
                log::info!("lightmap: connected to {url}");
                let (_, mut stream) = ws.split();
                loop {
                    match stream.next().await {
                        Some(Ok(Message::Text(text))) => handle_message(&text, &tx),
                        Some(Ok(Message::Close(_))) | None => break,
                        Some(Err(e)) => {
                            log::warn!("lightmap: stream error: {e}");
                            break;
                        }
                        _ => {}
                    }
                }
            }
            Err(e) => {
                log::warn!("lightmap: failed to connect to {url}: {e}");
            }
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

fn handle_message(text: &str, tx: &mpsc::Sender<LightmapUpdate>) {
    let Ok(msg) = serde_json::from_str::<WireLightmapMessage>(text) else {
        return;
    };
    // The half-float file where there is one: the preview beside it is
    // clipped at 1 and quantised to 8 bits, fit for a browser and not for the
    // lighting.
    use space_soup_engine::lightmaps::LightmapEncoding;
    let (b64, encoding) = match &msg.hdr_png_b64 {
        Some(h) => (h.as_str(), LightmapEncoding::F16),
        None => (msg.png_b64.as_str(), LightmapEncoding::Srgb8),
    };
    let Ok(png_bytes) = base64::engine::general_purpose::STANDARD.decode(b64) else {
        return;
    };
    let Some((rgba, linear, _, _)) = space_soup_engine::lightmaps::decode_png_rgba(&png_bytes, encoding) else {
        return;
    };
    let _ = tx.send(LightmapUpdate {
        object_id: msg.object_id,
        width: msg.width,
        height: msg.height,
        rgba,
        linear,
    });
}
