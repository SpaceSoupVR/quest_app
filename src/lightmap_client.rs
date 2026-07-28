#![cfg(target_os = "android")]

use std::sync::mpsc;
use std::time::Duration;

use base64::Engine;
use futures_util::StreamExt;
use serde::Deserialize;
use tokio_tungstenite::tungstenite::Message;

/// Same background-thread + dedicated current_thread runtime + channel
/// handoff pattern as network.rs's multiplayer client -- keeps the main
/// render thread's polling non-blocking (see LightmapUpdates::try_iter uses
/// in lib.rs), unlike debug_protocol's direct blocking TcpStream calls.
pub struct LightmapUpdate {
    pub object_id: String,
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

#[derive(Deserialize)]
struct WireLightmapMessage {
    object_id: String,
    width: u32,
    height: u32,
    png_b64: String,
}

/// Same hardcoded-localhost-dev-port precedent as debug_protocol's
/// 127.0.0.1:7778 -- this is a local dev-loop tool (the scene editor's own
/// FastAPI server), not a deployed/production endpoint.
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
    let Ok(png_bytes) = base64::engine::general_purpose::STANDARD.decode(&msg.png_b64) else {
        return;
    };
    let Ok(decoded) = image::load_from_memory(&png_bytes) else {
        return;
    };
    let rgba = decoded.to_rgba8();
    let _ = tx.send(LightmapUpdate {
        object_id: msg.object_id,
        width: msg.width,
        height: msg.height,
        rgba: rgba.into_raw(),
    });
}
