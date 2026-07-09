#![cfg(target_os = "android")]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use glam::{Quat, Vec3};
use tokio::net::TcpStream;
use tokio::sync::watch;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

use space_soup_protocol::{ClientMessage, PlayerId, Pose, ServerMessage};

use crate::avatar::{LocalPose, RemotePlayerState, Transform};

pub type RemotePlayers = Arc<Mutex<HashMap<PlayerId, RemotePlayerState>>>;

pub struct NetworkHandle {
    pub local_pose_tx: watch::Sender<LocalPose>,
    pub remote_players: RemotePlayers,
}

fn to_wire(t: Transform) -> Pose {
    Pose {
        position: t.position.to_array(),
        rotation: t.rotation.to_array(),
    }
}

fn from_wire(p: Pose) -> Transform {
    Transform {
        position: Vec3::from(p.position),
        rotation: Quat::from_array(p.rotation),
    }
}

/// Reads the multiplayer server address pushed alongside `game/` (see
/// `game_dir()`'s convention), falling back to a local default so the app
/// still runs standalone if no file was pushed.
pub fn server_url() -> String {
    let path = "/sdcard/Android/data/com.example.questapp/files/server_url.txt";
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "ws://127.0.0.1:9001".to_string())
}

/// Spawns a background OS thread owning a single-threaded Tokio runtime
/// (there's exactly one socket and no parallel work, so a multi-worker pool
/// would just be idle threads on a mobile device). Only plain data crosses
/// back into the frame loop — never engine types like `GameRuntime`.
pub fn spawn(server_url: String) -> NetworkHandle {
    let (local_pose_tx, local_pose_rx) = watch::channel(LocalPose {
        head: Transform {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        },
        left_hand: None,
        right_hand: None,
    });
    let remote_players: RemotePlayers = Arc::new(Mutex::new(HashMap::new()));

    let thread_remote_players = remote_players.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build network runtime");
        rt.block_on(run_client(server_url, local_pose_rx, thread_remote_players));
    });

    NetworkHandle {
        local_pose_tx,
        remote_players,
    }
}

async fn run_client(
    server_url: String,
    mut local_pose_rx: watch::Receiver<LocalPose>,
    remote_players: RemotePlayers,
) {
    loop {
        match tokio_tungstenite::connect_async(&server_url).await {
            Ok((ws, _)) => {
                log::info!("multiplayer: connected to {server_url}");
                if let Err(e) = run_session(ws, &mut local_pose_rx, &remote_players).await {
                    log::warn!("multiplayer: session ended: {e}");
                }
            }
            Err(e) => {
                log::warn!("multiplayer: failed to connect to {server_url}: {e}");
            }
        }
        remote_players.lock().unwrap().clear();
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

async fn run_session(
    ws: WebSocketStream<MaybeTlsStream<TcpStream>>,
    local_pose_rx: &mut watch::Receiver<LocalPose>,
    remote_players: &RemotePlayers,
) -> Result<(), Box<dyn std::error::Error>> {
    let (mut sink, mut stream) = ws.split();

    sink.send(Message::text(serde_json::to_string(&ClientMessage::Join)?))
        .await?;

    loop {
        tokio::select! {
            changed = local_pose_rx.changed() => {
                changed?;
                let pose = *local_pose_rx.borrow_and_update();
                let msg = ClientMessage::Update {
                    head: to_wire(pose.head),
                    left_hand: pose.left_hand.map(to_wire),
                    right_hand: pose.right_hand.map(to_wire),
                };
                sink.send(Message::text(serde_json::to_string(&msg)?)).await?;
            }
            incoming = stream.next() => {
                match incoming {
                    Some(Ok(Message::Text(text))) => handle_server_message(&text, remote_players),
                    Some(Ok(Message::Close(_))) | None => return Ok(()),
                    Some(Err(e)) => return Err(Box::new(e)),
                    _ => {}
                }
            }
        }
    }
}

fn handle_server_message(text: &str, remote_players: &RemotePlayers) {
    let Ok(msg) = serde_json::from_str::<ServerMessage>(text) else {
        return;
    };
    match msg {
        ServerMessage::Welcome { id } => {
            log::info!("multiplayer: joined as {id:?}");
        }
        ServerMessage::PlayerUpdate {
            id,
            head,
            left_hand,
            right_hand,
        } => {
            let state = RemotePlayerState {
                head: from_wire(head),
                left_hand: left_hand.map(from_wire),
                right_hand: right_hand.map(from_wire),
            };
            remote_players.lock().unwrap().insert(id, state);
        }
        ServerMessage::PlayerLeft { id } => {
            remote_players.lock().unwrap().remove(&id);
        }
        // Authoritative-simulation broadcast — quest_app doesn't send
        // ClientMessage::Input yet (it still runs its own local GameRuntime;
        // becoming a thin terminal that renders this instead is a bigger,
        // separate cutover), so it never receives one of these in practice.
        // Ignored rather than left unhandled so a real server response
        // doesn't crash a match instead of just being a no-op.
        ServerMessage::World(_) => {}
    }
}
