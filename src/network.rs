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

use space_soup_protocol::{
    ClientMessage, PlayerId, Pose, ServerMessage, WireHand, WireInputFrame, WireLocomotionInput,
    WirePlayerRig, WireTeleportTarget, WireWorld,
};

use crate::avatar::{LocalPose, RemotePlayerState, Transform};

pub type RemotePlayers = Arc<Mutex<HashMap<PlayerId, RemotePlayerState>>>;
pub type LatestWorld = Arc<Mutex<Option<WireWorld>>>;

#[derive(Debug, Clone)]
pub struct PendingInput {
    pub input: WireInputFrame,
    pub locomotion_input: WireLocomotionInput,
    pub rig: WirePlayerRig,
    pub teleport_target: Option<WireTeleportTarget>,
}

impl Default for PendingInput {
    fn default() -> Self {
        Self {
            input: WireInputFrame::default(),
            locomotion_input: WireLocomotionInput {
                move_stick: (0.0, 0.0),
                turn_stick_x: 0.0,
                teleport_pressed: false,
                teleport_released: false,
                teleport_hand: WireHand::Right,
                player_offset: None,
                player_yaw: None,
            },
            rig: WirePlayerRig::default(),
            teleport_target: None,
        }
    }
}

pub struct NetworkHandle {
    pub local_pose_tx: watch::Sender<LocalPose>,
    pub input_tx: watch::Sender<PendingInput>,
    pub remote_players: RemotePlayers,
    pub latest_world: LatestWorld,
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

pub fn server_url() -> String {
    let path = "/sdcard/Android/data/com.example.questapp/files/server_url.txt";
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "ws://127.0.0.1:9001".to_string())
}

pub fn spawn(server_url: String) -> NetworkHandle {
    let (local_pose_tx, local_pose_rx) = watch::channel(LocalPose {
        head: Transform {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        },
        left_hand: None,
        right_hand: None,
    });
    let (input_tx, input_rx) = watch::channel(PendingInput::default());
    let remote_players: RemotePlayers = Arc::new(Mutex::new(HashMap::new()));
    let latest_world: LatestWorld = Arc::new(Mutex::new(None));

    let thread_remote_players = remote_players.clone();
    let thread_latest_world = latest_world.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build network runtime");
        rt.block_on(run_client(
            server_url,
            local_pose_rx,
            input_rx,
            thread_remote_players,
            thread_latest_world,
        ));
    });

    NetworkHandle {
        local_pose_tx,
        input_tx,
        remote_players,
        latest_world,
    }
}

async fn run_client(
    server_url: String,
    mut local_pose_rx: watch::Receiver<LocalPose>,
    mut input_rx: watch::Receiver<PendingInput>,
    remote_players: RemotePlayers,
    latest_world: LatestWorld,
) {
    loop {
        match tokio_tungstenite::connect_async(&server_url).await {
            Ok((ws, _)) => {
                if let MaybeTlsStream::Plain(tcp) = ws.get_ref() {
                    let _ = tcp.set_nodelay(true);
                }
                log::info!("multiplayer: connected to {server_url}");
                if let Err(e) = run_session(
                    ws,
                    &mut local_pose_rx,
                    &mut input_rx,
                    &remote_players,
                    &latest_world,
                )
                .await
                {
                    log::warn!("multiplayer: session ended: {e}");
                }
            }
            Err(e) => {
                log::warn!("multiplayer: failed to connect to {server_url}: {e}");
            }
        }
        remote_players.lock().unwrap().clear();
        *latest_world.lock().unwrap() = None;
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

async fn run_session(
    ws: WebSocketStream<MaybeTlsStream<TcpStream>>,
    local_pose_rx: &mut watch::Receiver<LocalPose>,
    input_rx: &mut watch::Receiver<PendingInput>,
    remote_players: &RemotePlayers,
    latest_world: &LatestWorld,
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
            changed = input_rx.changed() => {
                changed?;
                let pending = input_rx.borrow_and_update().clone();
                let msg = ClientMessage::Input {
                    input: pending.input,
                    locomotion_input: pending.locomotion_input,
                    rig: pending.rig,
                    teleport_target: pending.teleport_target,
                };
                sink.send(Message::text(serde_json::to_string(&msg)?)).await?;
            }
            incoming = stream.next() => {
                match incoming {
                    Some(Ok(Message::Text(text))) => {
                        handle_server_message(&text, remote_players, latest_world)
                    }
                    Some(Ok(Message::Close(_))) | None => return Ok(()),
                    Some(Err(e)) => return Err(Box::new(e)),
                    _ => {}
                }
            }
        }
    }
}

fn handle_server_message(text: &str, remote_players: &RemotePlayers, latest_world: &LatestWorld) {
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
        ServerMessage::World(world) => {
            *latest_world.lock().unwrap() = Some(world);
        }
    }
}

