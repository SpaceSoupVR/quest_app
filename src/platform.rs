#![cfg(target_os = "android")]

use log::{error, info, warn};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use space_soup::renderer::xr_renderer::XrRenderer;
use space_soup::{Controllers, HandTrackers, Headset, VkContext, XrContext};

const ANDROID_LOOPER_ID_MAIN: u32 = 0;
const ANDROID_LOOPER_ID_INPUT: u32 = 1;

pub(crate) fn pump_android_events(exit: &mut bool) {
    use ndk::looper::{Poll, ThreadLooper};
    let Some(looper) = ThreadLooper::for_thread() else {
        return;
    };
    loop {
        let Ok(Poll::Event { ident, .. }) = looper.poll_all_timeout(std::time::Duration::ZERO)
        else {
            break;
        };
        match ident as u32 {
            ANDROID_LOOPER_ID_MAIN => match ndk_glue::poll_events() {
                Some(ndk_glue::Event::Destroy) => {
                    info!("pump_android_events: activity destroyed");
                    *exit = true;
                }
                Some(_) => {}
                None => break,
            },
            ANDROID_LOOPER_ID_INPUT => {
                // The drain thread usually got there first; nothing left means
                // stop, or a readable queue with no events would spin this loop.
                if drain_input_queue() == 0 {
                    break;
                }
            }
            _ => break,
        }
    }
}

/// Input events finished so far, for the log.
static INPUT_EVENTS_FINISHED: AtomicU64 = AtomicU64::new(0);

/// Finish every Android input event waiting in the queue. Returns how many.
///
/// Finished UNHANDLED: nothing in the app reads these -- controllers and hands
/// arrive through OpenXR -- but Android waits for each one to be finished, and
/// raises "not responding" when one sits for five seconds.
pub(crate) fn drain_input_queue() -> usize {
    let Some(queue) = ndk_glue::input_queue() else {
        return 0;
    };
    let mut taken = 0;
    loop {
        match queue.get_event() {
            Ok(Some(event)) => {
                taken += 1;
                let kind = match &event {
                    ndk::event::InputEvent::KeyEvent(_) => "key",
                    ndk::event::InputEvent::MotionEvent(_) => "motion",
                };
                // A key event is offered to the IME first. If it takes it,
                // Android finishes it and it must not be finished twice.
                if let Some(event) = queue.pre_dispatch(event) {
                    queue.finish_event(event, false);
                }
                let n = INPUT_EVENTS_FINISHED.fetch_add(1, Ordering::Relaxed) + 1;
                if n <= 20 || n % 200 == 0 {
                    info!("input: finished event #{n} ({kind})");
                }
            }
            Ok(None) => break,
            Err(e) => {
                warn!("input: could not read the input queue: {e}");
                break;
            }
        }
    }
    taken
}

/// Keeps Android's input queue empty for the life of the process.
///
/// A thread, because the frame loop is not always there to do it. Startup ran
/// 7.2 s before the loop pumped anything (measured 2026-09-10: begin 23:08:52.7,
/// probes 23:08:59.9), and a key event in that window timed out at 23:08:58 --
/// "Input dispatching timed out ... Waited 5001ms for KeyEvent" -- after which
/// Android kept re-raising the dialog every few seconds. Any later stall in the
/// loop, a scene switch or a pipeline build, would do the same.
pub(crate) fn spawn_input_drain() {
    let spawned = std::thread::Builder::new()
        .name("input-drain".into())
        .spawn(|| loop {
            drain_input_queue();
            std::thread::sleep(std::time::Duration::from_millis(50));
        });
    if let Err(e) = spawned {
        warn!("input: could not start the input drain thread: {e}");
    }
}

#[no_mangle]
pub unsafe extern "C" fn ANativeActivity_onCreate(
    activity: *mut std::ffi::c_void,
    saved_state: *mut std::ffi::c_void,
    saved_state_size: usize,
) {
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Debug)
            .with_tag("quest_app"),
    );
    info!("ANativeActivity_onCreate started");

    ndk_glue::init(activity as _, saved_state as _, saved_state_size, crate::run);
}

pub(crate) fn game_dir() -> PathBuf {
    PathBuf::from("/sdcard/Android/data/com.example.questapp/files/game")
}

/// The headset's lever file, beside the game folder so pushing a new game
/// never wipes it. See `space_soup::renderer::levers`.
pub(crate) fn levers_path() -> PathBuf {
    PathBuf::from("/sdcard/Android/data/com.example.questapp/files/levers.json")
}

pub(crate) struct XrSetup {
    pub(crate) xr: XrContext,
    pub(crate) headset: Headset,
    pub(crate) controllers: Controllers,
    pub(crate) hands: HandTrackers,
    pub(crate) renderer: XrRenderer,
}

pub(crate) fn init_xr() -> Result<XrSetup, Box<dyn std::error::Error>> {
    info!("init: creating XR context");
    let xr = {
        let mut attempts = 0u32;
        loop {
            match XrContext::new() {
                Ok(ctx) => break ctx,
                Err(e) if e.to_string().contains("no more") && attempts < 25 => {
                    warn!(
                        "xr: limit reached — previous session still cleaning up \
                           (attempt {}/25), retrying in 200ms",
                        attempts + 1
                    );
                    std::thread::sleep(std::time::Duration::from_millis(200));
                    attempts += 1;
                }
                Err(e) => return Err(e),
            }
        }
    };
    info!("init: creating Vulkan context");
    let vk = VkContext::new(&xr)?;
    info!("init: creating headset session");
    let headset = Headset::new(&xr, &vk)?;
    info!("init: creating controllers");
    let controllers = Controllers::new(&xr.instance, &headset.session)?;
    info!("init: creating hand trackers");
    let hands = HandTrackers::new(&xr, &headset.session)?;
    info!("init: creating XR renderer");
    let renderer = XrRenderer::new(&vk, &xr, &headset.session)?;
    info!("init: all subsystems ready");

    renderer.device().on_uncaptured_error(std::sync::Arc::new(|error| {
        error!("=== WGPU UNCAPTURED ERROR ===\n{error}\n=============================");
    }));

    Ok(XrSetup { xr, headset, controllers, hands, renderer })
}

pub(crate) const JOINT_NAMES: [&str; 26] = [
    "palm",
    "wrist",
    "thumb_meta",
    "thumb_prox",
    "thumb_dist",
    "thumb_tip",
    "index_meta",
    "index_prox",
    "index_inter",
    "index_dist",
    "index_tip",
    "middle_meta",
    "middle_prox",
    "middle_inter",
    "middle_dist",
    "middle_tip",
    "ring_meta",
    "ring_prox",
    "ring_inter",
    "ring_dist",
    "ring_tip",
    "little_meta",
    "little_prox",
    "little_inter",
    "little_dist",
    "little_tip",
];
