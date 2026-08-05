#![cfg(target_os = "android")]

use log::{error, info, warn};
use std::path::PathBuf;

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
                let Some(queue) = ndk_glue::input_queue() else {
                    break;
                };
                match queue.get_event() {
                    Ok(Some(event)) => queue.finish_event(event, false),
                    _ => break,
                }
            }
            _ => break,
        }
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

    renderer.device().on_uncaptured_error(Box::new(|error| {
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
