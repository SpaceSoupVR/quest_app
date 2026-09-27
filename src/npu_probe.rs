//! NPU GATE 0, step 2: can THIS APP reach the Hexagon DSP at all?
//!
//! Step 1 (read-only adb, 2026-09-17) found the FastRPC client library listed
//! as public (`libcdsprpc.so` in `/vendor/etc/public.libraries.txt`), an HTP
//! V69 skel on the DSP, and `/dev/adsprpc-smd` readable by others. None of that
//! says whether an app process, under its SELinux domain, may open a session.
//! Only trying from inside the app can.
//!
//! What this does, once, on its own thread, logging each result with the prefix
//! `NPU probe:` and changing nothing:
//! 1. `dlopen("libcdsprpc.so")` -- is the client loadable from the app sandbox?
//! 2. `remote_handle64_open` on a skel the VENDOR already ships
//!    (`libdspCV_skel.so`, seen in `/vendor/lib/rfsa/adsp`), in the default
//!    (signed) protection domain.
//! 3. The same after asking for an UNSIGNED protection domain -- the path a
//!    third-party app's own compute code would have to take.
//!
//! Every handle it opens it closes. A thread that blocks inside the driver
//! only blocks itself; nothing waits on it. See the roadmap doc
//! `docs/quest3-rendering-roadmap-2026-09.md` for what each outcome means.

/// Whether the probe runs at startup. Diagnostic only; costs one thread once.
pub const NPU_ACCESS_PROBE: bool = true;

/// `DSPRPC_CONTROL_UNSIGNED_MODULE` in the Hexagon SDK's `remote.h`.
const DSPRPC_CONTROL_UNSIGNED_MODULE: u32 = 2;
/// `CDSP_DOMAIN_ID` -- the compute DSP, where the HTP (NPU) lives.
const CDSP_DOMAIN_ID: i32 = 3;

/// The vendor skel opened as a reachability test, with its FastRPC URI.
const VENDOR_SKEL_URI: &str =
    "file:///libdspCV_skel.so?dspCV_skel_handle_invoke&_modver=1.0&_dom=cdsp\0";

#[repr(C)]
struct UnsignedModuleControl {
    domain: i32,
    enable: i32,
}

/// The loaded library and the three calls the probe needs.
#[cfg(target_os = "android")]
struct FastRpc {
    open: unsafe extern "C" fn(*const std::ffi::c_char, *mut u64) -> i32,
    close: unsafe extern "C" fn(u64) -> i32,
    session_control: unsafe extern "C" fn(u32, *mut std::ffi::c_void, u32) -> i32,
}

#[cfg(target_os = "android")]
extern "C" {
    fn dlopen(filename: *const std::ffi::c_char, flag: i32) -> *mut std::ffi::c_void;
    fn dlsym(handle: *mut std::ffi::c_void, symbol: *const std::ffi::c_char) -> *mut std::ffi::c_void;
    fn dlerror() -> *const std::ffi::c_char;
}

#[cfg(target_os = "android")]
const RTLD_NOW: i32 = 2;

#[cfg(target_os = "android")]
fn last_dl_error() -> String {
    // SAFETY: dlerror returns null or a NUL-terminated string owned by libdl.
    unsafe {
        let e = dlerror();
        if e.is_null() {
            "no error text".to_string()
        } else {
            std::ffi::CStr::from_ptr(e).to_string_lossy().into_owned()
        }
    }
}

/// Where to look for the FastRPC client, in order.
///
/// BY NAME FIRST, then by absolute path. `libcdsprpc.so` is listed in
/// `/vendor/etc/public.libraries.txt` and exists at all three of these, and the
/// bare name still failed with "library not found" (headset, 2026-09-17): being
/// a public vendor library only puts it in the `sphal` namespace, which an
/// app's own namespace does not necessarily link to by name. An absolute path
/// is the next thing to try before concluding the gate is shut.
#[cfg(target_os = "android")]
const CLIENT_PATHS: [&std::ffi::CStr; 3] = [
    c"libcdsprpc.so",
    c"/vendor/lib64/libcdsprpc.so",
    c"/system/vendor/lib64/libcdsprpc.so",
];

#[cfg(target_os = "android")]
fn load() -> Result<FastRpc, String> {
    // SAFETY: plain libdl calls with NUL-terminated literals; each symbol is
    // checked for null before it is transmuted to the signature `remote.h`
    // declares for it.
    unsafe {
        let mut lib = std::ptr::null_mut();
        for path in CLIENT_PATHS {
            lib = dlopen(path.as_ptr(), RTLD_NOW);
            if !lib.is_null() {
                log::info!("NPU probe: loaded {}", path.to_string_lossy());
                break;
            }
            log::info!(
                "NPU probe: dlopen({}) failed: {}",
                path.to_string_lossy(),
                last_dl_error(),
            );
        }
        if lib.is_null() {
            return Err("every path to the FastRPC client failed".to_string());
        }
        let sym = |name: &std::ffi::CStr| {
            let p = dlsym(lib, name.as_ptr());
            if p.is_null() {
                Err(format!("dlsym({}) failed: {}", name.to_string_lossy(), last_dl_error()))
            } else {
                Ok(p)
            }
        };
        Ok(FastRpc {
            open: std::mem::transmute::<*mut std::ffi::c_void, _>(sym(c"remote_handle64_open")?),
            close: std::mem::transmute::<*mut std::ffi::c_void, _>(sym(c"remote_handle64_close")?),
            session_control: std::mem::transmute::<*mut std::ffi::c_void, _>(sym(c"remote_session_control")?),
        })
    }
}

/// Try to open, and if it opens, close, the vendor skel. Returns the error code.
#[cfg(target_os = "android")]
fn try_open(rpc: &FastRpc, label: &str) -> i32 {
    let mut handle: u64 = 0;
    // SAFETY: the URI is NUL-terminated and `handle` outlives the call.
    let rc = unsafe { (rpc.open)(VENDOR_SKEL_URI.as_ptr().cast(), &mut handle) };
    if rc == 0 {
        log::info!("NPU probe: {label}: remote_handle64_open OK -- a cDSP session opened from the app");
        // SAFETY: `handle` came from a successful open.
        let close_rc = unsafe { (rpc.close)(handle) };
        log::info!("NPU probe: {label}: closed (rc {close_rc:#x})");
    } else {
        log::warn!("NPU probe: {label}: remote_handle64_open failed, rc {rc:#x} ({})", describe(rc));
    }
    rc
}

/// A reading of the FastRPC error codes most likely here. Anything else is
/// logged raw -- the number is the evidence, this is only a hint.
pub fn describe(rc: i32) -> &'static str {
    match rc as u32 {
        0 => "success",
        0x8000_0406 => "AEE_ECONNREFUSED-class: the DSP session was refused",
        0x8000_040D => "AEE_EUNSUPPORTED-class: not supported for this domain/process",
        0x8000_0400..=0x8000_04FF => "a FastRPC transport error (0x800004xx)",
        0x0000_0001..=0x0000_00FF => "an AEE error (likely the skel or its symbol, not the transport)",
        _ => "unrecognised",
    }
}

/// EVERY library `/vendor/etc/public.libraries.txt` offers, tried in turn.
///
/// The point is to tell a blanket policy from a specific one. If the app's
/// namespace refuses all eight, Horizon OS simply does not link app processes to
/// the vendor namespace and no amount of manifest declaration will help. If it
/// refuses only the DSP transport, that is a decision about the NPU, and the
/// others are a route to the hardware that IS open -- `libOpenCL.so` in
/// particular is a compute path to the Adreno, which is the other accelerator on
/// this SoC.
#[cfg(target_os = "android")]
const PUBLIC_VENDOR_LIBS: [&std::ffi::CStr; 8] = [
    c"libcdsprpc.so",
    c"libadsprpc.so",
    c"libsdsprpc.so",
    c"libhexagon.so",
    c"libOpenCL.so",
    c"libeva.so",
    c"libsynx.so",
    c"libdrm.so",
];

/// Which of them the app can actually load, by name.
#[cfg(target_os = "android")]
fn survey_public_libraries() {
    let mut open = Vec::new();
    for name in PUBLIC_VENDOR_LIBS {
        // SAFETY: a NUL-terminated literal, and the handle is deliberately left
        // open -- this runs once and the process keeps whatever it loaded.
        let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
        if handle.is_null() {
            log::info!("NPU probe: vendor lib {} NOT loadable", name.to_string_lossy());
        } else {
            open.push(name.to_string_lossy().into_owned());
        }
    }
    log::info!(
        "NPU probe: SURVEY {} of {} public vendor libraries loadable from the app: [{}]",
        open.len(),
        PUBLIC_VENDOR_LIBS.len(),
        open.join(", "),
    );
}

/// COULD WE WRITE OUR OWN CLIENT INSTEAD?
///
/// FastRPC is a userspace library talking to a character device by ioctl. If the
/// device node were open to us, the vendor library would be a convenience rather
/// than a gate, and a client could be written and bundled. So this asks the
/// device directly, from the app's own SELinux domain, which is the only place
/// the answer means anything.
///
/// `/dev/adsprpc-smd` is `crw-rw-r--  system system  u:object_r:vendor_qdsp_device`
/// -- others may read, not write, and the label is a vendor one. Both of those
/// have to give way, and the second is not something an app can ask for.
#[cfg(target_os = "android")]
fn probe_device_node() {
    const PATHS: [&std::ffi::CStr; 2] = [c"/dev/adsprpc-smd", c"/dev/subsys_cdsp"];
    for path in PATHS {
        for (label, flags) in [("read-write", 2), ("read-only", 0)] {
            // SAFETY: a NUL-terminated literal; the fd is closed immediately.
            let fd = unsafe { libc_open(path.as_ptr(), flags) };
            if fd >= 0 {
                log::info!(
                    "NPU probe: {} opened {} -- a hand-written FastRPC client is worth trying",
                    path.to_string_lossy(),
                    label,
                );
                unsafe { libc_close(fd) };
            } else {
                log::info!(
                    "NPU probe: {} refused {} (errno {})",
                    path.to_string_lossy(),
                    label,
                    std::io::Error::last_os_error(),
                );
            }
        }
    }
}

#[cfg(target_os = "android")]
extern "C" {
    #[link_name = "open"]
    fn libc_open(path: *const std::ffi::c_char, flags: i32) -> i32;
    #[link_name = "close"]
    fn libc_close(fd: i32) -> i32;
}

/// Run the probe on its own thread. Returns immediately.
#[cfg(target_os = "android")]
pub fn spawn() {
    if !NPU_ACCESS_PROBE {
        return;
    }
    let _ = std::thread::Builder::new().name("npu_probe".into()).spawn(|| {
        survey_public_libraries();
        probe_device_node();
        let rpc = match load() {
            Ok(r) => {
                log::info!("NPU probe: libcdsprpc.so loaded from the app process");
                r
            }
            Err(e) => {
                log::warn!("NPU probe: {e} -- the FastRPC client is NOT reachable from the app; Gate 0 fails here");
                return;
            }
        };
        let signed = try_open(&rpc, "default PD");
        let mut ctl = UnsignedModuleControl { domain: CDSP_DOMAIN_ID, enable: 1 };
        // SAFETY: `ctl` is the struct `remote.h` defines for this request, and
        // lives for the call.
        let ctl_rc = unsafe {
            (rpc.session_control)(
                DSPRPC_CONTROL_UNSIGNED_MODULE,
                (&mut ctl as *mut UnsignedModuleControl).cast(),
                std::mem::size_of::<UnsignedModuleControl>() as u32,
            )
        };
        log::info!("NPU probe: request unsigned PD on cDSP: rc {ctl_rc:#x} ({})", describe(ctl_rc));
        let unsigned = try_open(&rpc, "unsigned PD");
        log::info!(
            "NPU probe: SUMMARY loaded=yes default_pd_rc={signed:#x} unsigned_request_rc={ctl_rc:#x} unsigned_pd_rc={unsigned:#x}"
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_skel_uri_is_nul_terminated_and_targets_the_compute_dsp() {
        assert!(VENDOR_SKEL_URI.ends_with('\0'), "a C string without its NUL reads past the end");
        assert!(VENDOR_SKEL_URI.contains("_dom=cdsp"), "the HTP lives on the compute DSP");
        assert_eq!(std::mem::size_of::<UnsignedModuleControl>(), 8, "must match remote.h's two ints");
    }

    #[test]
    fn success_is_described_as_success() {
        assert_eq!(describe(0), "success");
        assert_ne!(describe(0x8000_0406u32 as i32), "unrecognised");
    }
}
