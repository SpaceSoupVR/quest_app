#!/usr/bin/env bash

set -e

# --print-env: resolve the NDK toolchain, print it as shell exports, and stop
# before doing any building or deploying. Lets `cargo` be driven directly
# without anyone hand-copying paths that then go stale.
PRINT_ENV=0

# --data-only: push game/ to the headset and relaunch, skipping the native
# build, the gradle build and the APK install entirely.
#
# Scene JSON, models and sounds are DATA. Changing a prop's position does not
# change a single byte of the APK, but the only path to the headset used to be
# `cargo build --release` + `gradlew assembleDebug` + `adb install`, so testing
# a level edit in VR cost minutes of compiling code that had not changed. This
# is the same push the full run does, without the part that is not needed.
#
# Assumes the app is already installed; if it is not, run without the flag once.
DATA_ONLY=false

case "${1:-}" in
    --print-env) PRINT_ENV=1 ;;
    --data-only)
        DATA_ONLY=true
        # Non-interactive by construction: nothing here needs a decision, and
        # this path is driven from the editor over a websocket where a `read`
        # prompt would hang forever.
        WANT_CLEAN=false
        WANT_DEPLOY=true
        WANT_EDITOR=false
        WANT_DASHBOARD=false
        ;;
    -h|--help)
        echo "usage: $0 [--print-env | --data-only]"
        echo "  --print-env  print the detected Android toolchain as shell exports and exit"
        echo "  --data-only  push game/ to the headset and relaunch; no build, no install"
        exit 0
        ;;
esac

if [ -t 1 ]; then
    C_STEP=$'\033[1;36m'
    C_OK=$'\033[1;32m'
    C_WARN=$'\033[1;33m'
    C_ERR=$'\033[1;31m'
    C_DIM=$'\033[2m'
    C_RESET=$'\033[0m'
else
    C_STEP=""; C_OK=""; C_WARN=""; C_ERR=""; C_DIM=""; C_RESET=""
fi

step()  { printf '\n%s==>%s %s%s%s\n'   "$C_STEP" "$C_RESET" "$C_STEP" "$1" "$C_RESET"; }
ok()    { printf '    %s%s%s\n'         "$C_OK" "$1" "$C_RESET"; }
warn()  { printf '    %sWARNING:%s %s\n' "$C_WARN" "$C_RESET" "$1"; }
fail()  { printf '    %sERROR:%s %s\n'  "$C_ERR" "$C_RESET" "$1"; }
detail(){ printf '    %s%s%s\n'         "$C_DIM" "$1" "$C_RESET"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XR_DIR="$(dirname "$SCRIPT_DIR")"
QUEST_APP_DIR="$SCRIPT_DIR"
DEBUG_VIEWER_DIR="$XR_DIR/space_soup_editor"
SPACE_SOUP_DIR="$XR_DIR/space_soup"
GAME_DIR="$XR_DIR/game"
PACKAGE="com.example.questapp"
REMOTE_GAME_DIR="/sdcard/Android/data/$PACKAGE/files/game"
REMOTE_SERVER_URL_FILE="/sdcard/Android/data/$PACKAGE/files/server_url.txt"
SERVER_URL="${QUEST_SERVER_URL:-ws://137.184.21.78:9001}"
DROPLET_SSH_HOST="${QUEST_SERVER_SSH_HOST:-root@137.184.21.78}"
DROPLET_SSH_KEY="${QUEST_SERVER_SSH_KEY:-$HOME/vr_digitalocean}"
DROPLET_SERVICE="space-soup-server.service"
HOST_TARGET=$(rustc -vV | awk '/host:/ {print $2}')
DASHBOARD_SESSION="quest_app"

# ── Host / SDK / NDK discovery ───────────────────────────────────────────────
# Previously this hardcoded macOS: $HOME/Library/Android/sdk and a
# darwin-x86_64 toolchain. That works on exactly one developer's machine layout
# and fails on Linux, on a different SDK location, or under a different user.
case "$(uname -s)" in
    Darwin) NDK_PREBUILT="darwin-x86_64"; DEFAULT_SDK="$HOME/Library/Android/sdk"; PKG_HINT="brew install tmux" ;;
    Linux)  NDK_PREBUILT="linux-x86_64";  DEFAULT_SDK="$HOME/Android/Sdk";         PKG_HINT="sudo apt install tmux  (or your distro's equivalent)" ;;
    MINGW*|MSYS*|CYGWIN*)
        # Git Bash / MSYS on Windows. Do not guess a toolchain here: MSYS also
        # rewrites leading-slash arguments into Windows paths, so adb targets
        # like /sdcard/... intermittently become C:/Program Files/Git/sdcard/...
        # and the push appears to fail at random. run_quest.ps1 is the supported
        # Windows path and has full parity with this script.
        fail "Running under $(uname -s). Use PowerShell instead: .\\run_quest.ps1"
        detail "Git Bash mangles adb's /sdcard/... paths, so deploys fail unpredictably."
        exit 1 ;;
    *)      NDK_PREBUILT="linux-x86_64";  DEFAULT_SDK="$HOME/Android/Sdk";         PKG_HINT="install tmux with your package manager" ;;
esac

# Everything from here to the end of the toolchain check exists to compile Rust
# for Android, and is skipped entirely on a data-only push.
#
# Not just a speed saving: this block ends in `exit 1` when the NDK is missing,
# so without the guard a level designer with adb but no Android toolchain could
# not push a scene they had just authored -- the script would refuse over a
# compiler it was never going to invoke.
if ! $DATA_ONLY; then

# Honour the standard variables first, then the per-OS default, then the other
# common install locations, so nobody has to edit this file to build.
SDK_HOME="${ANDROID_SDK_ROOT:-${ANDROID_SDK_HOME:-${ANDROID_HOME:-}}}"
if [ -z "$SDK_HOME" ] || [ ! -d "$SDK_HOME" ]; then
    for candidate in "$DEFAULT_SDK" "$HOME/Library/Android/sdk" "$HOME/Android/Sdk" "/usr/local/lib/android/sdk" "/opt/android-sdk"; do
        if [ -d "$candidate" ]; then SDK_HOME="$candidate"; break; fi
    done
fi

NDK_HOME="${ANDROID_NDK_ROOT:-${ANDROID_NDK_HOME:-}}"
if [ -z "$NDK_HOME" ] || [ ! -d "$NDK_HOME" ]; then
    # Newest NDK that actually has a toolchain for THIS host, rather than the
    # newest directory name -- a partially-installed NDK sorts highest and then
    # fails at link time with a confusing missing-compiler error.
    #
    # Deliberately no `sort -V` and no `tac`: neither exists on macOS. Using them
    # here made the pipeline produce nothing on a Mac, so the loop found zero
    # candidates and the script reported "Android NDK not found" while pointing
    # at the directory the NDK was sitting in. A portability script that is not
    # itself portable is worse than no script -- the error blames the user's
    # install. Ordering by dot-separated numeric fields works on BSD and GNU sort
    # alike, and `tail -1` replaces `tac | head`.
    candidates=""
    for dir in "$SDK_HOME"/ndk/*; do
        if [ -d "$dir/toolchains/llvm/prebuilt/$NDK_PREBUILT/bin" ]; then
            candidates="$candidates$(basename "$dir")
"
        fi
    done
    # Built with an explicit `if` rather than `[ ... ] && ...`: under `set -e` a
    # trailing false test is the loop's exit status and would kill the script.
    if [ -n "$candidates" ]; then
        newest=$(printf '%s' "$candidates" | sort -t. -k1,1n -k2,2n -k3,3n | tail -1)
        NDK_HOME="$SDK_HOME/ndk/$newest"
    fi
fi

if [ ! -d "$NDK_HOME" ]; then
    fail "Android NDK not found. Set ANDROID_NDK_ROOT, or install it via the SDK manager."
    detail "Looked under: ${SDK_HOME:-<no SDK found>}/ndk"
    exit 1
fi

NDK_BIN="$NDK_HOME/toolchains/llvm/prebuilt/$NDK_PREBUILT/bin"

# API level from gradle rather than hardcoded. Building below minSdk fails at
# link time as `ld.lld: error: unable to find library -lvulkan` (Vulkan ships
# from API 24 up), which reads as a missing dependency rather than a wrong level.
MIN_SDK=$(awk '/minSdk/ {print $2; exit}' "$QUEST_APP_DIR/android/app/build.gradle" 2>/dev/null || echo 29)
case "$MIN_SDK" in ''|*[!0-9]*) MIN_SDK=29 ;; esac
if [ "$MIN_SDK" -lt 24 ]; then MIN_SDK=24; fi

# The toolchain lives here, not in .cargo/config.toml, which deliberately holds
# no absolute paths any more.
#
# It used to pin one developer's macOS NDK, and these exports were believed to
# override it. They did not: cargo's [env] set the HYPHENATED names
# (CC_aarch64-linux-android) while these are UNDERSCORED, which are different
# variables -- a non-forced [env] entry only yields to a real variable of the
# same name. So cc-rs kept resolving the macOS path on every other machine, and
# reported it as a missing clang++.exe far downstream. cc-rs accepts either
# spelling, so with that file cleaned out these are what take effect.
#
# Without ANDROID_NDK_ROOT, physx-sys panics with
# `environment variable "ANDROID_NDK_ROOT" has not been set`.
export ANDROID_NDK_ROOT="$NDK_HOME"
export ANDROID_NDK_HOME="$NDK_HOME"
export CC_aarch64_linux_android="$NDK_BIN/aarch64-linux-android$MIN_SDK-clang"
export CXX_aarch64_linux_android="$NDK_BIN/aarch64-linux-android$MIN_SDK-clang++"
export AR_aarch64_linux_android="$NDK_BIN/llvm-ar"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$CC_aarch64_linux_android"

for tool in "$CC_aarch64_linux_android" "$CXX_aarch64_linux_android" "$AR_aarch64_linux_android"; do
    if [ ! -x "$tool" ]; then
        fail "NDK tool missing or not executable: $tool"
        exit 1
    fi
done
detail "NDK: $NDK_HOME ($NDK_PREBUILT, API $MIN_SDK)"

fi  # end: not DATA_ONLY (toolchain discovery)

# Emit the discovered toolchain as shell exports, so a plain `cargo build` gets
# the same values this script would use:  eval "$(./run.sh --print-env)"
# Discovery has to stay in one place; duplicating these paths into a shell
# profile is how they go stale and how absolute paths end up committed again.
if [ "${PRINT_ENV:-0}" = "1" ]; then
    printf 'export ANDROID_NDK_ROOT=%s
' "$NDK_HOME"
    printf 'export ANDROID_NDK_HOME=%s
' "$NDK_HOME"
    printf 'export CC_aarch64_linux_android=%s
' "$CC_aarch64_linux_android"
    printf 'export CXX_aarch64_linux_android=%s
' "$CXX_aarch64_linux_android"
    printf 'export AR_aarch64_linux_android=%s
' "$AR_aarch64_linux_android"
    printf 'export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER=%s
' "$CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER"
    exit 0
fi

cleanup() {
    # A data-only run started neither of these. `gradlew --stop` in particular
    # is not free -- it boots a JVM to tell a daemon we never launched to go
    # away, which is several seconds on a path whose entire point is finishing
    # in a few.
    $DATA_ONLY && return 0
    tmux kill-session -t "$DASHBOARD_SESSION" 2>/dev/null || true
    (cd "$QUEST_APP_DIR/android" && ./gradlew --stop >/dev/null 2>&1) || true
}
trap cleanup EXIT

WANT_DASHBOARD="${WANT_DASHBOARD:-true}"

if $WANT_DASHBOARD && ! command -v tmux >/dev/null 2>&1; then
    fail "tmux not found — install it with: $PKG_HINT"
    exit 1
fi

if [ -z "${WANT_CLEAN+x}" ]; then
    read -r -p "Clean all builds first? [y/N] " clean_reply
    case "$clean_reply" in
        [yY]*) WANT_CLEAN=true ;;
        *) WANT_CLEAN=false ;;
    esac
fi

if [ -z "${WANT_DEPLOY+x}" ]; then
    read -r -p "Upload to headset and run? [Y/n] " deploy_reply
    case "$deploy_reply" in
        [nN]*) WANT_DEPLOY=false ;;
        *) WANT_DEPLOY=true ;;
    esac
fi

if [ -z "${WANT_EDITOR+x}" ]; then
    WANT_EDITOR=false
    if $WANT_DEPLOY && [ -d "$DEBUG_VIEWER_DIR" ]; then
        WANT_EDITOR=true
        read -r -p "Do you want the scene editor? [Y/n] " editor_reply
        case "$editor_reply" in
            [nN]*) WANT_EDITOR=false ;;
            *) WANT_EDITOR=true ;;
        esac
    fi
fi

wait_for_remote_path() {
    local path="$1"
    local tries=0
    until adb shell "[ -e '$path' ] && echo exists" 2>/dev/null | grep -q exists; do
        tries=$((tries + 1))
        if [ "$tries" -ge 20 ]; then
            fail "timed out waiting for $path to exist on device."
            return 1
        fi
        sleep 1
    done
    return 0
}

wait_for_app_data_dir() {
    local data_dir="/sdcard/Android/data/$PACKAGE"
    step "Waiting for app data directory ($data_dir)..."

    if wait_for_remote_path "$data_dir" 2>/dev/null; then
        ok "Data directory exists."
        return 0
    fi

    detail "Data directory missing — launching app once to create it..."
    adb shell am start -n "$PACKAGE/android.app.NativeActivity" >/dev/null 2>&1 || true
    sleep 2
    adb shell am force-stop "$PACKAGE" >/dev/null 2>&1 || true

    if ! wait_for_remote_path "$data_dir"; then
        fail "app data directory never appeared. Is the package name correct?"
        exit 1
    fi
    ok "Data directory created."
}

# True if the remote file exists and is non-empty. Pure predicate — never exits.
verify_remote_file() {
    local path="$1" size
    size=$(adb shell "stat -c %s '$path' 2>/dev/null || echo 0" | tr -d '\r')
    [ "${size:-0}" -ge 1 ] 2>/dev/null
}

# Files that failed to push after retries — reported at the end instead of aborting.
PUSH_FAILURES=()

# push_file SRC DST — adb push with one retry + verify. Records failures instead of
# killing the whole run. Previously a single bad file aborted the entire push via
# `set -e` + verify_remote_file's `exit 1`, silently leaving every later asset (other
# models, the editor's external .bin animation buffers) off the headset. Always
# returns 0 so `set -e` can never trip on it.
push_file() {
    local src="$1" dst="$2" attempt size
    for attempt in 1 2; do
        if adb push "$src" "$dst" >/dev/null 2>&1 && verify_remote_file "$dst"; then
            size=$(adb shell "stat -c %s '$dst' 2>/dev/null || echo 0" | tr -d '\r')
            ok "OK: ${dst#"$REMOTE_GAME_DIR"/} ($size bytes)"
            return 0
        fi
        if [ "$attempt" -eq 1 ]; then
            warn "push failed for ${dst#"$REMOTE_GAME_DIR"/} — retrying..."
        fi
    done
    fail "could not push ${dst#"$REMOTE_GAME_DIR"/} after 2 attempts"
    PUSH_FAILURES+=("$dst")
    return 0
}

wait_for_tcp_listener() {
    local port="$1"
    local tries=0
    step "Waiting for listener on :$port..."
    # bash's own /dev/tcp rather than nc: netcat is absent by default on many
    # Linux images, and its -z flag differs between the BSD and GNU variants.
    until (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; do
        tries=$((tries + 1))
        if [ "$tries" -ge 30 ]; then
            fail "nothing started listening on :$port after 30s."
            exit 1
        fi
        sleep 1
    done
    ok "Listener on :$port is ready."
}

if $WANT_CLEAN; then
    step "Cleaning space_soup..."
    cd "$SPACE_SOUP_DIR"
    cargo clean

    if $WANT_EDITOR; then
        step "Cleaning debug_viewer..."
        cd "$DEBUG_VIEWER_DIR"
        cargo clean
    fi

    step "Cleaning quest_app..."
    cd "$QUEST_APP_DIR"
    cargo clean
    cd android && ./gradlew clean && cd ..
fi

if $WANT_EDITOR; then
    step "Pre-building debug_viewer..."
    cd "$DEBUG_VIEWER_DIR"
    cargo build --target "$HOST_TARGET"
fi

if $DATA_ONLY; then
    step "Data-only run — skipping cargo, gradle and APK install."
else

step "Building quest_app for Android..."
cd "$QUEST_APP_DIR"
cargo build --target aarch64-linux-android --release
mkdir -p android/jniLibs/arm64-v8a
cp target/aarch64-linux-android/release/libquest_app.so android/jniLibs/arm64-v8a/

CXX_SHARED="$NDK_HOME/toolchains/llvm/prebuilt/$NDK_PREBUILT/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
if [ -f "$CXX_SHARED" ]; then
    step "Copying libc++_shared.so into jniLibs ..."
    cp "$CXX_SHARED" android/jniLibs/arm64-v8a/libc++_shared.so
else
    warn "libc++_shared.so not found at $CXX_SHARED — app will fail to load on device"
fi

cd android
./gradlew assembleDebug

fi  # end: not DATA_ONLY

if $WANT_DEPLOY; then
    if ! $DATA_ONLY; then
    step "Installing APK..."
    adb install -r app/build/outputs/apk/debug/app-debug.apk

    tries=0
    until adb shell pm list packages | grep -q "$PACKAGE"; do
        tries=$((tries + 1))
        if [ "$tries" -ge 15 ]; then
            fail "$PACKAGE not found in package list after install."
            exit 1
        fi
        sleep 1
    done
    ok "Package installed and registered."
    fi  # end: not DATA_ONLY

    cd "$QUEST_APP_DIR"

    if $WANT_EDITOR; then
        step "Reversing TCP debug_viewer port 7778..."
        adb reverse tcp:7778 tcp:7778
    fi

    if [ ! -f "$GAME_DIR/manifest.json" ]; then
        fail "$GAME_DIR/manifest.json not found. Aborting."
        exit 1
    fi

    wait_for_app_data_dir

    step "Pushing game folder to Quest..."
    adb shell mkdir -p "$REMOTE_GAME_DIR"
    push_file "$GAME_DIR/manifest.json" "$REMOTE_GAME_DIR/manifest.json"

    if [ -f "$GAME_DIR/avatar_rig.json" ]; then
        push_file "$GAME_DIR/avatar_rig.json" "$REMOTE_GAME_DIR/avatar_rig.json"
    fi

    if [ -d "$GAME_DIR/scenes" ]; then
        adb shell mkdir -p "$REMOTE_GAME_DIR/scenes"
        shopt -s nullglob
        for f in "$GAME_DIR"/scenes/*.json; do
            push_file "$f" "$REMOTE_GAME_DIR/scenes/$(basename "$f")"
        done
        shopt -u nullglob
    fi

    if [ -d "$GAME_DIR/models" ]; then
        while IFS= read -r -d '' d; do
            rel="${d#"$GAME_DIR"/models}"
            adb shell mkdir -p "$REMOTE_GAME_DIR/models$rel"
        done < <(find "$GAME_DIR/models" -type d -print0)

        # -type f pulls in EVERYTHING: .glb meshes AND their sidecar .bin animation
        # buffers / .json matrices. push_file keeps going if any single file fails.
        while IFS= read -r -d '' f; do
            rel="${f#"$GAME_DIR"/models/}"
            push_file "$f" "$REMOTE_GAME_DIR/models/$rel"
        done < <(find "$GAME_DIR/models" -type f -print0)
    fi

    if [ -d "$GAME_DIR/sound" ]; then
        adb shell mkdir -p "$REMOTE_GAME_DIR/sound"
        shopt -s nullglob
        for f in "$GAME_DIR"/sound/*; do
            push_file "$f" "$REMOTE_GAME_DIR/sound/$(basename "$f")"
        done
        shopt -u nullglob
    fi

    if [ "${#PUSH_FAILURES[@]}" -gt 0 ]; then
        fail "${#PUSH_FAILURES[@]} file(s) FAILED to reach the headset:"
        for f in "${PUSH_FAILURES[@]}"; do detail "  ${f#"$REMOTE_GAME_DIR"/}"; done
        warn "Scene may render incomplete (missing meshes or .bin animation buffers). Re-run to retry."
    else
        ok "Game folder verified on device — all files pushed."
    fi

    # Skipped on a data-only push: it is an SSH round trip to a droplet that a
    # scene edit cannot possibly have affected, and the whole point of this path
    # is that it finishes in seconds.
    if $DATA_ONLY; then
        detail "Data-only — skipping multiplayer server check and server_url push."
    else
    step "Verifying multiplayer server is running on $DROPLET_SSH_HOST..."
    if [ -f "$DROPLET_SSH_KEY" ]; then
        if ssh -i "$DROPLET_SSH_KEY" -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new \
            "$DROPLET_SSH_HOST" "systemctl is-active --quiet $DROPLET_SERVICE || systemctl restart $DROPLET_SERVICE"; then
            ok "space-soup-server is active."
        else
            warn "Could not confirm space-soup-server is running on $DROPLET_SSH_HOST — multiplayer may not work. Check manually with: ssh -i $DROPLET_SSH_KEY $DROPLET_SSH_HOST systemctl status $DROPLET_SERVICE"
        fi
    else
        warn "SSH key not found at $DROPLET_SSH_KEY — skipping multiplayer server check (set QUEST_SERVER_SSH_KEY to override)."
    fi

    step "Pushing multiplayer server URL ($SERVER_URL)..."
    adb shell mkdir -p "$(dirname "$REMOTE_SERVER_URL_FILE")"
    echo "$SERVER_URL" | adb shell "cat > '$REMOTE_SERVER_URL_FILE'"
    if verify_remote_file "$REMOTE_SERVER_URL_FILE"; then
        ok "OK: server_url.txt ($SERVER_URL)"
    else
        warn "server_url.txt may not have written — client could fall back to its default server."
    fi
    fi  # end: not DATA_ONLY

    if $WANT_DASHBOARD; then
        step "Starting dev dashboard (tmux, running in background)..."

        tmux kill-session -t "$DASHBOARD_SESSION" 2>/dev/null || true
        tmux new-session -d -s "$DASHBOARD_SESSION" -n dev -x 220 -y 52

        tmux set-option -t "$DASHBOARD_SESSION" -g mouse on
        tmux set-option -t "$DASHBOARD_SESSION" -g status-style "bg=colour234,fg=colour51"
        tmux set-option -t "$DASHBOARD_SESSION" -g status-left "#[bold]  Quest App Dev  #[default]"
        tmux set-option -t "$DASHBOARD_SESSION" -g status-left-length 20
        tmux set-option -t "$DASHBOARD_SESSION" -g status-right "#[fg=colour244]%H:%M:%S "
        tmux set-option -t "$DASHBOARD_SESSION" -g pane-border-status top
        tmux set-option -t "$DASHBOARD_SESSION" -g pane-border-format " #{pane_title} "
        tmux set-option -t "$DASHBOARD_SESSION" -g pane-active-border-style "fg=colour51"
        tmux set-option -t "$DASHBOARD_SESSION" -g pane-border-style "fg=colour238"

        PANE_LOGCAT=$(tmux display-message -t "$DASHBOARD_SESSION:dev" -p '#{pane_id}')
        tmux send-keys -t "$PANE_LOGCAT" \
            "export DISABLE_AUTO_UPDATE=true; cd '$QUEST_APP_DIR' && adb logcat -s quest_app" C-m
        tmux select-pane -t "$PANE_LOGCAT" -T "logcat"

        if $WANT_EDITOR; then
            PANE_VIEWER=$(tmux split-window -h -t "$PANE_LOGCAT" -l "60%" -P -F '#{pane_id}')
            tmux send-keys -t "$PANE_VIEWER" \
                "export DISABLE_AUTO_UPDATE=true; cd '$DEBUG_VIEWER_DIR' && cargo run --target $HOST_TARGET" C-m
            tmux select-pane -t "$PANE_VIEWER" -T "debug_viewer"

            PANE_KEEPALIVE=$(tmux split-window -v -t "$PANE_LOGCAT" -l "40%" -P -F '#{pane_id}')
            tmux send-keys -t "$PANE_KEEPALIVE" \
                "export DISABLE_AUTO_UPDATE=true; while true; do adb reverse tcp:7778 tcp:7778 2>/dev/null; sleep 5; done" C-m
            tmux select-pane -t "$PANE_KEEPALIVE" -T "adb reverse keepalive"

            tmux select-pane -t "$PANE_VIEWER"
        fi

        ok "Dashboard running in background tmux session (no window opened)."
        detail "Attach any time with: tmux attach -t $DASHBOARD_SESSION"

        if $WANT_EDITOR; then
            wait_for_tcp_listener 7778
        fi
    fi

    if $DATA_ONLY; then
        # GameRuntime::load reads game/ once at startup, so a running app would
        # happily carry on showing the scene it loaded before the push. `am
        # start` alone resumes an existing process; the force-stop is what makes
        # the pushed data take effect.
        step "Restarting quest_app so it re-reads the pushed data..."
        adb shell am force-stop "$PACKAGE" >/dev/null 2>&1 || true
    else
        step "Launching quest_app on headset..."
    fi
    adb shell am start -n "$PACKAGE/android.app.NativeActivity"

    tries=0
    until adb shell pidof "$PACKAGE" >/dev/null 2>&1; do
        tries=$((tries + 1))
        if [ "$tries" -ge 20 ]; then
            warn "could not confirm $PACKAGE process started — check logcat."
            break
        fi
        sleep 1
    done
    ok "quest_app process running."

    step "All set."
    if $WANT_DASHBOARD; then
        if $WANT_EDITOR; then
            detail "Put on your headset and move around — debug_viewer shows the live scene + player."
        fi
        detail "Dashboard: tmux attach -t $DASHBOARD_SESSION"

        read -r -p "Press Enter when you're done to stop the dashboard and free up GPU/CPU... "
    fi
fi
