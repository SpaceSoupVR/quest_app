#!/usr/bin/env bash

set -e

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
HOST_TARGET=$(rustc -vV | awk '/host:/ {print $2}')
SDK_HOME="${ANDROID_SDK_HOME:-$HOME/Library/Android/sdk}"
NDK_HOME="${ANDROID_NDK_HOME:-$(ls -d "$SDK_HOME/ndk/"* 2>/dev/null | sort -V | tail -1)}"
DASHBOARD_SESSION="quest_app"


if ! command -v tmux >/dev/null 2>&1; then
    fail "tmux not found — install it with: brew install tmux"
    exit 1
fi
if ! command -v xterm >/dev/null 2>&1; then
    warn "xterm not found (install XQuartz for it: brew install --cask xquartz)."
    detail "Continuing — the dashboard will open in Terminal.app instead."
fi

resolve_display() {
    if ! command -v xterm >/dev/null 2>&1; then
        return 1
    fi

    local n="${DISPLAY#:}"
    if [ -n "$DISPLAY" ] && [ -S "/tmp/.X11-unix/X${n:-0}" ]; then
        printf '%s' "$DISPLAY"
        return 0
    fi
    if [ -S "/tmp/.X11-unix/X0" ]; then
        printf ':0'
        return 0
    fi

    if [ -d "/Applications/Utilities/XQuartz.app" ] || [ -d "/Applications/XQuartz.app" ]; then
        detail "Starting XQuartz (not running yet — first run after install needs this)..." >&2
        open -a XQuartz 2>/dev/null || true
        local tries=0
        while [ "$tries" -lt 20 ]; do
            if [ -S "/tmp/.X11-unix/X0" ]; then
                printf ':0'
                return 0
            fi
            tries=$((tries + 1))
            sleep 1
        done
    fi
    return 1
}

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

verify_remote_file() {
    # $1 = remote path
    local path="$1"
    local size
    size=$(adb shell "stat -c %s '$path' 2>/dev/null || echo 0" | tr -d '\r')
    if [ "$size" -lt 1 ] 2>/dev/null; then
        fail "$path missing or empty after push."
        exit 1
    fi
    ok "OK: $path ($size bytes)"
}

wait_for_tcp_listener() {
    # $1 = port
    local port="$1"
    local tries=0
    step "Waiting for listener on :$port..."
    until nc -z 127.0.0.1 "$port" 2>/dev/null; do
        tries=$((tries + 1))
        if [ "$tries" -ge 30 ]; then
            fail "nothing started listening on :$port after 30s."
            exit 1
        fi
        sleep 1
    done
    ok "Listener on :$port is ready."
}

step "Cleaning space_soup..."
cd "$SPACE_SOUP_DIR"
cargo clean

step "Cleaning debug_viewer..."
cd "$DEBUG_VIEWER_DIR"
cargo clean

step "Cleaning quest_app..."
cd "$QUEST_APP_DIR"
cargo clean
cd android && ./gradlew clean && cd ..

step "Pre-building debug_viewer..."
cd "$DEBUG_VIEWER_DIR"
cargo build --target "$HOST_TARGET"

step "Building quest_app for Android..."
cd "$QUEST_APP_DIR"
cargo build --target aarch64-linux-android --release
mkdir -p android/jniLibs/arm64-v8a
cp target/aarch64-linux-android/release/libquest_app.so android/jniLibs/arm64-v8a/

CXX_SHARED="$NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
if [ -f "$CXX_SHARED" ]; then
    step "Copying libc++_shared.so into jniLibs ..."
    cp "$CXX_SHARED" android/jniLibs/arm64-v8a/libc++_shared.so
else
    warn "libc++_shared.so not found at $CXX_SHARED — app will fail to load on device"
fi

cd android
./gradlew assembleDebug

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

cd "$QUEST_APP_DIR"

step "Reversing TCP debug_viewer port 7778..."
adb reverse tcp:7778 tcp:7778

if [ ! -f "$GAME_DIR/manifest.json" ]; then
    fail "$GAME_DIR/manifest.json not found. Aborting."
    exit 1
fi

wait_for_app_data_dir

step "Pushing game folder to Quest..."
adb shell mkdir -p "$REMOTE_GAME_DIR"
adb push "$GAME_DIR/manifest.json" "$REMOTE_GAME_DIR/manifest.json"
verify_remote_file "$REMOTE_GAME_DIR/manifest.json"

if [ -d "$GAME_DIR/scenes" ]; then
    adb shell mkdir -p "$REMOTE_GAME_DIR/scenes"
    shopt -s nullglob
    for f in "$GAME_DIR"/scenes/*.json; do
        fname=$(basename "$f")
        adb push "$f" "$REMOTE_GAME_DIR/scenes/$fname"
        verify_remote_file "$REMOTE_GAME_DIR/scenes/$fname"
    done
    shopt -u nullglob
fi

if [ -d "$GAME_DIR/models" ]; then
    adb shell mkdir -p "$REMOTE_GAME_DIR/models"
    shopt -s nullglob
    for f in "$GAME_DIR"/models/*; do
        fname=$(basename "$f")
        adb push "$f" "$REMOTE_GAME_DIR/models/$fname"
        verify_remote_file "$REMOTE_GAME_DIR/models/$fname"
    done
    shopt -u nullglob
fi

ok "Game folder verified on device."

step "Opening dev dashboard (tmux in xterm)..."

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

PANE_VIEWER=$(tmux split-window -h -t "$PANE_LOGCAT" -l "60%" -P -F '#{pane_id}')
tmux send-keys -t "$PANE_VIEWER" \
    "export DISABLE_AUTO_UPDATE=true; cd '$DEBUG_VIEWER_DIR' && cargo run --target $HOST_TARGET" C-m
tmux select-pane -t "$PANE_VIEWER" -T "debug_viewer"

PANE_KEEPALIVE=$(tmux split-window -v -t "$PANE_LOGCAT" -l "40%" -P -F '#{pane_id}')
tmux send-keys -t "$PANE_KEEPALIVE" \
    "export DISABLE_AUTO_UPDATE=true; while true; do adb reverse tcp:7778 tcp:7778 2>/dev/null; sleep 5; done" C-m
tmux select-pane -t "$PANE_KEEPALIVE" -T "adb reverse keepalive"

tmux select-pane -t "$PANE_VIEWER"

if XTERM_DISPLAY=$(resolve_display); then
    DISPLAY="$XTERM_DISPLAY" xterm -fa Monaco -fs 13 -bg '#101418' -fg '#d8f0ff' -geometry 220x52+40+40 \
        -T "quest_app dev" -e "tmux attach -t $DASHBOARD_SESSION" &
    ok "Dashboard opened in xterm."
else
    warn "No usable X11 display for xterm — opening the same dashboard in Terminal.app instead."
    osascript <<EOF
tell application "Terminal"
    do script "tmux attach -t $DASHBOARD_SESSION"
end tell
EOF
    ok "Dashboard opened in Terminal.app."
fi
detail "Reattach any time with: tmux attach -t $DASHBOARD_SESSION"

# ── 7. Wait for the listener to actually be ready ─────────────────────────────
wait_for_tcp_listener 7778

# ── 8. Launch Quest app ────────────────────────────────────────────────────────
step "Launching quest_app on headset..."
adb shell am start -n "$PACKAGE/android.app.NativeActivity"

# Confirm the process actually started
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
detail "Put on your headset and move around — debug_viewer shows the live scene + player."
detail "Dashboard: tmux attach -t $DASHBOARD_SESSION"