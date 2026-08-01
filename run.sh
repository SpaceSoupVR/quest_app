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
REMOTE_SERVER_URL_FILE="/sdcard/Android/data/$PACKAGE/files/server_url.txt"
SERVER_URL="${QUEST_SERVER_URL:-ws://137.184.21.78:9001}"
DROPLET_SSH_HOST="${QUEST_SERVER_SSH_HOST:-root@137.184.21.78}"
DROPLET_SSH_KEY="${QUEST_SERVER_SSH_KEY:-$HOME/vr_digitalocean}"
DROPLET_SERVICE="space-soup-server.service"
HOST_TARGET=$(rustc -vV | awk '/host:/ {print $2}')
SDK_HOME="${ANDROID_SDK_HOME:-$HOME/Library/Android/sdk}"
NDK_HOME="${ANDROID_NDK_HOME:-$(ls -d "$SDK_HOME/ndk/"* 2>/dev/null | sort -V | tail -1)}"
DASHBOARD_SESSION="quest_app"

cleanup() {
    tmux kill-session -t "$DASHBOARD_SESSION" 2>/dev/null || true
    (cd "$QUEST_APP_DIR/android" && ./gradlew --stop >/dev/null 2>&1) || true
}
trap cleanup EXIT

WANT_DASHBOARD="${WANT_DASHBOARD:-true}"

if $WANT_DASHBOARD && ! command -v tmux >/dev/null 2>&1; then
    fail "tmux not found — install it with: brew install tmux"
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
    if $WANT_DEPLOY; then
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

if $WANT_DEPLOY; then
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

    step "Launching quest_app on headset..."
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
