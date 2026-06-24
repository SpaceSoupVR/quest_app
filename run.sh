#!/usr/bin/env bash
# launch.sh — clean, build, deploy to Quest, push game folder, and open space_soup_editor window
# with readiness checks at every step instead of blind sleeps.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XR_DIR="$(dirname "$SCRIPT_DIR")"
QUEST_APP_DIR="$SCRIPT_DIR"
DEBUG_VIEWER_DIR="$XR_DIR/space_soup_editor"
SPACE_SOUP_DIR="$XR_DIR/space_soup"
GAME_DIR="$XR_DIR/game"
PACKAGE="com.example.questapp"
REMOTE_GAME_DIR="/sdcard/Android/data/$PACKAGE/files/game"
HOST_TARGET=$(rustc -vV | awk '/host:/ {print $2}')

# ── Helpers ────────────────────────────────────────────────────────────────────

wait_for_remote_path() {
    # $1 = remote path to wait for
    local path="$1"
    local tries=0
    until adb shell "[ -e '$path' ] && echo exists" 2>/dev/null | grep -q exists; do
        tries=$((tries + 1))
        if [ "$tries" -ge 20 ]; then
            echo "ERROR: timed out waiting for $path to exist on device."
            return 1
        fi
        sleep 1
    done
    return 0
}

wait_for_app_data_dir() {
    local data_dir="/sdcard/Android/data/$PACKAGE"
    echo "==> Waiting for app data directory ($data_dir)..."

    if wait_for_remote_path "$data_dir" 2>/dev/null; then
        echo "    Data directory exists."
        return 0
    fi

    echo "    Data directory missing — launching app once to create it..."
    adb shell am start -n "$PACKAGE/android.app.NativeActivity" >/dev/null 2>&1 || true
    sleep 2
    adb shell am force-stop "$PACKAGE" >/dev/null 2>&1 || true

    if ! wait_for_remote_path "$data_dir"; then
        echo "ERROR: app data directory never appeared. Is the package name correct?"
        exit 1
    fi
    echo "    Data directory created."
}

verify_remote_file() {
    # $1 = remote path
    local path="$1"
    local size
    size=$(adb shell "stat -c %s '$path' 2>/dev/null || echo 0" | tr -d '\r')
    if [ "$size" -lt 1 ] 2>/dev/null; then
        echo "ERROR: $path missing or empty after push."
        exit 1
    fi
    echo "    OK: $path ($size bytes)"
}

wait_for_tcp_listener() {
    # $1 = port
    local port="$1"
    local tries=0
    echo "==> Waiting for listener on :$port..."
    until nc -z 127.0.0.1 "$port" 2>/dev/null; do
        tries=$((tries + 1))
        if [ "$tries" -ge 30 ]; then
            echo "ERROR: nothing started listening on :$port after 30s."
            exit 1
        fi
        sleep 1
    done
    echo "    Listener on :$port is ready."
}

# ── 1. Clean all projects ─────────────────────────────────────────────────────
echo "==> Cleaning space_soup..."
cd "$SPACE_SOUP_DIR"
cargo clean

echo "==> Cleaning debug_viewer..."
cd "$DEBUG_VIEWER_DIR"
cargo clean

echo "==> Cleaning quest_app..."
cd "$QUEST_APP_DIR"
cargo clean
cd android && ./gradlew clean && cd ..

# ── 2. Pre-build debug_viewer so it starts instantly later ───────────────────
echo "==> Pre-building debug_viewer..."
cd "$DEBUG_VIEWER_DIR"
cargo build --target "$HOST_TARGET"

# ── 3. Build and deploy to Quest ──────────────────────────────────────────────
echo "==> Building quest_app for Android..."
cd "$QUEST_APP_DIR"
cargo build --target aarch64-linux-android --release
mkdir -p android/jniLibs/arm64-v8a
cp target/aarch64-linux-android/release/libquest_app.so android/jniLibs/arm64-v8a/
cd android
./gradlew assembleDebug

echo "==> Installing APK..."
adb install -r app/build/outputs/apk/debug/app-debug.apk

# Confirm install actually registered
tries=0
until adb shell pm list packages | grep -q "$PACKAGE"; do
    tries=$((tries + 1))
    if [ "$tries" -ge 15 ]; then
        echo "ERROR: $PACKAGE not found in package list after install."
        exit 1
    fi
    sleep 1
done
echo "    Package installed and registered."

cd "$QUEST_APP_DIR"

# ── 4. Set up ADB reverse for debug_viewer port ───────────────────────────────
echo "==> Reversing TCP debug_viewer port 7778..."
adb reverse tcp:7778 tcp:7778

# ── 5. Ensure app data dir exists, then push game folder ─────────────────────
if [ ! -f "$GAME_DIR/manifest.json" ]; then
    echo "ERROR: $GAME_DIR/manifest.json not found. Aborting."
    exit 1
fi

wait_for_app_data_dir

echo "==> Pushing game folder to Quest..."
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

echo "==> Game folder verified on device."

# ── 6. Open Terminal windows ───────────────────────────────────────────────────

# Terminal 1 — logcat
osascript <<EOF
tell application "Terminal"
    do script "export DISABLE_AUTO_UPDATE=true; cd '$QUEST_APP_DIR' && adb logcat -s quest_app"
    set bounds of front window to {0, 0, 900, 500}
end tell
EOF

# Terminal 2 — ADB reverse keepalive
osascript <<EOF
tell application "Terminal"
    do script "export DISABLE_AUTO_UPDATE=true; while true; do adb reverse tcp:7778 tcp:7778 2>/dev/null; sleep 5; done"
    set bounds of front window to {0, 520, 900, 800}
end tell
EOF

# Terminal 3 — Debug viewer window (already built, starts instantly)
osascript <<EOF
tell application "Terminal"
    do script "export DISABLE_AUTO_UPDATE=true; cd '$DEBUG_VIEWER_DIR' && cargo run --target $HOST_TARGET"
    set bounds of front window to {920, 0, 1800, 900}
end tell
EOF

# ── 7. Wait for the listener to actually be ready ─────────────────────────────
wait_for_tcp_listener 7778

# ── 8. Launch Quest app ────────────────────────────────────────────────────────
echo "==> Launching quest_app on headset..."
adb shell am start -n "$PACKAGE/android.app.NativeActivity"

# Confirm the process actually started
tries=0
until adb shell pidof "$PACKAGE" >/dev/null 2>&1; do
    tries=$((tries + 1))
    if [ "$tries" -ge 20 ]; then
        echo "WARNING: could not confirm $PACKAGE process started — check logcat."
        break
    fi
    sleep 1
done
echo "    quest_app process running."

echo "==> All terminals launched."
echo "    Put on your headset and move around — debug_viewer shows the live scene + player."