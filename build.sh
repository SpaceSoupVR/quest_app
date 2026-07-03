#NO NEED TO RUN THIS SCRIPT
set -e

PACKAGE="com.example.questapp"
TARGET="aarch64-linux-android"
ACTIVITY="android.app.NativeActivity"

SDK_HOME="${ANDROID_SDK_HOME:-$HOME/Library/Android/sdk}"
NDK_HOME="${ANDROID_NDK_HOME:-$(ls -d "$SDK_HOME/ndk/"* 2>/dev/null | sort -V | tail -1)}"
TOOLCHAIN="$NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin"
BUILD_TOOLS="$SDK_HOME/build-tools/$(ls "$SDK_HOME/build-tools" | sort -V | tail -1)"
PLATFORM_JAR="$SDK_HOME/platforms/android-35/android.jar"

if [ ! -f "$PLATFORM_JAR" ]; then
    PLATFORM_JAR="$SDK_HOME/platforms/android-34/android.jar"
fi
if [ ! -f "$PLATFORM_JAR" ]; then
    PLATFORM_JAR="$SDK_HOME/platforms/android-32/android.jar"
fi

export PATH="$TOOLCHAIN:$PATH"

echo "==> NDK    : $NDK_HOME"
echo "==> Tools  : $BUILD_TOOLS"
echo "==> Platform: $PLATFORM_JAR"
echo ""

echo "==> Building libquest_app.so ..."
cargo build --target "$TARGET" --release --lib

SO="target/$TARGET/release/libquest_app.so"
echo "==> Built: $SO"

echo "==> Copying .so into jniLibs ..."
mkdir -p android/jniLibs/arm64-v8a
cp "$SO" android/jniLibs/arm64-v8a/libquest_app.so

# space_soup_engine's PhysX integration pulls in C++ code linked against
# libc++_shared — without bundling it, the app fails to load at the
# dynamic-linker stage (before any Rust code, including the logger, runs),
# which looks like an instant, silent crash on launch.
CXX_SHARED="$NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
if [ -f "$CXX_SHARED" ]; then
    echo "==> Copying libc++_shared.so into jniLibs ..."
    cp "$CXX_SHARED" android/jniLibs/arm64-v8a/libc++_shared.so
else
    echo "==> WARNING: libc++_shared.so not found at $CXX_SHARED — app will fail to load on device"
fi

echo "==> Building APK with Gradle ..."
cd android
./gradlew assembleDebug --quiet
cd ..

APK="android/app/build/outputs/apk/debug/app-debug.apk"
echo "==> APK: $APK"

if command -v adb &>/dev/null; then
    DEVICES=$(adb devices | grep -v "List of" | grep "device$" | wc -l | tr -d ' ')
    if [ "$DEVICES" -eq 0 ]; then
        echo "==> No ADB devices found — connect your Quest and accept USB debugging"
        exit 1
    fi

    echo "==> Installing on device ..."
    adb install -r "$APK"

    echo "==> Launching $PACKAGE ..."
    adb shell am start -n "$PACKAGE/$ACTIVITY"

    echo ""
    echo "==> Logcat (Ctrl+C to stop):"
    adb logcat -s quest_app
else
    echo "==> adb not found — install Android SDK platform-tools"
    exit 1
fi