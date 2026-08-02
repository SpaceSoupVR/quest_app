<#
    Windows/PowerShell equivalent of run.sh -- build, deploy and run quest_app on
    a Quest, with a dev dashboard.

    We have devs on different systems, so this is kept feature-equivalent to
    run.sh rather than being a cut-down port. Differences are platform ones only:

      * NDK/SDK locations are detected per-platform (run.sh hardcodes the macOS
        darwin-x86_64 toolchain path, which is wrong everywhere else).
      * The NDK C/C++ toolchain is exported explicitly. physx-sys shells out to
        aarch64-linux-android-clang++ and fails with a bare "program not found"
        if it is not on PATH -- worth doing for everyone, not just Windows.
      * tmux has no Windows equivalent, so the dashboard opens separate
        PowerShell windows (logcat / editor / adb-reverse keepalive).
      * gradlew.bat instead of ./gradlew.

    Usage:
        .\run_quest.ps1                  # interactive, same prompts as run.sh
        .\run_quest.ps1 -Clean -Deploy   # skip prompts
        .\run_quest.ps1 -NoDeploy        # build only
#>
[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$Deploy,
    [switch]$NoDeploy,
    [switch]$NoEditor
)

$ErrorActionPreference = 'Stop'

function Step   ($m) { Write-Host ""; Write-Host "==> $m" -ForegroundColor Cyan }
function Ok     ($m) { Write-Host "    $m" -ForegroundColor Green }
function Warn   ($m) { Write-Host "    WARNING: $m" -ForegroundColor Yellow }
function Fail   ($m) { Write-Host "    ERROR: $m" -ForegroundColor Red }
function Detail ($m) { Write-Host "    $m" -ForegroundColor DarkGray }

function Add-PathFront([string]$dir) {
    if (-not $dir -or -not (Test-Path $dir)) { return }
    $parts = @($env:PATH -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    $exists = $parts | Where-Object { $_.TrimEnd('\\').ToLowerInvariant() -eq $dir.TrimEnd('\\').ToLowerInvariant() }
    if (-not $exists) { $env:PATH = "$dir;$env:PATH" }
}

# Make rustup/cargo visible even when the shell profile did not run.
$DefaultCargoBin = Join-Path $HOME '.cargo\bin'
Add-PathFront $DefaultCargoBin

$QuestAppDir = $PSScriptRoot
$XrDir       = Split-Path $QuestAppDir -Parent
$SpaceSoupDir = Join-Path $XrDir 'space_soup'
$GameDir      = Join-Path $XrDir 'game'
# frank_branch replaced the old egui space_soup_editor with the web editor.
$EditorDir    = Join-Path $XrDir 'scene_editor_web'

$Package            = 'com.example.questapp'
$RemoteGameDir      = "/sdcard/Android/data/$Package/files/game"
$RemoteServerUrlFile = "/sdcard/Android/data/$Package/files/server_url.txt"

$ServerUrl     = if ($env:QUEST_SERVER_URL) { $env:QUEST_SERVER_URL } else { 'ws://137.184.21.78:9001' }
$DropletSsh    = if ($env:QUEST_SERVER_SSH_HOST) { $env:QUEST_SERVER_SSH_HOST } else { 'root@137.184.21.78' }
$DropletSshKey = if ($env:QUEST_SERVER_SSH_KEY) { $env:QUEST_SERVER_SSH_KEY } else { Join-Path $HOME 'vr_digitalocean' }
$DropletService = 'space-soup-server.service'

$HostTarget = (& rustc -vV | Select-String '^host:').ToString().Split(' ')[1]

# ---------------------------------------------------------------- toolchain --
function Resolve-Sdk {
    if ($env:ANDROID_SDK_ROOT -and (Test-Path $env:ANDROID_SDK_ROOT)) { return $env:ANDROID_SDK_ROOT }
    if ($env:ANDROID_SDK_HOME -and (Test-Path $env:ANDROID_SDK_HOME)) { return $env:ANDROID_SDK_HOME }
    if ($env:ANDROID_HOME -and (Test-Path $env:ANDROID_HOME)) { return $env:ANDROID_HOME }

    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Android\Sdk'),
        (Join-Path $env:USERPROFILE 'AppData\Local\Android\Sdk'),
        'C:\Android\Sdk',
        'C:\Android\sdk',
        'C:\android-sdk'
    )

    foreach ($drive in @('D','E','F','G','H')) {
        $candidates += @(
            "$drive`:\Android\Sdk",
            "$drive`:\Android\sdk",
            "$drive`:\android-sdk",
            "$drive`:\Sdk\Android",
            "$drive`:\AndroidStudio\Sdk",
            "$drive`:\Program Files\Android\Sdk"
        )
    }

    foreach ($candidate in $candidates | Select-Object -Unique) {
        if (Test-Path $candidate) { return $candidate }
    }

    return $null
}

function Resolve-Ndk {
    if ($env:ANDROID_NDK_HOME -and (Test-Path $env:ANDROID_NDK_HOME)) { return $env:ANDROID_NDK_HOME }
    if ($env:ANDROID_NDK_ROOT -and (Test-Path $env:ANDROID_NDK_ROOT)) { return $env:ANDROID_NDK_ROOT }
    $sdk = Resolve-Sdk
    if ($sdk) {
        $ndkRoot = Join-Path $sdk 'ndk'
        if (Test-Path $ndkRoot) {
            $newest = Get-ChildItem $ndkRoot -Directory -ErrorAction SilentlyContinue |
                      Sort-Object Name | Select-Object -Last 1
            if ($newest) { return $newest.FullName }
        }
        $bundle = Join-Path $sdk 'ndk-bundle'
        if (Test-Path $bundle) { return $bundle }
    }
    return $null
}

function Resolve-JavaHome {
    if ($env:JAVA_HOME -and (Test-Path (Join-Path $env:JAVA_HOME 'bin\java.exe'))) { return $env:JAVA_HOME }

    $candidates = @(
        (Join-Path $env:ProgramFiles 'Android\Android Studio\jbr'),
        (Join-Path $env:ProgramFiles 'Android\Android Studio\jre'),
        (Join-Path ${env:ProgramFiles(x86)} 'Android\Android Studio\jre')
    )

    foreach ($drive in @('D','E','F','G','H')) {
        $candidates += @(
            "$drive`:\Program Files\Android\Android Studio\jbr",
            "$drive`:\Program Files\Android\Android Studio\jre",
            "$drive`:\Android\Android Studio\jbr",
            "$drive`:\Android\Android Studio\jre",
            "$drive`:\AndroidStudio\jbr"
        )
    }

    foreach ($candidate in $candidates | Select-Object -Unique) {
        if (Test-Path (Join-Path $candidate 'bin\java.exe')) { return $candidate }
    }

    return $null
}

$SdkHome = Resolve-Sdk
if (-not $SdkHome) {
    Fail 'Android SDK not found. Set ANDROID_SDK_ROOT/ANDROID_HOME or install Android SDK.'
    exit 1
}
$env:ANDROID_SDK_ROOT = $SdkHome
$env:ANDROID_HOME = $SdkHome

Add-PathFront (Join-Path $SdkHome 'platform-tools')
Add-PathFront (Join-Path $SdkHome 'cmdline-tools\latest\bin')
Add-PathFront (Join-Path $SdkHome 'tools\bin')

$JavaHome = Resolve-JavaHome
if ($JavaHome) {
    $env:JAVA_HOME = $JavaHome
    Add-PathFront (Join-Path $JavaHome 'bin')
    Detail "JAVA_HOME: $JavaHome"
} else {
    Warn 'JAVA_HOME not auto-detected. If Gradle fails, set JAVA_HOME to JDK or Android Studio jbr.'
}

$NdkHome = Resolve-Ndk
if (-not $NdkHome) {
    Fail 'Android NDK not found. Set ANDROID_NDK_HOME (or install it under the SDK).'
    exit 1
}
# Host tag differs per platform -- this is the line run.sh gets wrong off macOS.
$HostTag = if ($IsMacOS) { 'darwin-x86_64' } elseif ($IsLinux) { 'linux-x86_64' } else { 'windows-x86_64' }
$NdkBin  = Join-Path $NdkHome "toolchains\llvm\prebuilt\$HostTag\bin"
if (-not (Test-Path $NdkBin)) {
    Fail "NDK toolchain not found at $NdkBin"
    exit 1
}

# physx-sys invokes the C++ compiler directly; without these it dies with a bare
# "failed to find tool aarch64-linux-android-clang++".
#
# The API level is READ from the gradle minSdk rather than guessed. The NDK ships
# one clang per API level, and anything below 24 has no libvulkan -- so guessing
# too low fails at LINK time with "unable to find library -lvulkan", minutes into
# a release build that looked like it was working.
$ApiLevel = 29
$gradleApp = Join-Path $QuestAppDir 'android/app/build.gradle'
if (Test-Path $gradleApp) {
    $m = Select-String -Path $gradleApp -Pattern 'minSdk\s+(\d+)' | Select-Object -First 1
    if ($m) { $ApiLevel = [int]$m.Matches[0].Groups[1].Value }
}
if ($ApiLevel -lt 24) {
    Warn "minSdk $ApiLevel predates libvulkan; using API 24 for the native toolchain."
    $ApiLevel = 24
}
Detail "Android API level: $ApiLevel"
$clangCc  = Join-Path $NdkBin "aarch64-linux-android$ApiLevel-clang.cmd"
$clangCxx = Join-Path $NdkBin "aarch64-linux-android$ApiLevel-clang++.cmd"
if (-not (Test-Path $clangCc)) {
    $clangCc  = Join-Path $NdkBin "aarch64-linux-android$ApiLevel-clang"
    $clangCxx = Join-Path $NdkBin "aarch64-linux-android$ApiLevel-clang++"
}
Add-PathFront $NdkBin
$env:CC_aarch64_linux_android  = $clangCc
$env:CXX_aarch64_linux_android = $clangCxx
$env:AR_aarch64_linux_android  = (Join-Path $NdkBin 'llvm-ar.exe')
$env:CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER = $clangCc
Detail "SDK: $SdkHome"
Detail "NDK: $NdkHome"

foreach ($tool in @('adb', 'cargo', 'rustc')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Fail "$tool not found on PATH."
        exit 1
    }
}

# ------------------------------------------------------------------ prompts --
function Ask($question, $defaultYes) {
    $suffix = if ($defaultYes) { '[Y/n]' } else { '[y/N]' }
    $reply = Read-Host "$question $suffix"
    if ([string]::IsNullOrWhiteSpace($reply)) { return $defaultYes }
    return $reply -match '^[yY]'
}

$WantClean = $Clean.IsPresent
if (-not $Clean -and -not $Deploy -and -not $NoDeploy) {
    $WantClean = Ask 'Clean all builds first?' $false
}

$WantDeploy = $true
if ($NoDeploy) { $WantDeploy = $false }
elseif (-not $Deploy -and -not $Clean) { $WantDeploy = Ask 'Upload to headset and run?' $true }

$WantEditor = $false
if ($WantDeploy -and -not $NoEditor) { $WantEditor = Ask 'Do you want the scene editor?' $true }

# ------------------------------------------------------------------ helpers --
function Wait-RemotePath($path, $tries = 30) {
    for ($i = 0; $i -lt $tries; $i++) {
        $out = & adb shell "ls $path" 2>&1
        if ($LASTEXITCODE -eq 0 -and $out -notmatch 'No such file') { return $true }
        Start-Sleep -Seconds 1
    }
    return $false
}

function Wait-AppDataDir {
    Step "Waiting for app data dir on device..."
    # Launching once makes Android create the app's files/ directory.
    & adb shell am start -n "$Package/android.app.NativeActivity" | Out-Null
    if (Wait-RemotePath "/sdcard/Android/data/$Package/files" 30) {
        Ok 'App data dir present.'
    } else {
        Warn 'App data dir did not appear; pushing anyway (mkdir -p should create it).'
    }
}

function Verify-RemoteFile($remote) {
    $out = & adb shell "ls -l $remote" 2>&1
    if ($LASTEXITCODE -ne 0 -or $out -match 'No such file') {
        Fail "verify failed: $remote missing on device"
        exit 1
    }
}

function Wait-TcpListener($port, $tries = 30) {
    for ($i = 0; $i -lt $tries; $i++) {
        $c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        if ($c) { Ok "Listener on :$port is ready."; return $true }
        Start-Sleep -Seconds 1
    }
    Warn "No listener on :$port after $tries s."
    return $false
}

function Push-Tree($localRoot, $remoteRoot) {
    if (-not (Test-Path $localRoot)) { return }
    & adb shell "mkdir -p '$remoteRoot'" | Out-Null
    Get-ChildItem $localRoot -Recurse -Directory | ForEach-Object {
        $rel = $_.FullName.Substring($localRoot.Length).TrimStart('\','/') -replace '\\','/'
        & adb shell "mkdir -p '$remoteRoot/$rel'" | Out-Null
    }
    Get-ChildItem $localRoot -Recurse -File | ForEach-Object {
        $rel = $_.FullName.Substring($localRoot.Length).TrimStart('\','/') -replace '\\','/'
        & adb push $_.FullName "$remoteRoot/$rel" | Out-Null
        Verify-RemoteFile "$remoteRoot/$rel"
    }
}

$DashboardWindows = @()
function Start-DashboardWindow($title, $workDir, $command) {
    $script = "`$Host.UI.RawUI.WindowTitle = '$title'; Set-Location '$workDir'; $command"
    $p = Start-Process powershell -PassThru -ArgumentList @(
        '-NoExit', '-NoProfile', '-Command', $script
    )
    $script:DashboardWindows += $p
    return $p
}

function Stop-Dashboard {
    foreach ($p in $script:DashboardWindows) {
        if ($p -and -not $p.HasExited) {
            try { Stop-Process -Id $p.Id -Force -ErrorAction Stop } catch { }
        }
    }
    $gradlew = Join-Path $QuestAppDir 'android\gradlew.bat'
    if (Test-Path $gradlew) {
        Push-Location (Join-Path $QuestAppDir 'android')
        try { & $gradlew --stop 2>&1 | Out-Null } catch { }
        Pop-Location
    }
}

# -------------------------------------------------------------------- build --
try {
    if ($WantClean) {
        Step 'Cleaning space_soup...'
        Push-Location $SpaceSoupDir; & cargo clean; Pop-Location

        Step 'Cleaning quest_app...'
        Push-Location $QuestAppDir; & cargo clean; Pop-Location

        $gradlew = Join-Path $QuestAppDir 'android\gradlew.bat'
        if (Test-Path $gradlew) {
            Push-Location (Join-Path $QuestAppDir 'android')
            & $gradlew clean
            Pop-Location
        }
    }

    Step 'Building quest_app for Android...'
    Push-Location $QuestAppDir
    & cargo build --target aarch64-linux-android --release
    if ($LASTEXITCODE -ne 0) { Fail 'cargo build failed'; exit 1 }

    $jni = Join-Path $QuestAppDir 'android\jniLibs\arm64-v8a'
    New-Item -ItemType Directory -Force -Path $jni | Out-Null
    Copy-Item 'target\aarch64-linux-android\release\libquest_app.so' (Join-Path $jni 'libquest_app.so') -Force

    $cxxShared = Join-Path $NdkBin '..\sysroot\usr\lib\aarch64-linux-android\libc++_shared.so'
    if (Test-Path $cxxShared) {
        Step 'Copying libc++_shared.so into jniLibs ...'
        Copy-Item $cxxShared (Join-Path $jni 'libc++_shared.so') -Force
    } else {
        Warn "libc++_shared.so not found at $cxxShared - app will fail to load on device"
    }
    Pop-Location

    Step 'Building APK...'
    Push-Location (Join-Path $QuestAppDir 'android')
    & .\gradlew.bat assembleDebug
    if ($LASTEXITCODE -ne 0) { Fail 'gradle assembleDebug failed'; exit 1 }
    Pop-Location

    if (-not $WantDeploy) {
        Ok 'Build complete (deploy skipped).'
        exit 0
    }

    # ----------------------------------------------------------- deploy --
    Step 'Installing APK...'
    & adb install -r (Join-Path $QuestAppDir 'android\app\build\outputs\apk\debug\app-debug.apk')

    $installed = $false
    for ($i = 0; $i -lt 15; $i++) {
        if ((& adb shell pm list packages) -match [regex]::Escape($Package)) { $installed = $true; break }
        Start-Sleep -Seconds 1
    }
    if (-not $installed) { Fail "$Package not found in package list after install."; exit 1 }
    Ok 'Package installed and registered.'

    if ($WantEditor) {
        Step 'Reversing TCP editor port 7778...'
        & adb reverse tcp:7778 tcp:7778 | Out-Null
    }

    if (-not (Test-Path (Join-Path $GameDir 'manifest.json'))) {
        Fail "$GameDir\manifest.json not found. Aborting."
        exit 1
    }

    Wait-AppDataDir

    Step 'Pushing game folder to Quest...'
    & adb shell "mkdir -p '$RemoteGameDir'" | Out-Null
    & adb push (Join-Path $GameDir 'manifest.json') "$RemoteGameDir/manifest.json" | Out-Null
    Verify-RemoteFile "$RemoteGameDir/manifest.json"

    foreach ($cfg in @('avatar_rig.json', 'synthetic_hand.json')) {
        $local = Join-Path $GameDir $cfg
        if (Test-Path $local) {
            & adb push $local "$RemoteGameDir/$cfg" | Out-Null
            Verify-RemoteFile "$RemoteGameDir/$cfg"
        }
    }

    Push-Tree (Join-Path $GameDir 'scenes') "$RemoteGameDir/scenes"
    Push-Tree (Join-Path $GameDir 'models') "$RemoteGameDir/models"
    Push-Tree (Join-Path $GameDir 'sound')  "$RemoteGameDir/sound"
    Ok 'Game folder verified on device.'

    Step "Verifying multiplayer server is running on $DropletSsh..."
    if (Test-Path $DropletSshKey) {
        & ssh -i $DropletSshKey -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new `
            $DropletSsh "systemctl is-active --quiet $DropletService || systemctl restart $DropletService"
        if ($LASTEXITCODE -eq 0) {
            Ok 'space-soup-server is active.'
        } else {
            Warn "Could not confirm space-soup-server on $DropletSsh - multiplayer may not work."
        }
    } else {
        Warn "SSH key not found at $DropletSshKey - skipping server check (set QUEST_SERVER_SSH_KEY to override)."
    }

    Step "Pushing multiplayer server URL ($ServerUrl)..."
    $tmp = New-TemporaryFile
    [IO.File]::WriteAllText($tmp.FullName, $ServerUrl)
    & adb push $tmp.FullName $RemoteServerUrlFile | Out-Null
    Remove-Item $tmp.FullName -Force
    Verify-RemoteFile $RemoteServerUrlFile

    # -------------------------------------------------------- dashboard --
    Step 'Starting dev dashboard (separate windows)...'
    Start-DashboardWindow 'quest_app logcat' $QuestAppDir 'adb logcat -s quest_app' | Out-Null
    Ok 'logcat window opened.'

    if ($WantEditor) {
        if (Test-Path (Join-Path $EditorDir 'run_editor.ps1')) {
            Start-DashboardWindow 'scene editor' $EditorDir '.\run_editor.ps1 -NoBrowser' | Out-Null
            Ok 'scene editor starting (http://localhost:5173).'
        } else {
            Warn "scene editor not found at $EditorDir - skipping."
        }
        Start-DashboardWindow 'adb reverse keepalive' $QuestAppDir `
            'while ($true) { adb reverse tcp:7778 tcp:7778 2>$null | Out-Null; Start-Sleep -Seconds 5 }' | Out-Null
        Wait-TcpListener 5173 | Out-Null
    }

    Step 'Launching quest_app on headset...'
    & adb shell am start -n "$Package/android.app.NativeActivity" | Out-Null

    $running = $false
    for ($i = 0; $i -lt 20; $i++) {
        & adb shell pidof $Package 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) { $running = $true; break }
        Start-Sleep -Seconds 1
    }
    if ($running) { Ok 'quest_app process running.' }
    else { Warn 'could not confirm quest_app started - check the logcat window.' }

    Step 'All set.'
    if ($WantEditor) { Detail 'Editor: http://localhost:5173' }
    Read-Host 'Press Enter when you are done to stop the dashboard and free up GPU/CPU'
}
finally {
    Stop-Dashboard
}
