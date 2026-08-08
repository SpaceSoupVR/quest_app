<#
    Windows equivalent of run.sh — build, deploy and run quest_app on a Quest.

        .\run_quest.ps1                       # interactive, same prompts as run.sh
        .\run_quest.ps1 -Clean                # clean all builds first
        .\run_quest.ps1 -NoDeploy             # build only, do not touch the headset
        .\run_quest.ps1 -NoDashboard          # skip the logcat/viewer windows

    Full parity with run.sh. Everything that differs does so because Windows
    requires it, and each of those is commented where it happens:

      * tmux does not exist here, so the dashboard is separate PowerShell windows
      * ./gradlew -> .\gradlew.bat
      * the NDK's prebuilt toolchain is windows-x86_64, not darwin-x86_64, and its
        clang wrappers are .cmd files
      * the toolchain comes from here, not .cargo/config.toml, which no longer
        holds absolute paths. It used to pin a macOS NDK, and exporting these was
        believed to override it -- it did not, because cargo's [env] used the
        hyphenated names and these are underscored, which are different variables
      * `nc -z` -> System.Net.Sockets.TcpClient
      * `echo url | adb shell "cat > file"` -> write a temp file and adb push it

    Deliberately NOT run through Git Bash: bash on Windows rewrites leading-slash
    arguments into Windows paths (MSYS path conversion), which mangles adb targets
    like /sdcard/... into C:/Program Files/Git/sdcard/... intermittently. PowerShell
    passes them through untouched.
#>
param(
    [switch]$PrintEnv,
    [switch]$Clean,
    [switch]$NoDeploy,
    [switch]$NoDashboard,
    [switch]$NoEditor
)

# Do NOT use 'Stop' here. Native tools on this path (adb, gradle, cargo, ssh) write
# progress and warnings to stderr, and in Windows PowerShell that becomes a
# terminating NativeCommandError under 'Stop' -- which is what made run_editor.ps1
# tear down its own servers. Failures are checked explicitly via $LASTEXITCODE.
$ErrorActionPreference = 'Continue'

function Step   ($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Ok     ($m) { Write-Host "    $m"   -ForegroundColor Green }
function Warn   ($m) { Write-Host "    WARNING: $m" -ForegroundColor Yellow }
function Fail   ($m) { Write-Host "    ERROR: $m"   -ForegroundColor Red }
function Detail ($m) { Write-Host "    $m"   -ForegroundColor DarkGray }

$ScriptDir     = $PSScriptRoot
$XrDir         = Split-Path $ScriptDir -Parent
$QuestAppDir   = $ScriptDir
$DebugViewerDir= Join-Path $XrDir 'space_soup_editor'
$SpaceSoupDir  = Join-Path $XrDir 'space_soup'
$GameDir       = Join-Path $XrDir 'game'
$Package       = 'com.example.questapp'
$RemoteGameDir = "/sdcard/Android/data/$Package/files/game"
$RemoteUrlFile = "/sdcard/Android/data/$Package/files/server_url.txt"
$ServerUrl     = if ($env:QUEST_SERVER_URL)      { $env:QUEST_SERVER_URL }      else { 'ws://137.184.21.78:9001' }
$DropletHost   = if ($env:QUEST_SERVER_SSH_HOST) { $env:QUEST_SERVER_SSH_HOST } else { 'root@137.184.21.78' }
$DropletKey    = if ($env:QUEST_SERVER_SSH_KEY)  { $env:QUEST_SERVER_SSH_KEY }  else { Join-Path $env:USERPROFILE 'vr_digitalocean' }
$DropletService= 'space-soup-server.service'

# --- PATH repair -------------------------------------------------------------
# A PowerShell launched from an IDE or a scheduled task often has not run the
# user profile, so cargo/rustup are missing from PATH even though they are
# installed. Prepend the standard locations rather than failing with
# "cargo is not recognized".
function Add-PathFront([string]$dir) {
    if (-not $dir -or -not (Test-Path $dir)) { return }
    $parts = @($env:PATH -split ';' | Where-Object { $_ -and $_.Trim() })
    if ($parts -notcontains $dir) { $env:PATH = "$dir;$env:PATH" }
}
Add-PathFront (Join-Path $env:USERPROFILE '.cargo\bin')

# --- Android SDK / NDK discovery ---------------------------------------------
# run.sh defaults to ~/Library/Android/sdk (macOS). Windows installs land
# elsewhere and vary by installer, so probe the usual homes in order.
function Find-AndroidSdk {
    if ($env:ANDROID_SDK_ROOT -and (Test-Path $env:ANDROID_SDK_ROOT)) { return $env:ANDROID_SDK_ROOT }
    if ($env:ANDROID_SDK_HOME -and (Test-Path $env:ANDROID_SDK_HOME)) { return $env:ANDROID_SDK_HOME }
    if ($env:ANDROID_HOME     -and (Test-Path $env:ANDROID_HOME))     { return $env:ANDROID_HOME }
    foreach ($c in @(
        (Join-Path $env:LOCALAPPDATA 'Android\Sdk'),
        (Join-Path $env:USERPROFILE  'AppData\Local\Android\Sdk'),
        'G:\Android\Sdk', 'C:\Android\Sdk'
    )) { if (Test-Path $c) { return $c } }
    return $null
}

function Find-AndroidNdk([string]$sdk) {
    if ($env:ANDROID_NDK_ROOT -and (Test-Path $env:ANDROID_NDK_ROOT)) { return $env:ANDROID_NDK_ROOT }
    if ($env:ANDROID_NDK_HOME -and (Test-Path $env:ANDROID_NDK_HOME)) { return $env:ANDROID_NDK_HOME }
    # Standalone NDK installs (e.g. G:\Android\android-ndk-r26d) as well as the
    # SDK-managed ndk\<version> layout run.sh globs.
    $roots = @()
    if ($sdk) { $roots += (Join-Path $sdk 'ndk') }
    $roots += @('G:\Android', 'C:\Android', (Join-Path $env:LOCALAPPDATA 'Android'))
    foreach ($r in $roots) {
        if (-not (Test-Path $r)) { continue }
        $hit = Get-ChildItem $r -Directory -ErrorAction SilentlyContinue |
               Where-Object { Test-Path (Join-Path $_.FullName 'toolchains\llvm\prebuilt\windows-x86_64\bin') } |
               Sort-Object Name | Select-Object -Last 1
        if ($hit) { return $hit.FullName }
    }
    return $null
}

$SdkHome = Find-AndroidSdk
$NdkHome = Find-AndroidNdk $SdkHome
Add-PathFront (Join-Path $SdkHome 'platform-tools')   # adb

if (-not (Get-Command adb -ErrorAction SilentlyContinue)) {
    Fail "adb not found. Install Android platform-tools, or set ANDROID_SDK_ROOT."
    exit 1
}
if (-not $NdkHome) {
    Fail "Android NDK not found. Set ANDROID_NDK_ROOT to your NDK directory."
    exit 1
}

# --- Cross-compile environment ----------------------------------------------
# The toolchain lives here. .cargo/config.toml deliberately holds no absolute
# paths any more -- it used to pin a macOS NDK whose [env] names were hyphenated
# while these are underscored, so these never actually won and cc-rs resolved a
# darwin path on Windows. Without ANDROID_NDK_ROOT physx-sys fails with
# `environment variable "ANDROID_NDK_ROOT" has not been set`,
# or the link step dies looking for a darwin-x86_64 toolchain that does not exist.
$Prebuilt = Join-Path $NdkHome 'toolchains\llvm\prebuilt\windows-x86_64'
$NdkBin   = Join-Path $Prebuilt 'bin'

# Derive the API level from gradle rather than hardcoding it. Using a level below
# minSdk produces a link failure that reads as a missing library: building against
# API 21 fails with `ld.lld: error: unable to find library -lvulkan`, because
# Vulkan only ships from API 24 upward.
$MinSdk = 29
$gradleFile = Join-Path $QuestAppDir 'android\app\build.gradle'
if (Test-Path $gradleFile) {
    $m = Select-String -Path $gradleFile -Pattern 'minSdk\s+(\d+)' | Select-Object -First 1
    if ($m) { $MinSdk = [int]$m.Matches[0].Groups[1].Value }
}
$ApiLevel = [Math]::Max($MinSdk, 24)
Detail "NDK: $NdkHome (API $ApiLevel, from minSdk $MinSdk)"

# The NDK's clang wrappers are .cmd on Windows; the bare names do not exist.
$Clang   = Join-Path $NdkBin "aarch64-linux-android$ApiLevel-clang.cmd"
$ClangPP = Join-Path $NdkBin "aarch64-linux-android$ApiLevel-clang++.cmd"
$Ar      = Join-Path $NdkBin 'llvm-ar.exe'
foreach ($t in @($Clang, $ClangPP, $Ar)) {
    if (-not (Test-Path $t)) { Fail "NDK tool missing: $t"; exit 1 }
}
$env:ANDROID_NDK_ROOT                             = $NdkHome
$env:ANDROID_NDK_HOME                             = $NdkHome
$env:CC_aarch64_linux_android                     = $Clang
$env:CXX_aarch64_linux_android                    = $ClangPP
$env:AR_aarch64_linux_android                     = $Ar
$env:CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER    = $Clang

# -PrintEnv: emit the detected toolchain and stop, so cargo can be driven
# directly without anyone hand-copying paths that then go stale.
#   . ./run_quest.ps1 -PrintEnv
if ($PrintEnv) {
    Write-Output "`$env:ANDROID_NDK_ROOT = '$NdkHome'"
    Write-Output "`$env:ANDROID_NDK_HOME = '$NdkHome'"
    Write-Output "`$env:CC_aarch64_linux_android = '$Clang'"
    Write-Output "`$env:CXX_aarch64_linux_android = '$ClangPP'"
    Write-Output "`$env:AR_aarch64_linux_android = '$Ar'"
    Write-Output "`$env:CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER = '$Clang'"
    exit 0
}
Add-PathFront $NdkBin

$HostTarget = (& rustc -vV | Select-String '^host:\s*(.+)$').Matches[0].Groups[1].Value.Trim()

# --- Interactive choices (same questions, same defaults, as run.sh) ----------
$WantClean     = $Clean.IsPresent
$WantDeploy    = -not $NoDeploy.IsPresent
$WantDashboard = -not $NoDashboard.IsPresent
$WantEditor    = $false

if (-not $PSBoundParameters.ContainsKey('Clean')) {
    $WantClean = (Read-Host 'Clean all builds first? [y/N]') -match '^[yY]'
}
if (-not $PSBoundParameters.ContainsKey('NoDeploy')) {
    $WantDeploy = -not ((Read-Host 'Upload to headset and run? [Y/n]') -match '^[nN]')
}
if ($WantDeploy -and -not $NoEditor.IsPresent -and (Test-Path $DebugViewerDir)) {
    $WantEditor = -not ((Read-Host 'Do you want the scene editor? [Y/n]') -match '^[nN]')
}

$DashboardWindows = @()
$PushFailures     = @()

function Wait-RemotePath([string]$path, [int]$tries = 20) {
    for ($i = 0; $i -lt $tries; $i++) {
        if ((& adb shell "[ -e '$path' ] && echo exists" 2>$null) -match 'exists') { return $true }
        Start-Sleep -Seconds 1
    }
    return $false
}

function Wait-AppDataDir {
    $dataDir = "/sdcard/Android/data/$Package"
    Step "Waiting for app data directory ($dataDir)..."
    if (Wait-RemotePath $dataDir 5) { Ok 'Data directory exists.'; return $true }

    Detail 'Data directory missing - launching app once to create it...'
    & adb shell am start -n "$Package/android.app.NativeActivity" *> $null
    Start-Sleep -Seconds 2
    & adb shell am force-stop $Package *> $null

    if (-not (Wait-RemotePath $dataDir)) {
        Fail 'app data directory never appeared. Is the package name correct?'
        return $false
    }
    Ok 'Data directory created.'
    return $true
}

# True when the remote file exists and is non-empty. Pure predicate.
function Test-RemoteFile([string]$path) {
    $size = (& adb shell "stat -c %s '$path' 2>/dev/null || echo 0") -replace '\r',''
    return ([int]($size | Select-Object -First 1)) -ge 1
}

# adb push with one retry + verify. Records failures instead of aborting, so one
# bad file cannot silently leave every later asset (other models, the editor's
# sidecar .bin animation buffers) off the headset.
function Push-File([string]$src, [string]$dst) {
    $short = $dst -replace [regex]::Escape("$RemoteGameDir/"), ''
    foreach ($attempt in 1, 2) {
        & adb push $src $dst *> $null
        if ($LASTEXITCODE -eq 0 -and (Test-RemoteFile $dst)) {
            $size = (& adb shell "stat -c %s '$dst' 2>/dev/null || echo 0") -replace '\r',''
            Ok "OK: $short ($size bytes)"
            return
        }
        if ($attempt -eq 1) { Warn "push failed for $short - retrying..." }
    }
    Fail "could not push $short after 2 attempts"
    $script:PushFailures += $short
}

function Wait-TcpListener([int]$port, [int]$tries = 30) {
    Step "Waiting for listener on :$port..."
    for ($i = 0; $i -lt $tries; $i++) {
        try {
            $c = New-Object System.Net.Sockets.TcpClient
            $c.Connect('127.0.0.1', $port); $c.Close()
            Ok "Listener on :$port is ready."; return $true
        } catch { Start-Sleep -Seconds 1 }
    }
    Fail "nothing started listening on :$port after ${tries}s."
    return $false
}

# Opens a titled PowerShell window. tmux has no Windows equivalent, so each
# dashboard pane becomes its own window; they are closed in the finally block.
function Start-DashboardWindow([string]$title, [string]$workDir, [string]$command) {
    $inner = "`$host.UI.RawUI.WindowTitle='$title'; Set-Location '$workDir'; $command"
    $p = Start-Process powershell -PassThru -ArgumentList @(
        '-NoExit','-NoProfile','-ExecutionPolicy','Bypass','-Command', $inner
    )
    $script:DashboardWindows += $p
    return $p
}

try {
    if ($WantClean) {
        Step 'Cleaning space_soup...'
        Push-Location $SpaceSoupDir; & cargo clean; Pop-Location
        if ($WantEditor) {
            Step 'Cleaning debug_viewer...'
            Push-Location $DebugViewerDir; & cargo clean; Pop-Location
        }
        Step 'Cleaning quest_app...'
        Push-Location $QuestAppDir; & cargo clean; Pop-Location
        Push-Location (Join-Path $QuestAppDir 'android'); & .\gradlew.bat clean; Pop-Location
    }

    if ($WantEditor) {
        Step 'Pre-building debug_viewer...'
        Push-Location $DebugViewerDir
        & cargo build --target $HostTarget
        if ($LASTEXITCODE -ne 0) { Fail 'debug_viewer build failed.'; exit 1 }
        Pop-Location
    }

    Step 'Building quest_app for Android...'
    Push-Location $QuestAppDir
    & cargo build --target aarch64-linux-android --release
    if ($LASTEXITCODE -ne 0) { Fail 'cargo build failed.'; exit 1 }

    $jniDir = Join-Path $QuestAppDir 'android\jniLibs\arm64-v8a'
    New-Item -ItemType Directory -Force -Path $jniDir | Out-Null
    $builtSo = Join-Path $QuestAppDir 'target\aarch64-linux-android\release\libquest_app.so'
    if (-not (Test-Path $builtSo)) { Fail "build reported success but $builtSo is missing."; exit 1 }
    Copy-Item $builtSo (Join-Path $jniDir 'libquest_app.so') -Force

    # windows-x86_64, not darwin-x86_64.
    $CxxShared = Join-Path $Prebuilt 'sysroot\usr\lib\aarch64-linux-android\libc++_shared.so'
    if (Test-Path $CxxShared) {
        Step 'Copying libc++_shared.so into jniLibs ...'
        Copy-Item $CxxShared (Join-Path $jniDir 'libc++_shared.so') -Force
    } else {
        Warn "libc++_shared.so not found at $CxxShared - app will fail to load on device"
    }
    Pop-Location

    Push-Location (Join-Path $QuestAppDir 'android')
    & .\gradlew.bat assembleDebug
    if ($LASTEXITCODE -ne 0) { Fail 'gradle assembleDebug failed.'; exit 1 }
    Pop-Location

    if (-not $WantDeploy) { Step 'Build complete (deploy skipped).'; exit 0 }

    Step 'Installing APK...'
    $apk = Join-Path $QuestAppDir 'android\app\build\outputs\apk\debug\app-debug.apk'
    & adb install -r $apk
    if ($LASTEXITCODE -ne 0) { Fail 'adb install failed.'; exit 1 }

    $registered = $false
    for ($i = 0; $i -lt 15; $i++) {
        if ((& adb shell pm list packages) -match [regex]::Escape($Package)) { $registered = $true; break }
        Start-Sleep -Seconds 1
    }
    if (-not $registered) { Fail "$Package not found in package list after install."; exit 1 }
    Ok 'Package installed and registered.'

    if ($WantEditor) {
        Step 'Reversing TCP debug_viewer port 7778...'
        & adb reverse tcp:7778 tcp:7778
    }

    $manifest = Join-Path $GameDir 'manifest.json'
    if (-not (Test-Path $manifest)) { Fail "$manifest not found. Aborting."; exit 1 }

    if (-not (Wait-AppDataDir)) { exit 1 }

    Step 'Pushing game folder to Quest...'
    & adb shell mkdir -p $RemoteGameDir
    Push-File $manifest "$RemoteGameDir/manifest.json"

    $avatarRig = Join-Path $GameDir 'avatar_rig.json'
    if (Test-Path $avatarRig) { Push-File $avatarRig "$RemoteGameDir/avatar_rig.json" }

    $scenesDir = Join-Path $GameDir 'scenes'
    if (Test-Path $scenesDir) {
        & adb shell mkdir -p "$RemoteGameDir/scenes"
        Get-ChildItem $scenesDir -Filter *.json -File | ForEach-Object {
            Push-File $_.FullName "$RemoteGameDir/scenes/$($_.Name)"
        }
    }

    $modelsDir = Join-Path $GameDir 'models'
    if (Test-Path $modelsDir) {
        Get-ChildItem $modelsDir -Directory -Recurse | ForEach-Object {
            $rel = $_.FullName.Substring($modelsDir.Length).TrimStart('\') -replace '\\','/'
            & adb shell mkdir -p "$RemoteGameDir/models/$rel"
        }
        # EVERY file, not just .glb: the editor's sidecar .bin animation buffers
        # live beside the meshes, and a model without them loads with no clips.
        Get-ChildItem $modelsDir -File -Recurse | ForEach-Object {
            $rel = $_.FullName.Substring($modelsDir.Length).TrimStart('\') -replace '\\','/'
            Push-File $_.FullName "$RemoteGameDir/models/$rel"
        }
    }

    $soundDir = Join-Path $GameDir 'sound'
    if (Test-Path $soundDir) {
        & adb shell mkdir -p "$RemoteGameDir/sound"
        Get-ChildItem $soundDir -File | ForEach-Object {
            Push-File $_.FullName "$RemoteGameDir/sound/$($_.Name)"
        }
    }

    if ($PushFailures.Count -gt 0) {
        Fail "$($PushFailures.Count) file(s) FAILED to reach the headset:"
        $PushFailures | ForEach-Object { Detail "  $_" }
        Warn 'Scene may render incomplete (missing meshes or .bin animation buffers). Re-run to retry.'
    } else {
        Ok 'Game folder verified on device - all files pushed.'
    }

    Step "Verifying multiplayer server is running on $DropletHost..."
    if (Test-Path $DropletKey) {
        & ssh -i $DropletKey -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new `
            $DropletHost "systemctl is-active --quiet $DropletService || systemctl restart $DropletService"
        if ($LASTEXITCODE -eq 0) { Ok 'space-soup-server is active.' }
        else { Warn "Could not confirm space-soup-server on $DropletHost - multiplayer may not work. Check with: ssh -i $DropletKey $DropletHost systemctl status $DropletService" }
    } else {
        Warn "SSH key not found at $DropletKey - skipping multiplayer server check (set QUEST_SERVER_SSH_KEY to override)."
    }

    Step "Pushing multiplayer server URL ($ServerUrl)..."
    & adb shell mkdir -p (Split-Path $RemoteUrlFile -Parent).Replace('\','/')
    # run.sh pipes into `adb shell cat >`. Piping through PowerShell would add a
    # BOM and CRLF, so write a temp file with explicit LF and push that instead.
    $tmp = Join-Path ([System.IO.Path]::GetTempPath()) 'server_url.txt'
    [System.IO.File]::WriteAllText($tmp, "$ServerUrl`n", (New-Object System.Text.UTF8Encoding $false))
    & adb push $tmp $RemoteUrlFile *> $null
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    if (Test-RemoteFile $RemoteUrlFile) { Ok "OK: server_url.txt ($ServerUrl)" }
    else { Warn 'server_url.txt may not have written - client could fall back to its default server.' }

    if ($WantDashboard) {
        Step 'Starting dev dashboard (separate windows)...'
        Start-DashboardWindow 'quest_app - logcat' $QuestAppDir 'adb logcat -s quest_app' | Out-Null
        if ($WantEditor) {
            Start-DashboardWindow 'debug_viewer' $DebugViewerDir "cargo run --target $HostTarget" | Out-Null
            Start-DashboardWindow 'adb reverse keepalive' $QuestAppDir `
                'while ($true) { adb reverse tcp:7778 tcp:7778 2>$null; Start-Sleep -Seconds 5 }' | Out-Null
        }
        Ok 'Dashboard running in separate windows.'
        if ($WantEditor) { Wait-TcpListener 7778 | Out-Null }
    }

    Step 'Launching quest_app on headset...'
    & adb shell am start -n "$Package/android.app.NativeActivity"

    $running = $false
    for ($i = 0; $i -lt 20; $i++) {
        & adb shell pidof $Package *> $null
        if ($LASTEXITCODE -eq 0) { $running = $true; break }
        Start-Sleep -Seconds 1
    }
    if ($running) { Ok 'quest_app process running.' }
    else { Warn "could not confirm $Package process started - check logcat." }

    Step 'All set.'
    if ($WantDashboard) {
        if ($WantEditor) { Detail 'Put on your headset and move around - debug_viewer shows the live scene + player.' }
        Read-Host 'Press Enter when you are done to close the dashboard windows'
    }
}
finally {
    # run.sh's `trap cleanup EXIT`.
    foreach ($p in $DashboardWindows) {
        if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    }
    $gradleDir = Join-Path $QuestAppDir 'android'
    if (Test-Path (Join-Path $gradleDir 'gradlew.bat')) {
        Push-Location $gradleDir
        & .\gradlew.bat --stop *> $null
        Pop-Location
    }
}
