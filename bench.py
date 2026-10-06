#!/usr/bin/env python3
"""Measure what a frame costs on the Quest, unattended, from fixed viewpoints.

Plug the headset in, leave it on the desk, and run one of:

    python3 bench.py                  # every view, the shipped renderer
    python3 bench.py --ab             # every view, every perf_ab phase, 3 passes
    python3 bench.py --views hall_back,hallway --ab --passes 2
    python3 bench.py --levers '{"probe_trace": false}'     # a configuration
    python3 bench.py --compare ../docs/bench/2026-09-27_2210  # against a run
    python3 bench.py --trace 2        # each render pass's bins and stage times

In order, it:

  1. finds THE QUEST -- serial 2G0YC5ZG7706YV reporting model Quest_3 -- and
     touches no other device (the Motorola test phone is never used);
  2. tells the headset it is being worn, so it keeps rendering on the desk;
     locks the CPU and GPU clock levels, so a number measures the work and not
     the clock governor; enlarges the logcat buffer;
  3. starts the app if it is not running;
  4. for each view, writes the lever file with the camera pinned there,
     takes the shipped renderer's windows, a screenshot of what the headset
     shows (the system's own screenshot service), and the GPU's own counters
     (ovrgpuprofiler: busy, stalls, cache misses), then -- with --ab -- runs
     the A/B schedule there, waiting each time until the app's results file
     holds the windows it needs;
  5. puts everything back -- lever file removed, clock levels and the
     proximity sensor restored -- also when interrupted or when it fails;
  6. writes perf.jsonl, vrapi.log, report.md and report.json to the output
     folder, and prints the report.

The views are bench_views.json. The app side: space_soup/src/renderer/bench.rs
(the pinned camera), space_soup/src/perf_record.rs (the results file) and
space_soup/src/renderer/levers.rs (the lever file).
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import statistics
import signal
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

QUEST_SERIAL = "2G0YC5ZG7706YV"
QUEST_MODEL = "Quest_3"
PACKAGE = "com.example.questapp"
ACTIVITY = PACKAGE + "/android.app.NativeActivity"
FILES = "/sdcard/Android/data/" + PACKAGE + "/files"
LEVERS = FILES + "/levers.json"
PERF = FILES + "/perf.jsonl"
HERE = Path(__file__).resolve().parent
VIEWS_FILE = HERE / "bench_views.json"
ADB = "adb"

# Quest 3 levels (Meta, "CPU and GPU levels"): GPU 5 is 599 MHz, the top;
# CPU 4 is 1.92 GHz, what an app gets by default. The system properties
# override whatever the app asks for, and do not survive a reboot.
CPU_LEVEL_PROP = "debug.oculus.cpuLevel"
GPU_LEVEL_PROP = "debug.oculus.gpuLevel"
# Pauses the boundary, so a headset lying still on a desk is not interrupted
# by it. Like the levels, it does not survive a reboot.
GUARDIAN_PAUSE_PROP = "debug.oculus.guardian_pause"
# Where the system's screenshot service (MetaCam) writes.
SCREENSHOT_DIR = "/sdcard/Oculus/Screenshots"
# Lets the renderer copy its eye images out (`Levers::eye_capture`); read when
# the app starts, so setting it means a restart.
EYE_CAPTURE_PROP = "debug.spacesoup.eyecapture"
# Every pipeline logs its shader statistics -- registers, instructions,
# occupancy -- as `PIPESTATS` lines (`xr::vulkan::VkContext::new`). Also read
# when the app starts: set on a running app, it logs nothing (2026-10-01).
PIPESTATS_PROP = "debug.spacesoup.pipestats"
# What each of those reads on a fresh boot. A run restores what it found, but
# what it found can be an earlier run's leftovers: one killed before its
# `finally` left the Guardian paused and the clocks locked, the next run read
# those as "before" and put them back, and the user's own session then ran in
# bench state (2026-10-02). A value equal to the bench's own goes back to this.
STOCK = {GUARDIAN_PAUSE_PROP: "0", CPU_LEVEL_PROP: "", GPU_LEVEL_PROP: "", EYE_CAPTURE_PROP: "", PIPESTATS_PROP: ""}


class BenchError(Exception):
    """Something the person running the benchmark has to fix."""


class Device:
    """The Quest, and only the Quest: every command is pinned to its serial."""

    def __init__(self, serial: str):
        self.serial = serial

    def run(self, *args: str, check: bool = True, timeout: float = 120) -> str:
        cmd = [ADB, "-s", self.serial, *args]
        try:
            p = subprocess.run(cmd, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise BenchError("timed out: " + " ".join(cmd))
        if check and p.returncode != 0:
            err = p.stderr.decode(errors="replace").strip() or p.stdout.decode(errors="replace").strip()
            raise BenchError("failed (%d): %s\n  %s" % (p.returncode, " ".join(cmd), err))
        return p.stdout.decode(errors="replace")

    def shell(self, command: str, check: bool = True, timeout: float = 120) -> str:
        return self.run("shell", command, check=check, timeout=timeout)


def find_quest() -> Device:
    try:
        p = subprocess.run([ADB, "devices", "-l"], capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        raise BenchError("adb is not on PATH")
    if p.returncode != 0:
        raise BenchError("adb devices failed: " + p.stderr.strip())
    for line in p.stdout.splitlines()[1:]:
        fields = line.split()
        if len(fields) < 2 or fields[0] != QUEST_SERIAL:
            continue
        if fields[1] != "device":
            raise BenchError(
                "the Quest is connected but %r: put it on once, allow USB debugging, and run again" % fields[1]
            )
        if "model:" + QUEST_MODEL not in fields:
            raise BenchError("serial %s does not report model:%s (%s); refusing" % (QUEST_SERIAL, QUEST_MODEL, line.strip()))
        return Device(QUEST_SERIAL)
    raise BenchError(
        "the Quest 3 (%s) is not connected. No other device is ever used, "
        "so nothing was touched." % QUEST_SERIAL
    )


def load_views(names: str | None) -> list:
    doc = json.loads(VIEWS_FILE.read_text())
    views = doc["views"]
    by_name = {v["name"]: v for v in views}
    if len(by_name) != len(views):
        raise BenchError("bench_views.json names a view twice")
    if not names:
        return views
    picked = []
    for name in [n.strip() for n in names.split(",") if n.strip()]:
        view = None
        for sep, derive in (("-t", lambda v, q: turned_view(v, q / 100.0, name)),
                            ("-m", lambda v, q: moved_view(v, q / 10000.0, name)),
                            ("-y", lambda v, q: snap_turned_view(v, q, name)),
                            ("-f", lambda v, q: flashlight_moved_view(v, q / 10000.0, name))):
            base, found, q = name.rpartition(sep)
            if found and q.isdigit() and base in by_name:
                view = derive(by_name[base], int(q))
                break
        if view is None:
            if name not in by_name:
                raise BenchError("no view %r; bench_views.json has: %s" % (name, ", ".join(by_name)))
            view = by_name[name]
        if name not in [v["name"] for v in picked]:
            picked.append(view)
    return picked


# One eye pixel of the left eye's image across its middle, in radians: the
# frustum's tangents left and right (1.376 + 0.839) over its 1176 pixels.
EYE_PIXEL_RAD = (1.376 + 0.839) / 1176


def turned_view(view: dict, pixels: float, name: str) -> dict:
    """`view` turned right by `pixels` eye pixels about its eye -- `NAME-tQ`,
    Q in hundredths of a pixel (`hall_back-t25`; the app takes only letters,
    digits, '_' and '-' in a view's name):
    the same picture a fraction of a pixel over, to measure losslessly with
    --eye-capture whether small bright things hold their size and light as
    the head turns -- shimmer, without a video encoder in the way."""
    eye, at = view["eye"], view["at"]
    dx, dz = at[0] - eye[0], at[2] - eye[2]
    # Turned about +y by `a`: (x, z) -> (c x - s z, s x + c z). Facing +z,
    # right is -x, so a positive angle turns right.
    a = pixels * EYE_PIXEL_RAD
    c, s = math.cos(a), math.sin(a)
    turned = [eye[0] + c * dx - s * dz, at[1], eye[2] + s * dx + c * dz]
    return dict(view, name=name, at=turned, note="%s, turned %g eye pixels right" % (view["name"], pixels))


# What every bench view measures unless `--levers` says otherwise: the native
# 72 Hz frame. SpaceWarp ships ON (2026-09-29) and paces the app to 36 fps,
# which would halve every fps figure and hide the 13.9 ms budget this bench
# exists to hold; multiview is pinned because the app keeps whatever the last
# toggle left it at.
MEASURED = {"space_warp": False, "multiview": False}



def moved_view(view: dict, metres: float, name: str) -> dict:
    """`view` with the head moved `metres` to the right, still looking the
    same way -- `NAME-mQ`, Q in tenths of a millimetre (`hall_back-m20` is 2
    mm). A turn changes no surface's view of its reflection; a move does: the
    measure of reflections and highlights sliding as the head moves."""
    eye, at = view["eye"], view["at"]
    f = [at[i] - eye[i] for i in range(3)]
    r = [-f[2], 0.0, f[0]]  # forward x up, up = +y
    n = math.hypot(r[0], r[2]) or 1.0
    d = [r[0] / n * metres, 0.0, r[2] / n * metres]
    return dict(view, name=name, eye=[eye[i] + d[i] for i in range(3)], at=[at[i] + d[i] for i in range(3)],
                note="%s, moved %g mm right" % (view["name"], metres * 1000.0))

def flashlight_moved_view(view: dict, metres: float, name: str) -> dict:
    """`view` with its flashlight moved `metres` to the right, glass and aim
    point together, the head held still -- `NAME-fQ`, Q in tenths of a
    millimetre (`torch_pillar-f20` is 2 mm). A light that moves moves every
    shadow and highlight it makes; this measures whether they glide or crawl,
    as `-m` does for the head."""
    f = view.get("flashlight")
    if not f:
        raise BenchError("%s holds no flashlight to move" % view["name"])
    eye, at = view["eye"], view["at"]
    fw = [at[i] - eye[i] for i in range(3)]
    r = [-fw[2], 0.0, fw[0]]  # forward x up, up = +y
    n = math.hypot(r[0], r[2]) or 1.0
    d = [r[0] / n * metres, 0.0, r[2] / n * metres]
    moved = dict(f, at=[f["at"][i] + d[i] for i in range(3)], aim=[f["aim"][i] + d[i] for i in range(3)])
    return dict(view, name=name, flashlight=moved,
                note="%s, the flashlight moved %g mm right" % (view["name"], metres * 1000.0))


def snap_turned_view(view: dict, degrees: int, name: str) -> dict:
    """The same eyes looking the same way, with the rig turned `degrees` (0-359)
    as snap or smooth turning leaves it and the head turned back by the rest --
    `NAME-yQ`. Every picture must match the unturned one: a difference is
    something still kept in the player's frame instead of the world's."""
    return dict(view, name=name, rig_yaw=float(degrees),
                note="%s, rig turned %d degrees" % (view["name"], degrees))


def levers_for(view: dict, ab: bool, extra: dict, synced: bool = False) -> dict:
    """The lever file for one view: the extra levers, the pin, the schedule.
    `synced` blocks on the GPU every frame, so the render thread's wait is
    the GPU's time; the A/B schedule always runs synced."""
    levers = dict(MEASURED, **extra)
    levers["bench"] = {"name": view["name"], "eye": view["eye"], "at": view["at"]}
    # The same view with the rig turned as a snap turn leaves it (BenchPose::rig_yaw).
    if "rig_yaw" in view:
        levers["bench"]["rig_yaw"] = view["rig_yaw"]
    # A head that sways side to side (BenchPose::sway), for watching what moves.
    if "sway" in view:
        levers["bench"]["sway"] = view["sway"]
    # The player's flashlight held still at a place in the world (BenchPose::flashlight).
    if "flashlight" in view:
        levers["bench"]["flashlight"] = view["flashlight"]
    if ab:
        levers["ab_cycle"] = True
    if ab or synced:
        levers["gpu_sync"] = True
    return levers


def record_kind(r: dict) -> str:
    """Which step a window belongs to: `shipped` (pipelined, as the headset
    runs), `synced` (blocking on the GPU, so the wait is its time) or the
    A/B phase it ran under."""
    if r.get("cycle_len", 1) > 1 or r.get("phase", "-") not in ("-", ""):
        return r.get("phase", "-")
    return "synced" if "gpu_sync" in (r.get("levers") or "") else "shipped"


_last_lever_length = [-1]


def write_levers(dev: Device, levers: dict) -> None:
    """Pushed to a temporary name and moved over the lever file, so the app
    never reads half a file.

    The app notices a new file by its modification time and length
    (`LeverFile::poll`), and `adb push` stamps whole seconds: two files of the
    same length pushed within a second look like one, and the second view is
    never picked up -- `back_wall_self-y180` then `-y270` stalled a run
    (2026-10-01). A file as long as the last one written takes a trailing
    space, which JSON ignores."""
    text = json.dumps(levers, separators=(",", ":"))
    if len(text) == _last_lever_length[0]:
        text += " "
    _last_lever_length[0] = len(text)
    fd, tmp = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        dev.run("push", tmp, LEVERS + ".tmp")
        dev.shell("mv %s.tmp %s" % (LEVERS, LEVERS))
    finally:
        os.unlink(tmp)


def read_records(dev: Device) -> list:
    text = dev.shell("cat %s 2>/dev/null" % PERF, check=False)
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # the line being written as it was read
    return out


def wait_for_view(dev: Device, name: str, ab: bool, passes: int, windows: int,
                  stall_seconds: float, poll_seconds: float, synced: bool = False) -> list:
    """The view's measured windows of one step, once there are enough."""
    last_seen = -1
    last_change = time.monotonic()
    while True:
        records = read_records(dev)
        mine = [r for r in records if r.get("bench") == name]
        # The shipped windows and the schedule's are told apart by whether
        # the app was cycling, so neither step counts the other's windows.
        def this_step(r: dict) -> bool:
            if r.get("warmup") or (r.get("cycle_len", 1) > 1) != ab:
                return False
            return ab or (record_kind(r) == "synced") == synced
        measured = sorted((r for r in mine if this_step(r)), key=lambda r: r.get("window", 0))
        if ab:
            cycle = max((r.get("cycle_len", 0) for r in measured), default=0)
            need = cycle * passes if cycle > 1 else None
        else:
            need = windows
        if need and len(measured) >= need:
            return measured[:need]
        if len(mine) != last_seen:
            last_seen = len(mine)
            last_change = time.monotonic()
            print("  %s: %d measured window(s)%s" % (name, len(measured), " of %d" % need if need else ""), flush=True)
        if time.monotonic() - last_change > stall_seconds:
            message = stall_diagnosis(name, records, stall_seconds)
            blocked = launch_block(dev) if not records else None
            if blocked:
                message += " The Quest blocked the app's launch: " + blocked
            raise BenchError(message)
        time.sleep(poll_seconds)


def launch_block(dev: Device) -> str | None:
    """A system dialog in the way of the app's launch, from the recent log --
    e.g. "Switch to Controllers" for an app that declares no hand tracking,
    with the controllers asleep on the desk."""
    text = dev.run("logcat", "-d", "-t", "4000", check=False, timeout=60)
    for line in reversed(text.splitlines()):
        if PACKAGE in line and ("launch_blocked" in line or "Launch is blocked" in line):
            return line.strip()
    return None


def stall_diagnosis(name: str, records: list, waited: float) -> str:
    head = "no new window for view %r in %.0f s. " % (name, waited)
    if not records:
        return head + (
            "The app has written no results at all: the installed build may predate the "
            "benchmark mode (deploy it), or the app is not rendering (is the headset awake, "
            "and has tracking located it at least once?)."
        )
    last = records[-1]
    if last.get("bench") != name:
        return head + (
            "The app's last window was for %r with levers %r: it has not picked up this "
            "view's lever file." % (last.get("bench"), last.get("levers"))
        )
    return head + "The app stopped writing windows mid-view: is it still running (adb logcat -s quest_app)?"


# What the GPU's own counters are asked for: whether the frame waits on
# arithmetic, on textures or on memory, and how much hidden surface the
# hardware's low-resolution depth (LRZ) throws away before shading. Names as
# `ovrgpuprofiler -m` prints them; the tool reads at most 30, so when more
# match, the earlier keywords win.
PROFILER_KEYWORDS = (
    "clocks / second", "frequency", "shaders busy", "alu", "texture fetch stall", "stall",
    "texture l1 miss", "texture l2 miss", "lrz", "fragment", "bus busy", "utilization",
    "read total", "write total",
)
PROFILER_LIMIT = 30


def profiler_metrics(dev: Device) -> tuple:
    """The counters the GPU offers: [(id, name)], and the raw listing."""
    text = dev.shell("ovrgpuprofiler -m", check=False, timeout=30)
    metrics = []
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+)\s+(\S.*?)\s*$", line)
        if m:
            metrics.append((int(m.group(1)), m.group(2)))
    return metrics, text


def pick_metrics(metrics: list) -> list:
    """Up to 30 counters: every keyword gets its first few before any keyword
    gets more, so one family with many counters cannot crowd out the rest."""
    per_keyword: list = [[] for _ in PROFILER_KEYWORDS]
    for i, name in metrics:
        lower = name.lower()
        k = next((k for k, word in enumerate(PROFILER_KEYWORDS) if word in lower), None)
        if k is not None:
            per_keyword[k].append((i, name))
    if not any(per_keyword):
        return metrics[:PROFILER_LIMIT]
    chosen: list = []
    depth = 0
    while len(chosen) < PROFILER_LIMIT and any(len(group) > depth for group in per_keyword):
        for group in per_keyword:
            if len(group) > depth and len(chosen) < PROFILER_LIMIT:
                chosen.append(group[depth])
        depth += 1
    return chosen


def profile(dev: Device, ids: list, seconds: int) -> tuple:
    """Each counter's mean over `seconds` of `ovrgpuprofiler -r`, the first
    second left out (it can straddle the start), and the raw output."""
    text = dev.shell('timeout %d ovrgpuprofiler -r"%s"' % (seconds, ",".join(str(i) for i in ids)),
                     check=False, timeout=seconds + 30)
    samples: dict = {}
    for line in text.splitlines():
        m = re.match(r"^\s*(\S.*?)\s*:\s*(-?[\d.]+(?:[eE][-+]?\d+)?)\s*$", line)
        if m:
            samples.setdefault(m.group(1), []).append(float(m.group(2)))
    means = {k: statistics.mean(v[1:] if len(v) > 2 else v) for k, v in samples.items()}
    return means, text


# One render pass as the render-stage trace reports it, e.g.
#   Surface 1 | 1216x1344 | color 32bit, depth 24bit, stencil 0 bit, MSAA 4,
#   Mode: 1 (HwBinning) | 60 128x224 bins ( 60 rendered) | 5.08 ms | 130 stages
#   : Binning : 0.623ms Render : 1.877ms StoreColor : 0.309ms Preempt : 1.286ms
# (Meta, "ovrgpuprofiler"); the tool pads sizes as `588 x616`. Read loosely -- the fields are searched for rather
# than split by position -- because the only sample of the format is the
# documentation's, and the raw trace is kept beside the report either way.
TRACE_SURFACE = re.compile(r"^\s*Surface\s+\d+\s*\|")
TRACE_SIZE = re.compile(r"\|\s*(\d+)\s*x\s*(\d+)\s*\|")
TRACE_MSAA = re.compile(r"MSAA\s*(\d+)")
TRACE_MODE = re.compile(r"Mode:\s*\d*\s*\(?([A-Za-z]+)")
TRACE_BINS = re.compile(r"(\d+)\s+(\d+)x(\d+)\s+bins(?:\s*\(\s*(\d+)\s+rendered\))?")
TRACE_MS = re.compile(r"\|\s*([\d.]+)\s*ms\s*\|")
TRACE_STAGE = re.compile(r"([A-Za-z]+)\s*:\s*([\d.]+)\s*ms")


def summarize_trace(text: str, seconds: float) -> list:
    """Each KIND of render pass in a render-stage trace -- same size, MSAA,
    mode and bins -- with how often it ran, its mean time and each stage's
    mean time, heaviest first. A frame's passes are told apart by size: the
    eye buffer, the half-resolution reflection target, the shadow maps."""
    kinds: dict = {}
    for line in text.splitlines():
        if not TRACE_SURFACE.match(line):
            continue
        size = TRACE_SIZE.search(line)
        ms = TRACE_MS.search(line)
        if not size or not ms:
            continue
        msaa = TRACE_MSAA.search(line)
        mode = TRACE_MODE.search(line)
        bins = TRACE_BINS.search(line)
        key = "%sx%s msaa%s %s %s" % (size.group(1), size.group(2), msaa.group(1) if msaa else "?",
                                      mode.group(1) if mode else "?",
                                      "%s bins of %sx%s" % bins.group(1, 2, 3) if bins else "no bins")
        k = kinds.setdefault(key, {"surface": key, "count": 0, "ms": 0.0, "stages": {}})
        k["count"] += 1
        k["ms"] += float(ms.group(1))
        tail = line.split("stages", 1)[1] if "stages" in line else ""
        for stage, value in TRACE_STAGE.findall(tail):
            k["stages"][stage] = k["stages"].get(stage, 0.0) + float(value)
    out = []
    for k in kinds.values():
        n = k["count"]
        out.append({
            "surface": k["surface"],
            "count": n,
            "per_second": n / seconds if seconds else None,
            "mean_ms": k["ms"] / n,
            "stages_ms": {s: v / n for s, v in sorted(k["stages"].items(), key=lambda kv: -kv[1])},
        })
    out.sort(key=lambda k: -(k["mean_ms"] * k["count"]))
    return out


def trace(dev: Device, seconds: int) -> tuple:
    """A render-stage trace of `seconds` (`ovrgpuprofiler -t`): the summary
    and the raw output. Needs the app started in detailed profiling mode."""
    text = dev.shell("ovrgpuprofiler -t%d" % seconds, check=False, timeout=seconds + 90)
    return summarize_trace(text, seconds), text


# One draw call in a per-draw trace (`ovrgpuprofiler -t -x=...`), e.g.
#   Frame 1   : Draw 1   Label 0xffffffff
#       LRZ State: TestEnabled, WriteEnabled <0x03>
#       Clocks                                     :       58912.000
# Meta documents the numbers as comparable between draws of one trace only.
DRAW_HEAD = re.compile(r"Frame\s+(\d+)\s*:\s*Draw\s+(\d+)")
DRAW_METRIC = re.compile(r"^\s*([A-Za-z%/()][^:]*?)\s*:\s*(-?[\d.]+(?:[eE][-+]?\d+)?)\s*$")


def summarize_draws(text: str, limit: int = 40) -> list:
    """Each draw call of a per-draw trace, by its position -- `cb.draw`, the
    command buffer and the draw within it -- with every metric averaged over
    the frames that drew it; heaviest (Clocks) first. A draw's position is
    what identifies it: the tool has no names."""
    draws: dict = {}
    current = None
    # Draws are numbered within a command buffer, and the numbering starts
    # again at 1 in the next: `cb.draw` names one.
    cb, last = 0, None
    for line in text.splitlines():
        head = DRAW_HEAD.search(line)
        if head:
            n = int(head.group(2))
            if last is not None and n <= last:
                cb += 1
            last = n
            key = "%d.%d" % (cb, n)
            current = draws.setdefault(key, {"draw": key, "frames": 0, "sums": {}})
            current["frames"] += 1
            continue
        if current is None:
            continue
        m = DRAW_METRIC.match(line)
        if m:
            current["sums"][m.group(1)] = current["sums"].get(m.group(1), 0.0) + float(m.group(2))
    out = [{"draw": d["draw"], "frames": d["frames"],
            "metrics": {k: v / d["frames"] for k, v in d["sums"].items()}} for d in draws.values()]
    out.sort(key=lambda d: -d["metrics"].get("Clocks", 0.0))
    return out[:limit]


def trace_draws(dev: Device, seconds: int, metrics: str) -> tuple:
    """A per-draw trace (`ovrgpuprofiler -t -x=`) of `seconds` with the given
    metric ids: the summary and the raw output. Detailed mode, like `trace`."""
    # The ids ATTACH to `-x`, with no `=` and no space: `-x1,14,18`. Tried on
    # the Quest, 2026-09-28: `-x=ids` traces Clocks only, `-x ids` and
    # `-x -s=ids` track nothing ("-s ignored in drawcall mode"), and
    # `--drawcall=ids` captures no draws. A trace under 2 s may catch none.
    text = dev.shell("ovrgpuprofiler -t%d -x%s" % (max(seconds, 2), metrics), check=False, timeout=seconds + 120)
    return summarize_draws(text), text


def list_screenshots(dev: Device) -> list:
    text = dev.shell("ls -1 %s 2>/dev/null" % SCREENSHOT_DIR, check=False)
    return sorted(line.strip() for line in text.splitlines() if line.strip())


def screenshot(dev: Device, dest_stem: Path, timeout: float = 20.0, poll: float = 1.0) -> Path | None:
    """What the headset shows now, through the system's own screenshot
    service, pulled to `dest_stem` plus the device file's extension. `None`
    when no new file appeared."""
    before = set(list_screenshots(dev))
    dev.shell("am startservice -n com.oculus.metacam/.capture.CaptureService -a TAKE_SCREENSHOT", check=False)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(poll)
        new = [f for f in list_screenshots(dev) if f not in before]
        if not new:
            continue
        remote = "%s/%s" % (SCREENSHOT_DIR, new[-1])
        # Pulled once the file has stopped growing: the service writes it
        # after announcing it.
        size = None
        while time.monotonic() < deadline:
            now = dev.shell("stat -c %%s %s" % remote, check=False).strip()
            if now and now == size:
                break
            size = now
            time.sleep(poll)
        local = dest_stem.with_suffix(Path(new[-1]).suffix or ".jpg")
        dev.run("pull", remote, str(local), check=False)
        return local if local.exists() else None
    return None


def eye_capture(dev: Device, levers: dict, n: int, dest_stem: Path, timeout: float = 30.0,
                poll: float = 1.0) -> list | None:
    """Both eyes' finished images of one frame, from the renderer itself
    (`Levers::eye_capture`): the system's screenshot is ONE view, and a
    difference between the eyes shows no other way. Written beside
    `dest_stem` as `<name>_eye_left.png` and `_eye_right.png`; their names, or
    `None` when nothing came back."""
    remote = "%s/eyecapture_%d.bin" % (FILES, n)
    dev.shell("rm -f " + remote, check=False)
    write_levers(dev, dict(levers, eye_capture=n))
    deadline = time.monotonic() + timeout
    size = None
    while time.monotonic() < deadline:
        time.sleep(poll)
        now = dev.shell("stat -c %%s %s" % remote, check=False).strip()
        if now.isdigit() and now == size:
            break
        size = now if now.isdigit() else None
    else:
        return None
    local = dest_stem.parent / (dest_stem.name + "_eyes.bin")
    dev.run("pull", remote, str(local))
    dev.shell("rm -f " + remote, check=False)
    names = write_eye_pngs(local.read_bytes(), dest_stem)
    local.unlink()
    return names


VIDEO_DIR = "/sdcard/Oculus/VideoShots"


def record_video(dev: Device, seconds: float, dest: Path, timeout: float = 60.0, poll: float = 1.0) -> Path | None:
    """What the headset shows for `seconds`, through the system's own video
    capture -- the compositor's output, SpaceWarp's synthesised frames
    included, which no eye capture can show -- pulled to `dest`. `None` when
    no new file appeared."""
    listing = lambda: set(dev.shell("ls -1 %s 2>/dev/null" % VIDEO_DIR, check=False).split())
    before = listing()
    # The capture service's own debugging actions (its package's strings; the
    # START_CAPTURE some write-ups give is refused as invalid). It records
    # only while the display is on: run inside the bench, which holds the
    # proximity sensor closed.
    service = "am startservice -n com.oculus.metacam/.capture.CaptureService -a %s_INTERNAL_CAPTURE_TO_DISK"
    dev.shell(service % "START", check=False)
    time.sleep(seconds)
    dev.shell(service % "STOP", check=False)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(poll)
        new = sorted(f for f in listing() if f not in before and f.endswith(".mp4"))
        if not new:
            continue
        remote = "%s/%s" % (VIDEO_DIR, new[-1])
        # The file is there from the start, its header first; it is finished
        # once its size has held still for a few seconds after the stop.
        size, still = None, 0
        while time.monotonic() < deadline and still < 3:
            time.sleep(poll)
            now = dev.shell("stat -c %%s %s" % remote, check=False).strip()
            still = still + 1 if (now and now == size and int(now) > 4096) else 0
            size = now
        dev.run("pull", remote, str(dest), check=False)
        dev.shell("rm -f " + remote, check=False)
        return dest if dest.exists() else None
    return None


def write_eye_pngs(data: bytes, dest_stem: Path) -> list:
    """A capture (`EYES`, width, height as little-endian u32s, then each eye's
    RGBA rows) as two PNGs, left then right; their names."""
    if data[:4] != b"EYES":
        raise BenchError("not an eye capture")
    width, height = struct.unpack("<II", data[4:12])
    eye_bytes = width * height * 4
    if len(data) != 12 + 2 * eye_bytes:
        raise BenchError("an eye capture of %dx%d should be %d bytes, not %d" % (width, height, 12 + 2 * eye_bytes, len(data)))
    names = []
    for i, side in enumerate(("left", "right")):
        rgba = data[12 + i * eye_bytes:12 + (i + 1) * eye_bytes]
        rows = b"".join(b"\x00" + rgba[y * width * 4:(y + 1) * width * 4] for y in range(height))
        def chunk(kind: bytes, body: bytes) -> bytes:
            return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
        png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b""))
        path = dest_stem.parent / ("%s_eye_%s.png" % (dest_stem.name, side))
        path.write_bytes(png)
        names.append(path.name)
    return names


def median(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def summarize_view(records: list) -> dict:
    """Per phase: the median over passes of each number, and its spread."""
    phases: dict = {}
    for r in records:
        phases.setdefault(record_kind(r), []).append(r)
    out = {}
    for phase, rs in phases.items():
        xr_gpu = [r.get("xr", {}).get("app/gpu_frametime") for r in rs]
        xr_vals = [v for v in xr_gpu if v is not None]
        waits = [r.get("gpu_avg") for r in rs if r.get("gpu_avg") is not None]
        passes = {}
        for label in sorted({k for r in rs for k in r.get("pass", {})}):
            passes[label] = median([r.get("pass", {}).get(label) for r in rs])
        out[phase] = {
            "windows": len(rs),
            "levers": rs[0].get("levers"),
            "app_gpu_ms": median(xr_gpu),
            "app_gpu_spread_ms": (max(xr_vals) - min(xr_vals)) if len(xr_vals) > 1 else 0.0,
            "gpu_wait_ms": median([r.get("gpu_avg") for r in rs]),
            "gpu_wait_spread_ms": (max(waits) - min(waits)) if len(waits) > 1 else 0.0,
            "cpu_ms": median([r.get("cpu_avg") for r in rs]),
            "frame_ms": median([r.get("frame_ms") for r in rs]),
            "fps": median([r.get("fps") for r in rs]),
            "passes_ms": passes,
            "xr": {k: median([r.get("xr", {}).get(k) for r in rs]) for k in sorted({k for r in rs for k in r.get("xr", {})})},
        }
    return out


VRAPI_CLOCKS = re.compile(r"CPU\d*/GPU=(\d+)/(\d+),(\d+)/(\d+)MHz")
VRAPI_TEMP = re.compile(r"Temp=([\d.]+)C")


def summarize_vrapi(text: str) -> dict:
    """The clock levels, clocks and temperature the runtime logged."""
    cpu_levels, gpu_levels, gpu_mhz, temps = set(), set(), set(), []
    lines = 0
    for line in text.splitlines():
        m = VRAPI_CLOCKS.search(line)
        if not m:
            continue
        lines += 1
        cpu_levels.add(int(m.group(1)))
        gpu_levels.add(int(m.group(2)))
        gpu_mhz.add(int(m.group(4)))
        t = VRAPI_TEMP.search(line)
        if t:
            temps.append(float(t.group(1)))
    return {
        "lines": lines,
        "cpu_levels": sorted(cpu_levels),
        "gpu_levels": sorted(gpu_levels),
        "gpu_mhz": sorted(gpu_mhz),
        "temp_c": [min(temps), max(temps)] if temps else None,
    }


def fmt(v: float | None, digits: int = 2) -> str:
    return "-" if v is None else ("%." + str(digits) + "f") % v


def render_report(run: dict, views: dict, compare: dict | None) -> str:
    lines = ["# Quest benchmark %s" % run["started"], ""]
    lines.append("Build levers: `%s`. Clock levels: CPU %s, GPU %s. %s." % (
        run["extra_levers"] or "{}", run["cpu_level"], run["gpu_level"],
        "A/B schedule, %d pass(es)" % run["passes"] if run["ab"] else "%d window(s) a view" % run["windows"]))
    lines.append("")
    lines.append("`GPU ms` is the render thread's wait for the GPU to finish each frame, averaged over a "
                 "window and the median over passes; `cost` is what switching the feature off saves "
                 "against the baseline -- negative for an optimisation, whose lever switches it off. "
                 "`shipped` is the renderer with no schedule running, and should agree with `baseline`. "
                 "`spread` is the range over passes. `app GPU` is the runtime's own counter "
                 "(XR_META_performance_metrics); it LAGS -- smoothed over about a second -- so a window "
                 "right after a slow phase reads high, and it is shown for reference only.")
    for name, v in views.items():
        phases = v["phases"]
        lines += ["", "## %s" % name, ""]
        vr = v.get("vrapi") or {}
        if vr.get("lines"):
            lines.append("GPU level %s at %s MHz, CPU level %s, %s." % (
                "/".join(map(str, vr["gpu_levels"])), "/".join(map(str, vr["gpu_mhz"])),
                "/".join(map(str, vr["cpu_levels"])),
                "%.1f-%.1f C" % tuple(vr["temp_c"]) if vr.get("temp_c") else "no temperature"))
            if len(vr["gpu_mhz"]) > 1:
                lines.append("**The GPU clock changed during this view: its numbers are not comparable.**")
            lines.append("")
        # The A/B baseline when there is one, else the synced windows: both
        # block on the GPU, so their wait is its time.
        base = phases.get("baseline") or phases.get("synced") or phases.get("-")
        shipped = phases.get("shipped")
        if shipped:
            lines.append("**As shipped (pipelined): %s ms a frame, %s fps**; the runtime's app GPU %s ms." % (
                fmt(shipped["frame_ms"]), fmt(shipped["fps"], 1), fmt(shipped["app_gpu_ms"])))
            lines.append("")
        if v.get("screenshot"):
            lines.append("![%s](%s)" % (name, v["screenshot"]))
        for eye in v.get("eyes") or []:
            lines.append("![%s](%s)" % (eye, eye))
            lines.append("")
        lines.append("| phase | GPU ms | cost ms | spread | app GPU ms | frame ms | fps | CPU ms |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for phase, p in phases.items():
            cost = None
            if base and p is not base and base["gpu_wait_ms"] is not None and p["gpu_wait_ms"] is not None:
                cost = base["gpu_wait_ms"] - p["gpu_wait_ms"]
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                phase, fmt(p["gpu_wait_ms"]) if phase != "shipped" else "-", fmt(cost) if phase != "shipped" else "-",
                fmt(p["gpu_wait_spread_ms"]) if phase != "shipped" else "-",
                fmt(p["app_gpu_ms"]), fmt(p["frame_ms"]), fmt(p["fps"], 1), fmt(p["cpu_ms"])))
        if base and base["passes_ms"]:
            lines.append("")
            lines.append("Passes (baseline, one frame each): " + ", ".join(
                "%s %s" % (k, fmt(ms)) for k, ms in base["passes_ms"].items() if ms))
        if v.get("trace"):
            lines.append("")
            lines.append("Render passes (shipped renderer, `ovrgpuprofiler -t`, detailed profiling mode; "
                         "mean over the trace, heaviest first):")
            lines.append("")
            lines.append("| pass | a second | ms | stages (ms) |")
            lines.append("|---|---|---|---|")
            for k in v["trace"]:
                lines.append("| %s | %s | %s | %s |" % (
                    k["surface"], fmt(k["per_second"], 1), fmt(k["mean_ms"], 3),
                    ", ".join("%s %s" % (s, fmt(ms, 3)) for s, ms in k["stages_ms"].items())))
        if v.get("draws"):
            names = []
            for d in v["draws"]:
                names += [k for k in d["metrics"] if k not in names]
            lines.append("")
            lines.append("Draw calls (per-draw trace, heaviest first; numbers compare draws of one trace only):")
            lines.append("")
            lines.append("| draw | frames | " + " | ".join(names) + " |")
            lines.append("|---|---|" + "---|" * len(names))
            for d in v["draws"]:
                lines.append("| %s | %d | %s |" % (d["draw"], d["frames"], " | ".join(
                    fmt(d["metrics"].get(k), 1) for k in names)))
        if v.get("gpu_counters"):
            lines.append("")
            lines.append("GPU counters (shipped renderer, ovrgpuprofiler, mean over the capture):")
            lines.append("")
            lines.append("| counter | value |")
            lines.append("|---|---|")
            for k, val in v["gpu_counters"].items():
                lines.append("| %s | %s |" % (k, fmt(val, 3)))
        if compare and name in compare.get("views", {}):
            old = compare["views"][name]["phases"]
            old_base = old.get("baseline") or old.get("synced") or old.get("-")
            old_shipped = old.get("shipped") or old.get("-")
            if old_shipped and shipped and old_shipped.get("frame_ms") and shipped.get("frame_ms"):
                lines.append("")
                lines.append("Against %s: a frame %s -> %s ms as shipped." % (
                    compare["run"]["started"], fmt(old_shipped["frame_ms"]), fmt(shipped["frame_ms"])))
            if old_base and base and old_base.get("gpu_wait_ms") is not None and base["gpu_wait_ms"] is not None:
                d = base["gpu_wait_ms"] - old_base["gpu_wait_ms"]
                lines.append("")
                lines.append("Against %s: GPU %s -> %s ms (%+.2f ms, %s)." % (
                    compare["run"]["started"], fmt(old_base["gpu_wait_ms"]), fmt(base["gpu_wait_ms"]), d,
                    "slower" if d > 0 else "faster"))
    return "\n".join(lines) + "\n"


def set_prop(dev: Device, changed: list, warnings: list, prop: str, value) -> None:
    """Set a system property and queue its undo: back to what it was, or to
    `STOCK` when what it was is the bench's own value -- see `STOCK`."""
    before = dev.shell("getprop " + prop).strip()
    if before == str(value) and before != STOCK[prop]:
        warnings.append("%s was already %s, left by a run that never restored it; it goes back to %r"
                        % (prop, before, STOCK[prop]))
        before = STOCK[prop]
    dev.shell("setprop %s %s" % (prop, value))
    changed.append(("restore %s to %r" % (prop, before),
                    lambda: dev.shell("setprop %s '%s'" % (prop, before))))


def restore(dev: Device, changed: list, warnings: list) -> None:
    """Undo what was changed, newest first; each step on its own, so one that
    fails does not leave the others undone."""
    for what, undo in reversed(changed):
        try:
            undo()
        except (BenchError, OSError) as e:
            warnings.append("could not %s: %s" % (what, e))


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--views", help="comma-separated view names (default: every view in bench_views.json)")
    ap.add_argument("--ab", action="store_true", help="run the perf_ab schedule at each view")
    ap.add_argument("--passes", type=int, default=3, help="passes through the schedule with --ab (default 3)")
    ap.add_argument("--windows", type=int, default=3, help="windows a view without --ab (default 3)")
    ap.add_argument("--levers", default="", help="extra levers for every view, as JSON")
    ap.add_argument("--cpu-level", type=int, default=4, help="lock the CPU level (Quest 3: 0-6, 8; default 4)")
    ap.add_argument("--gpu-level", type=int, default=5, help="lock the GPU level (Quest 3: 0-5; default 5)")
    ap.add_argument("--no-lock", action="store_true", help="leave the clock levels to the governor")
    ap.add_argument("--no-profile", action="store_true", help="skip the GPU counters (ovrgpuprofiler)")
    ap.add_argument("--no-screenshots", action="store_true", help="skip the screenshot from each view")
    ap.add_argument("--eye-capture", action="store_true",
                    help="also both eyes' finished images at each view, from the renderer itself (restarts the app "
                         "with debug.spacesoup.eyecapture set, and afterwards without it)")
    ap.add_argument("--pipestats", action="store_true",
                    help="also every pipeline's shader statistics -- registers, instructions, occupancy -- into "
                         "pipestats.log (restarts the app with debug.spacesoup.pipestats set, and afterwards "
                         "without it)")
    ap.add_argument("--record", type=float, metavar="SECONDS",
                    help="also record what the headset shows at each view for this long, through the system's "
                         "video capture: SpaceWarp's synthesised frames included (`<view>.mp4`)")
    ap.add_argument("--profile-seconds", type=int, default=10, help="how long to read the GPU counters a view (default 10)")
    ap.add_argument("--trace", type=int, metavar="SECONDS",
                    help="capture a render-stage trace this long at each view (ovrgpuprofiler -t): each pass's "
                         "bins and binning, rendering, store and pre-emption times. Restarts the app in the "
                         "driver's detailed profiling mode and back out of it afterwards; skips the counters")
    ap.add_argument("--draws", metavar="IDS",
                    help="with --trace, also a per-draw trace of these metric ids (ovrgpuprofiler -x -m lists "
                         "them), e.g. 1,14,18,19,28,36 -- clocks, ALU use, wave occupancy, instruction cache "
                         "misses, fragments and ALU a fragment for every draw call. Some sets come back Clocks "
                         "only; that one is known to work")
    ap.add_argument("--stage-metrics", metavar="IDS",
                    help="with --trace, also a render-stage trace with these metric ids for every pass "
                         "(ovrgpuprofiler -t -s; `-m -t` lists them) -- per PASS, so a pass's counters stay apart "
                         "from the other passes' and the system's. The SoC takes only some sets: 19,13,18,1,14 "
                         "returned instruction-cache misses, stalls, ALU use and clocks; 19,13,14,3,1 returned "
                         "clocks and ALU alone (2026-10-01). Check `Captured N metrics` in stages_<view>.txt")
    ap.add_argument("--out", help="output folder (default ../docs/bench/<date_time>)")
    ap.add_argument("--compare", help="an earlier run's folder, to compare baselines against")
    ap.add_argument("--stall-seconds", type=float, default=120.0, help="give up on a view after this long with no new window")
    ap.add_argument("--poll-seconds", type=float, default=3.0, help=argparse.SUPPRESS)
    ap.add_argument("--print-levers", metavar="VIEW", help="print the lever file for VIEW and exit (no device)")
    args = ap.parse_args(argv)
    # A run stopped from outside -- a task runner's SIGTERM -- must put the
    # headset back like Ctrl-C does. Python's default SIGTERM ends the process
    # WITHOUT running `finally`, and one such stop left the Quest in detailed
    # profiling mode with its clocks locked (2026-09-28).
    def terminated(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminated)

    try:
        extra = json.loads(args.levers) if args.levers else {}
        if not isinstance(extra, dict) or "bench" in extra or "ab_cycle" in extra:
            raise BenchError("--levers takes a JSON object of levers, without bench or ab_cycle")
        if args.trace and args.ab:
            # Detailed profiling mode is on for the whole run, and its overhead
            # would sit inside every A/B number.
            raise BenchError("--trace runs in the driver's detailed profiling mode; run --ab separately")
        if args.draws and (not args.trace or not re.fullmatch(r"\d+(,\d+)*", args.draws)):
            raise BenchError("--draws takes comma-separated metric ids, with --trace")
        if args.stage_metrics and (not args.trace or not re.fullmatch(r"\d+(,\d+)*", args.stage_metrics)):
            raise BenchError("--stage-metrics takes comma-separated metric ids, with --trace")
    except json.JSONDecodeError as e:
        print("bench: --levers is not JSON: %s" % e, file=sys.stderr)
        return 2
    except BenchError as e:
        print("bench: %s" % e, file=sys.stderr)
        return 2

    try:
        views = load_views(args.print_levers or args.views)
    except BenchError as e:
        print("bench: %s" % e, file=sys.stderr)
        return 2
    if args.print_levers:
        print(json.dumps(levers_for(views[0], args.ab, extra)))
        return 0

    compare = None
    if args.compare:
        compare = json.loads((Path(args.compare) / "report.json").read_text())

    try:
        dev = find_quest()
    except BenchError as e:
        print("bench: %s" % e, file=sys.stderr)
        return 1

    started = datetime.datetime.now()
    out = Path(args.out) if args.out else HERE.parent / "docs" / "bench" / started.strftime("%Y-%m-%d_%H%M")
    out.mkdir(parents=True, exist_ok=True)
    run = {
        "started": started.strftime("%Y-%m-%d %H:%M"),
        "device": dev.serial,
        "ab": args.ab,
        "passes": args.passes,
        "windows": args.windows,
        "extra_levers": args.levers,
        "cpu_level": "governor" if args.no_lock else args.cpu_level,
        "gpu_level": "governor" if args.no_lock else args.gpu_level,
        "views": [v["name"] for v in views],
        "trace_seconds": args.trace,
    }
    warnings: list = []
    collected: dict = {}
    # (what, how to undo it), in the order things were changed on the device.
    changed: list = []
    capture = None
    capture_file = None
    pipestats_capture = None
    pipestats_file = None
    vrapi_path = out / "vrapi.log"
    failure = None
    try:
        dev.shell("input keyevent KEYCODE_WAKEUP", check=False)
        dev.shell("am broadcast -a com.oculus.vrpowermanager.prox_close")
        changed.append(("give the proximity sensor back",
                        lambda: dev.shell("am broadcast -a com.oculus.vrpowermanager.automation_disable")))
        props = [(GUARDIAN_PAUSE_PROP, 1)]
        if not args.no_lock:
            props += [(CPU_LEVEL_PROP, args.cpu_level), (GPU_LEVEL_PROP, args.gpu_level)]
        for prop, value in props:
            set_prop(dev, changed, warnings, prop, value)
        dev.run("logcat", "-G", "16M", check=False)
        if args.trace:
            # The render-stage trace needs the driver's detailed profiling
            # mode, which an app only picks up when it STARTS -- so it is
            # restarted into it, and afterwards restarted out of it rather
            # than left running with the profiling overhead in the next run.
            # (Undone newest first: detailed mode off, then the stop.)
            changed.append(("stop the app profiled in detailed mode",
                            lambda: dev.shell("am force-stop " + PACKAGE)))
            dev.shell("ovrgpuprofiler -e " + PACKAGE)
            changed.append(("leave detailed profiling mode", lambda: dev.shell("ovrgpuprofiler -d")))
            dev.shell("am force-stop " + PACKAGE)
        if args.eye_capture:
            # The swapchain is made copyable only when the app STARTS with the
            # property set, so it is restarted with it -- and, once the
            # property is put back, stopped, so the next run is not measured
            # with a copyable swapchain. (Undone newest first.)
            changed.append(("stop the app started for eye captures", lambda: dev.shell("am force-stop " + PACKAGE)))
            set_prop(dev, changed, warnings, EYE_CAPTURE_PROP, 1)
            dev.shell("am force-stop " + PACKAGE)
        if args.pipestats:
            # The same restart, in and out, as the eye captures; the lines are
            # read back from the moment the app starts.
            changed.append(("stop the app started for pipeline statistics",
                            lambda: dev.shell("am force-stop " + PACKAGE)))
            set_prop(dev, changed, warnings, PIPESTATS_PROP, 1)
            dev.shell("am force-stop " + PACKAGE)
            # STREAMED from before the start, not read back at the end: the
            # app logs every pipeline in one burst, and by the end of a run the
            # device's shared log buffer had wrapped past the first ten
            # (2026-10-01).
            pipestats_file = open(out / "pipestats.log", "wb")
            pipestats_capture = subprocess.Popen(
                [ADB, "-s", dev.serial, "logcat", "-v", "time", "-T", "1", "-e", "PIPESTATS"],
                stdout=pipestats_file, stderr=subprocess.DEVNULL,
            )
        dev.shell("rm -f " + PERF)
        # Brought to the front whether or not it is running: a process in the
        # background renders nothing. `am start` resumes a running one rather
        # than restarting it.
        running = bool(dev.shell("pidof " + PACKAGE, check=False).strip())
        print("%s the app" % ("resuming" if running else "starting"), flush=True)
        dev.shell("am start -n " + ACTIVITY)
        capture_file = open(vrapi_path, "wb")
        capture = subprocess.Popen([ADB, "-s", dev.serial, "logcat", "-v", "time", "-T", "1", "VrApi:I", "*:S"],
                                   stdout=capture_file, stderr=subprocess.DEVNULL)
        counter_ids: list = []
        if not args.no_profile and not args.trace:
            metrics, listing = profiler_metrics(dev)
            (out / "profiler_metrics.txt").write_text(listing)
            counter_ids = [i for i, _ in pick_metrics(metrics)]
            if not counter_ids:
                warnings.append("ovrgpuprofiler listed no counters; GPU counters skipped (see profiler_metrics.txt)")
        for view in views:
            name = view["name"]
            print("view %s: %s" % (name, view.get("note", "")), flush=True)
            mark = vrapi_path.stat().st_size
            # The shipped renderer first -- its windows, then the GPU's own
            # counters while it keeps drawing the same frame.
            write_levers(dev, levers_for(view, False, extra))
            if not any(what == "remove the lever file" for what, _ in changed):
                changed.append(("remove the lever file", lambda: dev.shell("rm -f " + LEVERS)))
            records = wait_for_view(dev, name, False, args.passes, args.windows, args.stall_seconds, args.poll_seconds)
            # The same frame blocking on the GPU each frame, whose wait is then
            # its time -- what `--compare` and a run without --ab compare by.
            write_levers(dev, levers_for(view, False, extra, synced=True))
            records += wait_for_view(dev, name, False, args.passes, args.windows, args.stall_seconds, args.poll_seconds,
                                     synced=True)
            write_levers(dev, levers_for(view, False, extra))
            # What the view looks like, from the headset itself.
            shot = None
            if not args.no_screenshots:
                shot = screenshot(dev, out / name)
                if shot is None:
                    warnings.append("no screenshot came back for %s" % name)
            eyes = None
            if args.eye_capture:
                eyes = eye_capture(dev, levers_for(view, False, extra), len(collected) + 1, out / name)
                write_levers(dev, levers_for(view, False, extra))
                if eyes is None:
                    warnings.append("no eye capture came back for %s (is the app built with it?)" % name)
            if args.record:
                video = record_video(dev, args.record, out / (name + ".mp4"))
                if video is None:
                    warnings.append("no video came back for %s" % name)
            counters: dict = {}
            if counter_ids:
                counters, raw = profile(dev, counter_ids, args.profile_seconds)
                (out / ("profiler_%s.txt" % name)).write_text(raw)
            passes: list = []
            draws: list = []
            if args.trace:
                # Per-draw traces are FLAKY on the Quest (2026-09-28): one can
                # capture no draws at all, and the counters come back only for
                # some metric sets -- 1,14,18,19,28,36 returned every one where
                # 1,14,18,28,36 returned Clocks alone, in the same session. So
                # a trace that came back empty or Clocks-only is taken again.
                if args.draws:
                    for attempt in range(2):
                        draws, raw = trace_draws(dev, args.trace, args.draws)
                        (out / ("draws_%s.txt" % name)).write_text(raw)
                        if any(len(d["metrics"]) > 1 for d in draws):
                            break
                    if not draws:
                        warnings.append("the per-draw trace at %s held no draws (see draws_%s.txt)" % (name, name))
                    elif not any(len(d["metrics"]) > 1 for d in draws):
                        warnings.append("the per-draw trace at %s returned Clocks only, twice: try another metric "
                                        "set (1,14,18,19,28,36 is known to work)" % name)
                passes, raw = trace(dev, args.trace)
                (out / ("trace_%s.txt" % name)).write_text(raw)
                if args.stage_metrics:
                    # Kept raw: whoever asked for the per-stage metrics reads them.
                    raw = dev.shell("ovrgpuprofiler -t%d -s%s" % (args.trace, args.stage_metrics),
                                    check=False, timeout=args.trace + 90)
                    (out / ("stages_%s.txt" % name)).write_text(raw)
                if not passes:
                    warnings.append("the trace at %s held no render passes (see trace_%s.txt)" % (name, name))
            # Then the schedule, every phase `passes` times.
            if args.ab:
                write_levers(dev, levers_for(view, True, extra))
                records += wait_for_view(dev, name, True, args.passes, args.windows, args.stall_seconds, args.poll_seconds)
            capture_file.flush()
            with open(vrapi_path, "rb") as f:
                f.seek(mark)
                vrapi = summarize_vrapi(f.read().decode(errors="replace"))
            collected[name] = {"phases": summarize_view(records), "vrapi": vrapi, "gpu_counters": counters,
                               "trace": passes, "draws": draws, "records": len(records),
                               "screenshot": shot.name if shot else None, "eyes": eyes}
    except BenchError as e:
        failure = str(e)
    except KeyboardInterrupt:
        failure = "interrupted"
    finally:
        if pipestats_capture is not None:
            # Stopped before the restore stops the app: pipelines built late --
            # a lever's measurement variant -- log when they are built.
            pipestats_capture.terminate()
            try:
                pipestats_capture.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pipestats_capture.kill()
            pipestats_file.close()
            if not (out / "pipestats.log").read_bytes().strip():
                warnings.append("no PIPESTATS lines came back (does the driver have "
                                "VK_KHR_pipeline_executable_properties? see the app's `vulkan:` log)")
        restore(dev, changed, warnings)
        if capture is not None:
            capture.terminate()
            try:
                capture.wait(timeout=10)
            except subprocess.TimeoutExpired:
                capture.kill()
        if capture_file is not None:
            capture_file.close()
        try:
            dev.run("pull", PERF, str(out / "perf.jsonl"), check=False)
        except BenchError as e:
            warnings.append("could not pull the results file: %s" % e)

    report = render_report(run, collected, compare) if collected else "# Quest benchmark %s\n\nNo view finished.\n" % run["started"]
    if warnings or failure:
        report += "\n" + "\n".join("- " + w for w in ([("**FAILED:** " + failure)] if failure else []) + warnings) + "\n"
    (out / "report.md").write_text(report)
    (out / "report.json").write_text(json.dumps({"run": run, "views": collected, "failure": failure, "warnings": warnings}, indent=1))
    print(report)
    print("written to %s" % out)
    return 1 if failure else 0


if __name__ == "__main__":
    sys.exit(main())
