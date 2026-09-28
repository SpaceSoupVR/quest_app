#!/usr/bin/env python3
"""Measure what a frame costs on the Quest, unattended, from fixed viewpoints.

Plug the headset in, leave it on the desk, and run one of:

    python3 bench.py                  # every view, the shipped renderer
    python3 bench.py --ab             # every view, every perf_ab phase, 3 passes
    python3 bench.py --views hall_back,hallway --ab --passes 2
    python3 bench.py --levers '{"probe_trace": false}'     # a configuration
    python3 bench.py --compare ../docs/bench/2026-09-27_2210  # against a run

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
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
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
        if name not in by_name:
            raise BenchError("no view %r; bench_views.json has: %s" % (name, ", ".join(by_name)))
        if name not in [v["name"] for v in picked]:
            picked.append(by_name[name])
    return picked


def levers_for(view: dict, ab: bool, extra: dict) -> dict:
    """The lever file for one view: the extra levers, the pin, the schedule."""
    levers = dict(extra)
    levers["bench"] = {"name": view["name"], "eye": view["eye"], "at": view["at"]}
    if ab:
        levers["ab_cycle"] = True
    return levers


def write_levers(dev: Device, levers: dict) -> None:
    """Pushed to a temporary name and moved over the lever file, so the app
    never reads half a file."""
    fd, tmp = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(levers, separators=(",", ":")))
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
                  stall_seconds: float, poll_seconds: float) -> list:
    """The view's measured windows, once there are enough of them."""
    last_seen = -1
    last_change = time.monotonic()
    while True:
        records = read_records(dev)
        mine = [r for r in records if r.get("bench") == name]
        # The shipped windows and the schedule's are told apart by whether
        # the app was cycling, so neither step counts the other's windows.
        measured = sorted(
            (r for r in mine if not r.get("warmup") and (r.get("cycle_len", 1) > 1) == ab),
            key=lambda r: r.get("window", 0),
        )
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


def median(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def summarize_view(records: list) -> dict:
    """Per phase: the median over passes of each number, and its spread."""
    phases: dict = {}
    for r in records:
        phases.setdefault(r.get("phase", "-"), []).append(r)
    out = {}
    for phase, rs in phases.items():
        xr_gpu = [r.get("xr", {}).get("app/gpu_frametime") for r in rs]
        xr_vals = [v for v in xr_gpu if v is not None]
        passes = {}
        for label in sorted({k for r in rs for k in r.get("pass", {})}):
            passes[label] = median([r.get("pass", {}).get(label) for r in rs])
        out[phase] = {
            "windows": len(rs),
            "levers": rs[0].get("levers"),
            "app_gpu_ms": median(xr_gpu),
            "app_gpu_spread_ms": (max(xr_vals) - min(xr_vals)) if len(xr_vals) > 1 else 0.0,
            "gpu_wait_ms": median([r.get("gpu_avg") for r in rs]),
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
    lines.append("`app GPU` is the runtime's own GPU time for the app's frame (XR_META_performance_metrics), "
                 "averaged over each window and the median over passes; `cost` is what switching the "
                 "feature off saves against the baseline -- negative for an optimisation, whose lever "
                 "switches it off. `shipped` is the renderer with no schedule running, and should agree "
                 "with `baseline`. `spread` is the range over passes. `wait` is the render thread's wait "
                 "for the GPU.")
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
        base = phases.get("baseline") or phases.get("-")
        if v.get("screenshot"):
            lines.append("![%s](%s)" % (name, v["screenshot"]))
            lines.append("")
        lines.append("| phase | app GPU ms | cost ms | spread | wait ms | frame ms | fps | CPU ms |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for phase, p in phases.items():
            cost = None
            if base and p is not base and base["app_gpu_ms"] is not None and p["app_gpu_ms"] is not None:
                cost = base["app_gpu_ms"] - p["app_gpu_ms"]
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                "shipped" if phase == "-" else phase, fmt(p["app_gpu_ms"]), fmt(cost), fmt(p["app_gpu_spread_ms"]), fmt(p["gpu_wait_ms"]),
                fmt(p["frame_ms"]), fmt(p["fps"], 1), fmt(p["cpu_ms"])))
        if base and base["passes_ms"]:
            lines.append("")
            lines.append("Passes (baseline, one frame each): " + ", ".join(
                "%s %s" % (k, fmt(ms)) for k, ms in base["passes_ms"].items() if ms))
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
            old_base = old.get("baseline") or old.get("-")
            if old_base and base and old_base.get("app_gpu_ms") is not None and base["app_gpu_ms"] is not None:
                d = base["app_gpu_ms"] - old_base["app_gpu_ms"]
                lines.append("")
                lines.append("Against %s: app GPU %s -> %s ms (%+.2f ms, %s)." % (
                    compare["run"]["started"], fmt(old_base["app_gpu_ms"]), fmt(base["app_gpu_ms"]), d,
                    "slower" if d > 0 else "faster"))
    return "\n".join(lines) + "\n"


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
    ap.add_argument("--profile-seconds", type=int, default=10, help="how long to read the GPU counters a view (default 10)")
    ap.add_argument("--out", help="output folder (default ../docs/bench/<date_time>)")
    ap.add_argument("--compare", help="an earlier run's folder, to compare baselines against")
    ap.add_argument("--stall-seconds", type=float, default=120.0, help="give up on a view after this long with no new window")
    ap.add_argument("--poll-seconds", type=float, default=3.0, help=argparse.SUPPRESS)
    ap.add_argument("--print-levers", metavar="VIEW", help="print the lever file for VIEW and exit (no device)")
    args = ap.parse_args(argv)

    try:
        extra = json.loads(args.levers) if args.levers else {}
        if not isinstance(extra, dict) or "bench" in extra or "ab_cycle" in extra:
            raise BenchError("--levers takes a JSON object of levers, without bench or ab_cycle")
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
    }
    warnings: list = []
    collected: dict = {}
    # (what, how to undo it), in the order things were changed on the device.
    changed: list = []
    capture = None
    capture_file = None
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
            before = dev.shell("getprop " + prop).strip()
            dev.shell("setprop %s %d" % (prop, value))
            changed.append(("restore %s to %r" % (prop, before),
                            lambda prop=prop, before=before: dev.shell("setprop %s '%s'" % (prop, before))))
        dev.run("logcat", "-G", "16M", check=False)
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
        if not args.no_profile:
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
            # What the view looks like, from the headset itself.
            shot = None
            if not args.no_screenshots:
                shot = screenshot(dev, out / name)
                if shot is None:
                    warnings.append("no screenshot came back for %s" % name)
            counters: dict = {}
            if counter_ids:
                counters, raw = profile(dev, counter_ids, args.profile_seconds)
                (out / ("profiler_%s.txt" % name)).write_text(raw)
            # Then the schedule, every phase `passes` times.
            if args.ab:
                write_levers(dev, levers_for(view, True, extra))
                records += wait_for_view(dev, name, True, args.passes, args.windows, args.stall_seconds, args.poll_seconds)
            capture_file.flush()
            with open(vrapi_path, "rb") as f:
                f.seek(mark)
                vrapi = summarize_vrapi(f.read().decode(errors="replace"))
            collected[name] = {"phases": summarize_view(records), "vrapi": vrapi, "gpu_counters": counters,
                               "records": len(records), "screenshot": shot.name if shot else None}
    except BenchError as e:
        failure = str(e)
    except KeyboardInterrupt:
        failure = "interrupted"
    finally:
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
