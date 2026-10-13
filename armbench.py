#!/usr/bin/env python3
"""Lever arms benched against each other in ONE app session: per view, the
arms in turn, `rounds` times, each `windows` synced windows (the GPU's wait
each frame -- bench.py's `synced`), so a difference between arms is the
arms', not a launch's or a deploy's.

    armbench.py <out dir> <view,view> '<[["label", {levers}, [marks]], ...]>' [rounds] [windows]

An arm's third element is the tokens its windows' lever summary must hold.
Boolean levers print as flags (`no_cave_twins`, `multiview=off`), so name
them; an arm with no token of its own (the shipped one) gives `[]` and is
told apart by holding none of the other arms' tokens.

A window counts for an arm only if the app logged it after that arm's lever
file was written, under those levers, and not as a warm-up (the app marks the
window that straddles a lever change). With `--drawlog`, each arm's first
visit to a view also saves that frame's draw log (`Levers::draw_log`) as
`drawlog_<view>_<arm>.txt`: every command buffer and draw with its
pipeline's label, to join to a per-draw trace.

The device setup and its undo are bench.py's own: proximity held, boundary
paused, CPU and GPU levels locked (4, 5), the lever file removed, everything
put back newest first. (Rewritten 2026-10-12: the scratch copy was lost to
the temp folder's clean-up.)
"""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402

STALL = 240.0


def wait_windows(dev, name: str, want: list, avoid: list, after: int, windows: int) -> list:
    since = time.monotonic()
    while True:
        mine = [r for r in bench.read_records(dev) if r.get("bench") == name]
        fresh = []
        for r in mine[after:]:
            tokens = (r.get("levers") or "").split(",")
            if r.get("warmup") or "gpu_sync" not in tokens:
                continue
            if all(m in tokens for m in want) and not any(m in tokens for m in avoid):
                fresh.append(r)
        if len(fresh) >= windows:
            return fresh[:windows]
        if time.monotonic() - since > STALL:
            block = bench.launch_block(dev)
            raise bench.BenchError("no window for %s under %s in %d s%s" % (name, want, STALL, "; " + block if block else ""))
        time.sleep(2.0)


def gpu_mhz(dev) -> str:
    """The GPU's clock now, in MHz, or "?" where the driver's file cannot be read."""
    try:
        text = dev.shell("cat /sys/class/kgsl/kgsl-3d0/gpuclk", check=False).strip()
        return str(int(text) // 1000000)
    except (ValueError, bench.BenchError):
        return "?"


def draw_log(dev, levers: dict, n: int, dest: Path) -> None:
    dev.run("logcat", "-c", check=False)
    bench.write_levers(dev, dict(levers, draw_log=n))
    deadline = time.monotonic() + 30
    text = ""
    while time.monotonic() < deadline:
        time.sleep(2.0)
        text = dev.run("logcat", "-d", "-s", "quest_app:W", check=False)
        if "DRAWLOG frame end" in text:
            break
    lines = [l.split("DRAWLOG ", 1)[1] for l in text.splitlines() if "DRAWLOG " in l]
    dest.write_text("\n".join(lines) + "\n")
    bench.write_levers(dev, levers)


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--drawlog"]
    want_log = "--drawlog" in sys.argv[1:]
    out = Path(args[0])
    out.mkdir(parents=True, exist_ok=True)
    views = bench.load_views(args[1])
    arms = json.loads(args[2])
    rounds = int(args[3]) if len(args) > 3 else 2
    windows = int(args[4]) if len(args) > 4 else 2
    dev = bench.find_quest()
    changed: list = []
    warnings: list = []
    results: dict = {}
    log_n = int(time.time()) % 100000
    try:
        dev.shell("input keyevent KEYCODE_WAKEUP", check=False)
        dev.shell("am broadcast -a com.oculus.vrpowermanager.prox_close")
        changed.append(("give the proximity sensor back",
                        lambda: dev.shell("am broadcast -a com.oculus.vrpowermanager.automation_disable")))
        for prop, value in [(bench.GUARDIAN_PAUSE_PROP, 1), (bench.CPU_LEVEL_PROP, 4), (bench.GPU_LEVEL_PROP, 5)]:
            bench.set_prop(dev, changed, warnings, prop, value)
        dev.run("logcat", "-G", "16M", check=False)
        dev.shell("am force-stop " + bench.PACKAGE)
        changed.append(("stop the app", lambda: dev.shell("am force-stop " + bench.PACKAGE)))
        dev.shell("rm -f " + bench.PERF, check=False)
        bench.write_levers(dev, bench.levers_for(views[0], False, arms[0][1], synced=True))
        changed.append(("remove the lever file", lambda: dev.shell("rm -f " + bench.LEVERS)))
        dev.shell("am start -n " + bench.ACTIVITY)
        with open(out / "windows.jsonl", "w") as log:
            for view in views:
                name = view["name"]
                for rnd in range(rounds):
                    for k, arm in enumerate(arms):
                        label, extra = arm[0], arm[1]
                        want = arm[2] if len(arm) > 2 else ["%s=%s" % kv for kv in extra.items()]
                        avoid = [m for j, other in enumerate(arms) if j != k
                                 for m in (other[2] if len(other) > 2 else []) if m not in want]
                        levers = bench.levers_for(view, False, extra, synced=True)
                        after = len([r for r in bench.read_records(dev) if r.get("bench") == name])
                        bench.write_levers(dev, levers)
                        got = wait_windows(dev, name, want, avoid, after, windows)
                        vals = [r["gpu_avg"] for r in got]
                        results.setdefault((name, label), []).extend(vals)
                        # The GPU's clock as the arm's last window ended: the
                        # governor moves between 599 and 640 MHz under load, 6%
                        # of a frame, which is more than most arms differ by.
                        clock = gpu_mhz(dev)
                        print("%-20s %-18s round %d: %s  [%s MHz]" % (name, label, rnd, ", ".join("%.2f" % v for v in vals), clock), flush=True)
                        for r in got:
                            log.write(json.dumps({"view": name, "arm": label, "round": rnd, "gpu_avg": r["gpu_avg"],
                                                  "gpu_mhz": clock, "levers": r.get("levers"), "pass": r.get("pass")}) + "\n")
                        if want_log and rnd == 0:
                            log_n += 1
                            draw_log(dev, levers, log_n, out / ("drawlog_%s_%s.txt" % (name, label)))
    except (bench.BenchError, KeyboardInterrupt) as e:
        print("armbench: %s" % e, file=sys.stderr)
        return 1
    finally:
        bench.restore(dev, changed, warnings)
        for w in warnings:
            print("warning:", w)
    print("\nmedian synced ms per arm (all windows, all rounds); the first arm is the reference")
    for view in views:
        name = view["name"]
        ref = statistics.median(results[(name, arms[0][0])])
        parts = []
        for arm in arms:
            v = results[(name, arm[0])]
            m = statistics.median(v)
            parts.append("%s %.2f (%+.2f, range %.2f)" % (arm[0], m, m - ref, max(v) - min(v)))
        print("%-20s %s" % (name, " | ".join(parts)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
