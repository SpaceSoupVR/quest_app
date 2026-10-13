#!/usr/bin/env python3
"""Per-draw GPU captures of bench views, several a view, for `drawrank.py`.

    tracedraws.py <out dir> <view,view> [captures] [metric ids] [prefix]

For each view the app is started fresh at it (a session that has been traced
stops picking up lever files after a view or two), one synced window is waited
for so the view has settled, and `captures` (default 3) per-draw traces are
taken with `ovrgpuprofiler -t2 -x<ids>` and saved as
`<out dir>/<prefix><k>/draws_<view>.txt` (prefix `trace_c` by default).

WHY SEVERAL. A capture is whatever run of command buffers the profiler
caught -- sometimes both eyes, often one, sometimes only the shadow maps -- and
some come back with Clocks alone. `drawrank.py` takes, per view, the capture
that pairs the most draws with the draw log.

The metric ids are `ovrgpuprofiler -x -m`'s; the default set is the one known
to come back whole (bench.py). SpaceWarp is on, as it ships.

The device setup and its undo are bench.py's own (see armbench.py).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
from armbench import wait_windows  # noqa: E402

DEFAULT_IDS = "1,14,18,19,28,36"


def main() -> int:
    out = Path(sys.argv[1])
    views = bench.load_views(sys.argv[2])
    captures = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    ids = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_IDS
    prefix = sys.argv[5] if len(sys.argv) > 5 else "trace_c"
    dev = bench.find_quest()
    changed: list = []
    warnings: list = []
    extra = {"space_warp": True}
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
        changed.append(("remove the lever file", lambda: dev.shell("rm -f " + bench.LEVERS)))
        dev.shell("ovrgpuprofiler -e " + bench.PACKAGE)
        changed.append(("leave detailed profiling mode", lambda: dev.shell("ovrgpuprofiler -d")))
        for view in views:
            name = view["name"]
            dev.shell("am force-stop " + bench.PACKAGE)
            dev.shell("rm -f " + bench.PERF, check=False)
            bench.write_levers(dev, bench.levers_for(view, False, extra, synced=True))
            dev.shell("am start -n " + bench.ACTIVITY)
            wait_windows(dev, name, [], [], 0, 1)
            # Unsynced while tracing, as bench.py traces: the sync's wait is not the frame's.
            bench.write_levers(dev, bench.levers_for(view, False, extra))
            time.sleep(4.0)
            for k in range(captures):
                text = dev.shell("ovrgpuprofiler -t2 -x%s" % ids, check=False, timeout=140)
                dest = out / ("%s%d" % (prefix, k))
                dest.mkdir(parents=True, exist_ok=True)
                (dest / ("draws_%s.txt" % name)).write_text(text)
                draws = text.count(": Draw ")
                whole = text.count("ALU / Fragment") if "36" in ids.split(",") else draws
                print("%-20s capture %d: %d draws%s" % (name, k, draws, "" if whole else " (Clocks only)"), flush=True)
    except (bench.BenchError, KeyboardInterrupt) as e:
        print("tracedraws: %s" % e, file=sys.stderr)
        return 1
    finally:
        bench.restore(dev, changed, warnings)
        for w in warnings:
            print("warning:", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
