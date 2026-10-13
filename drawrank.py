#!/usr/bin/env python3
"""Per-draw traces joined to draw logs and PIPESTATS: what each PIPELINE costs
in each bench view, with the statistics of the code the driver compiled for
it.

    drawrank.py <dir> [--md out.md] [--top N] [--traces 'trace_*']

`<dir>` holds what `armbench.py --drawlog` and `bench.py --trace --draws`
wrote (docs/bench/shader_opt_1012 is the first):

    arms/drawlog_<view>_shipped.txt   every command buffer and draw of one frame, by pipeline label
    arms.log                          the synced GPU ms per view (the `shipped` arm's median)
    trace_*/draws_<view>.txt          ovrgpuprofiler's per-draw metrics for one frame
    pipestats/pipestats.log           PIPESTATS lines

THE JOIN. The profiler has no names: a draw is its command buffer and its
number in it. The draw log has the names in the same order. Command buffers
are paired in order by their draw counts (the log's encoders that hold only
compute dispatches can merge or split in the trace, so those are paired
loosely and not attributed); a render encoder's draws pair one to one.

THE MILLISECONDS ARE AN ESTIMATE: the capture's clocks turned into time by
the passes the app times itself (`pass` in arms/windows.jsonl: the reflection
pass and the scene pass of each eye, the characters' cards) -- the timed
passes' milliseconds over their clocks, applied to every draw. Clocks compare
within one capture only, and a capture is whatever run of command buffers the
profiler caught: often one eye, which is then doubled.

BOUND BY (per pipeline, from the trace and the statistics):
  code    % Instruction Cache Miss >= 0.03: past the instruction cache (3,387
          instructions); every instruction costs two to three times as much.
  regs    occupancy <= 50% from the register footprint: half or fewer of the
          waves in flight; latency (texture reads, uniform loads) is not hidden.
  alu     % Shader ALU Capacity Utilized >= 40.
  geometry  no fragments shaded, or over 40 clocks a fragment: its clocks are
          not its shaded pixels' -- vertex work, the draw's own set-up, or
          rasterising depth alone (the prepass, a model wholly hidden).
  fill    none of those, and the clocks follow the fragments: the cost is how
          many pixels it shades (overdraw, or simply a large surface).
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipestats_table  # noqa: E402

HEAD = re.compile(r"Frame\s+(\d+)\s*:\s*Draw\s+(\d+)")
METRIC = re.compile(r"^\s*([A-Za-z%/()][^:]*?)\s*:\s*(-?[\d.]+(?:[eE][-+]?\d+)?)\s*$")
LRZ = re.compile(r"LRZ State:\s*(.*?)\s*<")


def read_trace(path: Path) -> list:
    """The first traced frame as command buffers, each a list of draws
    (dicts of metric -> value, plus `lrz`)."""
    cbs: list = []
    frame = None
    last = None
    cur = None
    for line in path.read_text(errors="replace").splitlines():
        h = HEAD.search(line)
        if h:
            f, n = int(h.group(1)), int(h.group(2))
            if frame is None:
                frame = f
            if f != frame:
                break
            if last is None or n <= last:
                cbs.append([])
            last = n
            cur = {}
            cbs[-1].append(cur)
            continue
        if cur is None:
            continue
        z = LRZ.search(line)
        if z:
            cur["lrz"] = z.group(1)
            continue
        m = METRIC.match(line)
        if m:
            cur[m.group(1)] = float(m.group(2))
    return cbs


def read_drawlog(path: Path) -> list:
    """Encoders with something in them: (label, [(kind, pipeline, count)])."""
    secs: list = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("begin "):
            secs.append((line[6:], []))
        elif line.startswith(("draw ", "dispatch ")) and secs:
            parts = line.split()
            kind, label = parts[0], " ".join(parts[2:-1])
            secs[-1][1].append((kind, label, int(parts[-1][2:])))
    return [s for s in secs if s[1]]


def pair(cbs: list, secs: list) -> tuple:
    """Each trace command buffer with the log encoder it is: [(cb index,
    encoder index or None)], and how many draws went unpaired.

    A capture is some run of consecutive command buffers -- it may start
    anywhere in the frame and hold one eye or less -- so the log's encoders
    are tried from every starting point (wrapping into the next frame), render
    encoders pairing by an equal draw count in order, and the start that pairs
    the most draws wins. Encoders holding only compute dispatches merge or
    split in the trace, so those pair loosely and are not attributed."""
    best = ([(i, None) for i in range(len(cbs))], sum(len(cb) for cb in cbs), -1)
    ring = secs + secs
    for start in range(len(secs)):
        out = []
        at = start
        got = 0
        for i, cb in enumerate(cbs):
            found = None
            for j in range(at, min(at + 5, len(ring))):
                draws = ring[j][1]
                if len(draws) == len(cb) and draws[0][0] == "draw":
                    found = j
                    break
            if found is None:
                out.append((i, None))
                continue
            out.append((i, found % len(secs)))
            got += len(cb)
            at = found + 1
        if got > best[2]:
            best = (out, sum(len(cb) for cb in cbs) - got, got)
    return best[0], best[1]


def synced_ms(arms_log: Path) -> dict:
    ms = {}
    for line in arms_log.read_text().splitlines():
        m = re.match(r"(\S+)\s+shipped (\d+\.\d+) \(", line)
        if m:
            ms[m.group(1)] = float(m.group(2))
    return ms


# The app's own GPU timers by encoder label, the left eye's first.
TIMERS = {"probe_pass": ("probe_l", "probe_r"), "ssr_scene_pass": ("scene_l", "scene_r"), "character_cards": ("cards", "cards")}


def pass_ms(windows: Path) -> dict:
    """view -> {timer: median ms over the shipped arm's windows}."""
    import json
    import statistics
    seen: dict = {}
    for line in windows.read_text().splitlines():
        r = json.loads(line)
        if r.get("arm") != "shipped" or not r.get("pass"):
            continue
        for k, v in r["pass"].items():
            seen.setdefault(r["view"], {}).setdefault(k, []).append(v)
    return {view: {k: statistics.median(v) for k, v in timers.items()} for view, timers in seen.items()}


def bound_by(row: dict, stats) -> str:
    if row["frags"] <= 0 or row["clocks_frag"] > 40:
        # Its clocks are not its shaded pixels': vertex work, what a draw
        # costs to set up, or depth alone.
        return "geometry"
    marks = []
    if row["icache"] >= 0.03:
        marks.append("code")
    if stats and 0 <= stats["occ"] <= 50:
        marks.append("regs")
    if row["alu_cap"] >= 40:
        marks.append("alu")
    return "+".join(marks) or "fill"


def view_rows(root: Path, view: str, stats: dict, timers: dict, traces_glob: str = "trace_*") -> tuple:
    """The view's pipelines, heaviest first, with `ms` for a whole frame
    (both eyes); and a note on what the capture held."""
    traces = sorted(root.glob("%s/draws_%s.txt" % (traces_glob, view)))
    log = root / "arms" / ("drawlog_%s_shipped.txt" % view)
    if not traces or not log.exists():
        return [], "no trace"
    secs = read_drawlog(log)
    # The capture that pairs the most draws.
    best = None
    for t in traces:
        cbs = read_trace(t)
        pairs, lost = pair(cbs, secs)
        got = sum(len(cb) for cb in cbs) - lost
        if best is None or got > best[0]:
            best = (got, cbs, pairs, lost)
    _, cbs, pairs, lost = best
    # Which eye an encoder is: its place among the log's encoders of its label.
    nth: dict = {}
    eye_of = {}
    for j, (label, _) in enumerate(secs):
        eye_of[j] = nth.get(label, 0)
        nth[label] = eye_of[j] + 1
    # Milliseconds a clock, from the passes the app times that the capture holds whole.
    timed_ms = 0.0
    timed_clocks = 0.0
    held: dict = {}
    for i, j in pairs:
        if j is None:
            continue
        label = secs[j][0]
        held.setdefault(label, set()).add(eye_of[j])
        if label in TIMERS and view in timers:
            timed_ms += timers[view].get(TIMERS[label][min(eye_of[j], 1)], 0.0)
            timed_clocks += sum(d.get("Clocks", 0.0) for d in cbs[i])
    per_clock = timed_ms / timed_clocks if timed_clocks else 0.0
    rows: dict = {}
    for i, j in pairs:
        if j is None:
            continue
        enc = secs[j]
        # An eye the capture missed is counted from the one it held.
        twice = 2.0 if nth.get(enc[0], 1) == 2 and len(held[enc[0]]) == 1 else 1.0
        for d, (kind, label, count) in zip(cbs[i], enc[1]):
            key = (enc[0], label)
            r = rows.setdefault(key, {"pass": enc[0], "label": label, "draws": 0, "clocks": 0.0, "frags": 0.0, "alu": 0.0,
                                      "ic": 0.0, "cap": 0.0, "occ": 0.0, "lrz": set(), "extra": {}, "ms": 0.0})
            c = d.get("Clocks", 0.0)
            f = d.get("Fragments Shaded", 0.0)
            r["draws"] += 1
            r["clocks"] += c
            r["ms"] += c * per_clock * twice
            r["frags"] += f
            r["alu"] += d.get("ALU / Fragment", 0.0) * f
            r["ic"] += d.get("% Instruction Cache Miss", 0.0) * c
            r["cap"] += d.get("% Shader ALU Capacity Utilized", 0.0) * c
            r["occ"] += d.get("% Wave Context Occupancy", 0.0) * c
            r["lrz"].add(d.get("lrz", "?"))
            for k, v in d.items():
                if k not in ("lrz", "Clocks", "Fragments Shaded", "ALU / Fragment", "% Instruction Cache Miss",
                             "% Shader ALU Capacity Utilized", "% Wave Context Occupancy"):
                    w = f if "/ Fragment" in k or "Per Pixel" in k else c
                    e = r["extra"].setdefault(k, [0.0, 0.0])
                    e[0] += v * w
                    e[1] += w
    out = []
    for r in rows.values():
        c, f = max(r["clocks"], 1e-9), max(r["frags"], 1e-9)
        row = {"pass": r["pass"], "label": r["label"], "draws": r["draws"], "clocks": r["clocks"], "ms": r["ms"],
               "frags": r["frags"], "clocks_frag": r["clocks"] / f if r["frags"] else 0.0, "alu_frag": r["alu"] / f,
               "icache": r["ic"] / c, "alu_cap": r["cap"] / c, "occupancy": r["occ"] / c, "lrz": ",".join(sorted(r["lrz"])),
               "extra": {k: (v[0] / v[1] if v[1] else 0.0) for k, v in r["extra"].items()}}
        row["stats"] = stats.get(r["label"])
        row["bound"] = bound_by(row, row["stats"])
        out.append(row)
    out.sort(key=lambda r: -r["ms"])
    eyes = len(held.get("ssr_scene_pass", ()))
    note = "the capture held %s; %d draws unpaired" % (
        {0: "no scene pass", 1: "one eye's scene pass (doubled)", 2: "both eyes"}[eyes], lost)
    return (out if eyes else []), note


def main() -> int:
    args = [a for a in sys.argv[1:]]
    md = None
    top = 8
    if "--md" in args:
        md = Path(args[args.index("--md") + 1])
        del args[args.index("--md"):args.index("--md") + 2]
    if "--top" in args:
        top = int(args[args.index("--top") + 1])
        del args[args.index("--top"):args.index("--top") + 2]
    traces_glob = "trace_*"
    if "--traces" in args:
        traces_glob = args[args.index("--traces") + 1]
        del args[args.index("--traces"):args.index("--traces") + 2]
    root = Path(args[0])
    # The fragment stage's statistics by label (the driver names it "Mesh"),
    # the last one logged for a label.
    stats = {}
    for r in pipestats_table.parse(str(root / "pipestats" / "pipestats.log")):
        if r["stage"] in ("Mesh", "Fragment"):
            stats[r["label"]] = r
    ms = synced_ms(root / "arms.log")
    timers = pass_ms(root / "arms" / "windows.jsonl")
    views = [v for v in ms if (root / "arms" / ("drawlog_%s_shipped.txt" % v)).exists()]
    lines = []
    totals: dict = {}
    counted = 0
    extra_names: list = []
    for view in views:
        rows, note = view_rows(root, view, stats, timers, traces_glob)
        if not rows:
            lines.append("\n### %s -- %.2f ms synced; %s\n" % (view, ms[view], note))
            continue
        counted += 1
        t = timers.get(view, {})
        lines.append("\n### %s -- %.2f ms synced (reflection pass %.2f + %.2f, scene pass %.2f + %.2f; %s)\n" % (
            view, ms[view], t.get("probe_l", 0), t.get("probe_r", 0), t.get("scene_l", 0), t.get("scene_r", 0), note))
        for r in rows:
            for k in r["extra"]:
                if k not in extra_names:
                    extra_names.append(k)
        lines.append("| pass | pipeline | draws | est. ms | fragments | clocks/frag | ALU/frag | I$ miss %% | ALU cap %% | occupancy %% |%s instr | alu32 | alu16 | special | tex | regs full+half | occ %% | bound by |" % (
            "".join(" %s |" % k for k in extra_names)))
        lines.append("|---|---|---|---|---|---|---|---|---|---|" + "---|" * (len(extra_names) + 8))
        for r in rows[:top]:
            s = r["stats"]
            st = ("%d | %d | %d | %d | %d | %d+%d | %d" % (s["instr"], s["alu32"], s["alu16"], s["cx"], s["tex"], s["full"], s["half"], s["occ"])) if s else "- | - | - | - | - | - | -"
            lines.append("| %s | %s | %d | %.2f | %.0f | %.2f | %.0f | %.3f | %.0f | %.0f |%s %s | %s |" % (
                r["pass"].replace("ssr_scene_pass", "scene"), r["label"], r["draws"], r["ms"], r["frags"],
                r["clocks_frag"], r["alu_frag"], r["icache"], r["alu_cap"], r["occupancy"],
                "".join(" %.2f |" % r["extra"].get(k, 0.0) for k in extra_names), st, r["bound"]))
        for r in rows:
            t = totals.setdefault(r["label"], {"ms": 0.0, "views": 0, "worst": (0.0, ""), "bound": {}, "stats": r["stats"], "cf": 0.0, "fr": 0.0})
            t["ms"] += r["ms"]
            t["views"] += 1 if r["ms"] >= 0.05 else 0
            if r["ms"] > t["worst"][0]:
                t["worst"] = (r["ms"], view)
            t["bound"][r["bound"]] = t["bound"].get(r["bound"], 0.0) + r["ms"]
            t["cf"] += r["clocks"]
            t["fr"] += r["frags"]
    head = ["## Pipelines by estimated GPU time, summed over the %d views with a usable capture\n" % counted,
            "| pipeline | sum est. ms | views >= 0.05 ms | worst view (ms) | clocks/frag | instr | alu32 | alu16 | special | tex | regs full+half | occ % | bound by (ms) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, t in sorted(totals.items(), key=lambda kv: -kv[1]["ms"])[:40]:
        s = t["stats"]
        st = ("%d | %d | %d | %d | %d | %d+%d | %d" % (s["instr"], s["alu32"], s["alu16"], s["cx"], s["tex"], s["full"], s["half"], s["occ"])) if s else "- | - | - | - | - | - | -"
        bound = ", ".join("%s %.1f" % kv for kv in sorted(t["bound"].items(), key=lambda kv: -kv[1]))
        head.append("| %s | %.2f | %d | %s (%.2f) | %.2f | %s | %s |" % (
            label, t["ms"], t["views"], t["worst"][1], t["worst"][0], t["cf"] / max(t["fr"], 1.0), st, bound))
    text = "\n".join(head + ["\n## By view (the %d heaviest pipelines each)" % top] + lines) + "\n"
    if md:
        md.write_text(text)
        print("written to %s" % md)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
