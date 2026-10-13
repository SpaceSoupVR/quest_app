#!/usr/bin/env python3
"""PIPESTATS lines (a bench.py --pipestats run's pipestats.log, or logcat) as
one row a pipeline stage: the compiled statistics the Adreno driver reports.

    pipestats_table.py <pipestats.log> [substring ...]      # rows whose label holds any substring

Columns: instr = Instruction count all; alu32 / alu16 = ALU instruction
counts; cx = Complex instruction count (special functions: rcp, rsq, sqrt,
sin, cos, log2, exp2); tex = texture reads; flow = flow control; pre = the
PREAMBLE's instructions (code the driver runs once a draw, hoisted from the
shader by itself: uniform-only expressions), with its ALU and complex counts;
regs = full + half registers = overall; occ = ALU fiber occupancy; mem =
memory reads + writes (spilled arrays).
"""
import re
import sys

FIELDS = [
    ("instr", "Instruction count all"), ("alu32", "ALU instruction count 32bit"), ("alu16", "ALU instruction count 16bit"),
    ("cx", "Complex instruction count"), ("tex", "Texture read instruction count"), ("flow", "Flow control instruction count"),
    ("pre", "Preamble instruction count all"), ("pre32", "Preamble ALU instruction count 32bit"),
    ("precx", "Preamble complex instruction count"), ("pretex", "Preamble texture read instruction count"),
    ("full", "Full precision register footprint per shader instance"), ("half", "Half precision register footprint per shader instance"),
    ("regs", "Overall register footprint per shader instance"), ("occ", "ALU fiber occupancy percentage"),
    ("memr", "Memory read instruction count"), ("memw", "Memory write instruction count"),
]


def parse(path: str) -> list:
    rows = []
    for line in open(path, errors="replace"):
        if "PIPESTATS " not in line:
            continue
        body = line.split("PIPESTATS ", 1)[1]
        m = re.match(r"(.*?) <(\w+) Shader>", body)
        if not m:
            continue
        stats = dict((k, int(v)) for k, v in re.findall(r"\[([^\]]+)\]=(-?\d+)", body))
        row = {"label": m.group(1).strip(), "stage": m.group(2)}
        for short, name in FIELDS:
            row[short] = stats.get(name, -1)
        rows.append(row)
    return rows


def main() -> int:
    rows = parse(sys.argv[1])
    wanted = sys.argv[2:]
    seen = set()
    print("%-52s %-4s %5s %5s %5s %4s %4s %4s | %4s %4s %3s %3s | %2s+%2s=%2s %3s%% %s" % (
        "pipeline", "stg", "instr", "alu32", "alu16", "cx", "tex", "flow", "pre", "p32", "pcx", "ptx", "f", "h", "rg", "occ", "mem"))
    for r in rows:
        if wanted and not any(w in r["label"] for w in wanted):
            continue
        key = (r["label"], r["stage"], r["instr"], r["regs"])
        if key in seen:
            continue
        seen.add(key)
        print("%-52s %-4s %5d %5d %5d %4d %4d %4d | %4d %4d %3d %3d | %2d+%2d=%2d %3d%% %d" % (
            r["label"][:52], r["stage"][:4], r["instr"], r["alu32"], r["alu16"], r["cx"], r["tex"], r["flow"],
            r["pre"], r["pre32"], r["precx"], r["pretex"], r["full"], r["half"], r["regs"], r["occ"], r["memr"] + r["memw"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
