#!/usr/bin/env python3
"""bench.py against a fake adb: its branching, without the headset.

    python3 bench_test.py

The fake adb stands in for the device: it answers `devices -l`, keeps system
properties and a few files, and plays the app -- writing results windows for
whatever view the lever file pins. Every call is logged, so the tests assert
what bench.py did to the device, not only what it printed. This proves the
script's decisions; it does not prove the headset measures anything.
"""

import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUEST = "2G0YC5ZG7706YV"
PHONE = "ZA223HJX4F"
PHASES = ["baseline", "half_viewport", "direct_path", "no_probes", "no_shadows", "no_sun_dynamic",
          "no_probe_blend", "no_portals", "no_direct_lights", "no_probe_trace", "no_proxies",
          "no_stationary", "no_light_culling", "full_res_reflections"]
# What each phase saves, in the fake app: the report must recover these.
COST = {p: 0.5 * i for i, p in enumerate(PHASES)}

FAKE_ADB = textwrap.dedent('''\
    #!/usr/bin/env python3
    import json, os, shlex, shutil, sys, time
    state = os.environ["FAKE_ADB_STATE"]
    args = sys.argv[1:]
    with open(os.path.join(state, "calls.jsonl"), "a") as f:
        f.write(json.dumps(args) + "\\n")
    devices = open(os.path.join(state, "devices.txt")).read()
    if args[:2] == ["devices", "-l"]:
        print(devices, end="")
        sys.exit(0)
    if args[0] != "-s":
        sys.exit("fake adb: no -s")
    serial, args = args[1], args[2:]
    if serial not in devices:
        sys.exit("error: device '%%s' not found" %% serial)
    fs = os.path.join(state, "fs")
    local = lambda p: os.path.join(fs, p.replace("/", "_"))
    props_path = os.path.join(state, "props.json")
    props = json.load(open(props_path))
    PHASES = %(phases)s
    COST = %(cost)s

    def play_the_app():
        """Write a few more results windows for whatever the lever file says."""
        if open(os.path.join(state, "mode")).read().strip() in ("silent", "blocked"):
            return
        app_path = os.path.join(state, "app.json")
        app = json.load(open(app_path)) if os.path.exists(app_path) else {"levers": None, "window": 0, "pos": 0, "warm": True}
        lp = local("/sdcard/Android/data/com.example.questapp/files/levers.json")
        levers = json.load(open(lp)) if os.path.exists(lp) else {}
        if levers != app["levers"]:
            app.update(levers=levers, pos=0, warm=True)
        ab = levers.get("ab_cycle", False)
        bench = (levers.get("bench") or {}).get("name")
        with open(local("/sdcard/Android/data/com.example.questapp/files/perf.jsonl"), "a") as f:
            for _ in range(5):
                phase = "warmup" if app["warm"] else (PHASES[app["pos"] %% len(PHASES)] if ab else "-")
                gpu = 40.0 - COST.get(phase, 0.0)
                f.write(json.dumps({
                    "window": app["window"], "t": app["window"] * 3.0, "phase": phase,
                    "cycle_pass": app["pos"] // len(PHASES) if ab else app["pos"],
                    "cycle_len": len(PHASES) if ab else 1, "warmup": app["warm"],
                    "levers": "bench=%%s" %% bench, "bench": bench, "ssr": False, "multiview": False,
                    "frames": 112, "cpu_avg": 4.0, "cpu_max": 5.0, "gpu_avg": gpu + 1.0, "gpu_max": gpu + 3.0,
                    "frame_ms": 41.7, "fps": 24.0, "pass": {"scene_l": 20.0, "probe_l": 3.0},
                    "xr": {"app/gpu_frametime": gpu},
                }) + "\\n")
                app["window"] += 1
                if app["warm"]:
                    app["warm"] = False
                else:
                    app["pos"] += 1
        json.dump(app, open(app_path, "w"))

    if args[0] == "shell":
        words = shlex.split(args[1])
        mode = open(os.path.join(state, "mode")).read().strip()
        if words[:2] == ["ovrgpuprofiler", "-m"]:
            print("1       Clocks / Second\\n2       GPU %% Bus Busy\\n3       %% Vertex Fetch Stall\\n"
                  "4       %% Texture Fetch Stall\\n5       Preemptions / second")
            sys.exit(0)
        if words[0] == "timeout" and words[2] == "ovrgpuprofiler":
            for second in range(3):
                print("%%%% Texture Fetch Stall                      :           %%.3f" %% (2.0 + second))
                print("GPU %% Bus Busy                             :          40.000")
            sys.exit(0)
        if words[0] == "getprop":
            print(props.get(words[1], ""))
        elif words[0] == "setprop":
            props[words[1]] = words[2] if len(words) > 2 else ""
            json.dump(props, open(props_path, "w"))
        elif words[0] == "rm":
            p = local(words[-1])
            if os.path.exists(p):
                os.remove(p)
        elif words[0] == "mv":
            os.replace(local(words[1]), local(words[2]))
        elif words[0] == "pidof":
            print("4242")
        elif words[:2] == ["am", "startservice"] and "TAKE_SCREENSHOT" in words:
            n = len([f for f in os.listdir(fs) if f.startswith("_sdcard_Oculus_Screenshots_")])
            with open(local("/sdcard/Oculus/Screenshots/com.example.questapp-%%d.jpg" %% n), "wb") as f:
                f.write(b"fake jpeg")
        elif words[0] == "ls":
            prefix = next(w for w in words if w.startswith("/")).replace("/", "_") + "_"
            for f in sorted(os.listdir(fs)):
                if f.startswith(prefix):
                    print(f[len(prefix):])
        elif words[:2] == ["stat", "-c"]:
            p = local(words[-1])
            if os.path.exists(p):
                print(os.path.getsize(p))
        elif words[0] == "cat":
            if words[1].endswith("perf.jsonl"):
                play_the_app()
            p = local(words[1])
            if os.path.exists(p):
                print(open(p).read(), end="")
        sys.exit(0)
    if args[0] == "push":
        shutil.copy(args[1], local(args[2]))
        sys.exit(0)
    if args[0] == "pull":
        p = local(args[1])
        if not os.path.exists(p):
            sys.exit("remote object does not exist")
        shutil.copy(p, args[2])
        sys.exit(0)
    if args[0] == "logcat":
        if "-G" in args:
            sys.exit(0)
        if "-d" in args:
            if open(os.path.join(state, "mode")).read().strip() == "blocked":
                print("09-27 23:10:10.712  3168  7537 D CaseDialogAnalytics: logDialogShown: "
                      "dialogId=common_system_dialog_app_launch_blocked_controller_required, "
                      "packageName=com.example.questapp")
            sys.exit(0)
        # Once a second on the headset; faster here, so every view gets some.
        for _ in range(3000):
            print("09-27 22:00:00.000 I/VrApi   ( 999): FPS=24/72,Prd=45ms,CPU4/GPU=4/5,1920/599MHz,Temp=33.5C/0.0C,App=39.00ms", flush=True)
            time.sleep(0.02)
    sys.exit(0)
''') % {"phases": repr(PHASES), "cost": repr(COST)}

QUEST_LINE = QUEST + "          device usb:1-1 product:eureka model:Quest_3 device:eureka transport_id:3"
PHONE_LINE = PHONE + "          device usb:1-2 product:rhode model:moto_g_52 device:rhode transport_id:4"


class BenchScript(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name)
        (self.state / "fs").mkdir()
        bindir = self.state / "bin"
        bindir.mkdir()
        adb = bindir / "adb"
        adb.write_text(FAKE_ADB)
        adb.chmod(adb.stat().st_mode | stat.S_IEXEC)
        self.env = dict(os.environ, FAKE_ADB_STATE=str(self.state), PATH=str(bindir) + os.pathsep + os.environ["PATH"])
        self.props = {"debug.oculus.gpuLevel": "", "debug.oculus.cpuLevel": "3", "debug.oculus.guardian_pause": "0"}
        (self.state / "props.json").write_text(json.dumps(self.props))
        (self.state / "mode").write_text("normal")
        self.devices([QUEST_LINE, PHONE_LINE])

    def tearDown(self):
        self.tmp.cleanup()

    def devices(self, lines):
        (self.state / "devices.txt").write_text("List of devices attached\n" + "\n".join(lines) + "\n\n")

    def bench(self, *args):
        out = self.state / "out"
        p = subprocess.run([sys.executable, str(HERE / "bench.py"), "--out", str(out), "--poll-seconds", "0.01", *args],
                           env=self.env, capture_output=True, text=True, timeout=120)
        return p, out

    def calls(self):
        path = self.state / "calls.jsonl"
        return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []

    def shell_calls(self):
        return [c[3] for c in self.calls() if c[:1] == ["-s"] and c[2] == "shell"]

    def test_an_ab_run_measures_every_phase_and_puts_the_headset_back(self):
        p, out = self.bench("--ab", "--passes", "2", "--views", "hall_back,hallway")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        # Only ever the Quest.
        for c in self.calls():
            if c[:2] != ["devices", "-l"]:
                self.assertEqual(c[:2], ["-s", QUEST], c)
        # Put back: clock levels as they were, the lever file gone, the
        # proximity sensor returned after it was taken.
        self.assertEqual(json.loads((self.state / "props.json").read_text()), self.props)
        self.assertFalse((self.state / "fs" / "_sdcard_Android_data_com.example.questapp_files_levers.json").exists())
        shell = self.shell_calls()
        prox = [i for i, c in enumerate(shell) if "prox_close" in c]
        back = [i for i, c in enumerate(shell) if "automation_disable" in c]
        self.assertTrue(prox and back and back[0] > prox[0], shell)
        # Every phase at both views, and each phase's cost recovered.
        report = json.loads((out / "report.json").read_text())
        self.assertIsNone(report["failure"])
        for view in ("hall_back", "hallway"):
            phases = report["views"][view]["phases"]
            self.assertEqual(sorted(phases), sorted(["-"] + PHASES))
            self.assertEqual(phases["-"]["app_gpu_ms"], phases["baseline"]["app_gpu_ms"], "shipped and baseline disagree")
            for phase in PHASES:
                self.assertEqual(phases[phase]["windows"], 2, phase)
                self.assertAlmostEqual(phases["baseline"]["app_gpu_ms"] - phases[phase]["app_gpu_ms"], COST[phase])
                self.assertAlmostEqual(phases["baseline"]["gpu_wait_ms"] - phases[phase]["gpu_wait_ms"], COST[phase])
            self.assertEqual(report["views"][view]["vrapi"]["gpu_mhz"], [599])
        self.assertIn("| no_portals | 37.50 | 3.50 |", (out / "report.md").read_text())
        self.assertTrue((out / "perf.jsonl").exists())
        # The shipped renderer's own windows, apart from the schedule's, and
        # the GPU's counters read while it drew them: the first second is
        # left out, and only counters that say where the time goes are asked.
        for view in ("hall_back", "hallway"):
            self.assertEqual(report["views"][view]["phases"]["-"]["windows"], 3)
            self.assertEqual(report["views"][view]["gpu_counters"]["% Texture Fetch Stall"], 3.5)
        # One screenshot a view, pulled beside the report and shown in it.
        for view in ("hall_back", "hallway"):
            self.assertTrue((out / (view + ".jpg")).exists(), view)
            self.assertEqual(report["views"][view]["screenshot"], view + ".jpg")
        self.assertIn("![hallway](hallway.jpg)", (out / "report.md").read_text())
        profiled = [c for c in self.shell_calls() if c.startswith("timeout")]
        self.assertEqual(len(profiled), 2)
        asked = profiled[0].split('-r"')[1].split('"')[0].split(",")
        self.assertEqual(sorted(asked), ["1", "2", "3", "4"], "Preemptions / second is not a question worth a slot")

    def test_a_plain_run_takes_the_shipped_renderer_at_each_view(self):
        p, out = self.bench("--views", "pillar", "--windows", "4", "--no-profile")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        report = json.loads((out / "report.json").read_text())
        self.assertEqual(list(report["views"]["pillar"]["phases"]), ["-"])
        self.assertEqual(report["views"]["pillar"]["phases"]["-"]["windows"], 4)
        pushed = [c for c in self.calls() if c[2:3] == ["push"]]
        self.assertEqual(len(pushed), 1)

    def test_the_phone_alone_is_never_touched(self):
        self.devices([PHONE_LINE])
        p, _ = self.bench("--views", "pillar")
        self.assertEqual(p.returncode, 1)
        self.assertIn("not connected", p.stderr)
        self.assertEqual([c for c in self.calls() if c[:1] == ["-s"]], [])

    def test_the_right_serial_on_the_wrong_model_is_refused(self):
        self.devices([QUEST + "          device usb:1-1 product:x model:Pixel_7 device:x"])
        p, _ = self.bench("--views", "pillar")
        self.assertEqual(p.returncode, 1)
        self.assertIn("refusing", p.stderr)
        self.assertEqual([c for c in self.calls() if c[:1] == ["-s"]], [])

    def test_an_app_that_writes_nothing_fails_and_still_puts_everything_back(self):
        (self.state / "mode").write_text("silent")
        p, out = self.bench("--views", "pillar", "--stall-seconds", "0.5")
        self.assertEqual(p.returncode, 1)
        report = json.loads((out / "report.json").read_text())
        self.assertIn("written no results at all", report["failure"])
        self.assertEqual(json.loads((self.state / "props.json").read_text()), self.props)
        self.assertTrue(any("automation_disable" in c for c in self.shell_calls()))
        self.assertTrue(any(c.startswith("rm -f") and c.endswith("levers.json") for c in self.shell_calls()))

    def test_a_launch_the_quest_blocked_is_named(self):
        # The first real run (2026-09-27): no controller-required declaration,
        # controllers asleep on the desk, and the app never started.
        (self.state / "mode").write_text("blocked")
        p, out = self.bench("--views", "pillar", "--stall-seconds", "0.5", "--no-profile")
        self.assertEqual(p.returncode, 1)
        failure = json.loads((out / "report.json").read_text())["failure"]
        self.assertIn("blocked the app's launch", failure)
        self.assertIn("controller_required", failure)
        self.assertEqual(json.loads((self.state / "props.json").read_text()), self.props)

    def test_the_lever_file_is_what_the_renderer_parses(self):
        p = subprocess.run([sys.executable, str(HERE / "bench.py"), "--print-levers", "hall_back", "--ab"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        levers = json.loads(p.stdout)
        self.assertEqual(set(levers["bench"]), {"name", "eye", "at"}, "the app refuses any other field")
        self.assertIs(levers["ab_cycle"], True)


if __name__ == "__main__":
    unittest.main()
