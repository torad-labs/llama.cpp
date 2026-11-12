"""The paid window must use pinned bytes, matched serving and real device coverage."""
import copy
import contextlib
import hashlib
import io
import itertools
import json
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qualification import (BASELINE_COMMIT, device_cases, check_run_identity, check_quality, check_stop_at,
                           verify_files, validate_arch_table, validate_device_csv)


class QualificationTest(unittest.TestCase):
    def test_requested_profile_requires_verified_36_session_gpu_evidence(self):
        import qualification
        valid = {
            "start": {"exit": 0},
            "finalize": {"report_present": True},
            "request": {"active_sessions": 36, "launch_sha256": "launch"},
            "capture": {
                "capture_verified": True,
                "launch_sha256": "launch",
                "readiness": {
                    "usable": True, "expected_sequences": 36, "actual_sequences": 36, "actual_rows": 108,
                    "gpu_window": {"start_ns": 130, "end_ns": 160, "duration_ns": 30, "kernel_count": 4},
                },
            },
        }
        self.assertEqual(qualification.profile_evidence(True, False, valid, {"launch_sha256": "launch"})["state"], "usable")
        cases = [
            (False, False, None, None, "not_requested"),
            (True, True, None, None, "invalid"),
            (True, False, None, None, "invalid"),
            (True, False, {"start": {"exit": 7}}, None, "start_failed"),
            (True, False, {"start": {"exit": 0}, "finalize": {"report_present": False}}, None, "invalid"),
            (True, False, {"start": {"exit": 0}, "finalize": {"report_present": True}}, None, "invalid"),
            (True, False, valid | {"request": {"active_sessions": 36}, "capture": valid["capture"] | {"launch_sha256": None}}, None, "invalid"),
            (True, False, valid | {"capture": valid["capture"] | {"capture_verified": False}}, {"launch_sha256": "launch"}, "invalid"),
            (True, False, valid | {"capture": valid["capture"] | {"launch_sha256": "other"}}, {"launch_sha256": "launch"}, "invalid"),
            (True, False, valid | {"capture": valid["capture"] | {"readiness": valid["capture"]["readiness"] | {"gpu_window": None}}}, {"launch_sha256": "launch"}, "invalid"),
        ]
        for requested, running, record, workload, expected in cases:
            with self.subTest(expected=expected, record=record):
                result = qualification.profile_evidence(requested, running, record, workload)
                self.assertEqual(result["state"], expected)
                self.assertEqual(result["capture_verified"], expected == "usable")

    def test_matrix_success_requires_every_planned_leg_and_verified_requested_profile(self):
        import qualification
        good = dict(plan=[["B","2",[[36,512]],[]]], servers=[
            dict(short_ok=True, runs=[dict(active=36, context=512, exit=0, passed=True,
                                           profile={"state": "not_requested"})])])
        self.assertTrue(qualification.matrix_succeeded(good))
        profiled = copy.deepcopy(good)
        profiled["plan"][0].append({"profile": True})
        profiled["servers"][0]["runs"][0]["profile"] = {"state": "usable", "capture_verified": True}
        self.assertTrue(qualification.matrix_succeeded(profiled))
        budgeted = copy.deepcopy(profiled)
        budgeted["plan"][0][4]["tokens"] = 1024
        budgeted["servers"][0]["runs"][0]["decode_tokens"] = 256
        self.assertFalse(qualification.matrix_succeeded(budgeted))
        budgeted["servers"][0]["runs"][0]["decode_tokens"] = 1024
        self.assertTrue(qualification.matrix_succeeded(budgeted))
        variants = [dict(plan=good["plan"], servers=[]), dict(plan=[], servers=[])]
        for scope, key, value in (("server","short_ok",False), ("server","error","failed"),
                                  ("server","guard","memory_reserve"), ("run","exit",1),
                                  ("run","passed",False), ("run","skipped","window"),
                                  ("run","profile",{"state":"start_failed"})):
            bad = copy.deepcopy(good)
            target = bad["servers"][0]
            if scope == "run":
                target = target["runs"][0]
            target[key] = value
            variants.append(bad)
        for state in ("exported_unverified", "invalid", "not_requested"):
            bad = copy.deepcopy(profiled)
            bad["servers"][0]["runs"][0]["profile"] = {"state": state, "capture_verified": False}
            variants.append(bad)
        bad = copy.deepcopy(good)
        bad["servers"][0]["runs"] = []
        variants.append(bad)
        for bad in variants:
            self.assertFalse(qualification.matrix_succeeded(bad), bad)

    def test_controller_exits_nonzero_but_keeps_teardown_marker_when_engine_is_missing(self):
        import subprocess
        import time
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"results").mkdir()
            (root/"results/selected-gpus.json").write_text(json.dumps(dict(
                selected=[dict(index=i,uuid=f"GPU-fixture-{i}") for i in range(4)])))
            (root/"engines.py").write_text(
                "def install_pending(root): return []\ndef installed(root, engine): return False\n")
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("server_matrix.py")),
                "--root", folder, "--deadline", str(time.time()+1200), "--tag", "fixture",
                "--plan", json.dumps([["B","2",[[36,512]],[]]])],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads((root/"results/matrix-fixture.json").read_text())
            self.assertIs(report["passed"], False)
            self.assertTrue((root/"results/gpu-work-complete-fixture").is_file())

    def test_controller_retains_failed_leg_and_returns_failure_after_cleanup(self):
        self.exercise_controller_failure(False, 300)

    def test_profile_controller_uses_the_same_six_minute_reserve_as_preparation(self):
        self.exercise_controller_failure(True, 480)

    def test_controller_launch_receipt_omits_options_unsupported_by_the_baseline(self):
        self.exercise_controller_failure(False, 300, prompt_chunk=False)

    def exercise_controller_failure(self, profiled, seconds_left, prompt_chunk=True):
        import runpy
        import subprocess
        import time
        from types import SimpleNamespace
        original_popen = subprocess.Popen
        original_run = subprocess.run
        import os
        original_killpg = os.killpg
        detached = []
        sessions = []

        def reap(child):
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)

        sentinel = original_popen([sys.executable, '-I', '-c', 'import signal; signal.pause()']) if profiled else None
        if sentinel is not None:
            self.addCleanup(reap, sentinel)

        def run(command, **kwargs):
            if command[0] == 'nsys':
                if command[1] == 'shutdown':
                    self.assertIn('--session=' + sessions[0], command)
                    kill_signal = int(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--kill=')))
                    for child in detached:
                        original_killpg(child.pid, kill_signal)
                        child.wait(timeout=1)
                return subprocess.CompletedProcess(command, 0, 'fixture Nsight version\n', '')
            return original_run(command, **kwargs)

        class ServerProcess:
            pid = 12345
            returncode = None
            def poll(self):
                return self.returncode
            def wait(self, timeout=None):
                self.returncode = 0
                return 0

        def launch(command, **kwargs):
            if any(Path(arg).name == "llama-server" for arg in command) and "--help" not in command:
                if profiled:
                    child = original_popen([sys.executable, '-I', '-c',
                        "import signal; signal.signal(signal.SIGINT,signal.SIG_IGN); signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); signal.pause()"],
                        start_new_session=True, stdout=subprocess.PIPE, text=True)
                    self.assertEqual(child.stdout.readline().strip(), 'ready')
                    child.stdout.close()
                    detached.append(child)
                    self.addCleanup(reap, child)
                    sessions.append(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--session-new=')))
                return ServerProcess()
            return original_popen(command, **kwargs)

        def response(request, **kwargs):
            if request.full_url.endswith("/health"):
                data = {"status": "ok"}
            elif request.full_url.endswith("/v1/chat/completions"):
                data = {"choices": [{"message": {"content": "4"}}]}
            else:
                data = {"tokens": [1]*96}
            return io.BytesIO(json.dumps(data).encode())

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"results").mkdir()
            (root/"results/selected-gpus.json").write_text(json.dumps(dict(
                selected=[dict(index=i,uuid=f"GPU-fixture-{i}") for i in range(4)])))
            (root/"engines/B/bin").mkdir(parents=True)
            (root/"engines/B/SOURCE").write_text("a"*40)
            server = root/"engines/B/bin/llama-server"
            server.write_text(f"#!{sys.executable}\nprint(" +
                              repr("--prompt-step N\n" + ("--prompt-chunk N" if prompt_chunk else "")) + ")\n")
            server.chmod(0o755)
            (root/"scripts/glm53").mkdir(parents=True)
            (root/"scripts/glm53/workload.py").write_text(
                "import json,pathlib,sys\n"
                "root=pathlib.Path(sys.argv[sys.argv.index('--root')+1])\n"
                "name=sys.argv[sys.argv.index('--name')+1]\n"
                "out=root/'results'/name; out.mkdir()\n"
                "(out/'run.json').write_text(json.dumps({'passed':False,'error':'fixture failure'}))\n"
                "sys.exit(7)\n")
            plan = [["B","2",[[36,512]],[],dict(label="fixture", profile=profiled)]]
            argv = ["server_matrix", "--root", folder, "--deadline", str(time.time()+seconds_left),
                    "--tag", "failed-leg", "--plan", json.dumps(plan)]
            memory = "\n".join(f"{i}, 98000, 1000, 97000, 100, 0, 2400, 10000, 40" for i in range(4))
            engine = SimpleNamespace(install_pending=lambda root: [], installed=lambda root, name: True)
            with mock.patch.object(sys, "argv", argv), mock.patch.object(sys, "path", sys.path.copy()), \
                    mock.patch.dict(sys.modules, {"engines": engine}), \
                    mock.patch("subprocess.Popen", side_effect=launch), \
                    mock.patch("subprocess.run", side_effect=run), \
                    mock.patch("subprocess.check_output", return_value=memory), \
                    mock.patch("urllib.request.urlopen", side_effect=response), mock.patch("os.killpg"), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exit_:
                runpy.run_path(str(Path(__file__).with_name("server_matrix.py")), run_name="__main__")
            self.assertEqual(exit_.exception.code, 1)
            report = json.loads((root/"results/matrix-failed-leg.json").read_text())
            self.assertFalse(report["passed"])
            row = report["servers"][0]["runs"][0]
            self.assertEqual(row["exit"], 7)
            self.assertFalse(row["passed"])
            self.assertEqual(row["profile"]["state"], "invalid" if profiled else "not_requested")
            self.assertTrue((root/"results/gpu-work-complete-failed-leg").exists())
            launch_record = json.loads((root/"results/failed-leg-00-B-mtp2-fixture/launch.json").read_text())
            self.assertEqual("--prompt-chunk" in launch_record["argv"], prompt_chunk)
            if profiled:
                self.assertTrue(detached)
                self.assertTrue(all(child.poll() is not None for child in detached), 'profile cleanup left its detached application alive')
                self.assertIsNone(sentinel.poll(), 'profile cleanup touched a process outside its session')
            for flag in ("LLAMA_DECODE_TRACE", "GGML_CUDA_NVTX", "GGML_GRAPH_TRACE", "LLAMA_TIME_DECODE"):
                self.assertEqual(launch_record["env"].get(flag), "1" if profiled else None)

    def test_retained_files_are_verified_without_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            content = bytes(range(256)) * 17
            (root / "shard").write_bytes(content)
            files = {"shard": {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}}
            self.assertEqual(verify_files(root, files), files)
            (root / "shard").write_bytes(content[:-1] + b"!")
            with self.assertRaisesRegex(ValueError, "identity"):
                verify_files(root, files)
            self.assertEqual((root / "shard").read_bytes(), content[:-1] + b"!")
            (root / "shard").unlink()
            with self.assertRaises(FileNotFoundError):
                verify_files(root, files)
            self.assertFalse((root / "shard").exists())

    def test_matched_serving_and_exact_baseline(self):
        meta = {"args": {"context": 1048576, "ubatch": 512, "slots": 3, "depth": 2,
                         "sampling": "served", "repeats": 3, "tokens": 384, "adaptive": 0.18},
                "binary_manifest": {"commit": BASELINE_COMMIT, "tarball_sha256": "a"*64},
                "engine_env": {"GGML_CUDA_ALLREDUCE": "internal", "GGML_CUDA_GRAPH_MAX": "0"},
                "profile": "", "instrumented": False, "model_sha256": "b"*64}
        expected = {"commit": BASELINE_COMMIT, "tarball_sha256": "a"*64, "model_sha256": "b"*64}
        self.assertTrue(check_run_identity(meta, expected, 1048576, 512, ""))
        for key, bad in (("sampling", "greedy"), ("slots", 1), ("depth", 1), ("ubatch", 1024),
                         ("context", 524288), ("adaptive", 0.0), ("tokens", 64), ("repeats", 1)):
            row = copy.deepcopy(meta)
            row["args"][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_run_identity(row, expected, 1048576, 512, "")
        for key, bad in (("profile", "sm120-direct"), ("instrumented", True), ("model_sha256", "c"*64)):
            row = copy.deepcopy(meta)
            row[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_run_identity(row, expected, 1048576, 512, "")
        row = copy.deepcopy(meta)
        row["binary_manifest"]["commit"] = "fdf8d4044a6002a2c7ad300b088a0fd9bbcd6175"
        with self.assertRaises(ValueError):
            check_run_identity(row, expected, 1048576, 512, "")

        # A timing run must reject hidden check/counter/dispatch overrides, even
        # when its top-level instrumentation label was accidentally left false.
        for key in ("GGML_CUDA_ROUTE_STATS", "GGML_CUDA_MMVQ_Q8_SM120_CHECK", "GGML_CUDA_Q8_TMA"):
            row = copy.deepcopy(meta)
            row["engine_env"][key] = "1"
            with self.subTest(env=key), self.assertRaises(ValueError):
                check_run_identity(row, expected, 1048576, 512, "")

    def test_candidate_records_the_complete_resolved_profile(self):
        resolved = {"GGML_CUDA_Q8_SPLIT": "1", "GGML_CUDA_Q8_TMA": "1"}
        expected = {"commit": "c"*40, "tarball_sha256": "a"*64,
                    "model_sha256": "b"*64, "resolved_env": resolved}
        meta = {"args": dict(context=1048576, ubatch=512, slots=3, depth=2,
                             sampling="served", repeats=3, tokens=384, adaptive=0.18),
                "binary_manifest": expected, "profile": "sm120-balanced", "instrumented": False,
                "model_sha256": expected["model_sha256"],
                "engine_env": resolved | {"GGML_CUDA_ALLREDUCE": "internal", "GGML_CUDA_GRAPH_MAX": "0"}}
        self.assertTrue(check_run_identity(meta, expected, 1048576, 512, "sm120-balanced"))
        for key in resolved:
            row = copy.deepcopy(meta)
            del row["engine_env"][key]
            with self.assertRaises(ValueError):
                check_run_identity(row, expected, 1048576, 512, "sm120-balanced")

    def test_device_receipt_requires_supported_passing_cases(self):
        header = '"backend_name","supported","passed","error_message"\n'
        good = '"CUDA0",1,1,""\n'
        self.assertTrue(validate_device_csv("backend info\n" + header + good*3, 3))
        for output in (header, header+good*2, header+good*4,
                       header+'"CUDA0",1,0,""\n',
                       header+'"CUDA0",0,1,""\n',
                       header+'"CUDA0",1,1,"mismatch"\n',
                       header+'"CPU",1,1,""\n',
                       header+good*2+'"CUDA0"\n'):
            with self.assertRaises(ValueError):
                validate_device_csv(output, 3)
        # test-backend-ops labels an unsupported case with every backend that rejects
        # one of its tensors, so a rejected case can carry any label but plain CUDA0.
        # Every CSV row is a case the matrix asked for, whatever its label.
        for extra in ('"CUDA0, CUDA0",0,0,"not supported"\n', '"CPU",0,0,"not supported"\n'):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "not supported"):
                validate_device_csv(header + good*3 + extra, 3)
        # A mismatch prints its diagnostic on the row's own line, before the quote.
        with self.assertRaisesRegex(ValueError, "test failed"):
            validate_device_csv(header + good*2 + '[GATED_DELTA_NET] ERR = 0.1 > 0.0005 "CUDA0",1,0,"test failed"\n', 3)
        # Every case the matrix runs states how many cases it must see.
        with self.assertRaises(ValueError):
            validate_device_csv(header + good*3, None)

    def test_every_backend_ops_case_pins_its_case_count(self):
        # A filter that silently matches fewer cases, or a backend that drops one,
        # must fail; only a probe that is meant to abort carries no count.
        for case in device_cases():
            if case["argv"][0] == "test-backend-ops" and "expected_failure" not in case:
                with self.subTest(case=case["name"]):
                    self.assertIsInstance(case["cases"], int)
                    self.assertGreater(case["cases"], 0)
        counts = {c["name"]: c["cases"] for c in device_cases()}
        for gpu in (0, 1):
            # 113 gated-delta-net and 72 cache-fusion KDA shapes, 11 of them at GLM's 32 heads a card;
            # 44+16+5 HC shapes.
            for name in [f"kda-c1-f0-gpu{gpu}"] + [f"kda-c{c}-f{f}-gpu{gpu}" for c in (2, 4) for f in (0, 1)]:
                self.assertEqual(counts[name], 185, name)
            for publication in (0, 1):
                self.assertEqual(counts[f"hc-publication-{publication}-gpu{gpu}"], 65)

    def test_long_context_direct_attention_requires_real_devices(self):
        cases = {case["name"]: case for case in device_cases()}
        for gpu in (0, 1):
            case = cases[f"direct-attention-400k-gpu{gpu}"]
            self.assertEqual(case["argv"], ["test-fattn-sparse-q8", "--direct-longctx", "--require-cuda"])
            self.assertEqual(case["env"]["CUDA_VISIBLE_DEVICES"], str(gpu))
            receipt = case["require"]
            self.assertRegex("16/16 long-context direct attention cases passed on 1 CUDA device", receipt)
            for incomplete in ("no CUDA device, skipped", "15/16 long-context direct attention cases passed on 1 CUDA device",
                               "0/0 long-context direct attention cases passed on 0 CUDA devices", "OK",
                               "116/16 long-context direct attention cases passed on 1 CUDA device",
                               "16/16 long-context direct attention cases passed on 1 CUDA device SKIPPED",
                               "16/16 long-context direct attention cases passed on 2 CUDA devices"):
                self.assertIsNone(re.search(receipt, incomplete), incomplete)

    def test_architecture_cases_require_every_device_to_match_the_cpu(self):
        # test-llama-archs exits 0 when every row is SKIP. On the two-card box each
        # card and the CPU must match the CPU logits and survive a save/load, and
        # both tensor splits must match; a tensor split has no roundtrip by design.
        ok, skip, fail = "\033[1;32mOK\033[0m", "\033[1;33mSKIP\033[0m", "\033[1;31mFAIL\033[0m"

        def table(arch, rows):
            head = "main: using seed 1234\n|     Model arch.|  Device|Config|   NMSE vs. CPU|Roundtrip|\n|---|---|---|---|---|\n"
            return head + "".join(f"|{arch:>16}|{device:>24}|{'MoE':>6}|{nmse:>15}|{trip:>20}|\n"
                                  for device, nmse, trip in rows)
        good = [("NVIDIA RTX PRO 6000 Blackwell Server Edition", ok + " (1.20e-07)", ok)] * 2 + [
            ("AMD EPYC 9354 32-Core Processor", ok + " (0.00e+00)", ok),
            ("Meta", ok + " (2.10e-07)", skip), ("Meta 3:7", ok + " (2.30e-07)", skip)]
        self.assertTrue(validate_arch_table(table("glm5next", good), "glm5next", 3))
        cpu_only = [good[2], ("Meta", skip + "           ", skip), ("Meta 3:7", skip + "           ", skip)]
        broken = {"all skipped": [(d, skip + "           ", skip) for d, _, _ in good],
                  "no GPU": cpu_only,
                  "one card missing": good[1:],
                  "NMSE failed": [good[0], (good[1][0], fail + " (3.00e-02)", ok)] + good[2:],
                  "MTP vocab": [(good[0][0], "\033[1;31mFAIL MTP vocab\033[0m (1.20e-07)", ok)] + good[1:],
                  "roundtrip failed": [(good[0][0], good[0][1], fail)] + good[1:],
                  "card roundtrip skipped": [(good[0][0], good[0][1], skip)] + good[1:],
                  "split skipped": good[:3] + [("Meta", skip + "           ", skip), good[4]],
                  # test-llama-archs marks a row OK unless its NMSE exceeds 1e-4, which NaN never does.
                  "NMSE nan": [(good[0][0], ok + " (nan)", ok)] + good[1:],
                  "NMSE -nan": good[:3] + [("Meta", ok + " (-nan)", skip), good[4]],
                  "NMSE inf": [good[0], (good[1][0], ok + " (inf)", ok)] + good[2:],
                  "NMSE above 1e-4": [(good[0][0], ok + " (2.00e-04)", ok)] + good[1:],
                  "NMSE unreadable": [(good[0][0], ok + " (fast)", ok)] + good[1:],
                  "unfinished row": good[:4] + [("Meta 3:7", "", "")]}
        for label, rows in broken.items():
            with self.subTest(label), self.assertRaises(ValueError):
                validate_arch_table(table("glm5next", rows), "glm5next", 3)
        with self.assertRaises(ValueError):
            validate_arch_table(table("qwen35", good), "glm5next", 3)
        cases = {c["name"]: c for c in device_cases()}
        for name, arch in (("glm-architecture", "glm5next"), ("glm-ordinary-cache", "glm5next"),
                           ("qwen-architecture", "qwen35")):
            self.assertEqual(cases[name].get("table"), dict(arch=arch, devices=3), name)

    def test_the_armed_stop_must_fall_inside_the_approved_cap(self):
        # The cap is the number of hours Marcos approved for the window; one by default.
        now = 1_000_000.0
        for stop_in, hours, accepted in ((3500, 1, True), (3700, 1, False), (5000, 1.5, True), (5500, 1.5, False),
                                         (0, 1, False), (-10, 1.5, False)):
            with self.subTest(stop_in=stop_in, hours=hours):
                if accepted:
                    self.assertTrue(check_stop_at(now + stop_in, now, hours))
                else:
                    with self.assertRaisesRegex(ValueError, f"at most {hours:g} hours? away"):
                        check_stop_at(now + stop_in, now, hours)

    def test_all_contexts_must_meet_distribution_contract(self):
        quality = {str(n): {"meanKld": 0, "sameTopP": 100} for n in (2048, 4096, 16384)}
        self.assertTrue(check_quality(quality))
        for n in quality:
            for key, bad in (("meanKld", 1.1e-6), ("meanKld", -1.1e-6),
                             ("meanKld", float("nan")), ("sameTopP", 99.98)):
                record = copy.deepcopy(quality)
                record[n][key] = bad
                with self.subTest(context=n, key=key), self.assertRaises(ValueError):
                    check_quality(record)
            record = copy.deepcopy(quality)
            del record[n]
            with self.assertRaises(ValueError):
                check_quality(record)

    def test_device_matrix_keeps_two_cards_and_coupled_variants(self):
        cases = device_cases()
        names = [case["name"] for case in cases]
        self.assertEqual(len(names), len(set(names)))
        for gpu in (0, 1):
            subset = [c for c in cases if c["env"].get("CUDA_VISIBLE_DEVICES") == str(gpu)]
            for legacy in (0, 1):
                name = f"q8-legacy-priority-{legacy}-gpu{gpu}"
                self.assertIn(name, names)
                probe = next(c for c in subset if c["name"] == name)
                self.assertEqual(probe["env"]["GGML_CUDA_Q8_SPLIT"], "1")
                self.assertEqual(probe["env"]["GGML_CUDA_Q8_TMA"], "1")
                self.assertEqual(probe["env"]["GGML_CUDA_MMVQ_Q8_SM120_LEGACY"], str(legacy))
                self.assertEqual(probe["env"]["GGML_CUDA_Q8_TMA_SLOTS"], "invalid")
                if legacy:
                    self.assertNotIn("expected_failure", probe)
                    self.assertEqual(probe["cases"], 203)
                else:
                    self.assertEqual(probe["expected_failure"],
                                     dict(exit=-6, message="GGML_CUDA_Q8_TMA_SLOTS must be 2 or 4"))
            for packing in (0, 1):
                case = next(c for c in subset if c["name"] == f"q8-meta-cache-p{packing}-gpu{gpu}")
                self.assertEqual(case["env"]["GGML_CUDA_Q8_SPLIT"], str(packing))
                self.assertIn("--meta-cache", case["argv"])
                self.assertEqual(case["require"], "16/16 meta Q8 cache rebuilds passed")
            self.assertIn(f"q8-channel-views-gpu{gpu}", names)
            for mode in ("direct", "tma2", "tma4"):
                self.assertTrue(any(c["name"] == f"q8-readers-{mode}-gpu{gpu}" for c in subset))
            for compute in ("f16", "bf16", "f32"):
                self.assertIn(f"q8-cublas-{compute}-gpu{gpu}", names)
            for unroll in (0, 1, 2, 4):
                for kind, count in (("q8", 106), ("fused", 94), ("concat", 17)):
                    case = next(c for c in subset if c["name"] == f"q8-{kind}-u{unroll}-gpu{gpu}")
                    self.assertEqual(case["cases"], count)
            for columns in (1, 2, 4):
                for front in (0, 1) if columns > 1 else (0,):
                    self.assertIn(f"kda-c{columns}-f{front}-gpu{gpu}", names)
            for slots in (2, 4, 6, 8):
                case = next(c for c in subset if c["name"] == f"iq3-s{slots}-gpu{gpu}")
                self.assertEqual(case["cases"], 72)
                self.assertNotIn("-p", case["argv"])
        for route in ("peer", "host", "mixed", "duplicate"):
            self.assertIn(f"capture-{route}", names)
        # The peer route's line, as the box printed it on October 24 (capture-peer.out); host staging is not it.
        peer = next(c for c in cases if c["name"] == "capture-peer")["require"]
        self.assertRegex("ok   card-to-card limit 131072 bytes: device inboxes exercised", f"^{peer}$")
        self.assertNotRegex("FAIL card-to-card limit 0 bytes: host staging only", f"^{peer}$")

    def test_kda_cases_select_every_shape_glm_builds_and_no_rows_mode(self):
        # glm5next's build_kda_layer never passes state_rows, so the rows-indexed
        # state read (src[6]) is not on GLM's path and CUDA rejects it by design.
        # Every other KDA gated-delta-net and KDA cache-fusion shape stays required.
        # The strings follow test-backend-ops' vars(), which -p searches as a regex.
        gdn = ("type={ty},head_count=4,head_size={s},n_seq_tokens={t},n_seqs={b},v_repeat={r},permuted={p},"
               "kda={kda},K={K},rows_mode={rows},cache_rows={c},raw_gates={g},qkv_view={q}")
        fusion = "type={ty},head_count=4,head_size={s},n_seq_tokens={t},n_seqs={b},K={K},v_repeat={r},raw_gates={g},cache_type={ty}"
        kda_cases = [c for c in device_cases() if c["name"].startswith("kda-")]
        self.assertEqual(len(kda_cases), 10)
        wrong = []
        for case in kda_cases:
            pattern = case["argv"][case["argv"].index("-p") + 1]
            for ty, s, t, b, r, p, kda, K, rows, c, g, q in itertools.product(
                    ("f32", "f16"), (16, 64, 128), (1, 4, 32, 128), (1, 3), (1, 2), (0, 1), (0, 1),
                    (1, 2, 9, 17), (0, 1), (-1, 4, 6), (0, 1), (0, 1)):
                values = dict(ty=ty, s=s, t=t, b=b, r=r, p=p, kda=kda, K=K, rows=rows, c=c, g=g, q=q)
                text = gdn.format(**values)
                if bool(re.search(pattern, text)) != (kda == 1 and rows == 0):
                    wrong.append((case["name"], text))
                for tail in ("", f",mem_size={c},kv_head=1"):
                    for row in ("", f",state_row={rows}"):
                        text = fusion.format(**values) + tail + row + (",kda=1" if kda else "")
                        if bool(re.search(pattern, text)) != (kda == 1):
                            wrong.append((case["name"], text))
        self.assertEqual(wrong[:3], [])


class ControlPreparationTest(unittest.TestCase):
    def test_quality_plan_isolates_engine_query_and_chunk_changes(self):
        import control_matrix
        plan = control_matrix.prepare()
        jobs = plan.get("quality_jobs", [])
        self.assertTrue(jobs, "The plan must execute distributions, not just declare a KL threshold.")
        expected = {
            ("engine-f32", "C", "f32", 512, "B", "f32", 512),
            ("engine-q8", "C", "q8", 512, "B", "q8", 512),
            ("chunk-f32", "C", "f32", 128, "C", "f32", 512),
            ("chunk-q8", "C", "q8", 128, "C", "q8", 512),
            ("query-512", "C", "q8", 512, "C", "f32", 512),
            ("query-128", "C", "q8", 128, "C", "f32", 128),
        }
        for context in (2048, 4096, 16384):
            selected = [job for job in jobs if job["context"] == context]
            references = {job["id"]: job for job in selected if job["action"] == "reference"}
            actual = set()
            for job in selected:
                if job["action"] != "compare":
                    continue
                ref = references[job["reference_id"]]
                self.assertLess(selected.index(ref), selected.index(job))
                actual.add((job["comparison"], job["engine"], job["query"], job["ubatch"],
                            ref["engine"], ref["query"], ref["ubatch"]))
            self.assertEqual(actual, expected)

    def test_quality_cli_runs_every_control_and_retains_numerical_failures(self):
        import os
        import time
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "results").mkdir()
            (root / "results/selected-gpus.json").write_text(json.dumps({
                "selected": [{"uuid": f"GPU-fixture-{i}"} for i in range(4)]}))
            (root / "model").mkdir()
            (root / "model/GLM-5.3-Flash-Uncensored-IQ3_XXS-00001-of-00003.gguf").write_bytes(b"public fixture")
            corpus = b"Public numerical-control fixture.\n"
            (root / "public-corpus.txt").write_bytes(corpus)
            monitors = root / "monitor"
            monitors.mkdir()
            (monitors / "free-mib").write_text("8192")
            smi = monitors / "nvidia-smi"
            smi.write_text("#!/usr/bin/env python3\nfrom pathlib import Path\n"
                           "free=Path(__file__).with_name('free-mib').read_text()\n"
                           "for i in range(4): print(f'GPU-fixture-{i}, {free}')\n")
            smi.chmod(0o755)
            environment = dict(os.environ, PATH=str(monitors) + os.pathsep + os.environ["PATH"])
            for engine in ("B", "C"):
                folder = root / "engines" / engine
                (folder / "bin").mkdir(parents=True)
                source = ("24c77a0b5c8dcb3352e8027cd7e34367bdaeec13+dirty-sha256:"
                          "098fcb005ad4b0575ece6786e1da219d49c5b337a6e72a06404c0635b0014510") if engine == "B" else "fixture-C"
                (folder / "SOURCE").write_text(source)
                binary = folder / "bin/llama-perplexity"
                binary.write_text(
                    "#!/usr/bin/env python3\n"
                    "import json,os,sys,signal,threading\n"
                    "from pathlib import Path\n"
                    "monitor=Path(__file__).resolve().parents[3]/'monitor'\n"
                    "if (monitor/'drop').exists() and not (monitor/'attempted').exists():\n"
                    " (monitor/'attempted').touch(); (monitor/'free-mib').write_text('4095')\n"
                    " def stopped(*args):\n"
                    "  (monitor/'terminated').touch(); sys.exit(2)\n"
                    " signal.signal(signal.SIGTERM,stopped); threading.Event().wait(5)\n"
                    "a=sys.argv; ref=Path(a[a.index('--kl-divergence-base')+1])\n"
                    "if '--kl-divergence' not in a:\n"
                    " ref.write_text(json.dumps({'query':os.environ['LLAMA_INDEXER_Q8_QUERY'],'ubatch':a[a.index('-ub')+1]}))\n"
                    "else:\n"
                    " assert ref.is_file() and ref.stat().st_size\n"
                    " bad=os.environ['LLAMA_INDEXER_Q8_QUERY']=='1' and a[a.index('-ub')+1]=='128'\n"
                    " print('Mean KLD:',0.016772 if bad else 0.0)\n"
                    " print('Same top p:',95.797 if bad else 100.0)\n")
                binary.chmod(0o755)
            out = root / "results/quality"
            run = subprocess.run([
                sys.executable, "-I", str(Path(__file__).with_name("control_matrix.py")),
                "--execute-quality", "--root", str(root), "--deadline", str(time.time() + 60),
                "--corpus-sha256", hashlib.sha256(corpus).hexdigest(), "--output", str(out),
            ], capture_output=True, text=True, timeout=70, env=environment)
            self.assertTrue((out / "quality.json").is_file(), run.stderr)
            report = json.loads((out / "quality.json").read_text())
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertTrue(report["completed"], report)
            self.assertFalse(report["passed"])
            self.assertEqual(len(report["runs"]), len(report["plan"]))
            comparisons = [row for row in report["runs"] if row["action"] == "compare"]
            self.assertEqual(len(comparisons), 18)
            self.assertEqual(sum(not row["quality"]["passed"] for row in comparisons), 6)
            self.assertEqual({row["comparison"] for row in comparisons}, {
                "engine-f32", "engine-q8", "chunk-f32", "chunk-q8", "query-512", "query-128"})
            for row in report["runs"]:
                self.assertEqual(row["exit"], 0)
                self.assertTrue((out / row["log"]).is_file())
                if row["action"] == "reference":
                    self.assertTrue((out / row["distribution"]).is_file())
            baseline_source = (root / "engines/B/SOURCE").read_text()
            for case in ("wrong-source", "unbounded-deadline", "reserve-at-start", "reserve-during-work"):
                with self.subTest(case=case):
                    (monitors / "free-mib").write_text("4095" if case == "reserve-at-start" else "8192")
                    if case == "reserve-during-work":
                        (monitors / "drop").touch()
                    (root / "engines/B/SOURCE").write_text("different-engine" if case == "wrong-source" else baseline_source)
                    rejected_out = root / "results" / case
                    rejected = subprocess.run([
                        sys.executable, "-I", str(Path(__file__).with_name("control_matrix.py")),
                        "--execute-quality", "--root", str(root),
                        "--deadline", "nan" if case == "unbounded-deadline" else str(time.time() + 60),
                        "--corpus-sha256", hashlib.sha256(corpus).hexdigest(), "--output", str(rejected_out),
                    ], capture_output=True, text=True, timeout=70, env=environment)
                    self.assertEqual(rejected.returncode, 1, rejected.stderr)
                    refusal = json.loads((rejected_out / "quality.json").read_text())
                    self.assertFalse(refusal["completed"])
                    if case == "reserve-during-work":
                        self.assertEqual(len(refusal["runs"]), 1)
                        self.assertFalse(refusal["runs"][0]["passed"])
                        self.assertTrue((monitors / "terminated").is_file())
                    else:
                        self.assertEqual(refusal["runs"], [])

    def test_serving_controls_keep_the_measured_recipe_and_use_abba(self):
        import control_matrix
        plan = control_matrix.prepare()
        self.assertEqual(plan["source_identity"]["baseline"], {
            "engine": "B", "commit": "24c77a0b5c8dcb3352e8027cd7e34367bdaeec13",
            "patch_sha256": "098fcb005ad4b0575ece6786e1da219d49c5b337a6e72a06404c0635b0014510",
            "source_record": "engines/B/SOURCE",
        })
        self.assertEqual(plan["serving_baseline"]["mtp_depth"], "2")
        self.assertEqual(plan["serving_baseline"]["indexer_query"], "q8")
        self.assertEqual(plan["serving_baseline"]["ubatch"], 512)
        self.assertEqual(plan["quality_contract"], {"meanKld_abs_max": 1e-6, "sameTopP_min": 99.99})
        entries = plan["server_matrix_plan"]
        self.assertEqual(sum(bool(entry[4].get("profile")) for entry in entries), 1)
        for depth in ("off", "1", "2"):
            group = [entry for entry in entries if entry[1] == depth and not entry[4].get("profile")]
            self.assertEqual([entry[0] for entry in group], ["B", "C", "C", "B"])
            for entry in group:
                self.assertEqual(entry[2], [[36, 512], [36, 8192]])
                self.assertNotIn("--diagnostic-ubatch", entry[4]["serve"])
                self.assertEqual(entry[4]["serve"], ["--transport", "internal-nccl-gather", "--indexer-query", "q8"])
        self.assertEqual(plan["profile_capture"]["active_sessions"], 36)
        self.assertTrue(plan["profile_capture"]["profile_requested"])
        self.assertEqual(plan["profile_capture"]["timing"], "gpu-cupti-only")

    def test_control_cli_writes_a_checked_server_matrix_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "control-plan.json"
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("control_matrix.py")),
                                     "--output", str(output), "--root", "/rental", "--deadline", "1234.5"],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(output.read_text())
            command = plan["execution"]["command"]
            self.assertEqual(command[command.index("--root") + 1], "/rental")
            self.assertEqual(float(command[command.index("--deadline") + 1]), 1234.5)
            self.assertEqual(json.loads(command[command.index("--plan") + 1]), plan["server_matrix_plan"])
            self.assertEqual(command[1], "/rental/scripts/glm53/server_matrix.py")
            self.assertEqual(plan["execution"]["quality_command"][2], "/rental/scripts/glm53/control_matrix.py")


class ServingRecipeTest(unittest.TestCase):
    def setUp(self):
        import serve
        self.serve = serve

    def test_default_retains_the_two_card_tensor_recipe(self):
        args = self.serve.arguments(["--model", "/models/first shard.gguf", "--mmproj", "/models/projector.gguf"])
        argv, env = self.serve.make_plan(args)
        self.assertEqual(argv, [str(self.serve.ROOT / "build/bin/llama-server"),
            "-m", "/models/first shard.gguf", "--mmproj", "/models/projector.gguf",
            "-ngl", "99", "-sm", "tensor", "-ts", "1,1", "--jinja", "-fa", "on",
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "-ctki", "q8_0",
            "-b", "2048", "-ub", "2048", "--prompt-step", "1024",
            "--alias", "GLM-5.3-Flash-Uncensored-IQ3_XXS", "--chat-template-file", str(self.serve.TEMPLATE),
            "--spec-type", "draft-mtp", "--spec-draft-n-max", "2",
            "--spec-draft-mtp-vocab", str(self.serve.DRAFT_VOCAB), "--spec-draft-mtp-adaptive", "0.18",
            "-c", "524288", "-np", "3", "--kv-unified", "--cache-ram", "32768",
            "--no-cache-idle-slots", "--metrics", "--host", "127.0.0.1", "--port", "8102",
            "--log-timestamps"])
        self.assertEqual(env, {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0,1",
            "GGML_CUDA_GRAPH_MAX": "0", "GGML_CUDA_ALLREDUCE": "internal",
            "LLAMA_INDEXER_Q8_QUERY": "1", "LLAMA_MTP_DEVICE_H": "1", "LLAMA_TOK_EMBD_GPU": "1",
            "LLAMA_KPOOL_DIRECT_INDICES": "0", "GGML_CUDA_MMQ_MOE_NCOLS": "0",
            "GGML_CUDA_MMQ_L2_PF": "1", "GGML_SCHED_DEBUG": "0",
            "LD_LIBRARY_PATH": str(self.serve.ROOT / "build/bin")})
        self.assertNotIn("--glm53-profile", argv)
        with self.assertRaisesRegex(ValueError, "transport"):
            self.serve.make_plan(self.serve.arguments(["--model", "/model.gguf", "--transport", "internal-nccl-gather"]))

    def test_dp_capacity_recipe_keeps_each_slots_cache_on_one_owner(self):
        for slots in (1, 4, 36, 40):
            for depth in ("2", "off"):
                with self.subTest(slots=slots, depth=depth):
                    args = self.serve.arguments(["--model", "/models/target.gguf", "--profile", "dp-4",
                                                 "--slots", str(slots), "--mtp-depth", depth])
                    argv, env = self.serve.make_plan(args)
                    self.assertIn("--prompt-chunk", argv)
                    self.assertEqual(argv[argv.index("--prompt-chunk") + 1], "128")
                    expected = {"-sm": "dp", "-ts": "1,1,1,1", "-fit": "off", "-np": str(slots),
                                "-c": str(slots * 442368), "-b": "2048", "-ub": "512", "--prompt-step": "512",
                                "--cache-type-k": "q8_0", "--cache-type-v": "q8_0", "-ctki": "q8_0", "-cts": "q8_0"}
                    for flag, value in expected.items():
                        self.assertEqual(argv[argv.index(flag) + 1], value)
                    self.assertNotIn("--kv-unified", argv)
                    # each slot keeps its own state in VRAM, and a slot taken by another conversation still saves its
                    # state to host RAM first, so copying every idle slot there when a request starts only held the
                    # others back: at 36 sessions on four PCIe cards each next turn's first token waited 5.5-14.5 s
                    # for it, and 1.3-1.6 s without it
                    self.assertEqual(argv[argv.index("--cache-ram") + 1], "32768")
                    self.assertIn("--no-cache-idle-slots", argv)
                    self.assertIn("--backend-sampling", argv)
                    self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1,2,3")
                    self.assertEqual(env["GGML_CUDA_ALLREDUCE"], "internal")
                    self.assertEqual(env["GGML_CUDA_GRAPH_MAX"], "0")
                    self.assertEqual(env["GGML_CUDA_GRAPH_MEM_STATS"], "0")
                    self.assertEqual(env["GGML_CUDA_GRAPH_NODES"], "32768")
                    self.assertEqual(env["GGML_CUDA_GRAPH_IDLE_MS"], "0")
                    self.assertNotIn("LLAMA_KPOOL_DIRECT_INDICES", env)
                    for flag, value in {"-ubd": "512", "-ctkd": "q8_0", "-ctvd": "q8_0"}.items():
                        if depth == "off":
                            self.assertNotIn(flag, argv)
                        else:
                            self.assertEqual(argv[argv.index(flag) + 1], value)
                    if depth == "off":
                        self.assertFalse(any(arg.startswith("--spec-") for arg in argv))
                    else:
                        self.assertEqual(argv[argv.index("--spec-draft-n-max") + 1], depth)

    def test_dp_transport_selection_scrubs_inherited_overrides(self):
        for slots, transport, inherited_transport in itertools.product((1, 36, 40),
                ('internal', 'internal-nccl-gather'), ('internal', 'internal-nccl-gather', 'nccl')):
            with self.subTest(slots=slots, transport=transport, inherited=inherited_transport):
                args = self.serve.arguments(['--model', '/model.gguf', '--profile', 'dp-4', '--slots', str(slots),
                                             '--transport', transport, '--dry-run'])
                _, settings = self.serve.make_plan(args)
                env = self.serve.execution_environment(settings, {'GGML_CUDA_ALLREDUCE': inherited_transport,
                                                                  'GGML_CUDA_Q8_TMA': '1', 'PATH': '/bin'})
                self.assertEqual(env['GGML_CUDA_ALLREDUCE'], transport)
                self.assertNotIn('GGML_CUDA_Q8_TMA', env)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.serve.main(args)
                self.assertEqual(json.loads(output.getvalue())['env']['GGML_CUDA_ALLREDUCE'], transport)

    def test_qualification_legacy_transport_cannot_override_explicit_choice(self):
        for explicit, legacy in itertools.product((None, 'internal', 'internal-nccl-gather'),
                                                   (None, 'internal', 'internal-nccl-gather')):
            options = ['--model', '/model.gguf', '--profile', 'dp-4']
            if explicit:
                options += ['--transport', explicit]
            args = self.serve.arguments(options)
            if explicit and legacy and explicit != legacy:
                with self.assertRaisesRegex(ValueError, 'transport'):
                    self.serve.resolve_transport(args, legacy)
            else:
                self.assertEqual(self.serve.resolve_transport(args, legacy), explicit or legacy or 'internal')

    def test_four_card_tensor_recipe_shares_one_pool_on_the_four_bit_cache(self):
        args = self.serve.arguments(["--model", "/models/target.gguf", "--profile", "tensor-4"])
        argv, env = self.serve.make_plan(args)
        expected = {"-sm": "tensor", "-ts": "1,1,1,1", "-np": "23", "-c": "524288", "-b": "2048", "-ub": "2048",
                    "--prompt-step": "1024", "--prompt-chunk": "128",
                    "--cache-type-k": "q4_0", "--cache-type-v": "q4_0", "-ctki": "q8_0",
                    "--spec-type": "draft-mtp", "--spec-draft-n-max": "2", "--spec-draft-mtp-adaptive": "0.18"}
        for flag, value in expected.items():
            self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertIn("--kv-unified", argv)
        for flag in ("--backend-sampling", "-fit", "-cts", "-ubd", "--spec-draft-model"):
            self.assertNotIn(flag, argv)
        two = self.serve.make_plan(self.serve.arguments(["--model", "/models/target.gguf"]))[1]
        self.assertNotIn("LLAMA_SPEC_COUPLED", two)
        # what the four cards serve with and the two do not: the reductions on the copy engines through host memory in
        # 8-bit blocks, graphs kept by shape, CUDA executables held between turns
        wire = {"GGML_CUDA_AR_WIRE_Q8": "1"}
        four = {"GGML_CUDA_AR_LL_MAX_BYTES": "0", "GGML_CUDA_AR_P2P_MAX_BYTES": "0", "GGML_CUDA_AR_COPY_THRESHOLD": "1",
                "GGML_CUDA_AR_COPY_PEER": "0", "GGML_CUDA_AR_GATHER_OWNER_SLICES": "0",
                "LLAMA_GRAPH_ANY_SEQS": "1", "LLAMA_GRAPH_SLOTS": "128", "GGML_META_CAPTURE_GRAPHS": "132",
                "GGML_CUDA_GRAPH_NODES": "262144", "GGML_CUDA_GRAPH_IDLE_MS": "600000", "CUDA_VISIBLE_DEVICES": "0,1,2,3"}
        for key in list(four) + list(wire):
            if key != "CUDA_VISIBLE_DEVICES":
                self.assertNotIn(key, two)
        self.assertEqual(env, {**two, **four, **wire, "LLAMA_SPEC_COUPLED": "1"})
        bf16 = self.serve.make_plan(self.serve.arguments(
            ["--model", "/models/target.gguf", "--profile", "tensor-4", "--reduction-wire", "bf16"]))[1]
        self.assertEqual(bf16, {**two, **four, "LLAMA_SPEC_COUPLED": "1"})
        # the coupled draws are the MTP head's: neither without a draft nor with the DFlash one
        for options in (["--mtp-depth", "off"], ["--dflash", "/models/dflash.gguf", "--dflash-depth", "15"]):
            other = self.serve.make_plan(self.serve.arguments(
                ["--model", "/models/target.gguf", "--profile", "tensor-4"] + options))[1]
            self.assertEqual(other, {**two, **four, **wire})

    def test_four_card_tensor_recipe_takes_its_sessions_pool_and_drafter(self):
        base = ["--model", "/models/target.gguf", "--profile", "tensor-4"]
        for depth, adaptive in (("3", "0"), ("4", "0.18")):
            argv, _ = self.serve.make_plan(self.serve.arguments(
                base + ["--slots", "36", "--context", "294912", "--mtp-depth", depth, "--mtp-adaptive", adaptive,
                        "--backend-sampling"]))
            for flag, value in {"-np": "36", "-c": "294912", "--spec-draft-n-max": depth,
                                "--spec-draft-mtp-adaptive": adaptive}.items():
                self.assertEqual(argv[argv.index(flag) + 1], value)
            self.assertIn("--backend-sampling", argv)
        argv, _ = self.serve.make_plan(self.serve.arguments(base + ["--dflash", "/models/draft.gguf", "--dflash-depth", "7"]))
        for flag, value in {"--spec-type": "draft-dflash", "--spec-draft-model": "/models/draft.gguf",
                            "--spec-draft-n-max": "7", "--spec-draft-device": "CUDA0", "--spec-draft-ngl": "99"}.items():
            self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertFalse(any(arg.startswith("--spec-draft-mtp") for arg in argv))
        # a session's limit is the engine's own switch, set only when asked for
        self.assertNotIn("LLAMA_KV_SEQ_CELLS", self.serve.make_plan(self.serve.arguments(base))[1])
        argv, env = self.serve.make_plan(self.serve.arguments(base + ["--context", "3014656", "--slot-context", "262144"]))
        self.assertEqual(argv[argv.index("-c") + 1], "3014656")
        self.assertEqual(env["LLAMA_KV_SEQ_CELLS"], "262144")
        # graphs sized for long sessions are kept 32 a context, not 128, unless a count is given; four captures beyond
        self.assertEqual((env["LLAMA_GRAPH_SLOTS"], env["GGML_META_CAPTURE_GRAPHS"]), ("32", "36"))
        for options, kept in ((["--kept-graphs", "64"], "64"),
                              (["--context", "3014656", "--slot-context", "262144", "--kept-graphs", "128"], "128")):
            env = self.serve.make_plan(self.serve.arguments(base + options))[1]
            self.assertEqual((env["LLAMA_GRAPH_SLOTS"], env["GGML_META_CAPTURE_GRAPHS"]), (kept, str(int(kept) + 4)))
        for options in (base + ["--slots", "0"], base + ["--slots", "257"], base + ["--slots", "8", "--context", "7"],
                        base + ["--kept-graphs", "0"], base + ["--kept-graphs", "257"],
                        ["--model", "/models/target.gguf", "--kept-graphs", "32"],
                        ["--model", "/models/target.gguf", "--profile", "dp-4", "--kept-graphs", "32"],
                        base + ["--slot-context", "255"], base + ["--slot-context", "524288"],
                        base + ["--context", "294912", "--slot-context", "294912"],
                        ["--model", "/models/target.gguf", "--slot-context", "262144"],
                        ["--model", "/models/target.gguf", "--profile", "dp-4", "--slot-context", "262144"],
                        base + ["--dflash", "/models/draft.gguf"], base + ["--dflash-depth", "7"],
                        base + ["--mtp-adaptive", "1.5"],
                        ["--model", "/models/target.gguf", "--context", "294912"],
                        ["--model", "/models/target.gguf", "--backend-sampling"],
                        ["--model", "/models/target.gguf", "--reduction-wire", "q8"],
                        ["--model", "/models/target.gguf", "--profile", "dp-4", "--reduction-wire", "bf16"],
                        ["--model", "/models/target.gguf", "--dflash", "/models/draft.gguf", "--dflash-depth", "7"],
                        ["--model", "/models/target.gguf", "--profile", "dp-4", "--mtp-depth", "3"]):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.serve.make_plan(self.serve.arguments(options))

    def test_dp_slot_defaults_and_bounds_do_not_change_tensor_capacity(self):
        args = self.serve.arguments(["--model", "/models/target.gguf", "--profile", "dp-4"])
        argv, _ = self.serve.make_plan(args)
        self.assertEqual(argv[argv.index("-np") + 1], "36")
        for options in (["--slots", "4"], ["--profile", "dp-4", "--slots", "0"],
                        ["--profile", "dp-4", "--slots", "41"]):
            with self.subTest(options=options):
                args = self.serve.arguments(["--model", "/models/target.gguf", *options])
                with self.assertRaisesRegex(ValueError, "slots"):
                    self.serve.make_plan(args)

    def test_depth_and_responsiveness_options_are_independent(self):
        for depth, query in itertools.product(["1", "2", "off"], ["q8", "f32"]):
            args = self.serve.arguments(["--model", "/models/target.gguf", "--mtp-depth", depth,
                                         "--indexer-query", query,
                                         "--prompt-step", "0", "--host", "0.0.0.0", "--port", "8090"])
            argv, env = self.serve.make_plan(args)
            self.assertEqual(env["LLAMA_INDEXER_Q8_QUERY"], "1" if query == "q8" else "0")
            self.assertNotIn("--mmproj", argv)
            self.assertEqual(argv[argv.index("--prompt-step") + 1], "0")
            self.assertEqual(argv[argv.index("--host") + 1], "0.0.0.0")
            self.assertEqual(argv[argv.index("--port") + 1], "8090")
            if depth == "off":
                self.assertFalse(any(arg.startswith("--spec-") for arg in argv))
            else:
                self.assertEqual(argv[argv.index("--spec-draft-n-max") + 1], depth)

    def test_diagnostic_microbatch_is_explicit_and_does_not_change_the_serving_baseline(self):
        baseline, _ = self.serve.make_plan(self.serve.arguments(["--model", "/models/target.gguf", "--profile", "dp-4"]))
        self.assertEqual(baseline[baseline.index("-ub") + 1], "512")
        for ubatch in (128, 512):
            with self.subTest(ubatch=ubatch):
                diagnostic, _ = self.serve.make_plan(self.serve.arguments([
                    "--model", "/models/target.gguf", "--profile", "dp-4", "--diagnostic-ubatch", str(ubatch)]))
                self.assertEqual(diagnostic[diagnostic.index("-ub") + 1], str(ubatch))

    def test_host_capacity_requires_mtp_and_a_sufficient_ram_budget(self):
        base = ["--model", "/model.gguf", "--profile", "dp-4", "--kv-host", "--slots", "72"]
        args = self.serve.arguments(base + ["--host-budget-mib", "200000"])
        argv, env = self.serve.make_plan(args)
        self.assertEqual(env["LLAMA_KV_HOST"], "1")
        self.assertEqual(argv[argv.index("-c") + 1], str(72 * 442368))
        self.assertEqual(argv[argv.index("--spec-draft-n-max") + 1], "2")
        self.assertNotIn("--kv-unified", argv)
        for extra in ([], ["--host-budget-mib", "198287"], ["--host-budget-mib", "nan"],
                      ["--host-budget-mib", "200000", "--mtp-depth", "off"],
                      ["--host-budget-mib", "200000", "--profile", "tensor-2"]):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.serve.make_plan(self.serve.arguments(base + extra))

    def test_cgroup_v1_memory_budget_is_honored(self):
        files = {
            "/proc/meminfo": "MemAvailable: 104857600 kB\n",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes": str(50 * 1024**3),
            "/sys/fs/cgroup/memory/memory.usage_in_bytes": str(10 * 1024**3),
        }

        class FakePath:
            def __init__(self, path):
                self.path = path

            def is_file(self):
                return self.path in files

            def read_text(self):
                return files[self.path]

            def with_name(self, name):
                from pathlib import PurePosixPath
                return FakePath(str(PurePosixPath(self.path).with_name(name)))

        with mock.patch.object(self.serve, "Path", side_effect=FakePath):
            self.assertEqual(self.serve.available_memory_mib(), 40960)

    def test_host_capacity_launch_refuses_insufficient_available_ram(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            model.touch()
            args = self.serve.arguments(["--model", str(model), "--server", sys.executable,
                "--profile", "dp-4", "--kv-host", "--slots", "72", "--host-budget-mib", "200000"])
            with mock.patch.object(self.serve.os, "execve") as execute, \
                    mock.patch.object(self.serve, "available_memory_mib", return_value=230000):
                with self.assertRaisesRegex(ValueError, "RAM"):
                    self.serve.main(args)
                execute.assert_not_called()

    def test_hidden_engine_settings_do_not_override_the_recipe(self):
        args = self.serve.arguments(["--model", "/models/target.gguf"])
        _, settings = self.serve.make_plan(args)
        inherited = {"PATH": "/bin", "HOME": "/home/operator", "GGML_CUDA_Q8_TMA": "1",
                     "LLAMA_ARG_GLM53_PROFILE": "sm120-balanced", "LLAMA_ARG_MODEL": "wrong.gguf",
                     "LLAMA_KPOOL_DIRECT_INDICES": "1", "CUDA_LAUNCH_BLOCKING": "1",
                     "CUDA_VISIBLE_DEVICES": "9", "LD_LIBRARY_PATH": "/wrong/build"}
        actual = self.serve.execution_environment(settings, inherited)
        self.assertEqual(actual, {"PATH": "/bin", "HOME": "/home/operator", **settings})

    def test_model_paths_keep_their_own_names_through_symlinks(self):
        # A Hugging Face cache snapshot links the first shard to a blob whose name carries no split suffix; the loader
        # finds shards 2 and 3 from the first shard's name, so the launcher must pass the link's path, not its target.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            blob = root / "blobs" / "9916f01db53f8a2bfd5db1d6eb8cd48f93ae2b732fe1dfce993450745334cae1"
            blob.parent.mkdir()
            blob.write_bytes(b"GGUF")
            shard = root / "snapshots" / "1bc0afad" / "IQ3_XXS" / "GLM-5.3-Flash-Uncensored-IQ3_XXS-00001-of-00003.gguf"
            shard.parent.mkdir(parents=True)
            shard.symlink_to(blob)
            projector = root / "snapshots" / "1bc0afad" / "mmproj-GLM-5.3-Flash-Uncensored-F16.gguf"
            projector.symlink_to(blob)
            args = self.serve.arguments(["--model", str(shard), "--mmproj", str(projector)])
            argv, _ = self.serve.make_plan(args)
            self.assertEqual(argv[argv.index("-m") + 1], str(shard))
            self.assertEqual(argv[argv.index("--mmproj") + 1], str(projector))
            relative = self.serve.arguments(["--model", "models/first.gguf"])
            argv, _ = self.serve.make_plan(relative)
            self.assertEqual(argv[argv.index("-m") + 1], str(Path.cwd() / "models/first.gguf"))

    def test_server_access_controls_survive_the_engine_scrub(self):
        # The scrub removes engine, profile and diagnostic overrides; the server's own key and TLS settings stay.
        args = self.serve.arguments(["--model", "/models/target.gguf"])
        _, settings = self.serve.make_plan(args)
        kept = {"LLAMA_API_KEY": "secret", "LLAMA_ARG_API_KEY_FILE": "/etc/llama/keys",
                "LLAMA_ARG_SSL_KEY_FILE": "/etc/llama/key.pem", "LLAMA_ARG_SSL_CERT_FILE": "/etc/llama/cert.pem"}
        inherited = {**kept, "PATH": "/bin", "LLAMA_ARG_MODEL": "wrong.gguf", "LLAMA_KPOOL_DIRECT_INDICES": "1",
                     "LLAMA_INDEXER_Q8_QUERY": "0", "GGML_CUDA_Q8_TMA": "1", "CUDA_VISIBLE_DEVICES": "9",
                     "NVIDIA_TF32_OVERRIDE": "0"}
        self.assertEqual(self.serve.execution_environment(settings, inherited), {**kept, "PATH": "/bin", **settings})
        self.assertEqual(self.serve.removed_overrides(inherited),
                         ["CUDA_VISIBLE_DEVICES", "GGML_CUDA_Q8_TMA", "LLAMA_ARG_MODEL", "LLAMA_INDEXER_Q8_QUERY",
                          "LLAMA_KPOOL_DIRECT_INDICES", "NVIDIA_TF32_OVERRIDE"])

    def test_dry_run_names_the_inherited_overrides_it_removes(self):
        args = self.serve.arguments(["--model", "/missing/model.gguf", "--server", "/missing/server", "--dry-run"])
        output = io.StringIO()
        with mock.patch.dict(self.serve.os.environ, {"LLAMA_KPOOL_DIRECT_INDICES": "1", "LLAMA_API_KEY": "secret"}), \
                mock.patch.object(self.serve.os, "execve") as execute, contextlib.redirect_stdout(output):
            self.assertEqual(self.serve.main(args), 0)
        execute.assert_not_called()
        plan = json.loads(output.getvalue())
        self.assertIn("LLAMA_KPOOL_DIRECT_INDICES", plan["removed"])
        self.assertNotIn("LLAMA_API_KEY", plan["removed"])
        self.assertNotIn("secret", output.getvalue())

    def test_dry_run_never_executes_and_does_not_require_model_downloads(self):
        args = self.serve.arguments(["--model", "/missing/model;literal.gguf", "--server", "/missing/server with spaces",
                                     "--dry-run"])
        output = io.StringIO()
        with mock.patch.object(self.serve.os, "execve") as execute, contextlib.redirect_stdout(output):
            self.assertEqual(self.serve.main(args), 0)
        execute.assert_not_called()
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["argv"][0], "/missing/server with spaces")
        self.assertEqual(plan["argv"][2], "/missing/model;literal.gguf")

    def test_real_launch_checks_inputs_and_executes_without_a_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model;literal.gguf"
            args = self.serve.arguments(["--model", str(model), "--server", sys.executable])
            with mock.patch.object(self.serve.os, "execve") as execute:
                with self.assertRaisesRegex(ValueError, "model"):
                    self.serve.main(args)
                execute.assert_not_called()
                model.touch()
                self.serve.main(args)
                executable, argv, env = execute.call_args.args
                self.assertEqual(executable, str(Path(sys.executable).resolve()))
                self.assertEqual(argv[2], str(model))
                self.assertEqual(env["LLAMA_KPOOL_DIRECT_INDICES"], "0")

    def test_published_assets_have_the_pinned_bytes_and_vocabulary_shape(self):
        self.serve.verify_assets()
        data = self.serve.DRAFT_VOCAB.read_bytes()
        self.assertEqual(len(data), 393216)
        ids = struct.unpack("<98304i", data)
        self.assertEqual(len(set(ids)), 98304)
        self.assertTrue(set(range(256)).issubset(ids))
        self.assertTrue(set(range(154820, 154856)).issubset(ids))
        self.assertTrue(all(0 <= token <= 154855 for token in ids))


class CapacityTest(unittest.TestCase):
    def setUp(self):
        from capacity import estimate
        self.estimate = estimate
        self.lower = [dict(device=d, used_mib=used, free_mib=97342-used)
                      for d, used in enumerate([90510, 88876, 88876, 88876])]
        self.upper = [dict(device=d, used_mib=used, free_mib=97342-used)
                      for d, used in enumerate([95798, 94164, 94164, 94164])]

    def test_measured_fixed_cost_and_mtp_cache_limit_every_card(self):
        resident = self.estimate(36, self.lower, 40, self.upper)
        self.assertEqual(resident['estimated_max']['slots'], 36)
        host = self.estimate(36, self.lower, 40, self.upper, host_kv=True, host_budget_mib=200000)
        self.assertEqual(host['estimated_max']['slots'], 72)
        self.assertEqual(host['estimated_max']['host_kv_mib'], 198288)
        self.assertEqual(host['cache_mib_per_slot'], 4819.5)
        self.assertAlmostEqual(host['staging_mib_per_card'], 1321.8359375)
        self.assertFalse(host['capacity_qualified'])
        self.assertFalse(host['quality_qualified'])
        limited = self.estimate(36, self.lower, 40, self.upper, host_kv=True, host_budget_mib=50000)
        self.assertEqual(limited['estimated_max']['slots'], 18)
        self.assertEqual(limited['estimated_max']['slots_per_card'], [5, 5, 4, 4])

    def test_bad_calibration_cannot_produce_a_capacity_claim(self):
        for field, value in [('peak_extra_mib', float('nan')), ('reserve_mib', float('inf')),
                             ('max_slots', 257)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.estimate(36, self.lower, 40, self.upper, **{field: value})
        changed = copy.deepcopy(self.upper)
        changed[0]['free_mib'] += 1024
        with self.assertRaises(ValueError):
            self.estimate(36, self.lower, 40, changed)
        with self.assertRaises(ValueError):
            self.estimate(36, self.lower, 40, self.upper, host_kv=True)


if __name__ == "__main__":
    unittest.main()
