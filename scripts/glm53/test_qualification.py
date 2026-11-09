"""The paid window must use pinned bytes, matched serving and real device coverage."""
import copy
import contextlib
import hashlib
import io
import itertools
import json
import re
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qualification import (BASELINE_COMMIT, device_cases, check_run_identity, check_quality, check_stop_at,
                           verify_files, validate_arch_table, validate_device_csv)


class QualificationTest(unittest.TestCase):
    def test_collection_status_cannot_hide_missing_or_failed_profile(self):
        import qualification
        cases = [
            (False, False, None, "not_requested"),
            (True, True, None, "invalid"),
            (True, False, None, "invalid"),
            (True, False, {"start": {"exit": 7}}, "start_failed"),
            (True, False, {"start": {"exit": 0}, "finalize": {"report_present": False}}, "invalid"),
            (True, False, {"start": {"exit": 0}, "finalize": {"report_present": True}}, "exported_unverified"),
        ]
        for requested, running, record, expected in cases:
            with self.subTest(expected=expected, record=record):
                result = qualification.profile_evidence(requested, running, record)
                self.assertEqual(result["state"], expected)
                self.assertFalse(result["capture_verified"])

    def test_matrix_success_requires_every_planned_leg_and_quality_check(self):
        import qualification
        good = dict(plan=[["B","2",[[36,512]],[]]], servers=[
            dict(short_ok=True, runs=[dict(active=36, context=512, exit=0, passed=True,
                                           profile={"state": "not_requested"})])])
        self.assertTrue(qualification.matrix_succeeded(good))
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

    def exercise_controller_failure(self, profiled, seconds_left):
        import runpy
        import subprocess
        import time
        from types import SimpleNamespace
        original_popen = subprocess.Popen

        class ServerProcess:
            pid = 12345
            returncode = None
            def poll(self):
                return self.returncode
            def wait(self, timeout=None):
                self.returncode = 0
                return 0

        def launch(command, **kwargs):
            if any(Path(arg).name == "llama-server" for arg in command):
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

    def test_dp_capacity_recipe_keeps_each_slots_cache_on_one_owner(self):
        for slots in (1, 4, 36, 40):
            for depth in ("2", "off"):
                with self.subTest(slots=slots, depth=depth):
                    args = self.serve.arguments(["--model", "/models/target.gguf", "--profile", "dp-4",
                                                 "--slots", str(slots), "--mtp-depth", depth])
                    argv, env = self.serve.make_plan(args)
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

    def test_four_card_tensor_recipe_shares_one_pool_on_the_four_bit_cache(self):
        args = self.serve.arguments(["--model", "/models/target.gguf", "--profile", "tensor-4"])
        argv, env = self.serve.make_plan(args)
        expected = {"-sm": "tensor", "-ts": "1,1,1,1", "-np": "23", "-c": "524288", "-b": "2048", "-ub": "2048",
                    "--prompt-step": "1024", "--cache-type-k": "q4_0", "--cache-type-v": "q4_0", "-ctki": "q8_0",
                    "--spec-type": "draft-mtp", "--spec-draft-n-max": "2", "--spec-draft-mtp-adaptive": "0.18"}
        for flag, value in expected.items():
            self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertIn("--kv-unified", argv)
        for flag in ("--backend-sampling", "-fit", "-cts", "-ubd", "--spec-draft-model"):
            self.assertNotIn(flag, argv)
        two = self.serve.make_plan(self.serve.arguments(["--model", "/models/target.gguf"]))[1]
        self.assertEqual(env, {**two, "CUDA_VISIBLE_DEVICES": "0,1,2,3"})

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
        for options in (base + ["--slots", "0"], base + ["--slots", "257"], base + ["--slots", "8", "--context", "7"],
                        base + ["--dflash", "/models/draft.gguf"], base + ["--dflash-depth", "7"],
                        base + ["--mtp-adaptive", "1.5"],
                        ["--model", "/models/target.gguf", "--context", "294912"],
                        ["--model", "/models/target.gguf", "--backend-sampling"],
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
