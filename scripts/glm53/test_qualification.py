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
        for key in ("GGML_CUDA_ROUTE_STATS", "GGML_CUDA_MMVQ_Q8_SM120_CHECK", "GGML_CUDA_Q8_TMA",
                    "CUDA_LAUNCH_BLOCKING", "NVIDIA_TF32_OVERRIDE"):
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
            for slots in (2, 4, 6, 8):
                self.assertEqual(counts[f"iq3-s{slots}-gpu{gpu}"], 81)
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
                self.assertEqual(case["cases"], 81)
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

    def test_default_two_card_tensor_recipe_carries_the_four_card_settings(self):
        args = self.serve.arguments(["--model", "/models/first shard.gguf", "--mmproj", "/models/projector.gguf"])
        argv, env = self.serve.make_plan(args)
        self.assertEqual(argv, [str(self.serve.ROOT / "build/bin/llama-server"),
            "-m", "/models/first shard.gguf", "--mmproj", "/models/projector.gguf",
            "-ngl", "99", "-sm", "tensor", "-ts", "1,1", "--jinja", "-fa", "on",
            "--cache-type-k", "q4_0", "--cache-type-v", "q4_0", "-ctki", "q8_0",
            "-b", "2048", "-ub", "2048", "--prompt-step", "1024",
            "--alias", "GLM-5.3-Flash-Uncensored-IQ3_XXS", "--chat-template-file", str(self.serve.TEMPLATE),
            "--prompt-chunk", "128",
            "--spec-type", "draft-mtp", "--spec-draft-n-max", "2",
            "--spec-draft-mtp-vocab", str(self.serve.DRAFT_VOCAB), "--spec-draft-mtp-adaptive", "0.18",
            "-c", "1843200", "-np", "12", "--kv-unified", "--cache-ram", "32768",
            "--no-cache-idle-slots", "--metrics", "--host", "127.0.0.1", "--port", "8102",
            "--log-timestamps"])
        self.assertEqual(env, {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0,1",
            "GGML_CUDA_GRAPH_MAX": "0", "GGML_CUDA_ALLREDUCE": "internal",
            "LLAMA_INDEXER_Q8_QUERY": "1", "LLAMA_MTP_DEVICE_H": "1", "LLAMA_TOK_EMBD_GPU": "1",
            "GGML_CUDA_MMQ_MOE_NCOLS": "0", "GGML_CUDA_MMQ_L2_PF": "1", "GGML_SCHED_DEBUG": "0",
            "GGML_CUDA_AR_WIRE_Q8": "1", "LLAMA_GRAPH_ANY_SEQS": "1", "LLAMA_GRAPH_SLOTS": "32",
            "GGML_META_CAPTURE_GRAPHS": "36", "GGML_CUDA_GRAPH_NODES": "262144",
            "GGML_CUDA_GRAPH_IDLE_MS": "600000", "LLAMA_GRAPH_DEVICE_RESERVE_MIB": "1024",
            "LLAMA_SPEC_COUPLED": "1", "LLAMA_KV_SEQ_CELLS": "307200",
            "LLAMA_STATE_SAVE_ASYNC": "1", "LLAMA_STATE_RESTORE_ASYNC": "1",
            "LD_LIBRARY_PATH": str(self.serve.ROOT / "build/bin")})
        self.assertNotIn("--glm53-profile", argv)

    def test_the_dp_profiles_are_gone(self):
        # production runs the four-card tensor layout and every read since November 9 ran on it; the DP
        # profiles were removed on November 16, 2026
        for profile in ("dp-2", "dp-4"):
            with self.subTest(profile=profile), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                self.serve.arguments(["--model", "/models/target.gguf", "--profile", profile])

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
        # the reductions on the copy engines through host memory in 8-bit blocks, graphs kept by shape, CUDA
        # executables held between turns, saves and restores beside the steps, the coupled MTP draws
        self.assertEqual(env, {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "GGML_CUDA_GRAPH_MAX": "0", "GGML_CUDA_ALLREDUCE": "internal",
            "LLAMA_INDEXER_Q8_QUERY": "1", "LLAMA_MTP_DEVICE_H": "1", "LLAMA_TOK_EMBD_GPU": "1",
            "GGML_CUDA_MMQ_MOE_NCOLS": "0", "GGML_CUDA_MMQ_L2_PF": "1", "GGML_SCHED_DEBUG": "0",
            "GGML_CUDA_AR_LL_MAX_BYTES": "0", "GGML_CUDA_AR_P2P_MAX_BYTES": "0", "GGML_CUDA_AR_COPY_THRESHOLD": "1",
            "GGML_CUDA_AR_COPY_PEER": "0", "GGML_CUDA_AR_GATHER_OWNER_SLICES": "0", "GGML_CUDA_AR_WIRE_Q8": "1",
            "LLAMA_GRAPH_ANY_SEQS": "1", "LLAMA_GRAPH_SLOTS": "128", "GGML_META_CAPTURE_GRAPHS": "132",
            "GGML_CUDA_GRAPH_NODES": "262144", "GGML_CUDA_GRAPH_IDLE_MS": "600000",
            "LLAMA_SPEC_COUPLED": "1", "LLAMA_STATE_SAVE_ASYNC": "1", "LLAMA_STATE_RESTORE_ASYNC": "1",
            "LD_LIBRARY_PATH": str(self.serve.ROOT / "build/bin")})
        # two cards serve the same settings; what differs is the devices in sight, the host-memory reduction route
        # no pair has been read on, the kept graphs a session limit sizes, the session limit itself and the fuller
        # cards' graph reserve
        two = self.serve.make_plan(self.serve.arguments(["--model", "/models/target.gguf"]))[1]
        self.assertEqual({key for key in set(env) | set(two) if env.get(key) != two.get(key)}, {
            "CUDA_VISIBLE_DEVICES", "GGML_CUDA_AR_LL_MAX_BYTES", "GGML_CUDA_AR_P2P_MAX_BYTES",
            "GGML_CUDA_AR_COPY_THRESHOLD", "GGML_CUDA_AR_COPY_PEER", "GGML_CUDA_AR_GATHER_OWNER_SLICES",
            "LLAMA_GRAPH_SLOTS", "GGML_META_CAPTURE_GRAPHS", "LLAMA_GRAPH_DEVICE_RESERVE_MIB",
            "LLAMA_KV_SEQ_CELLS"})
        bf16 = self.serve.make_plan(self.serve.arguments(
            ["--model", "/models/target.gguf", "--profile", "tensor-4", "--reduction-wire", "bf16"]))[1]
        self.assertEqual(bf16, {key: value for key, value in env.items() if key != "GGML_CUDA_AR_WIRE_Q8"})
        # the coupled draws are the MTP head's: neither without a draft nor with the DFlash one
        for options in (["--mtp-depth", "off"], ["--dflash", "/models/dflash.gguf", "--dflash-depth", "15"]):
            other = self.serve.make_plan(self.serve.arguments(
                ["--model", "/models/target.gguf", "--profile", "tensor-4"] + options))[1]
            self.assertEqual(other, {key: value for key, value in env.items() if key != "LLAMA_SPEC_COUPLED"})

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
                        base + ["--slot-context", "255"], base + ["--slot-context", "524288"],
                        base + ["--context", "294912", "--slot-context", "294912"],
                        ["--model", "/models/target.gguf", "--context", "400000", "--slot-context", "400000"],
                        base + ["--dflash", "/models/draft.gguf"], base + ["--dflash-depth", "7"],
                        base + ["--mtp-adaptive", "1.5"]):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.serve.make_plan(self.serve.arguments(options))

    def test_two_card_tensor_recipe_sizes_its_sessions_at_300k(self):
        base = ["--model", "/models/target.gguf"]
        # the plain recipe: twelve sessions reserved against a pool of six whole 307,200-token sessions, as the
        # four-card production server reserves 23 against a pool far below 23 whole sessions: decode is bounded
        # by the requests decoding at once, not by the slots reserved (Marcos, November 16, 2026)
        argv, env = self.serve.make_plan(self.serve.arguments(base))
        self.assertEqual((argv[argv.index("-np") + 1], argv[argv.index("-c") + 1]), ("12", "1843200"))
        self.assertEqual(env["LLAMA_KV_SEQ_CELLS"], "307200")
        self.assertEqual((env["LLAMA_GRAPH_SLOTS"], env["GGML_META_CAPTURE_GRAPHS"]), ("32", "36"))
        # the pool is sized once, from the cards; the reserved count and the limit move without resizing it
        argv, env = self.serve.make_plan(self.serve.arguments(base + ["--slots", "7"]))
        self.assertEqual((argv[argv.index("-np") + 1], argv[argv.index("-c") + 1]), ("7", "1843200"))
        self.assertEqual(env["LLAMA_KV_SEQ_CELLS"], "307200")
        argv, env = self.serve.make_plan(self.serve.arguments(base + ["--slot-context", "262144"]))
        self.assertEqual((argv[argv.index("-np") + 1], argv[argv.index("-c") + 1]), ("12", "1843200"))
        self.assertEqual(env["LLAMA_KV_SEQ_CELLS"], "262144")
        argv, env = self.serve.make_plan(self.serve.arguments(base + ["--slots", "1"]))
        self.assertEqual((argv[argv.index("-np") + 1], argv[argv.index("-c") + 1]), ("1", "1843200"))
        self.assertEqual(env["LLAMA_KV_SEQ_CELLS"], "307200")
        self.assertEqual(env["LLAMA_GRAPH_SLOTS"], "32")
        # a pool given without a limit leaves its sessions uncapped, as the four cards' pool does
        argv, env = self.serve.make_plan(self.serve.arguments(base + ["--context", "524288"]))
        self.assertEqual((argv[argv.index("-np") + 1], argv[argv.index("-c") + 1]), ("12", "524288"))
        self.assertNotIn("LLAMA_KV_SEQ_CELLS", env)
        # the two cards take the four cards' own sizing when it is asked for
        argv, env = self.serve.make_plan(self.serve.arguments(
            base + ["--slots", "23", "--context", "3014656", "--slot-context", "262144", "--kept-graphs", "64"]))
        self.assertEqual((argv[argv.index("-np") + 1], argv[argv.index("-c") + 1]), ("23", "3014656"))
        self.assertEqual((env["LLAMA_KV_SEQ_CELLS"], env["LLAMA_GRAPH_SLOTS"]), ("262144", "64"))
        self.assertEqual(env["GGML_META_CAPTURE_GRAPHS"], "68")

    def test_reduction_route_and_latent_cache_are_tensor_choices(self):
        route = {"GGML_CUDA_AR_LL_MAX_BYTES", "GGML_CUDA_AR_P2P_MAX_BYTES", "GGML_CUDA_AR_COPY_THRESHOLD",
                 "GGML_CUDA_AR_COPY_PEER", "GGML_CUDA_AR_GATHER_OWNER_SLICES"}
        # two cards keep the engine's own routes until a pair is read; the host route is a launcher option there
        env = self.serve.make_plan(self.serve.arguments(["--model", "/models/target.gguf"]))[1]
        self.assertFalse(route & set(env))
        env = self.serve.make_plan(self.serve.arguments(
            ["--model", "/models/target.gguf", "--reduction-route", "host"]))[1]
        self.assertTrue(route <= set(env))
        env = self.serve.make_plan(self.serve.arguments(
            ["--model", "/models/target.gguf", "--reduction-route", "engine"]))[1]
        self.assertFalse(route & set(env))
        self.assertEqual(env["GGML_CUDA_AR_WIRE_Q8"], "1")
        # four cards keep the host route by default and can be read on the engine's routes against it
        env = self.serve.make_plan(self.serve.arguments(
            ["--model", "/models/target.gguf", "--profile", "tensor-4", "--reduction-route", "engine"]))[1]
        self.assertFalse(route & set(env))
        # the cache the two cards served with before the four-card settings stays one flag away
        argv = self.serve.make_plan(self.serve.arguments(
            ["--model", "/models/target.gguf", "--latent-cache", "q8_0"]))[0]
        self.assertEqual((argv[argv.index("--cache-type-k") + 1], argv[argv.index("--cache-type-v") + 1]),
                         ("q8_0", "q8_0"))

    def test_prompt_step_ms_is_an_opt_in_and_its_step_is_the_most_a_step_takes(self):
        base = ["--model", "/models/target.gguf", "--profile", "tensor-4"]
        self.assertNotIn("--prompt-step-ms", self.serve.make_plan(self.serve.arguments(base))[0])
        argv = self.serve.make_plan(self.serve.arguments(base + ["--prompt-step-ms", "40"]))[0]
        self.assertEqual((argv[argv.index("--prompt-step-ms") + 1], argv[argv.index("--prompt-step") + 1]), ("40", "1024"))
        # with shapes the step is one pass; the time takes the count there, so the step is not held to the pass
        argv = self.serve.make_plan(self.serve.arguments(base + ["--prompt-shapes", "8", "--prompt-step-ms", "40"]))[0]
        self.assertEqual((argv[argv.index("--prompt-step-ms") + 1], argv[argv.index("--prompt-step") + 1]), ("40", "1024"))
        argv = self.serve.make_plan(self.serve.arguments(
            base + ["--prompt-shapes", "8", "--prompt-step-ms", "40", "--prompt-step", "512"]))[0]
        self.assertEqual(argv[argv.index("--prompt-step") + 1], "512")
        for ms in ("0", "-5"):
            with self.subTest(ms=ms), self.assertRaises(ValueError):
                self.serve.make_plan(self.serve.arguments(base + ["--prompt-step-ms", ms]))

    def test_prompt_shapes_session_step_and_low_depth_are_opt_ins(self):
        base = ["--model", "/models/target.gguf", "--profile", "tensor-4"]
        argv, env = self.serve.make_plan(self.serve.arguments(base))
        for key in ("LLAMA_PROMPT_SHAPES", "LLAMA_PROMPT_SHAPE_SMALL", "LLAMA_PROMPT_SHAPE_ROWS", "LLAMA_KV_PAD_SHARED"):
            self.assertNotIn(key, env)
        self.assertNotIn("--spec-draft-n-min", argv)
        self.assertEqual(argv[argv.index("--prompt-step") + 1], "1024")
        # 32 is the engine's switch as it was; a smaller count names the rows a short part keeps whole
        argv, env = self.serve.make_plan(self.serve.arguments(base + ["--prompt-shapes", "32"]))
        self.assertEqual(env["LLAMA_PROMPT_SHAPES"], "1")
        self.assertNotIn("LLAMA_PROMPT_SHAPE_SMALL", env)
        self.assertNotIn("LLAMA_PROMPT_SHAPE_ROWS", env)
        # with shapes a step takes one prompt pass, the largest the shapes make, unless the step is given
        self.assertEqual(argv[argv.index("--prompt-step") + 1], "256")
        argv = self.serve.make_plan(self.serve.arguments(base + ["--prompt-shapes", "8", "--prompt-step", "1024"]))[0]
        self.assertEqual(argv[argv.index("--prompt-step") + 1], "1024")
        argv, env = self.serve.make_plan(self.serve.arguments(
            base + ["--prompt-shapes", "8", "--prompt-shape-rows", "1024", "--session-step", "65536",
                    "--mtp-depth", "3", "--mtp-depth-min", "2"]))
        self.assertEqual((env["LLAMA_PROMPT_SHAPES"], env["LLAMA_PROMPT_SHAPE_SMALL"], env["LLAMA_PROMPT_SHAPE_ROWS"]),
                         ("1", "8", "1024"))
        self.assertEqual(argv[argv.index("--prompt-step") + 1], "1024")
        self.assertEqual(env["LLAMA_KV_PAD_SHARED"], "65536")
        self.assertEqual((argv[argv.index("--spec-draft-n-max") + 1], argv[argv.index("--spec-draft-n-min") + 1]), ("3", "2"))
        # the two-card tensor recipe takes the shapes, the low depth and the step; the step is the shared pool's
        argv, env = self.serve.make_plan(self.serve.arguments(
            ["--model", "/models/target.gguf", "--prompt-shapes", "8", "--mtp-depth-min", "2",
             "--session-step", "65536"]))
        self.assertEqual((env["LLAMA_PROMPT_SHAPES"], env["LLAMA_PROMPT_SHAPE_SMALL"]), ("1", "8"))
        self.assertEqual(env["LLAMA_KV_PAD_SHARED"], "65536")
        self.assertEqual(argv[argv.index("--spec-draft-n-min") + 1], "2")
        for options in (base + ["--prompt-shape-rows", "1024"],
                        base + ["--mtp-depth", "2", "--mtp-depth-min", "3"],
                        base + ["--mtp-depth", "off", "--mtp-depth-min", "1"],
                        base + ["--dflash", "/models/draft.gguf", "--dflash-depth", "7", "--mtp-depth-min", "1"]):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.serve.make_plan(self.serve.arguments(options))

    def test_prompt_cache_size_checkpoint_count_and_prefix_copy_are_launcher_options(self):
        base = ["--model", "/models/target.gguf", "--profile", "tensor-4"]
        argv, env = self.serve.make_plan(self.serve.arguments(base))
        self.assertEqual(argv[argv.index("--cache-ram") + 1], "32768")
        self.assertIn("--no-cache-idle-slots", argv)
        self.assertNotIn("--ctx-checkpoints", argv)
        self.assertNotIn("LLAMA_SLOT_PREFIX_COPY", env)
        argv, env = self.serve.make_plan(self.serve.arguments(
            base + ["--cache-ram", "65536", "--ctx-checkpoints", "8", "--prefix-copy"]))
        self.assertEqual((argv[argv.index("--cache-ram") + 1], argv[argv.index("--ctx-checkpoints") + 1]), ("65536", "8"))
        self.assertEqual(env["LLAMA_SLOT_PREFIX_COPY"], "1")
        for rows in (0, 64):
            with self.subTest(rows=rows):
                env = self.serve.make_plan(self.serve.arguments(base + ["--checkpoint-rows", str(rows)]))[1]
                self.assertEqual(env["LLAMA_RS_STAGE_ROWS"], str(rows))
        # -1 (no limit) and 0 (no cache, no checkpoints) are the server's own values and go through as given
        for value in ("-1", "0"):
            argv = self.serve.make_plan(self.serve.arguments(base + ["--cache-ram", value, "--ctx-checkpoints", "0"]))[0]
            self.assertEqual((argv[argv.index("--cache-ram") + 1], argv[argv.index("--ctx-checkpoints") + 1]), (value, "0"))
        # the two-card tensor recipe shares one pool too
        self.assertEqual(self.serve.make_plan(self.serve.arguments(
            ["--model", "/models/target.gguf", "--prefix-copy"]))[1]["LLAMA_SLOT_PREFIX_COPY"], "1")
        # the off switches of the streams that share cells and of the victim rule reach the engine only as launcher options
        self.assertFalse({"LLAMA_KPOOL_SHARED_CELLS_LEGACY", "LLAMA_SLOT_VICTIM_LRU_LEGACY"} & set(env))
        env = self.serve.make_plan(self.serve.arguments(base + ["--shared-cells-legacy", "--victim-lru-legacy"]))[1]
        self.assertEqual((env["LLAMA_KPOOL_SHARED_CELLS_LEGACY"], env["LLAMA_SLOT_VICTIM_LRU_LEGACY"]), ("1", "1"))
        for options in (base + ["--cache-ram", "-2"], base + ["--ctx-checkpoints", "-1"],
                        base + ["--checkpoint-rows", "-1"], base + ["--checkpoint-rows", "65"]):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.serve.make_plan(self.serve.arguments(options))

    def test_tensor_recipes_save_and_restore_beside_the_steps_by_default(self):
        base = ["--model", "/models/target.gguf", "--profile", "tensor-4"]
        env = self.serve.make_plan(self.serve.arguments(base))[1]
        self.assertEqual((env["LLAMA_STATE_SAVE_ASYNC"], env["LLAMA_STATE_RESTORE_ASYNC"]), ("1", "1"))
        # the options a four-card launch line gave before they were the default stay accepted
        env = self.serve.make_plan(self.serve.arguments(base + ["--save-async", "--restore-async"]))[1]
        self.assertEqual((env["LLAMA_STATE_SAVE_ASYNC"], env["LLAMA_STATE_RESTORE_ASYNC"]), ("1", "1"))
        # each off switch brings one of the two back to the server's thread
        env = self.serve.make_plan(self.serve.arguments(base + ["--save-sync"]))[1]
        self.assertNotIn("LLAMA_STATE_SAVE_ASYNC", env)
        self.assertEqual(env["LLAMA_STATE_RESTORE_ASYNC"], "1")
        env = self.serve.make_plan(self.serve.arguments(base + ["--restore-sync"]))[1]
        self.assertNotIn("LLAMA_STATE_RESTORE_ASYNC", env)
        self.assertEqual(env["LLAMA_STATE_SAVE_ASYNC"], "1")
        # the two-card recipe defaults the same way, with the same off switches
        env = self.serve.make_plan(self.serve.arguments(["--model", "/models/target.gguf"]))[1]
        self.assertEqual((env["LLAMA_STATE_SAVE_ASYNC"], env["LLAMA_STATE_RESTORE_ASYNC"]), ("1", "1"))
        env = self.serve.make_plan(self.serve.arguments(["--model", "/models/target.gguf", "--save-sync"]))[1]
        self.assertNotIn("LLAMA_STATE_SAVE_ASYNC", env)
        self.assertEqual(env["LLAMA_STATE_RESTORE_ASYNC"], "1")
        for options in (base + ["--save-sync", "--save-async"], base + ["--restore-sync", "--restore-async"],
                        ["--model", "/models/target.gguf", "--save-sync", "--save-async"]):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.serve.make_plan(self.serve.arguments(options))

    def test_the_rule_for_the_session_that_is_cut_has_a_setting_and_two_switches(self):
        base = ["--model", "/models/target.gguf", "--profile", "tensor-4"]
        env = self.serve.make_plan(self.serve.arguments(base))[1]
        for key in ("LLAMA_SLOT_VICTIM_LRU_LEGACY", "LLAMA_SLOT_VICTIM_IDLE_LEGACY", "LLAMA_SLOT_VICTIM_RETURN_S"):
            self.assertNotIn(key, env)
        env = self.serve.make_plan(self.serve.arguments(base + ["--victim-return-s", "45"]))[1]
        self.assertEqual(env["LLAMA_SLOT_VICTIM_RETURN_S"], "45")
        env = self.serve.make_plan(self.serve.arguments(base + ["--victim-idle-legacy"]))[1]
        self.assertEqual(env["LLAMA_SLOT_VICTIM_IDLE_LEGACY"], "1")
        self.assertNotIn("LLAMA_SLOT_VICTIM_LRU_LEGACY", env)
        env = self.serve.make_plan(self.serve.arguments(base + ["--victim-lru-legacy"]))[1]
        self.assertEqual(env["LLAMA_SLOT_VICTIM_LRU_LEGACY"], "1")
        for options in (base + ["--victim-return-s", "0"], base + ["--victim-return-s", "nan"],
                        base + ["--victim-return-s", "45", "--victim-idle-legacy"],
                        base + ["--victim-return-s", "45", "--victim-lru-legacy"]):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.serve.make_plan(self.serve.arguments(options))

    def test_state_copy_legacy_is_the_off_switch_of_the_batched_state_copies(self):
        for profile in ("tensor-2", "tensor-4"):
            base = ["--model", "/models/target.gguf", "--profile", profile]
            with self.subTest(profile=profile):
                self.assertNotIn("GGML_TENSOR_N_STAGE_LEGACY", self.serve.make_plan(self.serve.arguments(base))[1])
                env = self.serve.make_plan(self.serve.arguments(base + ["--state-copy-legacy"]))[1]
                self.assertEqual(env["GGML_TENSOR_N_STAGE_LEGACY"], "1")
        # an inherited setting never reaches the server: the option is how a box turns the batches off
        self.assertIn("GGML_TENSOR_N_STAGE_LEGACY", self.serve.removed_overrides({"GGML_TENSOR_N_STAGE_LEGACY": "1"}))

    def test_fixed_rows_is_an_opt_in_of_the_tensor_recipes(self):
        for profile in ("tensor-2", "tensor-4"):
            base = ["--model", "/models/target.gguf", "--profile", profile]
            plain_argv, plain_env = self.serve.make_plan(self.serve.arguments(base))
            # the engine's own switches, set only when asked for, and nothing else of the recipe moves with them
            self.assertNotIn("LLAMA_RS_FIXED_ROWS", plain_env)
            self.assertNotIn("GGML_CUDA_SSM_CONV_ROWS", plain_env)
            # the rows come with the kernel that keeps their conv-state update one launch; "unfused" is the rows alone
            rows, fused = {"LLAMA_RS_FIXED_ROWS": "1"}, {"GGML_CUDA_SSM_CONV_ROWS": "1"}
            for options, switches in ((["--fixed-rows"], {**rows, **fused}), (["--fixed-rows", "fused"], {**rows, **fused}),
                                      (["--fixed-rows", "unfused"], rows)):
                with self.subTest(profile=profile, options=options):
                    argv, env = self.serve.make_plan(self.serve.arguments(base + options))
                    self.assertEqual(env, {**plain_env, **switches})
                    self.assertEqual(argv, plain_argv)

    def test_slot_defaults_and_bounds_follow_the_profile(self):
        for options, expected in ((["--profile", "tensor-4"], "23"), (["--profile", "tensor-2"], "12"),
                                  (["--slots", "4"], "4")):
            with self.subTest(options=options):
                argv, _ = self.serve.make_plan(self.serve.arguments(["--model", "/models/target.gguf", *options]))
                self.assertEqual(argv[argv.index("-np") + 1], expected)
        for options in (["--slots", "0"], ["--slots", "257"]):
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
                self.assertEqual(env["GGML_CUDA_MMQ_L2_PF"], "1")

    def test_published_assets_have_the_pinned_bytes_and_vocabulary_shape(self):
        self.serve.verify_assets()
        data = self.serve.DRAFT_VOCAB.read_bytes()
        self.assertEqual(len(data), 393216)
        ids = struct.unpack("<98304i", data)
        self.assertEqual(len(set(ids)), 98304)
        self.assertTrue(set(range(256)).issubset(ids))
        self.assertTrue(set(range(154820, 154856)).issubset(ids))
        self.assertTrue(all(0 <= token <= 154855 for token in ids))


if __name__ == "__main__":
    unittest.main()
