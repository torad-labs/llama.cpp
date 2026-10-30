"""The paid window must use pinned bytes, matched serving and real device coverage."""
import copy
import hashlib
import itertools
import re
import tempfile
import unittest
from pathlib import Path

from qualification import (BASELINE_COMMIT, device_cases, check_run_identity, check_quality, check_stop_at,
                           verify_files, validate_arch_table, validate_device_csv)


class QualificationTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
