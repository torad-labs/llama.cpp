"""The paid window must use pinned bytes, matched serving and real device coverage."""
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from qualification import (BASELINE_COMMIT, device_cases, check_run_identity,
                           check_quality, verify_files, validate_device_csv)


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
                       header+'"CPU",1,1,""\n'):
            with self.assertRaises(ValueError):
                validate_device_csv(output, 3)

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
                self.assertIn(f"iq3-s{slots}-gpu{gpu}", names)
        for route in ("peer", "host", "mixed", "duplicate"):
            self.assertIn(f"capture-{route}", names)
        self.assertEqual(next(c for c in cases if c["name"] == "capture-peer")["require"],
                         "device inboxes exercised")


if __name__ == "__main__":
    unittest.main()
