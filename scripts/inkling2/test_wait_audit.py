"""Unit tests for scripts/inkling2/wait_audit.py."""
import copy
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import wait_audit


def _load_repo_evidence():
    path = REPO_ROOT / "reports" / "inkling2-wait-audit.json"
    return json.loads(path.read_text(encoding="utf-8"))


class WaitAuditTest(unittest.TestCase):
    def test_repo_evidence_passes(self):
        ev = _load_repo_evidence()
        self.assertEqual(wait_audit.check(ev, REPO_ROOT), [])

    def test_cli_check_passes(self):
        run = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "wait_audit.py"),
                "--check",
                "--evidence",
                "reports/inkling2-wait-audit.json",
                "--root",
                str(REPO_ROOT),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(run.returncode, 0, msg=run.stdout + run.stderr)
        self.assertIn("ok:", run.stdout)
        self.assertIn("barrier=0", run.stdout)

    def test_rejects_barrier_disposition(self):
        ev = copy.deepcopy(_load_repo_evidence())
        ev["sites"][0]["disposition"] = "barrier"
        problems = wait_audit.check(ev, REPO_ROOT)
        self.assertTrue(any("barrier" in p for p in problems), problems)

    def test_rejects_missing_host_data(self):
        ev = copy.deepcopy(_load_repo_evidence())
        for site in ev["sites"]:
            if site["disposition"] == "kept-host-data":
                site["host_data"] = "   "
                break
        problems = wait_audit.check(ev, REPO_ROOT)
        self.assertTrue(any("host_data" in p for p in problems), problems)

    def test_rejects_missing_counter_mechanism(self):
        ev = copy.deepcopy(_load_repo_evidence())
        for site in ev["sites"]:
            if site["disposition"] == "counter":
                site["counter_mechanism"] = ""
                break
        problems = wait_audit.check(ev, REPO_ROOT)
        self.assertTrue(any("counter_mechanism" in p for p in problems), problems)

    def test_rejects_hot_out_of_path(self):
        ev = copy.deepcopy(_load_repo_evidence())
        for site in ev["sites"]:
            if site["disposition"] == "out-of-path":
                site["hot_decode_path"] = True
                break
        problems = wait_audit.check(ev, REPO_ROOT)
        self.assertTrue(any("hot_decode_path=true" in p for p in problems), problems)

    def test_rejects_bad_anchor(self):
        ev = copy.deepcopy(_load_repo_evidence())
        ev["sites"][0]["anchor"] = "nonexistent_symbol_xyz_12345"
        problems = wait_audit.check(ev, REPO_ROOT)
        self.assertTrue(any("anchor" in p for p in problems), problems)

    def test_rejects_summary_mismatch_and_nonzero_barrier(self):
        ev = copy.deepcopy(_load_repo_evidence())
        ev["summary"]["barrier"] = 1
        problems = wait_audit.check(ev, REPO_ROOT)
        self.assertTrue(any("summary['barrier']" in p for p in problems), problems)

        ev2 = copy.deepcopy(_load_repo_evidence())
        ev2["summary"]["counter"] = 999
        problems2 = wait_audit.check(ev2, REPO_ROOT)
        self.assertTrue(any("summary['counter']" in p for p in problems2), problems2)

    def test_scan_skips_if0_and_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            target = root / "ggml/src/ggml-cuda/allreduce.cu"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                "\n".join([
                    "// cudaStreamSynchronize(commented);",
                    "#if 0",
                    "cudaDeviceSynchronize();",
                    "#if 1",
                    "cudaStreamSynchronize(nested);",
                    "#endif",
                    "#endif",
                    "void f() { cudaEventSynchronize(ev); }",
                ]) + "\n",
                encoding="utf-8",
            )
            hits = wait_audit.scan_sync_calls(root)
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0]["api"], "cudaEventSynchronize")
            self.assertEqual(hits[0]["line"], 8)

    def test_rejects_unaudited_sync_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            target = root / "ggml/src/ggml-cuda/allreduce.cu"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                "\n".join(
                    ["void counter_fn() { /* counter_anchor */ }"]
                    + [""] * 15
                    + [
                        "void sneaky_fn() {",
                        "    cudaStreamSynchronize(s);",
                        "}",
                    ]
                ) + "\n",
                encoding="utf-8",
            )
            ev = {
                "version": 1,
                "row": "inkling-counter-sync",
                "build": "test",
                "measured_at": "2026-11-24T00:00:00Z",
                "summary": {"counter": 1, "kept-host-data": 0, "out-of-path": 0, "barrier": 0},
                "sites": [
                    {
                        "id": "only-counter",
                        "file": "ggml/src/ggml-cuda/allreduce.cu",
                        "line": 1,
                        "symbol": "counter_fn",
                        "api": "counter",
                        "anchor": "counter_anchor",
                        "hot_decode_path": True,
                        "disposition": "counter",
                        "counter_mechanism": "device atomic release/acquire",
                        "reason": "test",
                    }
                ],
                "verification": {"tests": ["unit"]},
            }
            problems = wait_audit.check(ev, root)
            self.assertTrue(any("unaudited sync call" in p for p in problems), problems)

    def test_rejects_zero_hot_counters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            target = root / "ggml/src/ggml-cuda/allreduce.cu"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("void f() { cudaStreamSynchronize(s); }\n", encoding="utf-8")
            ev = {
                "version": 1,
                "row": "inkling-counter-sync",
                "build": "test",
                "measured_at": "2026-11-24T00:00:00Z",
                "summary": {"counter": 0, "kept-host-data": 1, "out-of-path": 0, "barrier": 0},
                "sites": [
                    {
                        "id": "kept-1",
                        "file": "ggml/src/ggml-cuda/allreduce.cu",
                        "line": 1,
                        "symbol": "f",
                        "api": "cudaStreamSynchronize",
                        "anchor": "cudaStreamSynchronize",
                        "hot_decode_path": True,
                        "disposition": "kept-host-data",
                        "host_data": "logits buffer",
                        "reason": "host reads logits",
                    }
                ],
                "verification": {"tests": ["unit"]},
            }
            problems = wait_audit.check(ev, root)
            self.assertTrue(any("no hot_decode_path site" in p for p in problems), problems)


if __name__ == "__main__":
    unittest.main()
