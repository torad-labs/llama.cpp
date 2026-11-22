"""Offline tests for kernel_sweep.py: no cmake, no build, no GPU.

The fake runner answers configure, build and timing commands from a fixture tree, including a build failure and a
nonzero binary exit. Run from this directory:

    python3 -m unittest test_kernel_sweep -v
"""
import contextlib
import io
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

import kernel_sweep

CONFIG = """
[build]
source_dir = "../.."
build_dir = "../../build"
target = "test-backend-ops"
jobs = 4
configure = ["cmake", "-S", "{source_dir}", "-B", "{build_dir}", "-DGGML_CUDA=ON", "-DCMAKE_CUDA_FLAGS=-D{macro}={value}"]
build = ["cmake", "--build", "{build_dir}", "--target", "{target}", "-j", "{jobs}"]
binary = "{build_dir}/bin/test-backend-ops"

[timing]
command = ["{binary}", "perf", "-b", "CUDA0", "-o", "MUL_MAT_ID", "-p", "n_mats=384,n_used=8,"]

[constants.MMQ_NWARPS]
define = "MMQ_NWARPS"
source = "ggml/src/ggml-cuda/mmq.cuh"
source_line = 16
default = 8
values = [4, 6, 8, 16]
op = "MUL_MAT Q3_K"
needs_guard = false

[constants.MMQ_NWARPS.rules]
power_of_two = true

[constants.MMQ_ITER_K]
define = "MMQ_ITER_K"
source = "ggml/src/ggml-cuda/mmq.cuh"
source_line = 14
default = 256
values = [128, 256, 512]
op = "MUL_MAT_ID Q3_K"
needs_guard = true
guard_hint = "wrap ggml/src/ggml-cuda/mmq.cuh:14 in #ifndef MMQ_ITER_K / #endif"

[constants.MMQ_ITER_K.rules]
multiple_of = 256

[constants.graph_slot_max_tokens]
define = "GRAPH_SLOT_MAX_TOKENS"
source = "src/llama-context.h"
source_line = 615
kind = "constexpr"
default = 32
values = [16, 32]
op = "llama context graph slots"
needs_guard = true
timing = []
"""

MMQ_CUH = """\
#ifndef MMQ_NWARPS
#define MMQ_NWARPS 8
#endif

#define MMQ_ITER_K 256
"""

LLAMA_CONTEXT_H = "    static constexpr uint32_t graph_slot_max_tokens = 32;\n"


def perf_text(us_per_case):
    lines = []
    for index, us in enumerate(us_per_case):
        lines.append(
            f"  MUL_MAT_ID(type_a=q3_K,n_mats=384,n_used=8,m=512,n=23,k=4608,case={index}):"
            f"   2048 runs - {us:8.2f} us/run -   1.23 TFLOP/run -   9.96 TFLOPS"
        )
    return "\n".join(lines) + "\n"


PERF = {
    4: perf_text([100.0, 150.0]),
    8: perf_text([80.0, 120.0]),
    16: perf_text([120.0, 180.0]),
}


class FakeRunner:
    """Answers the three command stages; remembers the value the last configure set."""

    def __init__(self):
        self.calls = []
        self.current_value = None
        self.build_failures = set()
        self.run_failures = set()
        self.perf_by_value = dict(PERF)

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), dict(kwargs)))
        if "perf" in argv:
            if self.current_value in self.run_failures:
                return subprocess.CompletedProcess(argv, 1, "", "CUDA error: launch failed")
            return subprocess.CompletedProcess(argv, 0, self.perf_by_value.get(self.current_value, ""), "")
        if "--build" in argv:
            if self.current_value in self.build_failures:
                return subprocess.CompletedProcess(argv, 3, "", "ninja: build stopped")
            return subprocess.CompletedProcess(argv, 0, "", "")
        match = re.search(r"-D[A-Za-z_][A-Za-z0-9_]*=(\d+)\b", " ".join(argv))
        if match:
            self.current_value = int(match.group(1))
        return subprocess.CompletedProcess(argv, 0, "", "")


class KernelSweepTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        config_dir = self.root / "scripts" / "hyperparameter"
        config_dir.mkdir(parents=True)
        cuda_dir = self.root / "ggml" / "src" / "ggml-cuda"
        cuda_dir.mkdir(parents=True)
        (cuda_dir / "mmq.cuh").write_text(MMQ_CUH, encoding="utf-8")
        src_dir = self.root / "src"
        src_dir.mkdir()
        (src_dir / "llama-context.h").write_text(LLAMA_CONTEXT_H, encoding="utf-8")
        self.config = config_dir / "kernel_constants.toml"
        self.config.write_text(CONFIG, encoding="utf-8")
        self.results = self.root / "results"

    def tearDown(self):
        self.tmp.cleanup()

    def run_sweep(self, fake, constants=("MMQ_NWARPS",)):
        out = io.StringIO()
        code = kernel_sweep.run(
            self.config,
            results_dir=str(self.results),
            constants=list(constants),
            runner=fake,
            out=out,
        )
        return code, out.getvalue()

    def records(self):
        files = sorted(self.results.glob("*.jsonl"))
        self.assertEqual(len(files), 1)
        lines = files[0].read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def test_plan_prints_commands_and_prunes_invalid_values(self):
        out = io.StringIO()
        code = kernel_sweep.plan(self.config, out=out)
        text = out.getvalue()

        self.assertEqual(code, 0)
        self.assertIn("-DCMAKE_CUDA_FLAGS=-DMMQ_NWARPS=4", text)
        self.assertIn("cmake --build", text)
        self.assertIn("perf -b CUDA0 -o MUL_MAT_ID -p n_mats=384,n_used=8,", text)
        self.assertIn("injectable: yes", text)
        # 6 is pruned by the power-of-two rule, so no command line for it exists.
        self.assertNotIn("-DMMQ_NWARPS=6", text)
        self.assertIn("6 (not a power of two)", text)
        # 128 is pruned by the divisibility rule of the K-quant block size.
        self.assertIn("128 (not a multiple of 256)", text)

    def test_plan_names_the_missing_guard(self):
        out = io.StringIO()
        code = kernel_sweep.plan(self.config, constants=["MMQ_ITER_K"], out=out)
        text = out.getvalue()

        self.assertEqual(code, 0)
        self.assertIn("injectable: no", text)
        self.assertIn("without #ifndef MMQ_ITER_K", text)
        self.assertIn("guard needed", text)
        # plan is a dry run: the commands it would run are still printed.
        self.assertIn("-DMMQ_ITER_K=256", text)

    def test_run_refuses_unguarded_constant_before_any_command(self):
        fake = FakeRunner()
        code, text = self.run_sweep(fake, constants=["MMQ_ITER_K"])

        self.assertEqual(code, 2)
        self.assertEqual(fake.calls, [])
        self.assertIn("refused", text)
        self.assertIn("#ifndef MMQ_ITER_K", text)
        self.assertFalse(self.results.exists())

    def test_run_refuses_constexpr_constant(self):
        fake = FakeRunner()
        code, text = self.run_sweep(fake, constants=["graph_slot_max_tokens"])

        self.assertEqual(code, 2)
        self.assertEqual(fake.calls, [])
        self.assertIn("constexpr", text)
        self.assertIn("no timing command", text)

    def test_run_records_values_and_jsonl_roundtrips(self):
        fake = FakeRunner()
        code, text = self.run_sweep(fake)

        self.assertEqual(code, 0)
        self.assertEqual(len(fake.calls), 9)  # 3 valid values x 3 stages
        records = self.records()
        self.assertEqual([record["value"] for record in records], [4, 8, 16])
        by_value = {record["value"]: record for record in records}
        self.assertEqual(by_value[4]["score_us_per_run"], 250.0)
        self.assertEqual(by_value[8]["score_us_per_run"], 200.0)
        self.assertEqual(by_value[16]["score_us_per_run"], 300.0)
        for record in records:
            self.assertEqual(record["status"], "ok")
            self.assertEqual(record["failed_stage"], None)
            self.assertEqual(record["exit_codes"], {"configure": 0, "build": 0, "run": 0})
            self.assertEqual(set(record["wall_seconds"]), {"configure", "build", "run", "total"})
            self.assertTrue(record["perf"][0]["case"].startswith("MUL_MAT_ID("))
            self.assertEqual(record["perf"][0]["n_runs"], 2048)
        # A sweep never puts a timeout on a configure, build or timing run.
        for _, kwargs in fake.calls:
            self.assertNotIn("timeout", kwargs)

    def test_build_failure_marks_value_invalid_and_sweep_goes_on(self):
        fake = FakeRunner()
        fake.build_failures = {4}
        code, _ = self.run_sweep(fake)

        self.assertEqual(code, 0)
        by_value = {record["value"]: record for record in self.records()}
        self.assertEqual(by_value[4]["status"], "invalid")
        self.assertEqual(by_value[4]["failed_stage"], "build")
        self.assertEqual(by_value[4]["invalid_reason"], "build exited 3")
        self.assertNotIn("run", by_value[4]["exit_codes"])
        self.assertEqual(by_value[8]["status"], "ok")
        self.assertEqual(by_value[16]["status"], "ok")

    def test_nonzero_binary_exit_marks_value_invalid(self):
        fake = FakeRunner()
        fake.run_failures = {4}
        code, _ = self.run_sweep(fake)

        self.assertEqual(code, 0)
        by_value = {record["value"]: record for record in self.records()}
        self.assertEqual(by_value[4]["status"], "invalid")
        self.assertEqual(by_value[4]["failed_stage"], "run")
        self.assertEqual(by_value[4]["invalid_reason"], "run exited 1")
        self.assertEqual(by_value[8]["status"], "ok")

    def test_unparsed_timing_output_marks_value_invalid(self):
        fake = FakeRunner()
        fake.perf_by_value[4] = ""
        code, _ = self.run_sweep(fake)

        self.assertEqual(code, 0)
        by_value = {record["value"]: record for record in self.records()}
        self.assertEqual(by_value[4]["status"], "invalid")
        self.assertEqual(by_value[4]["failed_stage"], "parse")
        self.assertIn("no perf lines", by_value[4]["invalid_reason"])

    def test_every_value_invalid_returns_one(self):
        fake = FakeRunner()
        fake.build_failures = {4, 8, 16}
        code, _ = self.run_sweep(fake)

        self.assertEqual(code, 1)
        self.assertTrue(all(record["status"] == "invalid" for record in self.records()))

    def test_report_best_and_delta_math(self):
        self.results.mkdir()
        records = [
            {"timestamp": "t1", "constant": "MMQ_NWARPS", "value": 1, "default": 2,
             "status": "ok", "score_us_per_run": 1.0},
            {"timestamp": "t2", "constant": "MMQ_NWARPS", "value": 2, "default": 2,
             "status": "ok", "score_us_per_run": 2.0},
            {"timestamp": "t3", "constant": "MMQ_NWARPS", "value": 3, "default": 2,
             "status": "invalid", "score_us_per_run": None},
        ]
        path = self.results / "20261122T010000Z.jsonl"
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )

        out = io.StringIO()
        code = kernel_sweep.report(self.results, out=out)
        text = out.getvalue()

        self.assertEqual(code, 0)
        self.assertIn("-50.0%", text)
        self.assertIn("default", text)
        self.assertIn("n/a", text)
        self.assertIn("best MMQ_NWARPS: 1", text)

    def test_report_reads_every_jsonl_and_keeps_the_last_record(self):
        self.results.mkdir()
        first = [
            {"timestamp": "t1", "constant": "X", "value": 8, "default": 8,
             "status": "ok", "score_us_per_run": 100.0},
        ]
        second = [
            {"timestamp": "t2", "constant": "X", "value": 8, "default": 8,
             "status": "ok", "score_us_per_run": 90.0},
            {"timestamp": "t3", "constant": "X", "value": 4, "default": 8,
             "status": "ok", "score_us_per_run": 45.0},
        ]
        (self.results / "20261122T010000Z.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in first), encoding="utf-8"
        )
        (self.results / "20261122T020000Z.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in second), encoding="utf-8"
        )

        out = io.StringIO()
        code = kernel_sweep.report(self.results, out=out)
        text = out.getvalue()

        self.assertEqual(code, 0)
        self.assertIn("2 file(s)", text)
        self.assertIn("90.000", text)
        self.assertIn("best X: 4", text)

    def test_cli_refusal_exit_code(self):
        fake = FakeRunner()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = kernel_sweep.main(
                ["run", str(self.config), "--results", str(self.results), "--constant", "MMQ_ITER_K"],
                runner=fake,
            )

        self.assertEqual(code, 2)
        self.assertEqual(fake.calls, [])
        self.assertIn("refused", out.getvalue())

    def test_unknown_constant_is_an_error(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = kernel_sweep.main(["plan", str(self.config), "--constant", "NOPE"])

        self.assertEqual(code, 2)
        self.assertIn("unknown constant", err.getvalue())

    def test_bad_rule_is_a_config_error_not_a_crash(self):
        self.config.write_text(
            CONFIG.replace("multiple_of = 256", "multiple_of = 0"), encoding="utf-8"
        )
        with self.assertRaises(kernel_sweep.SweepError):
            kernel_sweep.plan(self.config, out=io.StringIO())


if __name__ == "__main__":
    unittest.main()
