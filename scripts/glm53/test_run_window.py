"""The rental controller must retain receipts and stop its own failed work."""
import json
import os
import concurrent.futures
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

from qualification import device_cases
from run_window import SiblingStopped, Window, execute


class WindowProcessTest(unittest.TestCase):
    def test_negative_device_probe_accepts_only_its_declared_error(self):
        # A configuration probe may intentionally fail, but a different failure
        # (or an unexpected success) must close the window like any other error.
        for exit_code, message, accepted in ((3, "expected-config-error", True),
                                              (4, "expected-config-error", False),
                                              (3, "different-error", False),
                                              (0, "expected-config-error", False)):
            with self.subTest(exit=exit_code, message=message), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                binaries = root / "candidate-tests"
                binaries.mkdir()
                (binaries / "probe").symlink_to(sys.executable)
                window = Window(SimpleNamespace(root=root, stop_at=time.time()+630))
                window.profile_env = {}
                window.out.mkdir()
                case = dict(name="negative", argv=["probe", "-c",
                    "import sys; print(sys.argv[1], file=sys.stderr); raise SystemExit(int(sys.argv[2]))",
                    message, str(exit_code)], env={}, cases=None, require=None,
                    expected_failure=dict(exit=3, message="expected-config-error"))
                if accepted:
                    window.device_case(case)
                    self.assertFalse(window.failed.is_set())
                    self.assertEqual(window.command("next", [sys.executable, "-c", "print('continued')"]).strip(),
                                     "continued")
                else:
                    with self.assertRaises(RuntimeError):
                        window.device_case(case)
                    self.assertTrue(window.failed.is_set())
                receipts = json.loads((window.out / "commands.json").read_text())
                self.assertEqual(receipts[0]["exit"], exit_code)

    def test_real_command_keeps_arguments_and_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            window = Window(SimpleNamespace(root=root, stop_at=time.time()+630))
            window.out.mkdir()
            value = "an argument with spaces; $(no shell)"
            output = window.command("args", [sys.executable, "-c", "import sys; print(sys.argv[1])", value], limit=5)
            self.assertEqual(output.strip(), value)
            receipt, = json.loads((window.out / "commands.json").read_text())
            self.assertEqual(receipt["exit"], 0)
            self.assertEqual(receipt["argv"][-1], value)

    def test_deadline_terminates_only_the_launched_process_and_closes_window(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            window = Window(SimpleNamespace(root=root, stop_at=time.time()+630))
            window.out.mkdir()
            with self.assertRaises(subprocess.TimeoutExpired):
                window.command("timeout", [sys.executable, "-c",
                    "import os,time; print(os.getpid(), flush=True); time.sleep(60)"], limit=1)
            pid = int((window.out / "timeout.out").read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
            receipt, = json.loads((window.out / "commands.json").read_text())
            self.assertNotEqual(receipt["exit"], 0)
            with self.assertRaises(SiblingStopped):
                window.command("later", [sys.executable, "-c", "print('must not run')"])
            self.assertFalse((window.out / "later.out").exists())

    def test_one_failed_card_stops_the_other_cards_current_check(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            window = Window(SimpleNamespace(root=root, stop_at=time.time()+630))
            window.out.mkdir()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                running = pool.submit(window.command, "other-card", [sys.executable, "-c",
                    "import time; print('started', flush=True); time.sleep(60)"], limit=4)
                # Wait for a real started child; this is not a timing performance assertion.
                until = time.monotonic()+3
                while not (window.out / "other-card.out").exists() or not (window.out / "other-card.out").read_text():
                    if time.monotonic() > until:
                        self.fail("child did not start")
                    time.sleep(0.01)
                with self.assertRaises(RuntimeError):
                    window.command("failed-card", [sys.executable, "-c", "raise SystemExit(7)"], limit=3)
                with self.assertRaisesRegex(RuntimeError, "another command failed"):
                    running.result(timeout=5)
            receipts = json.loads((window.out / "commands.json").read_text())
            self.assertEqual({r["name"] for r in receipts}, {"other-card", "failed-card"})
            self.assertTrue(all(r["exit"] != 0 for r in receipts))

    def test_a_failure_on_either_card_is_reported_as_itself(self):
        # Whichever card fails first, the controller must name that case and its
        # reason, not the other card's stop, and the receipts must say which case
        # failed validation and which was stopped because of it.
        header = '"backend_name","op_name","op_params","test_mode","supported","passed","error_message"'
        for failing in (0, 1):
            with self.subTest(card=failing), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                binaries = root / "candidate-tests"
                binaries.mkdir()
                for name in ("probe", "test-backend-ops"):
                    (binaries / name).symlink_to(sys.executable)
                window = Window(SimpleNamespace(root=root, stop_at=time.time()+630))
                window.profile_env = {}
                window.out.mkdir()
                rejected = f'"CUDA0, CUDA0","GATED_DELTA_NET","kda=1,K=3,rows_mode=1","test","0","0","not supported"'
                cases = [dict(name=f"passing-gpu{failing}", argv=["test-backend-ops", "-c",
                              f"print({header!r}); print({header.replace('backend_name', 'CUDA0')!r}.replace('supported', '1')"
                              ".replace('passed', '1').replace('error_message', ''))"],
                              env={"CUDA_VISIBLE_DEVICES": str(failing)}, cases=1, require=None),
                         dict(name=f"rejected-gpu{failing}", argv=["test-backend-ops", "-c",
                              f"print({header!r}); print({rejected!r})"],
                              env={"CUDA_VISIBLE_DEVICES": str(failing)}, cases=1, require=None),
                         dict(name=f"waiting-gpu{1 - failing}", argv=["probe", "-c", "import time; time.sleep(60)"],
                              env={"CUDA_VISIBLE_DEVICES": str(1 - failing)}, cases=None, require=None)]
                with self.assertRaisesRegex(ValueError, f"rejected-gpu{failing}: .*not supported"):
                    window.card_matrix(cases)
                receipts = {r["name"]: r for r in json.loads((window.out / "commands.json").read_text())}
                self.assertEqual(set(receipts), {f"passing-gpu{failing}", f"rejected-gpu{failing}", f"waiting-gpu{1 - failing}"})
                self.assertIsNone(receipts[f"passing-gpu{failing}"]["error"])
                self.assertEqual(receipts[f"rejected-gpu{failing}"]["exit"], 0)
                self.assertIn("not supported", receipts[f"rejected-gpu{failing}"]["error"])
                self.assertIn("another command failed", receipts[f"waiting-gpu{1 - failing}"]["error"])

    def test_a_failing_backend_ops_run_names_its_failing_rows(self):
        # test-backend-ops exits 1 when any case fails; the receipt and the
        # controller's error must still name which cases failed and why.
        header = '"backend_name","op_name","op_params","test_mode","supported","passed","error_message"'
        failing = '[GATED_DELTA_NET] ERR = 0.1 > 0.0005 "CUDA0","GATED_DELTA_NET","kda=1,K=3","test","1","0","test failed"'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "candidate-tests").mkdir()
            (root / "candidate-tests" / "test-backend-ops").symlink_to(sys.executable)
            window = Window(SimpleNamespace(root=root, stop_at=time.time()+630))
            window.profile_env = {}
            window.out.mkdir()
            case = dict(name="kda-failing", argv=["test-backend-ops", "-c",
                        f"print({header!r}); print({failing!r}); raise SystemExit(1)"], env={}, cases=1, require=None)
            with self.assertRaisesRegex(RuntimeError, "kda-failing: exit 1; .*kda=1,K=3: test failed"):
                window.device_case(case)
            receipt, = json.loads((window.out / "commands.json").read_text())
            self.assertIn("kda=1,K=3: test failed", receipt["error"])

    def test_meta_backend_tests_fail_when_they_skip(self):
        # Both meta tests exit 0 when they find no GPU; the sampler must also run over
        # two different cards.
        cases = {c["name"]: c for c in device_cases()}
        outputs = {"meta-sampler": ("meta backend over CUDA0 and CUDA1\nok   case\nOK\n",
                                    ["no GPU for the meta backend, skipped\n", "meta backend over CUDA0 and CUDA0\nOK\n"]),
                   "meta-sourceless": ("OK\n", ["no GPU for the meta backend, skipped\n"])}
        for name, (passing, skipping) in outputs.items():
            for output in [passing] + skipping:
                with self.subTest(name=name, output=output), tempfile.TemporaryDirectory() as folder:
                    root = Path(folder)
                    (root / "candidate-tests").mkdir()
                    binary = root / "candidate-tests" / cases[name]["argv"][0]
                    binary.write_text(f"#!/bin/sh\nprintf '%s' '{output}'\n")
                    binary.chmod(0o755)
                    window = Window(SimpleNamespace(root=root, stop_at=time.time()+630))
                    window.profile_env = {}
                    window.out.mkdir()
                    if output == passing:
                        window.device_case(cases[name])
                    else:
                        with self.assertRaisesRegex(ValueError, f"{name}: required route was not exercised"):
                            window.device_case(cases[name])

    def test_outcome_names_the_stage_and_cause_of_a_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            window = Window(SimpleNamespace(root=root, stop_at=time.time()+7200))
            with self.assertRaisesRegex(ValueError, "stop-at"):
                execute(window)
            outcome = json.loads((window.out / "outcome.json").read_text())
            self.assertEqual((outcome["passed"], outcome["stage"]), (False, "preflight"))
            self.assertRegex(outcome["error"], "^ValueError: stop-at must be the armed deadline")

    def test_the_speed_comparison_comes_before_the_long_serving_checks(self):
        # One hour may not hold every leg; the distribution check and both
        # comparisons must land before the semantics, capacity and route legs.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            window = Window(SimpleNamespace(root=root, stop_at=time.time()+630))
            window.out.mkdir()
            calls = []
            window.preflight = lambda: calls.append("preflight")
            window.device_case = lambda case: calls.append(case["name"])
            window.card_matrix = lambda cases: calls.append("card-matrix")
            window.quality = lambda: calls.append("quality")
            window.compare = lambda label, context, ubatch: calls.append(label) or dict(passed=True)

            def serve(name, arm, context, ubatch, mode="timing"):
                calls.append(name)
                raise TimeoutError("serving deadline reached")
            window.serve = serve
            with self.assertRaises(TimeoutError):
                execute(window)
            served = calls[calls.index("card-matrix") + 1:]
            self.assertEqual(served[-4:], ["quality", "primary", "companion", "serving-semantics"])
            self.assertEqual(calls[:3], ["preflight", "capture-peer", "card-matrix"])
            outcome = json.loads((window.out / "outcome.json").read_text())
            self.assertEqual((outcome["passed"], outcome["stage"]), (False, "serving-semantics"))
            self.assertEqual(outcome["error"], "TimeoutError: serving deadline reached")


if __name__ == "__main__":
    unittest.main()
