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

from run_window import Window


class WindowProcessTest(unittest.TestCase):
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
            with self.assertRaises(TimeoutError):
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


if __name__ == "__main__":
    unittest.main()
