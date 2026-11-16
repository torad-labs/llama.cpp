import contextlib
import http.client
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).with_name("isolation.py")


class FailedResponse:
    status = 503

    def read(self):
        return b"unavailable"


class FailedConnection:
    def __init__(self, *args, **kwargs):
        pass

    def request(self, *args, **kwargs):
        pass

    def getresponse(self):
        return FailedResponse()

    def close(self):
        pass


class IsolationTest(unittest.TestCase):
    def test_all_failed_requests_make_the_run_incomplete(self):
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "report.json"
            stdout = io.StringIO()
            argv = [str(SCRIPT), "--base-url", "http://server.invalid/v1", "--sessions", "1", "--turns", "1",
                    "--filler", "1", "--system", "1", "--drop", "0", "--out", str(report)]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(http.client, "HTTPConnection", FailedConnection), \
                    mock.patch("time.sleep", return_value=None), contextlib.redirect_stdout(stdout):
                with self.assertRaises(SystemExit) as stopped:
                    runpy.run_path(SCRIPT, run_name="__main__")
            self.assertEqual(stopped.exception.code, 1)
            self.assertIn("INCOMPLETE", stdout.getvalue())
            data = json.loads(report.read_text())
            self.assertEqual(len(data["log"]), 3)
            self.assertTrue(all("error" in event for event in data["log"]))


if __name__ == "__main__":
    unittest.main()
