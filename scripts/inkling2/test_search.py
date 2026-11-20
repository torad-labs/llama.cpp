"""The search ends on the fastest settings, every trial reaches MLflow once, and the launcher starts with what it found."""
import json
import os
import subprocess
import sys
import tempfile
import threading
import tomllib
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import search
import serve

# A server that streams fastest when launched with drafts of 5 and 8 sampling threads, and dies with drafts of 7.
STAND_IN = r'''#!/usr/bin/env python3
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if "--help" in sys.argv:
    print("--prompt-chunk N")
    raise SystemExit(0)
with open(os.environ["STAND_IN_LAUNCHES"], "a") as launches:
    launches.write("launch\n")
def value(flag):
    return sys.argv[sys.argv.index(flag) + 1]
depth = int(value("--spec-draft-n-max")) if "--spec-draft-n-max" in sys.argv else 0
if depth == 7:
    raise SystemExit(3)
threads = int(os.environ.get("LLAMA_SAMPLE_THREADS", 4))
pause = 0.002 * (1 + abs(depth - 5)) * (1 + abs(threads - 8) / 8)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *_):
        pass

    def reply(self, body):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def do_GET(self):
        self.reply({"status": "ok"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/apply-template":
            return self.reply({"prompt": body["messages"][0]["content"]})
        if self.path == "/tokenize":
            return self.reply({"tokens": list(range(len(body["content"].split())))})
        self.send_response(200)
        self.end_headers()
        for n in range(body["n_predict"]):
            time.sleep(pause)
            self.wfile.write(b"data: " + json.dumps({"content": "x", "tokens": [n], "stop": False}).encode() + b"\n\n")
            self.wfile.flush()
        timings = {"predicted_n": body["n_predict"], "prompt_n": len(body["prompt"]), "cache_n": 0,
                   "draft_n": 2 * body["n_predict"], "draft_n_accepted": body["n_predict"]}
        self.wfile.write(b"data: " + json.dumps({"content": "", "tokens": [], "stop": True, "timings": timings}).encode() + b"\n\n")


ThreadingHTTPServer.request_queue_size = 64  # every session connects at once
ThreadingHTTPServer(("127.0.0.1", int(value("--port"))), Handler).serve_forever()
'''


class Tracking(BaseHTTPRequestHandler):
    """An MLflow tracking server that keeps what it is sent."""
    runs, calls = {}, []

    def log_message(self, *_):
        pass

    def reply(self, body, status=200):
        self.send_response(status)
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def do_GET(self):
        self.calls.append((self.path, self.headers.get("Authorization")))
        self.reply({"error_code": "RESOURCE_DOES_NOT_EXIST"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.calls.append((self.path, self.headers.get("Authorization")))
        route = self.path.rsplit("/", 1)[-1]
        if route == "create" and "experiments" in self.path:
            return self.reply({"experiment_id": "7"})
        if route == "create":
            run = str(len(self.runs))
            self.runs[run] = {"name": body["run_name"], "params": {}, "metrics": {}, "status": "RUNNING"}
            return self.reply({"run": {"info": {"run_id": run}}})
        run = self.runs[body["run_id"]]
        if route == "log-batch":
            run["params"].update({p["key"]: p["value"] for p in body.get("params", [])})
            run["metrics"].update({m["key"]: m["value"] for m in body.get("metrics", []) if m["step"] == 0})
        else:
            run["status"] = body["status"]
        self.reply({})


class SearchRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.folder.cleanup)
        root = Path(cls.folder.name)
        cls.out, cls.tuned, cls.launches = root / "out", root / "tuned.json", root / "launches"
        cls.server = root / "llama-server"
        cls.server.write_text(STAND_IN)
        cls.server.chmod(0o755)
        cls.env = {key: value for key, value in os.environ.items() if not key.startswith("MLFLOW_")}
        cls.env["STAND_IN_LAUNCHES"] = str(cls.launches)
        cls.first = cls.search("run")

    @classmethod
    def search(cls, verb, *options, env=None):
        words = {"run": ["--profile", "tensor-4-draft", "--server", str(cls.server), "--port", "18103", "--tokens", "24",
                         "--rounds", "2", "--tuned", os.devnull, "--knob=--draft-depth=5,7", "--knob=LLAMA_SAMPLE_THREADS=1,8"],
                 "log": ["--experiment", "inkling2/decode"],
                 "apply": ["--profile", "tensor-4-draft", "--tuned", str(cls.tuned)]}[verb]
        return subprocess.run([sys.executable, search.__file__, verb, "--out", str(cls.out), *words, *options],
                              env=env or cls.env, capture_output=True, text=True)

    def trials(self):
        return [json.loads(line) for line in (self.out / "trials.jsonl").read_text().splitlines()]

    def test_search_ends_on_the_fastest_settings(self):
        self.assertEqual(self.first.returncode, 0, self.first.stderr)
        best = json.loads((self.out / "best.json").read_text())
        self.assertEqual(best["settings"], {"--draft-depth": "5", "LLAMA_SAMPLE_THREADS": "8"})
        trials = self.trials()
        self.assertEqual(trials[0]["settings"], {})
        self.assertGreater(best["tokens_per_second"], 2 * trials[0]["tokens_per_second"])
        self.assertEqual(best["baseline_tokens_per_second"], trials[0]["tokens_per_second"])

    def test_a_trial_whose_server_dies_is_kept_with_its_error_and_the_search_goes_on(self):
        dead = [t for t in self.trials() if t["settings"].get("--draft-depth") == "7"]
        self.assertEqual(len(dead), 1)
        self.assertIsNone(dead[0]["tokens_per_second"])
        self.assertIn("exited 3", dead[0]["error"])
        self.assertEqual(len(self.trials()), 5)

    def test_a_trial_names_how_much_of_its_drafts_the_target_kept(self):
        self.assertEqual(self.trials()[0]["draft_accepted"], 0.5)

    def test_a_finished_trial_is_not_run_again(self):
        before = self.launches.read_text()
        again = self.search("run")
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertEqual(self.launches.read_text(), before)
        self.assertEqual(len(self.trials()), 5)

    def test_each_trial_reaches_mlflow_once_with_its_settings_and_rate(self):
        Tracking.runs, Tracking.calls = {}, []
        tracking = ThreadingHTTPServer(("127.0.0.1", 0), Tracking)
        threading.Thread(target=tracking.serve_forever, daemon=True).start()
        self.addCleanup(tracking.shutdown)
        env = {**self.env, "MLFLOW_TRACKING_URI": f"http://127.0.0.1:{tracking.server_port}",
               "MLFLOW_TRACKING_USERNAME": "seat", "MLFLOW_TRACKING_PASSWORD": "pass-word"}
        for _ in range(2):
            sent = self.search("log", env=env)
            self.assertEqual(sent.returncode, 0, sent.stderr)
            self.assertNotIn("pass-word", sent.stdout + sent.stderr)
        trials = self.trials()
        self.assertEqual(len(Tracking.runs), len(trials))
        self.assertTrue(all(auth == "Basic c2VhdDpwYXNzLXdvcmQ=" for _, auth in Tracking.calls))
        best = max((t for t in trials if t["tokens_per_second"]), key=lambda t: t["tokens_per_second"])
        run = next(r for r in Tracking.runs.values() if r["metrics"].get("tokens_per_second") == best["tokens_per_second"])
        self.assertEqual({k: v for k, v in run["params"].items() if k in best["settings"]}, best["settings"])
        self.assertEqual(run["status"], "FINISHED")
        self.assertEqual(sum(r["status"] == "FAILED" for r in Tracking.runs.values()), 1)

    def test_the_launcher_starts_the_profile_with_the_settings_the_search_kept(self):
        applied = self.search("apply")
        self.assertEqual(applied.returncode, 0, applied.stderr)

        def plan(*options):
            return serve.make_plan(serve.arguments(["--profile", "tensor-4-draft", "--tuned", str(self.tuned), *options]))

        argv, env = plan()
        self.assertEqual(argv[argv.index("--spec-draft-n-max") + 1], "5")
        self.assertEqual(env["LLAMA_SAMPLE_THREADS"], "8")
        argv, _ = plan("--draft-depth", "2")
        self.assertEqual(argv[argv.index("--spec-draft-n-max") + 1], "2")
        _, env = serve.make_plan(serve.arguments(["--profile", "tensor-4", "--tuned", str(self.tuned)]))
        self.assertNotIn("LLAMA_SAMPLE_THREADS", env)


class SearchLoad(unittest.TestCase):
    def test_a_session_decodes_over_the_context_the_load_was_asked_for(self):
        with tempfile.TemporaryDirectory() as folder:
            server = Path(folder) / "llama-server"
            server.write_text(STAND_IN)
            server.chmod(0o755)
            running = subprocess.Popen([str(server), "--port", "18104"],
                                       env={**os.environ, "STAND_IN_LAUNCHES": str(Path(folder) / "launches")})
            self.addCleanup(running.wait)
            self.addCleanup(running.kill)
            words = [sys.executable, search.__file__, "load", "--url", "http://127.0.0.1:18104", "--sessions", "3",
                     "--tokens", "8", "--rounds", "1"]
            for _ in range(100):
                plain = subprocess.run(words, capture_output=True, text=True)
                if plain.returncode == 0:
                    break
                time.sleep(0.05)
            long = subprocess.run(words + ["--context", "300"], capture_output=True, text=True)
        self.assertEqual((plain.returncode, long.returncode), (0, 0), plain.stderr + long.stderr)
        self.assertLess(json.loads(plain.stdout)["prompt_tokens"], 60)
        self.assertTrue(300 <= json.loads(long.stdout)["prompt_tokens"] < 400, long.stdout)


class SearchSpace(unittest.TestCase):
    def test_the_default_space_holds_only_values_the_registry_and_the_launcher_take(self):
        registry = {p["name"]: p for p in tomllib.loads((serve.ROOT / "tunables.toml").read_text())["param"]}
        for name, values in search.SPACE.items():
            for value in values:
                if name.startswith("--"):
                    serve.make_plan(serve.arguments(["--profile", "tensor-4-draft", "--tuned", os.devnull,
                                                     *serve.option_words(name, value)]))
                    continue
                knob = registry[name]
                self.assertTrue(knob["available"] and knob["set"] == "env", name)
                self.assertIn(knob["output"], ("same", "rounding"), name)
                if "values" in knob:
                    self.assertIn(int(value), knob["values"], name)
                elif knob["type"] == "int":
                    self.assertTrue(knob["min"] <= int(value) <= knob["max"], f"{name}={value}")
                else:
                    self.assertIn(value, ("0", "1"), name)


if __name__ == "__main__":
    unittest.main()
