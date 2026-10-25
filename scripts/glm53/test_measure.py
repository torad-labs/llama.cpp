import concurrent.futures
import http.server
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from measure import ServingLeg, completion_payload, server_command, validate_response


class ServingStandIn(http.server.BaseHTTPRequestHandler):
    def send(self, status, body):
        self.send_response(status)
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else json.dumps(body).encode())

    def do_GET(self):
        if self.path == "/health":
            self.server.health_calls += 1
            if self.server.health_calls == 2 and self.server.monitor_mode == "exit-during-health":
                self.server.marker.with_name("health").touch()
                time.sleep(0.25)
        self.send(200, {"status": "ok"})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            self.send(200, {"tokens": list(range(131072))})
            return
        tokens = payload["n_predict"]
        prompt_n = len(payload["prompt"]) if isinstance(payload["prompt"], list) else 4
        response = dict(content="answer", timings=dict(predicted_n=tokens, prompt_n=prompt_n, predicted_ms=100))
        if tokens != 32:
            self.server.marker.touch()
            time.sleep(0.25)
            case = self.server.case
            if case == "short-generation":
                response["timings"]["predicted_n"] -= 1
            elif case == "short-prompt":
                response["timings"]["prompt_n"] -= 1
            elif case == "empty":
                response["content"] = ""
            elif case == "application-error":
                response = {"error": "stand-in server error"}
            elif case == "http-error" or case == "concurrent-error" and payload["id_slot"] == 1:
                self.send(503, b'{"error":"stand-in unavailable"}')
                return
            elif case == "bad-json":
                self.send(200, b'not-json\xff')
                return
            elif case == "short-body":
                self.send_response(200)
                self.send_header("Content-Length", "1000")
                self.end_headers()
                self.wfile.write(b'{"content":"partial')
                return
        self.send(200, response)

    def log_message(self, *args):
        pass


MONITOR = '''
import json, sys, time
from datetime import datetime
from pathlib import Path
root = Path(sys.argv[0]).parent
mode = json.loads((root / "monitor.json").read_text())["mode"]
if mode == "exit-nonzero":
    print("stand-in NVML error", file=sys.stderr)
    raise SystemExit(7)
while True:
    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
    for card in ([0] if mode == "missing-card" else [0, 1]):
        memory = "N/A" if mode == "invalid-memory" and card == 1 else 1000 + 1000 * card
        print(f"{now}, {card}, {memory}, 200", flush=True)
    if mode == "exit-during-workload" and (root / "workload").exists():
        raise SystemExit(0)
    if mode == "exit-during-health" and (root / "health").exists():
        raise SystemExit(0)
    if mode == "stale":
        time.sleep(60)
    time.sleep(0.05)
'''


class ServingEvidence(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        root = self.root = Path(self.folder.name)
        binary = root / "bin"
        binary.mkdir()
        for name, source in (("llama-server", "import time; time.sleep(60)\n"), ("nvidia-smi", MONITOR)):
            path = binary / name
            path.write_text(f"#!{sys.executable}\n" + source)
            path.chmod(0o755)
        (binary / "manifest.json").write_text("{}")
        (binary / "monitor.json").write_text(json.dumps({"mode": "good"}))
        (root / "prompts.json").write_text(json.dumps([
            dict(category=name, prompt="public fixture") for name in ("english", "chinese", "code", "tools", "long")]))
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), ServingStandIn)
        self.server.case = None
        self.server.monitor_mode, self.server.health_calls = "good", 0
        self.server.marker = binary / "workload"
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.addCleanup(self.close_server)
        self.args = SimpleNamespace(root=root, fixtures=root, binary=binary, model=root / "model.gguf",
                                    run="leg", model_sha256="test-model", context=1048576, ubatch=512,
                                    deadline=time.time()+20, mode="capacity", profile="", port=self.server.server_port,
                                    slots=3, depth=2, adaptive=0.18, sampling="served", tokens=384, repeats=1)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join()

    def capture_records(self, leg):
        records = [json.loads(path.read_text()) for path in (leg.out / "http").glob("*.json")]
        return [row for row in records if row.get("path") == "/completion"]

    def test_rejected_completion_keeps_its_request_and_raw_reply(self):
        for case in ("short-generation", "short-prompt", "empty", "application-error", "http-error", "bad-json", "short-body"):
            with self.subTest(case=case):
                self.server.case = case
                self.args.run = case
                leg = ServingLeg(self.args)
                leg.out.mkdir(parents=True)
                with self.assertRaises(Exception) as caught:
                    leg.complete(dict(category="long", prompt=list(range(32768))), 64, 2, 32768)
                record, = self.capture_records(leg)
                self.assertEqual(record["request"]["id_slot"], 2)
                self.assertEqual(record["request"]["n_predict"], 64)
                body = (leg.out / record["body_file"]).read_bytes()
                if case == "bad-json":
                    self.assertEqual(body, b'not-json\xff')
                elif case == "short-body":
                    self.assertEqual(body, b'{"content":"partial')
                else:
                    response = json.loads(body)
                    if case == "short-generation":
                        self.assertEqual(response["timings"]["predicted_n"], 63)
                        self.assertIn("63", str(caught.exception))
                        self.assertIn("64", str(caught.exception))
                    elif case == "short-prompt":
                        self.assertEqual(response["timings"]["prompt_n"], 32767)
                        self.assertIn("32767", str(caught.exception))
                        self.assertIn("32768", str(caught.exception))
                    elif case in ("application-error", "http-error"):
                        self.assertIn("error", response)
                self.assertIn("http/", str(caught.exception))
                self.assertFalse((leg.out / "responses.jsonl").exists())

    def test_one_failed_concurrent_reply_does_not_discard_the_other_received_replies(self):
        self.server.case = "concurrent-error"
        leg = ServingLeg(self.args)
        leg.out.mkdir(parents=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            with self.assertRaises(Exception):
                list(pool.map(lambda slot: leg.complete(dict(category="concurrent", prompt="public"), 64, slot),
                              range(3)))
        records = self.capture_records(leg)
        self.assertEqual({row["request"]["id_slot"] for row in records}, {0, 1, 2})
        self.assertEqual(len(records), 3)
        self.assertEqual({row["status"] for row in records}, {200, 503})
        self.assertEqual(len({row["body_file"] for row in records}), 3)
        self.assertTrue(all((leg.out / row["body_file"]).stat().st_size for row in records))

    def test_capacity_requires_live_monitor_and_complete_numeric_memory_evidence(self):
        # The server satisfies the actual HTTP contract; only the external monitor differs between cases.
        for mode in ("exit-nonzero", "exit-during-workload", "exit-during-health", "missing-card", "invalid-memory", "stale", "good"):
            with self.subTest(monitor=mode):
                self.args.run = mode
                self.args.deadline = time.time()+20
                self.server.marker.unlink(missing_ok=True)
                self.server.marker.with_name("health").unlink(missing_ok=True)
                self.server.monitor_mode, self.server.health_calls = mode, 0
                (self.args.binary / "monitor.json").write_text(json.dumps({"mode": mode}))
                leg = ServingLeg(self.args)
                with patch.dict(os.environ, {"PATH": str(self.args.binary)+os.pathsep+os.environ["PATH"]}):
                    if mode == "good":
                        leg.run()
                    else:
                        with self.assertRaisesRegex((RuntimeError, ValueError), "memory|monitor"):
                            leg.run()
                workload = leg.out / "workload.json"
                self.assertEqual(workload.exists() and json.loads(workload.read_text())["passed"], mode == "good")
                exits = json.loads((leg.out / "process-exit.json").read_text())
                self.assertIn("monitor", exits)
                if mode == "exit-nonzero":
                    self.assertEqual(exits["monitor"]["before_cleanup"], 7)
                elif mode in ("exit-during-workload", "exit-during-health"):
                    self.assertEqual(exits["monitor"]["before_cleanup"], 0)
                if mode == "good":
                    evidence = json.loads((leg.out / "memory.json").read_text())
                    self.assertEqual(evidence["peak_mib"], {"0": 1000.0, "1": 2000.0})
                    self.assertEqual(set(evidence["sample_counts"]), {"0", "1"})
                    self.assertTrue(all(count >= 2 for count in evidence["sample_counts"].values()))


class ServedContract(unittest.TestCase):
    def test_sampling_is_the_frozen_served_distribution(self):
        payload = completion_payload("public prompt", 384, 0)
        for key, value in dict(temperature=1.0, top_p=0.95, repeat_penalty=1.1,
                               repeat_last_n=256, seed=530, ignore_eos=True,
                               cache_prompt=False, n_predict=384, id_slot=0).items():
            self.assertEqual(payload[key], value)
        self.assertNotIn("top_k", payload)  # Preserve the qualified server default.
        self.assertEqual(payload["prompt"], "public prompt")

    def test_baseline_and_candidate_use_same_model_and_native_mtp(self):
        args = SimpleNamespace(binary="/bundle with spaces/baseline", model="/models/part1.gguf",
                               fixtures="/fixtures", context=1048576, ubatch=512, profile="",
                               port=8102, slots=3, depth=2, adaptive=0.18)
        baseline = server_command(args)
        args.profile = "sm120-balanced"
        candidate = server_command(args)
        self.assertEqual(candidate[:-2], baseline)
        self.assertEqual(candidate[-2:], ["--glm53-profile", "sm120-balanced"])
        self.assertEqual(baseline[0], "/bundle with spaces/baseline/llama-server")
        for option, value in (("-sm", "tensor"), ("-ts", "1,1"), ("-ctk", "q8_0"),
                              ("-ctv", "q8_0"), ("-ctki", "q8_0"), ("-b", "4096"),
                              ("-np", "3"), ("--spec-type", "draft-mtp"),
                              ("--spec-draft-mtp-adaptive", "0.18")):
            self.assertEqual(baseline[baseline.index(option)+1], value)
        self.assertIn("/fixtures/glm-mtp-public-98304.i32", baseline)

    def test_real_response_requires_delivered_output_and_complete_prompt(self):
        row = {"content": "answer", "timings": {"predicted_n": 384, "prompt_n": 32768}}
        self.assertTrue(validate_response(row, 384, 32768))
        for bad in ({"content": "", "timings": row["timings"]},
                    {"content": "answer", "timings": {"predicted_n": 64, "prompt_n": 32768}},
                    {"content": "answer", "timings": {"predicted_n": 384, "prompt_n": 127}},
                    {"error": "request failed"}):
            with self.assertRaises(ValueError):
                validate_response(bad, 384, 32768)


if __name__ == "__main__":
    unittest.main()
