import concurrent.futures
import http.server
import io
import json
import os
from pathlib import Path
import socket
import struct
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
        if self.path == "/v1/chat/completions":
            self.send(200, {"choices": [{"message": {"tool_calls": [{"function": {
                "name": "weather", "arguments": '{"city":"Paris"}'}}]}}]})
            return
        tokens = payload["n_predict"]
        prompt_n = len(payload["prompt"]) if isinstance(payload["prompt"], list) else 4
        response = dict(content="answer", timings=dict(predicted_n=tokens, prompt_n=prompt_n, predicted_ms=100))
        if "grammar" in payload:
            response["content"] = '{"ok":true}'
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
            elif case in ("timeout-200", "timeout-503", "reset-200", "reset-503"):
                self.send_response(int(case[-3:]))
                self.send_header("Content-Length", "1000")
                self.end_headers()
                self.wfile.write(b'{"content":"partial')
                self.wfile.flush()
                time.sleep(0.15 if case.startswith("reset") else 0.5)
                if case.startswith("reset"):
                    self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                    self.rfile.close()
                    self.wfile.close()
                    self.connection.close()
                    self.wfile = io.BytesIO()  # The handler still flushes once after do_POST returns.
                return
            elif case == "stream-timeout" and payload.get("stream"):
                self.send_response(200)
                self.send_header("Content-Length", "1000")
                self.end_headers()
                self.wfile.write(b'data: {"content":"partial')
                self.wfile.flush()
                time.sleep(2)
                return
            elif case == "stream-success" and payload.get("stream"):
                self.send(200, b'data: {"content":"visible"}\n\n')
                return
            elif case == "stream-bursts" and payload.get("stream"):
                self.send_response(200)
                self.end_headers()
                for part in ([0], list(range(1, tokens-1)), [tokens-1]):
                    event = dict(content="x", tokens=part, stop=False)
                    self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                    self.wfile.flush()
                    time.sleep(0.02)
                response.update(content="", tokens=[], stop=True)
                self.wfile.write(b"data: " + json.dumps(response).encode() + b"\n\n")
                return
            elif case in ("stream-deadline", "stream-deadline-stall", "stream-deadline-framing") and payload.get("stream"):
                self.server.stream_deadline_barrier.wait()
                self.send_response(200)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                def send_event(event):
                    raw = b"data: " + json.dumps(event).encode() + b"\n\n"
                    header = f"{len(raw):x}".encode()
                    if case == "stream-deadline-framing" and event.get("tokens") == [6]:
                        for byte in header + b";padding=" + b"x"*40:
                            self.wfile.write(bytes([byte]))
                            self.wfile.flush()
                            time.sleep(0.05)
                    else:
                        self.wfile.write(header)
                    self.wfile.write(b"\r\n" + raw + b"\r\n")
                    self.wfile.flush()
                try:
                    for token in range(tokens):
                        send_event(dict(content="x", tokens=[token], stop=False))
                        time.sleep(2 if case == "stream-deadline-stall" and token == 11 else 0.05)
                    send_event(response | dict(content="", tokens=[], stop=True))
                    self.wfile.write(b"0\r\n\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            elif case == "chunked":
                self.send_response(200)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                raw = json.dumps(response).encode()
                for part in (raw[:3], raw[3:11], raw[11:]):
                    self.wfile.write(f"{len(part):x}\r\n".encode() + part + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
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

    def test_concurrent_streams_retain_ids_timestamps_and_equal_lengths(self):
        self.server.case = "stream-bursts"
        leg = ServingLeg(self.args)
        leg.out.mkdir(parents=True)
        result = leg.decode_group(dict(category="long", prompt=list(range(512))), 128, [0, 2], 512)
        self.assertTrue(result["common_decode"]["available"])
        self.assertGreater(result["common_decode"]["tokens_per_second"], 0)
        for row in result["streams"]:
            self.assertEqual(row["token_ids"], list(range(128)))
            self.assertEqual(row["request"]["n_predict"], 128)
            self.assertTrue(row["request"]["return_tokens"])
            self.assertEqual(len(row["bursts"]), 3)
            self.assertTrue((leg.out / row["response_record"]).with_suffix(".events.jsonl").is_file())

    def test_abba_runs_all_four_legs_with_shared_deadline_and_raw_streams(self):
        from measure import run_measurements
        self.server.case = "stream-bursts"
        self.args.mode = "scaling"
        self.args.sessions = [1]
        self.args.prompt_tokens = [512]
        self.args.tokens = 8
        self.args.baseline_binary = self.args.binary
        run_measurements(self.args)
        receipt = json.loads((self.root / "results/leg/abba.json").read_text())
        self.assertTrue(receipt["passed"])
        self.assertEqual(receipt["completed"], ["leg-A1", "leg-B1", "leg-B2", "leg-A2"])
        for name in receipt["completed"]:
            folder = self.root / "results" / name
            self.assertEqual(json.loads((folder / "run.json").read_text())["args"]["deadline"], self.args.deadline)
            measured = json.loads((folder / "decode-512-1-0.json").read_text())
            self.assertEqual(measured["streams"][0]["token_ids"], list(range(8)))
            self.assertTrue(json.loads((folder / "process-exit.json").read_text())["exit"] is not None)

    def check_stream_deadline(self, case):
        self.server.case = case
        self.args.mode = "scaling"
        self.args.sessions = [2]
        self.args.prompt_tokens = [512]
        self.args.tokens = 64
        started = []
        def arm_deadline():
            # This tests reads after HTTP 200, not process startup or prompt construction.
            started.append(time.monotonic())
            self.args.deadline = time.time() + 0.9
        self.server.stream_deadline_barrier = threading.Barrier(2, action=arm_deadline, timeout=5)
        leg = ServingLeg(self.args)
        with self.assertRaises(OSError):
            leg.run()
        self.assertEqual(len(started), 1)
        self.assertLess(time.monotonic() - started[0], 1.4)
        records = [row for row in self.capture_records(leg) if row["request"].get("stream")]
        self.assertEqual({row["request"]["id_slot"] for row in records}, {0, 1})
        for record in records:
            self.assertEqual(record["status"], 200)
            self.assertIn("TimeoutError", record["error"])
            body = leg.out / record["body_file"]
            self.assertIn(b"data:", body.read_bytes())
            events = [json.loads(line) for line in body.with_suffix(".events.jsonl").read_text().splitlines()]
            self.assertTrue(events)
            self.assertFalse(any(row["event"].get("stop") for row in events))
            self.assertFalse(body.with_suffix(".stream.json").exists())
        self.assertFalse((leg.out / "workload.json").exists())
        exits = json.loads((leg.out / "process-exit.json").read_text())
        self.assertIsNotNone(exits["exit"])
        with self.assertRaises(ProcessLookupError):
            os.kill(exits["pid"], 0)

    def test_progressing_streams_stop_at_deadline_and_clean_up(self):
        self.check_stream_deadline("stream-deadline")

    def test_stalled_stream_reads_use_the_remaining_deadline(self):
        self.check_stream_deadline("stream-deadline-stall")

    def test_partial_chunk_framing_cannot_extend_the_deadline(self):
        self.check_stream_deadline("stream-deadline-framing")

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

    def test_transport_failure_retains_received_body_prefix_and_original_cause(self):
        for case in ("timeout-200", "timeout-503", "reset-200", "reset-503"):
            with self.subTest(case=case):
                self.server.case = self.args.run = case
                leg = ServingLeg(self.args)
                leg.out.mkdir(parents=True)
                maximum = 0.35 if case.startswith("timeout") else 1
                with self.assertRaises(OSError) as caught:
                    leg.request("/completion", completion_payload("public", 64, 0), maximum=maximum)
                record, = self.capture_records(leg)
                self.assertEqual(record["status"], int(case[-3:]))
                self.assertEqual((leg.out / record["body_file"]).read_bytes(), b'{"content":"partial')
                self.assertIn("TimeoutError" if case.startswith("timeout") else "ConnectionResetError", record["error"])
                self.assertIn("http/", str(caught.exception))
                self.assertFalse((leg.out / "responses.jsonl").exists())

    def test_stream_timeout_retains_the_incomplete_event_before_a_line_arrives(self):
        self.server.case = "stream-timeout"
        self.args.deadline = time.time() + 1.5
        leg = ServingLeg(self.args)
        leg.out.mkdir(parents=True)
        with self.assertRaises(OSError) as caught:
            leg.semantics()
        record, = [row for row in self.capture_records(leg) if row["request"].get("stream")]
        self.assertEqual(record["status"], 200)
        self.assertEqual((leg.out / record["body_file"]).read_bytes(), b'data: {"content":"partial')
        self.assertIn("TimeoutError", record["error"])
        self.assertIn("http/", str(caught.exception))
        self.assertFalse((leg.out / "cancelled.json").exists())

    def test_chunked_completion_keeps_the_decoded_body_and_passes_validation(self):
        self.server.case = "chunked"
        leg = ServingLeg(self.args)
        leg.out.mkdir(parents=True)
        row = leg.complete(dict(category="english", prompt="public"), 64)
        record, = self.capture_records(leg)
        self.assertEqual(json.loads((leg.out / record["body_file"]).read_bytes()), row["response"])
        self.assertTrue(validate_response(row["response"], 64))

    def test_successful_stream_can_still_cancel_then_complete_a_followup(self):
        self.server.case = "stream-success"
        leg = ServingLeg(self.args)
        leg.out.mkdir(parents=True)
        leg.semantics()
        cancelled = json.loads((leg.out / "cancelled.json").read_text())
        self.assertEqual(cancelled["first_chunk"]["content"], "visible")
        self.assertTrue((leg.out / "cancel-followup.json").is_file())
        record, = [row for row in self.capture_records(leg) if row["request"].get("stream")]
        self.assertEqual((leg.out / record["body_file"]).read_bytes(), b'data: {"content":"visible"}\n\n')

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


class DecodeWindows(unittest.TestCase):
    def test_workload_records_initial_and_effective_prompts_without_prefill_rate(self):
        import hashlib
        import subprocess

        class Handler(ServingStandIn):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/apply-template":
                    self.send(200, {"prompt": "public fixture"})
                    return
                if self.path == "/tokenize":
                    self.send(200, {"tokens": list(range(1024))})
                    return
                self.server.requests.append(payload)
                count = payload["n_predict"]
                self.server.barrier.wait(timeout=5)
                self.send_response(200)
                self.end_headers()
                for part in ([0], list(range(1, count-1)), [count-1]):
                    self.wfile.write(b"data: " + json.dumps(dict(content="x", tokens=part)).encode() + b"\n\n")
                    self.wfile.flush()
                    time.sleep(0.03)
                cached = payload["cache_prompt"]
                final = dict(stop=True, id_slot=payload["id_slot"], tokens=[], timings=dict(
                    predicted_n=count, prompt_n=1 if cached else len(payload["prompt"]),
                    cache_n=len(payload["prompt"])-1 if cached else 0))
                self.wfile.write(b"data: " + json.dumps(final).encode() + b"\n\n")

            def do_GET(self):
                self.send(503, {"error": "metrics offline"})

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.requests, server.barrier = [], threading.Barrier(2)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root/"public-corpus.txt").write_text("public source text\n"*100)
                launch = root/"launch.json"
                launch.write_text(json.dumps(dict(argv=["/workspace/llama-server"])))
                command = [sys.executable, str(Path(__file__).with_name("workload.py")),
                    "--root", tmp, "--name", "short", "--binary", tmp, "--depth", "2",
                    "--port", str(server.server_port), "--slots", "2", "--active", "2",
                    "--context", "512", "--tokens", "32", "--deadline", str(time.time()+15),
                    "--launch-json", str(launch)]
                result = subprocess.run(command, capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                report = json.loads((root/"results/short/run.json").read_text())
                warmup = json.loads((root/"results/short/warmup.json").read_text())
                self.assertTrue(report["passed"])
                self.assertEqual(report["launch_sha256"], hashlib.sha256(launch.read_bytes()).hexdigest())
                self.assertEqual(report["initial_prompt_tokens"], 512)
                self.assertEqual(report["decode_prompt_tokens"], [528, 528])
                self.assertEqual(report["warmup_generated_tokens"], 32)
                self.assertEqual(report["warmup_tokens_per_session"], [16, 16])
                self.assertEqual(report["metrics"]["before"], dict(available=False, error_kind="HTTPError", status=503))
                self.assertEqual(report["metrics"]["after"], report["metrics"]["before"])
                self.assertEqual(report["warmup_seconds"], warmup["wall"]["seconds"])
                self.assertNotIn("prefill_tokens_per_second_wall", report)
                self.assertNotIn("context_tokens", report)
                self.assertNotIn("prefill_seconds", report)
                self.assertGreater(report["decode_finished_epoch"], report["decode_started_epoch"])
                for slot in range(2):
                    requests = [p for p in server.requests if p["id_slot"] == slot]
                    self.assertEqual([p["n_predict"] for p in requests], [16, 32])
                    self.assertEqual(requests[1]["prompt"], requests[0]["prompt"]+list(range(16)))
                    expected = hashlib.sha256(json.dumps(requests[1]["prompt"]).encode()).hexdigest()
                    self.assertEqual(report["decode_prompt_sha256"][slot], expected)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_dp_measurement_uses_shared_four_card_recipe(self):
        args = SimpleNamespace(recipe="dp-4", profile="", ubatch=512, slots=40, context=40*442368,
                               adaptive=0.18, binary="/bin", model="/model.gguf", port=8102, depth=2,
                               kv_host=True, host_budget_mib=300000)
        command = server_command(args)
        self.assertEqual(command[command.index("-sm")+1], "dp")
        self.assertEqual(command[command.index("-ts")+1], "1,1,1,1")
        self.assertIn("--backend-sampling", command)
        self.assertNotIn("--kv-unified", command)
        args.kv_host = False
        with self.assertRaises(ValueError):
            server_command(args)
        args.kv_host = True
        args.profile = "sm120-balanced"
        with self.assertRaises(ValueError):
            server_command(args)

    def test_common_interval_counts_tokens_in_bursts_and_excludes_prefill(self):
        from measure import common_decode_window
        rows = [dict(slot=0, bursts=[dict(at=1., tokens=[1, 2]), dict(at=3., tokens=[3]), dict(at=5., tokens=[4, 5])]),
                dict(slot=1, bursts=[dict(at=2., tokens=[6]), dict(at=3., tokens=[7, 8]), dict(at=6., tokens=[9, 10])])]
        result = common_decode_window(rows)
        self.assertTrue(result["available"])
        self.assertEqual((result["start"], result["end"]), (2., 5.))
        self.assertEqual(result["tokens"], 5)
        self.assertAlmostEqual(result["tokens_per_second"], 5/3)
        self.assertEqual([r["tokens"] for r in result["streams"]], [3, 2])

    def test_no_shared_decode_interval_is_explicit(self):
        from measure import common_decode_window
        rows = [dict(slot=0, bursts=[dict(at=1., tokens=[1]), dict(at=2., tokens=[2])]),
                dict(slot=1, bursts=[dict(at=3., tokens=[1]), dict(at=4., tokens=[2])])]
        self.assertEqual(common_decode_window(rows)["reason"], "no_common_decode_interval")
        self.assertIsNone(common_decode_window(rows)["tokens_per_second"])

    def test_stream_preserves_burst_ids_and_terminal_counters(self):
        from measure import read_completion_stream
        events = [dict(content="a", tokens=[7], stop=False),
                  dict(content="", tokens=[8, 9], stop=False),
                  dict(content="b", tokens=[10], stop=False),
                  dict(content="", tokens=[], stop=True, timings=dict(prompt_n=512, predicted_n=4, draft_n=3, draft_n_accepted=1))]
        clock = iter([1., 2., 3., 4.])
        lines = [b"data: " + json.dumps(e).encode() + b"\n" for e in events]
        result = read_completion_stream(lines, 4, 512, clock=lambda: next(clock))
        self.assertEqual(result["token_ids"], [7, 8, 9, 10])
        self.assertEqual([len(b["tokens"]) for b in result["bursts"]], [1, 2, 1])
        self.assertEqual(result["response"]["content"], "ab")
        self.assertEqual(result["response"]["timings"]["draft_n"], 3)
        for broken in (lines[:-1], lines[:1] + lines[-1:], [b'data: {"error":"failed"}\n']):
            with self.assertRaises(ValueError):
                read_completion_stream(broken, 4, 512)

    def test_bad_or_backwards_bursts_are_rejected(self):
        from measure import common_decode_window
        for bursts in ([dict(at=2., tokens=[1]), dict(at=1., tokens=[2])],
                       [dict(at=float("nan"), tokens=[1])]):
            with self.assertRaises(ValueError):
                common_decode_window([dict(slot=0, bursts=bursts)])


if __name__ == "__main__":
    unittest.main()
