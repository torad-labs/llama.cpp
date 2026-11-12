import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock
import urllib.parse

import dp_longctx


def chars_by_four(text):
    return max(1, len(text) // 4)


def measured(messages, tools=None):
    # the stand-in for the server's count: the rendered text's characters by four
    return chars_by_four(dp_longctx.session_text(messages, tools or []))


class SessionShapeTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        root = Path(self.folder.name)
        self.files = []
        for i in range(12):
            path = root / "src" / f"file{i}.cpp"
            path.parent.mkdir(exist_ok=True)
            path.write_text(f"// file {i}\n" + "".join(f"int f{i}_{j}(int x) {{ return x * {j}; }}\n" for j in range(400)))
            self.files.append(path)

    def tearDown(self):
        self.folder.cleanup()

    def test_session_reaches_the_target_within_two_percent(self):
        # A session is a Claude Code history: a system prompt, the tools, a user request, then Read calls and their
        # results (the corpus's files) until the whole rendered prompt measures the target, no more and no more than
        # two percent less.
        target = 20000
        messages = dp_longctx.build_session(self.files, target, measured, dp_longctx.TOOLS, seed=3)
        size = measured(messages, dp_longctx.TOOLS)
        self.assertLessEqual(size, target, (size, target))
        self.assertGreaterEqual(size, int(target * 0.98), (size, target))
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        steps = messages[2:]
        self.assertEqual(len(steps) % 2, 0)
        self.assertGreater(len(steps), 2)
        for call, result in zip(steps[::2], steps[1::2]):
            self.assertEqual(call["role"], "assistant")
            self.assertEqual(len(call["tool_calls"]), 1)
            tool_call = call["tool_calls"][0]
            self.assertEqual(tool_call["function"]["name"], "Read")
            self.assertEqual(result["role"], "tool")
            self.assertEqual(result["tool_call_id"], tool_call["id"])
            path = json.loads(tool_call["function"]["arguments"])["file_path"]
            self.assertTrue(Path(path).read_text().startswith(result["content"]), path)

    def test_session_never_passes_the_target_when_a_file_tokenizes_denser(self):
        # between exact counts the builder estimates the next file at the rate seen so far; a file that tokenizes
        # four times denser than the ones before it must still leave the history at or under the target, since the
        # server refuses a prompt over the slot
        def dense_measure(messages, tools=None):
            size = measured(messages, tools)
            for m in messages:
                if m["role"] == "tool" and m["content"].startswith("// dense"):
                    size += 3 * len(m["content"]) // 4
            return size
        for i in (1, 4, 6, 9, 11):
            self.files[i].write_text("// dense\n" + self.files[i].read_text())
        for seed in range(8):
            messages = dp_longctx.build_session(self.files, 20000, dense_measure, dp_longctx.TOOLS, seed=seed)
            self.assertLessEqual(dense_measure(messages, dp_longctx.TOOLS), 20000, seed)

    def test_two_seeds_read_the_files_in_different_orders(self):
        a = dp_longctx.build_session(self.files, 12000, measured, dp_longctx.TOOLS, seed=1)
        b = dp_longctx.build_session(self.files, 12000, measured, dp_longctx.TOOLS, seed=2)
        self.assertNotEqual(a[2]["tool_calls"][0]["function"]["arguments"], b[2]["tool_calls"][0]["function"]["arguments"])

    def test_next_turn_keeps_the_history_as_its_prefix(self):
        # Every later turn appends the reply and one more step; the earlier messages stay byte for byte, so the
        # server's cache can keep the prefix.
        messages = dp_longctx.build_session(self.files, 8000, measured, dp_longctx.TOOLS, seed=5)
        before = json.dumps(messages)
        longer = dp_longctx.next_turn(messages, "I read it.", self.files[0], "now summarize")
        self.assertEqual(json.dumps(longer[:len(messages)]), before)
        self.assertEqual(longer[len(messages)]["role"], "assistant")
        self.assertEqual(longer[len(messages)]["content"], "I read it.")
        self.assertEqual(longer[len(messages) + 1]["role"], "user")
        self.assertIn("now summarize", longer[len(messages) + 1]["content"])
        self.assertEqual([m["role"] for m in longer[len(messages) + 2:]], ["assistant", "tool"])


class SummaryTest(unittest.TestCase):
    def timing(self, stream, turn, prompt_n, cache_n, predicted_n, prompt_ms, predicted_ms):
        return {"stream": stream, "turn": turn, "wall_ms": prompt_ms + predicted_ms,
                "timings": {"prompt_n": prompt_n, "cache_n": cache_n, "predicted_n": predicted_n,
                            "prompt_ms": prompt_ms, "predicted_ms": predicted_ms}}

    def test_summary_names_the_turn_that_lost_its_cache(self):
        results = [
            self.timing("a", 1, 400000, 0, 64, 120000.0, 1000.0),
            self.timing("b", 1, 400000, 0, 64, 125000.0, 1100.0),
            self.timing("a", 2, 900, 400050, 64, 400.0, 1000.0),
            self.timing("b", 2, 400900, 0, 64, 121000.0, 1000.0),  # the slot lost its cache
        ]
        summary = dp_longctx.summarize(results)
        self.assertEqual(summary["cache_lost"], [{"stream": "b", "turn": 2, "prompt_n": 400900, "cache_n": 0}])
        turn1, turn2 = summary["turns"][0], summary["turns"][1]
        self.assertEqual(turn1["turn"], 1)
        self.assertEqual(turn1["streams"], 2)
        self.assertEqual(turn1["prompt_n_max"], 400000)
        self.assertAlmostEqual(turn1["prompt_ms_median"], 122500.0)
        self.assertAlmostEqual(turn1["decode_tps_median"], (63 / 1000.0 * 1000 + 63 / 1100.0 * 1000) / 2, places=3)
        self.assertIsNone(turn1["output_tps_aggregate"])
        self.assertEqual(turn2["prompt_n_max"], 400900)
        self.assertAlmostEqual(turn2["prompt_ms_median"], (400.0 + 121000.0) / 2)

    def test_producer_rate_including_zero_and_legacy_fallback(self):
        for n, producer, want in [(1, 0, 0), (256, 255, 255), (256, 7, 7)]:
            row = self.timing("a", 1, 10, 0, n, 10, 1000)
            row["timings"]["predicted_per_second"] = producer
            self.assertEqual(dp_longctx.summarize([row])["turns"][0]["decode_tps_median"], want)
        row = self.timing("a", 1, 10, 0, 256, 10, 1000)
        self.assertEqual(dp_longctx.summarize([row])["turns"][0]["decode_tps_median"], 255)
        for ms in [0, -1, None]:
            row["timings"]["predicted_ms"] = ms
            self.assertIsNone(dp_longctx.summarize([row])["turns"][0]["decode_tps_median"])

    def test_common_interval_counts_outputs_and_staggered_starts(self):
        for intervals, want in [([(0, 1), (2, 4)], 5), ([(0, 2), (1, 3)], 20 / 3), ([(2, 4)], 5)]:
            rows = []
            for i, (start, end) in enumerate(intervals):
                row = self.timing(str(i), 1, 10, 0, 10, 10, 1000)
                row.update(request_start_s=start, request_end_s=end)
                rows.append(row)
            self.assertAlmostEqual(dp_longctx.summarize(rows)["turns"][0]["output_tps_aggregate"], want)

    def test_diagnostics_sum_histograms_and_weight_ratios(self):
        rows = [self.timing(str(i), 1, 10, 0, 10, 10, 1000) for i in range(2)]
        for row, n, rounds in zip(rows, [1, 9], [1, 3]):
            row["timings"].update(draft_n=n, draft_n_accepted=1, draft_rounds=rounds,
                                  draft_delivered_n=2, draft_verify_attempts=2,
                                  draft_verify_rows_total=3, draft_replay_attempts=1,
                                  draft_verify_rows={"1": 1, ">8": 2})
        summary = dp_longctx.summarize(rows)["turns"][0]["diagnostics"]
        self.assertEqual(summary["draft_n"], 10)
        self.assertEqual(summary["draft_verify_rows"], {"1": 2, ">8": 4})
        self.assertEqual(summary["acceptance"], 0.2)
        self.assertEqual(summary["delivered_per_round"], 1)
        rows[0]["timings"] = dict(prompt_n=10, prompt_ms=1, predicted_n=1, predicted_ms=1,
                                   draft_n=0, draft_verify_attempts=3)
        zero = dp_longctx.summarize(rows[:1])["turns"][0]["diagnostics"]
        self.assertEqual(zero["draft_n"], 0)
        self.assertEqual(zero["draft_verify_attempts"], 3)
        self.assertIsNone(zero["acceptance"])
        self.assertIsNone(zero["draft_rounds"])
        rows[1]["timings"].pop("draft_verify_attempts")
        mixed = dp_longctx.summarize(rows)["turns"][0]["diagnostics"]
        self.assertEqual(mixed["observations"]["draft_verify_attempts"], 1)
        self.assertEqual(mixed["requests"], 2)

    def test_peak_is_per_card(self):
        self.assertEqual(dp_longctx.peak([[1, 2, 3, 4], [5, 1, 1, 1], [2, 2, 9, 2]]), [5, 2, 9, 4])
        self.assertEqual(dp_longctx.peak([]), [])


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class FakeServer:
    """urllib.request.urlopen stood in: the three routes the tool uses, tokens counted as characters by four, every
    request's timeout kept, and the tokenizer slow for every thread but the first it hears from, so that one stream's
    history takes longer to build than another's whole first turn"""

    def __init__(self, slow=0.0):
        self.slow = slow
        self.lock = threading.Lock()
        self.timeouts = []
        self.first_thread = None
        self.tokenize_at = []
        self.chat_at = []
        self.chat_bodies = []

    def __call__(self, request, timeout="unset"):
        route = urllib.parse.urlparse(request.full_url).path
        body = json.loads(request.data)
        with self.lock:
            self.timeouts.append(timeout)
            if self.first_thread is None:
                self.first_thread = threading.get_ident()
            first = threading.get_ident() == self.first_thread
        if route == "/apply-template":
            return FakeResponse({"prompt": dp_longctx.session_text(body["messages"], body["tools"])})
        if route == "/tokenize":
            if not first:
                time.sleep(self.slow)
            with self.lock:
                self.tokenize_at.append(time.monotonic())
            return FakeResponse({"tokens": [0] * chars_by_four(body["content"])})
        if route == "/v1/chat/completions":
            with self.lock:
                self.chat_at.append(time.monotonic())
                self.chat_bodies.append(body)
            n = chars_by_four(dp_longctx.session_text(body["messages"], body["tools"]))
            return FakeResponse({"choices": [{"message": {"content": "a reply"}, "finish_reason": "stop"}],
                                 "timings": {"prompt_n": n, "cache_n": 0, "prompt_ms": 10.0, "predicted_n": 4,
                                             "predicted_ms": 100.0, "predicted_per_second": 0,
                                             "draft_verify_attempts": 3, "draft_verify_rows": {">8": 2}}})
        raise AssertionError(route)


class ToolRunTest(unittest.TestCase):
    """main driven end to end, the server stood in and nvidia-smi absent"""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        root = Path(self.folder.name)
        for i in range(12):
            path = root / "src" / f"file{i}.cpp"
            path.parent.mkdir(exist_ok=True)
            path.write_text(f"// file {i}\n" + "".join(f"int f{i}_{j}(int x) {{ return x * {j}; }}\n" for j in range(400)))
        self.out = root / "results" / "run.json"

    def tearDown(self):
        self.folder.cleanup()

    def run_tool(self, fake, *extra):
        with mock.patch.object(dp_longctx.urllib.request, "urlopen", fake), \
                mock.patch.object(dp_longctx.subprocess, "run", side_effect=OSError("no nvidia-smi")):
            return dp_longctx.main(["--corpus", self.folder.name, "--context-tokens", "20000", "--out", str(self.out),
                                    "--memory-every", "0.1", *extra])

    def test_a_response_is_waited_for_as_long_as_the_server_works_unless_a_limit_is_given(self):
        # twelve 400k first turns at once took about an hour on four cards (docs/torad/glm53-dp.md), longer than a
        # fixed 3,600 s socket timeout, which would fail every stream of the tool's own headline case before its reply
        fake = FakeServer()
        self.assertEqual(self.run_tool(fake, "--streams", "1", "--turns", "1"), 0)
        self.assertEqual(set(fake.timeouts), {None})
        fake = FakeServer()
        self.assertEqual(self.run_tool(fake, "--streams", "1", "--turns", "1", "--timeout", "120"), 0)
        self.assertEqual(set(fake.timeouts), {120.0})

    def test_server_sampling_sends_no_sampling_field(self):
        # Claude Code sets no sampling: the read of what it is served with must leave the server's own in place
        fake = FakeServer()
        self.assertEqual(self.run_tool(fake, "--streams", "1", "--turns", "2", "--server-sampling"), 0)
        self.assertEqual(len(fake.chat_bodies), 2)
        for body in fake.chat_bodies:
            self.assertEqual(set(body), {"messages", "tools", "max_tokens", "stream"})
        fake = FakeServer()
        self.assertEqual(self.run_tool(fake, "--streams", "1", "--turns", "1"), 0)
        self.assertEqual(fake.chat_bodies[0]["temperature"], 0.0)

    def test_the_first_turns_leave_together(self):
        # a burst of Claude Code windows opens at once: no stream sends its first turn while another is still
        # building its history, however long that build takes
        fake = FakeServer(slow=0.05)
        self.assertEqual(self.run_tool(fake, "--streams", "2", "--turns", "1"), 0)
        self.assertEqual(len(fake.chat_at), 2)
        self.assertLess(max(fake.tokenize_at), min(fake.chat_at))

    def test_every_turn_is_on_disk_as_it_lands(self):
        # a twelve-stream 400k run takes over an hour on rented cards: a run stopped before its end keeps the turns
        # it measured, one JSON row a line beside the results file, and the results file holds the same rows
        fake = FakeServer()
        self.assertEqual(self.run_tool(fake, "--streams", "2", "--turns", "2"), 0)
        rows = [json.loads(line) for line in self.out.with_suffix(".partial.jsonl").read_text().splitlines()]
        self.assertEqual(sorted((r["stream"], r["turn"]) for r in rows),
                         [("s00", 1), ("s00", 2), ("s01", 1), ("s01", 2)])
        report = json.loads(self.out.read_text())
        self.assertEqual(sorted((r["stream"], r["turn"]) for r in report["results"]),
                         sorted((r["stream"], r["turn"]) for r in rows))

        self.assertEqual(report["results"], rows)
        for row in rows:
            self.assertLessEqual(row["request_start_s"], row["request_end_s"])
            self.assertAlmostEqual(row["wall_ms"], (row["request_end_s"] - row["request_start_s"]) * 1000)
            self.assertEqual(row["timings"]["draft_verify_rows"], {">8": 2})
        self.assertEqual(report["summary"]["turns"][0]["diagnostics"]["draft_verify_attempts"], 6)
        self.assertEqual(report["summary"]["turns"][0]["decode_tps_median"], 0)


if __name__ == "__main__":
    unittest.main()
