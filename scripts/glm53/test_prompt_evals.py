"""Goal: a traced log's prompt evals are split into what was built and what replayed, with nothing counted twice."""
import io
import unittest

import prompt_evals


def span(kind, ns, rest):
    return f"GLM53_TRACE event={kind} clock=host-monotonic host_ns={ns} thread=b7f75295dec32875 nvtx=1 {rest}"


def decode(event, phase, rows, host="slot:0,measure:0,build:0,alloc:0"):
    return (f"0.08.786.389 I graph_trace layer=llama event={event} context=0x1 result=0x2 uid=1 phase={phase} rows={rows} "
            f"seqs=1 outputs=0 owners=none slots=5 slots_allowed=1 nodes=142 buf_mib=406.5 buf_shared=1 host_us={host}")


def cuda(event, count):
    return [f"0.08.790.040 I graph_trace layer=cuda event={event} context=0x3 device=0 uid=7 key=0x4 ggml_nodes=95"] * count


# An iteration that reads 5 prompt rows on a graph it builds (60 ms: 5.5 ms on the host, every CUDA subgraph launched
# directly), one that reads 5 more on the kept graph (11 ms, replayed) with a checkpoint and the head's catch-up, and a
# reply's 3-row decode
LOG = [
    span("begin", 1_000_000, "stage=iteration iteration=1"),
    span("begin", 2_000_000, "stage=target_eval iteration=1 rows=5 offset=0"),
    span("mark", 2_000_001, "stage=batch iteration=1 seq=0 rows=5 prompt_rows=5 pos0=100"),
    "0.08.786.373 I graph_trace layer=sched event=alloc sched=0x5 nodes=5639 leafs=1422 splits=1 split_us=100 place_us=0 "
    "sync_us=5 plan_us=900 place_new_us=4000",
    decode("build", "target", 5, "slot:100,measure:0,build:400,alloc:5000"),
    *cuda("miss", 2), *cuda("direct", 2),
    span("end", 62_000_000, "stage=target_eval iteration=1 rows=5 offset=0"),
    span("end", 63_000_000, "stage=iteration iteration=1"),
    span("begin", 70_000_000, "stage=iteration iteration=2"),
    span("begin", 71_000_000, "stage=target_eval iteration=2 rows=5 offset=0"),
    span("mark", 71_000_001, "stage=batch iteration=2 seq=0 rows=5 prompt_rows=5 pos0=105"),
    decode("reuse", "target", 5),
    *cuda("replay", 2),
    span("end", 82_000_000, "stage=target_eval iteration=2 rows=5 offset=0"),
    "0.08.431.170 I slot create_check: id  0 | task 0 | created context checkpoint 1 of 32 (pos_min = 109, pos_max = 109, "
    "n_tokens = 110, size = 3.325 MiB, new buffers, 4.50 ms: target 3.321 MiB 4.36, draft 0.000 MiB 0.02, speculative 0.04)",
    span("begin", 83_000_000, "stage=catchup head=0 rows=5"),
    decode("build", "draft", 5, "slot:10,measure:0,build:100,alloc:700"),
    span("end", 85_500_000, "stage=catchup head=0 rows=5"),
    span("end", 90_000_000, "stage=iteration iteration=2"),
    span("begin", 100_000_000, "stage=iteration iteration=3"),
    span("begin", 101_000_000, "stage=target_eval iteration=3 rows=3 offset=0"),
    span("mark", 101_000_001, "stage=batch iteration=3 seq=0 rows=3 prompt_rows=0 pos0=110"),
    decode("reuse", "target", 3),
    *cuda("replay", 2),
    span("end", 111_000_000, "stage=target_eval iteration=3 rows=3 offset=0"),
    span("end", 112_000_000, "stage=iteration iteration=3"),
]


class PromptEvalsTest(unittest.TestCase):
    tool = prompt_evals

    def test_a_built_eval_is_split_into_host_and_run_and_a_kept_one_replays(self):
        prompt_evals, reply_evals, outside, counts = self.tool.read(LOG)
        self.assertEqual((counts["iterations"], counts["target_decodes"], counts["draft_decodes"], counts["unmatched"]),
                         (3, 3, 1, 0))
        built, replayed = prompt_evals["1-32"]
        self.assertEqual((built["built"], built["passes"], built["prompt"], built["direct"], built["replay"]), (1, 1, 5, 2, 0))
        self.assertAlmostEqual(built["ms"], 60.0)
        self.assertAlmostEqual(built["host_ms"], 5.5)
        self.assertAlmostEqual(built["run_ms"], 54.5)
        self.assertEqual((built["place_new"], built["plan"]), (4.0, 1.0))
        self.assertEqual((replayed["built"], replayed["direct"], replayed["replay"]), (0, 0, 2))
        self.assertAlmostEqual(replayed["ms"], 11.0)
        self.assertEqual(list(reply_evals), [3])
        # the second iteration's checkpoint and catch-up are its own, outside its target eval
        first, second = outside["1-32"]
        self.assertEqual((first["checkpoints"], second["checkpoints"]), (0, 1))
        self.assertAlmostEqual(second["checkpoint_ms"], 4.5)
        self.assertAlmostEqual(second["catchup_ms"], 2.5)
        self.assertEqual(second["catchup_built"], 1)

    def test_the_report_names_what_was_built_and_what_replayed(self):
        out = io.StringIO()
        self.tool.report("server.log", self.tool.read(LOG), True, out)
        text = out.getvalue()
        self.assertIn("server.log: 3 iterations, 3 target decodes and 1 of the head in the graph trace", text)
        self.assertRegex(text, r"(?m)^\s+1\s+1\s+1\.0 \|\s+60\.00\s+5\.50 =\s+0\.10\s+0\.00\s+0\.40\s+5\.00 \|\s+54\.50 \|"
                               r"\s+2\s+0\s+0\s+built$")
        self.assertRegex(text, r"(?m)^\s+1\s+0\s+1\.0 \|\s+11\.00\s+0\.00 = .* \|\s+11\.00 \|\s+0\s+0\s+2\s+replayed$")
        self.assertIn("context checkpoints: 1 with a time (median 4.50 ms, max 4.50; 1 into new buffers)", text)
        self.assertIn("it 1      p     5 rows     5 passes  1 built  1", text)

    def test_a_log_without_graph_lines_says_so_and_the_usage_is_checked(self):
        out = io.StringIO()
        self.tool.report("plain.log", self.tool.read([line for line in LOG if "graph_trace" not in line]), False, out)
        self.assertIn("no graph_trace line", out.getvalue())
        self.assertEqual(self.tool.main([], out), 2)
        self.assertEqual(self.tool.main(["a.log", "--other"], out), 2)


if __name__ == "__main__":
    unittest.main()
