"""The Inkling 2 Mini launcher's one profile dry-runs to a pinned plan."""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import serve_mini as mini

SERVER = str(mini.ROOT / "build/bin/llama-server")
MODEL = str(mini.ROOT / "models/Inkling-2-Mini-Uncensored/Inkling-2-Mini-Uncensored-AD-IQ3_S.gguf")
# the launch of the Mini searches, arm by arm (scripts/inkling2/serve_mini.py, its header): the flags of the searched base, q4_0 K and V,
# four sessions in the largest pool the dev frontier fit in 16,303 MiB, and -ub 512
PLAN = [SERVER, "-m", MODEL, "-ngl", "99", "--jinja", "-fa", "on", "-b", "2048", "-ub", "512",
        "--temp", "1.0", "--top-p", "1.0", "--top-k", "0", "--min-p", "0.0",
        "--kv-unified", "--cache-ram", "0", "--metrics", "--host", "127.0.0.1", "--port", "8103", "--fit", "off",
        "--cache-type-k", "q4_0", "--cache-type-v", "q4_0", "-c", "233472", "-np", "4"]
ENV = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0", "LLAMA_KV_SEQ_CELLS": "233472",
       "LLAMA_SLOT_PREFIX_COPY": "1", "LD_LIBRARY_PATH": str(mini.ROOT / "build/bin")}


def plan(*options):
    return mini.make_plan(mini.arguments(list(options)))


class PlanTest(unittest.TestCase):
    def test_the_default_is_the_launch_the_searches_found_best(self):
        argv, env = plan()
        self.assertEqual(argv, PLAN)
        self.assertEqual(env, ENV)
        self.assertEqual((argv, env), plan())

    def test_the_pack_is_named_and_placed_by_the_packs_convention(self):
        self.assertEqual(mini.PACKS, mini.ROOT / "models" / mini.SOURCE)
        self.assertEqual(mini.PACK, "Inkling-2-Mini-Uncensored-AD-IQ3_S.gguf")  # the name AtomicChat publishes it under
        self.assertEqual(plan()[0][2], str(mini.PACKS / mini.PACK))

    def test_one_shared_pool_a_session_may_take_whole(self):
        argv, env = plan()
        self.assertIn("--kv-unified", argv)
        self.assertEqual(env["LLAMA_KV_SEQ_CELLS"], argv[argv.index("-c") + 1])
        self.assertEqual(plan("--context", "100000")[1]["LLAMA_KV_SEQ_CELLS"], "100000")

    def test_model_server_address_sessions_and_pool_take_their_options(self):
        other = mini.ROOT / "other/bin/llama-server"
        argv, env = plan("--model", "/m/mini.gguf", "--server", str(other), "--host", "0.0.0.0", "--port", "9000",
                         "--slots", "2", "--context", "100000")
        expected = list(PLAN)
        for flag, value in (("-m", "/m/mini.gguf"), ("--host", "0.0.0.0"), ("--port", "9000"), ("-np", "2"), ("-c", "100000")):
            expected[expected.index(flag) + 1] = value
        expected[0] = str(other)
        self.assertEqual(argv, expected)
        self.assertEqual(env, {**ENV, "LLAMA_KV_SEQ_CELLS": "100000", "LD_LIBRARY_PATH": str(other.parent)})

    def test_loop_passes_set_what_the_engine_needs_of_a_fixed_loop_pack(self):
        # 0 is the default: the pack's own 3 passes, neither variable
        for options in ([], ["--loop-passes", "0"]):
            self.assertEqual(plan(*options), (PLAN, ENV))
        # the pack's own count is trained: the count alone
        self.assertEqual(plan("--loop-passes", "3")[1], {**ENV, "LLAMA_LOOP_PASSES": "3"})
        # a count under it is untrained, and the engine refuses it of a fixed-loop pack without ANY
        for passes in ("1", "2"):
            self.assertEqual(plan("--loop-passes", passes)[1], {**ENV, "LLAMA_LOOP_PASSES": passes, "LLAMA_LOOP_PASSES_ANY": "1"})
        # a count over it is untrained too, and its cache slots are built at load
        self.assertEqual(plan("--loop-passes", "5")[1], {**ENV, "LLAMA_LOOP_PASSES": "5", "LLAMA_LOOP_PASSES_ANY": "1",
                                                         "LLAMA_LOOP_PASSES_MAX": "5"})
        # the pass count is the engine's: the server's arguments never change with it
        for passes in ("1", "2", "3", "5"):
            self.assertEqual(plan("--loop-passes", passes)[0], PLAN)

    def test_out_of_range_values_are_refused_and_the_edges_accepted(self):
        for options in (["--port", "0"], ["--port", "65536"], ["--slots", "0"], ["--slots", "257"], ["--context", "0"],
                        ["--context", "255"], ["--context", "-1"], ["--loop-passes", "-1"], ["--loop-passes", "65"], ["--host", ""]):
            with self.subTest(options=options), self.assertRaises(ValueError):
                plan(*options)
        for options in (["--port", "1"], ["--port", "65535"], ["--slots", "1"], ["--slots", "256"], ["--context", "256"],
                        ["--loop-passes", "64"]):
            with self.subTest(options=options):
                plan(*options)

    def test_no_speculation_tensor_split_or_small_launcher_settings(self):
        for options in ([], ["--loop-passes", "5"], ["--slots", "8", "--context", "100000"]):
            argv, env = plan(*options)
            with self.subTest(options=options):
                for word in argv:
                    self.assertNotIn("spec", word.lower())
                    self.assertNotIn("draft", word.lower())
                for flag in ("-sm", "--split-mode", "-ts", "--tensor-split", "--mmproj", "--chat-template-file",
                             "--prompt-chunk", "--prompt-step", "--attn-mask-bits", "--alias"):
                    self.assertNotIn(flag, argv)
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
                for name in env:
                    self.assertFalse(name.startswith(("LLAMA_SPEC", "GGML_CUDA_AR_", "GGML_META_")), name)


class DryRunTest(unittest.TestCase):
    def dry_run(self, *options, env=None):
        run = subprocess.run([sys.executable, str(Path(mini.__file__)), "--dry-run", *options],
                             capture_output=True, text=True, env=env, check=True)
        return json.loads(run.stdout)

    def test_dry_run_prints_the_plan_and_starts_nothing(self):
        clean = {"PATH": os.environ.get("PATH", "")}
        out = self.dry_run(env=clean)
        self.assertEqual(out, {"argv": PLAN, "env": ENV, "removed": []})
        self.assertEqual(out, self.dry_run(env=clean))
        out = self.dry_run("--model", "/m/mini.gguf", "--loop-passes", "2", env=clean)
        self.assertEqual(out["argv"][1:3], ["-m", "/m/mini.gguf"])
        self.assertEqual(out["env"], {**ENV, "LLAMA_LOOP_PASSES": "2", "LLAMA_LOOP_PASSES_ANY": "1"})

    def test_dry_run_reports_and_exec_removes_inherited_engine_overrides(self):
        inherited = {"PATH": os.environ.get("PATH", ""), "LLAMA_LOOP_PASSES": "5", "GGML_CUDA_FATTN_DEQ_DENSE_LEGACY": "1",
                     "LLAMA_API_KEY": "k"}
        out = self.dry_run(env=inherited)
        self.assertEqual(out["removed"], ["GGML_CUDA_FATTN_DEQ_DENSE_LEGACY", "LLAMA_LOOP_PASSES"])
        env = mini.execution_environment(plan()[1], inherited)
        self.assertNotIn("LLAMA_LOOP_PASSES", env)
        self.assertNotIn("GGML_CUDA_FATTN_DEQ_DENSE_LEGACY", env)
        self.assertEqual(env["LLAMA_API_KEY"], "k")

    def test_a_launch_is_checked_for_its_pack_and_server_before_exec(self):
        with mock.patch.object(os, "execve") as execve:
            with self.assertRaisesRegex(ValueError, "missing local model"):
                mini.main(mini.arguments(["--model", "/nonexistent/pack.gguf", "--server", sys.executable]))
            with self.assertRaisesRegex(ValueError, "missing local server"):
                mini.main(mini.arguments(["--model", __file__, "--server", "/nonexistent/llama-server"]))
        execve.assert_not_called()

    def test_a_launch_replaces_the_process_with_the_planned_server(self):
        args = mini.arguments(["--model", __file__, "--server", sys.executable, "--loop-passes", "2"])
        argv, settings = mini.make_plan(args)
        with mock.patch.object(os, "execve") as execve:
            self.assertEqual(mini.main(args), 0)
        execve.assert_called_once()
        path, started, env = execve.call_args.args
        self.assertEqual((path, started), (argv[0], argv))
        self.assertEqual({name: env[name] for name in settings}, settings)


if __name__ == "__main__":
    unittest.main()
