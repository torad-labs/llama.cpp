"""The Inkling launcher's profiles dry-run to pinned plans."""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import serve

SERVER = str(serve.ROOT / "build/bin/llama-server")
PACKS = serve.ROOT / "models/Inkling-2-Small-abliterated-e4853745"
MMPROJ = str(PACKS / "mmproj-Inkling-2-Small-abliterated-e4853745-F16.gguf")
HOST_ROUTE = {"GGML_CUDA_ALLREDUCE": "internal", "GGML_CUDA_AR_LL_MAX_BYTES": "0", "GGML_CUDA_AR_P2P_MAX_BYTES": "0",
              "GGML_CUDA_AR_COPY_THRESHOLD": "1", "GGML_CUDA_AR_COPY_PEER": "0", "GGML_CUDA_AR_GATHER_OWNER_SLICES": "0"}
ENV = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0,1,2,3", "LLAMA_KV_SEQ_CELLS": "262144",
       "LD_LIBRARY_PATH": str(serve.ROOT / "build/bin"), **HOST_ROUTE}


def pinned(pack, kv, context):
    return [SERVER, "-m", str(PACKS / (pack + ".gguf")), "--mmproj", MMPROJ,
            "-ngl", "99", "-sm", "tensor", "-ts", "1,1,1,1", "--jinja", "-fa", "on",
            "--cache-type-k", kv, "--cache-type-v", kv,
            "-b", "2048", "-ub", "2048", "--prompt-step", "1024", "--prompt-chunk", "128",
            "--temp", "1.0", "--top-p", "1.0", "--top-k", "0", "--min-p", "0.0",
            "--alias", pack, "--chat-template-file", str(serve.TEMPLATE),
            "-c", context, "-np", "23", "--kv-unified", "--cache-ram", "32768", "--no-cache-idle-slots",
            "--metrics", "--host", "127.0.0.1", "--port", "8103", "--log-timestamps"]


def plan(*options):
    return serve.make_plan(serve.arguments(list(options)))


class ProfilePlanTest(unittest.TestCase):
    def test_tensor_4_is_the_default_and_serves_the_q3_pack_at_f16_with_nine_full_sessions(self):
        argv, env = plan()
        self.assertEqual(argv, pinned("Inkling-2-Small-abliterated-e4853745-Q3_K_M-imatrix", "f16", "2359296"))
        self.assertEqual(env, ENV)
        self.assertEqual((argv, env), plan("--profile", "tensor-4"))

    def test_tensor_4_capacity_serves_the_iq3_pack_at_q8_0_with_every_session_full(self):
        argv, env = plan("--profile", "tensor-4-capacity")
        self.assertEqual(argv, pinned("Inkling-2-Small-abliterated-e4853745-IQ3_XXS-imatrix", "q8_0", "6029312"))
        self.assertEqual(env, ENV)

    def test_tensor_4_quality_serves_the_q4_pack_at_f16_with_two_full_sessions(self):
        argv, env = plan("--profile", "tensor-4-quality")
        self.assertEqual(argv, pinned("Inkling-2-Small-abliterated-e4853745-Q4_K_M-imatrix", "f16", "524288"))
        self.assertEqual(env, ENV)

    def test_every_pool_holds_whole_sessions_and_the_profiles_name_packs_of_their_own(self):
        packs = set()
        for name in serve.PROFILES:
            argv, _ = plan("--profile", name)
            context = int(argv[argv.index("-c") + 1])
            self.assertEqual(context % serve.SLOT_CONTEXT, 0, name)
            self.assertLessEqual(context, serve.SLOTS * serve.SLOT_CONTEXT, name)
            packs.add(argv[argv.index("-m") + 1])
        self.assertEqual(len(packs), len(serve.PROFILES))

    def test_no_profile_carries_glm_only_settings_or_a_stock_pack(self):
        for name in serve.PROFILES:
            argv, env = plan("--profile", name)
            with self.subTest(profile=name):
                text = " ".join(argv).lower()
                for glm in ("-ctki", "--spec-", "glm", "unsloth", "ud-"):
                    self.assertNotIn(glm, text)
                for key in ("LLAMA_INDEXER_Q8_QUERY", "LLAMA_MTP_DEVICE_H", "LLAMA_SPEC_COUPLED", "GGML_CUDA_AR_WIRE_Q8"):
                    self.assertNotIn(key, env)


class OptionTest(unittest.TestCase):
    def test_packs_folder_model_and_projector_overrides(self):
        argv, _ = plan("--packs", "/data/packs")
        self.assertEqual(argv[1:5], ["-m", "/data/packs/Inkling-2-Small-abliterated-e4853745-Q3_K_M-imatrix.gguf",
                                     "--mmproj", "/data/packs/mmproj-Inkling-2-Small-abliterated-e4853745-F16.gguf"])
        argv, _ = plan("--model", "/m/first.gguf", "--mmproj", "/m/proj.gguf")
        self.assertEqual(argv[1:5], ["-m", "/m/first.gguf", "--mmproj", "/m/proj.gguf"])
        argv, _ = plan("--text-only")
        self.assertNotIn("--mmproj", argv)
        with self.assertRaises(ValueError):
            plan("--text-only", "--mmproj", "/m/proj.gguf")

    def test_sessions_pool_and_cache_type_take_their_options(self):
        argv, env = plan("--slots", "8", "--context", "4194304", "--slot-context", "524288", "--kv-cache", "q8_0")
        self.assertEqual(argv[argv.index("-np") + 1], "8")
        self.assertEqual(argv[argv.index("-c") + 1], "4194304")
        self.assertEqual(argv[argv.index("--cache-type-k") + 1], "q8_0")
        self.assertEqual(argv[argv.index("--cache-type-v") + 1], "q8_0")
        self.assertEqual(env["LLAMA_KV_SEQ_CELLS"], "524288")
        for options in (["--slots", "0"], ["--slots", "257"], ["--context", "262144"], ["--slot-context", "128"],
                        ["--port", "0"], ["--prompt-step", "-1"], ["--prompt-step-ms", "0"], ["--cache-ram", "-2"],
                        ["--host", "a\nb"]):
            with self.subTest(options=options), self.assertRaises(ValueError):
                plan(*options)

    def test_prompt_step_controls(self):
        argv, _ = plan("--prompt-step", "256", "--prompt-step-ms", "40")
        self.assertEqual(argv[argv.index("--prompt-step") + 1], "256")
        self.assertEqual(argv[argv.index("--prompt-step-ms") + 1], "40")
        self.assertEqual(argv[argv.index("--prompt-chunk") + 1], "128")
        self.assertNotIn("--prompt-step-ms", plan()[0])

    def test_reduction_wire_and_route(self):
        env = plan("--reduction-wire", "q8")[1]
        self.assertEqual(env["GGML_CUDA_AR_WIRE_Q8"], "1")
        env = plan("--reduction-route", "engine")[1]
        for key in HOST_ROUTE:
            self.assertNotIn(key, env)


class DryRunTest(unittest.TestCase):
    def dry_run(self, *options, env=None):
        run = subprocess.run([sys.executable, str(Path(serve.__file__)), "--dry-run", *options],
                             capture_output=True, text=True, env=env, check=True)
        return json.loads(run.stdout)

    def test_dry_run_prints_the_pinned_plan_and_starts_nothing(self):
        clean = {"PATH": os.environ.get("PATH", "")}
        out = self.dry_run("--profile", "tensor-4-capacity", env=clean)
        self.assertEqual(out, {"argv": plan("--profile", "tensor-4-capacity")[0], "env": ENV, "removed": []})
        self.assertEqual(out, self.dry_run("--profile", "tensor-4-capacity", env=clean))

    def test_dry_run_reports_and_exec_removes_inherited_engine_overrides(self):
        inherited = {"PATH": os.environ.get("PATH", ""), "GGML_CUDA_AR_WIRE_Q8": "1", "LLAMA_SPEC_COUPLED": "1",
                     "LLAMA_API_KEY": "k"}
        out = self.dry_run(env=inherited)
        self.assertEqual(out["removed"], ["GGML_CUDA_AR_WIRE_Q8", "LLAMA_SPEC_COUPLED"])
        env = serve.execution_environment(plan()[1], inherited)
        self.assertNotIn("GGML_CUDA_AR_WIRE_Q8", env)
        self.assertNotIn("LLAMA_SPEC_COUPLED", env)
        self.assertEqual(env["LLAMA_API_KEY"], "k")

    def test_a_changed_template_is_refused(self):
        with mock.patch.object(serve, "TEMPLATE_SHA256", "0" * 64), self.assertRaises(ValueError):
            serve.main(serve.arguments(["--dry-run"]))

    def test_a_launch_without_its_pack_is_refused_before_exec(self):
        args = serve.arguments(["--packs", "/nonexistent/packs", "--server", sys.executable])
        with mock.patch.object(os, "execve") as execve, self.assertRaisesRegex(ValueError, "missing local model"):
            serve.main(args)
        execve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
