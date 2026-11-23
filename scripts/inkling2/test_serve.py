"""The Inkling launcher's profiles dry-run to pinned plans."""
import json
import os
import tempfile
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
              "GGML_CUDA_AR_COPY_THRESHOLD": "1", "GGML_CUDA_AR_COPY_PEER": "0"}
ENV = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0,1,2,3", "LLAMA_KV_SEQ_CELLS": "262144",
       "LLAMA_TOK_EMBD_GPU": "1",
       "LLAMA_STATE_SAVE_ASYNC": "1", "LLAMA_STATE_RESTORE_ASYNC": "1",
       "LD_LIBRARY_PATH": str(serve.ROOT / "build/bin"), **HOST_ROUTE}
TENSOR4_ENV = {**ENV, "LLAMA_KV_SEQ_CELLS": "294912", "LLAMA_SLOT_PREFIX_COPY": "1", "LLAMA_INKLING2_MEAN_MASK_DECODE": "1",
               "GGML_CUDA_MMVQ_KQUANT_MMID_MAX": "23", "LLAMA_INKLING2_BUCKET_PREFILL_KV": "1", "GGML_CUDA_MMVQ_EXPERT_SORT_23": "1",
               "GGML_CUDA_FATTN_REL_QUERY_FIRST": "1", "LLAMA_INKLING2_COMPACT_GREEDY": "1",
               "LLAMA_INKLING2_KV_RUNS": "1", "GGML_CUDA_FATTN_REL_RUNS_SHARED": "1",
               "GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE": "1",
               "GGML_CUDA_FATTN_REL_PACKED_PREFILL": "1", "GGML_CUDA_F32_FEW": "1", "GGML_CUDA_MOE_FUSE": "1",
               "GGML_CUDA_MMQ_TMA": "1", "GGML_CUDA_INKLING2_L2_PERSIST": "1",
               "GGML_CUDA_AR_COPY_ENGINE_CAPTURE": "1",
               "LLAMA_GRAPH_SLOTS": "32", "GGML_META_CAPTURE_GRAPHS": "38",
               "LLAMA_RS_FIXED_ROWS": "1", "LLAMA_GRAPH_ANY_SEQS": "1",
               "GGML_CUDA_GRAPH_NODES": "262144", "GGML_CUDA_GRAPH_IDLE_MS": "600000",
               "LLAMA_RS_STAGE_ROWS": "2", "LLAMA_CHECKPOINT_RAM_MIB": "16384"}


def pinned(pack, kv, context, cache_ram="32768", projector=MMPROJ, swa_kv="q8_0"):
    argv = [SERVER, "-m", str(PACKS / (pack + ".gguf")), "--mmproj", projector,
            "-ngl", "99", "-sm", "tensor", "-ts", "1,1,1,1", "--jinja", "-fa", "on", "--poll", "0",
            "--cache-type-k", kv, "--cache-type-v", kv,
            "-b", "2048", "-ub", "2048", "--prompt-step", "1024", "--prompt-chunk", "128", "--attn-mask-bits"]
    if swa_kv:
        argv += ["--cache-type-k-swa", swa_kv, "--cache-type-v-swa", swa_kv]
    argv += ["--temp", "1.0", "--top-p", "1.0", "--top-k", "0", "--min-p", "0.0",
             "--alias", pack, "--chat-template-file", str(serve.TEMPLATE),
             "-c", context, "-np", "23", "--kv-unified", "--cache-ram", cache_ram, "--no-cache-idle-slots",
             "--metrics", "--host", "127.0.0.1", "--port", "8103", "--log-timestamps"]
    return argv


def plan(*options):
    return serve.make_plan(serve.arguments(list(options)))


class ProfilePlanTest(unittest.TestCase):
    def test_tensor_4_draft_layout_rides_the_flag(self):
        with tempfile.TemporaryDirectory() as root:
            packs = Path(root)
            draft = packs / serve.DRAFT
            plain, _ = plan("--packs", root)
            self.assertNotIn("--spec-type", plain)
            draft.touch()
            self.assertEqual(plain, plan("--packs", root)[0])
            auto, _ = plan("--packs", root, "--draft")
            for flag, value in {
                "--spec-type": "draft-dspark", "--spec-draft-model": str(draft),
                "--spec-draft-n-max": "3", "--spec-draft-sampling": "probabilistic",
                "--spec-draft-device": "CUDA3", "--spec-draft-type-k": "q8_0",
                "--cache-type-k": "q8_0", "--cache-type-k-swa": "q8_0", "-c": "3014656", "-ub": "256",
            }.items():
                self.assertEqual(auto[auto.index(flag) + 1], value)
            greedy, _ = plan("--packs", root, "--draft", "--draft-sampling", "greedy")
            self.assertEqual(greedy[greedy.index("--spec-draft-sampling") + 1], "greedy")
            override, _ = plan("--packs", root, "--draft", "--ubatch-size", "512")
            self.assertEqual(override[override.index("-ub") + 1], "512")
            self.assertEqual(plain, plan("--packs", root, "--no-draft")[0])
            control, _ = plan("--packs", root, "--draft-layout", "--no-draft")
            self.assertEqual(control, auto[:auto.index("--spec-type")])
            custom = ["--packs", root, "--model", str(packs / "custom.gguf")]
            self.assertNotIn("--spec-type", plan(*custom)[0])
            self.assertIn("--spec-type", plan(*custom, "--draft-model", str(draft))[0])
            with self.assertRaises(ValueError):
                plan(*custom, "--draft")
            with self.assertRaises(ValueError):
                plan("--no-draft", "--draft-model", str(draft))

    def test_tensor_4_defaults_to_the_qualified_q3_pack_and_pool(self):
        argv, env = plan()
        self.assertEqual(argv, pinned("Inkling-2-Small-abliterated-e4853745-Q3_K_M-imatrix", "f16", "3014656",
                                      cache_ram="262144", projector=str(PACKS / "mmproj-F16.gguf")) + ["--checkpoint-every", "32768"])
        self.assertEqual(env, TENSOR4_ENV)
        self.assertEqual((argv, env), plan("--profile", "tensor-4"))

    def test_tensor_4_defaults_to_the_two_measured_decode_changes(self):
        # the fastest measured serving configuration is the one profile (reports/inkling2-tensor4-implementation.md)
        argv, env = plan("--profile", "tensor-4")
        self.assertIn("GGML_CUDA_FATTN_REL_QUERY_FIRST", env)
        self.assertIn("LLAMA_INKLING2_COMPACT_GREEDY", env)
        self.assertEqual((argv, env), plan())

    def test_tensor_4_can_select_plain_engine_or_q8_wire_without_bf16_ownership(self):
        profile = ("--profile", "tensor-4")
        _, engine = plan(*profile, "--reduction-route", "engine")
        self.assertEqual({key for key in engine if key.startswith("GGML_CUDA")},
                         {"GGML_CUDA_MMVQ_KQUANT_MMID_MAX", "GGML_CUDA_MMVQ_EXPERT_SORT_23",
                          "GGML_CUDA_FATTN_REL_QUERY_FIRST", "GGML_CUDA_FATTN_REL_RUNS_SHARED",
                          "GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE",
                          "GGML_CUDA_FATTN_REL_PACKED_PREFILL", "GGML_CUDA_F32_FEW", "GGML_CUDA_MOE_FUSE",
                          "GGML_CUDA_MMQ_TMA", "GGML_CUDA_INKLING2_L2_PERSIST",
                          "GGML_CUDA_GRAPH_NODES", "GGML_CUDA_GRAPH_IDLE_MS"})
        _, q8 = plan(*profile, "--reduction-wire", "q8")
        self.assertEqual(q8["GGML_CUDA_AR_WIRE_Q8"], "1")
        self.assertEqual(q8["GGML_CUDA_AR_GATHER_Q8"], "1")
        self.assertNotIn("GGML_CUDA_AR_HOST_OWNER_ROWS", q8)
        with self.assertRaises(ValueError):
            plan(*profile, "--reduction-route", "engine", "--host-owner-rows")

    def test_explicit_tensor_4_overrides(self):
        argv, env = plan("--profile", "tensor-4", "--no-host-owner-rows", "--no-prefix-copy", "--no-mean-mask-decode", "--no-bucket-prefill-kv", "--no-stable-prefill", "--no-sort-expert-rows",
                         "--no-query-first-decode", "--no-compact-greedy",
                         "--no-kv-runs", "--no-packed-prefill", "--no-f32-few", "--no-moe-fuse", "--no-mmq-tma",
                         "--no-l2-persist", "--no-ar-copy-capture",
                         "--swa-kv-cache", "f16",
                         "--keep-graphs", "0", "--checkpoint-rows", "0", "--checkpoint-ram", "0",
                         "--cache-ram", "0", "--context", "1048576", "--slots", "8", "--slot-context", "131072",
                         "--expert-row-limit", "auto", "--checkpoint-every", "0")
        for flag, expected in [("--cache-ram", "0"), ("-c", "1048576"), ("-np", "8"), ("--checkpoint-every", "0")]:
            self.assertEqual(argv[argv.index(flag)+1], expected)
        self.assertEqual(env, {**HOST_ROUTE, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                               "LLAMA_KV_SEQ_CELLS": "131072", "LLAMA_TOK_EMBD_GPU": "1",
                               "LD_LIBRARY_PATH": str(serve.ROOT / "build/bin")})

    def test_every_pool_holds_a_full_session_and_fits_the_combined_session_limits(self):
        for name in serve.PROFILES:
            argv, env = plan("--profile", name)
            context = int(argv[argv.index("-c") + 1])
            slot_context = int(env["LLAMA_KV_SEQ_CELLS"])
            self.assertLessEqual(slot_context, context, name)
            self.assertLessEqual(context, serve.SLOTS * slot_context, name)

    def test_no_profile_carries_glm_only_settings_or_a_stock_pack(self):
        for name in serve.PROFILES:
            argv, env = plan("--profile", name)
            with self.subTest(profile=name):
                text = " ".join(argv).lower()
                for glm in ("-ctki", "glm", "unsloth", "ud-"):
                    self.assertNotIn(glm, text)
                for key in ("LLAMA_INDEXER_Q8_QUERY", "LLAMA_MTP_DEVICE_H", "LLAMA_SPEC_COUPLED", "GGML_CUDA_AR_WIRE_Q8"):
                    self.assertNotIn(key, env)


class OptionTest(unittest.TestCase):
    def test_cross_session_prefix_copy_can_be_disabled(self):
        self.assertEqual(plan("--prefix-copy")[1], TENSOR4_ENV)
        self.assertNotIn("LLAMA_SLOT_PREFIX_COPY", plan("--no-prefix-copy")[1])

    def test_prefill_kv_buckets_can_be_disabled(self):
        key = "LLAMA_INKLING2_BUCKET_PREFILL_KV"
        self.assertNotIn(key, plan("--no-bucket-prefill-kv")[1])
        self.assertEqual(plan("--bucket-prefill-kv")[1][key], "1")
        self.assertEqual(plan("--stable-prefill")[1]["GGML_CUDA_FATTN_REL_PREFILL_F32"], "1")
        self.assertNotIn("GGML_CUDA_FATTN_REL_PREFILL_F32", plan("--no-stable-prefill")[1])
        self.assertEqual(plan("--sort-expert-rows")[1]["GGML_CUDA_MMVQ_EXPERT_SORT_23"], "1")
        self.assertNotIn("GGML_CUDA_MMVQ_EXPERT_SORT_23", plan("--no-sort-expert-rows")[1])

    def test_decode_changes_switch_on_and_off_one_by_one(self):
        query_first, compact = "GGML_CUDA_FATTN_REL_QUERY_FIRST", "LLAMA_INKLING2_COMPACT_GREEDY"
        self.assertNotIn(query_first, plan("--no-query-first-decode")[1])
        self.assertNotIn(compact, plan("--no-compact-greedy")[1])
        self.assertNotIn(query_first, plan("--no-query-first-decode", "--no-compact-greedy")[1])
        self.assertNotIn(compact, plan("--no-query-first-decode", "--no-compact-greedy")[1])

    def test_mean_mask_decode_can_be_disabled(self):
        key = "LLAMA_INKLING2_MEAN_MASK_DECODE"
        self.assertNotIn(key, plan("--no-mean-mask-decode")[1])
        self.assertEqual(plan("--mean-mask-decode")[1][key], "1")

    def test_packs_folder_model_and_projector_overrides(self):
        argv, _ = plan("--packs", "/data/packs")
        self.assertEqual(argv[1:5], ["-m", "/data/packs/Inkling-2-Small-abliterated-e4853745-Q3_K_M-imatrix.gguf",
                                     "--mmproj", "/data/packs/mmproj-F16.gguf"])
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
                        ["--host", "a\nb"], ["--ubatch-size", "0"], ["--ubatch-size", "2049"]):
            with self.subTest(options=options), self.assertRaises(ValueError):
                plan(*options)

    def test_prompt_step_controls(self):
        argv, _ = plan("--prompt-step", "256", "--prompt-step-ms", "40", "--ubatch-size", "512")
        self.assertEqual(argv[argv.index("-ub") + 1], "512")
        self.assertEqual(argv[argv.index("-b") + 1], "2048")
        self.assertEqual(argv[argv.index("--prompt-step") + 1], "256")
        self.assertEqual(argv[argv.index("--prompt-step-ms") + 1], "40")
        self.assertEqual(argv[argv.index("--prompt-chunk") + 1], "128")
        self.assertNotIn("--prompt-step-ms", plan()[0])

    def test_q3_pack_can_keep_sliding_kv_at_f16(self):
        argv, _ = plan("--kv-cache", "q8_0", "--swa-kv-cache", "f16")
        self.assertEqual(argv[argv.index("-m") + 1], plan()[0][2])
        for side in ("k", "v"):
            self.assertEqual(argv[argv.index("--cache-type-" + side) + 1], "q8_0")
            self.assertEqual(argv[argv.index("--cache-type-" + side + "-swa") + 1], "f16")

    def test_packed_mask_can_be_disabled(self):
        self.assertIn("--attn-mask-bits", plan()[0])
        self.assertNotIn("--attn-mask-bits", plan("--no-attn-mask-bits")[0])
        self.assertIn("--attn-mask-bits", plan("--attn-mask-bits")[0])

    def test_reference_sinks_keep_packed_attention(self):
        argv, env = plan("--reference-sinks")
        self.assertEqual(argv, plan()[0])
        self.assertEqual(env, {**TENSOR4_ENV, "LLAMA_INKLING2_SHIFT_SINKS": "1"})

    def test_graph_retention(self):
        env = plan("--keep-graphs", "32")[1]
        self.assertEqual(env, plan("--kept-graphs", "32")[1])
        self.assertEqual(env["LLAMA_GRAPH_SLOTS"], "32")
        self.assertEqual(env["GGML_META_CAPTURE_GRAPHS"], "38")
        self.assertEqual(env["GGML_CUDA_GRAPH_NODES"], "262144")
        self.assertEqual(env["GGML_CUDA_GRAPH_IDLE_MS"], "600000")
        self.assertEqual(env["LLAMA_RS_FIXED_ROWS"], "1")
        self.assertEqual(env["LLAMA_GRAPH_ANY_SEQS"], "1")
        self.assertNotIn("LLAMA_GRAPH_SLOTS", plan("--keep-graphs", "0")[1])
        for count in ("-1", "257"):
            with self.assertRaises(ValueError):
                plan("--keep-graphs", count)

    def test_prompt_shapes_and_quantized_kv_knobs(self):
        env = plan("--prompt-shapes", "8", "--prompt-shape-rows", "256")[1]
        self.assertEqual(env["LLAMA_PROMPT_SHAPES"], "1")
        self.assertEqual(env["LLAMA_PROMPT_SHAPE_SMALL"], "8")
        self.assertEqual(env["LLAMA_PROMPT_SHAPE_ROWS"], "256")
        with self.assertRaises(ValueError):
            plan("--prompt-shape-rows", "256")
        self.assertEqual(plan("--kv-cache", "q8_0")[1]["GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE"], "1")
        self.assertEqual(plan("--kv-cache", "q4_0")[1]["GGML_CUDA_FATTN_REL_DECODE_Q4"], "1")

    def test_expert_row_limit_only_overrides_sm120_kquants(self):
        for count in ("8", "23", "32"):
            env = plan("--expert-row-limit", count)[1]
            self.assertEqual(env, {**TENSOR4_ENV, "GGML_CUDA_MMVQ_KQUANT_MMID_MAX": count})
        argv, env = plan("--profile", "tensor-4", "--expert-row-limit", "16", "--checkpoint-every", "8192")
        self.assertEqual(env["GGML_CUDA_MMVQ_KQUANT_MMID_MAX"], "16")
        self.assertEqual(argv[argv.index("--checkpoint-every") + 1], "8192")
        for count in ("7", "33"):
            with self.assertRaises(ValueError):
                plan("--expert-row-limit", count)

    def test_slot_exchange_controls(self):
        argv, env = plan("--cache-ram", "262144", "--save-async", "--restore-async", "--ctx-checkpoints", "8", "--checkpoint-every", "8192")
        self.assertEqual(argv[argv.index("--cache-ram") + 1], "262144")
        self.assertEqual(argv[argv.index("--ctx-checkpoints") + 1], "8")
        self.assertEqual(argv[argv.index("--checkpoint-every") + 1], "8192")
        with self.assertRaises(ValueError):
            plan("--ctx-checkpoints", "-1")
        with self.assertRaises(ValueError):
            plan("--checkpoint-every", "-1")
        self.assertIn("--no-cache-idle-slots", argv)
        self.assertEqual(env, TENSOR4_ENV)
        self.assertNotIn("LLAMA_STATE_SAVE_ASYNC", plan("--save-sync")[1])
        self.assertEqual(plan("--save-sync")[1]["LLAMA_STATE_RESTORE_ASYNC"], "1")
        self.assertNotIn("LLAMA_STATE_RESTORE_ASYNC", plan("--restore-sync")[1])
        self.assertEqual(plan("--restore-sync")[1]["LLAMA_STATE_SAVE_ASYNC"], "1")
        for key in ("LLAMA_STATE_SAVE_ASYNC", "LLAMA_STATE_RESTORE_ASYNC"):
            self.assertNotIn(key, plan("--cache-ram", "0")[1])

    def test_reduction_wire_and_route(self):
        env = plan("--reduction-wire", "q8")[1]
        self.assertEqual(env["GGML_CUDA_AR_WIRE_Q8"], "1")
        self.assertEqual(plan("--reduction-route", "host")[1], TENSOR4_ENV)
        self.assertEqual(plan("--reduction-route", "host-sliced")[1], {**TENSOR4_ENV, "GGML_CUDA_AR_GATHER_HOST_SLICES": "1"})
        env = plan("--reduction-route", "engine")[1]
        for key in HOST_ROUTE:
            self.assertNotIn(key, env)
        nccl_env = plan("--reduction-route", "engine", "--nccl-sliced-gather")[1]
        self.assertEqual(nccl_env, {**env, "GGML_CUDA_NCCL_GATHER_SLICES": "1", "LLAMA_INKLING2_GATHER_TRANSPOSE": "1"})
        self.assertEqual(plan("--host-owner-rows")[1], {**TENSOR4_ENV, "GGML_CUDA_AR_HOST_OWNER_ROWS": "1"})
        for incompatible in (("--nccl-sliced-gather",), ("--host-owner-rows", "--reduction-route", "engine"),
                             ("--host-owner-rows", "--reduction-wire", "q8")):
            with self.assertRaises(ValueError):
                plan(*incompatible)


class DryRunTest(unittest.TestCase):
    def dry_run(self, *options, env=None):
        run = subprocess.run([sys.executable, str(Path(serve.__file__)), "--dry-run", *options],
                             capture_output=True, text=True, env=env, check=True)
        return json.loads(run.stdout)

    def test_dry_run_prints_the_pinned_plan_and_starts_nothing(self):
        clean = {"PATH": os.environ.get("PATH", "")}
        out = self.dry_run(env=clean)
        self.assertEqual(out, {"argv": plan()[0], "env": TENSOR4_ENV, "removed": []})
        self.assertEqual(out, self.dry_run(env=clean))

    def test_qualifier_dry_run_uses_tensor_4_and_honors_opt_outs(self):
        options = ["--profile", "tensor-4", "--no-prefix-copy", "--no-mean-mask-decode", "--no-host-owner-rows", "--no-bucket-prefill-kv", "--stable-prefill",
                   "--expert-row-limit", "auto", "--checkpoint-every", "0"]
        run = subprocess.run([sys.executable, str(Path(serve.__file__).with_name("qualify.py")), "--dry-run", "--out", "/unused",
                              "--routes", "host-sliced", "engine", *options], capture_output=True, text=True, check=True)
        for route, launch in json.loads(run.stdout)["routes"].items():
            argv, env = plan(*options, "--reduction-route", route, "--log-verbosity", "4")
            self.assertEqual((launch["argv"], launch["settings"]), (argv, env))

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
