"""Pinned same-pack GLM qualification helpers; no provider operations.

Keep the device matrix executable before renting. Byte verification is read-only:
a missing or changed retained shard fails instead of starting a new download.
"""
import csv
import hashlib
import io
import math

BASELINE_COMMIT = "776b8f577dec45f2fba79b6a41ae8b782f5fc3b3"
BASELINE_ARCHIVE = "48358002368846e49d93da500a852f8b6fb51a53655d563fdbdacddb6c8aa4ef"


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16*1024*1024), b""):
            sha.update(chunk)
    return sha.hexdigest()

def validate_device_csv(output, count):
    records = csv.DictReader(io.StringIO("\n".join(line for line in output.splitlines() if line.startswith('"'))))
    rows = [row for row in records if row.get("backend_name") == "CUDA0"]
    if (not rows or count is not None and len(rows) != count
            or any(row.get("supported") not in ("true", "1") or row.get("passed") not in ("true", "1")
                   or row.get("error_message") for row in rows)):
        raise ValueError(f"missing, unsupported or failed CUDA cases ({len(rows)})")
    return True

def verify_files(root, files):
    verified = {}
    for name, expected in files.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"file escapes the declared root: {name}")
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError(f"file identity mismatch: {path}")
        verified[name] = expected
    return verified

def check_run_identity(meta, expected, context, ubatch, profile):
    settings = dict(context=context, ubatch=ubatch, slots=3, depth=2,
                    sampling="served", repeats=3, tokens=384, adaptive=0.18)
    manifest = meta.get("binary_manifest", {})
    engine_env = {k: v for k, v in meta.get("engine_env", {}).items() if k.startswith(("GGML_", "LLAMA_"))}
    expected_env = {"GGML_CUDA_ALLREDUCE": "internal", "GGML_CUDA_GRAPH_MAX": "0"} | expected.get("resolved_env", {})
    if (any(meta.get("args", {}).get(k) != v for k, v in settings.items())
            or any(manifest.get(k) != expected[k] for k in ("commit", "tarball_sha256"))
            or meta.get("model_sha256") != expected["model_sha256"]
            or meta.get("profile") != profile or meta.get("instrumented") is not False
            or engine_env != expected_env):
        raise ValueError("serving configuration or artifact identity changed")
    return True

def check_quality(records):
    if set(records) != {"2048", "4096", "16384"}:
        raise ValueError("all three distribution contexts are required")
    for ctx, summary in records.items():
        kld, top = summary.get("meanKld"), summary.get("sameTopP")
        if (not isinstance(kld, (float, int)) or not math.isfinite(kld) or abs(kld) > 1e-6
                or not isinstance(top, (float, int)) or not math.isfinite(top) or not 99.99 <= top <= 100):
            raise ValueError(f"distribution contract failed at context {ctx}")
    return True

def device_cases():
    cases = []

    def add(name, argv, env=None, count=None, require=None):
        cases.append(dict(name=name, argv=argv, env=env or {}, cases=count, require=require))

    def ops(name, op, env, count=None, pattern=None):
        argv = ["test-backend-ops", "test", "-b", "CUDA0", "-o", op, "--output", "csv", "-j", "1"]
        if pattern:
            argv += ["-p", pattern]
        add(name, argv, env, count)

    for gpu in (0, 1):
        card = {"CUDA_VISIBLE_DEVICES": str(gpu)}
        add(f"q8-storage-gpu{gpu}", ["test-q8-backend-storage", "CUDA0"], card)
        add(f"q8-meta-fallback-gpu{gpu}", ["test-q8-backend-storage", "CUDA0", "--meta"],
            card | {"GGML_CUDA_Q8_SPLIT": "1"}, require="2/2 meta packed reader fallbacks passed")
        for mode in ("direct", "tma2", "tma4"):
            env = card | {"GGML_CUDA_Q8_SPLIT": "1", "GGML_CUDA_Q8_TMA": "1" if mode.startswith("tma") else "0",
                          "GGML_CUDA_Q8_TMA_SLOTS": "4" if mode == "tma4" else "2",
                          "GGML_CUDA_MMVQ_Q8_SM120_CHECK": "1"}
            ops(f"q8-readers-{mode}-gpu{gpu}", "Q8_WEIGHT_READERS", env, 203)
        for compute in ("f16", "bf16", "f32"):
            add(f"q8-cublas-{compute}-gpu{gpu}", ["test-q8-backend-storage", "CUDA0", "--cublas"],
                card | {"GGML_CUDA_Q8_SPLIT": "1", "GGML_CUDA_CUBLAS_COMPUTE_TYPE": compute},
                require="8/8 cuBLAS packed/canonical comparisons passed")
        for unroll in (0, 1, 2, 4):
            for kind, op, pattern, count in (
                ("q8", "MUL_MAT", "type_a=q8_0,type_b=f32,.*n=(1|2|3|6|9),", 106),
                ("fused", "MUL_MAT_VEC_FUSION", "type=q8_0", 94),
                ("concat", "MUL_MAT_CONCAT", "type=q8_0", 17)):
                ops(f"q8-{kind}-u{unroll}-gpu{gpu}", op,
                    card | {"GGML_CUDA_Q8_SPLIT": "0", "GGML_CUDA_Q8_TMA": "0",
                            "GGML_CUDA_MMVQ_Q8_SM120_CHECK": "1", "GGML_CUDA_MMVQ_Q8_SM120_UNROLL": str(unroll)},
                    count, pattern)
        for columns in (1, 2, 4):
            for front in (0, 1) if columns > 1 else (0,):
                ops(f"kda-c{columns}-f{front}-gpu{gpu}", "GATED_DELTA_NET,GATED_DELTA_NET_CACHE_FUSION",
                    card | {"GGML_CUDA_KDA_COLUMNS": str(columns), "GGML_CUDA_KDA_FRONT_SHARED": str(front)},
                    pattern="kda=1")
        for slots in (2, 4, 6, 8):
            ops(f"iq3-s{slots}-gpu{gpu}", "GLM53_MOE_ROUTES",
                card | {"GGML_CUDA_IQ3_COMPACT": "1", "GGML_CUDA_IQ3_SLOTS": str(slots)},
                72 if slots == 8 else 24, None if slots == 8 else "n_embd=512,n_ff=256")
        for publication in (0, 1):
            ops(f"hc-publication-{publication}-gpu{gpu}", "DSV4_HC_PRE_FUSED,DSV4_HC_PRE_Q8_1,DSV4_HC_PRE_POST",
                card | {"GGML_CUDA_HC_FRONT_ONE": "1", "GGML_CUDA_HC_CTA_PUBLISH": str(publication)})
    for route, settings in (("peer", {}), ("host", {"GGML_CUDA_AR_P2P_MAX_BYTES": "0"}),
                            ("mixed", {"GGML_CUDA_AR_P2P_MAX_BYTES": "8192"}),
                            ("duplicate", {"CUDA_VISIBLE_DEVICES": "0"})):
        add(f"capture-{route}", ["test-backend-meta-capture"], settings | {"TEST_GLM_HC": "1"},
            require="device inboxes exercised" if route == "peer" else "OK")
    for name, argv in (
        ("glm-architecture", ["test-llama-archs", "-a", "glm5next", "-s", "1234"]),
        ("qwen-architecture", ["test-llama-archs", "-a", "qwen35", "-s", "1234"]),
        ("speculative", ["test-speculative-mtp"]), ("sampling", ["test-sampling"]),
        ("meta-sampler", ["test-backend-meta-sampler"]), ("meta-sourceless", ["test-backend-meta-sourceless"]),
        ("meta-capacity", ["test-backend-meta-capacity", "--gpu"]),
        ("recurrent-arena", ["test-recurrent-arena"]), ("server-counters", ["test-server-speculation-stats"])):
        add(name, argv)
    # Device ownership fallback and ordinary-cache fallback use a fresh process.
    add("glm-ordinary-cache", ["test-llama-archs", "-a", "glm5next", "-s", "1234"],
        {"GGML_CUDA_GDN_STATE_PERSIST": "0"})
    return cases
