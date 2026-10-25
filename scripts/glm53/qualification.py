"""Pinned same-pack GLM qualification helpers; no provider operations.

Keep the device matrix executable before renting. Byte verification is read-only:
a missing or changed retained shard fails instead of starting a new download.
"""
import csv
import hashlib
import io
import math
import re

BASELINE_COMMIT = "776b8f577dec45f2fba79b6a41ae8b782f5fc3b3"
BASELINE_ARCHIVE = "48358002368846e49d93da500a852f8b6fb51a53655d563fdbdacddb6c8aa4ef"


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16*1024*1024), b""):
            sha.update(chunk)
    return sha.hexdigest()

def validate_device_csv(output, count):
    # An unsupported case is labelled with every backend that rejected one of its
    # tensors, and a mismatch prints its diagnostic ahead of the row on its line.
    lines = []
    for line in output.splitlines():
        if not line.startswith('"') and '"CUDA0",' in line:
            line = line[line.index('"CUDA0",'):]
        if line.startswith('"'):
            lines.append(line)
    rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    bad = [row for row in rows if row.get("backend_name") != "CUDA0" or row.get("supported") not in ("true", "1")
           or row.get("passed") not in ("true", "1") or row.get("error_message")]
    if not isinstance(count, int) or len(rows) != count or bad:
        named = "; ".join(f"{row.get('backend_name')} {row.get('op_name')} {row.get('op_params')}: "
                          f"{row.get('error_message') or 'incomplete row'}" for row in bad[:5])
        raise ValueError(f"{len(rows)} cases reported for {count} required, {len(bad)} not passed on CUDA0"
                         + (f": {named}" if named else ""))
    return True

def validate_arch_table(output, arch, devices):
    # test-llama-archs exits 0 when every row is SKIP. Each single device must match
    # the CPU logits and its save/load roundtrip; both tensor splits must match too,
    # and a tensor split has no roundtrip by design. The binary prints OK unless the
    # NMSE exceeds 1e-4, which a NaN never does, so the value is read here.
    def nmse_ok(cell):
        match = re.fullmatch(r"OK \(([^)]*)\)", cell)
        try:
            return bool(match) and math.isfinite(float(match[1])) and float(match[1]) <= 1e-4
        except ValueError:
            return False
    single, splits = [], []
    for line in re.sub(r"\x1b\[[0-9;]*m", "", output).splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) == 7 and cells[1] == arch:
            (splits if cells[2] in ("Meta", "Meta 3:7") else single).append(cells[2:6])
    bad = [row for row in single + splits if not nmse_ok(row[2])]
    bad += [row for row in single if row[3] != "OK"] + [row for row in splits if row[3] not in ("OK", "SKIP")]
    if len(single) != devices or sorted(row[0] for row in splits) != ["Meta", "Meta 3:7"] or bad:
        raise ValueError(f"{arch}: {len(single)} of {devices} devices and {len(splits)} of 2 tensor splits reported"
                         + "".join(f"; {device} {config}: {nmse or 'no result'}, roundtrip {trip or 'none'}"
                                   for device, config, nmse, trip in bad))
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

def check_stop_at(stop_at, now, max_hours):
    # The provider stop is armed before the controller starts, inside the hours approved for the window.
    if not 0 < stop_at - now <= max_hours * 3600:
        raise ValueError(f"stop-at must be the armed deadline, at most {max_hours:g} hour{'' if max_hours == 1 else 's'} away")
    return True


def check_quality_context(ctx, summary):
    kld = summary.get("meanKld") if isinstance(summary, dict) else None
    top = summary.get("sameTopP") if isinstance(summary, dict) else None
    if (not isinstance(kld, (float, int)) or not math.isfinite(kld) or abs(kld) > 1e-6
            or not isinstance(top, (float, int)) or not math.isfinite(top) or not 99.99 <= top <= 100):
        raise ValueError(f"distribution contract failed at context {ctx}: meanKld={kld!r}, sameTopP={top!r}")
    return True


def check_quality(records):
    if set(records) != {"2048", "4096", "16384"}:
        raise ValueError("all three distribution contexts are required")
    for ctx, summary in records.items():
        check_quality_context(ctx, summary)
    return True

def device_cases():
    cases = []

    def add(name, argv, env=None, count=None, require=None, expected_failure=None):
        case = dict(name=name, argv=argv, env=env or {}, cases=count, require=require)
        if expected_failure is not None:
            case["expected_failure"] = expected_failure
        cases.append(case)

    def ops(name, op, env, count=None, pattern=None, expected_failure=None):
        argv = ["test-backend-ops", "test", "-b", "CUDA0", "-o", op, "--output", "csv", "-j", "1"]
        if pattern:
            argv += ["-p", pattern]
        add(name, argv, env, count, expected_failure=expected_failure)

    for gpu in (0, 1):
        card = {"CUDA_VISIBLE_DEVICES": str(gpu)}
        add(f"q8-storage-gpu{gpu}", ["test-q8-backend-storage", "CUDA0"], card)
        add(f"q8-meta-fallback-gpu{gpu}", ["test-q8-backend-storage", "CUDA0", "--meta"],
            card | {"GGML_CUDA_Q8_SPLIT": "1"}, require="2/2 meta packed reader fallbacks passed")
        for packing in (0, 1):
            add(f"q8-meta-cache-p{packing}-gpu{gpu}", ["test-q8-backend-storage", "CUDA0", "--meta-cache"],
                card | {"GGML_CUDA_Q8_SPLIT": str(packing)}, require="16/16 meta Q8 cache rebuilds passed")
        add(f"q8-channel-views-gpu{gpu}", ["test-q8-backend-storage", "CUDA0", "--views"],
            card | {"GGML_CUDA_Q8_SPLIT": "1"}, require="4/4 packed channel views passed")
        for mode in ("direct", "tma2", "tma4"):
            env = card | {"GGML_CUDA_Q8_SPLIT": "1", "GGML_CUDA_Q8_TMA": "1" if mode.startswith("tma") else "0",
                          "GGML_CUDA_Q8_TMA_SLOTS": "4" if mode == "tma4" else "2",
                          "GGML_CUDA_MMVQ_Q8_SM120_CHECK": "1"}
            ops(f"q8-readers-{mode}-gpu{gpu}", "Q8_WEIGHT_READERS", env, 203)
        # With TMA eligible, an invalid slot count must fail. Global legacy must
        # bypass TMA entirely, so the same unused setting must then be harmless.
        for legacy in (0, 1):
            ops(f"q8-legacy-priority-{legacy}-gpu{gpu}", "Q8_WEIGHT_READERS",
                card | {"GGML_CUDA_Q8_SPLIT": "1", "GGML_CUDA_Q8_TMA": "1",
                        "GGML_CUDA_Q8_TMA_SLOTS": "invalid", "GGML_CUDA_MMVQ_Q8_SM120_LEGACY": str(legacy),
                        "GGML_CUDA_MMVQ_Q8_SM120_CHECK": "0"}, 203 if legacy else None,
                expected_failure=None if legacy else
                    dict(exit=-6, message="GGML_CUDA_Q8_TMA_SLOTS must be 2 or 4"))
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
        # GLM never builds rows mode (src[6]), which CUDA rejects by design; the
        # fusion's vars end in kda=1, the op's carry rows_mode right after K.
        # That leaves 113 gated-delta-net and 72 cache-fusion cases registered.
        for columns in (1, 2, 4):
            for front in (0, 1) if columns > 1 else (0,):
                ops(f"kda-c{columns}-f{front}-gpu{gpu}", "GATED_DELTA_NET,GATED_DELTA_NET_CACHE_FUSION",
                    card | {"GGML_CUDA_KDA_COLUMNS": str(columns), "GGML_CUDA_KDA_FRONT_SHARED": str(front)},
                    185, "kda=1(,K=[0-9]+,rows_mode=0,|$)")
        for slots in (2, 4, 6, 8):
            ops(f"iq3-s{slots}-gpu{gpu}", "GLM53_MOE_ROUTES",
                card | {"GGML_CUDA_IQ3_COMPACT": "1", "GGML_CUDA_IQ3_SLOTS": str(slots)},
                72)
        for publication in (0, 1):
            ops(f"hc-publication-{publication}-gpu{gpu}", "DSV4_HC_PRE_FUSED,DSV4_HC_PRE_Q8_1,DSV4_HC_PRE_POST",
                card | {"GGML_CUDA_HC_FRONT_ONE": "1", "GGML_CUDA_HC_CTA_PUBLISH": str(publication)}, 65)
    for route, settings in (("peer", {}), ("host", {"GGML_CUDA_AR_P2P_MAX_BYTES": "0"}),
                            ("mixed", {"GGML_CUDA_AR_P2P_MAX_BYTES": "8192"}),
                            ("duplicate", {"CUDA_VISIBLE_DEVICES": "0"})):
        add(f"capture-{route}", ["test-backend-meta-capture"], settings | {"TEST_GLM_HC": "1"},
            require=r"ok +card-to-card limit \d+ bytes: device inboxes exercised" if route == "peer" else "OK")
    # Both meta tests exit 0 when they find no GPU; the sampler names the two cards it ran over.
    for name, argv, require in (
        ("speculative", ["test-speculative-mtp"], None), ("sampling", ["test-sampling"], None),
        ("meta-sampler", ["test-backend-meta-sampler"], "meta backend over CUDA0 and CUDA1"),
        ("meta-sourceless", ["test-backend-meta-sourceless"], "OK"),
        ("meta-capacity", ["test-backend-meta-capacity", "--gpu"], None),
        ("recurrent-arena", ["test-recurrent-arena"], None), ("server-counters", ["test-server-speculation-stats"], None)):
        add(name, argv, require=require)
    # Both cards and the CPU, each alone, then both tensor splits. Device ownership
    # fallback and ordinary-cache fallback use a fresh process.
    for name, arch, env in (("glm-architecture", "glm5next", None), ("qwen-architecture", "qwen35", None),
                            ("glm-ordinary-cache", "glm5next", {"GGML_CUDA_GDN_STATE_PERSIST": "0"})):
        add(name, ["test-llama-archs", "-a", arch, "-s", "1234"], env)
        cases[-1]["table"] = dict(arch=arch, devices=3)
    return cases
