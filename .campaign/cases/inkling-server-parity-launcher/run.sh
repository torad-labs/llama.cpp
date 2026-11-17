#!/bin/sh
# Cases for inkling-server-parity-launcher: scripts/inkling2 holds a launcher in the pattern of
# scripts/glm53 (serve.py, whose --dry-run prints the plan as JSON with "argv" and "env" and starts
# nothing). Its named profiles are the ones the capacity study proposes for four cards
# (docs/torad/research/inkling-2-small/05-capacity-and-tensor4.md): tensor-4, tensor-4-capacity and
# tensor-4-quality. Each dry-runs to a pinned plan carrying flash attention, the prompt step
# controls, a unified cache and reserved sessions with the study's values and the local packs the
# conversion work will produce; unit tests in the same folder hold the plans.
# Options are read by any of their server spellings, or the server's LLAMA_ARG_* variable.
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT" || { echo "run.sh: cannot cd to $ROOT" >&2; exit 2; }

U=d3c3e242c16b8cbf51400fad410cd69b22d64f35
T=$(mktemp -d /tmp/inkling-server-parity-launcher.XXXXXX)
LAUNCHER=scripts/inkling2/serve.py

fail() { printf '%s\n' "$*" > "$REASON"; exit 1; }

require_merged() {
    git cat-file -e "$U^{commit}" 2>/dev/null || fail "upstream commit $U is not in this repository"
    git merge-base --is-ancestor "$U" HEAD || fail "upstream $U is not merged into HEAD"
}

cat > "$T/plan.py" <<'PY'
import json, sys, os

ALIASES = {
    "model": (["-m", "--model"], "LLAMA_ARG_MODEL"),
    "mmproj": (["-mm", "--mmproj"], "LLAMA_ARG_MMPROJ"),
    "split_mode": (["-sm", "--split-mode"], "LLAMA_ARG_SPLIT_MODE"),
    "tensor_split": (["-ts", "--tensor-split"], "LLAMA_ARG_TENSOR_SPLIT"),
    "device": (["-dev", "--device"], "LLAMA_ARG_DEVICE"),
    "flash_attn": (["-fa", "--flash-attn"], "LLAMA_ARG_FLASH_ATTN"),
    "ctx": (["-c", "--ctx-size"], "LLAMA_ARG_CTX_SIZE"),
    "parallel": (["-np", "--parallel"], "LLAMA_ARG_N_PARALLEL"),
    "batch": (["-b", "--batch-size"], "LLAMA_ARG_BATCH"),
    "prompt_step": (["--prompt-step"], "LLAMA_ARG_PROMPT_STEP"),
    "prompt_chunk": (["--prompt-chunk"], "LLAMA_ARG_PROMPT_CHUNK"),
    "cache_k": (["-ctk", "--cache-type-k"], "LLAMA_ARG_CACHE_TYPE_K"),
    "cache_v": (["-ctv", "--cache-type-v"], "LLAMA_ARG_CACHE_TYPE_V"),
    "cache_idx": (["-ctki", "--cache-type-idx"], "LLAMA_ARG_CACHE_TYPE_IDX"),
    "cache_ram": (["-cram", "--cache-ram"], "LLAMA_ARG_CACHE_RAM"),
    "temp": (["--temp", "--temperature"], "LLAMA_ARG_TEMP"),
    "top_p": (["--top-p"], "LLAMA_ARG_TOP_P"),
    "top_k": (["--top-k"], "LLAMA_ARG_TOP_K"),
    "min_p": (["--min-p"], "LLAMA_ARG_MIN_P"),
}

def die(msg):
    print(msg)
    sys.exit(1)

def load(path):
    try:
        with open(path) as f:
            plan = json.load(f)
    except Exception as e:
        die(f"the dry run did not print one JSON plan: {e}")
    if not isinstance(plan, dict) or not isinstance(plan.get("argv"), list) or not isinstance(plan.get("env"), dict):
        die("the dry-run plan has no argv list and env object")
    if not all(isinstance(a, str) for a in plan["argv"]) or not all(isinstance(v, str) for v in plan["env"].values()):
        die("the dry-run plan's argv and env values are not all strings")
    return plan["argv"], plan["env"]

def opt(argv, env, name):
    """The last value given for the option, by flag (spaced or with =) or by its LLAMA_ARG_ variable; None if absent."""
    flags, var = ALIASES[name]
    value = None
    for i, a in enumerate(argv[1:], start=1):
        for fl in flags:
            if a == fl and i + 1 < len(argv):
                value = argv[i + 1]
            elif a.startswith(fl + "="):
                value = a[len(fl) + 1:]
    if value is None and var in env:
        value = env[var]
    return value

def flag(argv, env, flags, var):
    if any(a in flags for a in argv[1:]):
        return True
    return env.get(var, "").lower() in ("1", "true", "on", "enabled")

def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def cards(argv, env):
    ts = opt(argv, env, "tensor_split")
    if ts:
        parts = [p for p in ts.replace("/", ",").split(",") if p.strip()]
        vals = [num(p) for p in parts]
        if None in vals or len(set(vals)) != 1:
            die(f"the tensor split {ts} is not equal parts")
        return len(parts)
    dev = opt(argv, env, "device")
    if dev:
        return len([p for p in dev.replace("/", ",").split(",") if p.strip()])
    vis = env.get("CUDA_VISIBLE_DEVICES")
    if vis:
        return len([p for p in vis.split(",") if p.strip()])
    die("the plan names no cards: no tensor split, no device list, no CUDA_VISIBLE_DEVICES")

def expect(argv, env, name, want, label):
    got = opt(argv, env, name)
    if got is None:
        die(f"the plan gives no {label}")
    if num(want) is not None:
        if num(got) != num(want):
            die(f"the plan's {label} is {got}, not {want}")
    elif got != want:
        die(f"the plan's {label} is {got}, not {want}")

def main():
    case = sys.argv[1]
    paths = sys.argv[2:]
    plans = [load(p) for p in paths]
    argv, env = plans[0]
    if case == "split":
        sm = opt(argv, env, "split_mode")
        if sm != "tensor":
            die(f"split mode is {sm}, not tensor")
        n = cards(argv, env)
        if n != 4:
            die(f"the plan splits over {n} cards, not four")
    elif case == "fa":
        fa = opt(argv, env, "flash_attn")
        if fa not in ("on", "1", "true", "enabled"):
            die(f"flash attention is {fa}, not on")
    elif case == "unified":
        if not flag(argv, env, ("-kvu", "--kv-unified"), "LLAMA_ARG_KV_UNIFIED"):
            die("the plan does not ask for a unified KV cache (--kv-unified)")
        if any(a in ("-no-kvu", "--no-kv-unified") for a in argv[1:]):
            die("the plan turns the unified KV cache off")
    elif case == "slots":
        expect(argv, env, "parallel", "23", "reserved sessions (-np)")
    elif case == "pool":
        expect(argv, env, "ctx", "2359296", "pool (-c)")
    elif case == "session_limit":
        # a session's limit: an option or variable about a slot's or a sequence's tokens set to 262144
        # (the fork's --kv-unified-per-slot, LLAMA_KV_SEQ_CELLS, or one like them)
        given = [(argv[i - 1], argv[i]) for i in range(2, len(argv))]
        given += [(a.split("=", 1)[0], a.split("=", 1)[1]) for a in argv[1:] if a.startswith("--") and "=" in a]
        given += list(env.items())
        if not any(v == "262144" and any(w in k.lower() for w in ("slot", "seq", "session")) for k, v in given):
            die("no slot, sequence or session option or variable in the plan sets the 262144-token session limit")
    elif case == "prompt_step":
        expect(argv, env, "prompt_step", "1024", "prompt step (--prompt-step)")
    elif case == "prompt_chunk":
        expect(argv, env, "prompt_chunk", "128", "prompt chunk (--prompt-chunk)")
    elif case == "batch":
        expect(argv, env, "batch", "2048", "batch (-b)")
    elif case == "cache_ram":
        expect(argv, env, "cache_ram", "32768", "prompt cache RAM (--cache-ram)")
    elif case == "kv_f16":
        for name, label in (("cache_k", "K cache type"), ("cache_v", "V cache type")):
            got = opt(argv, env, name)
            if got is not None and got.lower() != "f16":
                die(f"the plan's {label} is {got}, not f16")
    elif case == "kv_q8":
        for name, label in (("cache_k", "K cache type"), ("cache_v", "V cache type")):
            got = opt(argv, env, name)
            if got is None or got.lower() != "q8_0":
                die(f"the plan's {label} is {got}, not q8_0")
    elif case == "wire_bf16":
        if env.get("GGML_CUDA_AR_WIRE_Q8", "0") not in ("0", ""):
            die("the plan sends reductions on the 8-bit wire (GGML_CUDA_AR_WIRE_Q8), not BF16")
    elif case == "no_glm":
        found = []
        if opt(argv, env, "cache_idx") is not None:
            found.append("an indexer cache type (-ctki)")
        for a in argv[1:]:
            if a.startswith("--spec-draft-mtp") or a == "draft-mtp":
                found.append(a)
            if "glm" in a.lower():
                found.append(a)
        for k in ("LLAMA_INDEXER_Q8_QUERY", "LLAMA_MTP_DEVICE_H", "LLAMA_SPEC_COUPLED"):
            if k in env:
                found.append(k)
        if found:
            die("the plan carries GLM-only settings: " + ", ".join(sorted(set(found))))
    elif case == "sampling":
        want = {"temp": 1.0, "top_p": 1.0, "top_k": 0.0, "min_p": 0.0}
        for name, w in want.items():
            got = opt(argv, env, name)
            if got is not None and num(got) != w:
                die(f"the plan sets {name} to {got}, not {w:g}")
    elif case == "pack":
        m = opt(argv, env, "model")
        if not m:
            die("the plan names no model pack (-m)")
        if not os.path.isabs(m) or not m.endswith(".gguf"):
            die(f"the plan's model {m} is not an absolute path to a GGUF")
    elif case == "mmproj":
        mm = opt(argv, env, "mmproj")
        if not mm:
            die("the plan names no vision projector (--mmproj)")
        if not os.path.isabs(mm) or not mm.endswith(".gguf"):
            die(f"the plan's projector {mm} is not an absolute path to a GGUF")
    elif case == "defaults":
        fa = opt(argv, env, "flash_attn")
        if fa not in ("on", "1", "true", "enabled"):
            die(f"flash attention is {fa}, not on")
        if not flag(argv, env, ("-kvu", "--kv-unified"), "LLAMA_ARG_KV_UNIFIED"):
            die("the plan does not ask for a unified KV cache")
        np_ = num(opt(argv, env, "parallel"))
        if not np_ or np_ < 1:
            die("the plan reserves no sessions (-np)")
        if opt(argv, env, "prompt_step") is None:
            die("the plan gives no prompt step (--prompt-step)")
        n = cards(argv, env)
        if n != 4 or opt(argv, env, "split_mode") != "tensor":
            die(f"the plan is not a tensor split over four cards ({n} cards)")
    elif case == "no_stock":
        for p, (a, e) in zip(paths, plans):
            for name in ("model", "mmproj"):
                v = opt(a, e, name)
                if v and "unsloth" in v.lower():
                    die(f"{os.path.basename(p)}: {name} {v} is a stock Unsloth pack")
    elif case == "distinct_packs":
        ms = [opt(a, e, "model") for a, e in plans]
        if None in ms or len(set(ms)) != len(ms):
            die("the profiles do not each name a pack of their own: " + ", ".join(map(str, ms)))
    elif case == "larger_pool":
        c0 = num(opt(*plans[0], "ctx")); c1 = num(opt(*plans[1], "ctx"))
        if c0 is None or c1 is None or not c1 > c0:
            die(f"tensor-4-capacity's pool {c1} is not larger than tensor-4's {c0}")
    elif case == "smaller_pool":
        c0 = num(opt(*plans[0], "ctx")); c1 = num(opt(*plans[1], "ctx"))
        if c0 is None or c1 is None or not c1 < c0:
            die(f"tensor-4-quality's pool {c1} is not smaller than tensor-4's {c0}")
    else:
        die("unknown check " + case)

main()
PY

# One dry run of a profile, kept for the cases that read it; fails the case if it does not run
dry() {
    require_merged
    [ -f "$LAUNCHER" ] || fail "$LAUNCHER does not exist"
    out="$T/plan-$1.json"
    if [ ! -f "$out.rc" ]; then
        python3 "$LAUNCHER" --profile "$1" --dry-run > "$out" 2> "$out.err"
        echo $? > "$out.rc"
    fi
    rc=$(cat "$out.rc")
    [ "$rc" -eq 0 ] || fail "$LAUNCHER --profile $1 --dry-run exited $rc: $(tail -n 1 "$out.err")"
    PLAN=$out
}

check() { # check <what> <plan files...>
    what=$1; shift
    msg=$(python3 "$T/plan.py" "$what" "$@" 2>&1) || fail "${msg:-the plan check crashed}"
}

on_profile() { dry "$1"; check "$2" "$PLAN"; }

case_unit_tests_pass() {
    require_merged
    [ -d scripts/inkling2 ] || fail "scripts/inkling2 does not exist"
    ( cd scripts/inkling2 && python3 -m unittest discover -p 'test_*.py' ) > "$T/unit.out" 2>&1
    rc=$?
    ran=$(sed -n -E 's/^Ran ([0-9]+) tests?.*/\1/p' "$T/unit.out" | tail -n 1)
    [ "$rc" -eq 0 ] || fail "the unit tests in scripts/inkling2 exited $rc: $(grep -m1 -E '^(FAIL|ERROR):' "$T/unit.out" || tail -n 1 "$T/unit.out")"
    [ -n "$ran" ] && [ "$ran" -gt 0 ] || fail "no unit test ran in scripts/inkling2"
}

case_dry_run_is_pinned() {
    dry tensor-4
    python3 "$LAUNCHER" --profile tensor-4 --dry-run > "$T/again.json" 2>/dev/null || fail "a second dry run of tensor-4 failed"
    cmp -s "$PLAN" "$T/again.json" || fail "two dry runs of tensor-4 printed different plans"
}

case_tensor4_split() { on_profile tensor-4 split; }
case_tensor4_fa() { on_profile tensor-4 fa; }
case_tensor4_unified() { on_profile tensor-4 unified; }
case_tensor4_slots() { on_profile tensor-4 slots; }
case_tensor4_pool() { on_profile tensor-4 pool; }
case_tensor4_session_limit() { on_profile tensor-4 session_limit; }
case_tensor4_prompt_step() { on_profile tensor-4 prompt_step; }
case_tensor4_prompt_chunk() { on_profile tensor-4 prompt_chunk; }
case_tensor4_batch() { on_profile tensor-4 batch; }
case_tensor4_cache_ram() { on_profile tensor-4 cache_ram; }
case_tensor4_kv_f16() { on_profile tensor-4 kv_f16; }
case_tensor4_wire_bf16() { on_profile tensor-4 wire_bf16; }
case_tensor4_no_glm() { on_profile tensor-4 no_glm; }
case_tensor4_sampling() { on_profile tensor-4 sampling; }
case_tensor4_pack() { on_profile tensor-4 pack; }
case_tensor4_mmproj() { on_profile tensor-4 mmproj; }

case_capacity_defaults() { on_profile tensor-4-capacity defaults; }
case_capacity_kv_q8() { on_profile tensor-4-capacity kv_q8; }
case_capacity_larger_pool() { dry tensor-4; a=$PLAN; dry tensor-4-capacity; check larger_pool "$a" "$PLAN"; }
case_capacity_pack() { on_profile tensor-4-capacity pack; }

case_quality_defaults() { on_profile tensor-4-quality defaults; }
case_quality_smaller_pool() { dry tensor-4; a=$PLAN; dry tensor-4-quality; check smaller_pool "$a" "$PLAN"; }
case_quality_pack() { on_profile tensor-4-quality pack; }

case_no_stock_packs() {
    dry tensor-4; a=$PLAN; dry tensor-4-capacity; b=$PLAN; dry tensor-4-quality; c=$PLAN
    check no_stock "$a" "$b" "$c"
}
case_distinct_packs() {
    dry tensor-4; a=$PLAN; dry tensor-4-capacity; b=$PLAN; dry tensor-4-quality; c=$PLAN
    check distinct_packs "$a" "$b" "$c"
}

run_case() {
    name=$1; fn=$2
    REASON="$T/$name.reason"; : > "$REASON"
    ( set +e; "$fn" ) > "$T/$name.log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "PASS $name"
    else
        why=$(head -n 1 "$REASON" | tr -d '\r')
        [ -n "$why" ] || why="exited $rc: $(tail -n 1 "$T/$name.log" | tr -d '\r')"
        echo "FAIL $name: $why"
    fi
}

run_case inkling2-launcher-unit-tests-pass                case_unit_tests_pass
run_case inkling2-dry-run-plan-is-pinned                  case_dry_run_is_pinned
run_case tensor-4-splits-tensor-over-four-equal-cards     case_tensor4_split
run_case tensor-4-flash-attention-on                      case_tensor4_fa
run_case tensor-4-unified-kv-cache                        case_tensor4_unified
run_case tensor-4-reserves-23-sessions                    case_tensor4_slots
run_case tensor-4-pool-of-2359296-tokens                  case_tensor4_pool
run_case tensor-4-session-limit-262144-tokens             case_tensor4_session_limit
run_case tensor-4-prompt-step-1024                        case_tensor4_prompt_step
run_case tensor-4-prompt-chunk-128                        case_tensor4_prompt_chunk
run_case tensor-4-batch-2048                              case_tensor4_batch
run_case tensor-4-prompt-cache-ram-32768-mib              case_tensor4_cache_ram
run_case tensor-4-kv-cache-f16                            case_tensor4_kv_f16
run_case tensor-4-reduction-wire-bf16                     case_tensor4_wire_bf16
run_case tensor-4-carries-no-glm-only-settings            case_tensor4_no_glm
run_case tensor-4-sampling-defaults-match-model-card      case_tensor4_sampling
run_case tensor-4-names-a-local-gguf-pack                 case_tensor4_pack
run_case tensor-4-names-a-vision-projector                case_tensor4_mmproj
run_case tensor-4-capacity-carries-serving-defaults       case_capacity_defaults
run_case tensor-4-capacity-kv-cache-q8_0                  case_capacity_kv_q8
run_case tensor-4-capacity-pool-larger-than-tensor-4      case_capacity_larger_pool
run_case tensor-4-capacity-names-a-local-gguf-pack        case_capacity_pack
run_case tensor-4-quality-carries-serving-defaults        case_quality_defaults
run_case tensor-4-quality-pool-smaller-than-tensor-4      case_quality_smaller_pool
run_case tensor-4-quality-names-a-local-gguf-pack         case_quality_pack
run_case inkling2-profiles-name-no-stock-unsloth-pack     case_no_stock_packs
run_case inkling2-profiles-name-distinct-packs            case_distinct_packs

rm -rf "$T"
exit 0
