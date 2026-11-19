#!/usr/bin/env python3
"""Qualify the local Inkling Q3_K_M TP4 server. Run on the four-card host; no provisioning or network transfers."""
import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import urllib.error
import urllib.request
from uuid import uuid4

import mixed_prefill
import serve

sys.path.append(str(serve.ROOT / "scripts/glm53"))
import measure as glm_measure

ROUTES = ("host", "host-sliced", "engine")
PHASES = ("chat", "balanced", "long", "restore")
SLOTS = 23
READER = 262144
SEPARATOR = "<<<ARCHIVE>>>"
FACTS = ("OPAL-5837", "MICA-6249")
SAMPLE = ("Trace this Python code. Work through the object references before answering. Return only the final list as JSON. "
          "a = [[n] for n in range(4)]; b = a[::2]; b[1].append(9); "
          "a[0] = [7]; print([sum(x) for x in b] + [sum(a[0])])")
READ_QUESTION = ("Retrieve the two deployment approval identifiers from this repository audit. "
                 "Many routine entries are unrelated. " + SEPARATOR +
                 " What are the approvals for modules elm and ash, in that order? Return only the two identifiers.")


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def tool_roundtrip(args, folder, anthropic):
    endpoint = "/v1/messages" if anthropic else "/v1/chat/completions"
    name = "read_ticket"
    schema = {"type": "object", "properties": {"ticket": {"type": "integer"}}, "required": ["ticket"]}
    tool = {"name": name, "description": "Read the deployment record for the requested ticket."}
    if anthropic:
        tool["input_schema"] = schema
    else:
        tool = {"type": "function", "function": {**tool, "parameters": schema}}
    turns = [{"role": "user", "content": "Read deployment ticket 812 using the tool. Give only its approval number."}]
    body = {"model": args.alias, "messages": turns, "temperature": 0, "max_tokens": args.chat_budget, "tools": [tool]}
    body.update({"output_config": {"effort": "none"}} if anthropic else {"reasoning_effort": "none"})
    response = api(args.url, endpoint, body, args.timeout)
    prefix = "anthropic" if anthropic else "openai"
    save(folder / (prefix + "-tool-request.json"), response)
    receipt = '{"approval":3927}'
    if anthropic:
        calls = [b for b in response["content"] if b["type"] == "tool_use"]
        valid = response["stop_reason"] == "tool_use" and len(calls) == 1 and calls[0]["name"] == name and calls[0]["input"] == {"ticket": 812}
        if valid:
            turns.extend([{"role": "assistant", "content": response["content"]},
                          {"role": "user", "content": [{"type": "tool_result", "tool_use_id": calls[0]["id"], "content": receipt}]}])
    else:
        choice = response["choices"][0]
        calls = choice["message"].get("tool_calls", [])
        valid = (choice["finish_reason"] == "tool_calls" and len(calls) == 1 and calls[0]["function"]["name"] == name
                 and json.loads(calls[0]["function"]["arguments"]) == {"ticket": 812})
        if valid:
            turns.extend([choice["message"], {"role": "tool", "tool_call_id": calls[0]["id"], "content": receipt}])
    answer = ""
    if valid:
        response = api(args.url, endpoint, body, args.timeout)
        save(folder / (prefix + "-tool-result.json"), response)
        if anthropic:
            answer = "".join(b["text"] for b in response["content"] if b["type"] == "text")
            valid = response["stop_reason"] == "end_turn"
        else:
            answer = response["choices"][0]["message"].get("content") or ""
            valid = response["choices"][0]["finish_reason"] == "stop"
    return {"api": endpoint, "answer": answer, "passed": valid and answer.strip() == "3927"}


def api(url, path, body=None, timeout=30):
    request = urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


EFFORT_LEVELS = {"none": "0", "minimal": "0.1", "low": "0.2", "medium": "0.7", "high": "0.9", "xhigh": "0.99", "max": "0.99"}
CHAT_ENDPOINTS = ("chat/completions", "messages", "responses")


def effort_request(endpoint, alias, messages, effort, budget):
    body = {"model": alias, "temperature": 0}
    if endpoint == "responses":
        body.update(input=messages, max_output_tokens=budget)
        if effort is not None:
            body["reasoning"] = {"effort": effort}
    else:
        body.update(messages=messages, max_tokens=budget)
        if effort is not None:
            if endpoint == "messages":
                body["output_config"] = {"effort": effort}
            else:
                body["reasoning_effort"] = effort
    return body


def effort_routes(args):
    cases = []
    rejected = []
    for endpoint in CHAT_ENDPOINTS:
        sample = [{"role": "user", "content": "Test"}]
        invalid = [(str(bad), effort_request(endpoint, args.alias, sample, bad, 1)) for bad in ("turbo", 0.5)]
        if endpoint in ("responses", "messages"):
            wrapper = "reasoning" if endpoint == "responses" else "output_config"
            invalid.append((wrapper, {**effort_request(endpoint, args.alias, sample, None, 1), wrapper: "high"}))
        for field, body in invalid:
            try:
                api(args.url, "/v1/" + endpoint, body, args.timeout)
            except urllib.error.HTTPError as exc:
                if exc.code != 400 or json.load(exc)["error"]["type"] != "invalid_request_error":
                    raise
                rejected.append({"api": endpoint, "invalid": field, "status": exc.code})
            else:
                raise ValueError(f"{endpoint} accepted malformed reasoning input {field!r}")

    for effort in [*EFFORT_LEVELS, None]:
        # Give each level a unique suffix so a cached prompt from a past probe cannot match.
        messages = [{"role": "user", "content": f"Reply OK for request {uuid4().hex}."}]
        fields = {} if effort is None else {"reasoning_effort": effort}
        rendered = api(args.url, "/apply-template", {"messages": messages, **fields})["prompt"]
        value = EFFORT_LEVELS[effort or "high"]
        if rendered.count("Thinking effort level: ") != 1 or f"Thinking effort level: {value}<|end_message|>" not in rendered:
            raise ValueError(f"unexpected template output for effort {effort!r}")
        ids = api(args.url, "/tokenize", {"content": rendered, "parse_special": True, "add_special": True})["tokens"]
        # A full cached prefix checks the effort token IDs even when every effort has the same token count.
        api(args.url, "/completion", {"prompt": ids[:-1], "id_slot": 0, "cache_prompt": False,
                                     "n_predict": 1, "temperature": 0}, args.timeout)
        for endpoint in CHAT_ENDPOINTS:
            body = effort_request(endpoint, args.alias, messages, effort, 1)
            body["id_slot"] = 0
            usage = api(args.url, "/v1/" + endpoint, body, args.timeout)["usage"]
            if endpoint == "messages":
                cached = usage["cache_read_input_tokens"]
                length = cached + usage["input_tokens"]
            elif endpoint == "responses":
                cached = usage["input_tokens_details"]["cached_tokens"]
                length = usage["input_tokens"]
            else:
                cached = usage["prompt_tokens_details"]["cached_tokens"]
                length = usage["prompt_tokens"]
            if length != len(ids) or not len(ids) - 1 <= cached <= len(ids):
                raise ValueError(f"{endpoint} effort {effort!r}: {cached}/{length} tokens reused, expected prefix of {len(ids) - 1}")
            cases.append({"api": endpoint, "effort": effort, "native_level": value, "prompt_tokens": length, "cached_tokens": cached})
    return {"cases": cases, "rejected": rejected}


def chat(args, folder):
    messages = [{"role": "user", "content": SAMPLE}]
    results = []
    levels = effort_routes(args)
    save(folder / "effort-routing.json", levels)
    for effort in ("none", "high"):
        for endpoint in CHAT_ENDPOINTS:
            body = effort_request(endpoint, args.alias, messages, effort, args.chat_budget)
            response = api(args.url, "/v1/" + endpoint, body, args.timeout)
            save(folder / (endpoint.replace("/", "-") + "-" + effort + ".json"), response)
            if endpoint == "messages":
                blocks = response["content"]
                answer = "".join(b["text"] for b in blocks if b["type"] == "text")
                reasoning = sum(len(b.get("thinking", "")) for b in blocks if b["type"] == "thinking")
                finished = response["stop_reason"] == "end_turn"
            elif endpoint == "responses":
                output = response["output"]
                answer = "".join(b["text"] for item in output if item["type"] == "message"
                                 for b in item["content"] if b["type"] == "output_text")
                reasoning = sum(len(b.get("text", "")) for item in output if item["type"] == "reasoning"
                                for b in item.get("summary", []) + item.get("content", []))
                finished = response["status"] == "completed"
            else:
                choice = response["choices"][0]
                answer = choice["message"].get("content") or ""
                reasoning = len(choice["message"].get("reasoning_content") or "")
                finished = choice["finish_reason"] == "stop"
            try:
                parsed = json.loads(answer.strip().removeprefix("```json").removesuffix("```").strip())
            except ValueError:
                parsed = None
            results.append({"api": endpoint, "effort": effort, "answer": answer, "reasoning_chars": reasoning,
                            "finished": finished, "correct": parsed == [0, 11, 7]})
    tools = [tool_roundtrip(args, folder, anthropic) for anthropic in (False, True)]
    native = {x["api"]: x["reasoning_chars"] > 0 for x in results if x["effort"] == "high"}
    return {"cases": results, "rendered_efforts": EFFORT_LEVELS, "effort_routing": levels, "high_reasoning_delivered": native, "tool_roundtrips": tools,
            "passed": all(native.values()) and all(x["finished"] and x["correct"] for x in results)
            and all(x["passed"] for x in tools)}


def balanced(args, folder):
    gate = threading.Barrier(SLOTS)
    def client(slot):
        # Identical lengths, distinct first tokens and fixed slots make repeated routes comparable.
        vocab = args.vocab
        prompt = [vocab[slot % len(vocab)]] + [vocab[(slot + n * 7) % len(vocab)] for n in range(127)]
        body = dict(prompt=prompt, id_slot=slot, temperature=0, cache_prompt=False, ignore_eos=True,
                    n_predict=args.balanced_tokens, stream=True, return_tokens=True)
        gate.wait(timeout=30)
        start = time.monotonic()
        with mixed_prefill.post(args.url, "/completion", body, args.timeout) as response:
            row = glm_measure.read_completion_stream(response, args.balanced_tokens, len(prompt))
        row.update(slot=slot, elapsed=time.monotonic() - start, ttft_ms=(row["bursts"][0]["at"] - start) * 1000)
        timings = row["response"]["timings"]
        if row["response"].get("truncated") or timings["cache_n"] != 0:
            raise ValueError(f"slot {slot} reused or truncated its prompt")
        save(folder / f"client-{slot:02d}.json", row)
        return row

    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=SLOTS) as pool:
        rows = list(pool.map(client, range(SLOTS)))
    seconds = time.monotonic() - start
    common = glm_measure.common_decode_window(rows)
    return {"passed": common["available"] and all(x["tokens"] > 0 for x in common["streams"]),
            "sessions": SLOTS, "common_decode": common, "wall_tokens_per_second": SLOTS * args.balanced_tokens / seconds,
            "median_ttft_ms": statistics.median(x["ttft_ms"] for x in rows), "max_ttft_ms": max(x["ttft_ms"] for x in rows),
            "tokens_by_slot": [x["token_ids"] for x in rows]}


def tokenize(url, text):
    return api(url, "/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]


def archive_prompt(args):
    prompt = api(args.url, "/apply-template", {"messages": [{"role": "user", "content": READ_QUESTION}],
                                               "reasoning_effort": "none"})["prompt"]
    left, right = prompt.split(SEPARATOR)
    ids, suffix = tokenize(args.url, left + "\n"), tokenize(args.url, "\n" + right)
    filler = tokenize(args.url, "".join(f"\nAudit entry {i}: src/module_{i % 37}/build.py passed its interface checks."
                                       for i in range(257)))
    if not filler:
        raise ValueError("empty archive filler")
    def fill_to(n):
        missing = n - len(ids)
        if missing < 0:
            raise ValueError("archive cannot fit its instructions")
        ids.extend((filler * ((missing + len(filler) - 1) // len(filler)))[:missing])

    for percent, module, fact in [(17, "elm", FACTS[0]), (73, "ash", FACTS[1])]:
        fill_to(args.reader_tokens * percent // 100)
        ids.extend(tokenize(args.url, f"\nRelease approval for module {module}: {fact}.\n"))
    fill_to(args.reader_tokens - len(suffix))
    return ids + suffix


def workload(args, folder, phase):
    args.url = f"http://127.0.0.1:{args.port}"
    if phase == "chat":
        return chat(args, folder)
    args.vocab = tokenize(args.url, "Lake forest red blue green stone water earth east cloud black sky five " +
                          " ".join(f"file_{i} /tmp/log-{i}.txt" for i in range(SLOTS)))
    if not args.vocab:
        raise ValueError("tokenizer returned no IDs")
    if phase == "balanced":
        return balanced(args, folder)
    if phase == "long":
        options = SimpleNamespace(**vars(args), sessions=SLOTS, peer_prompt=128, warm_chunks=16, check_restore=False)
        result = mixed_prefill.measure(options, archive_prompt(args))
        answer = result["reader_text"].upper()
        found = re.search(FACTS[0] + r"\b.{0,80}?\b" + FACTS[1], answer, re.S)
        result.update(retrieval_correct=bool(found), passed=bool(found) and result["peers_spanning_reader"] == SLOTS - 1
                      and all(p["tokens_during_reader"] > 0 for p in result["peers"]))
        return result
    # Keep the exact prefix, four-slot displacement order and tokenizer vocabulary of the standalone probe.
    vocab = tokenize(args.url, " red blue green sky water cloud earth rain")
    result = mixed_prefill.compare_restore(args, vocab)
    def matches(same):
        return (same["same_greedy_token"] and same["common_top_tokens"] == 32 and
                same["max_common_logprob_difference"] <= args.logprob_tolerance)

    same = result["comparisons"]["restored_vs_resident"]
    result["passed"] = (result["cached_tokens"] == {"cold": 0, "resident": args.reader_tokens, "restored": args.reader_tokens}
                        and matches(same))
    reference = result["same_slot_file_control"]
    if reference:
        reference["passed"] = reference["timings"]["cache_n"] == args.reader_tokens and matches(reference["restored_vs_file"])
        result["passed"] &= reference["passed"]
    return result


def gpu_info():
    cmd = ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version", "--format=csv,noheader,nounits"]
    rows = list(csv.reader(subprocess.check_output(cmd, text=True, timeout=20).splitlines(), skipinitialspace=True))
    if (len(rows) != 4 or len({r[1] for r in rows}) != 4 or len({r[2] for r in rows}) != 4
            or any("RTX PRO 6000" not in r[3].upper() or "BLACKWELL" not in r[3].upper() or int(r[4]) < 90000 for r in rows)):
        raise ValueError(f"expected four distinct 96 GB RTX PRO 6000 Blackwell GPUs, found {rows}")
    return rows


def readiness(proc, args, deadline, folder):
    url = f"http://127.0.0.1:{args.port}"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited {proc.returncode}; see {folder / 'server.log'}")
        try:
            if api(url, "/health", timeout=2).get("status") == "ok":
                break
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1)
    else:
        raise TimeoutError("server startup deadline expired")
    props = api(url, "/props")
    model = api(url, "/v1/models")["data"][0]
    save(folder / "server-info.json", {"props": props, "model": model})
    meta = model["meta"]
    if not (props["total_slots"] == SLOTS and model["id"] == args.alias and meta["n_ctx"] == serve.CODING_SLOT_CONTEXT
            and meta["n_embd"] == 4608 and 590e9 < meta["n_params"] < 600e9 and "Q3_K - Medium" in meta["ftype"]):
        raise ValueError("running model, file type or slot geometry does not match the Inkling tensor-4 recipe")
    log = (folder / "server.log").read_text(errors="replace")
    counts = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", log)
    if not any(int(a) == int(b) and int(a) > 1 for a, b in counts):
        raise ValueError("startup did not report a complete GPU layer offload")
    if "Meta(CUDA0,CUDA1,CUDA2,CUDA3)" not in log:
        raise ValueError("startup did not report four distinct CUDA tensor ranks")


def gguf_parts(path):
    match = re.fullmatch(r"(.*)-(\d{5})-of-(\d{5})\.gguf", path.name)
    if not match:
        return [path]
    stem, first, total = match.groups()
    if int(first) != 1 or int(total) == 0:
        raise ValueError("specify the first GGUF shard")
    return [path.with_name(f"{stem}-{i:05d}-of-{total}.gguf") for i in range(1, int(total) + 1)]


def launch_plan(args, route):
    options = ["--profile", args.profile, "--server", str(args.server), "--packs", str(args.packs), "--port", str(args.port),
               "--reduction-route", route, "--log-verbosity", "4"]
    if args.cache_ram is not None:
        options += ["--cache-ram", str(args.cache_ram)]
    if args.expert_row_limit is not None:
        options += ["--expert-row-limit", str(args.expert_row_limit)]
    if args.ubatch_size is not None:
        options += ["--ubatch-size", str(args.ubatch_size)]
    if args.context is not None:
        options += ["--context", str(args.context)]
    if args.prompt_step is not None:
        options += ["--prompt-step", str(args.prompt_step)]
    if args.prompt_step_ms is not None:
        options += ["--prompt-step-ms", str(args.prompt_step_ms)]
    if args.checkpoint_every is not None:
        options += ["--checkpoint-every", str(args.checkpoint_every)]
    if args.nccl_sliced_gather and route == "engine":
        options.append("--nccl-sliced-gather")
    if args.host_owner_rows is not None and route != "engine":
        options.append("--host-owner-rows" if args.host_owner_rows else "--no-host-owner-rows")
    if args.reduction_wire is not None:
        options += ["--reduction-wire", args.reduction_wire]
    if args.prefix_copy is not None:
        options.append("--prefix-copy" if args.prefix_copy else "--no-prefix-copy")
    if args.restore_reference_dir:
        options += ["--slot-save-path", args.restore_reference_dir]
    if args.model:
        options += ["--model", str(args.model)]
    if args.mmproj:
        options += ["--mmproj", str(args.mmproj)]
    if args.bucket_prefill_kv is not None:
        options.append("--bucket-prefill-kv" if args.bucket_prefill_kv else "--no-bucket-prefill-kv")
    if args.stable_prefill is not None:
        options.append("--stable-prefill" if args.stable_prefill else "--no-stable-prefill")
    if args.sort_expert_rows is not None:
        options.append("--sort-expert-rows" if args.sort_expert_rows else "--no-sort-expert-rows")
    if args.mean_mask_decode is not None:
        options.append("--mean-mask-decode" if args.mean_mask_decode else "--no-mean-mask-decode")
    if args.text_only:
        options += ["--text-only"]
    argv, settings = serve.make_plan(serve.arguments(options))
    return {"command": [sys.executable, str(Path(serve.__file__).resolve()), *options], "argv": argv, "settings": settings}


def identity(args, plan):
    argv = plan["argv"]
    assets = gguf_parts(Path(argv[argv.index("-m") + 1]))
    if "--mmproj" in argv:
        assets += [Path(argv[argv.index("--mmproj") + 1])]
    metadata = {}
    for path in assets:
        st = path.stat()
        metadata[str(path.resolve())] = [st.st_size, st.st_mtime_ns]
    binary = Path(argv[0])
    libs = sorted({p.resolve() for p in binary.parent.glob("*.so*") if p.is_file()})
    sources = [Path(__file__), Path(serve.__file__), Path(serve.glm53.__file__), Path(mixed_prefill.__file__),
               Path(glm_measure.__file__), serve.TEMPLATE]
    head = args.source_head
    checkout = (serve.ROOT / ".git").exists()
    if not head and checkout:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=serve.ROOT, text=True).strip()
    if not head or not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", head):
        raise ValueError("a source bundle without .git needs --source-head with its full Git revision")
    source_digest = args.source_bundle_sha256
    if args.source_bundle:
        actual = glm_measure.artifact_digest(args.source_bundle)
        if source_digest and source_digest.lower() != actual:
            raise ValueError("source bundle checksum differs from --source-bundle-sha256")
        source_digest = actual
    if (source_digest and not re.fullmatch(r"[0-9a-fA-F]{64}", source_digest)) or (not checkout and not source_digest):
        raise ValueError("provide --source-bundle or the SHA256 verified when deploying it")
    return {"gpus": gpu_info(), "assets_size_mtime_ns": metadata,
            "binaries_sha256": {str(p): glm_measure.artifact_digest(p) for p in [binary, *libs]},
            "scripts_sha256": {p.name + ":" + p.parent.name: glm_measure.artifact_digest(p) for p in sources},
            "model_manifest_sha256": glm_measure.artifact_digest(args.sha256_manifest) if args.sha256_manifest else None,
            "git_head": head.lower(), "working_source_bundle_sha256": source_digest.lower() if source_digest else None,
            "launch": {r: launch_plan(args, r) for r in ROUTES},
            "workload": {k: getattr(args, k) for k in ("reader_tokens", "reader_output_tokens", "peer_tokens",
                                                        "balanced_tokens", "chat_budget", "logprob_tolerance")}}


def check_hashes(args, ident):
    if not args.sha256_manifest:
        return
    entries = {}
    for line in args.sha256_manifest.read_text().splitlines():
        m = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", line)
        if m:
            entries[(args.sha256_manifest.parent / m[2]).resolve()] = m[1].lower()
    for filename in ident["assets_size_mtime_ns"]:
        path = Path(filename)
        if path not in entries or glm_measure.artifact_digest(path) != entries[path]:
            raise ValueError("missing or mismatched GGUF checksum: " + str(path))


def compare(records):
    rows = {}
    for route, phases in records.items():
        metrics = {}
        if "balanced" in phases:
            metrics["balanced_decode_tps"] = phases["balanced"]["common_decode"]["tokens_per_second"]
        if "long" in phases:
            q = phases["long"]
            metrics.update(reader_ttft_ms=q["reader_ttft_ms"], peer_aggregate_tps=q["reader_aggregate_peer_tokens_per_second"],
                           peer_p95_gap_ms=q["overlapping_delivery_gaps"]["p95_ms"])
        rows[route] = {"metrics": metrics}
    for route, row in rows.items():
        row["ratio_to_host"] = {k: v / rows["host"]["metrics"][k] for k, v in row["metrics"].items()
                                if v is not None and rows.get("host", {}).get("metrics", {}).get(k)}
        if "balanced" in records[route] and "balanced" in records.get("host", {}):
            baseline = records["host"]["balanced"]["tokens_by_slot"]
            candidate = records[route]["balanced"]["tokens_by_slot"]
            row["first_different_decode_position_by_slot"] = [
                next((i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]), None) for a, b in zip(baseline, candidate)]
        if "restore" in records[route] and "restore" in records.get("host", {}):
            baseline, candidate = records["host"]["restore"], records[route]["restore"]
            arms = {}
            for arm in ("cold", "resident", "restored"):
                a, b = baseline["top_logprobs"][arm], candidate["top_logprobs"][arm]
                common = a.keys() & b.keys()
                arms[arm] = {"same_greedy_token": baseline["greedy_token"][arm] == candidate["greedy_token"][arm],
                             "common_top_tokens": len(common), "max_common_logprob_difference":
                             max((abs(a[k] - b[k]) for k in common), default=None)}
            row["top32_vs_host"] = arms
    return rows


def backup(out, dest):
    if dest:
        target = dest / out.name
        if target.exists() and (not (target / "run-id").is_file() or (target / "run-id").read_text() != (out / "run-id").read_text()):
            raise ValueError("a different run already owns the backup folder: " + str(target))
        shutil.copytree(out, target, dirs_exist_ok=True)


def publish_log(path, logfile):
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not path.is_symlink():
        raise ValueError("the monitoring log path is already a regular file: " + str(path))
    link = path.with_name("." + path.name + "-" + uuid4().hex)
    link.symlink_to(os.path.relpath(logfile.resolve(), path.parent.resolve()))
    link.replace(path)


def supervise(args):
    deadline = time.monotonic() + args.seconds
    plans = {r: launch_plan(args, r) for r in args.routes}
    args.alias = plans[args.routes[0]]["argv"][plans[args.routes[0]]["argv"].index("--alias") + 1]
    if args.dry_run:
        print(json.dumps({"routes": plans, "phases": args.phases, "sessions": SLOTS, "cold_reader": READER,
                          "reader_outputs": args.reader_output_tokens, "deadline_seconds": args.seconds}, indent=2))
        return 0
    out = args.out.resolve()
    if args.backup_dir:
        target = args.backup_dir.resolve() / out.name
        if target.is_relative_to(out) or out.is_relative_to(target):
            raise ValueError("backup and results folders must be separate")
    if out.exists() and not args.resume:
        raise ValueError("result folder exists; select a fresh folder or use --resume")
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not (out / "run-id").exists():
        (out / "run-id").write_text(uuid4().hex + "\n")
    manifest = out / "identity.json"
    ident = identity(args, next(iter(plans.values())))
    if manifest.exists():
        if json.loads(manifest.read_text()) != ident:
            raise ValueError("host, assets, binaries or workload changed; use a fresh output folder")
    else:
        check_hashes(args, ident)
        save(manifest, ident)
        with (out / "topology.txt").open("w") as topo:
            for opts in (["-m"], ["-p2p", "r"], ["-p2p", "w"]):
                subprocess.run(["nvidia-smi", "topo", *opts], stdout=topo, stderr=subprocess.STDOUT, timeout=30)
    backup(out, args.backup_dir)
    records = {r: {} for r in ROUTES}
    for route in ROUTES:
        for phase in PHASES:
            ref = out / route / (phase + ".done.json")
            if ref.is_file():
                records[route][phase] = json.loads((out / json.loads(ref.read_text())).read_text())
    for route in args.routes:
        pending = [s for s in args.phases if s not in records[route]]
        if not pending:
            continue
        with socket.socket() as check:
            check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            check.bind(("127.0.0.1", args.port))
            check.listen()
        folder = out / route / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:7])
        folder.mkdir(parents=True)
        save(folder / "plan.json", plans[route])
        with (folder / "server.log").open("w") as log:
            publish_log(args.server_log_link, folder / "server.log")
            proc = subprocess.Popen(plans[route]["command"], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                readiness(proc, args, min(deadline, time.monotonic() + args.startup_timeout), folder)
                (folder / "gpu-loaded.txt").write_text(subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index,pci.bus_id,memory.used,memory.free", "--format=csv"], text=True, timeout=30))
                for phase in pending:
                    work = folder / phase
                    work.mkdir()
                    save(work / "arguments.json", {k: v for k, v in vars(args).items()
                         if k in ("port", "alias", "reader_tokens", "reader_output_tokens", "peer_tokens", "balanced_tokens",
                                  "chat_budget", "timeout", "logprob_tolerance", "restore_reference_dir")})
                    print(f"{route} {phase}: {work}", flush=True)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("qualification deadline expired")
                    offset = (folder / "server.log").stat().st_size
                    with (work / "client.log").open("w") as client, (work / "pcie.log").open("w") as pcie:
                        monitor = subprocess.Popen(["nvidia-smi", "dmon", "-s", "t", "-d", "1", "-o", "DT"],
                                                   stdout=pcie, stderr=subprocess.STDOUT, start_new_session=True)
                        try:
                            subprocess.run([sys.executable, str(Path(__file__).resolve()), "_worker", phase, str(work)],
                                           stdout=client, stderr=subprocess.STDOUT, check=True, timeout=remaining)
                        finally:
                            glm_measure.terminate(monitor)
                    result_file = work / "result.json"
                    result = json.loads(result_file.read_text())
                    if phase == "restore":
                        with (folder / "server.log").open("rb") as serverlog:
                            serverlog.seek(offset)
                            stage_log = serverlog.read().decode(errors="replace")
                        writes = [l for l in stage_log.splitlines() if f"cache_save reason=saved slot=0 tokens={READER} bytes=" in l]
                        loads = [l for l in stage_log.splitlines() if f"cache_restore reason=restored slot=1 tokens={READER} bytes=" in l]
                        result["ram_transfer_log"] = {"saves": writes, "restores": loads}
                        result["passed"] &= bool(writes and loads)
                        save(result_file, result)
                    if not result["passed"]:
                        raise ValueError(f"{route} {phase} failed; see {result_file}")
                    ref = out / route / (phase + ".done.json")
                    save(ref, str(result_file.relative_to(out)))
                    records[route][phase] = result
                    save(out / "comparison.json", compare(records))
                    backup(out, args.backup_dir)
            finally:
                glm_measure.terminate(proc)
                backup(out, args.backup_dir)
    return 0


def main():
    if sys.argv[1:2] == ["_worker"]:
        phase, folder = sys.argv[2], Path(sys.argv[3])
        args = SimpleNamespace(**json.loads((folder / "arguments.json").read_text()))
        save(folder / "result.json", workload(args, folder, phase))
        return 0
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True, help="fresh result folder on persistent storage")
    p.add_argument("--backup-dir", type=Path, help="copy reports and logs into this second mounted directory after each stage")
    p.add_argument("--resume", action="store_true", help="reuse successful stages with identical assets, software and workload")
    p.add_argument("--dry-run", action="store_true", help="print launch plans and exit before accessing GPUs or model files")
    p.add_argument("--server", type=Path, default=serve.ROOT / "build/bin/llama-server")
    p.add_argument("--profile", choices=["tensor-4"], default="tensor-4", help="Q3_K_M serving profile to qualify")
    p.add_argument("--packs", type=Path, default=serve.PACKS)
    p.add_argument("--model", type=Path, help="alternate path to this calibrated abliterated Q3_K_M, or its first shard")
    p.add_argument("--mmproj", type=Path)
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--context", type=int, help="shared KV pool to qualify, in tokens")
    p.add_argument("--prefix-copy", action=argparse.BooleanOptionalAction, help="qualify active text prefix sharing")
    p.add_argument("--prompt-step", type=int, help="prompt tokens per step while peers generate")
    p.add_argument("--prompt-step-ms", type=int, help="target time in milliseconds for a mixed prompt step")
    p.add_argument("--checkpoint-every", type=int, help="save rollback state at this token interval")
    p.add_argument("--nccl-sliced-gather", action="store_true", help="use native sliced NCCL gathers on the engine route")
    p.add_argument("--host-owner-rows", action=argparse.BooleanOptionalAction, help="use owner-only BF16 convolution reductions on host routes")
    p.add_argument("--reduction-wire", choices=["bf16", "q8"], help="wire precision for eligible copy-engine reductions")
    p.add_argument("--mean-mask-decode", action=argparse.BooleanOptionalAction, help="opt in on all selected transport routes")
    p.add_argument("--bucket-prefill-kv", action=argparse.BooleanOptionalAction, help="override global F16 prefill buckets on selected routes")
    p.add_argument("--stable-prefill", action=argparse.BooleanOptionalAction, help="override FP32 relative-attention prefill on selected routes")
    p.add_argument("--sort-expert-rows", action=argparse.BooleanOptionalAction, help="override 23-row expert sorting on selected routes")
    p.add_argument("--sha256-manifest", type=Path, help="verify every GGUF against sha256sum entries relative to this manifest")
    p.add_argument("--source-head", help="full Git revision used for an exported checkout without .git")
    p.add_argument("--source-bundle", type=Path, help="hash the deployed source archive or source checksum manifest")
    p.add_argument("--source-bundle-sha256", help="SHA256 already verified on deployment, or expected digest for --source-bundle")
    p.add_argument("--server-log-link", type=Path, help="stable symlink updated per route for a monitor following the log by name")
    p.add_argument("--cache-ram", type=int, metavar="MIB", help="override the profile's displaced-session cache budget")
    p.add_argument("--expert-row-limit", type=serve.expert_row_limit, metavar="N|auto", help="qualify the routed kernel row limit or engine dispatch")
    p.add_argument("--ubatch-size", type=int, help="physical microbatch size for the serving run")
    p.add_argument("--routes", choices=ROUTES, nargs="+", default=ROUTES)
    p.add_argument("--phases", choices=PHASES, nargs="+", default=PHASES)
    p.add_argument("--port", type=int, default=8103)
    p.add_argument("--seconds", type=int, default=14400, help="wall time for preflight and all routes, followed by cleanup")
    p.add_argument("--startup-timeout", type=int, default=900)
    p.add_argument("--timeout", type=int, default=3600, help="maximum idle HTTP seconds, also bounded by total wall time")
    p.set_defaults(reader_tokens=READER)
    p.add_argument("--reader-output-tokens", type=int, default=64, help="64 for retrieval; 32768 also fills the 294912-token slot")
    p.add_argument("--peer-tokens", type=int, default=32768)
    p.add_argument("--balanced-tokens", type=int, default=128)
    p.add_argument("--chat-budget", type=int, default=2048)
    p.add_argument("--restore-reference-dir", help="shared local directory for a temporary same-slot reference snapshot (about 8.5 GiB at 262144 tokens)")
    p.add_argument("--logprob-tolerance", type=float, default=1e-4, help="maximum restored vs resident top-32 logprob difference")
    args = p.parse_args()
    if (not 1 <= args.reader_output_tokens <= serve.CODING_SLOT_CONTEXT - READER or
            min(args.seconds, args.startup_timeout, args.timeout, args.chat_budget) < 1 or
            args.peer_tokens <= 16 or args.balanced_tokens < 2 or args.cache_ram is not None and (args.cache_ram == 0 or args.cache_ram < -1) or
            not 0 <= args.logprob_tolerance < float("inf")):
        p.error("invalid token, timeout, cache or logprob limit")
    if args.restore_reference_dir:
        args.restore_reference_dir = str(Path(args.restore_reference_dir).resolve())
        if not args.dry_run:
            Path(args.restore_reference_dir).mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    def expired(*_):
        raise TimeoutError("qualification deadline expired")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, args.seconds)
    try:
        return supervise(args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    sys.exit(main())
