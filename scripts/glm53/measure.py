"""One pinned serving leg on an already approved dedicated rental.

Retains raw HTTP responses, launch identity and exit status. All work shares the
operator's deadline, and cleanup terminates only children launched by this run.
"""
import argparse
import concurrent.futures
import contextlib
import csv
from datetime import datetime
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import sqlite3
import subprocess
import threading
import time
from typing import Any
import urllib.error
import urllib.request


def completion_payload(prompt, tokens, slot):
    return dict(prompt=prompt, n_predict=tokens, id_slot=slot, temperature=1.0,
                top_p=0.95, repeat_penalty=1.1, repeat_last_n=256, seed=530,
                ignore_eos=True, cache_prompt=False)

def server_command(args):
    threads = []
    for flag in ("threads", "threads_batch"):
        value = getattr(args, flag, None)
        if value is not None:
            if value < 1:
                raise ValueError("thread counts must be positive")
            threads += ["--" + flag.replace("_", "-"), str(value)]
    if getattr(args, "recipe", "tensor-2") == "dp-4":
        return dp_plan(args)[0] + threads
    argv = [str(Path(args.binary) / "llama-server"), "-m", str(args.model), "-ngl", "99",
            "-sm", "tensor", "-ts", "1,1", "--jinja", "-fa", "on", "-ctk", "q8_0",
            "-ctv", "q8_0", "-ctki", "q8_0", "-b", "4096", "-ub", str(args.ubatch),
            "-c", str(args.context), "-np", str(args.slots), "--kv-unified", "--cache-ram", "0",
            "--no-cache-idle-slots", "--metrics", "--host", "127.0.0.1", "--port", str(args.port),
            "--spec-type", "draft-mtp", "--spec-draft-n-max", str(args.depth),
            "--spec-draft-mtp-vocab", str(Path(args.fixtures) / "glm-mtp-public-98304.i32"),
            "--spec-draft-mtp-adaptive", str(args.adaptive)]
    return argv + (["--glm53-profile", args.profile] if args.profile else []) + threads


def dp_plan(args):
    import serve
    if args.profile or args.ubatch != 512 or args.context != args.slots*serve.DP_SLOT_CONTEXT or args.adaptive != 0.18:
        raise ValueError("DP measurements require the documented private-cache recipe, ubatch 512 and adaptive 0.18")
    argv = ["--profile", "dp-4", "--model", str(args.model), "--server", str(Path(args.binary)/"llama-server"),
            "--port", str(args.port), "--slots", str(args.slots), "--mtp-depth", str(args.depth)]
    if getattr(args, "kv_host", False):
        argv += ["--kv-host"]
    if getattr(args, "host_budget_mib", None) is not None:
        argv += ["--host-budget-mib", str(args.host_budget_mib)]
    if getattr(args, "mmproj", None):
        argv += ["--mmproj", str(args.mmproj)]
    return serve.make_plan(serve.arguments(argv))

def validate_response(response, tokens, prompt_tokens=None):
    response = response if isinstance(response, dict) else {}
    timings = response.get("timings")
    timings = timings if isinstance(timings, dict) else {}
    if (not isinstance(response.get("content"), str) or not response["content"].strip()
            or "error" in response or timings.get("predicted_n") != tokens
            or prompt_tokens is not None and timings.get("prompt_n") != prompt_tokens):
        raise ValueError(f"missing delivered output or incomplete prompt/generation: predicted_n="
                         f"{timings.get('predicted_n')!r}, expected {tokens}; prompt_n={timings.get('prompt_n')!r}, "
                         f"expected {prompt_tokens!r}; error={response.get('error')!r}")
    return True


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n")


def artifact_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as artifact:
        for chunk in iter(lambda: artifact.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def nsys_version(nsys='nsys', timeout: float = 30) -> dict[str, Any]:
    result: dict[str, Any] = dict(command=[nsys, '--version'], exit=None, stdout='', stderr='')
    try:
        run = subprocess.run(result['command'], capture_output=True, text=True, timeout=timeout)
        result.update(exit=run.returncode, stdout=run.stdout, stderr=run.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    version = next((line.strip() for line in result['stdout'].splitlines() if line.strip()), None)
    if version is not None:
        result['version'] = version
    return result


def nsys_export_path(output):
    return output.with_name(output.name + '-export.sqlite')


def export_nsys_capture(output, devices, request, deadline, nsys='nsys'):
    """Export one raw report and verify an attributed CUDA decode window without replacing its artifacts."""
    from decode_trace import capture_readiness
    report = output.with_name(output.name + '.nsys-rep')
    sqlite = nsys_export_path(output)
    result: dict[str, Any] = dict(report=str(report), sqlite=str(sqlite), launch_sha256=request.get('launch_sha256'),
                  capture_verified=False, report_present=report.is_file() and report.stat().st_size > 0)
    if not result['report_present']:
        result['failure'] = 'missing_raw_report'
    elif sqlite.exists():
        result['failure'] = 'sqlite_output_exists'
    elif time.time() >= deadline:
        result['failure'] = 'export_deadline_reached'
    else:
        exporter = nsys_version(nsys, timeout=min(30, max(0.001, deadline-time.time())))
        captured = request.get('capture_nsys') or {}
        compatibility = dict(capture_version=captured.get('version'), exporter_version=exporter.get('version'),
                             state='unknown')
        if exporter.get('exit') != 0:
            compatibility['state'] = 'exporter_unavailable'
            result['failure'] = 'exporter_unavailable'
        elif compatibility['capture_version'] == compatibility['exporter_version']:
            compatibility['state'] = 'exact-version'
        elif compatibility['capture_version']:
            compatibility['state'] = 'version_mismatch'
            result['failure'] = 'exporter_version_mismatch'
        else:
            compatibility['state'] = 'capture_version_missing'
            result['failure'] = 'capture_version_missing'
        exporter['compatibility'] = compatibility
        result['exporter'] = exporter
        if 'failure' not in result:
            command = [nsys, 'export', '--type', 'sqlite', '--output', str(sqlite), str(report)]
            try:
                exported = subprocess.run(command, capture_output=True, text=True,
                                          timeout=min(180, max(0.001, deadline-time.time())))
                result['export'] = dict(argv=command, exit=exported.returncode,
                                        stdout=exported.stdout, stderr=exported.stderr)
            except (OSError, subprocess.TimeoutExpired) as exc:
                result['export_error'] = f'{type(exc).__name__}: {exc}'
            if result.get('export', {}).get('exit') != 0:
                result.setdefault('failure', 'sqlite_export_failed')
            elif not sqlite.is_file() or sqlite.stat().st_size == 0:
                result['failure'] = 'missing_sqlite_export'
            else:
                try:
                    with sqlite3.connect(sqlite.resolve().as_uri() + '?mode=ro', uri=True) as db:
                        result['readiness'] = capture_readiness(db, devices, expected_sequences=request.get('active_sessions'))
                except (OSError, sqlite3.Error, ValueError) as exc:
                    result['validation_error'] = f'{type(exc).__name__}: {exc}'
                else:
                    result['capture_verified'] = result['readiness']['usable'] is True
                    if not result['capture_verified']:
                        result['failure'] = 'capture_not_ready'
    if result['report_present']:
        result['report_sha256'] = artifact_digest(report)
    if sqlite.is_file() and sqlite.stat().st_size:
        result['sqlite_sha256'] = artifact_digest(sqlite)
    write_json(output.with_name(output.name + '-export.json'), result)
    return result


def shutdown_nsys_session(session, kill_signal=signal.SIGKILL):
    command = ['nsys', 'shutdown', '--session=' + session, '--kill=' + str(int(kill_signal))]
    result: dict[str, Any] = dict(argv=command, exit=None)
    try:
        stopped = subprocess.run(command, capture_output=True, text=True, timeout=30)
        result.update(exit=stopped.returncode, stdout=stopped.stdout, stderr=stopped.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        result['error'] = f'{type(error).__name__}: {error}'
    return result


def finish_nsys_capture(process, session, output, deadline):
    """Drain an owned Nsight exporter before its process group is terminated, even after an application abort."""
    report = output.with_name(output.name+'.nsys-rep')
    result = dict(report=str(report), report_present=False, capture_verified=False,
                  deadline_reached=time.time() >= deadline)
    if not result['deadline_reached']:
        stopped_ok = False
        try:
            stopped = subprocess.run(['nsys', 'stop', '--session='+session], capture_output=True, text=True,
                                     timeout=min(60, max(0.001, deadline-time.time())))
            result['stop'] = dict(exit=stopped.returncode, stdout=stopped.stdout, stderr=stopped.stderr)
            stopped_ok = stopped.returncode == 0
        except (OSError, subprocess.TimeoutExpired) as exc:
            result['stop_error'] = str(exc)
        # After a target abort, 'stop' may refuse because automatic export has already started.
        # 'shutdown' or killing the launch process here would truncate that export.
        if (not stopped_ok or not report.is_file()) and process is not None and process.poll() is None:
            wait = min(120, max(0, deadline-time.time()))
            if wait:
                try:
                    process.wait(timeout=wait)
                except subprocess.TimeoutExpired:
                    result['export_wait_expired'] = True
        result['deadline_reached'] = time.time() >= deadline
    result['report_present'] = report.is_file() and report.stat().st_size > 0
    # The report must be exported and checked with decode_trace.capture_readiness on the analysis machine.
    write_json(output.with_name(output.name+'-finalize.json'), result)
    return result


def retained_chunks(response, saved, timeout):
    while True:
        timeout()
        try:
            raw = response.read1(65536)
        except http.client.IncompleteRead as exc:
            saved.write(exc.partial)
            saved.flush()
            raise
        if not raw:
            # read1() does not itself reject EOF before Content-Length is exhausted.
            remaining = getattr(response, "length", None)
            if remaining:
                raise http.client.IncompleteRead(b"", remaining)
            timeout()
            return
        saved.write(raw)
        saved.flush()
        timeout()
        yield raw


def retain_body(response, saved, timeout):
    return b"".join(retained_chunks(response, saved, timeout))


def retained_lines(response, saved, timeout):
    pending = b""
    for chunk in retained_chunks(response, saved, timeout):
        pending += chunk
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            yield line + b"\n"
    if pending:
        yield pending


def read_completion_stream(lines, tokens, prompt_tokens=None, clock=time.monotonic, on_event=None):
    """Native /completion stream. One event may deliver several MTP/UTF-8 tokens."""
    bursts, ids, content, final = [], [], [], None
    for line in lines:
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        event = json.loads(data)
        at = clock()
        if on_event:
            on_event(dict(at=at, event=event))
        if not isinstance(event, dict) or "error" in event or final is not None:
            raise ValueError("invalid, failed or duplicate completion stream event")
        part = event.get("tokens", [])
        if not isinstance(part, list) or any(type(t) is not int or t < 0 for t in part):
            raise ValueError("stream did not return valid token IDs")
        if not isinstance(event.get("content", ""), str):
            raise ValueError("stream did not return text")
        if part:
            bursts.append(dict(at=at, tokens=part))
            ids.extend(part)
        content.append(event.get("content", ""))
        if event.get("stop") is True:
            final = event
    if final is None or len(ids) != tokens:
        raise ValueError(f"incomplete completion stream: terminal={final is not None}, token IDs={len(ids)}, expected={tokens}")
    response = final | dict(content="".join(content), tokens=ids)
    validate_response(response, tokens, prompt_tokens)
    return dict(response=response, bursts=bursts, token_ids=ids)


def common_decode_window(rows):
    """Delivered tokens in (latest first burst, earliest last burst]; no interpolation across bursts."""
    unavailable = dict(available=False, tokens_per_second=None, reason="no_common_decode_interval")
    if not rows or any(not row["bursts"] for row in rows):
        return unavailable
    for row in rows:
        previous = -math.inf
        for burst in row["bursts"]:
            at = burst["at"]
            if not math.isfinite(at) or at < previous or not burst["tokens"]:
                raise ValueError("invalid stream burst timeline")
            previous = at
    start = max(row["bursts"][0]["at"] for row in rows)
    end = min(row["bursts"][-1]["at"] for row in rows)
    if end <= start:
        return unavailable | dict(start=start, end=end)
    streams = []
    for row in rows:
        count = sum(len(b["tokens"]) for b in row["bursts"] if start < b["at"] <= end)
        streams.append(dict(slot=row["slot"], tokens=count, tokens_per_second=count/(end-start)))
    count = sum(s["tokens"] for s in streams)
    return dict(available=True, start=start, end=end, seconds=end-start, tokens=count,
                tokens_per_second=count/(end-start), streams=streams,
                boundary="start exclusive, end inclusive; client receive times; whole bursts")


def terminate(child):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


class ServingLeg:
    def __init__(self, args):
        self.args = args
        self.out = args.root / "results" / args.run
        self.url = f"http://127.0.0.1:{args.port}"
        self.request_lock = threading.Lock()
        self.request_number = 0
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("GGML_", "LLAMA_", "TEST_"))}
        self.env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1",
                        GGML_CUDA_ALLREDUCE="internal", GGML_CUDA_GRAPH_MAX="0",
                        LD_LIBRARY_PATH=f"{args.binary}:{args.root}/cuda-runtime")
        self.cards = 4 if getattr(args, "recipe", "tensor-2") == "dp-4" else 2
        if self.cards == 4:
            import serve
            self.env = serve.execution_environment(dp_plan(args)[1], self.env)
            self.env["LD_LIBRARY_PATH"] = f"{args.binary}:{args.root}/cuda-runtime"
        if args.mode in ("capacity", "concurrency"):
            self.env["GGML_CUDA_MMVQ_Q8_SM120_CHECK"] = "1"
        if args.mode == "diagnostic":
            self.env["GGML_CUDA_ROUTE_STATS"] = "1"

    def timeout(self, maximum=300):
        left = self.args.deadline - time.time()
        if left <= 0:
            raise TimeoutError("serving deadline reached")
        return min(maximum, left)

    def response_record(self):
        with self.request_lock:
            self.request_number += 1
            folder = self.out / "http"
            folder.mkdir(exist_ok=True)
            return folder / f"{self.request_number:06d}.json"

    @contextlib.contextmanager
    def response_deadline(self, response):
        stream = response.fp if isinstance(response, urllib.error.HTTPError) else response
        # A chunked read can keep receiving framing bytes without returning. Interrupt the socket at the deadline.
        connection = stream.fp.raw._sock
        expired = threading.Event()
        def expire():
            expired.set()
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # The response may already have closed its socket.
        timer = threading.Timer(self.timeout(math.inf), expire)
        timer.daemon = True
        timer.start()
        try:
            yield
            self.timeout()
        except Exception as exc:
            if expired.is_set():
                raise TimeoutError("serving deadline reached") from exc
            raise
        finally:
            timer.cancel()
            timer.join()

    @contextlib.contextmanager
    def capture_response(self, path, data=None, maximum=300, record=None, context=None):
        record = record or self.response_record()
        raw = record.with_suffix(".body")
        meta = dict(path=path, request=data, context=context, status=None,
                    body_file=str(raw.relative_to(self.out)), started_epoch=time.time())
        write_json(record, meta)
        try:
            body = None if data is None else json.dumps(data, ensure_ascii=False).encode()
            req = urllib.request.Request(self.url + path, data=body, headers={"Content-Type": "application/json"})
            try:
                response = urllib.request.urlopen(req, timeout=self.timeout(maximum))
            except urllib.error.HTTPError as exc:
                meta["status"] = exc.code
                with exc, raw.open("wb") as saved, self.response_deadline(exc):
                    retain_body(exc, saved, lambda: self.timeout(maximum))
                raise
            meta["status"] = response.status
            with response, raw.open("wb") as saved, self.response_deadline(response):
                yield response, saved
        except Exception as exc:
            meta["error"] = f"{type(exc).__name__}: {exc}"
            message = f"request {path} failed: {meta['error']}; record {record.relative_to(self.out)}"
            if isinstance(exc, ValueError):
                raise ValueError(message) from exc
            if isinstance(exc, (OSError, http.client.IncompleteRead)):
                raise OSError(message) from exc
            raise RuntimeError(message) from exc
        finally:
            meta["ended_epoch"] = time.time()
            write_json(record, meta)

    def request(self, path, data=None, maximum=300, record=None, context=None):
        with self.capture_response(path, data, maximum, record, context) as (response, saved):
            return json.loads(retain_body(response, saved, lambda: self.timeout(maximum)))

    def complete(self, prompt, tokens=384, slot=0, prompt_tokens=None, repeat=None):
        before = time.monotonic()
        payload = completion_payload(prompt["prompt"], tokens, slot)
        record = self.response_record()
        result = self.request("/completion", payload, record=record,
                              context=dict(category=prompt["category"], repeat=repeat, slot=slot,
                                           expected_tokens=tokens, expected_prompt_tokens=prompt_tokens))
        try:
            validate_response(result, tokens, prompt_tokens)
        except ValueError as exc:
            raise ValueError(f"{prompt['category']} slot {slot}: {exc}; record {record.relative_to(self.out)}") from exc
        elapsed = time.monotonic() - before
        raw = prompt["prompt"].encode() if isinstance(prompt["prompt"], str) else json.dumps(prompt["prompt"]).encode()
        return dict(category=prompt["category"], elapsed=elapsed,
                    prompt_sha256=hashlib.sha256(raw).hexdigest(), request=payload, response=result,
                    response_record=str(record.relative_to(self.out)))

    def complete_stream(self, prompt, tokens, slot=0, prompt_tokens=None, repeat=None):
        payload = completion_payload(prompt["prompt"], tokens, slot) | dict(stream=True, return_tokens=True)
        record = self.response_record()
        started = time.monotonic()
        with self.capture_response("/completion", payload, record=record,
                                   context=dict(category=prompt["category"], repeat=repeat, slot=slot,
                                                expected_tokens=tokens, expected_prompt_tokens=prompt_tokens)) as (response, saved):
            with record.with_suffix(".events.jsonl").open("w") as events:
                def keep_event(event):
                    events.write(json.dumps(event, allow_nan=False) + "\n")
                    events.flush()
                result = read_completion_stream(retained_lines(response, saved, self.timeout), tokens, prompt_tokens, on_event=keep_event)
        raw = prompt["prompt"].encode() if isinstance(prompt["prompt"], str) else json.dumps(prompt["prompt"]).encode()
        result.update(slot=slot, request=payload, prompt_sha256=hashlib.sha256(raw).hexdigest(),
                      started=started, ended=time.monotonic(), response_record=str(record.relative_to(self.out)))
        result["owner_device"] = slot % self.cards if self.cards == 4 else None
        write_json(record.with_suffix(".stream.json"), result)
        return result

    def decode_group(self, prompt, tokens, slots, prompt_tokens=None, repeat=None):
        slots = list(slots)
        if not slots or len(set(slots)) != len(slots) or any(type(s) is not int or s < 0 or s >= self.args.slots for s in slots):
            raise ValueError("decode group needs distinct configured slots")
        barrier = threading.Barrier(len(slots))
        def run(slot):
            barrier.wait(timeout=self.timeout(10))
            return self.complete_stream(prompt, tokens, slot, prompt_tokens, repeat)
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(slots)) as pool:
            rows = list(pool.map(run, slots))
        return dict(slots=slots, output_tokens_per_stream=tokens, streams=rows,
                    common_decode=common_decode_window(rows), http_seconds=time.monotonic()-started,
                    native_timings=[r["response"]["timings"] for r in rows])

    def long_prompt(self, prompts, count):
        prompt = next(p for p in prompts if p["category"] == "long")
        ids = self.request("/tokenize", dict(content=prompt["prompt"], add_special=True))["tokens"]
        if len(ids) < count or count <= 128:
            raise ValueError("public fixture does not contain the declared token count")
        # Preserve the instruction and assistant marker at the end.
        return dict(category="long", prompt=ids[:count-128] + ids[-128:])

    def semantics(self):
        payload = completion_payload("Return the requested JSON object.", 64, 0)
        payload.update(ignore_eos=False, grammar='root ::= "{\\\"ok\\\":true}"')
        response = self.request("/completion", payload)
        write_json(self.out / "grammar.json", dict(request=payload, response=response))
        if json.loads(response.get("content", "")) != {"ok": True}:
            raise ValueError("grammar fallback did not preserve the constrained result")

        payload = completion_payload("Write a short explanation of binary search.", 64, 0)
        payload["mirostat"] = 2  # Unsupported backend chain must use the existing host fallback.
        response = self.request("/completion", payload)
        write_json(self.out / "unsupported-chain.json", dict(request=payload, response=response))
        validate_response(response, 64)

        tool = dict(type="function", function=dict(name="weather", description="Get weather for a city",
                    parameters=dict(type="object", properties=dict(city=dict(type="string", enum=["Paris"])),
                                    required=["city"], additionalProperties=False)))
        payload = dict(messages=[dict(role="user", content="Call weather for Paris.")], tools=[tool],
                       tool_choice=dict(type="function", function=dict(name="weather")),
                       max_tokens=128, temperature=1.0, top_p=0.95, seed=530)
        response = self.request("/v1/chat/completions", payload)
        write_json(self.out / "tool.json", dict(request=payload, response=response))
        calls = response["choices"][0]["message"].get("tool_calls", [])
        if (len(calls) != 1 or calls[0]["function"]["name"] != "weather"
                or json.loads(calls[0]["function"]["arguments"]) != {"city": "Paris"}):
            raise ValueError("forced tool call did not preserve the requested schema")

        payload = completion_payload("Explain merge sort and give several examples.", 4096, 0) | {"stream": True}
        first = None
        with self.capture_response("/completion", payload, maximum=60) as (stream, saved):
            for line in retained_lines(stream, saved, lambda: self.timeout(60)):
                if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                    item = json.loads(line[6:])
                    if item.get("content"):
                        first = item
                        break  # Closing the HTTP stream cancels this request.
        if first is None:
            raise ValueError("stream produced no output before cancellation")
        write_json(self.out / "cancelled.json", dict(request=payload, first_chunk=first))
        followup = self.complete(dict(category="cancel-followup", prompt="List three prime numbers."), 32, 0)
        write_json(self.out / "cancel-followup.json", followup)

    def workload(self, prompts):
        if self.args.mode == "scaling":
            for count in self.args.prompt_tokens:
                prompt = self.long_prompt(prompts, count)
                for active in self.args.sessions:
                    for repeat in range(-1, self.args.repeats):
                        result = self.decode_group(prompt, self.args.tokens, range(active), count, repeat)
                        result.update(prompt_tokens=count, warmup=repeat < 0, repeat=repeat)
                        write_json(self.out / f"decode-{count}-{active}-{repeat}.json", result)
            return
        mode = self.args.mode
        if mode == "semantics":
            self.semantics()
        elif mode in ("capacity", "concurrency"):
            count = 131072 if mode == "capacity" else 32768
            prompt = self.long_prompt(prompts, count)
            write_json(self.out / "single.json", self.complete(prompt, 64, 0, count))
            if mode == "concurrency":
                started = time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                    rows = list(pool.map(lambda slot: self.complete(prompt, 64, slot, count), range(3)))
                write_json(self.out / "concurrent.json", dict(wall_seconds=time.monotonic()-started, rows=rows))
                write_json(self.out / "tail.json", self.complete(prompt, 64, 0, count))
        else:
            chosen = [p for p in prompts if p["category"] in ("english", "chinese", "code", "tools")]
            if len(chosen) != 4 or len({p["category"] for p in chosen}) != 4:
                raise ValueError("the four public categories are required")
            for repeat in range(self.args.repeats if mode == "timing" else 1):
                for prompt in chosen:
                    row = self.complete(prompt, self.args.tokens, repeat=repeat) | dict(repeat=repeat)
                    timing = row["response"]["timings"]
                    if not math.isfinite(timing.get("predicted_ms", math.nan)) or timing["predicted_ms"] <= 0:
                        raise ValueError("invalid prediction timing")
                    with (self.out / "responses.jsonl").open("a") as output:
                        output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    print(json.dumps(dict(category=row["category"], repeat=repeat, timings=timing)), flush=True)

    def memory_samples(self):
        text = (self.out / "gpu.csv").read_text()
        # The monitor can be in the middle of a line when the file is read.
        lines = text[:text.rfind("\n")+1].splitlines()
        samples = {device: [] for device in range(self.cards)}
        for row in csv.reader(lines):
            try:
                stamp, device, memory, _power = row
                stamp = datetime.strptime(stamp.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
                device, memory = int(device), float(memory)
                if device not in samples or not math.isfinite(memory) or memory < 0:
                    raise ValueError("invalid device or memory value")
                if samples[device] and stamp < samples[device][-1][0]:
                    raise ValueError("sample time moved backwards")
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"invalid GPU memory sample {row!r}: {exc}") from exc
            samples[device].append((stamp, memory))
        return samples

    def memory_evidence(self, monitor, started=None, ended=None):
        until = time.monotonic() + self.timeout(3)
        while True:
            status = monitor.poll()
            if status is not None:
                raise RuntimeError(f"GPU memory monitor exited before cleanup ({status}); see gpu.csv and gpu-monitor.log")
            samples = self.memory_samples()
            if all(samples.values()):
                if started is None:
                    return None
                covered = {}
                for device, rows in samples.items():
                    before = [i for i, row in enumerate(rows) if row[0] <= started]
                    after = [i for i, row in enumerate(rows) if row[0] >= ended]
                    if before and after:
                        covered[device] = rows[before[-1]:after[0]+1]
                if len(covered) == self.cards:
                    for device, rows in covered.items():
                        if any(b[0]-a[0] > 2 for a, b in zip(rows, rows[1:])):
                            raise ValueError(f"GPU {device} memory samples have a gap over 2 seconds during the workload")
                    return dict(workload_start_epoch=started, workload_end_epoch=ended,
                                peak_mib={str(device): max(memory for _, memory in rows) for device, rows in samples.items()},
                                sample_counts={str(device): len(rows) for device, rows in samples.items()})
            if time.monotonic() >= until:
                raise ValueError(f"incomplete GPU memory evidence for all {self.cards} cards over the workload; "
                                 f"sample counts: { {device: len(rows) for device, rows in samples.items()} }")
            time.sleep(min(0.05, max(0, until-time.monotonic())))

    def run(self):
        self.out.mkdir(parents=True, exist_ok=False)
        self.timeout()
        prompts = json.loads((self.args.fixtures / "prompts.json").read_text())
        meta = dict(args=vars(self.args), argv=server_command(self.args), model_sha256=self.args.model_sha256,
                    binary_manifest=json.loads((self.args.binary / "manifest.json").read_text()),
                    profile=self.args.profile, instrumented=self.args.mode in ("capacity", "concurrency", "diagnostic"),
                    engine_env={k: v for k, v in self.env.items() if k.startswith(("GGML_", "LLAMA_", "CUDA_"))})
        write_json(self.out / "run.json", meta)
        child, monitor = None, None
        with ((self.out / "server.log").open("w") as log, (self.out / "gpu.csv").open("w") as gpu,
              (self.out / "gpu-monitor.log").open("w") as monitor_log):
            try:
                if self.args.mode in ("capacity", "concurrency", "diagnostic"):
                    monitor = subprocess.Popen(["stdbuf", "-oL", "nvidia-smi", "--query-gpu=timestamp,index,memory.used,power.draw",
                                                "--format=csv,noheader,nounits", "--loop-ms=500"],
                                               stdout=gpu, stderr=monitor_log, start_new_session=True)
                child = subprocess.Popen(meta["argv"], env=self.env, stdout=log, stderr=log, start_new_session=True)
                ready_until = min(self.args.deadline, time.time()+300)
                while True:
                    if monitor is not None and monitor.poll() is not None:
                        raise RuntimeError(f"GPU memory monitor exited during readiness ({monitor.returncode}); see gpu-monitor.log")
                    if child.poll() is not None:
                        raise RuntimeError(f"server exited during readiness ({child.returncode})")
                    if time.time() >= ready_until:
                        raise TimeoutError("server readiness deadline reached")
                    try:
                        if self.request("/health", maximum=3).get("status") == "ok":
                            break
                    except (OSError, ValueError):
                        pass
                    time.sleep(min(1, self.timeout()))
                write_json(self.out / "props.json", self.request("/props"))
                if self.args.profile:
                    resolved = dict(re.findall(r"GLM profile " + re.escape(self.args.profile) +
                                               r": ((?:GGML_|LLAMA_)\w+)=(\d+)", (self.out / "server.log").read_text()))
                    if len(resolved) != 13:
                        raise ValueError("server did not report the complete resolved profile")
                    meta["engine_env"].update(resolved)
                    write_json(self.out / "run.json", meta)
                # Untimed warmup is identical for both arms.
                self.request("/completion", completion_payload("Hello", 32, 0) | dict(temperature=0, top_k=1))
                if monitor is not None:
                    self.memory_evidence(monitor)
                started = time.time()
                self.workload(prompts)
                ended = time.time()
                if monitor is not None:
                    write_json(self.out / "memory.json", self.memory_evidence(monitor, started, ended))
                if child.poll() is not None or self.request("/health", maximum=5).get("status") != "ok":
                    raise RuntimeError("server did not remain healthy after the workload")
                if monitor is not None and monitor.poll() is not None:
                    raise RuntimeError(f"GPU memory monitor exited before cleanup ({monitor.returncode}); see gpu-monitor.log")
                write_json(self.out / "workload.json", dict(passed=True, mode=self.args.mode))
            finally:
                before = None if child is None else child.poll()
                monitor_before = None if monitor is None else monitor.poll()
                terminate(child)
                terminate(monitor)
                write_json(self.out / "process-exit.json", dict(before_cleanup=before,
                           pid=None if child is None else child.pid, exit=None if child is None else child.returncode,
                           monitor=dict(before_cleanup=monitor_before, pid=None if monitor is None else monitor.pid,
                                        exit=None if monitor is None else monitor.returncode)))


def run_measurements(args):
    baseline = getattr(args, "baseline_binary", None)
    if baseline is None:
        ServingLeg(args).run()
        return
    if args.mode != "scaling":
        raise ValueError("ABBA comparisons require scaling mode")
    folder = args.root / "results" / args.run
    legs = [("A1", baseline), ("B1", args.binary), ("B2", args.binary), ("A2", baseline)]
    if any((folder.parent / (args.run + "-" + name)).exists() for name, _ in legs):
        raise ValueError("ABBA output directories must be new")
    folder.mkdir(parents=True, exist_ok=False)
    receipt = dict(order=[name for name, _ in legs], completed=[], deadline=args.deadline, passed=False)
    write_json(folder / "abba.json", receipt)
    for name, binary in legs:
        leg = argparse.Namespace(**(vars(args) | dict(run=args.run + "-" + name, binary=binary)))
        ServingLeg(leg).run()
        receipt["completed"].append(leg.run)
        write_json(folder / "abba.json", receipt)
    receipt["passed"] = True
    write_json(folder / "abba.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "fixtures", "binary", "model"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--ubatch", type=int, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--mode", choices=("timing", "semantics", "capacity", "concurrency", "diagnostic", "scaling"), default="timing")
    parser.add_argument("--recipe", choices=("tensor-2", "dp-4"), default="tensor-2")
    parser.add_argument("--baseline-binary", type=Path, help="Run scaling in baseline/candidate/candidate/baseline order")
    parser.add_argument("--kv-host", action="store_true")
    parser.add_argument("--host-budget-mib", type=float)
    parser.add_argument("--mmproj", type=Path)
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[512, 4096])
    parser.add_argument("--profile", default="")
    parser.add_argument("--port", type=int, default=8102)
    parser.add_argument("--slots", type=int, default=3)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--adaptive", type=float, default=0.18)
    parser.add_argument("--sampling", choices=("served",), default="served")
    parser.add_argument("--tokens", type=int, default=384)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--threads-batch", type=int)
    args = parser.parse_args()
    if args.mode == "scaling" and (args.tokens < 2 or args.repeats < 1 or
            any(s < 1 or s > args.slots for s in args.sessions) or any(t <= 128 for t in args.prompt_tokens)):
        parser.error("scaling needs at least two output tokens, a positive repeat count, configured slots and prompts over 128 tokens")
    if args.recipe != "dp-4" and (args.kv_host or args.host_budget_mib is not None or args.mmproj):
        parser.error("host KV and projector measurement options require the DP recipe")
    if Path(args.run).name != args.run or args.run in (".", ".."):
        parser.error("run must be one new output directory name")
    def interrupted(signum, frame):
        raise InterruptedError(f"serving leg interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    run_measurements(args)


if __name__ == "__main__":
    main()
