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
import subprocess
import threading
import time
import urllib.error
import urllib.request


def completion_payload(prompt, tokens, slot):
    return dict(prompt=prompt, n_predict=tokens, id_slot=slot, temperature=1.0,
                top_p=0.95, repeat_penalty=1.1, repeat_last_n=256, seed=530,
                ignore_eos=True, cache_prompt=False)

def server_command(args):
    argv = [str(Path(args.binary) / "llama-server"), "-m", str(args.model), "-ngl", "99",
            "-sm", "tensor", "-ts", "1,1", "--jinja", "-fa", "on", "-ctk", "q8_0",
            "-ctv", "q8_0", "-ctki", "q8_0", "-b", "4096", "-ub", str(args.ubatch),
            "-c", str(args.context), "-np", str(args.slots), "--kv-unified", "--cache-ram", "0",
            "--no-cache-idle-slots", "--metrics", "--host", "127.0.0.1", "--port", str(args.port),
            "--spec-type", "draft-mtp", "--spec-draft-n-max", str(args.depth),
            "--spec-draft-mtp-vocab", str(Path(args.fixtures) / "glm-mtp-public-98304.i32"),
            "--spec-draft-mtp-adaptive", str(args.adaptive)]
    return argv + (["--glm53-profile", args.profile] if args.profile else [])

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


def retain_body(response, saved):
    try:
        raw = response.read()
    except http.client.IncompleteRead as exc:
        saved.write(exc.partial)
        raise
    saved.write(raw)
    return raw


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
                with exc, raw.open("wb") as saved:
                    retain_body(exc, saved)
                raise
            meta["status"] = response.status
            with response, raw.open("wb") as saved:
                yield response, saved
        except Exception as exc:
            meta["error"] = f"{type(exc).__name__}: {exc}"
            message = f"request {path} failed: {meta['error']}; record {record.relative_to(self.out)}"
            if isinstance(exc, ValueError):
                raise ValueError(message) from exc
            if isinstance(exc, OSError):
                raise OSError(message) from exc
            raise RuntimeError(message) from exc
        finally:
            meta["ended_epoch"] = time.time()
            write_json(record, meta)

    def request(self, path, data=None, maximum=300, record=None, context=None):
        with self.capture_response(path, data, maximum, record, context) as (response, saved):
            return json.loads(retain_body(response, saved))

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
            for line in stream:
                saved.write(line)
                self.timeout()
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
        samples = {0: [], 1: []}
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
                if len(covered) == 2:
                    for device, rows in covered.items():
                        if any(b[0]-a[0] > 2 for a, b in zip(rows, rows[1:])):
                            raise ValueError(f"GPU {device} memory samples have a gap over 2 seconds during the workload")
                    return dict(workload_start_epoch=started, workload_end_epoch=ended,
                                peak_mib={str(device): max(memory for _, memory in rows) for device, rows in samples.items()},
                                sample_counts={str(device): len(rows) for device, rows in samples.items()})
            if time.monotonic() >= until:
                raise ValueError(f"incomplete GPU memory evidence for both cards over the workload; "
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "fixtures", "binary", "model"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--ubatch", type=int, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--mode", choices=("timing", "semantics", "capacity", "concurrency", "diagnostic"), default="timing")
    parser.add_argument("--profile", default="")
    parser.add_argument("--port", type=int, default=8102)
    parser.add_argument("--slots", type=int, default=3)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--adaptive", type=float, default=0.18)
    parser.add_argument("--sampling", choices=("served",), default="served")
    parser.add_argument("--tokens", type=int, default=384)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        parser.error("run must be one new output directory name")
    def interrupted(signum, frame):
        raise InterruptedError(f"serving leg interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    ServingLeg(args).run()


if __name__ == "__main__":
    main()
