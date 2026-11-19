#!/usr/bin/env python3
"""Prepare public calibration text or report routed-expert coverage."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlencode
from urllib.request import urlopen

LLAMA_REV = "bafa0420c1e57edf04d71587d32fe57d45df7b95"
DATASET = "HuggingFaceTB/smoltalk2"
DATA_REV = "fc6cc2103c066455aade5d7fbb346039ae36ca5e"
SPLIT = "smolagents_toolcalling_traces_think"
CODE = ("include/llama.h", "src/llama-quant.cpp", "scripts/server-test-function-call.py",
        "ggml/src/ggml-backend.cpp", "tools/server/tests/unit/test_tool_call.py")
DOCS = ("tools/server/README.md", "tools/server/README-dev.md", "docs/build.md", "README.md")


def fetch_file(name):
    url = f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{LLAMA_REV}/{name}"
    with urlopen(url, timeout=60) as response:
        return name + "\n\n" + response.read().decode()


def fetch_trace(index):
    params = dict(dataset=DATASET, revision=DATA_REV, config="SFT", split=SPLIT, offset=index, length=1)
    with urlopen("https://datasets-server.huggingface.co/rows?" + urlencode(params), timeout=60) as response:
        if response.headers.get("X-Revision") != DATA_REV:
            raise ValueError("dataset viewer returned another revision")
        record, = json.load(response)["rows"]
    if record["row_idx"] != index or record["truncated_cells"]:
        raise ValueError(f"dataset viewer did not return complete row {index}")
    return record["row"]


def render_trace(template, row):
    tools = [json.loads(line) for group in row["chat_template_kwargs"]["xml_tools"] for line in group.splitlines() if line]
    messages = []
    pending = []
    for source in row["messages"]:
        message = dict(source)
        if message["role"] == "assistant":
            content = message["content"]
            reasoning = re.match(r"<think>(.*?)</think>\s*", content, re.S)
            if reasoning:
                message["reasoning_content"] = reasoning[1]
                content = content[reasoning.end():]
            calls = []
            for text in re.findall(r"<tool_call>(.*?)</tool_call>", content, re.S):
                try:
                    call = json.loads(text)
                except json.JSONDecodeError:
                    call = ast.literal_eval(text)
                calls.append({"type": "function", "function": call})
            if calls:
                pending = [entry["function"]["name"] for entry in calls]
                message["tool_calls"] = calls
            message["content"] = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.S).strip()
        elif message["role"] == "tool" and pending:
            message["name"] = pending.pop(0)
        messages.append(message)
    return template.render(messages=messages, tools=tools, add_generation_prompt=False)


def blocks(items, n=8192):
    for item in items:
        for start in range(0, len(item), n):
            yield item[start:start + n]


def corpus(output):
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def fail(message):
        raise ValueError(message)

    template_path = Path(__file__).resolve().parents[2] / "models/templates/thinkingmachines-Inkling-2-Small.jinja"
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    environment.filters["tojson"] = json.dumps
    environment.globals["raise_exception"] = fail
    template = environment.from_string(template_path.read_text())
    with ThreadPoolExecutor(max_workers=4) as pool:
        code = list(pool.map(fetch_file, CODE))
        docs = list(pool.map(fetch_file, DOCS))
        traces = [render_trace(template, row) for row in pool.map(fetch_trace, range(0, 1024, 41))]
    pieces = zip_longest(blocks(code), blocks(docs), blocks(traces))
    text = "\n\n".join(item for group in pieces for item in group if item).encode()
    with output.open("xb") as out:
        out.write(text)
    print(f"{hashlib.sha256(text).hexdigest()}  {output} ({len(text):,} bytes)")


def coverage(path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
    import gguf
    import numpy as np

    matrix = gguf.GGUFReader(path)
    tensors = {tensor.name: tensor for tensor in matrix.tensors}
    chunks = int(matrix.fields["imatrix.chunk_count"].contents())
    length = int(matrix.fields["imatrix.chunk_size"].contents())
    observed = []
    for layer in range(48):
        projections = []
        for part, width in (("gate_up", 4608), ("down", 2304)):
            name = f"blk.{layer}.ffn_{part}_exps.weight"
            sums = tensors[name + ".in_sum2"]
            counts = tensors[name + ".counts"].data.reshape(-1)
            if tuple(sums.shape) != (width, 384) or counts.size != 384 or not np.all(np.isfinite(counts) & (counts >= 0)):
                raise ValueError(f"invalid shape or counts for {name}")
            projections.append(counts)
        if not np.array_equal(*projections):
            raise ValueError(f"gate/up and down routing counts disagree at layer {layer}")
        observed.append(projections[0])
    counts = np.stack(observed)
    print(f"{chunks} chunks, {length} tokens/chunk. {counts.shape[0]} layers, {counts.shape[1]} experts/layer.")
    print(f"Unseen layer/expert pairs: {int((counts == 0).sum())}/{counts.size}. "
          f"Pairs with fewer than 16 observations: {int((counts < 16).sum())}.")
    print(f"Observations per layer/expert pair: min {counts.min():g}, P1 {np.percentile(counts, 1):g}, "
          f"median {np.median(counts):g}, max {counts.max():g}.")
    routes = counts.sum(axis=1, dtype=np.float64)
    print(f"Routed token assignments per layer: {routes.min():g} to {routes.max():g}; expected {8 * chunks * length} for complete chunks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("corpus", "coverage"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "corpus":
        corpus(args.path)
    else:
        coverage(args.path)
