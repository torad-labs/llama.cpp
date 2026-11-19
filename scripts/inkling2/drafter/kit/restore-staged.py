#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.15"
# dependencies = ["gguf==0.19.0", "numpy==2.3.5", "requests==2.32.5", "PyYAML==6.0.3", "tqdm==4.67.1"]
# ///
"""Stage pinned stock shards and auxiliary tensors together, then run the sealed restore."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import requests

REVISION = "75970601c5da75212b2ffb914344e1b7dae2008d"


def load(path):
    spec = importlib.util.spec_from_file_location("release_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fetch_aux(layer, entry, cache):
    path = cache / f"{layer}.bin"
    size = entry["size"]
    if path.is_file() and path.stat().st_size == size:
        return
    part = path.with_suffix(".part")
    print(f"Fetching auxiliary tensor for layer {layer}", flush=True)
    with part.open("ab") as out, requests.Session() as session:
        if out.tell() > size:
            raise ValueError(f"oversized auxiliary tensor for layer {layer}")
        while out.tell() < size:
            end = min(out.tell() + (128 << 20), size)
            for attempt in range(5):
                first = entry["offset"] + out.tell()
                last = entry["offset"] + end - 1
                try:
                    headers = {"Range": f"bytes={first}-{last}", "Accept-Encoding": "identity"}
                    with session.get(entry["url"], params={"stage_aux": f"{first}-{last}"},
                                     headers=headers, stream=True, timeout=(20, 90)) as reply:
                        reply.raise_for_status()
                        expected = f"bytes {first}-{last}/{entry['file_size']}"
                        if reply.status_code != 206 or reply.headers.get("Content-Range") != expected:
                            raise ValueError(f"unexpected HTTP range for layer {layer}")
                        if not any(h.headers.get("X-Linked-ETag", "").strip('"') == entry["sha256"]
                                   and h.headers.get("X-Repo-Commit") == REVISION for h in reply.history):
                            raise ValueError(f"unpinned response for auxiliary layer {layer}")
                        for chunk in reply.iter_content(1 << 20):
                            if out.tell() + len(chunk) > end:
                                raise ValueError(f"oversized HTTP chunk for layer {layer}")
                            out.write(chunk)
                    if out.tell() != end:
                        raise requests.ConnectionError(f"short auxiliary range for layer {layer}")
                    break
                except requests.RequestException:
                    if out.tell() == end:
                        break
                    if attempt == 4:
                        raise
                    time.sleep(2**attempt)
    part.replace(path)
    print(f"Staged auxiliary layer {layer} ({size} bytes)", flush=True)


def materialize_staged(materializer, release, output, cache, auxiliary, **extra):
    script = Path(__file__).resolve()
    unwrap = release / "gguf-delta-supplement.py"

    def launch(argv, **kwargs):
        if argv[:3] == [sys.executable, str(unwrap), "unwrap"]:
            argv = [sys.executable, str(script), "--unwrap", str(unwrap), str(auxiliary), *argv[2:]]
        return subprocess.Popen(argv, **kwargs)

    original = materializer.subprocess
    materializer.subprocess = SimpleNamespace(Popen=launch, run=subprocess.run, PIPE=subprocess.PIPE)
    try:
        return materializer.materialize(release, output, cache, **extra)
    finally:
        materializer.subprocess = original


def unwrap_staged(path, auxiliary):
    tool = load(path)

    class LocalRanges(tool.Ranges):
        def contents(self, layer, relative, size):
            entry = self.table[layer]
            tool.p.require(0 <= relative and 0 < size <= tool.p.BLOCK and relative + size <= entry["size"],
                           "invalid auxiliary reference")
            with (auxiliary / f"{layer}.bin").open("rb") as source:
                return tool.p.read_at(source, relative, size)

    tool.Ranges = LocalRanges
    tool.main()


def restore(root, fetch_stock=None, stock_workers=8, aux_workers=4):
    release = root / "release"
    materializer = load(release / "materialize-inkling.py")
    supplement = load(release / "gguf-delta-supplement.py")
    for name, digest in materializer.PINS.items():
        materializer.require(materializer.digest(release / name) == digest, f"unexpected restore input: {name}")
    entries = json.loads((release / "stock-manifest.json").read_text())
    ranges, _ = supplement.catalog(release / "delta-supplement-ranges.json")
    cache, auxiliary = root / ".inkling-stock-75970601", root / "auxiliary"
    for path in (cache, auxiliary):
        path.mkdir(exist_ok=True, mode=0o700)
    receipt = json.loads(materializer.receipt(release, "supplement-receiver.json").read_text())
    with (cache / ".materialize.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        missing_aux = sum(entry["size"] for layer, entry in ranges.items() if not (auxiliary / f"{layer}.bin").is_file())
        output_size = receipt["target_bytes"] + (release / materializer.PROJECTOR).stat().st_size
        materializer.check_space(cache, root, entries, output_size + missing_aux)
        with ThreadPoolExecutor(max_workers=stock_workers) as stock, ThreadPoolExecutor(max_workers=aux_workers) as tensors:
            tasks = [stock.submit(fetch_stock or materializer.fetch, entry, cache) for entry in entries]
            tasks.extend(tensors.submit(fetch_aux, layer, entry, auxiliary) for layer, entry in ranges.items())
            for task in tasks:
                task.result()
    print("All pinned sources are staged. Restoring the verified target.", flush=True)
    dest = materialize_staged(materializer, release, root / "model", cache, auxiliary,
                              pins=materializer.PINS, proj_sha=materializer.PROJECTOR_SHA)
    print(f"Verified model and projector in {dest}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "--unwrap":
        path, directory = map(Path, sys.argv[2:4])
        del sys.argv[1:4]
        unwrap_staged(path, directory)
    elif len(sys.argv) == 2:
        restore(Path(sys.argv[1]).resolve())
    else:
        sys.exit("usage: restore-staged.py ROOT")
