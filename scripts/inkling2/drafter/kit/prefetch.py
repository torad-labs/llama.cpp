#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.15"
# dependencies = ["gguf==0.19.0", "numpy==2.3.5", "requests==2.32.5", "PyYAML==6.0.3", "tqdm==4.67.1", "huggingface-hub==2.4.0", "hf-xet==1.8.0"]
# ///
"""The restore's fetches alone, started before the release has finished arriving.

prefetch.py ROOT INPUTS stages the pinned stock shards and auxiliary tensors where the sealed restore looks for them
(ROOT/.inkling-stock-75970601 and ROOT/auxiliary), by the restore's own steps and under the restore's own lock, so a
restore started meanwhile waits for it and then finds every source in place. INPUTS holds the four release files the
fetches read. The restore still checks its inputs' digests and holds the pack it writes to the release's sums."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

from huggingface_hub import hf_hub_download

restore = __import__("restore-staged")


def fetch(entry, cache):
    """restore-xet.py's fetch of one stock shard, line for line (that script runs the whole restore when imported)."""
    path = cache / entry["path"]
    if path.is_file() and path.stat().st_size == entry["size"]:
        return
    repo, revision = "bartowski/Inkling-2-Small-GGUF", restore.REVISION
    prefix = f"/{repo}/resolve/{revision}/"
    url = urlsplit(entry["url"])
    if url.hostname != "huggingface.co" or not url.path.startswith(prefix):
        raise ValueError(f"unexpected stock source for {path.name}")
    filename = url.path[len(prefix):]
    print(f"Fetching pinned stock shard via Xet: {filename}", flush=True)
    began = time.monotonic()
    download = Path(hf_hub_download(repo_id=repo, revision=revision, filename=filename, local_dir=cache / "xet"))
    fetched = time.monotonic() - began
    with download.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if download.stat().st_size != entry["size"] or digest != entry["sha256"]:
        raise ValueError(f"Xet shard mismatch for {path.name}")
    download.replace(path)
    path.with_suffix(path.suffix + ".part").unlink(missing_ok=True)
    print(f"Verified Xet stock shard: {path.name} ({entry['size'] / fetched / 1e6:.0f} MB/s in {fetched:.0f} s)", flush=True)


def main():
    root, inputs = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    supplement = restore.load(inputs / "gguf-delta-supplement.py")
    entries = json.loads((inputs / "stock-manifest.json").read_text())
    ranges, _ = supplement.catalog(inputs / "delta-supplement-ranges.json")
    cache, auxiliary = root / ".inkling-stock-75970601", root / "auxiliary"
    for path in (cache, auxiliary):
        path.mkdir(exist_ok=True, mode=0o700)
    began = time.monotonic()
    with (cache / ".materialize.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with ThreadPoolExecutor(max_workers=workers) as stock, ThreadPoolExecutor(max_workers=4) as tensors:
            tasks = [stock.submit(fetch, entry, cache) for entry in entries]
            tasks.extend(tensors.submit(restore.fetch_aux, layer, entry, auxiliary) for layer, entry in ranges.items())
            for task in tasks:
                task.result()
    print(f"All pinned sources are staged in {time.monotonic() - began:.0f} s.", flush=True)


if __name__ == "__main__":
    main()
