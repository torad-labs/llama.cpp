#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.15"
# dependencies = ["gguf==0.19.0", "numpy==2.3.5", "requests==2.32.5", "PyYAML==6.0.3", "tqdm==4.67.1", "huggingface-hub==2.4.0", "hf-xet==1.8.0"]
# ///
"""Use native Xet for stock shards and keep the release's exact verification."""
import hashlib
from pathlib import Path
import sys
from urllib.parse import urlsplit

from huggingface_hub import hf_hub_download


restore = __import__("restore-staged")


def fetch(entry, cache):
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
    download = Path(hf_hub_download(repo_id=repo, revision=revision, filename=filename, local_dir=cache / "xet"))
    with download.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if download.stat().st_size != entry["size"] or digest != entry["sha256"]:
        raise ValueError(f"Xet shard mismatch for {path.name}")
    download.replace(path)
    path.with_suffix(path.suffix + ".part").unlink(missing_ok=True)
    print(f"Verified Xet stock shard: {path.name}", flush=True)


restore.restore(Path(sys.argv[1]).resolve(), fetch_stock=fetch, stock_workers=2, aux_workers=4)
