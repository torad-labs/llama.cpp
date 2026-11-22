#!/usr/bin/env python3
"""Prepare fixed replay tokens from the exact completed client probe.

Use the trusted probe.py from that capture and its matching tokenizer on
loopback. Only its prepare() helper runs: no completions or servers are
started. Reject any changed common or private prompt hashes.
"""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import time
from types import SimpleNamespace
from urllib.parse import urlsplit


def token_digest(tokens):
    return hashlib.sha256(",".join(str(t) for t in tokens).encode("ascii")).hexdigest()


def replay_manifest(reference, cases, build_sha256, model_sha256):
    clients = sorted(reference["clients"], key=lambda c: c["client"])
    n_common = reference["common"]
    common = None
    private, hashes, forced = [], [], []
    if len(clients) != len(cases) or reference["status"] != "full_reuse":
        raise ValueError("reference is not the complete recorded workload")
    for (wire, prepared), client in zip(cases, clients):
        ids = json.loads(wire)["prompt"]
        prefix, tail = ids[:n_common], ids[n_common:]
        tail_sha = token_digest(tail)
        if (prepared["client"] != client["client"] or len(tail) != reference["private"] or
                tail_sha != client["private_sha256"] or token_digest(prefix) != reference["common_sha256"]):
            raise ValueError(f"regenerated prompt does not match client {client['client']}")
        if len(client["token_ids"]) < 4:
            raise ValueError(f"client {client['client']} has fewer than four forced tokens")
        if common is None:
            common = prefix
        private.append(tail)
        hashes.append(tail_sha)
        forced.append(client["token_ids"][:4])
    return {
        "build_sha256": build_sha256, "model_sha256": model_sha256,
        "common": common, "private": private, "forced": forced,
        "common_sha256": reference["common_sha256"], "private_sha256": hashes,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", required=True, type=Path, help="trusted probe.py used for the reference run")
    p.add_argument("--reference", required=True, type=Path, help="stock_a.json")
    p.add_argument("--tokenizer", required=True, help="loopback URL for the original model's tokenizer")
    p.add_argument("--build-sha256", required=True)
    p.add_argument("--model-sha256", required=True)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    url = urlsplit(args.tokenizer)
    if url.scheme != "http" or url.hostname not in ("127.0.0.1", "::1", "localhost") or url.username or url.password:
        p.error("use the trusted local tokenizer on loopback")
    reference = json.loads(args.reference.read_bytes())
    config = SimpleNamespace(**{k: reference[k] for k in ("sessions", "common", "private", "seed", "arrival")})
    probe = runpy.run_path(str(args.probe.resolve()))
    common_sha, cases = probe["prepare"](config, url, time.monotonic() + reference["timeout_s"])
    if common_sha != reference["common_sha256"]:
        p.error("regenerated common prompt differs from the recorded prompt")
    manifest = replay_manifest(reference, cases, args.build_sha256, args.model_sha256)
    with args.out.open("x") as f:
        json.dump(manifest, f, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main()
