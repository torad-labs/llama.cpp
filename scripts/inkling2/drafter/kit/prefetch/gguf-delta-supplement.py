#!/usr/bin/env python3
"""Compact or expand a GGUFDEL1 stream using verified, immutable alternate public tensor ranges."""
import argparse
from contextlib import nullcontext, ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import struct
import sys
import time

import gguf
import requests

script = Path(__file__).resolve().with_name("gguf-delta.py")
spec = importlib.util.spec_from_file_location("inkling_primary_delta", script)
p = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p
spec.loader.exec_module(p)

MAGIC = b"GGSUPP01"
A_COPY, A_XOR, A_BITXOR = 128, 129, 130
TAIL = struct.Struct("<Q32s")
BASE_URL = "https://huggingface.co/bartowski/Inkling-2-Small-GGUF/resolve/75970601c5da75212b2ffb914344e1b7dae2008d/"
WINDOW = 32 << 20


def canon(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).digest()


def catalog(path):
    obj = json.loads(path.read_text())
    for key, t in obj.items():
        part = Path(t["path"])
        p.require(key == str(int(key)) and 0 <= int(key) < 128 and not part.is_absolute() and ".." not in part.parts
                  and t["url"] == BASE_URL + t["path"] and t["qtype"] == "Q4_K"
                  and re.fullmatch("[0-9a-f]{64}", t["sha256"]) and 0 <= t["offset"] < t["offset"] + t["size"] <= t["file_size"],
                  "invalid pinned range catalog")
    return {int(k): v for k, v in obj.items()}, canon(obj)


class Ranges:
    def __init__(self, table, local=None):
        self.table, self.local = table, local
        self.http = requests.Session()
        self.shard = None
        self.start = 0
        self.data = b""
        self.network = 0
        self.verified_files = set()

    def contents(self, layer, relative, size):
        t = self.table[layer]
        p.require(0 <= relative and 0 < size <= p.BLOCK and relative + size <= t["size"], "invalid auxiliary reference")
        if self.shard == layer and self.start <= relative and relative + size <= self.start + len(self.data):
            begin = relative - self.start
            return self.data[begin:begin + size]
        length = min(WINDOW, t["size"] - relative)
        absolute = t["offset"] + relative
        if self.local:
            path = self.local / t["path"]
            if path not in self.verified_files:
                p.require(path.stat().st_size == t["file_size"] and p.digest(path) == t["sha256"], "local aux shard hash mismatch")
                self.verified_files.add(path)
            with path.open("rb") as f:
                data = p.read_at(f, absolute, length)
        else:
            end = absolute + length - 1
            for attempt in range(4):
                try:
                    with self.http.get(t["url"] + f"?aux_v1={absolute}-{end}", headers={"Range": f"bytes={absolute}-{end}",
                                       "Accept-Encoding": "identity"}, timeout=(12, 90), stream=True) as r:
                        r.raise_for_status()
                        p.require(r.status_code == 206 and r.headers.get("Content-Range") == f"bytes {absolute}-{end}/{t['file_size']}",
                                  "server did not return the pinned range")
                        p.require(any(h.headers.get("X-Linked-ETag", "").strip('"') == t["sha256"]
                                      and h.headers.get("X-Repo-Commit") == "75970601c5da75212b2ffb914344e1b7dae2008d"
                                      for h in r.history), "HF did not attest the pinned revision and LFS identity")
                        data = r.content
                        p.require(len(data) == length, "short public tensor range")
                        self.network += len(data)
                        break
                except requests.RequestException:
                    if attempt == 3:
                        raise
                    time.sleep(1 << attempt)
        self.shard, self.start, self.data = layer, relative, data
        return data[:size]


def layout(args):
    table, key = catalog(args.ranges)
    reader = gguf.GGUFReader(args.target)
    tensors = {t.name: t for t in reader.tensors}
    result = {}
    for layer, base in table.items():
        t = tensors[f"blk.{layer}.ffn_down_exps.weight"]
        p.require(t.tensor_type.name == base["qtype"] and list(map(int,t.shape)) == base["shape"]
                  and t.n_bytes == base["size"], f"target layout differs for layer {layer}")
        result[layer] = [int(t.data_offset), int(t.n_bytes)]
    with args.target.open("rb") as f:
        header_hash = hashlib.sha256(p.exact(f, reader.data_offset)).hexdigest()
    return dict(target_bytes=args.target.stat().st_size, header_bytes=reader.data_offset, header_sha256=header_hash,
                catalog=key.hex(), ranges=result)


def primary_create():
    # Preserve the native record bytes for later replay. Only the selected alternate-base tensors need raw literals.
    selected = False
    name = Path(__file__).resolve().with_name("delta-supplement-ranges.json")
    if "--ranges" in sys.argv:
        ix = sys.argv.index("--ranges")
        name = Path(sys.argv[ix+1])
        del sys.argv[ix:ix+2]
    table, _ = catalog(name)
    names = {f"blk.{layer}.ffn_down_exps.weight" for layer in table}
    regular_encode, regular_segments = p.encode, p.segments
    def marked_segments(tensors, size):
        nonlocal selected
        for start, length, tensor in regular_segments(tensors, size):
            selected = tensor is not None and tensor.name in names
            yield start, length, tensor
    def raw_unmatched(raw, base, codec, bitplanes):
        if selected and base is None and raw.count(0) != len(raw):
            return p.RAW, raw
        return regular_encode(raw, base, codec, bitplanes)
    p.segments = marked_segments
    p.encode = raw_unmatched
    sys.argv[1] = "create"
    p.main()


def primary_record(src):
    head = p.exact(src, p.REC.size)
    kind, idx, off, n, stored = p.REC.unpack(head)
    if kind == p.END:
        p.require((idx, off, n, stored) == (p.NONREF, 0, 0, 32), "invalid primary footer")
    else:
        p.require(kind in range(len(p.KINDS)) and 0 < n <= p.BLOCK and stored <= n, "invalid primary record")
    return head, kind, n, p.exact(src, stored)


def wrap(args, ranges, catalog_sha, src, out):
    lay = json.loads(args.layout.read_text())
    header = p.exact(src, 48)
    p.require(header[:8] == p.MAGIC and header[8:40] == p.manifest(args.manifest)[1], "wrong primary header or base manifest")
    size, = struct.unpack("<Q", header[40:])
    candidates = sorted((start, start + n, int(layer)) for layer, (start, n) in lay["ranges"].items())
    p.require(lay["catalog"] == catalog_sha.hex() and lay["target_bytes"] == size and
              len(candidates) == len(ranges.table) and set(map(int,lay["ranges"])) == set(ranges.table), "wrong target layout")
    for ix, (start, end, layer) in enumerate(candidates):
        p.require(lay["header_bytes"] <= start < end <= size and end-start == ranges.table[layer]["size"]
                  and (ix == 0 or candidates[ix-1][1] <= start), "invalid target tensor bounds")
    limit = int(args.max_gib * p.GIB)
    primary_sha, supplement_sha, hdr_sha = hashlib.sha256(header), hashlib.sha256(), hashlib.sha256()
    primary_size, stored_size, cursor, chosen, rank, candidate_bytes, zliteral_bytes = len(header), 0, 0, 0, 0, 0, 0
    codec = p.Zstd(3)
    def emit(data):
        nonlocal stored_size
        p.require(stored_size + len(data) <= limit, "supplement exceeds cap")
        out.write(data)
        supplement_sha.update(data)
        stored_size += len(data)
    emit(MAGIC + catalog_sha + header)
    while True:
        head, kind, n, data = primary_record(src)
        primary_sha.update(head)
        primary_sha.update(data)
        primary_size += len(head) + len(data)
        if kind == p.END:
            p.require(cursor == size and src.read(1) == b"", "incomplete or trailing primary stream")
            emit(head + data + TAIL.pack(primary_size, primary_sha.digest()))
            p.require(stored_size < primary_size, "supplement did not improve on primary archive")
            return dict(primary_bytes=primary_size, primary_sha256=primary_sha.hexdigest(), target_bytes=size,
                        target_sha256=data.hex(), supplement_bytes=stored_size, supplement_sha256=supplement_sha.hexdigest(),
                        alternate_references=chosen, candidate_bytes=candidate_bytes, ordinary_compressed_literal_bytes=zliteral_bytes,
                        public_download_bytes=ranges.network)
        if cursor < lay["header_bytes"]:
            p.require(cursor + n <= lay["header_bytes"] and kind in (p.RAW, p.ZERO, p.ZRAW), "invalid source header extent")
            chunk = data if kind == p.RAW else (bytes(n) if kind == p.ZERO else codec.decompress(data, n))
            hdr_sha.update(chunk)
            if cursor + n == lay["header_bytes"]:
                p.require(hdr_sha.hexdigest() == lay["header_sha256"], "wrong target header")
        while rank < len(candidates) and candidates[rank][1] <= cursor:
            rank += 1
        if rank < len(candidates):
            start, end, layer = candidates[rank]
            if start <= cursor and cursor + n <= end:
                candidate_bytes += n
                if kind == p.ZRAW:
                    zliteral_bytes += n
                if kind == p.RAW:
                    p.require(len(data) == n, "invalid raw primary literal")
                    public = ranges.contents(layer, cursor - start, n)
                    code, body = p.encode(data, public, codec, True)
                    kinds = {p.COPY: A_COPY, p.ZXOR: A_XOR, p.BITXOR: A_BITXOR}
                    if code in kinds and 32 + len(body) < len(data):
                        data = hashlib.sha256(public).digest() + body
                        head = p.REC.pack(kinds[code], layer, cursor-start, n, len(data))
                        chosen += 1
        emit(head)
        emit(data)
        cursor += n


def unwrap(args, ranges, catalog_sha, src, out):
    prefix = p.exact(src, 88)
    p.require(prefix[:40] == MAGIC + catalog_sha and prefix[40:48] == p.MAGIC
              and prefix[48:80] == p.manifest(args.manifest)[1], "wrong supplementary catalog or primary manifest")
    observed_sha = hashlib.sha256(prefix)
    primary_sha = hashlib.sha256(prefix[40:])
    size, = struct.unpack("<Q", prefix[80:])
    cursor, written = 0, 48
    out.write(prefix[40:])
    codec = p.Zstd(3)
    while True:
        head = p.exact(src, p.REC.size)
        kind, idx, off, n, stored = p.REC.unpack(head)
        p.require(stored <= p.BLOCK + 32,"excessive payload")
        data = p.exact(src, stored)
        observed_sha.update(head)
        observed_sha.update(data)
        if kind == p.END:
            tail = p.exact(src, TAIL.size)
            observed_sha.update(tail)
            primary_sha.update(head + data)
            written += len(head) + len(data)
            p.require((idx,off,n,stored) == (p.NONREF,0,0,32) and cursor == size and
                      TAIL.unpack(tail) == (written, primary_sha.digest()) and src.read(1) == b"", "primary archive digest mismatch")
            if args.receipt:
                rec = json.loads(args.receipt.read_text())
                p.require(rec["supplement_sha256"] == observed_sha.hexdigest() and rec["target_sha256"] == data.hex()
                          and rec["primary_sha256"] == primary_sha.hexdigest(), "supplement differs from receipt")
            # The downstream original restorer can only publish its output after all transport checks finish.
            out.write(head + data)
            return dict(primary_bytes=written, primary_sha256=primary_sha.hexdigest(), target_sha256=data.hex(),
                        supplement_sha256=observed_sha.hexdigest(),public_download_bytes=ranges.network)
        p.require(0 < n <= p.BLOCK and cursor + n <= size,"invalid record extent")
        if kind in (A_COPY, A_XOR, A_BITXOR):
            p.require(idx in ranges.table and 32<=stored<n and (kind!=A_COPY or stored==32), "invalid auxiliary payload")
            public = ranges.contents(idx, off, n)
            p.require(hashlib.sha256(public).digest() == data[:32], "public reference range SHA-256 mismatch")
            if kind == A_COPY:
                data = public
            else:
                diff = codec.decompress(data[32:], n)
                if kind == A_BITXOR:
                    p.require(n % 8 == 0, "invalid bitplane extent")
                    diff = p.planes(diff, True)
                data = p.xor(diff, public)
            head = p.REC.pack(p.RAW, p.NONREF, 0, n, n)
        else:
            p.require(kind in range(len(p.KINDS)) and stored<=n, "invalid primary record")
        out.write(head)
        out.write(data)
        primary_sha.update(head)
        primary_sha.update(data)
        written += len(head) + len(data)
        cursor += n


def publish_received(args):
    sender = json.loads(args.primary_receipt.read_text())
    wrapped = json.loads(args.supplement_receipt.read_text())
    p.require(all(sender[a] == wrapped[b] for a,b in (("delta_bytes","primary_bytes"),("delta_sha256","primary_sha256"),
                     ("target_bytes","target_bytes"),("target_sha256","target_sha256"))),
              "sender and receiver disagree about the primary archive or original model")
    p.require(not os.path.lexists(args.output) and not args.partial.is_symlink(), "output exists or partial is a symlink")
    before = p.identity(args.partial.stat())
    p.require(before[2] == wrapped["supplement_bytes"] and p.digest(args.partial) == wrapped["supplement_sha256"]
              and before == p.identity(args.partial.stat()), "local supplement failed SHA-256 verification")
    os.link(args.partial,args.output)
    args.partial.unlink()
    return dict(source_primary_sha256=sender["delta_sha256"],target_sha256=sender["target_sha256"],
                supplement_sha256=wrapped["supplement_sha256"],supplement_bytes=wrapped["supplement_bytes"])


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "primary-create":
        primary_create()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command",required=True)
    recv = subs.add_parser("publish",help="verify sender and receiver receipts and atomically retain the supplement")
    for flag in ("primary-receipt","supplement-receipt","partial","output"):
        recv.add_argument("--"+flag,type=Path,required=True)
    for cmd in ("layout","wrap","unwrap"):
        a = subs.add_parser(cmd)
        a.add_argument("--ranges",type=Path,required=True)
        if cmd == "layout":
            a.add_argument("--target",type=Path,required=True)
            continue
        a.add_argument("--manifest",type=Path,required=True)
        a.add_argument("--input",type=Path, default=Path("-"))
        a.add_argument("--output",type=Path, required=True)
        a.add_argument("--aux-dir", type=Path, help="optional full locally cached alternate shards, SHA-256 verified")
        if cmd == "wrap":
            a.add_argument("--layout",type=Path,required=True)
            a.add_argument("--max-gib",type=float,required=True)
        else:
            a.add_argument("--receipt",type=Path)
    args = parser.parse_args()
    if args.command == "publish":
        print(json.dumps(publish_received(args),indent=2))
        return
    if args.command == "layout":
        print(json.dumps(layout(args),indent=2))
        return
    table, key = catalog(args.ranges)
    with ExitStack() as st:
        ranges = Ranges(table,args.aux_dir)
        st.callback(ranges.http.close)
        src = sys.stdin.buffer if args.input == Path("-") else st.enter_context(args.input.open("rb"))
        dest = sys.stdout.buffer if args.output == Path("-") else st.enter_context(p.publish(args.output))
        result = (wrap if args.command == "wrap" else unwrap)(args,ranges,key,src,dest)
        dest.flush()
    print(json.dumps(result,indent=2),file=sys.stderr if args.output == Path("-") else sys.stdout)


if __name__=="__main__":
    main()
