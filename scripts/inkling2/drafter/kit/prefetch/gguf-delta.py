#!/usr/bin/env python3
"""Lossless, tensor-aligned, streaming GGUF delta against immutable local HF shards."""
import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack, nullcontext, redirect_stdout
import ctypes
from ctypes.util import find_library
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import struct
import sys
import subprocess
import tempfile

import gguf
import numpy as np

MAGIC = b"GGUFDEL1"
# kind, base shard, absolute base offset, output length, stored length
REC = struct.Struct("<BBQII")
COPY, ZERO, RAW, ZRAW, ZXOR, BITXOR, END = range(7)
KINDS = ["copy", "zero", "raw", "zstd", "xor-zstd", "bitplane-xor-zstd"]
NONREF = 255
BLOCK = 4 << 20
GIB = 1 << 30


def require(ok, message):
    if not ok:
        raise ValueError(message)


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple
    qtype: str
    start: int
    size: int
    shard: int = NONREF


def dims(d):
    a = list(map(int, d))
    while len(a) > 1 and a[-1] == 1:
        a.pop()
    return tuple(a)


def index(path, shard=NONREF):
    reader = gguf.GGUFReader(path)
    return [Tensor(t.name, dims(t.shape), t.tensor_type.name, int(t.data_offset), int(t.n_bytes), shard)
            for t in reader.tensors]


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def manifest(path):
    spec = json.loads(path.read_text())
    require(isinstance(spec, list) and 0 < len(spec) <= NONREF, "invalid base shard list")
    names = set()
    for s in spec:
        name = s["path"]
        require(isinstance(name, str) and Path(name).name == name and name not in names
                and name not in (".", "..") and s["size"] > 0
                and re.fullmatch("[0-9a-f]{64}", s["sha256"]), "invalid base shard descriptor")
        names.add(name)
    data = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return spec, hashlib.sha256(data).digest()


def open_bases(stack, spec, folder, verify, partial):
    files, table = {}, {}
    for i, s in enumerate(spec):
        path = folder / s["path"]
        if partial and (not path.is_file() or Path(str(path.resolve()) + ".aria2").exists()
                        or path.stat().st_size != s["size"]):
            continue
        f = stack.enter_context(path.open("rb"))
        opened = identity(os.fstat(f.fileno()))
        require(opened[2] == s["size"], f"base size mismatch: {path}")
        if verify:
            require(hashlib.file_digest(f, "sha256").hexdigest() == s["sha256"], f"base SHA-256 mismatch: {path}")
        require(identity(os.fstat(f.fileno())) == opened, f"base changed while verifying: {path}")
        files[i] = f
        for t in index(path, i):
            require(t.name not in table, f"duplicate tensor in base shards: {t.name}")
            require(t.start + t.size <= s["size"], f"invalid base tensor: {t.name}")
            table[t.name] = t
    require(files or partial, "no complete base shards available")
    return files, table


def add_catalog(path, manifest_hash, spec, files, table):
    catalog = json.loads(path.read_text())
    require(catalog["manifest_sha256"] == manifest_hash.hex(), "catalog belongs to a different base manifest")
    catalog_table = {}
    require(len(catalog["shards"]) == len(spec), "incomplete base catalog")
    for i, shard in enumerate(catalog["shards"]):
        require(shard["shard"] == i and 0 < shard["header_bytes"] < spec[i]["size"], "invalid catalog shard")
        if i in files:
            # Check every available local header even when the estimate skips full payload hashing.
            h = hashlib.sha256(read_at(files[i], 0, shard["header_bytes"])).hexdigest()
            require(h == shard["header_sha256"], f"base header SHA-256 mismatch in shard {i}")
        for name, qtype, shape, size, offset in shard["tensors"]:
            t = Tensor(name, dims(shape), qtype, offset, size, i)
            require(name not in catalog_table and 0 < size and shard["header_bytes"] <= offset <= spec[i]["size"] - size,
                    f"invalid catalog tensor {name}")
            require(i not in files or table.get(name) == t, f"catalog/local disagreement: {name}")
            catalog_table[name] = t
    require(all(catalog_table.get(k) == v for k, v in table.items()), "catalog omitted a local tensor")
    return catalog_table


class Zstd:
    def __init__(self, level):
        self.level = level
        self.lib = ctypes.CDLL(find_library("zstd") or "libzstd.so.1")
        signatures = {"ZSTD_compressBound": [ctypes.c_size_t], "ZSTD_isError": [ctypes.c_size_t],
                      "ZSTD_compress": [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int],
                      "ZSTD_decompress": [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]}
        for name, sig in signatures.items():
            func = getattr(self.lib, name)
            func.argtypes = sig
            func.restype = ctypes.c_size_t if name != "ZSTD_isError" else ctypes.c_uint

    def compress(self, raw):
        capacity = self.lib.ZSTD_compressBound(len(raw))
        dst = ctypes.create_string_buffer(capacity)
        n = self.lib.ZSTD_compress(dst, capacity, raw, len(raw), self.level)
        require(not self.lib.ZSTD_isError(n), "zstd compression failed")
        return dst[:n]

    def decompress(self, data, length):
        dst = ctypes.create_string_buffer(length)
        n = self.lib.ZSTD_decompress(dst, length, data, len(data))
        require(n == length and not self.lib.ZSTD_isError(n), "invalid zstd frame or decoded size")
        return dst[:n]


def xor(a, b):
    return np.bitwise_xor(np.frombuffer(a, np.uint8), np.frombuffer(b, np.uint8)).tobytes()


def planes(data, reverse=False):
    # Both reshape dimensions divide the buffer exactly when its length is a multiple of eight.
    bits = np.unpackbits(np.frombuffer(data, np.uint8))
    return np.packbits(bits.reshape((8, -1) if reverse else (-1, 8)).T).tobytes()


def encode(raw, base, codec, bitplanes):
    if base is not None and raw == base:
        return COPY, b""
    if raw.count(0) == len(raw):
        return ZERO, b""
    if base is not None:
        diff = xor(raw, base)
        best = ZXOR, codec.compress(diff)
        if bitplanes and len(raw) % 8 == 0:
            candidate = codec.compress(planes(diff))
            if len(candidate) < len(best[1]):
                best = BITXOR, candidate
        # Subpercent deltas win against any ordinary quantized literal, and a literal isn't needed for correctness.
        if len(best[1]) < len(raw) // 100:
            return best
    else:
        best = RAW, raw
    literal = codec.compress(raw)
    return min((best, (ZRAW, literal), (RAW, raw)), key=lambda k: len(k[1]))


def read_at(f, pos, n):
    result = os.pread(f.fileno(), n, pos)
    require(len(result) == n, f"short read: offset={pos} length={n}")
    return result


def matching(t, table):
    b = table.get(t.name)
    return b if b and (b.shape, b.qtype, b.size) == (t.shape, t.qtype, t.size) else None


def segments(tensors, size):
    pos = 0
    for t in sorted(tensors, key=lambda t: t.start):
        require(pos <= t.start and t.start + t.size <= size, f"overlapping or incomplete target tensor: {t.name}")
        if t.start > pos:
            yield pos, t.start - pos, None
        yield t.start, t.size, t
        pos = t.start + t.size
    if pos < size:
        yield pos, size - pos, None


@contextmanager
def publish(path):
    require(not os.path.lexists(path), f"output exists: {path}")
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            yield out
            out.flush()
            os.fsync(out.fileno())
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


def identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def verify_open_bases(files, identities):
    require(all(identity(os.fstat(f.fileno())) == identities[i] for i, f in files.items()),
            "a base shard changed during processing")


def create(args, files, table, manifest_hash, identities, stdout):
    before = args.target.stat()
    size = before.st_size
    cap = min(size - 1, int(args.max_gib * GIB))
    require(cap > 0, "delta cap must be positive")
    h, archive_hash, counts = hashlib.sha256(), hashlib.sha256(), Counter()
    codec = Zstd(args.level)
    written = 0
    sink = nullcontext(stdout) if args.output == Path("-") else publish(args.output)
    with args.target.open("rb") as src, sink as out:
        def emit(data):
            nonlocal written
            require(written + len(data) <= cap, f"delta exceeds cap of {cap} bytes")
            out.write(data)
            archive_hash.update(data)
            written += len(data)
        emit(MAGIC + manifest_hash + struct.pack("<Q", size))
        for start, length, t in segments(index(args.target), size):
            b = matching(t, table) if t else None
            for offset in range(0, length, BLOCK):
                n = min(BLOCK, length - offset)
                raw = read_at(src, start + offset, n)
                h.update(raw)
                base = read_at(files[b.shard], b.start + offset, n) if b else None
                kind, payload = encode(raw, base, codec, args.bitplanes)
                use = kind in (COPY, ZXOR, BITXOR)
                require(written + REC.size + len(payload) + REC.size + 32 <= cap,
                        f"delta exceeds cap of {cap} bytes")
                emit(REC.pack(kind, b.shard if use else NONREF, b.start + offset if use else 0, n, len(payload)))
                emit(payload)
                counts[KINDS[kind]] += n
        verify_open_bases(files, identities)
        require(identity(before) == identity(args.target.stat()), "target changed during delta creation")
        if args.expect_sha256:
            require(h.hexdigest() == args.expect_sha256, "target SHA-256 differs from independent receipt")
        emit(REC.pack(END, NONREF, 0, 0, 32) + h.digest())
        out.flush()
    return dict(target_bytes=size, target_sha256=h.hexdigest(), delta_bytes=written, delta_sha256=archive_hash.hexdigest(),
                target_bytes_by_encoding=counts)


def exact(f, n):
    result = f.read(n)
    require(len(result) == n, "truncated delta")
    return result


def restore(args, files, identities, src):
    codec, h = Zstd(1), hashlib.sha256()
    size, = struct.unpack("<Q", exact(src, 8))
    # No output path means verify/decode the complete stream without storing another GGUF.
    with (publish(args.output) if args.output else open(os.devnull, "wb")) as out:
        written = 0
        while True:
            kind, idx, off, n, stored = REC.unpack(exact(src, REC.size))
            if kind == END:
                require((idx, off, n, stored) == (NONREF, 0, 0, 32) and written == size
                        and h.digest() == exact(src, 32) and src.read(1) == b"", "invalid footer or target SHA-256")
                if args.expect_sha256:
                    require(h.hexdigest() == args.expect_sha256, "target differs from independent SHA-256")
                verify_open_bases(files, identities)
                break
            require(0 < n <= BLOCK and written + n <= size and stored <= n, "invalid record length")
            ref = kind in (COPY, ZXOR, BITXOR)
            require(idx in files if ref else (idx == NONREF and off == 0), "invalid base reference")
            data = exact(src, stored)
            if kind in (COPY, ZERO):
                require(not stored, "invalid reference payload")
                raw = read_at(files[idx], off, n) if ref else bytes(n)
            elif kind == RAW:
                require(stored == n, "invalid literal length")
                raw = data
            elif kind in (ZRAW, ZXOR, BITXOR):
                raw = codec.decompress(data, n)
                if kind == BITXOR:
                    require(n % 8 == 0, "invalid bitplane length")
                    raw = planes(raw, True)
                if ref:
                    raw = xor(raw, read_at(files[idx], off, n))
            else:
                raise ValueError(f"unknown record type {kind}")
            out.write(raw)
            h.update(raw)
            written += n
    return dict(verified_bytes=size, target_sha256=h.hexdigest(), restored=bool(args.output))


def native_layout(args):
    # Use the producer's own GGUF metadata serializer and quantization policy, including imatrix overrides.
    with tempfile.TemporaryDirectory(prefix="inkling-delta-layout-") as d:
        zero = Path(d) / "empty-input.gguf"
        opts = [str(args.quantizer.resolve()), "--tensor-stdin"]
        if args.imatrix:
            opts += ["--imatrix", str(args.imatrix.resolve())]
        opts += [str(args.descriptor.resolve()), str(zero), "Q3_K_M", "1"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        def run(cmd):
            return subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, env=env)
        plan = run(opts[:1] + ["--dry-run"] + opts[1:])
        require(plan.returncode == 0, "native dry-run failed: " + plan.stdout[-1024:].decode(errors="replace"))
        probe = run(opts)
        require(probe.returncode != 0 and b"at offset 0" in probe.stdout and b"short read of " in probe.stdout
                and zero.is_file(), "empty input did not stop at first tensor; cannot measure header")
        reserve = zero.stat().st_size
        with zero.open("rb") as f:
            require(reserve > 0 and all(b.count(0) == len(b) for b in iter(lambda: f.read(BLOCK), b"")),
                    "unexpected bytes in native header reservation")
    align = gguf.GGUFReader(args.descriptor).alignment
    tensor = re.compile(r"^\[\s*(\d+)/\s*(\d+)\]\s+(\S+)\s+-\s+\[([^\]]+)\], type =\s*(\w+),\s*(.+)$")
    items, total, pos = [], 0, reserve
    for line in plan.stdout.decode(errors="replace").splitlines():
        m = tensor.match(line)
        if not m:
            continue
        i, total = int(m[1]), int(m[2])
        require(i == len(items) + 1, "nonsequential native layout")
        shape = dims(int(x) for x in m[4].split(","))
        selected = re.search(r"\((\w+)\)\s*$", m[6])
        qtype = (selected[1] if selected else m[5]).upper()
        block, width = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[qtype]]
        require(shape[0] % block == 0, f"unaligned quantization: {m[3]}")
        size = math.prod(shape) // block * width
        items.append(Tensor(m[3], shape, qtype, pos, size))
        pos += (size + align - 1) // align * align
    require(items and len(items) == total, "incomplete native tensor plan")
    return items, pos, reserve


def estimate(args, files, table, spec):
    with args.target.open("rb") as f:
        snapshot = os.fstat(f.fileno()).st_size
        magic = f.read(4)
        if magic == b"GGUF":
            tensors, final = index(args.target), snapshot
            reserve = min(t.start for t in tensors)
        else:
            require(magic == bytes(4) and args.descriptor and args.quantizer,
                    "growing pack needs --descriptor and --quantizer with the producer's --imatrix")
            tensors, final, reserve = native_layout(args)
            require(snapshot >= reserve and snapshot <= final, "unexpected growing target length")
            require(read_at(f, reserve - 1, 1) == bytes(1), "invalid header boundary")
        groups, bands = {}, {}
        found, total_payload, record_count = 0, 0, 0
        all_known = args.base_index is not None or len(files) == len(spec)
        limit = args.sample_kib * 1024
        codec, rng = Zstd(args.level), random.Random(args.seed)
        for t in tensors:
            stock = table.get(t.name)
            b = matching(t, table)
            role = re.sub(r"^blk\.\d+\.", "blk.*.", t.name)
            src = stock.qtype if stock else ("absent" if all_known else "unknown")
            layer = re.match(r"^blk\.(\d+)\.", t.name)
            band = "global"
            if layer:
                lower = int(layer[1]) // args.layer_band * args.layer_band
                band = f"layers-{lower}-{lower + args.layer_band - 1}"
            shape = "x".join(map(str, t.shape))
            if stock and not b and stock.shape != t.shape:
                src += "-different-shape"
            family = f"{role}/{shape}/{src}-to-{t.qtype}"
            group = groups.setdefault(family, Counter())
            band_group = bands.setdefault(family + "/" + band, Counter())
            total_payload += t.size
            record_count += (t.size + BLOCK - 1) // BLOCK
            trial = Counter(target_bytes=t.size, target_tensors=1)
            available = (b is None or b.shard in files) and (stock is not None or all_known)
            if available and t.start + t.size <= snapshot:
                trial.update(eligible_bytes=t.size, eligible_tensors=1, base_matched_tensors=int(bool(b)))
                found += 1
                length = min(limit, t.size)
                n_samples = args.expert_samples if role.endswith("_exps.weight") and args.expert_samples else args.samples
                spans = min(n_samples, t.size // length)
                for j in range(spans):
                    low = j * t.size // spans
                    last = (j + 1) * t.size // spans - length
                    offset = rng.randint(low, last)
                    raw = read_at(f, t.start + offset, length)
                    base = read_at(files[b.shard], b.start + offset, length) if b else None
                    kind, encoded = encode(raw, base, codec, args.bitplanes)
                    trial["sample_bytes"] += length
                    trial["stored_sample_bytes"] += len(encoded)
                    trial["plain_zstd_sample_bytes"] += min(len(raw), len(codec.compress(raw)))
                    trial[KINDS[kind] + "_samples"] += 1
            group.update(trial)
            band_group.update(trial)
        # Header and all target gaps are charged as literals. This also avoids extrapolating from a live zero header.
        structure = final - total_payload
        overhead = 48 + REC.size + 32 + (record_count + len(tensors) + 1 + (structure + BLOCK - 1) // BLOCK) * REC.size
        def project(buckets):
            delta, plain, unknown = structure + overhead, structure + overhead, 0
            for g in buckets.values():
                if g["sample_bytes"]:
                    g["projected_delta_bytes"] = round(g["target_bytes"] * g["stored_sample_bytes"] / g["sample_bytes"])
                    delta += g["projected_delta_bytes"]
                    plain += round(g["target_bytes"] * g["plain_zstd_sample_bytes"] / g["sample_bytes"])
                else:
                    unknown += g["target_bytes"]
            return delta + unknown, plain + unknown, unknown

        predicted, plain, unknown = project(groups)
        depth_estimate, _, depth_unknown = project(bands)
        return dict(status="sampled projection; late layers may differ", verified_all_base_sha256=args.verify_bases
                    and len(files) == len(spec), available_base_shards=len(files), total_base_shards=len(spec),
                    snapshot_bytes=snapshot, final_bytes=final, header_bytes=reserve, sampled_tensors=found,
                    tensor_bytes_with_completed_samples=sum(g["eligible_bytes"] for g in groups.values()),
                    total_tensor_bytes=total_payload, total_tensors=len(tensors), sample_bytes=sum(g["sample_bytes"] for g in groups.values()),
                    complete_base_tensor_catalog=all_known, unsampled_family_bytes=unknown,
                    projected_delta_bytes_with_unsampled_families_raw=predicted,
                    layer_band_width=args.layer_band, unsampled_depth_family_bytes=depth_unknown,
                    projected_delta_bytes_with_unsampled_depth_families_raw=depth_estimate,
                    projected_plain_zstd_bytes_with_unsampled_families_raw=plain,
                    groups=groups, depth_groups=bands)


def receive(args):
    _, manifest_hash = manifest(args.manifest)
    receipt = json.loads(args.receipt.read_text())
    require(not os.path.lexists(args.output) and not args.partial.is_symlink(), "output exists or incoming file is a symlink")
    with args.partial.open("rb") as f:
        before = identity(os.fstat(f.fileno()))
        require(before[2] == receipt["delta_bytes"], "transferred length differs from sender")
        require(exact(f, 40) == MAGIC + manifest_hash, "wrong archive or base manifest")
        size, = struct.unpack("<Q", exact(f, 8))
        require(size == receipt["target_bytes"], "target length differs from sender")
        f.seek(-REC.size - 32, os.SEEK_END)
        require(exact(f, REC.size) == REC.pack(END, NONREF, 0, 0, 32), "invalid or missing footer")
        target_hash = exact(f, 32).hex()
        require(target_hash == receipt["target_sha256"] and (not args.expect_sha256 or target_hash == args.expect_sha256),
                "target SHA-256 footer mismatch")
        f.seek(0)
        require(hashlib.file_digest(f, "sha256").hexdigest() == receipt["delta_sha256"], "transferred SHA-256 mismatch")
        require(identity(args.partial.stat()) == before, "partial changed during validation")
        os.fsync(f.fileno())
        os.link(args.partial, args.output)
    args.partial.unlink()
    return dict(delta_bytes=before[2], delta_sha256=receipt["delta_sha256"], target_sha256=target_hash,
                verified="transfer length, archive SHA-256, pinned manifest identity and footer")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    recv = sub.add_parser("receive", help="verify a successful sender receipt and publish a local partial atomically")
    for arg in ("partial", "output", "receipt", "manifest"):
        recv.add_argument("--" + arg, type=Path, required=True)
    recv.add_argument("--expect-sha256")
    for command in ("estimate", "create", "restore"):
        s = sub.add_parser(command)
        s.add_argument("--manifest", type=Path, required=True)
        s.add_argument("--base-dir", type=Path, required=True)
        if command in ("estimate", "create"):
            s.add_argument("--target", type=Path, required=True)
            s.add_argument("--level", type=int, default=3, choices=range(1, 20), metavar="1..19")
            s.add_argument("--bitplanes", action="store_true", help="also try reordering the XOR into eight bitplanes")
        if command in ("create", "restore"):
            s.add_argument("--expect-sha256", help="independent digest for the complete original target GGUF")
            s.add_argument("--output", type=Path, required=command == "create",
                           help="new file; create --output - streams to stdout; omit on restore to verify in memory")
        if command == "create":
            s.add_argument("--max-gib", type=float, required=True, help="abort and clean up if the delta exceeds this size")
        if command == "restore":
            s.add_argument("--delta", type=Path, required=True, help="archive file or - for stdin")
        if command == "estimate":
            s.add_argument("--base-index", type=Path, help="pinned complete stock header catalog for missing shards")
            s.add_argument("--layer-band", type=int, default=8, help="also project grouped by adjacent layer depth")
            s.add_argument("--expert-samples", type=int, help="override --samples on fused expert tensors, up to 384")
            s.add_argument("--verify-bases", action="store_true", help="SHA-256 every available complete base shard")
            s.add_argument("--descriptor", type=Path, help="exact sparse producer input for a growing target")
            s.add_argument("--quantizer", type=Path, help="same llama-quantize as the live producer")
            s.add_argument("--imatrix", type=Path, help="same file path as the live producer's --imatrix")
            s.add_argument("--sample-kib", type=int, default=256)
            s.add_argument("--samples", type=int, default=3, help="stratified random blocks per completed tensor")
            s.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()
    if args.action == "receive":
        print(json.dumps(receive(args), indent=2))
        return
    if args.action == "estimate":
        require(1 <= args.sample_kib <= BLOCK // 1024 and 1 <= args.samples <= 384
                and (args.expert_samples is None or 1 <= args.expert_samples <= 384) and args.layer_band > 0,
                "invalid sampling budget")
    spec, manifest_hash = manifest(args.manifest)
    streaming = args.action == "create" and args.output == Path("-")
    if getattr(args, "output", None) and not streaming:
        require(args.output != Path("-"), "restore verification reads --delta - and needs no --output")
        require(not os.path.lexists(args.output), f"output exists: {args.output}")
    stdout = sys.stdout.buffer
    with ExitStack() as stack:
        if streaming:
            stack.enter_context(redirect_stdout(sys.stderr))
        if args.action == "restore":
            src = sys.stdin.buffer if args.delta == Path("-") else stack.enter_context(args.delta.open("rb"))
            require(exact(src, 40) == MAGIC + manifest_hash, "wrong archive or base manifest")
        files, table = open_bases(stack, spec, args.base_dir, args.action != "estimate" or args.verify_bases,
                                  args.action == "estimate")
        identities = {i: identity(os.fstat(f.fileno())) for i, f in files.items()}
        if args.action == "estimate":
            if args.base_index:
                table = add_catalog(args.base_index, manifest_hash, spec, files, table)
            require(files or args.base_index, "no complete shards or pinned catalog for sampling")
            result = estimate(args, files, table, spec)
        elif args.action == "create":
            result = create(args, files, table, manifest_hash, identities, stdout)
        else:
            result = restore(args, files, identities, src)
    print(json.dumps(result, sort_keys=True, indent=2), file=sys.stderr if streaming else sys.stdout)


if __name__ == "__main__":
    main()
