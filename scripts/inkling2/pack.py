#!/usr/bin/env python3
"""Stream the pinned Inkling checkpoint to llama-quantize without a materialized BF16 model."""
import argparse
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "gguf-py"), str(ROOT)]
import gguf
import torch
from conversion.inkling2 import Inkling2Model

REVISION = "e4853745aa9cb4587e8959d3c80b92e1bef99a4a"
REPO = "benthecarman/Inkling-2-Small-abliterated"
STOCK_IMATRIX = "7e7897f261bbf80526484e41aa1e1d3627ddb1fdbb7f0e9f099b03b0852d4810"


def file_checksum(path, git_blob=False):
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode()) if git_blob else hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(source):
    manifest = Path(__file__).with_name("abliterated-e4853745.checksums")

    def verify(line):
        expected, size, name = line.split(" ", 2)
        path = source / name
        if path.stat().st_size != int(size):
            raise ValueError(f"wrong source length: {path}")
        if file_checksum(path, git_blob=len(expected) == 40) != expected:
            raise ValueError(f"source checksum mismatch: {path}")
        return name

    with ThreadPoolExecutor(max_workers=4) as pool:
        for name in pool.map(verify, manifest.read_text().splitlines()):
            print(f"verified {name}", file=sys.stderr, flush=True)


def check_imatrix(path, tensors):
    matrix = gguf.GGUFReader(path)
    entries = {entry.name: entry for entry in matrix.tensors}
    unobserved = 0
    for name, info in tensors.items():
        if not name.endswith((".ffn_gate_up_exps.weight", ".ffn_down_exps.weight")):
            continue
        experts, _, width = info.shape
        sums = entries.get(name + ".in_sum2")
        counts = entries.get(name + ".counts")
        if sums is None or tuple(sums.shape) != (width, experts):
            raise ValueError(f"missing or incompatible expert importance weights for {name}")
        if counts is None or counts.data.size != experts or (counts.data < 0).any():
            raise ValueError(f"invalid calibration counts for {name}")
        unobserved += int((counts.data == 0).sum())
    return unobserved


def write_descriptors(writer, path):
    writer.write_header_to_file(path)
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    handle, = writer.fout
    tensors, = writer.tensors
    align = writer.data_alignment
    weights_size = sum(writer.ggml_pad(info.nbytes, align) for info in tensors.values())
    handle.truncate(writer.ggml_pad(handle.tell(), align) + weights_size)
    writer.close()


def stream_quantize(command, tensors, producer_workers=1):
    pending = dict(tensors)
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0) as process:
        try:
            for line in process.stdout:
                if not line.startswith(b"LLAMA_TENSOR\t"):
                    sys.stderr.buffer.write(line)
                    continue
                name = line.rstrip(b"\r\n").split(b"\t", 1)[1].decode()
                info = pending.pop(name)
                if info.tensor is None or info.tensor.nbytes != info.nbytes:
                    raise ValueError(f"missing tensor data for {name}")
                if isinstance(info.tensor, gguf.LazyChunkedTensor) and producer_workers > 1:
                    info.tensor.tofile(process.stdin, workers=producer_workers)
                else:
                    info.tensor.tofile(process.stdin)
                info.tensor = None
            process.stdin.close()
            if process.wait():
                raise RuntimeError(f"llama-quantize exited with status {process.returncode}")
            if pending:
                raise ValueError(f"quantizer did not request {len(pending)} tensors")
        finally:
            if process.poll() is None:
                process.kill()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="snapshot directory, including all 24 pinned safetensors shards")
    parser.add_argument("output", type=Path, help="new Q3_K_M GGUF path")
    parser.add_argument("--quantizer", required=True, type=Path, help="this fork's llama-quantize executable")
    parser.add_argument("--imatrix", required=True, type=Path)
    parser.add_argument("--calibration", required=True, help="use stock-bartowski or describe the derivative model and calibration corpus")
    parser.add_argument("--threads", type=int, default=0, help="native quantizer threads")
    parser.add_argument("--producer-workers", type=int, default=1,
                        help="number of packed expert row chunks to prepare concurrently (default: 1)")
    parser.add_argument("--buffer-mib", type=int, default=256, help="maximum native row-slab workspace in MiB")
    args = parser.parse_args()
    if args.output.exists() or args.threads < 0 or args.buffer_mib <= 0 or args.producer_workers < 1:
        parser.error("output must be a new file, threads nonnegative and buffer size and producer worker count positive")
    digest = file_checksum(args.imatrix)
    if (args.calibration == "stock-bartowski") != (digest == STOCK_IMATRIX):
        parser.error("the stock-bartowski label requires the pinned matrix and that matrix requires its stock label")

    verify_source(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output.parent).free < 265 * (1 << 30):
        parser.error("the output filesystem needs at least 265 GiB free")
    with tempfile.TemporaryDirectory(dir=args.output.parent, prefix=".inkling-pack-") as tmp, torch.inference_mode():
        descriptor = Path(tmp) / "descriptors.gguf"
        pack = Path(tmp) / "packed.gguf"
        model = Inkling2Model(args.source, gguf.LlamaFileType.MOSTLY_BF16, descriptor)
        model.prepare_tensors()
        model.prepare_metadata(vocab_only=False)
        writer = model.gguf_writer
        tensors, = writer.tensors
        unobserved = check_imatrix(args.imatrix, tensors)
        print(f"Expert matrix rows using unit importance: {unobserved}", file=sys.stderr, flush=True)
        writer.add_uint32("quantize.imatrix.unobserved_expert_rows", unobserved)
        writer.add_string(gguf.Keys.General.SOURCE_REPO_URL, f"https://huggingface.co/{REPO}/tree/{REVISION}")
        writer.add_string("quantize.imatrix.source", args.calibration)
        writer.add_string("quantize.imatrix.sha256", digest)
        write_descriptors(writer, descriptor)
        command = [str(args.quantizer.resolve()), "--tensor-stdin", "--imatrix", str(args.imatrix),
                   "--max-buffer-size", str(args.buffer_mib), str(descriptor), str(pack), "Q3_K_M", str(args.threads)]
        stream_quantize(command, tensors, args.producer_workers)
        os.link(pack, args.output)


if __name__ == "__main__":
    main()
