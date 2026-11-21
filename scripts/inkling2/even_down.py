#!/usr/bin/env python3
"""Rewrite a pack's routed down projections from Q4_K and Q5_K to Q4_1 and Q5_1, value for value.

Under -sm tensor four cards share each expert's hidden width of 2,304, which is the down projection's K. In K-quant
blocks of 256 the cards get 512/512/512/768 of it and every layer waits for the card with 768. In blocks of 32 each
card gets 576. Each 32-wide sub-block of a Q4_K or Q5_K block becomes one Q4_1 or Q5_1 block with the same quants:
its scale d*sc and its offset -dmin*m, exact in f32, are rounded once to f16. The down projections grow by 1/9.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(1, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402
from gguf.quants import Q4_K  # noqa: E402

T = gguf.GGMLQuantizationType
TARGET = {T.Q4_K: T.Q4_1, T.Q5_K: T.Q5_1}
EXPERTS_PER_CHUNK = 16  # rows are converted a slab at a time: the host holds the new tensor and one slab of work


def split_k(blocks):
    """d*sc and -dmin*m of each sub-block of K-quant blocks [n, bytes], rounded to f16 as [n, 8, 2] bytes each."""
    n = blocks.shape[0]
    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32)
    dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32)
    sc, mn = Q4_K.get_scale_min(blocks[:, 4:16])
    scale = (d * sc.astype(np.float32)).astype(np.float16)
    offset = (-(dmin * mn.astype(np.float32))).astype(np.float16)
    return scale.view(np.uint8).reshape(n, 8, 2), offset.view(np.uint8).reshape(n, 8, 2)


def nibbles(q):
    """32 quants of a sub-block [..., 32] as Q4_1/Q5_1 pack their low nibbles: value j low, value j + 16 high."""
    return (q[..., :16] & 0x0F) | ((q[..., 16:] & 0x0F) << 4)


def q4_k_to_q4_1(blocks):
    n = blocks.shape[0]
    scale, offset = split_k(blocks)
    qs = blocks[:, 16:144].reshape(n, 4, 1, 32) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
    out = np.empty((n, 8, 20), dtype=np.uint8)
    out[:, :, 0:2] = scale
    out[:, :, 2:4] = offset
    out[:, :, 4:20] = nibbles((qs & 0x0F).reshape(n, 8, 32))
    return out.reshape(n, 160)


def q5_k_to_q5_1(blocks):
    n = blocks.shape[0]
    scale, offset = split_k(blocks)
    qh = blocks[:, 16:48].reshape(n, 1, 32)
    ql = blocks[:, 48:176].reshape(n, 4, 1, 32) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
    high = (qh >> np.arange(8, dtype=np.uint8).reshape(1, 8, 1)) & 1  # sub-block s takes bit s of qh
    q = (ql & 0x0F).reshape(n, 8, 32) | (high << 4)
    out = np.empty((n, 8, 24), dtype=np.uint8)
    out[:, :, 0:2] = scale
    out[:, :, 2:4] = offset
    out[:, :, 4:8] = np.packbits(high.astype(np.uint8), axis=-1, bitorder="little")
    out[:, :, 8:24] = nibbles(q)
    return out.reshape(n, 192)


CONVERT = {T.Q4_K: q4_k_to_q4_1, T.Q5_K: q5_k_to_q5_1}


def converts(tensor):
    return tensor.name.endswith(".ffn_down_exps.weight") and tensor.tensor_type in CONVERT


def convert_tensor(tensor):
    """the whole new tensor, its experts converted a slab at a time"""
    src_size = gguf.GGML_QUANT_SIZES[tensor.tensor_type][1]
    dst_size = gguf.GGML_QUANT_SIZES[TARGET[tensor.tensor_type]][1] * 8
    data = tensor.data.reshape(-1, tensor.data.shape[-2], tensor.data.shape[-1])  # [experts, rows, row bytes]
    n_expert, n_rows, row_bytes = data.shape
    blocks_per_row = row_bytes // src_size
    out = np.empty((n_expert, n_rows, blocks_per_row * dst_size), dtype=np.uint8)
    for e in range(0, n_expert, EXPERTS_PER_CHUNK):
        slab = np.ascontiguousarray(data[e:e + EXPERTS_PER_CHUNK]).reshape(-1, src_size)
        out[e:e + EXPERTS_PER_CHUNK] = CONVERT[tensor.tensor_type](slab).reshape(-1, n_rows, blocks_per_row * dst_size)
    return out.reshape(tensor.data.shape[:-1] + (blocks_per_row * dst_size,))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="a pack GGUF (one file)")
    parser.add_argument("output", type=Path, help="the new GGUF path")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new file")

    reader = gguf.GGUFReader(args.source)
    if not any(converts(tensor) for tensor in reader.tensors):
        downs = sorted({t.tensor_type.name for t in reader.tensors if t.name.endswith(".ffn_down_exps.weight")})
        sys.exit(f"{args.source}: no ffn_down_exps tensor is Q4_K or Q5_K (down types: {', '.join(downs) or 'none'}), "
                 "so there is nothing to convert")
    arch =reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    writer = gguf.GGUFWriter(args.output, arch=arch, endianess=reader.endianess)
    alignment = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment.contents()
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        sub_type = field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), field.types[0], sub_type=sub_type)

    grown = 0
    for tensor in reader.tensors:
        shape, nbytes, qtype = tensor.data.shape, tensor.n_bytes, tensor.tensor_type
        if converts(tensor):
            src_size = gguf.GGML_QUANT_SIZES[qtype][1]
            dst_size = gguf.GGML_QUANT_SIZES[TARGET[qtype]][1] * 8
            shape = shape[:-1] + (shape[-1] // src_size * dst_size,)
            nbytes = nbytes // src_size * dst_size
            grown += nbytes - tensor.n_bytes
            qtype = TARGET[qtype]
        writer.add_tensor_info(tensor.name, shape, tensor.data.dtype, nbytes, qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        if converts(tensor):
            print(f"{tensor.name}: {tensor.tensor_type.name} -> {TARGET[tensor.tensor_type].name}", file=sys.stderr, flush=True)
            writer.write_tensor_data(convert_tensor(tensor), tensor_endianess=reader.endianess)
        else:
            writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    writer.close()
    print(f"the down projections grew by {grown / (1 << 30):.2f} GiB", file=sys.stderr)


if __name__ == "__main__":
    main()
