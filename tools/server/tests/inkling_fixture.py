"""The synthetic Inkling 2 model, a matching vision projector and a test image, for the inkling server tests."""
import glob
import os
import struct
import subprocess
import zlib

import numpy as np

TEMPLATE = "../../../models/templates/thinkingmachines-Inkling-2-Small.jinja"


def bin_dir() -> str:
    server = os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server")
    return os.path.dirname(os.path.abspath(server))


def write_model(folder: str, stdev: float | None = None) -> str:
    """The inkling2 model test-llama-archs builds from its fixture, from the same build as the server."""
    cmd = [os.path.join(bin_dir(), "test-llama-archs"), "-a", "inkling2", "-o", folder]
    if stdev is not None:
        cmd += ["-d", str(stdev)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    models = sorted(glob.glob(os.path.join(folder, "inkling2*.gguf")))
    assert models, f"test-llama-archs wrote no inkling2 model in {folder}"
    return models[0]


def read_u32(path: str, key: str) -> int:
    """One integer value of a GGUF's key-value section."""
    sizes = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    with open(path, "rb") as f:
        def rd(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]

        def value(t):
            if t == 8:
                return f.read(rd("Q"))
            if t == 9:
                et, n = rd("I"), rd("Q")
                return [value(et) for _ in range(n)]
            return rd(sizes[t])
        assert f.read(4) == b"GGUF"
        rd("I")
        rd("Q")
        for _ in range(rd("Q")):
            name = f.read(rd("Q")).decode()
            v = value(rd("I"))
            if name == key:
                return int(v)
    raise KeyError(key)


def _gguf_value(v) -> bytes:
    if isinstance(v, bool):
        return struct.pack("<I?", 7, v)
    if isinstance(v, int):
        return struct.pack("<II", 4, v)
    if isinstance(v, float):
        return struct.pack("<If", 6, v)
    if isinstance(v, str):
        return struct.pack("<IQ", 8, len(v)) + v.encode()
    assert all(isinstance(x, float) for x in v)
    return struct.pack("<IIQ", 9, 6, len(v)) + struct.pack(f"<{len(v)}f", *v)


def write_gguf(path: str, kv: dict, tensors: dict) -> None:
    """A GGUF v3 file of F32 tensors (the gguf package needs PyYAML, which the test venv does not carry)."""
    align = 32
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)))
    for key, v in kv.items():
        out += struct.pack("<Q", len(key)) + key.encode() + _gguf_value(v)
    offset, blobs = 0, []
    for name, t in tensors.items():
        t = np.ascontiguousarray(t, dtype=np.float32)
        out += struct.pack("<Q", len(name)) + name.encode() + struct.pack("<I", t.ndim)
        out += struct.pack(f"<{t.ndim}Q", *reversed(t.shape)) + struct.pack("<IQ", 0, offset)
        blob = t.tobytes() + b"\0" * (-t.nbytes % align)
        blobs.append(blob)
        offset += len(blob)
    out += b"\0" * (-len(out) % align)
    with open(path, "wb") as f:
        f.write(bytes(out))
        for blob in blobs:
            f.write(blob)


def write_mmproj(path: str, text_embd: int, seed: int = 1) -> int:
    """An inkling2v projector with one ViT block and the layout conversion/inkling2.py writes, projecting into a
    text model text_embd wide; returns the image tokens a 16 x 16 image takes (4 x 4 patches merged 2 x 2)."""
    rng = np.random.default_rng(seed)
    n_embd, n_ff, hidden, patch, grid, merge = 32, 64, 64, 4, 4, 2

    def w(*shape):
        return rng.normal(0.0, 0.1, shape)
    tensors = {
        "v.patch_embd.weight": w(n_embd, 3, patch, patch), "v.patch_embd.bias": w(n_embd),
        "v.position_embd.weight": w(grid * grid, n_embd),
        "v.class_embd": w(n_embd), "v.class_position_embd.weight": w(n_embd),
        "v.pre_ln.weight": np.ones(n_embd), "v.pre_ln.bias": np.zeros(n_embd),
        "mm.1.weight": w(hidden, n_embd * merge * merge), "mm.1.bias": w(hidden),
        "mm.2.weight": w(text_embd, hidden), "mm.2.bias": w(text_embd),
    }
    for part in ("q", "k", "v", "out"):
        tensors[f"v.blk.0.attn_{part}.weight"] = w(n_embd, n_embd)
        tensors[f"v.blk.0.attn_{part}.bias"] = w(n_embd)
    for ln in ("ln1", "ln2"):
        tensors[f"v.blk.0.{ln}.weight"] = np.ones(n_embd)
        tensors[f"v.blk.0.{ln}.bias"] = np.zeros(n_embd)
    tensors.update({"v.blk.0.ffn_up.weight": w(n_ff, n_embd), "v.blk.0.ffn_up.bias": w(n_ff),
                    "v.blk.0.ffn_down.weight": w(n_embd, n_ff), "v.blk.0.ffn_down.bias": w(n_embd)})
    kv = {
        "general.architecture": "clip", "clip.has_vision_encoder": True, "clip.projector_type": "inkling2v",
        "clip.vision.embedding_length": n_embd, "clip.vision.feed_forward_length": n_ff,
        "clip.vision.block_count": 1, "clip.vision.attention.head_count": 2,
        "clip.vision.attention.layer_norm_epsilon": 1e-6, "clip.vision.projection_dim": text_embd,
        "clip.vision.image_size": grid * patch, "clip.vision.patch_size": patch,
        "clip.vision.image_mean": [0.5, 0.5, 0.5], "clip.vision.image_std": [0.5, 0.5, 0.5],
        "clip.vision.spatial_merge_size": merge,
    }
    write_gguf(path, kv, tensors)
    return (grid // merge) ** 2


def png(width: int = 16, height: int = 16, seed: int = 0) -> bytes:
    """An RGB PNG of random pixels."""
    pixels = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    raw = b"".join(b"\0" + row.tobytes() for row in pixels)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
