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


def write_gguf(path: str, kv: dict, tensors: dict, raw_kv: list | None = None) -> None:
    """A GGUF v3 file of F32 tensors (the gguf package needs PyYAML, which the test venv does not carry). raw_kv: key
    and value entries as another GGUF holds them (read_gguf), and a bytes value in kv is written as its typed value."""
    align = 32
    raw_kv = raw_kv or []
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv) + len(raw_kv)))
    for key, v in kv.items():
        out += struct.pack("<Q", len(key)) + key.encode() + (v if isinstance(v, bytes) else _gguf_value(v))
    for entry in raw_kv:
        out += entry
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


def read_gguf(path: str) -> tuple[dict, dict]:
    """A GGUF's key-value entries as the file holds them (key, type and value bytes, by key) and its F32 tensors."""
    sizes = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    with open(path, "rb") as f:
        data = f.read()
    pos = 4

    def rd(fmt):
        nonlocal pos
        v = struct.unpack_from("<" + fmt, data, pos)[0]
        pos += struct.calcsize("<" + fmt)
        return v

    def skip(t):
        nonlocal pos
        if t == 8:
            n = rd("Q")
            pos += n
        elif t == 9:
            et, n = rd("I"), rd("Q")
            for _ in range(n):
                skip(et)
        else:
            pos += struct.calcsize("<" + sizes[t])
    assert data[:4] == b"GGUF"
    rd("I")
    n_tensors, n_kv = rd("Q"), rd("Q")
    kv, align = {}, 32
    for _ in range(n_kv):
        start = pos
        n = rd("Q")
        name = data[pos:pos + n].decode()
        pos += n
        t = rd("I")
        if name == "general.alignment":
            align = struct.unpack_from("<I", data, pos)[0]
        skip(t)
        kv[name] = data[start:pos]
    infos = []
    for _ in range(n_tensors):
        n = rd("Q")
        name = data[pos:pos + n].decode()
        pos += n
        shape = [rd("Q") for _ in range(rd("I"))]
        infos.append((name, shape, rd("I"), rd("Q")))
    pos += -pos % align
    tensors = {}
    for name, shape, kind, offset in infos:
        if kind == 0:
            tensors[name] = np.frombuffer(data, np.float32, int(np.prod(shape)), pos + offset).reshape(shape[::-1])
    return kv, tensors


def write_dspark_draft(path: str, target: str, seed: int = 4558, *, tp4: bool = False) -> None:
    """An Inkling DSpark draft for the inkling2 target, laid out as the drafter kit's rehearsal fixture: two layers,
    a Markov head of rank 8 and a block of 8, with the target's tokenizer, embeddings and LM head.
    The TP4 fixture has at least one complete KV group and 32 FFN channels per rank."""
    kv_t, t_t = read_gguf(target)
    width = read_u32(target, "inkling2.embedding_length")
    layers = read_u32(target, "inkling2.block_count")
    padded = read_u32(target, "inkling2.vocab_size")
    vocab = read_u32(target, "inkling2.vocab_size_unpadded")
    heads, kv_heads, dim, ff, rank = (8, 4, 64, 128, 8) if tp4 else (4, 2, 32, 64, 8)
    kv = {
        "general.architecture": "dflash", "dflash.context_length": 512, "dflash.embedding_length": width,
        "dflash.feed_forward_length": ff, "dflash.block_count": 2, "dflash.attention.head_count": heads,
        "dflash.attention.head_count_kv": kv_heads, "dflash.attention.key_length": dim,
        "dflash.attention.value_length": dim, "dflash.rope.dimension_count": dim, "dflash.rope.freq_base": 10000.0,
        "dflash.attention.layer_norm_rms_epsilon": 1e-5, "dflash.vocab_size": padded,
        "dflash.vocab_size_unpadded": vocab, "dflash.block_size": 8,
        "dflash.target_layers": struct.pack("<IIQ2i", 9, 5, 2, 0, layers),
        "dflash.dspark.markov_rank": rank, "dflash.sample_from_anchor": False,
        "dflash.has_confidence_head": False, "dflash.attention.causal": False, "dflash.attention.sliding_window": 32,
        "dflash.attention.sliding_window_pattern": struct.pack("<IIQ2?", 9, 7, 2, True, False),
        "dflash.final_logit_softcapping": 30.0, "tokenizer.ggml.mask_token_id": 0,
    }
    raw = [v for k, v in kv_t.items() if k.startswith("tokenizer.") and k != "tokenizer.ggml.mask_token_id"]
    rng = np.random.default_rng(seed)

    def w(*shape):
        return rng.normal(0.0, 0.035, shape)
    tensors = {name: t_t[name] for name in ("token_embd.weight", "token_embd_norm.weight", "output.weight") if name in t_t}
    tensors.update({
        "dspark_fuse.weight": rng.normal(0.0, 0.05, (3, 2)), "dspark_norm.weight": np.ones(width),
        "output_norm.weight": np.ones(width), "mask_embd.weight": w(width), "markov_w1.weight": w(padded, rank),
        "joint_proj.weight": w(3*rank, 2*(rank + width)), "joint_proj.bias": w(3*rank),
        "dspark_hidden_out.weight": w(width, rank), "dspark_hidden_out.bias": w(width),
    })
    for layer in range(2):
        p = f"blk.{layer}."
        for name in ("attn_norm", "ffn_norm", "dflare_norm"):
            tensors[p + name + ".weight"] = np.ones(width)
        for name in ("attn_q_norm", "attn_k_norm", "attn_k_ctx_norm"):
            tensors[p + name + ".weight"] = np.ones(dim)
        for name in ("attn_k", "attn_v", "attn_k_ctx", "attn_v_ctx"):
            tensors[p + name + ".weight"] = w(kv_heads*dim, width)
        tensors[p + "attn_q.weight"] = w(heads*dim, width)
        tensors[p + "attn_output.weight"] = w(width, heads*dim)
        tensors[p + "ffn_gate.weight"] = w(ff, width)
        tensors[p + "ffn_up.weight"] = w(ff, width)
        tensors[p + "ffn_down.weight"] = w(width, ff)
    write_gguf(path, kv, tensors, raw)


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
