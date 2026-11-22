#!/usr/bin/env python3
"""Statistics of the K and V values stored in the global layers' KV cache, and a simulation of one-byte and 4-bit
formats on those real values. Offline: it reads a slot state file, needs no model and no GPU, only numpy.

Why: plan items M4, A1, A2 (docs/torad/research/inkling-2-small/15-kernels-shaped-to-the-architecture.md section 2 and
16-six-techniques-addendum.md section 1). They choose the format for item 3.4 (one byte per K/V value in the global
layers) and decide whether K needs a static per-channel scale (KIVI with SmoothQuant's migration). The real gate for a
format is still the KL capture against F16: these numbers only say which candidates are worth that run.

USAGE (the four-card box, the real model; about 16K to 32K tokens a state)
-------------------------------------------------------------------------
1. Serve with --slot-save-path, one slot, F16 caches, flash attention on. This is the tensor-4 launch with the pool
   cut to one session. States saved from quantized caches parse too: rows of q4_0 or q8_0 decode on read into the
   stored (rotated) domain the kernel sees, so statistics and format simulations run on deployed values rather than
   F16 stand-ins, and the reader reports the header's rotation-on-store (n_rot_k 128 for K, n_rot_v 64 for V):

     mkdir -p /tmp/kvstates
     python3 scripts/inkling2/serve.py --profile tensor-4 --packs /srv/models/inkling-2-small-abliterated-q3km \\
         --server "$PWD/build/bin/llama-server" --no-draft --text-only --slots 1 --context 49152 \\
         --slot-context 40960 --cache-ram 0 --slot-save-path /tmp/kvstates --port 8103 &
     until curl -sf localhost:8103/health; do sleep 5; done

2. Put one prompt of about 24K tokens in slot 0, then save the slot. K and V are about 32 KiB a token (8 layers x K
   and V x 2 KiB), so 24K tokens is 0.8 GiB of the file (it is larger by the window cache, 160 MiB, and the context
   checkpoints the server appends). 16K tokens gives thirds of 5K cells for the depth check; 32K is 1 GiB.
   Make two states from different text (code, prose) for the cross-prompt check; or send the parity capture's
   prompts as "prompt" if A1 should be over them. C++ is about 3 characters a token: check tokens_evaluated.

     python3 - <<'EOF' > /tmp/kvstates/req_code.json
     import json, pathlib
     text = "".join(p.read_text(errors="replace") for p in sorted(pathlib.Path("src").glob("*.cpp")))[:75000]
     print(json.dumps({"prompt": text, "n_predict": 1, "id_slot": 0, "cache_prompt": True, "temperature": 0}))
     EOF
     curl -s localhost:8103/completion -H 'Content-Type: application/json' -d @/tmp/kvstates/req_code.json \\
         | python3 -c 'import sys, json; print("tokens", json.load(sys.stdin)["tokens_evaluated"])'
     curl -s -X POST 'localhost:8103/slots/0?action=save' -H 'Content-Type: application/json' \\
         -d '{"filename": "code.state"}'
     curl -s -X POST 'localhost:8103/slots/0?action=erase'

   Then the same with the prose (docs/torad/research/inkling-2-small/*.md) saved as prose.state. The save answer has
   n_saved (tokens) and n_written (bytes, with the appended checkpoints, which this script does not read). Stop the
   server after the last save.

3. Run (python3 -c "import numpy" must work; if not: python3 -m pip install --target ~/kvlibs numpy, then
   export PYTHONPATH=~/kvlibs):

     python3 scripts/inkling2/kv_stats.py /tmp/kvstates/code.state /tmp/kvstates/prose.state --out /tmp/kvstats

   It prints the summary and writes kv_stats.json and kv_stats.txt in --out. Memory stays under 200 MB, run time is
   about 30 s for a 32K-token state. --no-sim skips the format simulation; --sim-cells 0 simulates every cell instead
   of 16384 evenly spread ones. --layer-ids sets the model layer of each cache layer (default 5, 11, ... as in
   Small). Exit 2: the file is not a state this script reads; exit 3: the self-check failed.

Self-check (hard assertions; --selfcheck warn lists the offenders and goes on, off skips): every stored V vector (one
cell, one KV head, 128 values) has RMS 1 (RMS norm, no gain, before the store; it adds eps, so a token with a tiny V
comes out a little under 1 and the bound is [0.9, 1.01]), and every K vector has L2 norm at most 22.6274 (L2 clip,
plus 0.2% for the F16 rounding). If the parser reads the right bytes they hold. A failure names the layer, cell row,
position and head. Checked on states of a real server: the synthetic Inkling 2 test model (single slot, three slots
with one stream each, unified pool with a two-range sequence) and the Mini model (8 global layers) on CPU.

State file (src/llama-context.cpp state_seq_save_file, src/llama-kv-cache.cpp state_write), little endian:
  u32 'ggsq', u32 version 4, u32 n_words, n_words x u32 server token state (skipped)
  global cache: u32 n_stream, per stream u32 cell_count (0: nothing else), per cell {i32 pos, u32 n_seq_id,
  n_seq_id x i32}, u32 v_trans, u32 n_layer, u32 n_rot_k, u32 n_rot_v, per layer {i32 type, u64 row_bytes, rows of K},
  per layer the same for V. The window cache, the convolution state and the server's appendix follow: not read.
  A row is one cell, n_heads x head_dim values, in cell order. Under -sm tensor ggml_backend_meta_buffer_get_tensor
  gathers each row from the cards in card order (ggml-backend-meta.cpp, ggml_backend_meta_get_dense), so the rows are
  logical, KV heads in order, whatever the split.

Formats simulated (input: the F16 values as stored; every error is against them):
  fp8_e4m3       E4M3, scale 1, saturating at 448, round to nearest even; 8 bits a value
  q8_0           ggml Q8_0: blocks of 32, d = amax/127 kept as F16, roundf; 8.5 bits
  fp4_e2m1_b16   e2m1 values, a scale per 16 values as E4M3 (amax/6), no outer scale; 4.5 bits
  q4_0           ggml Q4_0: blocks of 32, d = max/-8 as F16; 4.5 bits (the 4-bit cache the engine has today)
  "+chan"        the same with a static per-channel scale s_c: stored = x / s_c, read back * s_c, with s_c the channel
                 RMS over the sample divided by its geometric mean over the head's channels (per layer and KV head).
                 "+chan@first" (from the second state on) takes s_c from the first state: the cross-prompt case.
Per format: the relative L2 error of each (cell, KV head) vector (mean, p99), and the relative RMS error of q.k for
random Gaussian q, a proxy. The q.k error is the exact expectation over q, not a sample: for q ~ N(0, diag(w^2)) the
squared error of q.k is sum_c w_c^2 (khat_c - k_c)^2, so the ratio of its sum over the sample to that of q.k itself
needs no draws (a few draws shared by all cells are noisy when K has a hot channel; the tests check the closed form
against actual draws). "iso": w = 1, which makes it the energy-weighted relative L2 error (large vectors count more
than in the per-vector mean). "prof": w is the channel RMS profile of K, as if Q and K shared their large channels.
"""

import argparse
import json
import math
import os
import struct
import sys
import time
from collections import namedtuple

import numpy as np

STATE_MAGIC = 0x67677371  # 'ggsq'
STATE_VERSION = 4
TOKEN_NULL = 0xFFFFFFFF
TYPES = {0: ("f32", 4), 1: ("f16", 2), 30: ("bf16", 2)}
# Quantized rows decode to f32 on read, in the stored (rotated) domain the kernel sees: per-block scale first,
# then the values of one ggml block. GGML_CUDA_FATTN_REL_DECODE_* reads exactly these bytes on device.
QUANT_TYPES = {8: ("q4_0", 18, 32), 13: ("q8_0", 34, 32)}  # id -> (name, block bytes, values a block)


def _dequant_q(buf, name):
    """One flat layer's quantized rows as float32 [values], blocks of d + q."""
    n_blocks = buf.size // (QUANT_TYPES[8 if name == "q4_0" else 13][1])
    b = buf.reshape(n_blocks, QUANT_TYPES[8 if name == "q4_0" else 13][1])
    d = b[:, :2].copy().view("<f2").astype(np.float32).reshape(-1)   # one f16 scale a block; copy before view:
    # the column slice is strided, and numpy only reinterprets itemsize on contiguous memory
    if name == "q8_0":
        return (b[:, 2:].view(np.int8).astype(np.float32) * d[:, None]).reshape(-1)
    lo = (b[:, 2:] & 0x0F).astype(np.float32)
    hi = (b[:, 2:] >> 4).astype(np.float32)
    q = np.empty((n_blocks, 32), np.float32)
    q[:, :16] = lo    # ggml's canonical q4_0: byte j holds value j in its low nibble and value j+16 in its
    q[:, 16:] = hi    # high nibble (quantize_row_q4_0_ref, ggml-quants.c; the CUDA store in cpy-utils.cuh,
                      # and device readers in fattn-rel-decode.cuh, all agree)
    return ((q - 8.0) * d[:, None]).reshape(-1)

HEAD_DIM = 128
FIRST_GLOBAL_LAYER = 5  # the global layers are 5, 11, ..., 47 in Small and 5, 11, 17, 23 in Mini
GLOBAL_LAYER_STEP = 6
K_NORM_MAX = 22.6274  # qk_l2_clip_threshold
K_TOL = 2e-3  # slack for the F16 rounding of the stored values (at most 2^-11 on the norm)
V_RMS_MIN = 0.9  # RMS norm gives sqrt(ms / (ms + eps)): a token with a tiny V comes out under 1 (0.994 seen in Mini)
V_RMS_MAX = 1.01  # the norm never gives more than 1; the slack is the F16 rounding
EDGE = 0.99  # K "at the clip" is norm >= 0.99 * the clip; V "low" is RMS < 0.99
HOT = 2.0  # a hot channel has RMS over this many times the median channel of its head
E4M3_MAX = 448.0
N_THIRDS = 3
TOP_K = 8


class StateError(Exception):
    """The file is not a sequence state this script can read."""


class SelfCheckError(AssertionError):
    """Stored values break the bounds the architecture gives them."""


# ---------------------------------------------------------------------------------------------------------------------
# the state file
# ---------------------------------------------------------------------------------------------------------------------

class _Reader:
    def __init__(self, f, size):
        self.f, self.size, self.off = f, size, 0

    def peek(self, n):
        if n < 0 or self.off + n > self.size:
            raise StateError(f"the file ends inside the cache (byte {self.off}, wanted {n} more, {self.size - self.off} left)")
        self.f.seek(self.off)
        data = self.f.read(n)
        if len(data) != n:
            raise StateError(f"short read at byte {self.off}")
        return data

    def take(self, n):
        data = self.peek(n)
        self.off += n
        return data

    def skip(self, n):
        if n < 0 or self.off + n > self.size:
            raise StateError(f"the file ends inside the cache (byte {self.off}, wanted {n} more, {self.size - self.off} left)")
        self.off += n

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def left(self):
        return self.size - self.off


def _read_tokens(raw):
    """The server writes a marker, a version, the token list and the media list; an older writer wrote the plain list."""
    w = np.frombuffer(raw, dtype="<u4")
    if len(w) >= 3 and w[0] == TOKEN_NULL and w[1] == 1 and 3 + int(w[2]) <= len(w):
        return w[3:3 + int(w[2])].astype(np.int64)
    return w.astype(np.int64)


def _read_cells(rd, cell_count):
    """Per cell {i32 pos, u32 n_seq_id, n_seq_id x i32}. One seq id a cell is the common case and a fixed stride."""
    need = cell_count * 12
    if rd.off + need <= rd.size:
        rec = np.frombuffer(rd.peek(need), dtype=[("pos", "<i4"), ("n", "<u4"), ("sid", "<i4")])
        if (rec["n"] == 1).all():
            rd.skip(need)
            return rec["pos"].astype(np.int64), rec["sid"].astype(np.int64)
    pos = np.empty(cell_count, np.int64)
    first = np.full(cell_count, -1, np.int64)
    for i in range(cell_count):
        p, n = struct.unpack("<iI", rd.take(8))
        if n > 4096:
            raise StateError(f"cell {i} claims {n} sequence ids: the cell records are not in the layout this script reads")
        ids = struct.unpack(f"<{n}i", rd.take(4 * n)) if n else ()
        pos[i] = p
        first[i] = ids[0] if ids else -1
    return pos, first


class KVState:
    """One saved sequence: the global layers' K and V rows of its cells, in file order, read on demand."""

    def __init__(self, path, head_dim=HEAD_DIM):
        self.path = str(path)
        self.size = os.path.getsize(path)
        with open(path, "rb") as f:
            self._parse(f, head_dim)

    def _parse(self, f, head_dim):
        rd = _Reader(f, self.size)
        magic, version, n_words = struct.unpack("<3I", rd.take(12))
        if magic != STATE_MAGIC:
            raise StateError(f"{self.path}: not a sequence state file (magic {magic:#010x}, wanted {STATE_MAGIC:#010x})")
        if version != STATE_VERSION:
            raise StateError(f"{self.path}: state version {version}, this script reads version {STATE_VERSION}")
        self.tokens = _read_tokens(rd.take(4 * n_words))
        self.n_stream = rd.u32()
        if not 1 <= self.n_stream <= 4096:
            raise StateError(f"{self.path}: {self.n_stream} streams: the base cache header is not where it should be")
        self.stream = None
        for s in range(self.n_stream):
            n = rd.u32()
            if n == 0:
                continue
            if n > rd.left() // 8:  # a cell takes 8 bytes of meta at least
                raise StateError(f"{self.path}: {n} cells claimed but {rd.left()} bytes are left in the file")
            if self.stream is not None:
                raise StateError(f"{self.path}: cells in two streams; save one sequence of one slot")
            self.stream = s
            self._parse_stream(rd, n, head_dim)
        if self.stream is None:
            raise StateError(f"{self.path}: the global cache holds no cells of this sequence")
        self.swa = None  # the window cache follows: its stream count and its cell count, as a cross-check
        if rd.left() >= 8:
            self.swa = struct.unpack("<2I", rd.peek(8))

    def _parse_stream(self, rd, cell_count, head_dim):
        self.cell_count = cell_count
        self.pos, self.seq = _read_cells(rd, cell_count)
        v_trans, n_layer, self.n_rot_k, self.n_rot_v = struct.unpack("<4I", rd.take(16))
        if v_trans > 1 or not 1 <= n_layer <= 4096:
            raise StateError(f"{self.path}: cache header v_trans={v_trans} n_layer={n_layer} is not plausible: the cell "
                             "records before it are not in the layout this script reads")
        if v_trans:
            raise StateError(f"{self.path}: V is stored transposed (no flash attention); serve with -fa on")
        # A state from a rotated cache (n_rot_k/n_rot_v nonzero: a quantized type rotates by Walsh-Hadamard on store) is
        # read as-is, in its stored domain: the rotation is orthogonal and Q is multiplied by the same rotation at
        # attention time, so dot products and per-block relative errors computed on the stored values are exactly what
        # the kernel sees; block scales sit along storage order either way. An orthogonal transform keeps each vector's
        # L2 norm, so the bounds the self-check tests stand here too. The header still carries n_rot_k/n_rot_v and
        # info()/render_text report them.
        self.n_layer = n_layer
        self.off = {"K": [], "V": []}
        types = {"K": set(), "V": set()}
        for kind in ("K", "V"):
            for _ in range(n_layer):
                t, row = struct.unpack("<iQ", rd.take(12))
                types[kind].add((t, row))
                self.off[kind].append(rd.off)
                rd.skip(cell_count * row)
        both = types["K"] | types["V"]
        if len(both) != 1:
            raise StateError(f"{self.path}: layers differ in type or row size ({sorted(both)}): not supported")
        t, row = both.pop()
        if t in QUANT_TYPES:
            name, blk, vpb = QUANT_TYPES[t]
            vals = row // blk * vpb
            if row % blk or head_dim % vpb or vals % head_dim:
                raise StateError(f"{self.path}: row of {row} bytes is not whole {name} blocks of "
                                 f"{blk} bytes covering whole heads of {head_dim}")
            self.type_name, self.row_bytes = name, row
            self.n_heads = vals // head_dim
            self.head_dim = head_dim
        elif t not in TYPES:
            raise StateError(f"{self.path}: cache type id {t} is not one of f32/f16/bf16/q4_0/q8_0")
        else:
            self.type_name, elem = TYPES[t]
            if row % (elem * head_dim):
                raise StateError(f"{self.path}: row of {row} bytes is not a whole number of heads of {head_dim} {self.type_name}")
            self.row_bytes = row
            self.n_heads = row // (elem * head_dim)
            self.head_dim = head_dim


    def read(self, kind, layer, c0, c1):
        """Cells c0..c1 (file order) of one layer as float32 [cells, heads, head_dim]."""
        n = c1 - c0
        buf = np.empty(n * self.row_bytes, np.uint8)
        with open(self.path, "rb") as f:
            f.seek(self.off[kind][layer] + c0 * self.row_bytes)
            if f.readinto(memoryview(buf)) != buf.size:
                raise StateError(f"{self.path}: short read of {kind} layer {layer}")
        if self.type_name in ("q4_0", "q8_0"):
            return _dequant_q(buf, self.type_name).reshape(n, self.n_heads, self.head_dim)
        if self.type_name == "bf16":
            x = (buf.view("<u2").astype(np.uint32) << 16).view(np.float32)
        else:
            x = buf.view("<f2" if self.type_name == "f16" else "<f4").astype(np.float32)
        return x.reshape(n, self.n_heads, self.head_dim)

    def third_index(self):
        """0, 1, 2 for the first, middle and last third of the cells by position."""
        n = self.cell_count
        order = np.argsort(self.pos, kind="stable")
        rank = np.empty(n, np.int64)
        rank[order] = np.arange(n)
        return ((rank * N_THIRDS) // n).astype(np.int64)

    def info(self):
        return {"path": self.path, "bytes": self.size, "tokens_in_header": int(len(self.tokens)), "cells": int(self.cell_count),
                "pos_min": int(self.pos.min()), "pos_max": int(self.pos.max()), "n_stream": int(self.n_stream),
                "stream": int(self.stream), "n_layer": int(self.n_layer), "n_heads": int(self.n_heads),
                "head_dim": int(self.head_dim), "type": self.type_name, "n_rot_k": int(self.n_rot_k),
                "n_rot_v": int(self.n_rot_v), "window_cache": list(self.swa) if self.swa else None}


# ---------------------------------------------------------------------------------------------------------------------
# per layer statistics and the self check
# ---------------------------------------------------------------------------------------------------------------------

class Accum:
    """Running statistics of one layer and one kind (K or V), fed with chunks of cells."""

    def __init__(self, kind, layer, label, state, cfg):
        self.kind, self.layer, self.label, self.state, self.cfg = kind, layer, label, state, cfg
        h, d = state.n_heads, state.head_dim
        self.max_abs = np.zeros((h, d), np.float32)
        self.sumsq = np.zeros((N_THIRDS, h, d), np.float64)
        self.count = np.zeros(N_THIRDS, np.int64)
        self.hist = np.zeros(256, np.int64)  # by the biased exponent of the float32 magnitude; 0 is an exact zero
        self.vec_min, self.vec_max, self.vec_sum, self.vec_n = math.inf, -math.inf, 0.0, 0
        self.at_min = self.at_max = None
        self.bad = 0
        self.offenders = []
        self.edge = 0  # K: vectors at the clip; V: vectors with RMS under 0.99
        if kind == "K":
            self.lo, self.hi = 0.0, cfg.k_norm_max * (1.0 + cfg.k_tol)
        else:
            self.lo, self.hi = cfg.v_rms_min, cfg.v_rms_max

    def update(self, x, c0, third):
        with np.errstate(invalid="ignore", over="ignore"):  # wrong bytes give inf and nan: the check below reports them
            self._update(x, c0, third)

    def _update(self, x, c0, third):
        n, h, d = x.shape
        ax = np.abs(x)
        self.max_abs = np.maximum(self.max_abs, ax.max(axis=0))
        self.hist += np.bincount((ax.view(np.uint32) >> 23).ravel(), minlength=256)
        sq = x * x
        onehot = (third[:, None] == np.arange(N_THIRDS)[None, :]).T.astype(np.float32)
        self.sumsq += (onehot @ sq.reshape(n, h * d)).astype(np.float64).reshape(N_THIRDS, h, d)
        self.count += onehot.sum(axis=1).astype(np.int64)
        ss = sq.sum(axis=2)
        vec = np.sqrt(ss) if self.kind == "K" else np.sqrt(ss / d)
        self.vec_sum += float(vec.sum(dtype=np.float64))
        self.vec_n += vec.size
        self.edge += int((vec >= EDGE * self.cfg.k_norm_max).sum() if self.kind == "K" else (vec < EDGE).sum())
        for name, fn, better in (("min", np.argmin, lambda a, b: a < b), ("max", np.argmax, lambda a, b: a > b)):
            i = np.unravel_index(fn(vec), vec.shape)
            if better(float(vec[i]), getattr(self, "vec_" + name)):
                setattr(self, "vec_" + name, float(vec[i]))
                setattr(self, "at_" + name, (int(c0 + i[0]), int(i[1])))
        if self.cfg.selfcheck != "off":
            bad = ~((vec >= self.lo) & (vec <= self.hi))  # a NaN is bad too
            if bad.any():
                self.bad += int(bad.sum())
                for i, hh in np.argwhere(bad)[:max(0, 8 - len(self.offenders))]:
                    self.offenders.append({"cell": int(c0 + i), "pos": int(self.state.pos[c0 + i]), "head": int(hh),
                                           "value": float(vec[i, hh])})
                if self.cfg.selfcheck == "fail":
                    raise SelfCheckError(self.describe(int(bad.sum()), n * h, c0, c0 + n))

    def describe(self, n_bad, n_vec, c0, c1):
        what = "L2 norm" if self.kind == "K" else "RMS"
        rows = "\n".join(f"  model layer {self.label}, cell row {o['cell']}, position {o['pos']}, kv head {o['head']}: "
                         f"{what} {o['value']:.6g}" for o in self.offenders)
        return (f"{self.kind} self-check failed in {self.state.path}: global layer index {self.layer} (model layer "
                f"{self.label}) has {n_bad} of {n_vec} vectors in cell rows {c0}..{c1 - 1} with {what} outside "
                f"[{self.lo:.6g}, {self.hi:.6g}]. Offenders:\n{rows}\n"
                "Stored V rows must have RMS 1 (RMS norm without gain; a little under 1 for a token with a tiny V) and K rows "
                "L2 norm <= 22.6274 (L2 clip). If most vectors fail, the parser is reading the wrong bytes: check --head-dim, "
                "that the server ran F16 caches with -fa on and that the file is a slot state. For a few real outliers use "
                "--selfcheck warn.")

    def finish(self):
        total = int(self.hist.sum())
        sumsq = self.sumsq.sum(axis=0)
        n = int(self.count.sum())
        rms = np.sqrt(sumsq / max(n, 1))
        thirds = None
        if (self.count > 0).all():
            thirds = np.sqrt(self.sumsq / self.count[:, None, None])
        ch = np.unravel_index(int(self.max_abs.argmax()), self.max_abs.shape)

        def below(k):  # share of elements with |x| < 2^-k, zeros included
            return float(self.hist[:127 - k].sum() / total)

        hist = {int(e) - 127: int(c) for e, c in enumerate(self.hist) if c and e}  # count by floor(log2 |x|), zeros apart
        return {"kind": self.kind, "n_cells": n, "n_vectors": self.vec_n, "max_abs": self.max_abs, "rms": rms,
                "rms_thirds": thirds, "third_cells": self.count.copy(), "max_abs_all": float(self.max_abs.max()),
                "max_abs_at": (int(ch[0]), int(ch[1])), "rms_all": float(np.sqrt(sumsq.sum() / (n * sumsq.size))),
                "vec_min": self.vec_min, "vec_max": self.vec_max, "vec_mean": self.vec_sum / max(self.vec_n, 1),
                "vec_min_at": self.at_min, "vec_max_at": self.at_max, "frac_lt_2m6": below(6), "frac_lt_2m10": below(10),
                "frac_zero": float(self.hist[0] / total), "edge_share": self.edge / max(self.vec_n, 1), "hist_log2": hist,
                "violations": self.bad, "offenders": self.offenders, "limits": [self.lo, self.hi]}


def layer_label(cfg, li):
    if cfg.layer_ids:
        return cfg.layer_ids[li] if li < len(cfg.layer_ids) else None
    return FIRST_GLOBAL_LAYER + GLOBAL_LAYER_STEP * li


def eff_dim_pass(state, chunk_cells=4096, log=None):
    """Per-head effective dimensionality of K and V in the stored (rotated) domain, per global layer.

    The participation ratio ED = (sum eigenvalues)^2 / sum(eigenvalue^2) of one head's Gram matrix answers the
    truncating-basis question: an ED near the head size means energy is spread and a low-rank basis before store
    saves nothing worth its error; a small ED means most components are noise a truncation could drop. Also
    reports how many components carry 90/95/99 percent of the energy."""
    d = state.head_dim
    fracs = (0.90, 0.95, 0.99)

    def one(kind, li):
        heads = state.n_heads
        gram = np.zeros((heads, d, d), np.float64)
        for c0 in range(0, state.cell_count, chunk_cells):
            x = state.read(kind, li, c0, min(c0 + chunk_cells, state.cell_count))
            gram += np.einsum("nhd,nhe->hde", x.astype(np.float64), x.astype(np.float64))
        ed, n_frac = [], [[None] * len(fracs) for _ in range(heads)]
        for h in range(heads):
            lam = np.linalg.eigvalsh(gram[h])[::-1]
            lam = np.maximum(lam, 0.0)
            total = lam.sum()
            if total <= 0:
                ed.append(float(d)); [nf.__setitem__(i, float(d)) for i, nf in enumerate(n_frac[h])]
                continue
            ed.append(float(total ** 2 / np.square(lam).sum()))
            cum = np.cumsum(lam) / total
            for fi, f in enumerate(fracs):
                n_frac[h][fi] = int(np.searchsorted(cum, f - 1e-12) + 1)
        row = {"ed_mean": float(np.mean(ed)), "ed_min": float(min(ed)), "ed_max": float(max(ed))}
        for fi, f in enumerate(fracs):
            allv = [n_frac[h][fi] for h in range(heads)]
            row[f"n{int(f*100)}_mean"] = float(np.mean(allv))
            row[f"n{int(f*100)}_max"] = int(max(allv))
        return row

    out = []
    for li in range(state.n_layer):
        entry = {}
        for kind in ("K", "V"):
            t0 = time.time()
            entry[kind] = one(kind, li)
            if log:
                log(f"  eff-dim {kind} layer {li + 1}/{state.n_layer} {time.time() - t0:.1f}s")
        out.append(entry)
    return out


def collect_stats(state, cfg, log=None):
    third = state.third_index()
    out = []
    for li in range(state.n_layer):
        entry = {}
        for kind in ("K", "V"):
            t0 = time.time()
            acc = Accum(kind, li, layer_label(cfg, li), state, cfg)
            for c0 in range(0, state.cell_count, cfg.chunk_cells):
                c1 = min(c0 + cfg.chunk_cells, state.cell_count)
                acc.update(state.read(kind, li, c0, c1), c0, third[c0:c1])
            entry[kind] = acc.finish()
            if log:
                log(f"  stats {kind} layer {li + 1}/{state.n_layer} {time.time() - t0:.1f}s")
        out.append(entry)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# channel profiles: scales and the A2 measures
# ---------------------------------------------------------------------------------------------------------------------

def make_scale(rms, floor=1e-3):
    """Static scale per channel from the channel RMS [H, D]: geometric mean 1 in each head. A dead channel is
    floored at 1e-3 of the head's mean RMS so the scale stays finite."""
    r = np.maximum(rms.astype(np.float64), floor * rms.mean(axis=-1, keepdims=True))
    r = np.maximum(r, 1e-30)
    return (r / np.exp(np.log(r).mean(axis=-1, keepdims=True))).astype(np.float32)


def pearson(a, b):
    """Pearson correlation along the last axis; nan where either side is flat."""
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1))
    return np.where(den > 0, (a * b).sum(axis=-1) / np.where(den > 0, den, 1.0), np.nan)


def top_overlap(a, b, k=TOP_K):
    """Share of the k largest channels of a that are among the k largest of b, along the last axis."""
    ia = np.argsort(-a, axis=-1)[..., :k]
    ib = np.argsort(-b, axis=-1)[..., :k]
    ma = np.zeros(a.shape, bool)
    mb = np.zeros(b.shape, bool)
    np.put_along_axis(ma, ia, True, axis=-1)
    np.put_along_axis(mb, ib, True, axis=-1)
    return (ma & mb).sum(axis=-1) / float(k)


def static_fit(rms, log=True):
    """rms [G, ..., D]: G groups (thirds or prompts) of a channel profile. Each group's profile is taken without its
    own level (log2 RMS minus its mean over channels, or RMS over its mean minus 1). A static profile is the mean over
    the groups. Returns the share of the variance (over groups and channels) the static profile explains, the spread
    of the static profile and the drift of the groups around it (standard deviations, log2 units when log)."""
    rms = rms.astype(np.float64)
    u = np.log2(np.maximum(rms, 1e-12)) if log else rms / np.maximum(rms.mean(axis=-1, keepdims=True), 1e-30) - 1.0
    u = u - u.mean(axis=-1, keepdims=True)
    m = u.mean(axis=0)
    tot = (u * u).mean(axis=(0, -1))
    sta = (m * m).mean(axis=-1)
    share = np.where(tot > 1e-18, sta / np.where(tot > 1e-18, tot, 1.0), np.nan)
    return share, np.sqrt(sta), np.sqrt(np.maximum(tot - sta, 0.0))


def compare_profiles(rms):
    """rms [G, H, D] -> the pair measures over all pairs of groups and the static fit, per head."""
    pairs = [(a, b) for a in range(len(rms)) for b in range(a + 1, len(rms))]
    lg = np.log2(np.maximum(rms.astype(np.float64), 1e-12))
    out = {"pairs": pairs,
           "r_lin": np.stack([pearson(rms[a].astype(np.float64), rms[b].astype(np.float64)) for a, b in pairs]),
           "r_log": np.stack([pearson(lg[a], lg[b]) for a, b in pairs]),
           "top_overlap": np.stack([top_overlap(rms[a], rms[b]) for a, b in pairs])}
    hot = rms > HOT * np.median(rms, axis=-1, keepdims=True)  # [G, H, D]
    out["hot_all"] = hot.all(axis=0).sum(axis=-1)  # per head: hot in every group
    out["hot_any"] = hot.any(axis=0).sum(axis=-1)  # per head: hot in some group
    for name, flag in (("log", True), ("lin", False)):
        share, spread, drift = static_fit(rms, log=flag)
        out["static_share_" + name], out["static_spread_" + name], out["static_drift_" + name] = share, spread, drift
    return out


def summarize_compare(c, n_cells=None):
    """Summary over heads and pairs. n_cells: the fewest cells in a group, for the floor that sampling alone puts on the
    drift (std of log2 of an RMS from n Gaussian values: 1.4427 / sqrt(2 n); heavy tails raise it)."""
    def lo_mean(a):
        a = np.asarray(a, np.float64)
        ok = a[np.isfinite(a)]
        return (float(ok.min()), float(ok.mean())) if ok.size else (None, None)
    s = {"pairs": c["pairs"]}
    for key in ("r_lin", "r_log", "top_overlap"):
        s[key + "_min"], s[key + "_mean"] = lo_mean(c[key])
    for key in ("static_share_log", "static_share_lin", "static_spread_log", "static_drift_log"):
        s[key + "_min"], s[key + "_mean"] = lo_mean(c[key])
    s["hot_all"], s["hot_any"] = int(np.sum(c["hot_all"])), int(np.sum(c["hot_any"]))  # channels, summed over heads
    s["sampling_floor_log2"] = 1.4427 / math.sqrt(2.0 * n_cells) if n_cells else None
    s["per_head"] = {k: c[k] for k in ("r_lin", "r_log", "top_overlap", "static_share_log", "static_share_lin")}
    return s


def channel_spread(rms):
    """Per head: largest channel RMS over the median channel RMS, and which channel."""
    med = np.maximum(np.median(rms, axis=-1), 1e-30)
    return rms.max(axis=-1) / med, rms.argmax(axis=-1)


# ---------------------------------------------------------------------------------------------------------------------
# formats: each takes a float32 array whose last axis is a whole number of blocks and returns the values read back
# ---------------------------------------------------------------------------------------------------------------------

def _f16(x):
    return x.astype(np.float16).astype(np.float32)


def fmt_e4m3(x):
    """Nearest E4M3 value, ties to even, saturating at 448: spacing 2^(e-3) from 2^-6 up, 2^-9 below it."""
    a = np.abs(x)
    _, e = np.frexp(a)  # a = m * 2^e with m in [0.5, 1), so floor(log2 a) = e - 1
    step = np.ldexp(np.float32(1.0), np.maximum(e - 1, -6) - 3)
    q = np.minimum(np.rint(a / step) * step, np.float32(E4M3_MAX))
    return np.copysign(q, x)


def _roundf(v):
    """C roundf in float32: half away from zero (np.rint goes to even, and floor(|v| + 0.5) is wrong near 0.5)."""
    t = np.trunc(v)
    return t + np.where(np.abs(v - t) >= 0.5, np.sign(v), np.float32(0.0))


def fmt_q8_0(x):
    """ggml quantize_row_q8_0_ref and dequantize_row_q8_0."""
    b = x.reshape(-1, 32)
    d = np.abs(b).max(axis=1, keepdims=True) / np.float32(127.0)
    inv = np.where(d != 0, np.float32(1.0) / np.where(d != 0, d, np.float32(1.0)), np.float32(0.0))
    return (_roundf(b * inv) * _f16(d)).reshape(x.shape)


def fmt_q4_0(x):
    """ggml quantize_row_q4_0_ref and dequantize_row_q4_0."""
    b = x.reshape(-1, 32)
    big = b[np.arange(len(b)), np.abs(b).argmax(axis=1)][:, None]  # signed value of the first largest magnitude
    d = big / np.float32(-8.0)
    inv = np.where(d != 0, np.float32(1.0) / np.where(d != 0, d, np.float32(1.0)), np.float32(0.0))
    q = np.minimum(np.floor(b * inv + np.float32(8.5)), np.float32(15.0))
    return ((q - np.float32(8.0)) * _f16(d)).reshape(x.shape)


E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
# the middle of two grid values goes to the neighbour with the even code, so it goes up when the upper code is even
_E2M1_MID = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], np.float32)
_E2M1_EDGE = np.where([(i + 1) % 2 == 0 for i in range(7)], _E2M1_MID, np.nextafter(_E2M1_MID, np.float32(np.inf)))


def fmt_fp4(x):
    """e2m1 values with a scale per 16 values: E4M3(amax / 6), no per-tensor scale."""
    b = x.reshape(-1, 16)
    s = fmt_e4m3(np.abs(b).max(axis=1, keepdims=True) / np.float32(6.0))
    ok = s > 0
    a = np.where(ok, np.abs(b) / np.where(ok, s, np.float32(1.0)), np.float32(0.0))
    q = E2M1[np.searchsorted(_E2M1_EDGE, a, side="right")]
    return (np.copysign(q, b) * s).reshape(x.shape)


Format = namedtuple("Format", "name bits fn sat")
FORMATS = (Format("fp8_e4m3", 8.0, fmt_e4m3, E4M3_MAX), Format("q8_0", 8.5, fmt_q8_0, None),
           Format("fp4_e2m1_b16", 4.5, fmt_fp4, None), Format("q4_0", 4.5, fmt_q4_0, None))


def simulate_kind(state, li, kind, scales, formats, rms, cfg):
    """Quantize and read back the cells of one layer in every format and every scale option.
    scales: {suffix: s [H, D] or None}. Returns {variant: metrics} and {variant: per-vector relative errors}."""
    n, h, d = state.cell_count, state.n_heads, state.head_dim
    stride = 1 if cfg.sim_cells <= 0 else max(1, -(-n // cfg.sim_cells))
    w2 = (rms * rms) / np.maximum((rms * rms).mean(axis=-1, keepdims=True), 1e-30)  # q.k weights of the "prof" q
    names = [f.name + suffix for suffix in scales for f in formats]
    rel = {v: [] for v in names}
    acc = {v: {"num": 0.0, "den": 0.0, "pnum": 0.0, "sat": 0, "zero": 0} for v in names}
    pden = 0.0
    n_used = 0
    for c0 in range(0, n, cfg.chunk_cells):
        c1 = min(c0 + cfg.chunk_cells, n)
        x = state.read(kind, li, c0, c1)
        x = x[(np.arange(c0, c1) % stride) == 0]
        if not len(x):
            continue
        n_used += len(x)
        den = np.einsum("nhd,nhd->nh", x, x)
        live = den > 0
        pden += float((x * x * w2).sum(dtype=np.float64))
        for suffix, s in scales.items():
            xin = x if s is None else x / s
            for f in formats:
                v = f.name + suffix
                y = f.fn(xin.reshape(-1, d)).reshape(x.shape)
                if s is not None:
                    y = y * s
                dl = y - x
                num = np.einsum("nhd,nhd->nh", dl, dl)
                a = acc[v]
                a["num"] += float(num[live].sum(dtype=np.float64))
                a["den"] += float(den[live].sum(dtype=np.float64))
                a["zero"] += int((~live).sum())
                rel[v].append(np.sqrt(num[live] / den[live]).astype(np.float32))
                if f.sat:
                    a["sat"] += int((np.abs(xin) > f.sat).sum())
                if kind == "K":
                    a["pnum"] += float((dl * dl * w2).sum(dtype=np.float64))
    metrics, pooled = {}, {}
    bits = {f.name: f.bits for f in formats}
    for suffix in scales:
        for f in formats:
            v = f.name + suffix
            r = np.concatenate(rel[v]) if rel[v] else np.zeros(0, np.float32)
            a = acc[v]
            m = {"bits": bits[f.name], "vectors": int(len(r)), "zero_vectors": a["zero"],
                 "rel_l2_mean": float(r.mean()) if len(r) else None,
                 "rel_l2_p99": float(np.percentile(r, 99)) if len(r) else None,
                 "rel_l2_max": float(r.max()) if len(r) else None}
            if f.sat:
                m["saturated"] = a["sat"]
            if kind == "K":
                m["qk_rel_rms_iso"] = math.sqrt(a["num"] / a["den"]) if a["den"] > 0 else None
                m["qk_rel_rms_prof"] = math.sqrt(a["pnum"] / pden) if pden > 0 else None
            metrics[v], pooled[v] = m, r
    return metrics, pooled, n_used


def simulate_state(state, stats, cfg, first_stats=None, log=None):
    """The format simulation for every layer of a state. The scale of "+chan" is the state's own channel RMS; for a
    state after the first, "+chan@first" takes it from the first state."""
    formats = [f for f in FORMATS if f.name in cfg.formats]
    out = {}
    for kind in ("K", "V"):
        layers, pooled = [], {}
        for li in range(state.n_layer):
            rms = stats[li][kind]["rms"]
            scales = {"": None, "+chan": make_scale(rms)}
            if first_stats is not None:
                scales["+chan@first"] = make_scale(first_stats[li][kind]["rms"])
            t0 = time.time()
            m, p, n_used = simulate_kind(state, li, kind, scales, formats, rms, cfg)
            layers.append({"cells_used": n_used, "variants": m})
            for v, r in p.items():
                pooled.setdefault(v, []).append(r)
            if log:
                log(f"  sim {kind} layer {li + 1}/{state.n_layer} {time.time() - t0:.1f}s")
        allv = {}
        for v, rs in pooled.items():
            r = np.concatenate(rs)
            allv[v] = {"vectors": int(len(r)), "rel_l2_mean": float(r.mean()) if len(r) else None,
                       "rel_l2_p99": float(np.percentile(r, 99)) if len(r) else None,
                       "rel_l2_max": float(r.max()) if len(r) else None}
            for key in ("qk_rel_rms_iso", "qk_rel_rms_prof"):
                if key in layers[0]["variants"][v]:  # pooled over layers by the cells each used
                    w = np.array([L["cells_used"] for L in layers], np.float64)
                    e = np.array([np.nan if L["variants"][v][key] is None else L["variants"][v][key] ** 2 for L in layers])
                    allv[v][key] = float(np.sqrt((w * e).sum() / w.sum())) if np.isfinite(e).all() else None
        out[kind] = {"layers": layers, "all": allv}
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------------------------------------------------

def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return float(f"{v:.6g}") if math.isfinite(v) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def kind_result(st):
    ratio, ch = channel_spread(st["rms"])
    worst = int(ratio.argmax())
    return {"n_cells": st["n_cells"], "n_vectors": st["n_vectors"], "max_abs_all": st["max_abs_all"],
            "max_abs_at": {"head": st["max_abs_at"][0], "channel": st["max_abs_at"][1]}, "rms_all": st["rms_all"],
            "vec_min": st["vec_min"], "vec_max": st["vec_max"], "vec_mean": st["vec_mean"],
            "vec_min_at": st["vec_min_at"], "vec_max_at": st["vec_max_at"], "frac_lt_2m6": st["frac_lt_2m6"],
            "frac_lt_2m10": st["frac_lt_2m10"], "frac_zero": st["frac_zero"], "edge_share": st["edge_share"],
            "hist_log2": st["hist_log2"],
            "violations": st["violations"], "offenders": st["offenders"], "limits": st["limits"],
            "spread_max_over_median_by_head": ratio, "spread_worst_head": worst, "spread_worst_channel": int(ch[worst]),
            "max_abs": st["max_abs"], "rms": st["rms"], "rms_thirds": st["rms_thirds"], "third_cells": st["third_cells"]}


def analyze(states, cfg, log=None):
    """Statistics, A2 and the simulation for each state, and the comparison between states."""
    res = {"tool": "kv_stats.py", "format": 1, "config": {k: v for k, v in vars(cfg).items() if k != "states"},
           "states": [], "cross_states": {}}
    all_stats = []
    for si, state in enumerate(states):
        if log:
            log(f"state {si + 1}/{len(states)}: {state.path}")
        stats = collect_stats(state, cfg, log)
        all_stats.append(stats)
        if getattr(cfg, "eff_dim", False):
            e = eff_dim_pass(state, chunk_cells=cfg.chunk_cells, log=log)
            res["states"][si]["eff_dim"] = [{"index": li, "layer_id": layer_label(cfg, li),
                                             "K": e[li]["K"], "V": e[li]["V"]}
                                            for li in range(state.n_layer)]
        layers = []
        for li in range(state.n_layer):
            e = {"index": li, "layer_id": layer_label(cfg, li)}
            for kind in ("K", "V"):
                e[kind] = kind_result(stats[li][kind])
                th = stats[li][kind]["rms_thirds"]
                e[kind]["a2_thirds"] = (summarize_compare(compare_profiles(th), int(stats[li][kind]["third_cells"].min()))
                                        if th is not None else None)
            layers.append(e)
        res["states"].append({"file": state.info(), "layers": layers})
    if not cfg.no_sim:
        for si, state in enumerate(states):
            first = all_stats[0] if si > 0 else None
            res["states"][si]["sim"] = simulate_state(state, all_stats[si], cfg, first, log)
    if len(states) > 1:
        res["cross_states"] = cross_states(states, all_stats, cfg)
    return _clean(res)


def cross_states(states, all_stats, cfg):
    """A2 across prompts: per layer and kind, the channel RMS profiles of all states compared."""
    out = []
    for li in range(states[0].n_layer):
        e = {"index": li, "layer_id": layer_label(cfg, li)}
        for kind in ("K", "V"):
            rms = np.stack([s[li][kind]["rms"] for s in all_stats])
            e[kind] = summarize_compare(compare_profiles(rms), min(s.cell_count for s in states))
        out.append(e)
    return out


def _g(v, spec=".4g"):
    return "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)


def _pct(v, spec=".3f"):
    return "nan" if v is None else format(100.0 * v, spec)


def _ext(fn, a):
    a = [v for v in a if v is not None]  # a layer of all-nan values has no extreme
    return fn(a) if a else None


def _mean(a):
    a = [v for v in a if v is not None]
    return sum(a) / len(a) if a else None


def render_text(res):
    L = []
    cfg = res["config"]
    for si, st in enumerate(res["states"]):
        f = st["file"]
        L.append("=" * 118)
        L.append(f"state {si + 1}: {f['path']}")
        L.append(f"  {f['bytes'] / 2 ** 30:.3f} GiB, {f['cells']} cells (positions {f['pos_min']}..{f['pos_max']}), "
                 f"{f['tokens_in_header']} tokens in the header, window cache {f['window_cache']}")
        L.append(f"  {f['n_layer']} global layers, {f['n_heads']} KV heads x {f['head_dim']}, {f['type']}, V not transposed, "
                 f"rotation k/v {f['n_rot_k']}/{f['n_rot_v']}; cache layer i is called model layer 5 + 6 i (--layer-ids to change)")
        bad = sum(L_[k]["violations"] for L_ in st["layers"] for k in "KV")
        K0, V0 = st["layers"][0]["K"]["limits"], st["layers"][0]["V"]["limits"]
        if cfg["selfcheck"] == "off":
            L.append("  self-check: off")
        else:
            kmax = _ext(max, [L_["K"]["vec_max"] for L_ in st["layers"]])
            vmin = _ext(min, [L_["V"]["vec_min"] for L_ in st["layers"]])
            vmax = _ext(max, [L_["V"]["vec_max"] for L_ in st["layers"]])
            L.append(f"  self-check: {'FAILED, ' + str(bad) + ' vectors outside' if bad else 'passed'}: K vector norm max "
                     f"{_g(kmax)} (limit {_g(K0[1])}); V vector RMS min {_g(vmin, '.5g')} max {_g(vmax, '.5g')} "
                     f"(allowed {_g(V0[0])}..{_g(V0[1])})")
            L.append(f"  K vectors at the clip (norm >= 99% of it): {_pct(_mean([L_['K']['edge_share'] for L_ in st['layers']]), '.2f')}%"
                     f"; V vectors with RMS under 0.99: {_pct(_mean([L_['V']['edge_share'] for L_ in st['layers']]), '.3f')}%")
            for L_ in st["layers"]:
                for k in "KV":
                    for o in L_[k]["offenders"]:
                        L.append(f"    offender {k} model layer {L_['layer_id']} cell row {o['cell']} pos {o['pos']} "
                                 f"head {o['head']} value {_g(o['value'], '.6g')}")
        L.append("")
        L.append("M4  stored values, all cells (% columns are shares of elements; at clip: K vectors with norm >= 99% of the clip; "
                 "<0.99: V vectors)")
        L.append("layer |  K max|x|     rms  norm_max norm_mean at clip% %<2^-6 %<2^-10 |  V max|x|     rms  rms_min  rms_max <0.99% "
                 "%<2^-6 %<2^-10")
        for L_ in st["layers"]:
            k, v = L_["K"], L_["V"]
            L.append(f"{str(L_['layer_id']):>5} | {_g(k['max_abs_all'], '8.4g')} {_g(k['rms_all'], '7.4g')} {_g(k['vec_max'], '8.5g')} "
                     f"{_g(k['vec_mean'], '9.5g')} {_pct(k['edge_share'], '8.2f')} {_pct(k['frac_lt_2m6'], '6.3f')} "
                     f"{_pct(k['frac_lt_2m10'], '7.3f')} | {_g(v['max_abs_all'], '8.4g')} {_g(v['rms_all'], '7.4g')} "
                     f"{_g(v['vec_min'], '8.5g')} {_g(v['vec_max'], '8.5g')} {_pct(v['edge_share'], '6.3f')} "
                     f"{_pct(v['frac_lt_2m6'], '6.3f')} {_pct(v['frac_lt_2m10'], '7.3f')}")
        L.append("")
        L.append("A1  channel profile: largest channel RMS over the median channel RMS, worst head, and where (head, channel)")
        L.append("layer   K ratio  (head, ch)    V ratio  (head, ch)    channel of the largest |x|: K, V")
        for L_ in st["layers"]:
            k, v = L_["K"], L_["V"]
            kk = f"{_g(_ext(max, k['spread_max_over_median_by_head']), '8.3f')}  ({k['spread_worst_head']}, {k['spread_worst_channel']})"
            vv = f"{_g(_ext(max, v['spread_max_over_median_by_head']), '8.3f')}  ({v['spread_worst_head']}, {v['spread_worst_channel']})"
            L.append(f"{str(L_['layer_id']):>5} {kk:<20} {vv:<20} {k['max_abs_at']['channel']}, {v['max_abs_at']['channel']}")
        L.append("")
        L.append("A2  across depth: first, middle, last third of the cells by position (r, top8: min over heads and the 3 pairs)")
        L.extend(_a2_legend())
        for L_ in st["layers"]:
            L.append(f"{str(L_['layer_id']):>5} " + "  |  ".join(_a2_cells(L_[kind]["a2_thirds"]) for kind in "KV"))
        if "sim" in st:
            L.extend(_render_sim(st))
        L.append("")
    if res["cross_states"]:
        L.append("=" * 118)
        L.append(f"A2  across prompts: the {len(res['states'])} states compared (r, top8: min over heads and pairs of states)")
        L.extend(_a2_legend())
        for e in res["cross_states"]:
            L.append(f"{str(e['layer_id']):>5} " + "  |  ".join(_a2_cells(e[kind]) for kind in "KV"))
    L.append("")
    L.append("Reading the simulation: errors are against the stored F16 values. q.k is a proxy; the gate for a format is the KL "
             "capture against F16.")
    return "\n".join(L)


A2_HEAD = "    K r_lin  r_log  top8  R2_log R2_lin spread drift floor  hot all/any"


def _a2_legend():
    return ["    r = Pearson of the channel RMS profiles (linear, log2); top8 = share of the 8 largest channels two groups share;",
            "    R2 = share of the channel-profile variance (over groups and channels) that a static profile explains (log2, linear);",
            "    spread = std of the static profile, drift = std of the groups around it (log2 units; R2, spread, drift: mean over heads);",
            "    floor = the drift that sampling alone gives for Gaussian values (a drift near it is noise; heavy tails raise the floor);",
            "    hot = channels over 2x their head's median RMS, summed over heads: in every group / in some group (A1: a set that is large always)",
            "layer" + A2_HEAD + "  |" + A2_HEAD.replace("K", "V", 1)]


def _a2_cells(a):
    if a is None:
        return " " * 63 + "n/a"
    return (f"{_g(a['r_lin_min'], '9.4f')} {_g(a['r_log_min'], '6.3f')} {_g(a['top_overlap_min'], '5.2f')} "
            f"{_g(a['static_share_log_mean'], '6.3f')} {_g(a['static_share_lin_mean'], '6.3f')} "
            f"{_g(a['static_spread_log_mean'], '6.3f')} {_g(a['static_drift_log_mean'], '5.3f')} {_g(a['sampling_floor_log2'], '5.3f')} "
            f"{a['hot_all']:>5}/{a['hot_any']:<5}")


def _render_sim(st):
    L = [""]
    sim = st["sim"]
    for kind in "KV":
        names = list(sim[kind]["all"])
        bits = {v: sim[kind]["layers"][0]["variants"][v]["bits"] for v in names}
        L.append(f"Format simulation, {kind}: relative L2 error of a (cell, head) vector in percent, mean / p99 "
                 f"({sim[kind]['layers'][0]['cells_used']} cells a layer)"
                 + ("; qk = relative RMS error of q.k for Gaussian q, a proxy (iso: unit variance, prof: K channel RMS profile)"
                    if kind == "K" else ""))
        head = f"{'variant':<22} {'bits':>4} " + " ".join(f"{'L' + str(e['layer_id']):>11}" for e in st["layers"]) + \
               f" | {'all mean':>8} {'all p99':>8} {'sat':>5}" + (f" {'qk iso':>7} {'qk prof':>7}" if kind == "K" else "")
        L.append(head)
        for v in names:
            cols = []
            for lay in sim[kind]["layers"]:
                m = lay["variants"][v]
                cols.append(f"{_pct(m['rel_l2_mean'], '5.2f')}/{_pct(m['rel_l2_p99'], '5.2f')}".rjust(11))
            a = sim[kind]["all"][v]
            sat = sum(lay["variants"][v].get("saturated", 0) for lay in sim[kind]["layers"])
            row = f"{v:<22} {_g(bits[v], '4.1f')} " + " ".join(cols) + f" | {_pct(a['rel_l2_mean'], '8.3f')} " \
                  f"{_pct(a['rel_l2_p99'], '8.3f')} {sat:>5}"
            if kind == "K":
                row += f" {_pct(a.get('qk_rel_rms_iso'), '7.3f')} {_pct(a.get('qk_rel_rms_prof'), '7.3f')}"
            L.append(row)
        L.append("")
    return L


# ---------------------------------------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("states", nargs="+", help="slot state files, from POST /slots/ID?action=save")
    p.add_argument("--out", default="kv_stats_out", help="directory for kv_stats.json and kv_stats.txt (default: kv_stats_out)")
    p.add_argument("--head-dim", type=int, default=HEAD_DIM, help="values in a KV head (Inkling 2 Small and Mini: 128)")
    p.add_argument("--layer-ids", type=lambda s: [int(x) for x in s.split(",")],
                   help="model layer index of each cache layer, comma separated (default: 5, 11, 17, ...)")
    p.add_argument("--k-norm-max", type=float, default=K_NORM_MAX, help="L2 clip of K (default: %(default)s)")
    p.add_argument("--k-tol", type=float, default=K_TOL, help="relative slack on it for the F16 stored values (default: %(default)s)")
    p.add_argument("--v-rms-min", type=float, default=V_RMS_MIN,
                   help="lowest RMS of a V vector (1 less the RMS norm's eps effect; default: %(default)s)")
    p.add_argument("--v-rms-max", type=float, default=V_RMS_MAX, help="highest RMS of a V vector (default: %(default)s)")
    p.add_argument("--selfcheck", choices=["fail", "warn", "off"], default="fail",
                   help="fail: stop at the first chunk with a vector outside the bounds; warn: count and list them; off")
    p.add_argument("--chunk-cells", type=int, default=4096, help="cells read at a time (default: %(default)s)")
    p.add_argument("--sim-cells", type=int, default=16384,
                   help="cells simulated a layer, evenly spread; 0 for all (default: %(default)s)")
    p.add_argument("--formats", type=lambda s: s.split(","), default=[f.name for f in FORMATS],
                   help="formats to simulate, comma separated (default: all of " + ",".join(f.name for f in FORMATS) + ")")
    p.add_argument("--no-sim", action="store_true", help="statistics only")
    p.add_argument("--eff-dim", action="store_true",
                   help="per-head effective dimensionality (participation ratio of the Gram spectrum in the stored "
                        "domain) per global layer for K and V: settles whether a truncating basis before store beats "
                        "a flat low-bit format on bytes or error")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress lines")
    cfg = p.parse_args(argv)
    unknown = set(cfg.formats) - {f.name for f in FORMATS}
    if unknown:
        p.error(f"unknown formats {sorted(unknown)}")
    return cfg


def main(argv=None):
    cfg = parse_args(argv)
    log = None if cfg.quiet else (lambda m: print(m, file=sys.stderr, flush=True))
    try:
        states = [KVState(path, cfg.head_dim) for path in cfg.states]
        for s in states[1:]:
            if (s.n_layer, s.n_heads) != (states[0].n_layer, states[0].n_heads):
                raise StateError(f"{s.path}: {s.n_layer} layers of {s.n_heads} heads, the first state has "
                                 f"{states[0].n_layer} of {states[0].n_heads}")
        res = analyze(states, cfg, log)
    except (StateError, OSError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except SelfCheckError as err:
        print(f"SELF-CHECK FAILED: {err}", file=sys.stderr)
        return 3
    text = render_text(res)
    os.makedirs(cfg.out, exist_ok=True)
    with open(os.path.join(cfg.out, "kv_stats.json"), "w") as f:
        json.dump(res, f, allow_nan=False, separators=(",", ":"))
    with open(os.path.join(cfg.out, "kv_stats.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"wrote {os.path.join(cfg.out, 'kv_stats.json')} and kv_stats.txt", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
