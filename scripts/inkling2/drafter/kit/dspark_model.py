"""PyTorch copy of the fork's Inkling 2 DSpark draft graph (src/models/dflash.cpp, graph_inkling2), for training.

It reads every tensor from the draft GGUF the server loads and writes the trained ones back into a copy of that file,
so nothing about the draft is converted on the way in or out. What the server does to the numbers is done here too:

  weights   a matrix the file stores in BF16 is used as its BF16 value. The trainer keeps an f32 master of it, because
            a learning-rate-sized step is far smaller than a BF16 step and would be rounded away.
  matmul    the engine multiplies a BF16 matrix by its input cast to BF16, summing in f32 (cuBLAS and mul_mat_f), or
            by the f32 input when the product is one column or small (mul_mat_vec_f). round_x picks which.
  KV cache  the keys and values of a layer, the ones made from the target's states and the block's own, pass through
            the cache type the server gives that layer: f16 for a sliding-window layer, Q8_0 for a global one as the
            launcher sets them, and a Q8_0 cache of a head size that is a multiple of 64 holds them rotated by a
            Hadamard matrix (llama-kv-cache.cpp). Rounding passes the gradient straight through.

The graph, as the fork builds it:
  inject    for draft layer il: context = rms(sum_t softmax(fuse[il])_t * state_t) * dflare_norm,
            k = rope(rms(k_ctx(context)) * k_ctx_norm), v = v_ctx(context); f_rnn = the same mix of row n_layer with
            dspark_norm.
  block     [anchor, mask x (block-1)] at positions n..n+block-1; the anchor's embedding is rms-normed with the
            target's epsilon and token_embd_norm, a mask is mask_embd as stored. Per layer: pre-norm attention (q and k
            rms-normed per head, rope, scale 1/sqrt(head)), residual, pre-norm SiLU feed-forward, residual. Every block
            row sees the whole block and the context the anchor sees: positions n-W..n-1 for a sliding-window layer,
            every position below n for a global one.
  head      row i >= 1: joint = [state, markov_w1[prev], hidden_i, f_rnn(n-1)] -> joint_proj; gate, candidate and
            latent are its three chunks; state = cand + gate*(state - cand); logits_i = cap*tanh(W(hidden_i +
            hidden_out(tanh(latent)))/cap). prev is the anchor, then the first maximum of each row.
"""
from __future__ import annotations

import dataclasses
import math
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F

_gguf = None


def import_gguf(gguf_py: str = ""):
    """The gguf package: the engine source's own when its folder is given, the installed one otherwise."""
    global _gguf
    if _gguf is None:
        if gguf_py:
            sys.path.insert(0, gguf_py)
        import gguf
        import gguf.quants  # noqa: F401
        _gguf = gguf
    return _gguf


@dataclasses.dataclass
class DSparkConfig:
    n_layer: int
    n_embd: int
    n_head: int
    n_head_kv: int
    head_dim: int
    n_ff: int
    rope_theta: float
    rms_eps: float
    vocab_padded: int
    vocab_unpadded: int
    block_size: int
    markov_rank: int
    target_layers: tuple
    swa_pattern: tuple
    swa_window: int      # the file's value: a sliding-window layer sees this many positions before the anchor
    softcap: float
    tok_norm_eps: float  # the target's epsilon, which the anchor's embedding is normed with

    @property
    def n_target(self) -> int:
        return len(self.target_layers)


def field(reader, key: str, default=None):
    f = reader.fields.get(key)
    return default if f is None else f.contents()


def config_from_gguf(reader, tok_norm_eps=None) -> DSparkConfig:
    def need(key):
        value = field(reader, key)
        if value is None:
            raise SystemExit("the draft has no %s" % key)
        return value

    eps = float(need("dflash.attention.layer_norm_rms_epsilon"))
    return DSparkConfig(
        n_layer=int(need("dflash.block_count")),
        n_embd=int(need("dflash.embedding_length")),
        n_head=int(need("dflash.attention.head_count")),
        n_head_kv=int(need("dflash.attention.head_count_kv")),
        head_dim=int(need("dflash.attention.key_length")),
        n_ff=int(need("dflash.feed_forward_length")),
        rope_theta=float(need("dflash.rope.freq_base")),
        rms_eps=eps,
        vocab_padded=int(need("dflash.vocab_size")),
        vocab_unpadded=int(need("dflash.vocab_size_unpadded")),
        block_size=int(need("dflash.block_size")),
        markov_rank=int(need("dflash.dspark.markov_rank")),
        target_layers=tuple(int(x) for x in need("dflash.target_layers")),
        swa_pattern=tuple(bool(x) for x in need("dflash.attention.sliding_window_pattern")),
        swa_window=int(need("dflash.attention.sliding_window")),
        softcap=float(need("dflash.final_logit_softcapping")),
        tok_norm_eps=eps if tok_norm_eps is None else float(tok_norm_eps),
    )


def target_eps(gguf_py: str, path: str) -> float:
    """The epsilon of the target's norms, read from its GGUF."""
    reader = import_gguf(gguf_py).GGUFReader(path)
    arch = field(reader, "general.architecture")
    eps = field(reader, "%s.attention.layer_norm_rms_epsilon" % arch)
    if eps is None:
        raise SystemExit("%s has no %s.attention.layer_norm_rms_epsilon" % (path, arch))
    return float(eps)


TOP = {
    "fuse": "dspark_fuse.weight",
    "dspark_norm": "dspark_norm.weight",
    "output_norm": "output_norm.weight",
    "mask_embd": "mask_embd.weight",
    "markov_w1": "markov_w1.weight",
    "joint_proj": "joint_proj.weight",
    "joint_proj_b": "joint_proj.bias",
    "hidden_out": "dspark_hidden_out.weight",
    "hidden_out_b": "dspark_hidden_out.bias",
}
LAYER = {
    "attn_norm": "attn_norm", "ffn_norm": "ffn_norm", "dflare_norm": "dflare_norm",
    "wq": "attn_q", "wk": "attn_k", "wv": "attn_v", "wo": "attn_output",
    "k_ctx": "attn_k_ctx", "v_ctx": "attn_v_ctx",
    "q_norm": "attn_q_norm", "k_norm": "attn_k_norm", "k_ctx_norm": "attn_k_ctx_norm",
    "ffn_gate": "ffn_gate", "ffn_up": "ffn_up", "ffn_down": "ffn_down",
}
# the matrices the block and the injection multiply by; the Markov head's are in MARKOV
MATRICES = ("wq", "wk", "wv", "wo", "k_ctx", "v_ctx", "ffn_gate", "ffn_up", "ffn_down")
MARKOV = ("markov_w1", "joint_proj", "hidden_out")


def tensor_names(n_layer: int) -> dict:
    """Parameter name -> the name of its tensor in the GGUF."""
    names = dict(TOP)
    for i in range(n_layer):
        for p, g in LAYER.items():
            names["layers.%d.%s" % (i, p)] = "blk.%d.%s.weight" % (i, g)
    return names


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def rope(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    """NeoX rotation of x [..., heads, head_dim] at positions pos [...], one for each leading index of x."""
    hd = x.shape[-1]
    scale = torch.tensor(theta, dtype=torch.float32, device=x.device) ** (-2.0 / hd)
    inv = scale ** torch.arange(hd // 2, dtype=torch.float32, device=x.device)
    ang = pos.to(torch.float32)[..., None, None] * inv
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def ste_bf16(x: torch.Tensor) -> torch.Tensor:
    """x as its BF16 value, the gradient passing as if it were not rounded."""
    return x + (x.to(torch.bfloat16).to(torch.float32) - x).detach()


def q8_0(x: torch.Tensor) -> torch.Tensor:
    """x [..., n] as ggml stores it in Q8_0: blocks of 32 along the last axis, each an f16 step and 32 int8."""
    b = x.reshape(-1, 32)
    d = b.abs().amax(dim=1, keepdim=True) / 127.0
    inv = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    v = b * inv
    a = v.abs()
    whole = torch.floor(a)
    q = torch.sign(v) * (whole + torch.floor(2.0 * (a - whole)))   # roundf: halves away from zero
    return (q * d.half().float()).reshape(x.shape)


def hadamard(n: int, device) -> torch.Tensor:
    """The orthonormal Sylvester matrix ggml_gen_hadamard writes: symmetric, its own inverse."""
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    if h.shape[0] != n:
        raise SystemExit("a rotation of %d is not a power of two" % n)
    return (h / math.sqrt(n)).to(torch.float32).to(device)


def rotation_sizes(kind: str, head_dim: int) -> tuple:
    """The rotations a cache of this type takes for keys and values (llama_kv_cache's constructor); 0 is none."""
    if kind != "q8_0" or head_dim % 64 != 0:
        return 0, 0
    n_k = 64
    while head_dim % (2 * n_k) == 0:
        n_k *= 2
    return n_k, 64


class KVStore:
    """What a layer's cache does to the keys and values written into it."""

    def __init__(self, kind: str, rot_k: int, rot_v: int, device):
        if kind not in ("f32", "f16", "q8_0"):
            raise SystemExit("no cache type %s here (f32, f16, q8_0)" % kind)
        self.kind = kind
        self.rot_k, self.rot_v = rot_k, rot_v
        self.hk = hadamard(rot_k, device) if rot_k else None
        self.hv = hadamard(rot_v, device) if rot_v else None

    def describe(self) -> str:
        return self.kind + (" rot k%d v%d" % (self.rot_k, self.rot_v) if self.rot_k or self.rot_v else "")

    def _store(self, x: torch.Tensor, h) -> torch.Tensor:
        if self.kind == "f32":
            return x
        if self.kind == "f16":
            y = x.half().float()
        else:
            y = x
            if h is not None:
                y = (y.reshape(*x.shape[:-1], -1, h.shape[0]) @ h).reshape(x.shape)
            y = q8_0(y)
            if h is not None:
                y = (y.reshape(*x.shape[:-1], -1, h.shape[0]) @ h).reshape(x.shape)
        return x + (y - x).detach()

    def k(self, x: torch.Tensor) -> torch.Tensor:
        return self._store(x, self.hk)

    def v(self, x: torch.Tensor) -> torch.Tensor:
        return self._store(x, self.hv)


class RoundedMM(torch.autograd.Function):
    """x @ wr.T, where wr is the weight as the file stores it and w its f32 master, which takes the gradient."""

    @staticmethod
    def forward(ctx, x, w, wr):
        ctx.save_for_backward(x, wr)
        return x @ wr.T

    @staticmethod
    def backward(ctx, g):
        x, wr = ctx.saved_tensors
        gx = g @ wr if ctx.needs_input_grad[0] else None
        gw = None
        if ctx.needs_input_grad[1]:
            gw = g.reshape(-1, g.shape[-1]).T @ x.reshape(-1, x.shape[-1])
        return gx, gw, None


def attend(q: torch.Tensor, k_blk: torch.Tensor, v_blk: torch.Tensor, pieces: list, scale: float) -> torch.Tensor:
    """Attention of the block's queries over the context pieces and the block itself.

    q [B, Q, H, d]; k_blk, v_blk [G, B, Q, d]; query head h reads key head h // (H/G).
    A piece is (K, V, vis): K, V [G, R, d] shared by the anchors, or [G, B, W, d] one set an anchor; vis [B, R|W].
    Returns [B, Q, H*d].
    """
    B, Q, H, d = q.shape
    G = k_blk.shape[0]
    gqa = H // G
    qg = (q.view(B, Q, G, gqa, d).permute(2, 0, 1, 3, 4).reshape(G, B, Q * gqa, d) * scale).contiguous()
    scores = []
    for K, _, vis in pieces:
        if K.dim() == 3:
            s = torch.matmul(qg.view(G, B * Q * gqa, d), K.transpose(1, 2)).view(G, B, Q * gqa, K.shape[1])
        else:
            s = torch.matmul(qg, K.transpose(2, 3))
        scores.append(s.masked_fill_(~vis[None, :, None, :], float("-inf")))
    scores.append(torch.matmul(qg, k_blk.transpose(2, 3)))
    # the block is never masked, so the largest score is a number; exp in place keeps one tensor a piece
    top = scores[-1].detach().amax(-1, keepdim=True)
    for s in scores[:-1]:
        top = torch.maximum(top, s.detach().amax(-1, keepdim=True))
    out = None
    total = None
    for i, s in enumerate(scores):
        e = s.sub_(top).exp_()
        V = v_blk if i == len(pieces) else pieces[i][1]
        if V.dim() == 3:
            o = torch.matmul(e.view(G, B * Q * gqa, e.shape[-1]), V).view(G, B, Q * gqa, d)
        else:
            o = torch.matmul(e, V)
        z = e.sum(-1, keepdim=True)
        out = o if out is None else out + o
        total = z if total is None else total + z
    out = out / total
    return out.view(G, B, Q, gqa, d).permute(1, 2, 0, 3, 4).reshape(B, Q, H * d)


class Layer(torch.nn.Module):
    def __init__(self, c: DSparkConfig):
        super().__init__()

        def p(*shape):
            return torch.nn.Parameter(torch.zeros(*shape))

        self.attn_norm = p(c.n_embd)
        self.ffn_norm = p(c.n_embd)
        self.dflare_norm = p(c.n_embd)
        self.wq = p(c.n_head * c.head_dim, c.n_embd)
        self.wk = p(c.n_head_kv * c.head_dim, c.n_embd)
        self.wv = p(c.n_head_kv * c.head_dim, c.n_embd)
        self.wo = p(c.n_embd, c.n_head * c.head_dim)
        self.k_ctx = p(c.n_head_kv * c.head_dim, c.n_embd)
        self.v_ctx = p(c.n_head_kv * c.head_dim, c.n_embd)
        self.q_norm = p(c.head_dim)
        self.k_norm = p(c.head_dim)
        self.k_ctx_norm = p(c.head_dim)
        self.ffn_gate = p(c.n_ff, c.n_embd)
        self.ffn_up = p(c.n_ff, c.n_embd)
        self.ffn_down = p(c.n_embd, c.n_ff)


class DSpark(torch.nn.Module):
    def __init__(self, cfg: DSparkConfig):
        super().__init__()
        c = cfg
        self.cfg = cfg

        def p(*shape):
            return torch.nn.Parameter(torch.zeros(*shape))

        self.layers = torch.nn.ModuleList(Layer(c) for _ in range(c.n_layer))
        self.fuse = p(c.n_layer + 1, c.n_target)
        self.dspark_norm = p(c.n_embd)
        self.output_norm = p(c.n_embd)
        self.mask_embd = p(c.n_embd)
        self.markov_w1 = p(c.vocab_padded, c.markov_rank)
        self.joint_proj = p(3 * c.markov_rank, 2 * c.markov_rank + 2 * c.n_embd)
        self.joint_proj_b = p(3 * c.markov_rank)
        self.hidden_out = p(c.n_embd, c.markov_rank)
        self.hidden_out_b = p(c.n_embd)
        # the target's vocabulary tensors, as the server holds them: never trained
        self.register_buffer("tok_norm", torch.zeros(c.n_embd), persistent=False)
        self.register_buffer("head_w", torch.zeros(0, c.n_embd), persistent=False)
        self.names = tensor_names(c.n_layer)
        self.stored_bf16 = set()   # parameters the file stores in BF16
        self.round_x = True        # a block or injection product casts its input to BF16 too
        self.round_x_markov = False  # so does a product of the Markov head (the CPU backend's do; the GPU's small ones do not)
        self.kv = []               # a KVStore a layer
        self._wr = {}              # the BF16 value of each such matrix, in f32, as of the last refresh()
        self._embd = None          # (rows of token_embd.weight as stored, its ggml type)
        self._gguf = None

    # ---- loading and writing ----

    @classmethod
    def from_gguf(cls, path: str, device, gguf_py: str = "", vocab: str = "", tok_norm_eps=None,
                  kv_type: str = "q8_0", kv_type_swa: str = "f16", kv_rot: str = "auto",
                  round_x: bool = True) -> "DSpark":
        gguf = import_gguf(gguf_py)
        reader = gguf.GGUFReader(path)
        arch = field(reader, "general.architecture")
        if arch != "dflash":
            raise SystemExit("%s is %s, not a dflash draft" % (path, arch))
        cfg = config_from_gguf(reader, tok_norm_eps)
        model = cls(cfg)
        model._gguf = gguf
        model.round_x = round_x
        tensors = {t.name: t for t in reader.tensors}
        params = dict(model.named_parameters())
        missing = [g for g in model.names.values() if g not in tensors]
        if missing:
            raise SystemExit("%s has no %s" % (path, ", ".join(missing[:6])))
        model.to(device)
        with torch.no_grad():
            for pname, gname in model.names.items():
                t = tensors[gname]
                if t.tensor_type not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.BF16):
                    raise SystemExit("%s is %s: a trained tensor is F32 or BF16" % (gname, t.tensor_type.name))
                data = gguf.quants.dequantize(np.asarray(t.data), t.tensor_type)
                value = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))
                param = params[pname]
                # a vector may be stored as one row
                if [n for n in value.shape if n != 1] != [n for n in param.shape if n != 1]:
                    raise SystemExit("%s is %s in the file, %s in the graph" % (gname, tuple(value.shape), tuple(param.shape)))
                param.copy_(value.reshape(param.shape).to(device))
                if t.tensor_type == gguf.GGMLQuantizationType.BF16:
                    model.stored_bf16.add(pname)
                del data, value

            vreader = reader
            if "output.weight" not in tensors:
                if not vocab:
                    raise SystemExit("%s has no vocabulary tensors: name the file that has them with --vocab" % path)
                vreader = gguf.GGUFReader(vocab)
            vt = {t.name: t for t in vreader.tensors}
            for name in ("token_embd.weight", "token_embd_norm.weight", "output.weight"):
                if name not in vt:
                    raise SystemExit("no %s among the vocabulary tensors" % name)
            model.tok_norm = torch.from_numpy(np.array(vt["token_embd_norm.weight"].data, dtype=np.float32)).to(device)
            out = vt["output.weight"]
            if out.data.shape[0] < cfg.vocab_unpadded:
                raise SystemExit("output.weight has %d rows, the vocabulary %d" % (out.data.shape[0], cfg.vocab_unpadded))
            # the output matrix in whole rows, a chunk at a time, so the host never holds it in f32
            head = torch.empty(cfg.vocab_unpadded, cfg.n_embd, device=device)
            for r0 in range(0, cfg.vocab_unpadded, 8192):
                r1 = min(r0 + 8192, cfg.vocab_unpadded)
                rows = gguf.quants.dequantize(np.asarray(out.data[r0:r1]), out.tensor_type)
                head[r0:r1] = torch.from_numpy(np.ascontiguousarray(rows, dtype=np.float32)).to(device)
            model.head_w = head
            model._embd = (vt["token_embd.weight"].data, vt["token_embd.weight"].tensor_type, vreader)

        for il in range(cfg.n_layer):
            kind = kv_type_swa if cfg.swa_pattern[il] else kv_type
            rot_k, rot_v = rotation_sizes(kind, cfg.head_dim) if kv_rot == "auto" else (0, 0)
            if kv_rot not in ("auto", "off"):
                rot_k, rot_v = (int(x) for x in kv_rot.split(","))
                if kind != "q8_0":
                    rot_k = rot_v = 0
            model.kv.append(KVStore(kind, rot_k, rot_v, device))
        model.refresh()
        return model

    def describe(self) -> str:
        c = self.cfg
        kinds = ", ".join("%d %s %s" % (il, "window" if c.swa_pattern[il] else "global", st.describe())
                          for il, st in enumerate(self.kv))
        n = sum(p.numel() for p in self.parameters())
        return ("draft: %d layers, width %d, heads %d/%d x %d, block %d, taps %s, window %d, %.1fM parameters, "
                "%d stored in BF16, input cast to BF16 %s; caches: %s" % (
                    c.n_layer, c.n_embd, c.n_head, c.n_head_kv, c.head_dim, c.block_size, list(c.target_layers),
                    c.swa_window, n / 1e6, len(self.stored_bf16), "yes" if self.round_x else "no", kinds))

    @torch.no_grad()
    def refresh(self) -> None:
        """Take the BF16 value of every BF16 matrix again; called after each step of the optimizer."""
        for name, param in self.named_parameters():
            if name in self.stored_bf16 and name.rsplit(".", 1)[-1] in MATRICES:
                self._wr[name] = param.to(torch.bfloat16).to(torch.float32)

    def stored_value(self, name: str, param: torch.Tensor) -> torch.Tensor:
        """A parameter as the file would hold it."""
        if name in self.stored_bf16:
            return param.detach().to(torch.bfloat16)
        return param.detach().to(torch.float32)

    def export(self, source: str, path: str) -> int:
        """Write the parameters over the same tensors of a copy of the source draft. Returns the tensors written."""
        gguf = self._gguf
        shutil.copyfile(source, path)
        params = dict(self.named_parameters())
        for check in (False, True):
            reader = gguf.GGUFReader(path, "r" if check else "r+")
            tensors = {t.name: t for t in reader.tensors}
            for pname, gname in self.names.items():
                t = tensors[gname]
                value = self.stored_value(pname, params[pname]).cpu().contiguous()
                if t.tensor_type == gguf.GGMLQuantizationType.BF16:
                    raw = value.view(torch.int16).numpy().view(np.uint8).reshape(t.data.shape)
                else:
                    raw = value.numpy().reshape(t.data.shape)
                if check:
                    if not np.array_equal(np.asarray(t.data), raw):
                        raise SystemExit("%s: %s read back differs from what was written" % (path, gname))
                else:
                    t.data[...] = raw
            if not check:
                reader.data.flush()
            del reader, tensors
        return len(self.names)

    # ---- the graph ----

    def mm(self, x: torch.Tensor, name: str, w: torch.Tensor) -> torch.Tensor:
        if name not in self.stored_bf16:
            return x @ w.T
        if self.round_x:
            x = ste_bf16(x)
        return RoundedMM.apply(x, w, self._wr[name])

    def mm_markov(self, x: torch.Tensor, name: str, w: torch.Tensor) -> torch.Tensor:
        if name not in self.stored_bf16:
            return x @ w.T
        return (ste_bf16(x) if self.round_x_markov else x) @ ste_bf16(w).T

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """Rows of the target's token embedding, dequantized as the engine reads them. ids [n] -> [n, n_embd]."""
        data, kind, _ = self._embd
        host = ids.detach().cpu().numpy().astype(np.int64)
        uniq, inverse = np.unique(host, return_inverse=True)
        rows = self._gguf.quants.dequantize(np.ascontiguousarray(data[uniq]), kind)
        rows = torch.from_numpy(np.ascontiguousarray(rows, dtype=np.float32)).to(self.head_w.device)
        return rows[torch.from_numpy(inverse).to(self.head_w.device)]

    def mix(self, f: torch.Tensor, row: int, norm: torch.Tensor) -> torch.Tensor:
        """f [r, taps, n_embd] f32 -> [r, n_embd]."""
        w = torch.softmax(self.fuse[row], dim=-1)
        return rms_norm((f * w[None, :, None]).sum(1), norm, self.cfg.rms_eps)

    def inject_layer(self, il: int, f: torch.Tensor, pos: torch.Tensor) -> tuple:
        """The keys and values layer il's cache holds for target states f [r, taps, n_embd] at positions pos [r].

        Returns (k, v), each [G, r, head_dim].
        """
        c = self.cfg
        layer = self.layers[il]
        context = self.mix(f, il, layer.dflare_norm)
        k = self.mm(context, "layers.%d.k_ctx" % il, layer.k_ctx).view(-1, c.n_head_kv, c.head_dim)
        v = self.mm(context, "layers.%d.v_ctx" % il, layer.v_ctx).view(-1, c.n_head_kv, c.head_dim)
        k = rope(rms_norm(k, layer.k_ctx_norm, c.rms_eps), pos, c.rope_theta)
        store = self.kv[il]
        return store.k(k).permute(1, 0, 2), store.v(v).permute(1, 0, 2)

    def f_rnn(self, f: torch.Tensor) -> torch.Tensor:
        return self.mix(f, self.cfg.n_layer, self.dspark_norm)

    def block(self, emb: torch.Tensor, pos: torch.Tensor, pieces: list) -> torch.Tensor:
        """The block's hidden rows after the final norm.

        emb [B, n_embd]: the anchors' embedding rows (embed()); pos [B, block]: the rows' positions;
        pieces: for each layer, the context pieces attend() takes. Returns [B, block, n_embd].
        """
        c = self.cfg
        B, Q = pos.shape
        H, G, d = c.n_head, c.n_head_kv, c.head_dim
        anchors = rms_norm(emb, self.tok_norm, c.tok_norm_eps)
        x = torch.cat([anchors[:, None, :], self.mask_embd[None, None, :].expand(B, Q - 1, c.n_embd)], dim=1)
        scale = 1.0 / math.sqrt(d)
        for il, layer in enumerate(self.layers):
            name = "layers.%d." % il
            h = rms_norm(x, layer.attn_norm, c.rms_eps)
            q = self.mm(h, name + "wq", layer.wq).view(B, Q, H, d)
            k = self.mm(h, name + "wk", layer.wk).view(B, Q, G, d)
            v = self.mm(h, name + "wv", layer.wv).view(B, Q, G, d)
            q = rope(rms_norm(q, layer.q_norm, c.rms_eps), pos, c.rope_theta)
            k = rope(rms_norm(k, layer.k_norm, c.rms_eps), pos, c.rope_theta)
            store = self.kv[il]
            k = store.k(k).permute(2, 0, 1, 3)
            v = store.v(v).permute(2, 0, 1, 3)
            x = x + self.mm(attend(q, k, v, pieces[il], scale), name + "wo", layer.wo)
            h = rms_norm(x, layer.ffn_norm, c.rms_eps)
            gate = F.silu(self.mm(h, name + "ffn_gate", layer.ffn_gate))
            x = x + self.mm(gate * self.mm(h, name + "ffn_up", layer.ffn_up), name + "ffn_down", layer.ffn_down)
        return rms_norm(x, self.output_norm, c.rms_eps)

    def base_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """The output layer on hidden rows, before the Markov head and the soft cap."""
        return hidden @ self.head_w.T

    def head(self, hidden: torch.Tensor, ids: torch.Tensor, f_ctx: torch.Tensor, prev=None, n_rows=None) -> torch.Tensor:
        """Logits of draft rows 1..n_rows-1 over the unpadded vocabulary.

        hidden [B, block, n_embd]; ids [B] the anchors; f_ctx [B, n_embd] = f_rnn of the position before the anchor.
        prev [B, n_rows-2]: the token before each of rows 2.., when the chain is given; otherwise each row's first
        maximum is the next row's, as the server chains them. Returns [B, n_rows-1, vocab].
        """
        c = self.cfg
        r = c.markov_rank
        n_rows = c.block_size if n_rows is None else n_rows
        state = hidden.new_zeros(hidden.shape[0], r)
        cur = ids
        rows = []
        for i in range(1, n_rows):
            w1 = self.markov_w1[cur]
            if "markov_w1" in self.stored_bf16:
                w1 = ste_bf16(w1)
            joint = torch.cat([state, w1, hidden[:, i], f_ctx], dim=-1)
            joint = self.mm_markov(joint, "joint_proj", self.joint_proj) + self.joint_proj_b
            gate = torch.sigmoid(joint[:, :r])
            cand = torch.tanh(joint[:, r:2 * r])
            state = cand + gate * (state - cand)
            latent = torch.tanh(joint[:, 2 * r:])
            corr = self.mm_markov(latent, "hidden_out", self.hidden_out) + self.hidden_out_b
            logits = c.softcap * torch.tanh(((hidden[:, i] + corr) @ self.head_w.T) / c.softcap)
            rows.append(logits)
            if i + 1 < n_rows:
                cur = prev[:, i - 1] if prev is not None else logits.argmax(dim=-1)
        return torch.stack(rows, dim=1)
