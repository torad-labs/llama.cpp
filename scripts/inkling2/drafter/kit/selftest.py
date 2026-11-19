"""Checks of the draft trainer that need no GPU and no server, run before any card is paid for.

  selftest.py [--gguf-py DIR] [--work DIR]
      1. the PyTorch graph against what the engine computed for a small draft (parity/fixture-dump)
      2. the rounding the trainer copies from the engine (BF16, Q8_0, the Hadamard rotation) against gguf's own
      3. a small draft and a recording of a small made-up target, written here in the recording tool's format:
         the loader's pairing of anchors, tokens and distributions against the generator's own arrays; the batched
         attention and step against a plain one-anchor forward; the two-pass gradient against one graph; the
         acceptance count against a case worked by hand; a draft written and read back
      4. train_dspark.py on that recording, start to end: the loss falls, the target accepts more of the draft on
         the requests kept out, and the tuned draft loads
      5. a recording of prompts written as the target's own chat, every row of them kept: the anchors the loader finds
         in what the model wrote, the distributions the trainer makes for them from the recorded states, a step and
         its gradient at them, and train_dspark.py at them alone teaching the draft the target's own continuations

  selftest.py --parity DUMP --draft GGUF [--vocab GGUF] [--device cuda]
      only the first check, against another dump of dspark_parity (the real draft's, on a card that holds it)
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import dspark_model  # noqa: E402
from dspark_data import Batch, Context, Loaded, Recording  # noqa: E402
from dspark_model import DSpark  # noqa: E402
from dspark_step import Step  # noqa: E402

FAILED = []


def check(ok: bool, text: str) -> bool:
    print("%s  %s" % ("ok  " if ok else "FAIL", text), flush=True)
    if not ok:
        FAILED.append(text)
    return ok


# ---- 1. the engine's own numbers ----

def parity(gguf_py: str, draft: str, dump: str, vocab: str, device, strict: bool) -> bool:
    """The graph against a dump of dspark_parity: the engine on CPU, its caches F16, the context all before the anchor."""
    with open(os.path.join(dump, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    S, T, E, V, block = meta["n_ctx_rows"], meta["n_target"], meta["n_embd"], meta["n_vocab"], meta["block"]
    feats = torch.from_numpy(np.fromfile(os.path.join(dump, "features.bin"), dtype=np.float32).reshape(S, T, E)).to(device)
    f_ref = torch.from_numpy(np.fromfile(os.path.join(dump, "f_rnn.bin"), dtype=np.float32).reshape(S, E)).to(device)
    ref = torch.from_numpy(np.fromfile(os.path.join(dump, "logits.bin"), dtype=np.float32).reshape(block, V)).to(device)
    model = DSpark.from_gguf(draft, device, gguf_py, vocab, None, "f16", "f16", "off", True)
    c = model.cfg
    if (c.n_target, c.n_embd, c.block_size) != (T, E, block) or c.vocab_unpadded != V:
        print("      %s is not the draft of this dump" % draft)
        return False
    print("      " + model.describe())
    ctx = Context(feats=feats, pos=torch.arange(S, device=device), c_last=S, n_gen=0)
    n = int(meta["anchor_pos"])
    one = torch.tensor([n], device=device)
    ids = torch.tensor([meta["anchor_id"]], device=device)
    chain = ref[1:].argmax(-1)
    best = None
    with torch.no_grad():
        f_rnn = model.f_rnn(feats)
        d_f = float((f_rnn - f_ref).abs().max() / f_ref.abs().max())
        for round_x in (True, False):
            model.round_x = model.round_x_markov = round_x
            batch = Batch(cut=one, a=torch.zeros_like(one), row0=one.clone(), ids=ids, prev=None)
            step = Step(model, ctx, batch, "none")
            pos = one[:, None] + torch.arange(block, device=device)[None, :]
            hidden = model.block(model.embed(ids), pos, step.pieces(0, 1))
            f_ctx = model.f_rnn(feats[n - 1:n])
            rows = torch.cat([model.base_logits(hidden[:, 0])[:, None, :],
                              model.head(hidden, ids, f_ctx, chain[None, :-1])], dim=1)[0]
            free = model.head(hidden, ids, f_ctx)[0].argmax(-1)
            span = ref.amax(-1) - ref.amin(-1)
            d = (rows - ref).abs().amax(-1) / span
            top2 = ref.topk(2, dim=-1).values
            # a row whose first two choices the engine holds closer than the difference may name either
            same = (rows.argmax(-1) == ref.argmax(-1)) | ((top2[:, 0] - top2[:, 1]) < 2 * (rows - ref).abs().amax(-1))
            r = {"round_x": round_x, "d": [float(x) for x in d], "same": int(same.sum()), "free": int((free == chain).sum())}
            print("      inputs cast to BF16 %-3s: rows differ by %s of their range; %d of %d rows name the engine's "
                  "token, the free chain %d of %d" % ("yes" if round_x else "no", " ".join("%.1e" % x for x in r["d"]),
                                                      r["same"], block, r["free"], block - 1))
            if best is None or max(r["d"]) < max(best["d"]):
                best = r
            if not model.stored_bf16:
                break
    tol_f, tol_d = (1e-4, 0.02) if strict else (0.05, 0.05)
    print("      f_rnn differs by %.1e of its largest value" % d_f)
    return d_f <= tol_f and max(best["d"]) <= tol_d and best["same"] >= (block if strict else block - 1)


def check_parity(gguf_py: str, device) -> None:
    dump = os.path.join(HERE, "parity", "fixture-dump")
    names = ("fixture-draft.gguf", "fixture-draft-paired.gguf")
    ok = False
    for name in names:
        print("      against the engine's dump, as %s" % name)
        if parity(gguf_py, os.path.join(HERE, "parity", name), dump, "", device, True):
            ok = True
            break
    check(ok, "the graph computes what the engine computed for the fixture draft")


# ---- 2. rounding ----

def check_rounding(gguf) -> None:
    rng = np.random.default_rng(11)
    x = (rng.standard_normal((64, 256)) * np.exp(rng.standard_normal((64, 1)) * 3)).astype(np.float32)
    x[3] = 0.0
    x[5, :32] = 0.0
    t = torch.from_numpy(x)
    kinds = gguf.GGMLQuantizationType
    theirs = gguf.quants.dequantize(gguf.quants.quantize(x, kinds.Q8_0), kinds.Q8_0)
    check(np.array_equal(dspark_model.q8_0(t).numpy(), np.asarray(theirs, dtype=np.float32)), "Q8_0 as gguf rounds it")
    theirs = gguf.quants.dequantize(gguf.quants.quantize(x, kinds.BF16), kinds.BF16)
    check(np.array_equal(t.to(torch.bfloat16).float().numpy(), np.asarray(theirs, dtype=np.float32)), "BF16 as gguf rounds it")
    check(np.array_equal(dspark_model.ste_bf16(t).numpy(), t.to(torch.bfloat16).float().numpy()), "the straight-through BF16 is the BF16 value")
    for n in (64, 128):
        h = dspark_model.hadamard(n, "cpu")
        check(bool(torch.allclose(h @ h, torch.eye(n), atol=1e-6)) and bool(torch.equal(h, h.T))
              and bool(torch.allclose(h[0], torch.full((n,), n ** -0.5))), "the rotation of %d is symmetric and its own inverse" % n)
    check(dspark_model.rotation_sizes("q8_0", 128) == (128, 64) and dspark_model.rotation_sizes("q8_0", 64) == (64, 64)
          and dspark_model.rotation_sizes("q8_0", 32) == (0, 0) and dspark_model.rotation_sizes("f16", 128) == (0, 0),
          "rotation sizes: 128 and 64 for a Q8_0 cache of head 128, none for F16 or a head of 32")
    store = dspark_model.KVStore("q8_0", 128, 64, "cpu")
    k = torch.from_numpy(rng.standard_normal((5, 2, 128)).astype(np.float32))
    h = dspark_model.hadamard(128, "cpu")
    want = dspark_model.q8_0(k @ h) @ h
    check(bool(torch.allclose(store.k(k), want, atol=1e-6)), "a Q8_0 cache stores a key rotated and gives it back rotated again")
    h = dspark_model.hadamard(64, "cpu")
    want = (dspark_model.q8_0(k.view(5, 2, 2, 64) @ h) @ h).view(5, 2, 128)
    check(bool(torch.allclose(store.v(k), want, atol=1e-6)), "a value is rotated in halves of 64")
    xg = torch.randn(7, 12, requires_grad=True)
    w = torch.randn(5, 12, requires_grad=True)
    wr = w.detach().to(torch.bfloat16).float()
    y = dspark_model.RoundedMM.apply(xg, w, wr)
    g = torch.randn_like(y)
    gx, gw = torch.autograd.grad((y * g).sum(), (xg, w))
    y2 = xg @ (w + (wr - w).detach()).T
    hx, hw = torch.autograd.grad((y2 * g).sum(), (xg, w))
    check(bool(torch.allclose(y, y2)) and bool(torch.allclose(gx, hx, atol=1e-6)) and bool(torch.allclose(gw, hw, atol=1e-5)),
          "the product by a rounded matrix has the gradient of the straight-through one")


# ---- 3. a made-up target, its draft and its recordings ----

E, VOCAB, PADDED, RANK, BLOCK, WINDOW, TOPK, LATENT = 256, 96, 100, 8, 8, 16, 16, 6
HEADS, KV_HEADS, HEAD, FF = 4, 2, 128, 128
TAPS = (0, 3)
SHARED = 4
CAP = 30.0
TOK_MODEL, TOK_END, TOK_EOM, TOK_USER = 90, 91, 92, 93     # the made-up target's chat tokens, inside its vocabulary


class Target:
    """A target small enough to write by hand: its state is a 6-number memory of the tokens read, laid in E numbers,
    and its next-token logits are its output matrix on that state."""

    def __init__(self, rng):
        self.emb = rng.standard_normal((PADDED, E)).astype(np.float32)
        self.z = (rng.standard_normal((PADDED, LATENT)) * 1.2).astype(np.float32)
        q, _ = np.linalg.qr(rng.standard_normal((LATENT, LATENT)))
        self.turn = (0.6 * q).astype(np.float32)
        u, _ = np.linalg.qr(rng.standard_normal((E, LATENT)))
        self.lay = (u * 2.0).astype(np.float32)
        self.out = rng.standard_normal((PADDED, E)).astype(np.float32)
        self.rng = rng

    def read(self, memory: np.ndarray, token: int):
        """The memory after one more token, and the two rows the recording keeps of that position."""
        memory = np.tanh(self.z[token] + memory @ self.turn)
        state = (self.lay @ memory).astype(np.float32)
        return memory, np.stack([0.2 * self.emb[token], state])

    def logits(self, state: np.ndarray) -> np.ndarray:
        """Over the unpadded vocabulary, from a state as the recording holds it (BF16)."""
        return (CAP * np.tanh((self.out[:VOCAB] @ state) / CAP)).astype(np.float32)


def as_bf16(x: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(torch.bfloat16).view(torch.int16).numpy()


def from_bf16(x: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.ascontiguousarray(x)).view(torch.bfloat16).float().numpy()


def write_draft(gguf, target: Target, path: str, rng) -> None:
    """A draft shaped as the real one, in small: BF16 matrices, a head of 128, a window layer and a global one."""
    kinds = gguf.GGMLQuantizationType
    w = gguf.GGUFWriter(path, "dflash")
    for key, value in (("context_length", 512), ("embedding_length", E), ("feed_forward_length", FF), ("block_count", 2),
                       ("attention.head_count", HEADS), ("attention.head_count_kv", KV_HEADS),
                       ("attention.key_length", HEAD), ("attention.value_length", HEAD), ("rope.dimension_count", HEAD),
                       ("vocab_size", PADDED), ("vocab_size_unpadded", VOCAB), ("block_size", BLOCK),
                       ("dspark.markov_rank", RANK), ("attention.sliding_window", WINDOW)):
        w.add_uint32("dflash." + key, value)
    w.add_float32("dflash.rope.freq_base", 10000.0)
    w.add_float32("dflash.attention.layer_norm_rms_epsilon", 1e-6)
    w.add_float32("dflash.final_logit_softcapping", CAP)
    w.add_array("dflash.target_layers", list(TAPS))
    w.add_array("dflash.attention.sliding_window_pattern", [True, False])

    def f32(name, value):
        w.add_tensor(name, np.ascontiguousarray(value, dtype=np.float32))

    def bf16(name, rows, cols, scale=None):
        value = (rng.standard_normal((rows, cols)) * (scale or cols ** -0.5)).astype(np.float32)
        w.add_tensor(name, gguf.quants.quantize(value, kinds.BF16), raw_dtype=kinds.BF16)

    f32("token_embd.weight", target.emb)
    f32("token_embd_norm.weight", np.ones(E))
    f32("output.weight", target.out)
    f32("dspark_fuse.weight", np.zeros((3, len(TAPS))))
    f32("dspark_norm.weight", np.ones(E))
    f32("output_norm.weight", np.ones(E))
    f32("mask_embd.weight", rng.standard_normal(E) * 0.5)
    bf16("markov_w1.weight", PADDED, RANK, 0.3)
    bf16("joint_proj.weight", 3 * RANK, 2 * RANK + 2 * E)
    f32("joint_proj.bias", np.zeros(3 * RANK))
    bf16("dspark_hidden_out.weight", E, RANK, 0.3)
    f32("dspark_hidden_out.bias", np.zeros(E))
    for i in range(2):
        b = "blk.%d." % i
        for name, size in (("attn_norm", E), ("ffn_norm", E), ("dflare_norm", E), ("attn_q_norm", HEAD),
                           ("attn_k_norm", HEAD), ("attn_k_ctx_norm", HEAD)):
            f32(b + name + ".weight", np.ones(size))
        for name in ("attn_k", "attn_v", "attn_k_ctx", "attn_v_ctx"):
            bf16(b + name + ".weight", KV_HEADS * HEAD, E)
        bf16(b + "attn_q.weight", HEADS * HEAD, E)
        bf16(b + "attn_output.weight", E, HEADS * HEAD)
        bf16(b + "ffn_gate.weight", FF, E)
        bf16(b + "ffn_up.weight", FF, E)
        bf16(b + "ffn_down.weight", E, FF)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def marked_prompt(target: Target, prefix: list, turns: int, tail: str) -> tuple:
    """A prompt in the made-up target's own chat: the shared prefix, then turns of a user's message and the model's
    messages after it, the model's words drawn from the target itself as it read the prompt so far.
    tail: "turn" ends it there, "user" adds a user's message, "open" adds one and a message the model has not ended.
    Returns the tokens, the (first, last) position of each run the model wrote, and the positions after a user's
    message, where a continuation may be cut."""
    rng = target.rng
    tokens, runs, ends = [], [], []
    memory = np.zeros(LATENT, dtype=np.float32)
    state = None

    def put(token) -> None:
        nonlocal memory, state
        memory, row = target.read(memory, int(token))
        state = from_bf16(as_bf16(row[-1]))
        tokens.append(int(token))

    def word() -> int:
        logits = target.logits(state)[:TOK_MODEL].astype(np.float64)
        p = np.exp(logits - logits.max())
        return int(rng.choice(TOK_MODEL, p=p / p.sum()))

    def user() -> None:
        put(TOK_USER)
        for t in rng.integers(0, TOK_MODEL, int(rng.integers(1, 9))):
            put(t)
        put(TOK_EOM)
        ends.append(len(tokens))

    for t in prefix:
        put(t)
    for k in range(turns):
        user()
        if k % 3 == 2:
            put(TOK_END)            # an end of turn after the user's message: nothing the model wrote
        first = len(tokens)
        for _ in range(int(rng.integers(1, 4))):
            put(TOK_MODEL)
            for _ in range(int(rng.integers(0, 12))):
                put(word())
            put(TOK_EOM)
        if k % 4 != 3:
            put(TOK_END)            # the model ends its turn, but for one turn in four that stops at its message
        runs.append((first, len(tokens) - 1))
    if tail != "turn":
        user()
    if tail == "open":
        first = len(tokens)
        put(TOK_MODEL)
        for _ in range(int(rng.integers(2, 9))):
            put(word())
        runs.append((first, len(tokens) - 1))
    return tokens, runs, ends


def marked(target: Target, prefix: list, count: int) -> tuple:
    """`count` such prompts with up to two continuations each, and one with none: the shapes and prompts
    write_recording takes, and the runs the model wrote in each."""
    rng = target.rng
    shapes, prompts, spans = [], [], []
    for number in range(count):
        tokens, runs, ends = marked_prompt(target, prefix, int(rng.integers(3, 7)), ("user", "turn", "open")[number % 3])
        cuts = [] if number == 4 else sorted(int(c) for c in rng.choice(ends, size=min(2, len(ends)), replace=False))
        shapes.append((len(tokens), [(cut, int(rng.integers(20, 50))) for cut in cuts]))
        prompts.append(tokens)
        spans.append(runs)
    return shapes, prompts, spans


def write_recording(target: Target, path: str, shapes: list, prefix: list, prompts=None) -> list:
    """A recording in dspark_tool's layout. shapes: for each request (prompt length, [(cut, states)]).
    prompts: the tokens of each request, in place of made-up ones; the recording then keeps every row of them and
    names the chat's tokens, as the recorder does. Returns what was generated, for the loader to be checked against."""
    rng = target.rng
    os.makedirs(os.path.join(path, "req"), exist_ok=True)
    dtype = np.dtype([("lse", "<f4"), ("ids", "<i4", (TOPK,)), ("logits", "<f4", (TOPK,))])
    meta = {"taps": list(TAPS), "n_embd": E, "n_vocab": VOCAB, "dtype": "bf16", "row_bytes": len(TAPS) * E * 2,
            "topk": TOPK, "topk_bytes": dtype.itemsize}
    if prompts is not None:
        meta.update(tok_model=TOK_MODEL, tok_end=TOK_END)
    with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)

    def top(state_bf16: np.ndarray):
        logits = target.logits(from_bf16(state_bf16))
        order = np.argsort(-logits, kind="stable")[:TOPK]
        peak = float(logits.max())
        lse = peak + float(np.log(np.exp(logits.astype(np.float64) - peak).sum()))
        p = np.exp(logits.astype(np.float64) - lse)
        return (lse, order, logits[order]), int(rng.choice(VOCAB, p=p / p.sum())), logits

    truth = []
    with open(os.path.join(path, "index.jsonl"), "w", encoding="utf-8") as index:
        for number, (n_tokens, cuts) in enumerate(shapes):
            if prompts is None:
                tokens = np.array(prefix + [int(t) for t in rng.integers(0, VOCAB, n_tokens - len(prefix))], dtype=np.int32)
            else:
                tokens = np.array(prompts[number], dtype=np.int32)
            memory = np.zeros(LATENT, dtype=np.float32)
            memories, rows = [], []
            for t in tokens:
                memory, row = target.read(memory, int(t))
                memories.append(memory)
                rows.append(as_bf16(row))
            rows = np.stack(rows)                               # [n_tokens, taps, E] BF16 as int16
            name = "r%03d" % number
            base = os.path.join(path, "req", name)
            if number == 0:
                rows[:SHARED].tofile(os.path.join(path, "prefix.states"))
                tokens[:SHARED].tofile(os.path.join(path, "prefix.tokens"))
            last = int(n_tokens) if prompts is not None else max([cut for cut, _ in cuts], default=SHARED)
            rows[SHARED:last].tofile(base + ".states")
            tokens.tofile(base + ".tokens")
            segments, gen, topk, kept = [], [], [], []
            for cut, n_states in cuts:
                memory = memories[cut - 1]
                row, token, logits = top(rows[cut - 1, -1])
                seg_tokens, seg_rows, seg_logits = [token], [], [logits]
                topk.append(row)
                for _ in range(n_states):
                    memory, state = target.read(memory, token)
                    state = as_bf16(state)
                    seg_rows.append(state)
                    row, token, logits = top(state[-1])
                    topk.append(row)
                    seg_tokens.append(token)
                    seg_logits.append(logits)
                segments.append({"cut": cut, "tokens": seg_tokens, "n_states": n_states, "states_row0": len(gen),
                                 "topk_row0": len(topk) - n_states - 1, "stop": "length", "g_max": n_states})
                gen.extend(seg_rows)
                kept.append({"cut": cut, "tokens": seg_tokens, "states": seg_rows, "logits": seg_logits})
            with open(base + ".gen.states", "wb") as f:
                for state in gen:
                    state.tofile(f)
            stored = np.zeros(len(topk), dtype=dtype)
            for k, (lse, ids, logits) in enumerate(topk):
                stored["lse"][k], stored["ids"][k], stored["logits"][k] = lse, ids, logits
            stored.tofile(base + ".gen.topk")
            index.write(json.dumps({"id": "syn-%d" % number, "file": name, "n_tokens": int(n_tokens), "shared_len": SHARED,
                                    "states_rows": last - SHARED, "gen_rows": len(gen), "topk_rows": len(topk),
                                    "segments": segments}) + "\n")
            truth.append({"tokens": tokens, "rows": rows, "segments": kept})
    return truth


def check_loader(rec: Recording, truth: list, target: Target, note: str = "") -> None:
    """What the loader hands the trainer for each anchor, against what the generator knows happened there."""
    ok_tokens = ok_states = ok_top = ok_valid = True
    anchors = 0
    for entry, known in zip(rec.entries, truth):
        req = rec.load(entry)
        loaded = Loaded(req, torch.device("cpu"))
        batch, x, rows, valid = loaded.batch(np.arange(req.n_anchors), BLOCK)
        feats = loaded.ctx.feats.view(torch.int16).numpy()
        number = 0
        for seg in known["segments"]:
            n_states = len(seg["states"])
            cut = seg["cut"]
            if n_states:
                # the context an anchor may see: the prompt below the cut, then the continuation's states before it
                row0 = int(batch.row0[number])
                ok_states &= np.array_equal(feats[:cut], known["rows"][:cut])
                ok_states &= all(np.array_equal(feats[row0 + g], seg["states"][g]) for g in range(n_states))
                ok_states &= all(int(loaded.ctx.pos[row0 + g]) == cut + g for g in range(n_states))
            for a in range(n_states):
                ok_tokens &= int(batch.cut[number]) == cut and int(batch.a[number]) == a and int(batch.ids[number]) == seg["tokens"][a]
                ok_states &= int(batch.row0[number]) == row0
                for i in range(1, BLOCK):
                    there = a + i <= n_states
                    ok_valid &= bool(valid[number, i - 1]) == there
                    if not there:
                        continue
                    ok_tokens &= int(x[number, i - 1]) == seg["tokens"][a + i]
                    if i >= 2:
                        ok_tokens &= int(batch.prev[number, i - 2]) == seg["tokens"][a + i - 1]
                    # draft row i is taught the distribution the target drew token a+i from
                    logits = seg["logits"][a + i]
                    row = int(rows[number, i - 1])
                    ids = loaded.tk_ids[row].numpy()
                    logp = logits - (logits.max() + np.log(np.exp(logits - logits.max()).sum()))
                    ok_top &= int(ids[0]) == int(np.argmax(logits)) and np.allclose(loaded.tk_logp[row].numpy(), logp[ids], atol=1e-4)
                number += 1
        anchors += number
        ok_tokens &= number == req.n_anchors
    check(ok_tokens, "the loader names each anchor's token, the tokens the target drew after it and the ones before each row (%d anchors)%s" % (anchors, note))
    check(ok_states, "an anchor's context rows are the prompt below its cut and its continuation's states before it" + note)
    check(ok_top, "draft row i is paired with the distribution the target drew its token from" + note)
    check(ok_valid, "a row past the end of a continuation is not counted" + note)


def check_prompt(rec: Recording, truth: list, spans: list, target: Target, model: DSpark) -> None:
    """The anchors the loader finds in the prompts, and what it and the trainer hand a step at each, against the
    generator's own arrays."""
    import train_dspark
    ok_found = ok_tokens = ok_valid = ok_top = True
    anchors = bare = 0
    for entry, known, runs in zip(rec.entries, truth, spans):
        want = [(p, last) for first, last in runs for p in range(first, last)]
        n, end = rec.model_rows(entry)
        ok_found &= [(int(a), int(b)) for a, b in zip(n, end)] == want
        req = rec.load(entry)
        ok_found &= req.n_prompt == len(want) and req.c_last == len(known["tokens"])
        bare += 0 if req.n_anchors else 1
        loaded = Loaded(req, torch.device("cpu"))
        batch, x, rows, valid = loaded.batch_prompt(np.arange(req.n_prompt), BLOCK)
        ids, logp = train_dspark.target_topk(model, loaded.ctx, rows, TOPK)
        tokens = known["tokens"]
        for number, (p, last) in enumerate(want):
            ok_tokens &= int(batch.cut[number]) == p and int(batch.a[number]) == 0 and int(batch.ids[number]) == int(tokens[p])
            for i in range(1, BLOCK):
                there = p + i <= last
                ok_valid &= bool(valid[number, i - 1]) == there
                if not there:
                    continue
                ok_tokens &= int(x[number, i - 1]) == int(tokens[p + i]) and int(rows[number, i - 1]) == p + i - 1
                if i >= 2:
                    ok_tokens &= int(batch.prev[number, i - 2]) == int(tokens[p + i - 1])
                # draft row i is taught the distribution the target holds for the token at p+i: from its state at p+i-1
                logits = target.logits(from_bf16(known["rows"][p + i - 1, -1]))
                want_logp = logits - (logits.max() + np.log(np.exp(logits - logits.max()).sum()))
                got = ids[number, i - 1].numpy()
                ok_top &= bool(want_logp[got[0]] >= want_logp.max() - 1e-4) and np.allclose(logp[number, i - 1].numpy(), want_logp[got], atol=1e-4)
        anchors += len(want)
    check(ok_found and anchors > 0, "the prompt's anchors are every token the model wrote that has another of its turn after it (%d anchors)" % anchors)
    check(bare > 0, "a request with no continuation still has the anchors of its prompt")
    check(ok_tokens, "at a prompt's anchor the loader names its token, the prompt's tokens after it and the ones before each row")
    check(ok_valid, "a row past the end of the model's turn is not counted")
    check(ok_top, "draft row i of a prompt's anchor is taught the target's distribution at the position before its token")


def check_attend() -> None:
    g = torch.Generator().manual_seed(5)
    B, Q, H, G, d, R, W = 3, 4, 4, 2, 8, 7, 5
    q = torch.randn(B, Q, H, d, generator=g)
    k_blk, v_blk = torch.randn(G, B, Q, d, generator=g), torch.randn(G, B, Q, d, generator=g)
    ks, vs = torch.randn(G, R, d, generator=g), torch.randn(G, R, d, generator=g)
    kw, vw = torch.randn(G, B, W, d, generator=g), torch.randn(G, B, W, d, generator=g)
    vis_s = torch.rand(B, R, generator=g) < 0.6
    vis_w = torch.rand(B, W, generator=g) < 0.6
    vis_s[1] = False
    vis_w[2] = False
    got = dspark_model.attend(q, k_blk, v_blk, [(ks, vs, vis_s), (kw, vw, vis_w)], 0.37)
    want = torch.zeros(B, Q, H, d)
    for b in range(B):
        for h in range(H):
            kv = h // (H // G)
            keys = torch.cat([ks[kv][vis_s[b]], kw[kv, b][vis_w[b]], k_blk[kv, b]])
            values = torch.cat([vs[kv][vis_s[b]], vw[kv, b][vis_w[b]], v_blk[kv, b]])
            want[b, :, h] = torch.softmax(q[b, :, h] @ keys.T * 0.37, -1) @ values
    check(bool(torch.allclose(got, want.reshape(B, Q, H * d), atol=1e-5)), "the batched attention is each anchor's own masked softmax")


def one_anchor(model: DSpark, loaded: Loaded, batch: Batch, number: int, emb: torch.Tensor) -> torch.Tensor:
    """The draft at one anchor, with nothing shared: its own context rows, each layer's keys made for it alone."""
    c = model.cfg
    cut, a, row0 = int(batch.cut[number]), int(batch.a[number]), int(batch.row0[number])
    n = cut + a
    rows = torch.cat([torch.arange(cut), row0 + torch.arange(a)])
    pos = loaded.ctx.pos[rows]
    pieces = []
    for il in range(c.n_layer):
        keep = rows[pos >= n - c.swa_window] if c.swa_pattern[il] else rows
        k, v = model.inject_layer(il, loaded.ctx.take(keep), loaded.ctx.pos[keep])
        pieces.append([(k, v, torch.ones(1, keep.numel(), dtype=torch.bool))])
    block_pos = n + torch.arange(c.block_size)[None, :]
    hidden = model.block(emb[number:number + 1], block_pos, pieces)
    before = rows[-1:]
    return model.head(hidden, batch.ids[number:number + 1], model.f_rnn(loaded.ctx.take(before)), batch.prev[number:number + 1])


@contextlib.contextmanager
def unrounded(model: DSpark):
    """The draft with nothing rounded on its way (caches in f32, inputs not cast to BF16).

    Rounding is a step function: two right computations that differ in the last bit before it can land a step apart
    after it. With it off, what is left between two ways of computing the same thing is the order of their sums, so
    the comparison is held to a tolerance a wrong row or mask cannot pass.
    """
    kv, round_x = model.kv, model.round_x
    model.kv = [dspark_model.KVStore("f32", 0, 0, torch.device("cpu")) for _ in kv]
    model.round_x = False
    try:
        yield
    finally:
        model.kv, model.round_x = kv, round_x


def step_difference(model: DSpark, rec: Recording, args, seen: set, prompt: bool = False) -> tuple:
    worst, span, each = 0.0, 1.0, []
    with torch.no_grad():
        for entry in rec.entries[:6]:
            req = rec.load(entry)
            count = req.n_prompt if prompt else req.n_anchors
            if not count:
                continue
            loaded = Loaded(req, torch.device("cpu"))
            batch, _, _, _ = loaded.batch_prompt(np.arange(count), BLOCK) if prompt else loaded.batch(np.arange(count), BLOCK)
            step = Step(model, loaded.ctx, batch, "none", args.cells, args.max_batch, args.chunk)
            emb = model.embed(batch.ids)
            got = torch.cat([step.logits(i, j, emb) for i, j in step.runs()])
            for number in range(0, count, 3):
                want = one_anchor(model, loaded, batch, number, emb)[0]
                each.append(float((got[number] - want).abs().max()))
                worst = max(worst, each[-1])
                span = max(span, float(want.max() - want.min()))
                cut, a = int(batch.cut[number]), int(batch.a[number])
                seen.add("start" if a == 0 else "inside")
                seen.add("short" if cut + a < WINDOW else "long")
                seen.add("window in the continuation" if a >= WINDOW else "window across the cut")
    return worst, span, float(np.median(each))


def check_step(model: DSpark, rec: Recording, args) -> None:
    seen = set()
    with unrounded(model):
        worst, span, _ = step_difference(model, rec, args, seen)
    check(worst <= 1e-4 * span, "with nothing rounded, a step's logits are each anchor's own forward (largest difference %.1e of a range of %.1f)" % (worst, span))
    worst, span, middle = step_difference(model, rec, args, seen)
    check(worst <= 2e-2 * span and middle <= 2e-3 * span,
          "as served, they differ by the rounding alone (largest difference %.1e, the middle one %.1e, of a range of %.1f)" % (worst, middle, span))
    check(len(seen) == 6, "anchors at a cut and inside, shorter than the window and longer, a window over the cut and past it: %s" % sorted(seen))


def check_step_prompt(model: DSpark, rec: Recording, args) -> None:
    seen = set()
    with unrounded(model):
        worst, span, _ = step_difference(model, rec, args, seen, True)
    check(worst <= 1e-4 * span and {"short", "long"} <= seen,
          "with nothing rounded, a step at the prompt's anchors is each anchor's own forward (largest difference %.1e of a range of %.1f)" % (worst, span))
    worst, span, middle = step_difference(model, rec, args, seen, True)
    check(worst <= 2e-2 * span and middle <= 2e-3 * span,
          "as served, it differs by the rounding alone (largest difference %.1e, the middle one %.1e, of a range of %.1f)" % (worst, middle, span))
    with unrounded(model):
        worst, span, _ = step_difference(model, rec, args, seen)
    check(worst <= 1e-4 * span, "a step at the continuations' anchors is each anchor's own forward too when the whole prompt is read "
          "(largest difference %.1e of a range of %.1f)" % (worst, span))


def gradient_difference(model: DSpark, loaded: Loaded, chosen: np.ndarray, args, w_rows: torch.Tensor, kind: str = "gen") -> tuple:
    import train_dspark
    grads = {}
    for mode in ("two", "one"):
        r = train_dspark.step_gradient(model, loaded, chosen, args, w_rows, mode, kind, TOPK)
        grads[mode] = {name: (None if p.grad is None else p.grad.clone()) for name, p in model.named_parameters()}
    silent = [name for name, g in grads["two"].items() if g is None or float(g.abs().max()) == 0.0]
    worst, where = 0.0, ""
    for name, g1 in grads["one"].items():
        g2 = grads["two"][name]
        if g1 is None or g2 is None:
            continue
        d = float((g1 - g2).abs().max() / g1.abs().max().clamp_min(1e-20))
        if d > worst:
            worst, where = d, name
    for p in model.parameters():
        p.grad = None
    return worst, where, silent, r["runs"]


def check_gradient(model: DSpark, rec: Recording, args, kind: str = "gen") -> None:
    import train_dspark
    req = rec.load(rec.entries[0])
    loaded = Loaded(req, torch.device("cpu"))
    w_rows = torch.tensor([1.0 if i < train_dspark.DEPTH else args.tail_weight for i in range(BLOCK - 1)])
    chosen = np.arange(req.n_anchors if kind == "gen" else req.n_prompt)[::2]
    at = "" if kind == "gen" else ", at the prompt's anchors"
    with unrounded(model):
        worst, where, _, runs = gradient_difference(model, loaded, chosen, args, w_rows, kind)
    check(worst < 1e-4, "with nothing rounded, the two-pass gradient is the one-graph gradient%s (%d runs; largest difference %.1e of a tensor's largest, in %s)" % (at, runs, worst, where))
    worst, where, silent, runs = gradient_difference(model, loaded, chosen, args, w_rows, kind)
    check(not silent, "every parameter of the draft takes a gradient%s%s" % (at, ": none for " + ", ".join(silent) if silent else ""))
    check(worst < 2e-2, "as served, the two differ by the rounding alone%s (largest difference %.1e, in %s)" % (at, worst, where))


def check_meter() -> None:
    import train_dspark
    # two anchors, three rows, a vocabulary of 5, the target's top 2 kept
    logits = torch.full((2, 3, 5), -5.0)
    for (b, r), token in {(0, 0): 1, (0, 1): 2, (0, 2): 3, (1, 0): 4, (1, 1): 0, (1, 2): 1}.items():
        logits[b, r, token] = 5.0
    tk_ids = torch.tensor([[[1, 0], [0, 2], [3, 4]], [[2, 4], [0, 1], [1, 3]]])
    p = torch.tensor([[[0.7, 0.2], [0.5, 0.3], [0.6, 0.1]], [[0.6, 0.3], [0.9, 0.05], [0.8, 0.1]]])
    x = torch.tensor([[1, 2, 0], [2, 0, 1]])
    meter = train_dspark.Meter(3)
    meter.add(logits, x, tk_ids, p.log(), torch.tensor([10, 20]))
    got = meter.result()["all"]
    # anchor 0: draft 1,2,3; drawn 1,2,0: rows alive 1,1,1 -> 0.7 + 0.3 + 0.6
    # anchor 1: draft 4,0,1; drawn 2,0,1: row 0 gets 0.3 and misses, so the rest never counts
    want_rows = [(0.7 + 0.3) / 2, 0.3 / 2, 0.6 / 2]
    # the ceiling names the first choice: anchor 0 -> 0.7, then drawn 1 == first 1 so 0.5, then drawn 2 != first 0 so stop
    # anchor 1 -> 0.6, drawn 2 == first 2 so 0.9, drawn 0 == first 0 so 0.8
    want_ceiling = (0.7 + 0.5 + 0.6 + 0.9 + 0.8) / 2
    check(np.allclose(got["row_accepted"], want_rows, atol=1e-6) and abs(got["ceiling_3"] - want_ceiling) < 1e-6
          and abs(got["counted_3"] - (2 + 0) / 2) < 1e-6,
          "the acceptance count is the hand-worked one (%.3f a round, ceiling %.3f)" % (got["accepted_3"], got["ceiling_3"]))


def check_export(model: DSpark, draft: str, work: str, gguf_py: str) -> None:
    path = os.path.join(work, "written.gguf")
    with torch.no_grad():
        kept = {name: p.clone() for name, p in model.named_parameters()}
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.01)
        model.refresh()
        count = model.export(draft, path)
        again = DSpark.from_gguf(path, torch.device("cpu"), gguf_py, "", 1e-5)
        mine = dict(model.named_parameters())
        same = all(torch.equal(p, model.stored_value(name, mine[name]).float()) for name, p in again.named_parameters())
        moved = sum(1 for name, p in again.named_parameters() if not torch.equal(p, kept[name].to(torch.bfloat16).float()
                                                                                 if name in model.stored_bf16 else kept[name]))
        untouched = torch.equal(again.head_w, model.head_w) and torch.equal(again.tok_norm, model.tok_norm)
        for name, p in model.named_parameters():
            p.copy_(kept[name])
        model.refresh()
    check(same and moved == count and untouched and os.path.getsize(path) == os.path.getsize(draft),
          "a written draft reads back as the %d tensors it was given, the target's own tensors as they were" % count)


def check_training(work: str, draft: str, train: str, held: str, gguf_py: str) -> None:
    out = os.path.join(work, "out")
    command = [sys.executable, "-I", os.path.join(HERE, "train_dspark.py"), "--draft", draft, "--tok-norm-eps", "1e-5",
               "--train", train, "--eval", held, "--out", out, "--epochs", "24", "--anchors", "64", "--lr", "2e-3",
               "--warmup", "0.1", "--eval-anchors", "64", "--device", "cpu", "--log-every", "50", "--cells", "6000",
               "--max-batch", "32", "--chunk", "64", "--save-every", "100", "--eval-every", "200"]
    if gguf_py:
        command += ["--gguf-py", gguf_py]
    done = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print("".join("      | " + line + "\n" for line in done.stdout.splitlines()[-34:]), end="")
    if not check(done.returncode == 0, "train_dspark.py runs to its end on the made-up recording"):
        return
    with open(os.path.join(out, "summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    check(s["loss_last"] < 0.85 * s["loss_first"], "the loss falls: %.3f over the first steps, %.3f over the last (%d steps)" % (
        s["loss_first"], s["loss_last"], s["steps"]))
    key = "accepted_%d" % 3
    check(s["after"][key] > s["before"][key] + 0.05, "on requests kept out, the target accepts %.3f of a round before and %.3f after, "
          "of a ceiling of %.3f" % (s["before"][key], s["after"][key], s["after"]["ceiling_3"]))
    check(s["after"][key] <= s["after"]["ceiling_3"] + 1e-6 and s["before"]["accepted_7"] <= s["before"]["ceiling_7"] + 1e-6,
          "no draft is counted above the ceiling")
    for name in ("eval-before.json", "eval-after.json", "train.jsonl", "draft-tuned.gguf"):
        check(os.path.isfile(os.path.join(out, name)), "it leaves " + name)
    kept = sorted(f for f in os.listdir(out) if f.startswith("ckpt-"))
    check(len(kept) == 2, "it keeps the last two sets of parameters (%s)" % ", ".join(kept))
    tuned = DSpark.from_gguf(os.path.join(out, "draft-tuned.gguf"), torch.device("cpu"), gguf_py, "", 1e-5)
    stock = DSpark.from_gguf(draft, torch.device("cpu"), gguf_py, "", 1e-5)
    last = torch.load(os.path.join(out, kept[-1]), map_location="cpu")
    moved = sum(1 for (name, p), (_, q) in zip(tuned.named_parameters(), stock.named_parameters()) if not torch.equal(p, q))
    check(moved == len(tuned.names) and torch.equal(tuned.head_w, stock.head_w),
          "the tuned draft loads, all %d of its tensors moved and the target's own did not" % moved)
    check(set(last) == set(dict(tuned.named_parameters())), "the kept parameters name every tensor of the draft")
    # going on from kept parameters, and evaluating alone
    command = [sys.executable, "-I", os.path.join(HERE, "train_dspark.py"), "--draft", draft, "--tok-norm-eps", "1e-5",
               "--eval", held, "--out", os.path.join(work, "out-eval"), "--eval-only", "--init", os.path.join(out, kept[-1]),
               "--device", "cpu", "--eval-anchors", "64"]
    if gguf_py:
        command += ["--gguf-py", gguf_py]
    done = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    ok = done.returncode == 0
    if ok:
        with open(os.path.join(work, "out-eval", "summary.json"), encoding="utf-8") as f:
            again = json.load(f)
        ok = again["before"][key] > s["before"][key] + 0.05
    else:
        print("".join("      | " + line + "\n" for line in done.stdout.splitlines()[-12:]), end="")
    check(ok, "an evaluation alone, from kept parameters, runs and scores above the stock draft")


def check_training_prompt(work: str, draft: str, train: str, held: str, gguf_py: str) -> None:
    """train_dspark.py at the prompts' anchors: beside the continuations' for a few steps, then alone, where what the
    draft learns of the target's own continuations can only have come from the prompts."""
    base = [sys.executable, "-I", os.path.join(HERE, "train_dspark.py"), "--draft", draft, "--tok-norm-eps", "1e-5",
            "--train", train, "--eval", held, "--prompt-anchors", "--prompt-eom", str(TOK_EOM), "--anchors", "64",
            "--lr", "2e-3", "--warmup", "0.1", "--eval-anchors", "64", "--device", "cpu", "--log-every", "50",
            "--cells", "6000", "--max-batch", "32", "--chunk", "64", "--save-every", "100", "--eval-every", "200"]
    if gguf_py:
        base += ["--gguf-py", gguf_py]

    def run(out: str, more: list, lines: int):
        done = subprocess.run(base + ["--out", out] + more, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print("".join("      | " + line + "\n" for line in done.stdout.splitlines()[-lines:]), end="")
        if done.returncode != 0:
            return None, []
        with open(os.path.join(out, "summary.json"), encoding="utf-8") as f:
            summary = json.load(f)
        with open(os.path.join(out, "train.jsonl"), encoding="utf-8") as f:
            return summary, [r["kind"] for r in map(json.loads, f) if "kind" in r]

    s, kinds = run(os.path.join(work, "out-mix"), ["--epochs", "2", "--prompt-max", "48"], 12)
    if check(s is not None, "train_dspark.py runs to its end at the continuations' anchors and the prompts' together"):
        check(kinds.count("gen") > 0 and kinds.count("prompt") > 0 and s["prompt_anchors"] > 0,
              "its steps are of both kinds: %d at the continuations' anchors, %d at the prompts'" % (kinds.count("gen"), kinds.count("prompt")))
        pairing = s.get("prompt_pairing")
        check(pairing is not None and pairing[1] > max(pairing[0], pairing[2]) + 1.0,
              "its own check of the prompt's pairing ran: %s" % (pairing,))
    s, kinds = run(os.path.join(work, "out-prompt"), ["--epochs", "20", "--no-gen-anchors"], 30)
    if not check(s is not None, "train_dspark.py runs to its end at the prompts' anchors alone"):
        return
    check(set(kinds) == {"prompt"}, "every one of its %d steps is at a prompt's anchors" % len(kinds))
    check(s["loss_last"] < 0.85 * s["loss_first"], "the loss falls: %.3f over the first steps, %.3f over the last (%d steps)" % (
        s["loss_first"], s["loss_last"], s["steps"]))
    key = "accepted_%d" % 3
    check(s["after"][key] > s["before"][key] + 0.05, "taught at the prompts alone, the draft has %.3f of a round accepted in the target's own "
          "continuations on requests kept out, from %.3f before, of a ceiling of %.3f" % (s["after"][key], s["before"][key], s["after"]["ceiling_3"]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gguf-py", default="", help="the engine source's gguf-py folder (the installed gguf otherwise)")
    p.add_argument("--work", default="", help="where the made-up files go (a temporary folder otherwise)")
    p.add_argument("--parity", default="", help="a dump of dspark_parity to check the graph against, and nothing else")
    p.add_argument("--draft", default="", help="the draft that dump was made with")
    p.add_argument("--vocab", default="")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.manual_seed(7)
    gguf = dspark_model.import_gguf(args.gguf_py)
    device = torch.device(args.device)
    if args.parity:
        if not args.draft:
            p.error("--parity takes the --draft the dump was made with")
        ok = parity(args.gguf_py, args.draft, args.parity, args.vocab, device, False)
        check(ok, "the graph computes what the engine computed in %s" % args.parity)
        raise SystemExit(1 if FAILED else 0)

    print("torch %s, numpy %s, gguf from %s" % (torch.__version__, np.__version__, os.path.dirname(gguf.__file__)))
    check_parity(args.gguf_py, device)
    check_rounding(gguf)
    check_attend()
    check_meter()

    with tempfile.TemporaryDirectory(prefix="dspark-selftest-") as tmp:
        work = args.work or tmp
        os.makedirs(work, exist_ok=True)
        rng = np.random.default_rng(20261119)
        target = Target(rng)
        draft = os.path.join(work, "draft.gguf")
        write_draft(gguf, target, draft, rng)
        prefix = [int(t) for t in rng.integers(0, VOCAB, SHARED)]
        # prompts and cuts chosen so anchors fall before and past the window, at a cut and deep in a continuation
        def shapes(count: int) -> list:
            out = []
            for _ in range(count):
                n = int(rng.integers(50, 170))
                first = int(rng.integers(SHARED + 1, n // 2))
                out.append((n, [(first, int(rng.integers(30, 60))), (int(rng.integers(n // 2, n + 1)), int(rng.integers(30, 60)))]))
            return out

        train_shapes = [(40, [(9, 12), (22, 30), (40, 20)]), (90, [(30, 0), (55, 24), (88, 18)]), (64, [(12, 40), (60, 26)]),
                        (120, [(70, 30), (119, 22)]), (33, [(6, 3), (33, 28)]), (75, [(40, 25), (74, 30)])] + shapes(14)
        held_shapes = [(50, [(8, 20), (45, 30)]), (100, [(60, 30), (99, 25)]), (70, [(20, 35), (66, 0)])] + shapes(3)
        train = os.path.join(work, "a")
        held = os.path.join(work, "heldout")
        truth = write_recording(target, train, train_shapes, prefix)
        write_recording(target, held, held_shapes, prefix)
        chat = os.path.join(work, "chat")
        chat_shapes, chat_prompts, chat_spans = marked(target, prefix, 16)
        chat_truth = write_recording(target, chat, chat_shapes, prefix, chat_prompts)
        rec = Recording(train)
        check_loader(rec, truth, target)
        rec_chat = Recording(chat)
        rec_chat.use_prompt(TOK_EOM)
        check_loader(rec_chat, chat_truth, target, ", in a recording read with its whole prompts")

        import train_dspark
        targs = train_dspark.parse_args(["--draft", draft, "--out", work, "--train", train, "--cells", "400",
                                         "--max-batch", "5", "--chunk", "7", "--device", "cpu"])
        model = DSpark.from_gguf(draft, torch.device("cpu"), args.gguf_py, "", 1e-5)
        print("      " + model.describe())
        check(len(model.stored_bf16) == 21 and model.kv[0].describe() == "f16" and model.kv[1].describe() == "q8_0 rot k128 v64",
              "the made-up draft is read as the real one is: BF16 matrices, an F16 window cache, a rotated Q8_0 global one")
        paired, off, worst = train_dspark.check_alignment(model, rec, torch.device("cpu"))
        check(paired >= 0.95 and worst < 1e-3, "the trainer's own pairing check passes on a right recording (%.1f%%), and the "
              "distribution it makes from a recorded state is the recorded one (%.1e apart at most)" % (100 * paired, worst))
        check_step(model, rec, targs)
        check_gradient(model, rec, targs)
        check_prompt(rec_chat, chat_truth, chat_spans, target, model)
        check_step_prompt(model, rec_chat, targs)
        check_gradient(model, rec_chat, targs, "prompt")
        check_export(model, draft, work, args.gguf_py)
        check_training(work, draft, train, held, args.gguf_py)
        check_training_prompt(work, draft, chat, held, args.gguf_py)

    print()
    if FAILED:
        print("%d checks failed:" % len(FAILED))
        for text in FAILED:
            print("  " + text)
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
