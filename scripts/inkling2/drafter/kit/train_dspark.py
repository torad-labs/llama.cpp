"""Train the Inkling 2 DSpark draft on a recording of the served target, and say how much more of its drafts the
target would accept.

The draft is read from the GGUF the server loads and the trained one is written over a copy of it, so the result is
loaded by the launcher as the stock one is. Training sees what the server computes: BF16 weights, and each layer's
keys and values as its cache stores them (dspark_model.py).

A step takes --anchors places of one request where the server could have asked for a draft and runs the draft there
as the server would: the states the target had before that place as context, the token it had just written as the
anchor. Row i of the draft is taught the distribution the target drew its i-th next token from (the top 64 of it), with
the tokens the target did draw standing before it, which is what the row sees whenever the server gets to use it:
a row counts only when every row before it was accepted, and an accepted row is the token the target drew.

What is measured, before and after, on a recording kept out of training: the draft tokens the target would accept
a round, at the depth the launcher drafts (3) and at the whole block (7). A draft token is accepted when the target
draws it, so the chance is what the target's distribution gives the draft's first choice; "ceiling" is the same count
for a draft that always names the target's most likely token, the most any draft of that depth could get.

Usage:
  train_dspark.py --draft DRAFT.gguf --target TARGET.gguf --train REC [REC...] --eval REC --out DIR [--epochs N] [--lr X]
  train_dspark.py --draft DRAFT.gguf --target TARGET.gguf --eval REC --out DIR --eval-only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import dspark_model  # noqa: E402
from dspark_data import Loaded, Recording, stream, stream_pairs  # noqa: E402
from dspark_step import Step  # noqa: E402

BUCKETS = ((0, 32768, "under 32K"), (32768, 65536, "32K to 64K"), (65536, 131072, "64K to 128K"),
           (131072, 1 << 62, "128K and over"), (0, 1 << 62, "all"))
DEPTH = 3   # the launcher's draft depth
WALK_DEPTHS = (3, 5, 7)   # the depths a walk of the server's rounds is counted at
WALK_FLOORS = (0.3, 0.5, 0.7, 0.9)   # chances a draft gives its own first choice, under which a round would stop drafting


def say(text: str) -> None:
    print("%s %s" % (time.strftime("%H:%M:%S", time.gmtime()), text), flush=True)


def loss_rows(logits, tk_ids, tk_logp, weight, mode_weight: float):
    """logits [B, R, vocab]; tk_ids, tk_logp [B, R, k] the target's top k and their log-probabilities; weight [B, R].

    Returns the weighted sum of the loss, and of its two parts: the cross-entropy against the target's top k, and
    against its first choice alone.
    """
    lq = logits.gather(-1, tk_ids) - torch.logsumexp(logits, -1, keepdim=True)
    kd = -(tk_logp.exp() * lq).sum(-1)
    ce = -lq[..., 0]
    return ((kd + mode_weight * ce) * weight).sum(), (kd * weight).sum().detach(), (ce * weight).sum().detach()


@torch.no_grad()
def target_logp(model, ctx, rows: torch.Tensor) -> torch.Tensor:
    """The target's log-probabilities for the token after each context row of `rows` [N] -> [N, vocab]: its output
    state at the row, the recording's last tap, through its own output matrix and soft cap, as the engine makes its
    logits."""
    h = ctx.feats[rows.to(ctx.feats.device), -1].to(rows.device).float()
    cap = model.cfg.softcap
    logits = cap * torch.tanh((h @ model.head_w.T) / cap)
    return logits - torch.logsumexp(logits, -1, keepdim=True)


@torch.no_grad()
def target_topk(model, ctx, rows: torch.Tensor, k: int):
    """The target's first k choices for the token after each context row of `rows`, and their log-probabilities.
    rows [...] -> ids [..., k], log-probabilities [..., k]."""
    top = torch.topk(target_logp(model, ctx, rows.reshape(-1)), k, dim=-1)
    return top.indices.reshape(*rows.shape, k), top.values.reshape(*rows.shape, k)


def step_gradient(model, loaded: Loaded, chosen: np.ndarray, args, w_rows: torch.Tensor, grad: str = "two",
                  kind: str = "gen", topk: int = 64) -> dict:
    """Leave the gradient of one step in the parameters' .grad and return its numbers. kind: "gen" for anchors of
    the continuations, "prompt" for anchors of the prompt, whose distributions are made here `topk` wide."""
    block = model.cfg.block_size
    batch, _, rows, valid = loaded.batch(chosen, block) if kind == "gen" else loaded.batch_prompt(chosen, block)
    weight = valid.to(torch.float32) * w_rows[None, :]
    total = weight.sum()
    step = Step(model, loaded.ctx, batch, grad, args.cells, args.max_batch, args.chunk)
    emb = model.embed(batch.ids)
    for p in model.parameters():
        p.grad = None
    sums = [0.0, 0.0, 0.0]
    whole = None
    for i, j in step.runs():
        logits = step.logits(i, j, emb)
        if kind == "gen":
            tk_ids, tk_logp = loaded.tk_ids[rows[i:j]], loaded.tk_logp[rows[i:j]]
        else:
            tk_ids, tk_logp = target_topk(model, loaded.ctx, rows[i:j], topk)
        loss, kd, ce = loss_rows(logits, tk_ids, tk_logp, weight[i:j], args.mode_weight)
        if grad == "one":
            whole = loss if whole is None else whole + loss
        else:
            (loss / total).backward()
        for k, value in enumerate((loss, kd, ce)):
            sums[k] += float(value.detach())
    if whole is not None:
        (whole / total).backward()
    step.finish()
    total = float(total)
    return {"loss": sums[0] / total, "kd": sums[1] / total, "mode": sums[2] / total, "anchors": int(batch.cut.numel()),
            "context": int(step.c_max), "runs": len(step.runs())}


class Meter:
    """What the target would accept of the draft, summed over anchors and split by how long the context was."""

    def __init__(self, rows: int):
        self.rows = rows
        self.sums = {name: {"anchors": 0, "acc": np.zeros(rows), "hard": np.zeros(rows), "ceil": np.zeros(rows),
                            "mode": np.zeros(rows), "kd": np.zeros(rows)} for _, _, name in BUCKETS}

    @torch.no_grad()
    def add(self, logits, x, tk_ids, tk_logp, n) -> None:
        """logits [B, R, vocab] of anchors at positions n [B]; x [B, R] what the target drew; its top k for each row."""
        p = tk_logp.exp()
        one = p.new_ones(p.shape[0], 1)
        d = logits.argmax(-1)
        p_draft = (p * (tk_ids == d[..., None])).sum(-1)            # the chance the target draws the draft's token
        hit = (d == x).to(p.dtype)
        alive = torch.cat([one, hit[:, :-1].cumprod(1)], 1)         # every row before it was accepted, on this path
        first = tk_ids[..., 0]
        hit_first = (first == x).to(p.dtype)
        alive_first = torch.cat([one, hit_first[:, :-1].cumprod(1)], 1)
        values = {
            "acc": alive * p_draft,
            "hard": hit.cumprod(1),
            "ceil": alive_first * p[..., 0],
            "mode": (d == first).to(p.dtype),
            "kd": -(p * (logits.gather(-1, tk_ids) - torch.logsumexp(logits, -1, keepdim=True))).sum(-1),
        }
        for lo, hi, name in BUCKETS:
            m = (n >= lo) & (n < hi)
            count = int(m.sum())
            if count:
                s = self.sums[name]
                s["anchors"] += count
                for key, value in values.items():
                    s[key] += value[m].sum(0).double().cpu().numpy()

    def result(self) -> dict:
        out = {}
        for _, _, name in BUCKETS:
            s = self.sums[name]
            k = s["anchors"]
            if not k:
                continue
            out[name] = {
                "anchors": k,
                "accepted_%d" % DEPTH: float(s["acc"][:DEPTH].sum() / k),
                "accepted_%d" % self.rows: float(s["acc"].sum() / k),
                "ceiling_%d" % DEPTH: float(s["ceil"][:DEPTH].sum() / k),
                "ceiling_%d" % self.rows: float(s["ceil"].sum() / k),
                "counted_%d" % DEPTH: float(s["hard"][:DEPTH].sum() / k),
                "counted_%d" % self.rows: float(s["hard"].sum() / k),
                "row_accepted": [float(v) for v in s["acc"] / k],
                "row_names_first_choice": [float(v) for v in s["mode"] / k],
                "cross_entropy": float(s["kd"].mean() / k),
            }
        return out


def table(result: dict, rows: int) -> str:
    lines = ["%-14s %8s %12s %12s %12s %12s" % ("context", "anchors", "accepted/%d" % DEPTH, "ceiling/%d" % DEPTH,
                                                "accepted/%d" % rows, "ceiling/%d" % rows)]
    for _, _, name in BUCKETS:
        r = result.get(name)
        if r:
            lines.append("%-14s %8d %12.3f %12.3f %12.3f %12.3f" % (
                name, r["anchors"], r["accepted_%d" % DEPTH], r["ceiling_%d" % DEPTH],
                r["accepted_%d" % rows], r["ceiling_%d" % rows]))
    return "\n".join(lines)


def spread(count: int, most: int) -> np.ndarray:
    """`most` of range(count), evenly apart."""
    if not most or count <= most:
        return np.arange(count)
    return np.unique(np.linspace(0, count - 1, most).round().astype(np.int64))


def top_chance(logits: torch.Tensor, rows: int = 32) -> torch.Tensor:
    """The chance each draft row gives its first choice, a few anchors at a time so no second copy of the logits is held."""
    out = torch.empty(logits.shape[:-1], dtype=torch.float32, device=logits.device)
    for s in range(0, logits.shape[0], rows):
        part = logits[s:s + rows].float()
        out[s:s + rows] = (part.max(dim=-1).values - torch.logsumexp(part, dim=-1)).exp()
    return out


def with_continuations(rec: Recording) -> list:
    return sorted((e for e in rec.entries if e.get("gen_rows", 0) > 0), key=lambda e: (e["n_tokens"], e["id"]))


@torch.no_grad()
def evaluate(model, rec: Recording, args, device, n_requests: int, n_anchors: int) -> dict:
    """The same anchors every time it is asked the same way: requests evenly apart by length, anchors evenly apart in
    each, only those whose continuation goes on for the whole block."""
    block = model.cfg.block_size
    entries = with_continuations(rec)
    entries = [entries[i] for i in spread(len(entries), n_requests)]
    meter = Meter(block - 1)
    for req in stream(rec, entries):
        full = np.nonzero(req.an_a + (block - 1) <= req.an_nst)[0]
        if full.size == 0:
            continue
        chosen = full[spread(full.size, n_anchors)]
        loaded = Loaded(req, device, feats_device=args.feats_device or None)
        batch, x, rows, _ = loaded.batch(chosen, block)
        step = Step(model, loaded.ctx, batch, "none", args.cells, args.max_batch, args.chunk)
        emb = model.embed(batch.ids)
        for i, j in step.runs():
            logits = step.logits(i, j, emb)
            meter.add(logits, x[i:j], loaded.tk_ids[rows[i:j]], loaded.tk_logp[rows[i:j]], step.n[i:j])
        del loaded, step, batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return meter.result()


@torch.no_grad()
def overlap(logits: torch.Tensor, tk_ids: torch.Tensor, tk_logp: torch.Tensor, rows: int = 32):
    """What a draft row and the target's row share, the chance a token sampled from the draft is kept when the target
    checks it by the two chances: the sum over the target's top k of the smaller chance (low), and that plus the
    smaller of what each leaves outside the top k (high). Both [.., rows of the block]."""
    low = torch.empty(logits.shape[:-1], dtype=torch.float32, device=logits.device)
    high = torch.empty_like(low)
    for s in range(0, logits.shape[0], rows):
        part = logits[s:s + rows].float()
        lq = part.gather(-1, tk_ids[s:s + rows]) - torch.logsumexp(part, dim=-1, keepdim=True)
        p, q = tk_logp[s:s + rows].float().exp(), lq.exp()
        both = torch.minimum(p, q).sum(-1)
        low[s:s + rows] = both
        high[s:s + rows] = both + torch.minimum((1 - p.sum(-1)).clamp_min(0), (1 - q.sum(-1)).clamp_min(0))
    return low, high


@torch.no_grad()
def walk(model, rec: Recording, args, device, n_requests: int, most: int = 8192) -> dict:
    """What a server's rounds keep. A round drafts `depth` tokens behind the last one written and keeps those the
    target drew, up to the first it did not; the target's own token there is the last one written of the next round.
    So a round starts right behind every miss, and rounds stand more often where the draft is wrong than anchors
    evenly apart do. Every anchor of every continuation is run once, and the rounds are then walked on the host: for
    the draft, and for a draft that always names the target's first choice (the most a draft can keep).

    The draft's rounds are walked again for each floor of WALK_FLOORS: a round stops drafting at the first row whose
    first choice the draft itself gives less than the floor, so the target checks fewer rows it would not keep. Each
    row follows the draft's own first choices, as the server chains them, which is the given chain up to the first
    miss and what the server's round would see after it.

    Last, a draft that samples its rows and is checked by the two chances (a row is kept with the chance the two rows
    share), which the engine does not do for this draft: each row is given the target's own tokens before it, as a
    kept row has, and is kept or not against one fixed stream of numbers. "sampled" counts what the target's top k
    shows of the share, "sampled_most" adds what the rest could hold."""
    block = model.cfg.block_size
    depths = [d for d in WALK_DEPTHS if d < block]
    whos = ("draft", "first", "sampled", "sampled_most")
    rng = np.random.default_rng(args.seed)
    share_sum = np.zeros((2, block - 1), dtype=np.float64)
    share_n = np.zeros(block - 1, dtype=np.int64)
    sums = {name: {(who, d): [0, 0] for who in whos for d in depths} for _, _, name in BUCKETS}
    floors = {name: {(f, d): [0, 0, 0] for f in WALK_FLOORS for d in depths} for _, _, name in BUCKETS}
    each = []   # a line a continuation: its request, its cut, and [rounds, kept] for the draft and the first choice
    entries = with_continuations(rec)
    entries = [entries[i] for i in spread(len(entries), n_requests)]
    for req in stream(rec, entries):
        count = int(req.an_a.size)
        able = np.nonzero(req.an_a + min(depths) <= req.an_nst)[0]
        if able.size == 0:
            continue
        hits = np.zeros((len(whos), count, block - 1), dtype=np.int64)
        chance = np.zeros((count, block - 1), dtype=np.float32)
        share = np.zeros((2, count, block - 1), dtype=np.float32)
        loaded = Loaded(req, device, feats_device=args.feats_device or None)
        for s in range(0, able.size, most):
            chosen = able[s:s + most]
            batch, x, rows, valid = loaded.batch(chosen, block)
            batch.prev = None
            step = Step(model, loaded.ctx, batch, "none", args.cells, args.max_batch, args.chunk)
            emb = model.embed(batch.ids)
            for i, j in step.runs():
                hidden, ids, f_ctx = step.states(i, j, emb)
                logits = model.head(hidden, ids, f_ctx, None, None)
                d = logits.argmax(-1)
                chance[chosen[i:j]] = top_chance(logits).cpu().numpy()
                del logits
                first = loaded.tk_ids[rows[i:j]][..., 0]
                hits[0, chosen[i:j]] = ((d == x[i:j]) & valid[i:j]).cpu().numpy()
                hits[1, chosen[i:j]] = ((first == x[i:j]) & valid[i:j]).cpu().numpy()
                logits = model.head(hidden, ids, f_ctx, x[i:j, :-1], None)
                low, high = overlap(logits, loaded.tk_ids[rows[i:j]], loaded.tk_logp[rows[i:j]])
                del logits, hidden
                ok = valid[i:j].float()
                share[0, chosen[i:j]] = (low * ok).cpu().numpy()
                share[1, chosen[i:j]] = (high * ok).cpu().numpy()
                share_n += valid[i:j].sum(0).cpu().numpy()
                share_sum[0] += (low * ok).sum(0).double().cpu().numpy()
                share_sum[1] += (high * ok).sum(0).double().cpu().numpy()
            del step, batch, emb
        del loaded
        if device.type == "cuda":
            torch.cuda.empty_cache()
        draw = rng.random((count, block - 1), dtype=np.float32)
        hits[2] = draw < share[0]
        hits[3] = draw < share[1]
        lead = np.cumprod(hits, axis=2).cumsum(axis=2)      # [who, anchor, r]: rows kept of the first r + 1
        # [floor, anchor, r]: rows drafted of the first r + 1, stopping at the first one under the floor
        sure = np.stack([np.cumprod(chance >= f, axis=1).cumsum(axis=1) for f in WALK_FLOORS])
        start = 0
        while start < count:
            n, cut = int(req.an_nst[start]), int(req.an_cut[start])
            for fi, f in enumerate(WALK_FLOORS):
                for dep in depths:
                    a = 0
                    while a + dep <= n:
                        drafted = int(sure[fi, start + a, dep - 1])
                        kept = int(lead[0, start + a, drafted - 1]) if drafted else 0
                        for lo, hi, name in BUCKETS:
                            if lo <= cut + a < hi:
                                trio = floors[name][(f, dep)]
                                trio[0] += 1
                                trio[1] += kept
                                trio[2] += drafted
                        a += kept + 1
            line = {"id": req.id, "cut": cut, "anchors": n}
            for w, who in enumerate(whos):
                for dep in depths:
                    a = 0
                    own = [0, 0]
                    while a + dep <= n:
                        kept = int(lead[w, start + a, dep - 1])
                        own[0] += 1
                        own[1] += kept
                        for lo, hi, name in BUCKETS:
                            if lo <= cut + a < hi:
                                pair = sums[name][(who, dep)]
                                pair[0] += 1
                                pair[1] += kept
                        a += kept + 1
                    line["%s_%d" % (who, dep)] = own
            each.append(line)
            start += n
    out = {}
    for _, _, name in BUCKETS:
        if not sums[name][("draft", depths[0])][0]:
            continue
        out[name] = {}
        for dep in depths:
            rounds, kept = sums[name][("draft", dep)]
            f_rounds, f_kept = sums[name][("first", dep)]
            out[name]["rounds_%d" % dep] = rounds
            out[name]["walked_%d" % dep] = kept / max(1, rounds)
            out[name]["ceiling_rounds_%d" % dep] = f_rounds
            out[name]["walked_ceiling_%d" % dep] = f_kept / max(1, f_rounds)
            for who in ("sampled", "sampled_most"):
                w_rounds, w_kept = sums[name][(who, dep)]
                out[name]["walked_%s_%d" % (who, dep)] = w_kept / max(1, w_rounds)
        out[name]["floors"] = [
            {"depth": dep, "floor": f, "rounds": trio[0], "kept": trio[1] / max(1, trio[0]), "drafted": trio[2] / max(1, trio[0])}
            for dep in depths for f in WALK_FLOORS for trio in [floors[name][(f, dep)]]]
    out["each"] = each
    # the share of each draft row, over every anchor that has the row: [low, high] a row
    out["row_share"] = [[float(share_sum[k, r] / max(1, int(share_n[r]))) for k in range(2)] for r in range(block - 1)]
    return out


def walk_table(result: dict) -> str:
    depths = [d for d in WALK_DEPTHS if any("walked_%d" % d in r for r in result.values())]
    lines = ["%-14s %8s" % ("context", "rounds") + "".join(" %9s %10s" % ("kept/%d" % d, "ceiling/%d" % d) for d in depths)]
    for _, _, name in BUCKETS:
        r = result.get(name)
        if r:
            lines.append("%-14s %8d" % (name, r["rounds_%d" % depths[0]]) + "".join(
                " %9.3f %10.3f" % (r["walked_%d" % d], r["walked_ceiling_%d" % d]) for d in depths))
    if any("walked_sampled_%d" % depths[0] in r for r in result.values() if isinstance(r, dict)):
        lines.append("")
        lines.append("a draft that samples its rows, checked by the two chances (not in the engine for this draft): kept, "
                     "by what the target's top k shows and with what the rest could add")
        lines.append("%-14s" % "context" + "".join(" %9s %10s" % ("kept/%d" % d, "most/%d" % d) for d in depths))
        for _, _, name in BUCKETS:
            r = result.get(name)
            if r:
                lines.append("%-14s" % name + "".join(
                    " %9.3f %10.3f" % (r["walked_sampled_%d" % d], r["walked_sampled_most_%d" % d]) for d in depths))
    if result.get("row_share"):
        lines.append("what a draft row and the target's row share, row by row: " + " ".join(
            "%.3f-%.3f" % (lo, hi) for lo, hi in result["row_share"]))
    for _, _, name in BUCKETS:
        r = result.get(name)
        if not r or not r.get("floors"):
            continue
        lines.append("")
        lines.append("%s: a round that stops drafting at the first row its draft gives under a floor" % name)
        lines.append("%5s %6s %8s %8s %8s %8s %8s %13s" % ("depth", "floor", "rounds", "drafted", "kept", "written", "checked", "written/check"))
        for d in depths:
            rows = [{"depth": d, "floor": None, "rounds": r["rounds_%d" % d], "kept": r["walked_%d" % d], "drafted": float(d)}]
            rows += [f for f in r["floors"] if f["depth"] == d]
            for f in rows:
                lines.append("%5d %6s %8d %8.3f %8.3f %8.3f %8.3f %13.3f" % (
                    d, "none" if f["floor"] is None else "%.2f" % f["floor"], f["rounds"], f["drafted"], f["kept"],
                    f["kept"] + 1, f["drafted"] + 1, (f["kept"] + 1) / (f["drafted"] + 1)))
    return "\n".join(lines)


@torch.no_grad()
def check_alignment(model, rec: Recording, device, feats_device=None, requests: int = 3, most: int = 4096) -> tuple:
    """The loader's pairing of states and top-k rows, asked of the target itself: its last state at a position,
    through its own output matrix, names first the token that the top-k row of the next position names first.
    Returns the share of rows where it does, and how far the distribution made from the state stands from the
    recorded one: over a row's recorded choices, the difference of the two log-probabilities weighted by the recorded
    chance, as the mean over the rows and the largest of them."""
    if rec.taps[-1] != max(rec.taps):
        raise SystemExit("the recording's last tap is not the target's output state")
    same = total = 0
    off, worst = 0.0, 0.0
    for req in stream(rec, with_continuations(rec)[:requests]):
        loaded = Loaded(req, device, feats_device=feats_device)
        cut, a, g0, t0, _ = (x[torch.from_numpy(spread(req.n_anchors, most)).to(device)] for x in loaded.an)
        start = a == 0
        # the state after token a of a continuation with the row its next token was drawn from, and the prompt's
        # state before a cut with the continuation's first row
        rows = torch.cat([loaded.ctx.c_last + g0 + a, cut[start] - 1])
        tk = torch.cat([t0 + 1, t0[start]])
        top = (loaded.ctx.take(rows)[:, -1] @ model.head_w.T).argmax(-1)
        same += int((top == loaded.tk_ids[tk, 0]).sum())
        total += int(rows.numel())
        for s in range(0, rows.numel(), 256):
            logp = target_logp(model, loaded.ctx, rows[s:s + 256])
            ids, kept = loaded.tk_ids[tk[s:s + 256]], loaded.tk_logp[tk[s:s + 256]]
            d = (kept.exp() * (logp.gather(-1, ids) - kept).abs()).sum(-1)
            off += float(d.sum())
            worst = max(worst, float(d.max()) if d.numel() else 0.0)
        del loaded, logp
    if total == 0:
        raise SystemExit("%s has no continuation to check" % rec.path)
    return same / total, off / total, worst


@torch.no_grad()
def check_prompt_pairing(model, rec: Recording, device, feats_device=None, most: int = 512):
    """The loader's pairing of a prompt's anchors, asked of the target itself: the state the loader names for a draft
    row gives the prompt's token at that row more chance than the states one position before and after it do.
    Returns the mean log-probability of the token from the three states (before, the one named, after), or None when
    no request of the recording holds enough anchors to ask."""
    block = model.cfg.block_size
    for entry in sorted(rec.entries, key=lambda e: (e["n_tokens"], e["id"])):
        if rec.model_rows(entry)[0].size < 16:
            continue
        req = rec.load(entry)
        loaded = Loaded(req, device, feats_device=feats_device)
        _, x, rows, valid = loaded.batch_prompt(spread(req.n_prompt, most), block)
        sums = [0.0, 0.0, 0.0]
        for s in range(0, rows.shape[0], 64):
            r, t, v = rows[s:s + 64].reshape(-1), x[s:s + 64].reshape(-1, 1), valid[s:s + 64].reshape(-1)
            for k, shift in enumerate((-1, 0, 1)):
                logp = target_logp(model, loaded.ctx, (r + shift).clamp(0, loaded.ctx.c_last - 1))
                sums[k] += float(logp.gather(-1, t)[:, 0][v].sum())
        count = max(1, int(valid.sum()))
        return [value / count for value in sums]
    return None


def plan_epoch(recs: list, args, rng) -> list:
    """(recording, entry, [(kind, anchor numbers) of each step]) for one pass over the recordings, requests and
    anchors shuffled. A step is of one kind: anchors of the request's continuations ("gen"), or of its prompt."""
    entries = []
    for rec in recs:
        for e in sorted(rec.entries, key=lambda e: (e["n_tokens"], e["id"])):
            n_gen = 0 if args.no_gen_anchors else int(e.get("gen_rows", 0))
            n_prompt = int(rec.model_rows(e)[0].size) if rec.prompt is not None else 0
            if n_gen or n_prompt:
                entries.append((rec, e, n_gen, n_prompt))
    out = []
    for k in rng.permutation(len(entries)):
        rec, e, n_gen, n_prompt = entries[int(k)]
        steps = []
        for kind, count, most in (("gen", n_gen, args.max_anchors_per_request), ("prompt", n_prompt, args.prompt_max)):
            if not count:
                continue
            order = rng.permutation(count)
            if most:
                order = order[:most]
            steps += [(kind, np.sort(g)) for g in np.array_split(order, max(1, math.ceil(order.size / args.anchors)))]
        if n_gen and n_prompt:
            steps = [steps[int(i)] for i in rng.permutation(len(steps))]
        out.append((rec, e, steps))
    return out


def lr_at(step: int, total: int, base: float, warmup: float) -> float:
    warm = max(1, int(total * warmup))
    if step < warm:
        return base * (step + 1) / warm
    return base * 0.5 * (1.0 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def save_checkpoint(model, out: str, step: int, keep: int = 2) -> None:
    path = os.path.join(out, "ckpt-%07d.pt" % step)
    torch.save({name: p.detach().cpu() for name, p in model.named_parameters()}, path + ".tmp")
    os.replace(path + ".tmp", path)
    old = sorted(f for f in os.listdir(out) if f.startswith("ckpt-") and f.endswith(".pt"))
    for name in old[:-keep]:
        os.remove(os.path.join(out, name))


def write_json(path: str, value) -> None:
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(value, f, indent=1)
        f.write("\n")
    os.replace(path + ".tmp", path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--draft", required=True, help="the draft GGUF the server loads")
    p.add_argument("--vocab", default="", help="a GGUF with the target's token_embd, token_embd_norm and output, when the draft has none")
    p.add_argument("--target", default="", help="the target GGUF, for the epsilon of its norms")
    p.add_argument("--tok-norm-eps", type=float, default=None, help="that epsilon, when no target file is at hand")
    p.add_argument("--gguf-py", default="", help="the engine source's gguf-py folder (the installed gguf otherwise)")
    p.add_argument("--train", default=[], nargs="*", help="the recordings to train on")
    p.add_argument("--eval", default="", help="the recording kept out of training")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--anchors", type=int, default=256, help="anchors a step, all of one request")
    p.add_argument("--max-anchors-per-request", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--prompt-anchors", action="store_true", help="also train at every token the model wrote in the prompts: "
                   "the target's distribution there is made from the output state the recording holds")
    p.add_argument("--prompt-eom", type=int, default=200010, help="the token that ends a message, which those anchors are found by")
    p.add_argument("--prompt-max", type=int, default=0, help="at most so many of them a request each pass (0: all)")
    p.add_argument("--no-gen-anchors", action="store_true", help="leave the continuations' anchors out of training")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup", type=float, default=0.05, help="share of the steps the rate climbs over")
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--mode-weight", type=float, default=0.25, help="weight of naming the target's first choice")
    p.add_argument("--tail-weight", type=float, default=0.25, help="weight of the rows past the launcher's depth")
    p.add_argument("--eval-requests", type=int, default=0, help="requests of the held-out recording (0: all)")
    p.add_argument("--eval-anchors", type=int, default=128, help="anchors of each")
    p.add_argument("--eval-every", type=int, default=0, help="a small evaluation every so many steps")
    p.add_argument("--eval-small-requests", type=int, default=8)
    p.add_argument("--eval-small-anchors", type=int, default=64)
    p.add_argument("--save-every", type=int, default=0, help="keep the parameters every so many steps")
    p.add_argument("--init", default="", help="parameters kept by --save-every, to go on from")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--walk", action="store_true", help="also walk the server's rounds over every anchor of the held-out "
                   "recording: what a round keeps as the server counts it, for the draft and for the most a draft can keep")
    p.add_argument("--walk-before", default="", help="the walk of this draft before training, as an earlier run wrote it: "
                   "it stands as the walk before, and one request is walked now to show that the walk runs")
    p.add_argument("--kv-type", default="q8_0", help="cache type of the draft's global layers (--spec-draft-type-k/v)")
    p.add_argument("--kv-type-swa", default="f16", help="cache type of its sliding-window layers (--cache-type-k-swa)")
    p.add_argument("--kv-rot", default="auto", help="auto, off, or K,V: the rotation a Q8_0 cache stores in")
    p.add_argument("--no-gemm", action="store_true", help="multiply BF16 matrices by the f32 input, not its BF16 value")
    p.add_argument("--cells", type=int, default=2_500_000, help="anchors x context rows one attention holds")
    p.add_argument("--max-batch", type=int, default=64)
    p.add_argument("--chunk", type=int, default=4096, help="context rows made into keys at once")
    p.add_argument("--one-graph", action="store_true", help="one graph a step, for checking the two-pass gradient")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=20261119)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--feats-device", default="", help="where a request's recorded states are held (the draft's device otherwise): "
                   "another card or cpu, when the longest requests leave the draft's card too little room")
    args = p.parse_args(argv)
    if not args.train and not args.eval:
        p.error("name a recording: --train, --eval or both")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    eps = dspark_model.target_eps(args.gguf_py, args.target) if args.target else args.tok_norm_eps
    model = dspark_model.DSpark.from_gguf(args.draft, device, args.gguf_py, args.vocab, eps, args.kv_type,
                                          args.kv_type_swa, args.kv_rot, not args.no_gemm)
    say(model.describe())
    say("the anchor's embedding is normed with epsilon %g" % model.cfg.tok_norm_eps)
    if args.init:
        state = torch.load(args.init, map_location="cpu")
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.copy_(state[name].to(device))
        model.refresh()
        say("parameters from %s" % args.init)
    block = model.cfg.block_size
    trains = [Recording(path) for path in args.train]
    held = Recording(args.eval) if args.eval else None
    for rec in trains + [held]:
        if rec is not None and rec.taps != model.cfg.target_layers:
            raise SystemExit("%s holds the target's layers %s, the draft reads %s" % (rec.path, list(rec.taps), list(model.cfg.target_layers)))
    paired, off, worst = check_alignment(model, trains[0] if trains else held, device, args.feats_device or None)
    say("states and top-k rows pair up: the target's state names the recorded first choice in %.2f%% of rows, and the "
        "distribution made from it stands %.4f from the recorded one on average (%.4f at most)" % (100 * paired, off, worst))
    if paired < 0.95:
        raise SystemExit("the recording's states and top-k rows do not pair up as the loader reads them")
    summary = {"draft": args.draft, "paired": paired, "logp_off": [off, worst], "arguments": vars(args)}
    if args.prompt_anchors and trains:
        if off > 0.25:
            raise SystemExit("the target's distribution made from its recorded state is not the recorded one: no prompt anchors")
        seen = ended = count = 0
        for rec in trains:
            rec.use_prompt(args.prompt_eom)
            for e in rec.entries:
                cuts = [s["cut"] for s in e.get("segments", [])]
                if cuts:
                    tokens = rec.tokens(e)
                    seen += len(cuts)
                    ended += sum(1 for c in cuts if int(tokens[c - 1]) == args.prompt_eom)
                count += int(rec.model_rows(e)[0].size)
        say("the prompts hold %d anchors of their own; %d of the %d cuts follow the end of a message" % (count, ended, seen))
        summary["prompt_anchors"] = count
        if seen and ended < 0.99 * seen:
            raise SystemExit("token %d does not end the message before a cut: it is not the end of a message here" % args.prompt_eom)
        if count == 0:
            raise SystemExit("the prompts hold no anchor: the model's token %s is in none of them" % trains[0].meta.get("tok_model"))
        means = check_prompt_pairing(model, trains[0], device, args.feats_device or None)
        if means is not None:
            say("a prompt's token has a mean log-probability of %.3f from the state the loader pairs it with, %.3f from the "
                "one before and %.3f from the one after" % (means[1], means[0], means[2]))
            summary["prompt_pairing"] = means
            if means[1] < max(means[0], means[2]) + 1.0:
                raise SystemExit("the prompt's anchors are not paired with the states their tokens were drawn from")

    before = None
    if held is not None:
        t = time.time()
        before = evaluate(model, held, args, device, args.eval_requests, args.eval_anchors)
        write_json(os.path.join(args.out, "eval-before.json"), before)
        say("before training, on %s (%.0fs):\n%s" % (held.path, time.time() - t, table(before, block - 1)))
        summary["before"] = before.get("all")
        if args.walk and args.walk_before:
            with open(args.walk_before, encoding="utf-8") as f:
                summary["walked_before"] = json.load(f).get("all")
            t = time.time()
            try:
                walk(model, held, args, device, 1)
                say("before training, the walk in %s stands, and a walk of one request runs (%.0fs)" % (args.walk_before, time.time() - t))
            except Exception:
                # training goes on: the tuned draft can be walked by a run of its own once the walk is mended
                summary["walk_error"] = traceback.format_exc()
                say("the walk does not run, so the rounds are not walked after this training:\n%s" % summary["walk_error"])
                args.walk = False
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        elif args.walk:
            t = time.time()
            walked = walk(model, held, args, device, args.eval_requests)
            write_json(os.path.join(args.out, "walk-before.json"), walked)
            say("before training, the server's rounds walked over %s (%.0fs):\n%s" % (held.path, time.time() - t, walk_table(walked)))
            summary["walked_before"] = walked.get("all")
    if args.eval_only or not trains:
        write_json(os.path.join(args.out, "summary.json"), summary)
        return

    rng = np.random.default_rng(args.seed)
    plan = [plan_epoch(trains, args, rng) for _ in range(args.epochs)]
    total = sum(len(groups) for epoch in plan for _, _, groups in epoch)
    if args.max_steps:
        total = min(total, args.max_steps)
    if total == 0:
        raise SystemExit("%s has no continuation to train on" % ", ".join(rec.path for rec in trains))
    kinds = [kind for epoch in plan for _, _, groups in epoch for kind, _ in groups][:total]
    say("%d steps: %d epochs of %d requests, %d anchors a step; %d steps at the continuations' anchors, %d at the prompts'" % (
        total, args.epochs, len(plan[0]), args.anchors, kinds.count("gen"), kinds.count("prompt")))
    topk = trains[0].topk
    opt = torch.optim.AdamW(list(model.parameters()), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0,
                            fused=device.type == "cuda")
    w_rows = torch.tensor([1.0 if i < DEPTH else args.tail_weight for i in range(block - 1)], device=device)
    grad = "one" if args.one_graph else "two"
    losses = []
    done = 0
    t0 = time.time()
    with open(os.path.join(args.out, "train.jsonl"), "w", encoding="utf-8") as log:
        for epoch_no, epoch in enumerate(plan):
            if done >= total:
                break
            for (_, entry, groups), req in zip(epoch, stream_pairs([(rec, e) for rec, e, _ in epoch])):
                if req.n_anchors != entry.get("gen_rows", 0):
                    raise SystemExit("%s: %d anchors read, %d planned" % (req.id, req.n_anchors, entry.get("gen_rows", 0)))
                loaded = Loaded(req, device, feats_device=args.feats_device or None)
                for kind, chosen in groups:
                    rate = lr_at(done, total, args.lr, args.warmup)
                    for group in opt.param_groups:
                        group["lr"] = rate
                    t = time.time()
                    r = step_gradient(model, loaded, chosen, args, w_rows, grad, kind, topk)
                    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip))
                    if not math.isfinite(r["loss"]) or not math.isfinite(norm):
                        raise SystemExit("step %d on %s: loss %s, gradient norm %s" % (done, req.id, r["loss"], norm))
                    opt.step()
                    model.refresh()
                    done += 1
                    losses.append(r["loss"])
                    now = time.time()
                    r["loss_" + kind] = r["loss"]
                    r.update(step=done, total=total, epoch=epoch_no, request=req.id, kind=kind, lr=rate, grad_norm=norm, seconds=now - t,
                             time=now, elapsed=now - t0, eta_s=(now - t0) / done * (total - done),
                             anchors_per_s=len(chosen) / max(now - t, 1e-9))
                    log.write(json.dumps(r) + "\n")
                    log.flush()
                    if done % args.log_every == 0 or done == total:
                        recent = losses[-args.log_every:]
                        say("step %d/%d %s loss %.4f (kd %.4f, first choice %.4f) grad %.2f lr %.2e context %d, %.2fs a step, %.0f min left" % (
                            done, total, kind, sum(recent) / len(recent), r["kd"], r["mode"], norm, rate, r["context"],
                            (time.time() - t0) / done, (time.time() - t0) / done * (total - done) / 60))
                    if args.save_every and done % args.save_every == 0:
                        save_checkpoint(model, args.out, done)
                    if args.eval_every and held is not None and done % args.eval_every == 0 and done < total:
                        small = evaluate(model, held, args, device, args.eval_small_requests, args.eval_small_anchors)
                        say("after %d steps, a small evaluation:\n%s" % (done, table(small, block - 1)))
                        log.write(json.dumps({"step": done, "eval": small.get("all"), "time": time.time()}) + "\n")
                        log.flush()
                    if done >= total:
                        break
                del loaded
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if done >= total:
                    break
    edge = max(1, min(20, len(losses) // 5))
    summary.update(steps=done, seconds=time.time() - t0, loss_first=sum(losses[:edge]) / edge,
                   loss_last=sum(losses[-edge:]) / edge)
    say("trained %d steps in %.0f min: loss %.4f over the first %d steps, %.4f over the last" % (
        done, (time.time() - t0) / 60, summary["loss_first"], edge, summary["loss_last"]))
    # the tuned draft first: nothing after this can lose it
    del opt
    for p in model.parameters():
        p.grad = None
    if device.type == "cuda":
        torch.cuda.empty_cache()
    tuned = os.path.join(args.out, "draft-tuned.gguf")
    written = model.export(args.draft, tuned)
    say("wrote %s: %d tensors over a copy of the draft, read back the same" % (tuned, written))
    summary["tuned"] = tuned
    write_json(os.path.join(args.out, "summary.json"), summary)

    if held is not None:
        t = time.time()
        after = evaluate(model, held, args, device, args.eval_requests, args.eval_anchors)
        write_json(os.path.join(args.out, "eval-after.json"), after)
        say("after training, on %s (%.0fs):\n%s" % (held.path, time.time() - t, table(after, block - 1)))
        summary["after"] = after.get("all")
        if args.walk:
            t = time.time()
            walked = walk(model, held, args, device, args.eval_requests)
            write_json(os.path.join(args.out, "walk-after.json"), walked)
            say("after training, the server's rounds walked over %s (%.0fs):\n%s" % (held.path, time.time() - t, walk_table(walked)))
            summary["walked_after"] = walked.get("all")
        key = "accepted_%d" % DEPTH
        if before and "all" in before and "all" in after:
            say("accepted a round at depth %d: %.3f before, %.3f after, of a ceiling of %.3f" % (
                DEPTH, before["all"][key], after["all"][key], after["all"]["ceiling_%d" % DEPTH]))
    write_json(os.path.join(args.out, "summary.json"), summary)


if __name__ == "__main__":
    main()
