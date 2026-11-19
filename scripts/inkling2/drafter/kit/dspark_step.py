"""The draft's forward for many anchors of one request at once, with the gradient the server's layout allows.

Every anchor of a request reads the same prompt, so the keys and values the draft makes from the target's states are
made once a step and shared:

  a global layer   keeps a key and a value for every prompt row below the step's largest cut and for every written
                   state; an anchor sees the prompt rows below its own cut and the first a states of its continuation.
  a window layer   sees the W positions before the anchor, so the step keeps only the rows some anchor's window holds
                   and each anchor indexes its own W of them.

Anchors run through the block in runs that fit the attention's memory. With grad "two" the keys and values are made
without a graph and stand as leaves: each run's loss is carried back to them and to the block's parameters as soon as
it is computed, and finish() then makes the keys and values again, a chunk of rows at a time, and carries the
collected gradient on to the parameters that made them. The result is the gradient of the whole step, and no graph
ever spans more than one run or one chunk. Grad "one" builds the same step as one graph, for checking the other.
"""
from __future__ import annotations

import torch

from dspark_data import Batch, Context


class Step:
    def __init__(self, model, ctx: Context, batch: Batch, grad: str = "none", cells: int = 2_500_000,
                 max_batch: int = 64, chunk: int = 4096):
        if grad not in ("none", "two", "one"):
            raise SystemExit("grad is none, two or one")
        self.model, self.ctx, self.batch, self.grad = model, ctx, batch, grad
        self.cells, self.max_batch, self.chunk = cells, max_batch, chunk
        c = model.cfg
        dev = batch.cut.device
        cut, a, row0 = batch.cut, batch.a, batch.row0
        if cut.numel() > 1 and bool((cut[1:] < cut[:-1]).any()):
            raise SystemExit("the anchors of a step come in the order of their cuts")
        self.n = cut + a                                           # the anchor's position
        self.c_max = int(cut.max())
        if self.c_max > ctx.c_last:
            raise SystemExit("a cut at %d, the context keeps %d prompt rows" % (self.c_max, ctx.c_last))
        self.rows_glob = torch.cat([torch.arange(self.c_max, device=dev),
                                    ctx.c_last + torch.arange(ctx.n_gen, device=dev)])
        w = c.swa_window
        q = self.n[:, None] - w + torch.arange(w, device=dev)[None, :]
        self.swa_vis = q >= 0
        row = torch.where(q < cut[:, None], q, row0[:, None] + q - cut[:, None]).clamp_min(0)
        uniq = torch.unique(row[self.swa_vis])
        if uniq.numel() == 0:
            uniq = row.new_zeros(1)
        rowmap = torch.full((ctx.feats.shape[0],), -1, dtype=torch.int64, device=dev)
        rowmap[uniq] = torch.arange(uniq.numel(), device=dev)
        self.swa_idx = rowmap[row].clamp_min(0)
        self.rows_swa = uniq
        self.glob = [il for il in range(c.n_layer) if not c.swa_pattern[il]]
        self.swa = [il for il in range(c.n_layer) if c.swa_pattern[il]]
        self.K, self.V = {}, {}
        for layers, rows in ((self.glob, self.rows_glob), (self.swa, self.rows_swa)):
            if layers:
                self._inject(layers, rows)
        # what the runs are cut by, on the host
        self._cut = cut.tolist()
        self._lo = (row0 - ctx.c_last).tolist()
        self._hi = (row0 + a - ctx.c_last).tolist()

    def _inject(self, layers: list, rows: torch.Tensor) -> None:
        model, ctx = self.model, self.ctx
        c = model.cfg
        with torch.set_grad_enabled(self.grad == "one"):
            if self.grad == "one":
                f = ctx.take(rows)
                for il in layers:
                    self.K[il], self.V[il] = model.inject_layer(il, f, ctx.pos[rows])
                return
            shape = (c.n_head_kv, rows.numel(), c.head_dim)
            for il in layers:
                self.K[il] = torch.empty(shape, device=rows.device)
                self.V[il] = torch.empty(shape, device=rows.device)
            for i in range(0, rows.numel(), self.chunk):
                r = rows[i:i + self.chunk]
                f = ctx.take(r)
                for il in layers:
                    k, v = model.inject_layer(il, f, ctx.pos[r])
                    self.K[il][:, i:i + r.numel()] = k
                    self.V[il][:, i:i + r.numel()] = v
        if self.grad == "two":
            for il in layers:
                self.K[il].requires_grad_()
                self.V[il].requires_grad_()

    def runs(self) -> list:
        """(i, j) ranges of anchors, each small enough that anchors x the context rows they attend fit `cells`."""
        out = []
        total = len(self._cut)
        i = 0
        while i < total:
            lo, hi = self._lo[i], self._hi[i]
            j = i + 1
            while j < total and j - i < self.max_batch:
                span = self._cut[j] + max(0, max(hi, self._hi[j]) - min(lo, self._lo[j]))
                if (j + 1 - i) * span > self.cells:
                    break
                lo, hi = min(lo, self._lo[j]), max(hi, self._hi[j])
                j += 1
            out.append((i, j))
            i = j
        return out

    def pieces(self, i: int, j: int) -> list:
        """What each layer's attention sees of the context for anchors i..j-1."""
        ctx, b = self.ctx, self.batch
        dev = self.n.device
        cut, a, row0 = b.cut[i:j], b.a[i:j], b.row0[i:j]
        cm = int(cut.max())
        g0, g1 = min(self._lo[i:j]), max(self._hi[i:j])
        vis_p = torch.arange(cm, device=dev)[None, :] < cut[:, None]
        vis_g = None
        if g1 > g0:
            rows = ctx.c_last + torch.arange(g0, g1, device=dev)
            vis_g = (rows[None, :] >= row0[:, None]) & (rows[None, :] < (row0 + a)[:, None])
        out = []
        for il in range(self.model.cfg.n_layer):
            K, V = self.K[il], self.V[il]
            if self.model.cfg.swa_pattern[il]:
                idx = self.swa_idx[i:j]
                out.append([(K[:, idx], V[:, idx], self.swa_vis[i:j])])
                continue
            p = [(K[:, :cm], V[:, :cm], vis_p)]
            if vis_g is not None:
                p.append((K[:, self.c_max + g0: self.c_max + g1], V[:, self.c_max + g0: self.c_max + g1], vis_g))
            out.append(p)
        return out

    def states(self, i: int, j: int, emb: torch.Tensor):
        """What the head reads for anchors i..j-1: the block's states, the anchor tokens, and the state before each
        anchor. A caller that wants the rows under two chains runs the head on them twice."""
        model, ctx, b = self.model, self.ctx, self.batch
        block = model.cfg.block_size
        pos = self.n[i:j, None] + torch.arange(block, device=self.n.device)[None, :]
        hidden = model.block(emb[i:j], pos, self.pieces(i, j))
        cut, a, row0 = b.cut[i:j], b.a[i:j], b.row0[i:j]
        before = torch.where(a > 0, row0 + a - 1, cut - 1)       # the row of the position before the anchor
        f_ctx = model.f_rnn(ctx.take(before))
        return hidden, b.ids[i:j], f_ctx

    def logits(self, i: int, j: int, emb: torch.Tensor, n_rows=None) -> torch.Tensor:
        """Draft rows 1.. of anchors i..j-1: [j-i, rows-1, vocab]. emb: the embedding rows of all the step's anchors."""
        hidden, ids, f_ctx = self.states(i, j, emb)
        prev = None if self.batch.prev is None else self.batch.prev[i:j]
        return self.model.head(hidden, ids, f_ctx, prev, n_rows)

    def finish(self) -> None:
        """Carry the gradient the keys and values collected on to the parameters of the injection (grad "two")."""
        if self.grad != "two":
            return
        model, ctx = self.model, self.ctx
        for layers, rows in ((self.glob, self.rows_glob), (self.swa, self.rows_swa)):
            layers = [il for il in layers if self.K[il].grad is not None or self.V[il].grad is not None]
            if not layers:
                continue
            for i in range(0, rows.numel(), self.chunk):
                r = rows[i:i + self.chunk]
                f = ctx.take(r)
                outs, grads = [], []
                for il in layers:
                    k, v = model.inject_layer(il, f, ctx.pos[r])
                    for out, leaf in ((k, self.K[il]), (v, self.V[il])):
                        if leaf.grad is not None:
                            outs.append(out)
                            grads.append(leaf.grad[:, i:i + r.numel()])
                torch.autograd.backward(outs, grads)
