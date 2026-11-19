"""A recording of dspark_tool, read for training: one request at a time, as the tool wrote it.

A request is a prompt and some continuations. A continuation starts at a cut of the prompt: its token j stands at
position cut + j, row j of the request's written states is the target's state at that position, and row j of its
top-k rows is the distribution token j was drawn from (so the state at cut + j goes with top-k row j + 1, and the
prompt's state at cut - 1 with top-k row 0).

An anchor is a place the server could ask for a draft: a continuation and a count a of its tokens already read. The
draft then knows the states of the prompt below the cut and of the continuation's first a tokens, is given token a as
the last one accepted, and its row i predicts token a + i, which the target drew from top-k row a + i.

A prompt holds anchors of its own (use_prompt): every token the model wrote in the prompt's own part is one, with
the prompt before it as its context. Its row i predicts the prompt's token i places on, and the distribution the
target would draw that token from is not stored: it is the target's output state of the position before it, which
the recording holds as the last tap, through the target's own output matrix. Such an anchor reaches to the end of the
model's turn it stands in.

The states stay in the file's BF16 on the host and on the device; nothing holds a request in f32.
"""
from __future__ import annotations

import dataclasses
import json
import os
import queue
import threading

import numpy as np
import torch


@dataclasses.dataclass
class Context:
    """The target's states a request's anchors can see, on the device."""
    feats: torch.Tensor   # [rows, taps, n_embd] BF16: row p the prompt's position p for p < c_last, row c_last + g written state g
    pos: torch.Tensor     # [rows] the position each row stands at
    c_last: int           # prompt rows kept: the largest cut
    n_gen: int            # written states

    def take(self, rows: torch.Tensor) -> torch.Tensor:
        """The states of `rows` in f32, on the device the rows are named on (feats may be kept on another)."""
        return self.feats[rows.to(self.feats.device)].to(rows.device).float()


@dataclasses.dataclass
class Batch:
    """Anchors of one request, in the order of (cut, a)."""
    cut: torch.Tensor     # [A] the cut of the anchor's continuation
    a: torch.Tensor       # [A] tokens of the continuation read before the anchor
    row0: torch.Tensor    # [A] the context row of the continuation's first written state
    ids: torch.Tensor     # [A] the anchor token
    prev: torch.Tensor    # [A, rows - 2] the token before each of draft rows 2.., or None to chain the draft's own


class Request:
    """One request of a recording, on the host."""

    def __init__(self):
        self.id = ""
        self.n_tokens = 0
        self.c_last = 0
        self.n_gen = 0
        self.parts = []       # int16 [rows, taps, n_embd] pieces of the context, in row order
        self.gen_pos = None   # [n_gen] the position of each written state
        self.topk = None      # the top-k rows, as stored
        self.tok = None       # [top-k rows] the token drawn from each
        # one entry an anchor
        self.an_cut = self.an_a = self.an_g0 = self.an_t0 = self.an_nst = None
        # the prompt's anchors, when the recording was asked for them
        self.ptok = None      # [n_tokens] the prompt
        self.pr_n = None      # the position of each
        self.pr_end = None    # the last position of the model's turn it stands in

    @property
    def n_anchors(self) -> int:
        return int(self.an_cut.size)

    @property
    def n_prompt(self) -> int:
        return 0 if self.pr_n is None else int(self.pr_n.size)


class Recording:
    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.taps = tuple(int(t) for t in self.meta["taps"])
        self.n_embd = int(self.meta["n_embd"])
        self.row_bytes = int(self.meta["row_bytes"])
        self.topk = int(self.meta["topk"])
        if self.meta.get("dtype", "bf16") != "bf16":
            raise SystemExit("%s holds %s states, not bf16" % (path, self.meta["dtype"]))
        if self.row_bytes != len(self.taps) * self.n_embd * 2:
            raise SystemExit("%s: a row of %d bytes is not %d taps of %d" % (path, self.row_bytes, len(self.taps), self.n_embd))
        self.topk_dtype = np.dtype([("lse", "<f4"), ("ids", "<i4", (self.topk,)), ("logits", "<f4", (self.topk,))])
        if self.topk_dtype.itemsize != int(self.meta["topk_bytes"]):
            raise SystemExit("%s: a top-k row of %s bytes is not lse, %d ids, %d logits" % (path, self.meta["topk_bytes"], self.topk, self.topk))
        self.entries = []
        with open(os.path.join(path, "index.jsonl"), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.entries.append(json.loads(line))
        self.prompt = None          # (model, end of turn, end of message) tokens, once the prompt's anchors are asked for
        self._model_rows = {}

    def use_prompt(self, tok_eom: int) -> None:
        """From here on a request carries the anchors of its prompt too, and every row of the prompt the recording
        holds, whatever its cuts."""
        for key in ("tok_model", "tok_end"):
            if key not in self.meta:
                raise SystemExit("%s does not name %s, which the prompt's anchors are found by" % (self.path, key))
        self.prompt = (int(self.meta["tok_model"]), int(self.meta["tok_end"]), int(tok_eom))

    def prompt_rows(self, entry: dict) -> int:
        """The rows of an entry's prompt whose states the recording holds."""
        return min(int(entry["n_tokens"]), int(entry["shared_len"]) + int(entry.get("states_rows", 0)))

    def model_rows(self, entry: dict) -> tuple:
        """(n, end): the anchors of an entry's prompt. n: each position of the prompt's own part that the model wrote
        and that has another of its turn after it; end: the last position of that turn. A turn is the model's messages
        one after another, each from the model's token to the end of its message, and the token that ends the turn
        when it follows one."""
        key = entry["file"]
        if key not in self._model_rows:
            tok_model, tok_end, tok_eom = self.prompt
            lo, hi = int(entry["shared_len"]), self.prompt_rows(entry)
            empty = np.empty(0, dtype=np.int64)
            if hi - lo < 2:
                self._model_rows[key] = (empty, empty)
            else:
                t = self.tokens(entry)[lo:hi]
                idx = np.arange(lo, hi, dtype=np.int64)
                eom = t == tok_eom
                last_eom = np.maximum.accumulate(np.where(eom, idx, lo - 1))
                start = np.concatenate([[lo], last_eom[:-1] + 1])       # where the message of each position begins
                last_model = np.maximum.accumulate(np.where(t == tok_model, idx, -1))
                wrote = last_model >= start
                wrote |= (t == tok_end) & np.concatenate([[False], (eom & wrote)[:-1]])
                edge = np.diff(np.concatenate([[0], wrote.astype(np.int8), [0]]))
                first, last = np.nonzero(edge == 1)[0], np.nonzero(edge == -1)[0] - 1
                end = np.zeros(hi - lo, dtype=np.int64)
                for a, b in zip(first, last):
                    end[a:b + 1] = lo + b
                keep = wrote & (idx < end)
                self._model_rows[key] = (idx[keep], end[keep])
        return self._model_rows[key]

    def _rows(self, name: str, first: int, count: int) -> np.ndarray:
        shape = (count, len(self.taps), self.n_embd)
        if count == 0:
            return np.empty(shape, dtype=np.int16)
        path = os.path.join(self.path, name)
        data = np.fromfile(path, dtype=np.int16, count=count * shape[1] * shape[2], offset=first * self.row_bytes)
        if data.size != count * shape[1] * shape[2]:
            raise SystemExit("%s: %d rows from %d asked, the file is shorter" % (path, count, first))
        return data.reshape(shape)

    def load(self, entry: dict, prompt_rows=None) -> Request:
        """The request of an index entry. prompt_rows: keep that many rows of the prompt, when the entry has no
        continuation to say how many (a recording read back for the live check)."""
        req = Request()
        req.id = entry["id"]
        req.n_tokens = int(entry["n_tokens"])
        shared = int(entry["shared_len"])
        segments = entry.get("segments", [])
        states = topk = 0
        for s in segments:
            if s["states_row0"] != states or s["topk_row0"] != topk or len(s["tokens"]) != s["n_states"] + 1:
                raise SystemExit("%s: the continuation at %d does not follow the one before it" % (req.id, s["cut"]))
            if not shared < s["cut"] <= req.n_tokens:
                raise SystemExit("%s: a cut at %d of %d tokens, %d shared" % (req.id, s["cut"], req.n_tokens, shared))
            states += s["n_states"]
            topk += s["n_states"] + 1
        if states != entry.get("gen_rows", 0) or topk != entry.get("topk_rows", 0):
            raise SystemExit("%s: its continuations hold %d states and %d top-k rows, the index says %d and %d" % (
                req.id, states, topk, entry.get("gen_rows", 0), entry.get("topk_rows", 0)))
        live = [s for s in segments if s["n_states"] > 0]
        if prompt_rows is None and self.prompt is not None:
            prompt_rows = max([self.prompt_rows(entry)] + [s["cut"] for s in live])
        req.c_last = max([s["cut"] for s in live], default=0) if prompt_rows is None else int(prompt_rows)
        if req.c_last > req.n_tokens:
            raise SystemExit("%s: %d rows of a prompt of %d" % (req.id, req.c_last, req.n_tokens))
        req.n_gen = states
        n_pre = min(shared, req.c_last)
        base = os.path.join("req", entry["file"])
        req.parts = [self._rows("prefix.states", 0, n_pre), self._rows(base + ".states", 0, req.c_last - n_pre),
                     self._rows(base + ".gen.states", 0, states)]
        if topk:
            req.topk = np.fromfile(os.path.join(self.path, base + ".gen.topk"), dtype=self.topk_dtype)
            if req.topk.size != topk:
                raise SystemExit("%s: %d top-k rows on disk, %d in the index" % (req.id, req.topk.size, topk))
        else:
            req.topk = np.empty(0, dtype=self.topk_dtype)
        req.tok = np.zeros(topk, dtype=np.int64)
        req.gen_pos = np.zeros(states, dtype=np.int64)
        cut, a, g0, t0, nst = [], [], [], [], []
        for s in segments:
            n = s["n_states"]
            req.tok[s["topk_row0"]: s["topk_row0"] + n + 1] = s["tokens"]
            req.gen_pos[s["states_row0"]: s["states_row0"] + n] = s["cut"] + np.arange(n)
            steps = np.arange(n, dtype=np.int64)
            cut.append(np.full(n, s["cut"], dtype=np.int64))
            a.append(steps)
            g0.append(np.full(n, s["states_row0"], dtype=np.int64))
            t0.append(s["topk_row0"] + steps)
            nst.append(np.full(n, n, dtype=np.int64))
        empty = np.empty(0, dtype=np.int64)
        req.an_cut, req.an_a, req.an_g0, req.an_t0, req.an_nst = (
            np.concatenate(x) if x else empty for x in (cut, a, g0, t0, nst))
        if self.prompt is not None:
            req.ptok = self.tokens(entry)
            req.pr_n, req.pr_end = self.model_rows(entry)
            if req.pr_end.size and int(req.pr_end.max()) >= req.c_last:
                raise SystemExit("%s: a turn of the prompt ends at %d, past the %d rows kept" % (req.id, int(req.pr_end.max()), req.c_last))
        return req

    def tokens(self, entry: dict) -> np.ndarray:
        data = np.fromfile(os.path.join(self.path, "req", entry["file"] + ".tokens"), dtype=np.int32)
        if data.size != entry["n_tokens"]:
            raise SystemExit("%s: %d tokens on disk, %d in the index" % (entry["id"], data.size, entry["n_tokens"]))
        return data


class Loaded:
    """A request on the device."""

    def __init__(self, req: Request, device, rows_a_copy: int = 4096, feats_device=None):
        self.req = req
        taps, n_embd = (req.parts[0].shape[1], req.parts[0].shape[2])
        total = req.c_last + req.n_gen
        held = device if feats_device is None else feats_device   # the states are the one large thing of a request
        feats = torch.empty((total, taps, n_embd), dtype=torch.bfloat16, device=held)
        row = 0
        for part in req.parts:
            for i in range(0, part.shape[0], rows_a_copy):
                piece = torch.from_numpy(part[i:i + rows_a_copy]).to(held)
                feats[row:row + piece.shape[0]] = piece.view(torch.bfloat16)
                row += piece.shape[0]
        pos = torch.cat([torch.arange(req.c_last, dtype=torch.int64), torch.from_numpy(req.gen_pos)]).to(device)
        self.ctx = Context(feats=feats, pos=pos, c_last=req.c_last, n_gen=req.n_gen)
        self.tk_ids = torch.from_numpy(req.topk["ids"].astype(np.int64)).to(device)
        self.tk_logp = torch.from_numpy(req.topk["logits"] - req.topk["lse"][:, None]).to(device)
        self.tok = torch.from_numpy(req.tok).to(device)
        self.an = [torch.from_numpy(x).to(device) for x in (req.an_cut, req.an_a, req.an_g0, req.an_t0, req.an_nst)]
        self.ptok = None if req.ptok is None else torch.from_numpy(req.ptok.astype(np.int64)).to(device)
        self.pr = None if req.pr_n is None else [torch.from_numpy(x).to(device) for x in (req.pr_n, req.pr_end)]

    def batch(self, chosen: np.ndarray, block: int):
        """The anchors numbered `chosen` (ascending), with what the target did after each.

        Returns (batch, x, rows, valid): x [A, block-1] the token the target drew for each draft row, rows [A, block-1]
        the top-k row it was drawn from, valid [A, block-1] whether the continuation got that far.
        """
        index = torch.from_numpy(np.asarray(chosen, dtype=np.int64)).to(self.tok.device)
        cut, a, g0, t0, nst = (x[index] for x in self.an)
        step = torch.arange(1, block, device=index.device)
        valid = a[:, None] + step[None, :] <= nst[:, None]
        rows = (t0[:, None] + step[None, :]).clamp_max(self.tok.numel() - 1)
        x = self.tok[rows]
        return Batch(cut=cut, a=a, row0=self.ctx.c_last + g0, ids=self.tok[t0], prev=x[:, :-1]), x, rows, valid

    def batch_prompt(self, chosen: np.ndarray, block: int):
        """The prompt's anchors numbered `chosen` (ascending), as batch() returns the others: an anchor at position n
        is a continuation cut at n with nothing read yet. x [A, block-1] is the prompt's token at each draft row, and
        rows [A, block-1] the context row whose output state the target would draw it from: the position before it.
        """
        index = torch.from_numpy(np.asarray(chosen, dtype=np.int64)).to(self.ptok.device)
        n, end = (x[index] for x in self.pr)
        at = n[:, None] + torch.arange(1, block, device=index.device)[None, :]
        valid = at <= end[:, None]
        at = at.clamp_max(self.ctx.c_last - 1)
        x = self.ptok[at]
        batch = Batch(cut=n, a=torch.zeros_like(n), row0=torch.full_like(n, self.ctx.c_last), ids=self.ptok[n], prev=x[:, :-1])
        return batch, x, at - 1, valid


def stream(recording: Recording, entries: list):
    """The requests of `entries`, each read from disk while the one before it is being used."""
    return stream_pairs([(recording, e) for e in entries])


def stream_pairs(pairs: list):
    """The requests of (recording, entry) pairs, each read from disk while the one before it is being used."""
    q = queue.Queue(maxsize=1)

    def work():
        try:
            for recording, e in pairs:
                q.put(recording.load(e))
            q.put(None)
        except BaseException as err:  # the reader's failure is the caller's
            q.put(err)

    threading.Thread(target=work, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            return
        if isinstance(item, BaseException):
            raise item
        yield item
