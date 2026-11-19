"""Check a recording of `dspark_tool capture` before anything is trained on it.

  files     every file has the rows its index line says, and the continuations add up
  taps      no recorded layer is empty or holds a value that is not a number
  teacher   the last tap is the state the model's output layer reads, so its logits can be computed again from
            the recording: cap * tanh(W h / cap). They are compared with the top of the distribution the recorder
            saw on the box, at the row each one is said to belong to
  read      (--ref DIR --ref-split DIR) every row the run read is the row one session reading alone gets. The
            reference is the same plan read by one session with no continuation; a second one-session read, ended at
            every cut, shows how far apart two right reads are when their batches end in different places, and that
            sets what counts as far
  written   (--replay DIR, with the two above) every row the model wrote is the row a plain read of the same tokens
            gets: the replay is each continuation read back as a prompt (make_replay.py)

Usage: check_record.py --gguf-py DIR --model GGUF --rec DIR [--ref DIR --ref-split DIR [--replay DIR]]
Exits 1 when a check fails.
"""
import argparse
import json
import os
import sys

import numpy as np


def bf16(raw: np.ndarray) -> np.ndarray:
    return (raw.astype(np.uint32) << 16).view(np.float32)


def size_of(path: str) -> int:
    return os.path.getsize(path) if os.path.exists(path) else -1


class Recording:
    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.taps = self.meta["taps"]
        self.n_embd = self.meta["n_embd"]
        self.row_bytes = self.meta["row_bytes"]
        self.topk = self.meta["topk"]
        self.topk_bytes = self.meta["topk_bytes"]
        self.index = {}
        with open(os.path.join(path, "index.jsonl"), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    j = json.loads(line)
                    self.index[j["id"]] = j
        self.keep = max(self.meta["k_tools"], self.meta["k_sys"])

    def req(self, entry: dict, suffix: str) -> str:
        return os.path.join(self.path, "req", entry["file"] + suffix)

    def state_rows(self, path: str, row0: int, rows: int) -> np.ndarray:
        """Rows of a states file as float32, [rows, taps, n_embd]."""
        raw = np.fromfile(path, dtype=np.uint16, count=rows * self.row_bytes // 2, offset=row0 * self.row_bytes)
        if raw.size != rows * self.row_bytes // 2:
            raise SystemExit("%s is short at row %d" % (path, row0))
        return bf16(raw).reshape(rows, len(self.taps), self.n_embd)

    def position(self, entry: dict, pos: int) -> tuple:
        """The states file and row that hold a position the request was read at."""
        if pos < entry["shared_len"]:
            return os.path.join(self.path, "prefix.states"), pos
        return self.req(entry, ".states"), pos - entry["shared_len"]


class Report:
    def __init__(self):
        self.failed = []

    def check(self, name: str, ok: bool, text: str) -> None:
        print("%-8s %s  %s" % (name, "ok  " if ok else "FAIL", text), flush=True)
        if not ok:
            self.failed.append(name)


def check_files(rec: Recording, report: Report) -> None:
    bad = []
    n_seg = n_stop = n_len = n_gen = 0
    rows = 0
    stop_ids = {rec.meta["tok_end"]}
    prefix_tokens = None
    if rec.keep > 0:
        prefix_tokens = np.fromfile(os.path.join(rec.path, "prefix.tokens"), dtype=np.int32)
        have = size_of(os.path.join(rec.path, "prefix.states"))
        used = max((e["shared_len"] for e in rec.index.values()), default=0)
        if prefix_tokens.size < used or have < used * rec.row_bytes or have % rec.row_bytes != 0:
            bad.append("prefix: %d tokens and %d bytes of states for %d kept positions" % (prefix_tokens.size, have, used))

    for e in rec.index.values():
        why = []
        n, shared = e["n_tokens"], e["shared_len"]
        if size_of(rec.req(e, ".tokens")) != 4 * n:
            why.append("tokens")
        if e["states_rows"] != n - shared or size_of(rec.req(e, ".states")) != e["states_rows"] * rec.row_bytes:
            why.append("states")
        if not rec.meta["no_cuts"]:
            if size_of(rec.req(e, ".gen.states")) != e["gen_rows"] * rec.row_bytes:
                why.append("gen.states")
            if size_of(rec.req(e, ".gen.topk")) != e["topk_rows"] * rec.topk_bytes:
                why.append("gen.topk")
        g_rows = t_rows = 0
        last = shared
        for s in e["segments"]:
            if s["states_row0"] != g_rows or s["topk_row0"] != t_rows or len(s["tokens"]) != s["n_states"] + 1:
                why.append("segment rows at %d" % s["cut"])
            if s["cut"] <= last and s is not e["segments"][0] or s["cut"] <= shared or s["cut"] > n:
                why.append("cut order at %d" % s["cut"])
            if s["stop"] == "length" and s["n_states"] != s["g_max"]:
                why.append("length at %d" % s["cut"])
            if s["stop"] == "stop" and s["n_states"] > s["g_max"]:
                why.append("stop at %d" % s["cut"])
            if s["stop"] not in ("stop", "length"):
                why.append("ending at %d" % s["cut"])
            if min(s["tokens"]) < 0 or max(s["tokens"]) >= rec.meta["n_vocab"]:
                why.append("token range at %d" % s["cut"])
            if s["stop"] == "stop" and s["tokens"][-1] in stop_ids:
                n_stop += 1
            n_len += s["stop"] == "length"
            g_rows += s["n_states"]
            t_rows += s["n_states"] + 1
            last = s["cut"]
            n_seg += 1
        n_gen += g_rows
        if g_rows != e["gen_rows"] or t_rows != e["topk_rows"]:
            why.append("segment totals")
        if shared > 0 and prefix_tokens is not None and "tokens" not in why:
            head = np.fromfile(rec.req(e, ".tokens"), dtype=np.int32, count=shared)
            if not np.array_equal(head, prefix_tokens[:shared]):
                why.append("prefix tokens")
        rows += e["states_rows"]
        if why:
            bad.append("%s: %s" % (e["id"], ", ".join(why)))
    report.check("files", not bad, "%d requests, %d rows read, %d continuations (%d ended by the model, %d by length), "
                 "%d rows written by the model%s" % (len(rec.index), rows, n_seg, n_stop, n_len, n_gen,
                                                    "".join("\n           " + b for b in bad[:12])))


def sample_points(rec: Recording, count: int, seed: int) -> list:
    """Top-k rows and the state each belongs to: (states file, row, top-k file, row, what)."""
    points = []
    for e in rec.index.values():
        topk = rec.req(e, ".gen.topk")
        for s in e["segments"]:
            path, row = rec.position(e, s["cut"] - 1)
            points.append((path, row, topk, s["topk_row0"], "cut"))
            for j in range(s["n_states"]):
                points.append((rec.req(e, ".gen.states"), s["states_row0"] + j, topk, s["topk_row0"] + j + 1, "written"))
    rng = np.random.default_rng(seed)
    if len(points) > count:
        cuts = [p for p in points if p[4] == "cut"]
        rest = [p for p in points if p[4] != "cut"]
        take_cuts = min(len(cuts), count // 3)
        picked = [cuts[i] for i in rng.choice(len(cuts), take_cuts, replace=False)] if take_cuts else []
        take_rest = min(len(rest), count - take_cuts)
        picked += [rest[i] for i in rng.choice(len(rest), take_rest, replace=False)] if take_rest else []
        points = picked
    return points


def check_teacher(rec: Recording, report: Report, gguf_py: str, model: str, count: int, seed: int) -> None:
    sys.path.insert(0, gguf_py)
    from gguf import GGUFReader, quants  # noqa: E402

    points = sample_points(rec, count, seed)
    if not points:
        report.check("teacher", rec.meta["no_cuts"], "the recording has no continuation to compare")
        return

    reader = GGUFReader(model)
    arch = reader.fields["general.architecture"].contents()
    layers = int(reader.fields[arch + ".block_count"].contents())
    cap = float(reader.fields[arch + ".final_logit_softcapping"].contents())
    unpadded = int(reader.fields[arch + ".vocab_size_unpadded"].contents())
    if rec.taps[-1] != layers:
        report.check("teacher", False, "the last tap is layer %d, and the output layer reads layer %d" % (rec.taps[-1], layers))
        return
    weight = next(t for t in reader.tensors if t.name == "output.weight")

    # the states, every tap of them, and what the recorder saw
    states = np.empty((len(points), len(rec.taps), rec.n_embd), dtype=np.float32)
    lse = np.empty(len(points), dtype=np.float32)
    ids = np.empty((len(points), rec.topk), dtype=np.int32)
    seen = np.empty((len(points), rec.topk), dtype=np.float32)
    for i, (path, row, topk, topk_row, _) in enumerate(points):
        states[i] = rec.state_rows(path, row, 1)[0]
        raw = np.fromfile(topk, dtype=np.uint8, count=rec.topk_bytes, offset=topk_row * rec.topk_bytes)
        lse[i] = raw[:4].view(np.float32)[0]
        ids[i] = raw[4:4 + 4 * rec.topk].view(np.int32)
        seen[i] = raw[4 + 4 * rec.topk:].view(np.float32)

    finite = bool(np.isfinite(states).all())
    rms = np.sqrt((states.astype(np.float64) ** 2).mean(axis=(0, 2)))
    report.check("taps", finite and bool((rms > 0).all()),
                 "rms of each recorded layer over %d rows: %s" % (len(points), ", ".join(
                     "%d: %.4g" % (t, v) for t, v in zip(rec.taps, rms))))

    # the logits again, a few thousand rows of the output layer at a time
    h = np.ascontiguousarray(states[:, -1, :])
    logits = np.empty((len(points), unpadded), dtype=np.float32)
    step = 2048
    for a in range(0, unpadded, step):
        b = min(unpadded, a + step)
        w = quants.dequantize(np.asarray(weight.data[a:b]), weight.tensor_type).astype(np.float32, copy=False)
        logits[:, a:b] = h @ w.reshape(b - a, rec.n_embd).T
    if cap > 0:
        logits = cap * np.tanh(logits / cap)

    again_lse = np.empty(len(points), dtype=np.float64)
    first = np.empty(len(points), dtype=np.int64)
    shared = np.empty(len(points), dtype=np.float64)
    for i in range(len(points)):
        row = logits[i]
        first[i] = int(row.argmax())
        again_lse[i] = float(row[first[i]]) + float(np.log(np.exp((row - row[first[i]]).astype(np.float64)).sum()))
        best = np.argpartition(row, -rec.topk)[-rec.topk:]
        shared[i] = len(set(best.tolist()) & set(ids[i].tolist())) / rec.topk
    again = np.take_along_axis(logits, np.clip(ids, 0, unpadded - 1), axis=1)
    diff = np.abs(again - seen)
    same = first == ids[:, 0]
    top1 = float(same.mean())
    overlap = float(shared.mean())
    kinds = {}
    for kind in ("cut", "written"):
        mask = np.array([p[4] == kind for p in points])
        if mask.any():
            kinds[kind] = (int(mask.sum()), float(diff[mask].mean()), float(same[mask].mean()))
    ok = float(np.abs(again_lse - lse).mean()) < 0.1 and int(ids.min()) >= 0 and \
        all(v[1] < 0.1 and v[2] >= 0.9 for v in kinds.values())
    report.check("teacher", ok, "%d rows, output layer %s, cap %g: the same first token on %.1f%%, %.1f%% of the top %d shared, "
                 "logits apart by %.4f on average and %.4f at most, logsumexp apart by %.4f on average; %s" % (
                     len(points), weight.tensor_type.name, cap, 100 * top1, 100 * overlap, rec.topk, float(diff.mean()),
                     float(diff.max()), float(np.abs(again_lse - lse).mean()),
                     "; ".join("%s: %d rows, %.4f apart, first token %.1f%%" % (k, v[0], v[1], 100 * v[2])
                               for k, v in kinds.items())))


def row_gaps(rec_a: Recording, path_a: str, row_a: int, rec_b: Recording, path_b: str, row_b: int, rows: int) -> tuple:
    """How far rows of two states files are apart: for each row, the largest difference in a recorded layer against
    the largest value that layer has in the row, and the worst layer of the row."""
    gaps = np.empty(rows, dtype=np.float64)
    finite = True
    step = max(1, (64 << 20) // rec_a.row_bytes)
    for r0 in range(0, rows, step):
        n = min(step, rows - r0)
        a = rec_a.state_rows(path_a, row_a + r0, n)
        b = rec_b.state_rows(path_b, row_b + r0, n)
        finite = finite and bool(np.isfinite(a).all()) and bool(np.isfinite(b).all())
        scale = np.maximum(np.abs(a).max(axis=2), 1e-30)
        gaps[r0:r0 + n] = (np.abs(a - b).max(axis=2) / scale).max(axis=1)
    return gaps, finite


def spread(values: np.ndarray) -> str:
    if values.size == 0:
        return "none"
    return "median %.2e, 99%% %.2e, most %.2e over %d rows" % (
        float(np.median(values)), float(np.quantile(values, 0.99)), float(values.max()), values.size)


FAR_LEAST = 0.02  # nothing nearer than this is called far, however alike two right reads are
FAR_TIMES = 4.0   # far is this many times the widest gap between two right reads


def join(parts: list) -> np.ndarray:
    return np.concatenate(parts) if parts else np.empty(0)


def same_plan(rec: Recording, other: Recording) -> int:
    """Requests of rec that the other recording does not hold with the same tokens."""
    missing = 0
    for rid, e in rec.index.items():
        o = other.index.get(rid)
        if o is None or o["n_tokens"] != e["n_tokens"] or o["shared_len"] != e["shared_len"]:
            missing += 1
    return missing


def check_exact(rec: Recording, ref: Recording, split: Recording, replay, report: Report) -> None:
    """A state depends a little on where the batches that read its request ended, so two right reads are near each
    other and not equal. How near is measured first, between two one-session reads of the same plan: one in whole
    batches, one ended at every cut. Far is a few times the widest gap between those two. No row the run read, and no
    row the model wrote, may be far from the row a plain read gets, and their spread may not be wider than a few times
    the spread between the two right reads."""
    if same_plan(rec, ref) or same_plan(rec, split) or ref.taps != rec.taps or split.taps != rec.taps:
        report.check("read", False, "the references do not hold the recording's requests and layers")
        return

    floor, test = [], []
    finite = True
    worst = (0.0, "")
    for rid, e in rec.index.items():
        rows = e["states_rows"]
        a, ok_a = row_gaps(split, split.req(split.index[rid], ".states"), 0, ref, ref.req(ref.index[rid], ".states"), 0, rows)
        b, ok_b = row_gaps(rec, rec.req(e, ".states"), 0, ref, ref.req(ref.index[rid], ".states"), 0, rows)
        finite = finite and ok_a and ok_b
        floor.append(a[a > 0])  # the rows the two right reads did split differently
        test.append(b)
        if b.size and float(b.max()) > worst[0]:
            worst = (float(b.max()), rid)
    floor, test = join(floor), join(test)
    if floor.size < 100:
        report.check("read", False, "no floor: the two one-session reads differ on %d rows only" % floor.size)
        return
    far = max(FAR_LEAST, FAR_TIMES * float(floor.max()))

    def judge(name: str, values: np.ndarray, what: str, where: str) -> None:
        n_far = int((values > far).sum())
        ok = finite and values.size > 0 and n_far == 0 and \
            float(np.median(values)) <= 3 * max(float(np.median(floor)), 1e-4) and \
            float(np.quantile(values, 0.99)) <= 3 * max(float(np.quantile(floor, 0.99)), 1e-3)
        report.check(name, ok, "%s: %d of %d rows far (over %.3g), %s%s. Two one-session reads against each other, on the "
                     "rows their batches split differently: %s" % (what, n_far, values.size, far, spread(values), where,
                                                                   spread(floor)))

    prefix = ""
    if rec.keep > 0:
        used = max(e["shared_len"] for e in rec.index.values())
        p, ok_p = row_gaps(rec, os.path.join(rec.path, "prefix.states"), 0, ref, os.path.join(ref.path, "prefix.states"), 0, used)
        finite = finite and ok_p
        prefix = "; the shared prefix, %d rows: most %.2e" % (used, float(p.max()))
        test = np.concatenate([test, p])
    judge("read", test, "rows the run read against one session reading alone", "; farthest in %s%s" % (worst[1], prefix))

    if replay is None:
        return
    if replay.taps != rec.taps:
        report.check("written", False, "the replay holds other layers")
        return
    written = []
    missing = 0
    worst = (0.0, "")
    for rid, e in rec.index.items():
        for s in e["segments"]:
            if s["n_states"] == 0:
                continue
            key = "%s@%d" % (rid, s["cut"])
            r = replay.index.get(key)
            if r is None or r["n_tokens"] != s["cut"] + s["n_states"] or r["shared_len"] != e["shared_len"]:
                missing += 1
                continue
            g, ok_g = row_gaps(rec, rec.req(e, ".gen.states"), s["states_row0"],
                               replay, replay.req(r, ".states"), s["cut"] - e["shared_len"], s["n_states"])
            finite = finite and ok_g
            written.append(g)
            if float(g.max()) > worst[0]:
                worst = (float(g.max()), key)
    written = join(written)
    if missing or written.size == 0:
        report.check("written", False, "%d continuations are not in the replay, %d rows compared" % (missing, written.size))
        return
    judge("written", written, "rows the model wrote against the same tokens read back as a prompt", "; farthest in %s" % worst[1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gguf-py", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--rec", required=True)
    p.add_argument("--ref", help="the same plan read by one session, no continuation")
    p.add_argument("--ref-split", help="the same, each read ended at every cut")
    p.add_argument("--replay", help="the recording's continuations read back as prompts")
    p.add_argument("--teacher-rows", type=int, default=768)
    p.add_argument("--seed", type=int, default=20261119)
    args = p.parse_args()

    rec = Recording(args.rec)
    report = Report()
    print("recording %s: taps %s, %d bytes a row, top %d, kept prefix %d and %d" % (
        args.rec, rec.taps, rec.row_bytes, rec.topk, rec.meta["k_tools"], rec.meta["k_sys"]), flush=True)
    check_files(rec, report)
    if not report.failed:
        check_teacher(rec, report, args.gguf_py, args.model, args.teacher_rows, args.seed)
        if args.ref and args.ref_split:
            ref, split = Recording(args.ref), Recording(args.ref_split)
            replay = Recording(args.replay) if args.replay else None
            check_files(ref, report)
            check_files(split, report)
            if replay is not None:
                check_files(replay, report)
            if not report.failed:
                check_exact(rec, ref, split, replay, report)
    if report.failed:
        print("failed: %s" % ", ".join(report.failed))
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
