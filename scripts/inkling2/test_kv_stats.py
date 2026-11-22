"""kv_stats.py on synthetic arrays and fabricated state files only: no model, no server."""
import contextlib
import io
import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import kv_stats as ks

F32 = np.float32
HD = 32  # head size of the fabricated caches: a multiple of every block size


# ---------------------------------------------------------------------------------------------------------------------
# a fabricated state, written the way llama_kv_cache::state_write writes it
# ---------------------------------------------------------------------------------------------------------------------

def token_words(n_tokens, marker=True):
    toks = np.arange(100, 100 + n_tokens, dtype="<u4")
    if not marker:
        return toks.tobytes()
    return struct.pack("<3I", ks.TOKEN_NULL, 1, n_tokens) + toks.tobytes() + struct.pack("<I", 0)  # no media


def row_bytes(arr, dtype):
    if dtype == "bf16":  # the top half of each float32
        return (np.ascontiguousarray(arr).astype(F32).view(np.uint32) >> 16).astype("<u2").tobytes()
    return np.ascontiguousarray(arr).astype(dtype).tobytes()


def build_state(k, v, pos=None, *, n_stream=1, stream=0, seq_ids=None, dtype=np.float16, type_id=1, v_trans=0,
                n_rot=(0, 0), n_tokens=None, marker=True, swa=(1, 5), magic=ks.STATE_MAGIC, version=ks.STATE_VERSION,
                tail=b"\xde\xad" * 64):
    """k, v: [layers, cells, heads * head_dim]. Returns the bytes of a slot state file."""
    n_layer, n_cells, width = k.shape
    pos = np.arange(n_cells) if pos is None else pos
    words = token_words(n_cells if n_tokens is None else n_tokens, marker)
    out = struct.pack("<3I", magic, version, len(words) // 4) + words
    out += struct.pack("<I", n_stream)
    for s in range(n_stream):
        if s != stream:
            out += struct.pack("<I", 0)
            continue
        out += struct.pack("<I", n_cells)
        for i in range(n_cells):
            ids = [0] if seq_ids is None else seq_ids[i]
            out += struct.pack("<iI", int(pos[i]), len(ids)) + struct.pack(f"<{len(ids)}i", *ids)
        out += struct.pack("<4I", v_trans, n_layer, *n_rot)
        for arr in (k, v):
            for layer in range(n_layer):
                data = row_bytes(arr[layer], dtype)
                out += struct.pack("<iQ", type_id, len(data) // n_cells) + data
    out += struct.pack("<I", swa[0]) + struct.pack("<I", swa[1])  # the window cache and the rest: not read
    return out + tail


def make_kv(rng, n_layer, n_cells, n_heads, k_profile=None, v_profile=None):
    """K clipped to the L2 bound, V RMS normalized with eps 1e-6, as the graph stores them."""
    shape = (n_layer, n_cells, n_heads, HD)
    kp = np.ones((n_layer, 1, n_heads, HD)) if k_profile is None else k_profile
    vp = np.ones((n_layer, 1, n_heads, HD)) if v_profile is None else v_profile
    k = rng.standard_normal(shape) * kp * 3.0
    k *= np.minimum(1.0, ks.K_NORM_MAX / (np.linalg.norm(k, axis=-1, keepdims=True) + 1e-6))
    v = rng.standard_normal(shape) * vp
    v /= np.sqrt((v * v).mean(axis=-1, keepdims=True) + 1e-6)
    return (k.reshape(n_layer, n_cells, -1).astype(np.float16), v.reshape(n_layer, n_cells, -1).astype(np.float16))


def write(directory, name, blob):
    path = Path(directory) / name
    path.write_bytes(blob)
    return path


def cfg_for(*extra, states=("x",)):
    return ks.parse_args([*states, "--head-dim", str(HD), "-q", *extra])


# ---------------------------------------------------------------------------------------------------------------------
# the formats
# ---------------------------------------------------------------------------------------------------------------------

def e4m3_grid():
    """Every finite non-negative E4M3 value, from the bit patterns: the index is the code."""
    vals = []
    for code in range(127):  # 0x7f is the NaN
        e, m = code >> 3, code & 7
        vals.append(m / 8.0 * 2.0 ** -6 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7))
    return np.array(vals)


def e4m3_ref(v, grid):
    a = abs(float(v))
    sign = math.copysign(1.0, float(v))
    if a >= grid[-1]:
        return sign * 448.0  # the cast saturates
    i = int(np.searchsorted(grid, a))  # grid[i - 1] < a <= grid[i]
    if i == 0:
        return 0.0
    lo, hi = grid[i - 1], grid[i]
    if a - lo < hi - a:
        return sign * lo
    if hi - a < a - lo:
        return sign * hi
    return sign * (lo if (i - 1) % 2 == 0 else hi)  # a tie goes to the even code


def q8_ref(block):
    """quantize_row_q8_0_ref and dequantize_row_q8_0, one block, scalar float32 arithmetic."""
    amax = F32(0)
    for x in block:
        amax = max(amax, abs(F32(x)))
    d = F32(amax / F32(127))
    inv = F32(F32(1) / d) if d != 0 else F32(0)
    d16 = F32(np.float16(d))
    out = []
    for x in block:
        x0 = F32(F32(x) * inv)
        out.append(F32(math.copysign(math.floor(abs(float(x0)) + 0.5), float(x0)) * d16))  # roundf
    return np.array(out, F32)


def q4_ref(block):
    """quantize_row_q4_0_ref and dequantize_row_q4_0, one block."""
    amax, big = F32(0), F32(0)
    for x in block:
        if amax < abs(F32(x)):
            amax, big = abs(F32(x)), F32(x)
    d = F32(big / F32(-8))
    inv = F32(F32(1) / d) if d != 0 else F32(0)
    d16 = F32(np.float16(d))
    return np.array([F32((min(15, int(F32(F32(F32(x) * inv) + F32(8.5)))) - 8) * d16) for x in block], F32)


class Formats(unittest.TestCase):
    def test_e4m3_known_values(self):
        x = np.array([0.3, 17.0, 450.0, 1000.0, 0.01, 2.0 ** -10, 1.5 * 2.0 ** -9, 1e-4, -0.3, 0.0, 15.9], F32)
        want = [0.3125, 16.0, 448.0, 448.0, 0.009765625, 0.0, 2.0 ** -8, 0.0, -0.3125, 0.0, 16.0]
        np.testing.assert_array_equal(ks.fmt_e4m3(x), np.array(want, F32))

    def test_e4m3_matches_the_grid_with_ties_to_even(self):
        grid = e4m3_grid()
        self.assertEqual(grid.max(), 448.0)
        self.assertEqual(grid[1], 2.0 ** -9)
        rng = np.random.default_rng(3)
        x = np.concatenate([rng.standard_normal(4000) * s for s in (1e-3, 1e-2, 0.1, 1, 10, 100)]).astype(F32)
        mids = ((grid[:-1] + grid[1:]) / 2).astype(F32)  # every tie, exact in float32
        x = np.concatenate([x, mids, -mids, grid.astype(F32), np.array([447, 448, 449, 464, 470, 1e6], F32)])
        want = np.array([e4m3_ref(v, grid) for v in x], F32)
        np.testing.assert_array_equal(ks.fmt_e4m3(x), want)

    def test_q8_0_known_blocks(self):
        b = np.zeros(32, F32)
        b[:5] = [127, 63.5, -63.5, 0.5, -0.5]  # amax 127: d = 1, and roundf takes the middle away from zero
        got = ks.fmt_q8_0(b[None, :])[0]
        np.testing.assert_array_equal(got[:5], [127, 64, -64, 1, -1])
        c = np.zeros(32, F32)
        c[0], c[1] = 1.0, 0.5  # d = 1/127 kept as F16
        d16 = F32(np.float16(F32(1) / F32(127)))
        np.testing.assert_array_equal(ks.fmt_q8_0(c[None, :])[0][:2], [F32(127) * d16, F32(64) * d16])
        self.assertFalse(ks.fmt_q8_0(np.zeros((1, 32), F32)).any())  # an all-zero block stays zero

    def test_q8_0_and_q4_0_match_the_c_code(self):
        rng = np.random.default_rng(5)
        blocks = [(rng.standard_normal(32) * s).astype(F32) for s in (1e-3, 0.05, 1, 3, 22) for _ in range(60)]
        blocks.append(np.full(32, 0.49999997, F32))  # floor(x + 0.5) gets this wrong in float32
        arr = np.stack(blocks)
        np.testing.assert_array_equal(ks.fmt_q8_0(arr), np.stack([q8_ref(b) for b in blocks]))
        np.testing.assert_array_equal(ks.fmt_q4_0(arr), np.stack([q4_ref(b) for b in blocks]))

    def test_q4_0_known_block(self):
        b = np.zeros(32, F32)
        b[:4] = [-8, 7, 7.99, 0.6]  # the largest magnitude is -8: d = 1; 7.99 clips to code 15
        np.testing.assert_array_equal(ks.fmt_q4_0(b[None, :])[0][:4], [-8, 7, 7, 1])

    def test_fp4_grid_ties_and_block_scale(self):
        b = np.array([6, .25, .75, 1.25, 1.75, 2.5, 3.5, 5.0, -.25, -.75, -1.25, -1.75, -2.5, -3.5, -5.0, 0.1], F32)
        got = ks.fmt_fp4(b[None, :])[0]  # amax 6: scale exactly 1; the middle goes to the even code
        np.testing.assert_array_equal(got, [6, 0, 1, 1, 2, 2, 4, 4, 0, -1, -1, -2, -2, -4, -4, 0])
        c = np.zeros(16, F32)
        c[0] = 6.5  # scale 6.5 / 6 = 1.0833 -> E4M3 1.125; 6.5 / 1.125 = 5.78 -> 6; read back 6 * 1.125
        self.assertEqual(ks.fmt_fp4(c[None, :])[0][0], F32(6.75))
        self.assertFalse(ks.fmt_fp4(np.zeros((1, 16), F32)).any())

    def test_relative_error_of_the_formats_is_in_the_expected_range(self):
        x = np.random.default_rng(1).standard_normal((2000, 128)).astype(F32)
        err = {f.name: np.linalg.norm(f.fn(x) - x, axis=1) / np.linalg.norm(x, axis=1) for f in ks.FORMATS}
        self.assertTrue(0.015 < err["fp8_e4m3"].mean() < 0.035)  # 3 mantissa bits: half step 2^-4 at worst
        self.assertTrue(0.003 < err["q8_0"].mean() < 0.012)
        self.assertTrue(0.05 < err["fp4_e2m1_b16"].mean() < 0.15)
        self.assertTrue(0.05 < err["q4_0"].mean() < 0.2)


class Scales(unittest.TestCase):
    def test_scale_has_geometric_mean_one_and_the_profile_ratios(self):
        rms = np.tile(np.array([1.0, 2.0, 4.0, 8.0], F32), (3, 8))  # [heads, channels]
        s = ks.make_scale(rms)
        np.testing.assert_allclose(np.exp(np.log(s).mean(axis=-1)), 1.0, rtol=1e-6)
        np.testing.assert_allclose(s[:, 1] / s[:, 0], 2.0, rtol=1e-6)

    def test_dead_channel_gets_a_finite_scale(self):
        rms = np.ones((1, 16), F32)
        rms[0, 3] = 0.0
        s = ks.make_scale(rms)
        self.assertTrue(np.isfinite(s).all() and (s > 0).all())

    def test_channel_scale_helps_a_block_format_with_one_large_channel(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((4000, 32)).astype(F32)
        x[:, 5] *= 40.0  # a hot channel decides every block's scale
        rms = np.sqrt((x * x).mean(axis=0))[None, :]
        s = ks.make_scale(rms)[0]
        rel = lambda y: np.linalg.norm(y - x, axis=1).mean() / np.linalg.norm(x, axis=1).mean()
        plain, scaled = rel(ks.fmt_q8_0(x)), rel(ks.fmt_q8_0(x / s) * s)
        self.assertLess(scaled, plain)
        # E4M3 keeps a relative error in every value, so it gains nothing from the scale
        self.assertAlmostEqual(rel(ks.fmt_e4m3(x)), rel(ks.fmt_e4m3(x / s) * s), delta=0.003)


class ProfileMeasures(unittest.TestCase):
    def test_pearson(self):
        a = np.arange(10.0)[None, :]
        np.testing.assert_allclose(ks.pearson(a, 3 * a + 1), [1.0])
        np.testing.assert_allclose(ks.pearson(a, -a), [-1.0])
        self.assertTrue(np.isnan(ks.pearson(a, np.ones((1, 10)))[0]))  # a flat profile has no correlation

    def test_top_overlap(self):
        a = np.arange(32.0)[None, :]
        np.testing.assert_allclose(ks.top_overlap(a, a), [1.0])
        np.testing.assert_allclose(ks.top_overlap(a, a[:, ::-1]), [0.0])
        b = a.copy()
        b[0, 31], b[0, 0] = 0.0, 100.0  # one of the 8 largest replaced
        np.testing.assert_allclose(ks.top_overlap(a, b), [7 / 8])

    def test_static_fit_with_a_known_split(self):
        rng = np.random.default_rng(4)
        d = 4096
        base = rng.standard_normal(d)  # static profile of std 1 in log2 units
        for noise, want in ((0.0, 1.0), (0.5, 1.0 / 1.25), (1.0, 0.5)):
            groups = np.stack([2.0 ** (base + noise * rng.standard_normal(d)) for _ in range(40)])[:, None, :]
            share, spread, drift = ks.static_fit(groups)
            self.assertAlmostEqual(float(share[0]), want, delta=0.03)
            self.assertAlmostEqual(float(spread[0]), 1.0, delta=0.06)
            self.assertAlmostEqual(float(drift[0]), noise, delta=0.06)

    def test_static_fit_ignores_the_level_of_a_group(self):
        base = np.random.default_rng(6).standard_normal(64)
        groups = np.stack([2.0 ** (base + shift) for shift in (-3.0, 0.0, 2.5)])[:, None, :]
        share, _, drift = ks.static_fit(groups)
        self.assertAlmostEqual(float(share[0]), 1.0, places=9)
        self.assertAlmostEqual(float(drift[0]), 0.0, places=9)
        share, _, _ = ks.static_fit(groups, log=False)
        self.assertAlmostEqual(float(share[0]), 1.0, places=9)

    def test_independent_profiles_explain_nothing(self):
        rng = np.random.default_rng(8)
        groups = 2.0 ** rng.standard_normal((3, 1, 4096))
        c = ks.compare_profiles(groups)
        self.assertLess(float(c["static_share_log"][0]), 0.45)  # 1/3 for three independent groups
        self.assertLess(abs(float(c["r_log"].mean())), 0.06)
        self.assertEqual(c["pairs"], [(0, 1), (0, 2), (1, 2)])

    def test_hot_channels_in_every_group_and_in_some(self):
        rms = np.ones((3, 2, 32), F32)  # 3 groups, 2 heads
        rms[:, 0, 3] = 5.0  # head 0: channel 3 hot in every group
        rms[1, 0, 7] = 4.0  # ... and channel 7 in one group
        rms[2, 1, 9] = 3.0  # head 1: channel 9 in one group only
        c = ks.compare_profiles(rms)
        np.testing.assert_array_equal(c["hot_all"], [1, 0])
        np.testing.assert_array_equal(c["hot_any"], [2, 1])
        s = ks.summarize_compare(c, 100)
        self.assertEqual((s["hot_all"], s["hot_any"]), (1, 3))

    def test_channel_spread(self):
        rms = np.ones((2, 16), F32)
        rms[1, 9] = 12.0
        ratio, ch = ks.channel_spread(rms)
        np.testing.assert_allclose(ratio, [1.0, 12.0])
        self.assertEqual(int(ch[1]), 9)


# ---------------------------------------------------------------------------------------------------------------------
# the file
# ---------------------------------------------------------------------------------------------------------------------

class Parser(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.rng = np.random.default_rng(11)
        self.k, self.v = make_kv(self.rng, 3, 50, 2)

    def parse(self, blob, name="s.state"):
        return ks.KVState(write(self.dir.name, name, blob), HD)

    def test_round_trip(self):
        pos = self.rng.permutation(50) + 7  # cells are not in position order after restores
        st = self.parse(build_state(self.k, self.v, pos))
        self.assertEqual((st.cell_count, st.n_layer, st.n_heads, st.head_dim, st.type_name), (50, 3, 2, HD, "f16"))
        np.testing.assert_array_equal(st.pos, pos)
        self.assertEqual(st.swa, (1, 5))
        self.assertEqual(len(st.tokens), 50)
        for kind, arr in (("K", self.k), ("V", self.v)):
            for layer in range(3):
                got = st.read(kind, layer, 0, 50)
                np.testing.assert_array_equal(got.reshape(50, -1), arr[layer].astype(F32))
                part = st.read(kind, layer, 13, 29)  # a chunk in the middle
                np.testing.assert_array_equal(part.reshape(16, -1), arr[layer][13:29].astype(F32))
        info = st.info()
        self.assertEqual((info["cells"], info["pos_min"], info["pos_max"]), (50, int(pos.min()), int(pos.max())))

    def test_thirds_follow_position_not_file_order(self):
        pos = np.arange(60)[::-1].copy()  # the file lists the last position first
        k, v = make_kv(self.rng, 1, 60, 1)
        st = self.parse(build_state(k, v, pos))
        third = st.third_index()
        self.assertEqual(list(third[:20]), [2] * 20)
        self.assertEqual(list(third[-20:]), [0] * 20)
        self.assertEqual(sorted(np.bincount(third)), [20, 20, 20])

    def test_stream_other_than_the_first_and_plain_token_list(self):
        st = self.parse(build_state(self.k, self.v, n_stream=4, stream=2, marker=False, n_tokens=7))
        self.assertEqual((st.n_stream, st.stream, st.cell_count), (4, 2, 50))
        self.assertEqual(list(st.tokens), list(range(100, 107)))
        np.testing.assert_array_equal(st.read("K", 1, 0, 5).reshape(5, -1), self.k[1][:5].astype(F32))

    def test_tokens_of_the_server(self):
        st = self.parse(build_state(self.k, self.v, n_tokens=9))
        self.assertEqual(list(st.tokens), list(range(100, 109)))

    def test_cells_with_two_sequence_ids_take_the_slow_path(self):
        ids = [[0, 3] if i % 7 == 0 else [0] for i in range(50)]
        st = self.parse(build_state(self.k, self.v, seq_ids=ids))
        np.testing.assert_array_equal(st.pos, np.arange(50))
        np.testing.assert_array_equal(st.read("V", 2, 0, 50).reshape(50, -1), self.v[2].astype(F32))

    def test_f32_and_bf16_rows(self):
        st = self.parse(build_state(self.k, self.v, dtype=np.float32, type_id=0))
        self.assertEqual(st.type_name, "f32")
        np.testing.assert_array_equal(st.read("K", 0, 0, 50).reshape(50, -1), self.k[0].astype(F32))
        st = self.parse(build_state(self.k, self.v, dtype="bf16", type_id=30), "bf.state")
        self.assertEqual(st.type_name, "bf16")
        want = (self.k[0].astype(F32).view(np.uint32) & 0xFFFF0000).view(F32)  # what 8 mantissa bits keep
        np.testing.assert_array_equal(st.read("K", 0, 0, 50).reshape(50, -1), want)

    def test_rotated_on_store_parses_with_the_rotation_reported(self):
        # a quantized cache stores its values rotated on store; the reader works in that stored domain
        st = self.parse(build_state(self.k, self.v, n_rot=(128, 0)))
        self.assertEqual((st.n_rot_k, st.n_rot_v), (128, 0))
        np.testing.assert_array_equal(st.read("K", 1, 0, 50).reshape(50, -1), self.k[1].astype(F32))  # as stored
        self.assertEqual(st.info()["n_rot_k"], 128)  # reported by info()

    def test_wrong_files_are_refused_with_a_reason(self):
        good = build_state(self.k, self.v)
        cases = [("magic", build_state(self.k, self.v, magic=0x12345678), "not a sequence state"),
                 ("version", build_state(self.k, self.v, version=3), "version 3"),
                 ("transposed", build_state(self.k, self.v, v_trans=1), "transposed"),
                 ("q8_0", build_state(self.k, self.v, type_id=8), "not F32, F16 or BF16"),
                 ("truncated", good[:len(good) // 2], "ends inside"),
                 ("empty", good[:8], "ends inside")]
        for name, blob, text in cases:
            with self.subTest(name), self.assertRaisesRegex(ks.StateError, text):
                self.parse(blob, name + ".state")
        with self.assertRaisesRegex(ks.StateError, "whole number of heads"):
            ks.KVState(write(self.dir.name, "hd.state", good), 48)

    def test_absurd_cell_count_is_refused_without_allocating(self):
        good = bytearray(build_state(self.k, self.v))
        at = 12 + len(token_words(50)) + 4  # the first stream's cell count
        good[at:at + 4] = struct.pack("<I", 4_000_000_000)
        with self.assertRaisesRegex(ks.StateError, "cells claimed"):
            self.parse(bytes(good))

    def test_no_cells_and_two_streams_are_refused(self):
        empty = struct.pack("<3I", ks.STATE_MAGIC, ks.STATE_VERSION, 0) + struct.pack("<I", 1) + struct.pack("<I", 0)
        with self.assertRaisesRegex(ks.StateError, "no cells"):
            self.parse(empty + b"\0" * 16)
        two = build_state(self.k, self.v, n_stream=2, stream=0)
        # a second stream with cells: replace its zero count (the 4 bytes after the first stream's data) with 1
        k2, v2 = make_kv(self.rng, 3, 5, 2)
        first = build_state(self.k, self.v, n_stream=1)
        second = build_state(k2, v2, n_stream=1)
        head = len(token_words(50)) + 12 + 4  # up to the first stream's cell count
        body1 = first[head:len(first) - 64 - 8]
        body2 = second[len(token_words(5)) + 12 + 4:len(second) - 64 - 8]
        blob = first[:12 + len(token_words(50))] + struct.pack("<I", 2) + body1 + body2 + b"\0" * 80
        with self.assertRaisesRegex(ks.StateError, "two streams"):
            self.parse(blob)
        self.assertGreater(len(two), 0)


# ---------------------------------------------------------------------------------------------------------------------
# statistics and the self check
# ---------------------------------------------------------------------------------------------------------------------

class Statistics(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.rng = np.random.default_rng(21)
        n_layer, self.n, self.h = 2, 90, 2
        self.kp = 2.0 ** self.rng.uniform(-2, 1, (n_layer, 1, self.h, HD))  # a known channel profile
        self.vp = 2.0 ** self.rng.uniform(-1, 1, (n_layer, 1, self.h, HD))
        self.k, self.v = make_kv(self.rng, n_layer, self.n, self.h, self.kp, self.vp)
        self.pos = self.rng.permutation(self.n)

    def state(self, k=None, v=None, name="s.state"):
        blob = build_state(self.k if k is None else k, self.v if v is None else v, self.pos)
        return ks.KVState(write(self.dir.name, name, blob), HD)

    def test_channel_statistics_against_numpy(self):
        st = self.state()
        out = ks.collect_stats(st, cfg_for())
        order = np.argsort(self.pos)
        for li in range(2):
            for kind, arr in (("K", self.k), ("V", self.v)):
                got = out[li][kind]
                x = arr[li].astype(np.float64).reshape(self.n, self.h, HD)
                np.testing.assert_allclose(got["max_abs"], np.abs(x).max(axis=0), rtol=1e-6)
                np.testing.assert_allclose(got["rms"], np.sqrt((x * x).mean(axis=0)), rtol=1e-5)
                xs = x[order]
                for t in range(3):
                    part = xs[t * 30:(t + 1) * 30]  # 90 cells: exact thirds by position
                    np.testing.assert_allclose(got["rms_thirds"][t], np.sqrt((part * part).mean(axis=0)), rtol=1e-5)
                self.assertEqual(got["n_cells"], self.n)
                self.assertEqual(got["max_abs_all"], float(np.abs(x).max()))
                self.assertAlmostEqual(got["frac_lt_2m6"], float((np.abs(x) < 2.0 ** -6).mean()), places=12)
                self.assertAlmostEqual(got["frac_lt_2m10"], float((np.abs(x) < 2.0 ** -10).mean()), places=12)
                norms = np.linalg.norm(x, axis=2) if kind == "K" else np.sqrt((x * x).mean(axis=2))
                self.assertAlmostEqual(got["vec_max"], float(norms.max()), places=5)
                self.assertAlmostEqual(got["vec_min"], float(norms.min()), places=5)
                self.assertEqual(got["violations"], 0)

    def test_rotated_on_store_state_runs_statistics_and_holds_its_bounds(self):
        # A deployed quantized cache stores its values rotated (Hadamard on store: K mixes its whole head,
        # V within groups of values; the real model carries n_rot_k 128 and n_rot_v 64 at head size 128).
        # An orthonormal rotation preserves per-vector norms, so the bounds hold and the vector-level
        # statistics match the unrotated state's exactly; per-channel figures do not carry across and are
        # deliberately not asserted here. The synthetic fixture keeps K-only rotation at its own head size.
        def hadamard(n):
            h = np.ones((1, 1), np.float32)
            while h.shape[0] < n:
                h = np.kron(h, np.array([[1.0, 1.0], [1.0, -1.0]], np.float32))
            return h / np.sqrt(float(n))

        kl = self.k.astype(np.float32)                      # (n_layer, cells, heads * head_dim), rows as stored
        per = kl.shape[-1] // self.h                        # values a head: the rotation mixes one whole head
        kr3 = kl.reshape(kl.shape[0], kl.shape[1], self.h, per) @ hadamard(per)
        kr = kr3.reshape(kl.shape).astype(self.k.dtype)
        blob = build_state(kr, self.v, self.pos, n_rot=(per, 0))
        st = ks.KVState(write(self.dir.name, "rot.state", blob), HD)
        rot = ks.collect_stats(st, cfg_for())
        self.assertEqual((st.n_rot_k, st.n_rot_v), (HD, 0))
        for li in range(len(rot)):
            x = kr[li].astype(np.float64).reshape(self.n, self.h, per)  # the stored values are what the reader sees
            norms = np.linalg.norm(x, axis=2)                           # one vector a cell and head, as everywhere else
            got = rot[li]["K"]
            self.assertAlmostEqual(got["vec_max"], float(norms.max()), places=5)
            self.assertAlmostEqual(got["vec_min"], float(norms.min()), places=5)
            self.assertEqual(got["n_cells"], self.n)
            self.assertEqual(got["violations"], 0)
            self.assertEqual(rot[li]["V"]["violations"], 0)

    def test_chunking_does_not_change_the_numbers(self):
        st = self.state()
        a = ks.collect_stats(st, cfg_for("--chunk-cells", "7"))
        b = ks.collect_stats(st, cfg_for("--chunk-cells", "4096"))
        for li in range(2):
            for kind in "KV":
                for key in ("max_abs", "rms", "rms_thirds"):
                    np.testing.assert_allclose(a[li][kind][key], b[li][kind][key], rtol=1e-6)
                self.assertEqual(a[li][kind]["hist_log2"], b[li][kind]["hist_log2"])

    def test_magnitude_shares_of_known_values(self):
        k = np.zeros((1, 20, HD), np.float16)
        k[0, :, :] = 1.0
        k[0, :10, :16] = 0.01  # |x| < 2^-6: 160 of 640 elements
        k[0, :5, 16:20] = 0.0  # 20 exact zeros, which are below any threshold too
        k[0, 5:8, 20:24] = 2.0 ** -11  # below 2^-10
        v = np.ones((1, 20, HD), np.float16)
        st = ks.KVState(write(self.dir.name, "m.state", build_state(k, v)), HD)
        got = ks.collect_stats(st, cfg_for("--selfcheck", "off"))[0]["K"]
        x = k[0].astype(np.float64)
        self.assertAlmostEqual(got["frac_lt_2m6"], (np.abs(x) < 2.0 ** -6).mean(), places=12)
        self.assertAlmostEqual(got["frac_lt_2m10"], (np.abs(x) < 2.0 ** -10).mean(), places=12)
        self.assertAlmostEqual(got["frac_zero"], 20 / 640, places=12)

    def test_valid_cache_passes_the_self_check(self):
        out = ks.collect_stats(self.state(), cfg_for())
        self.assertLessEqual(out[0]["K"]["vec_max"], ks.K_NORM_MAX * (1 + ks.K_TOL))
        self.assertAlmostEqual(out[1]["V"]["vec_min"], 1.0, delta=2e-3)
        self.assertAlmostEqual(out[1]["V"]["vec_max"], 1.0, delta=2e-3)

    def test_k_over_the_clip_names_layer_cell_and_head(self):
        k = self.k.copy().reshape(2, self.n, self.h, HD)
        k[1, 17, 1, :] *= 5.0 / np.linalg.norm(k[1, 17, 1, :].astype(np.float64))  # norm 5 ...
        k[1, 17, 1, :] *= 6.0  # ... times 6: 30 > 22.63
        st = self.state(k=k.reshape(2, self.n, -1))
        with self.assertRaises(ks.SelfCheckError) as cm:
            ks.collect_stats(st, cfg_for("--layer-ids", "5,11"))
        msg = str(cm.exception)
        for part in ("K self-check failed", "model layer 11", "cell row 17", f"position {int(self.pos[17])}", "kv head 1",
                     "L2 norm"):
            self.assertIn(part, msg)
        self.assertIsInstance(cm.exception, AssertionError)

    def test_v_off_by_five_percent_fails_and_warn_mode_lists_it(self):
        v = self.v.copy().reshape(2, self.n, self.h, HD)
        v[0, 3, 0, :] = (v[0, 3, 0, :].astype(F32) * 1.05).astype(np.float16)
        st = self.state(v=v.reshape(2, self.n, -1))
        with self.assertRaisesRegex(ks.SelfCheckError, r"(?s)V self-check failed.*cell row 3.*kv head 0"):
            ks.collect_stats(st, cfg_for())
        out = ks.collect_stats(st, cfg_for("--selfcheck", "warn"))
        self.assertEqual(out[0]["V"]["violations"], 1)
        self.assertEqual(out[0]["V"]["offenders"][0]["cell"], 3)
        self.assertEqual(out[1]["V"]["violations"], 0)
        ks.collect_stats(st, cfg_for("--selfcheck", "off"))  # off: no check, no failure

    def test_v_with_tiny_activations_is_low_but_not_a_violation(self):
        # eps is added to the mean square: a token whose V was tiny comes out under 1 (0.994 seen in Mini), and 0.9 is the floor
        v = self.v.copy().reshape(2, self.n, self.h, HD)
        v[0, 5, 1, :] = (v[0, 5, 1, :].astype(F32) * 0.95).astype(np.float16)  # RMS 0.95
        out = ks.collect_stats(self.state(v=v.reshape(2, self.n, -1)), cfg_for())
        self.assertAlmostEqual(out[0]["V"]["vec_min"], 0.95, delta=2e-3)
        self.assertEqual(out[0]["V"]["violations"], 0)
        self.assertAlmostEqual(out[0]["V"]["edge_share"], 1.0 / (self.n * self.h), places=12)  # counted as under 0.99
        v[0, 6, 0, :] = (v[0, 6, 0, :].astype(F32) * 0.5).astype(np.float16)  # RMS 0.5: not the eps effect
        with self.assertRaisesRegex(ks.SelfCheckError, r"(?s)cell row 6.*kv head 0"):
            ks.collect_stats(self.state(v=v.reshape(2, self.n, -1)), cfg_for())
        out = ks.collect_stats(self.state(v=v.reshape(2, self.n, -1)), cfg_for("--v-rms-min", "0.4"))  # a wider bound
        self.assertEqual(out[0]["V"]["violations"], 0)

    def test_share_of_k_vectors_at_the_clip(self):
        k = self.k.copy().reshape(2, self.n, self.h, HD)
        k[0, :10] = (k[0, :10].astype(F32) * (ks.K_NORM_MAX / np.linalg.norm(k[0, :10].astype(F32), axis=-1, keepdims=True))
                     ).astype(np.float16)  # ten cells with every head at the clip
        out = ks.collect_stats(self.state(k=k.reshape(2, self.n, -1)), cfg_for())
        self.assertGreaterEqual(out[0]["K"]["edge_share"], 10 * self.h / (self.n * self.h))
        self.assertAlmostEqual(out[0]["K"]["vec_max"], ks.K_NORM_MAX, delta=0.02)

    def test_nan_in_k_is_a_violation(self):
        k = self.k.copy()
        k[0, 4, 5] = np.nan
        with self.assertRaises(ks.SelfCheckError):
            ks.collect_stats(self.state(k=k), cfg_for())

    def test_wrong_bytes_fail_at_once(self):
        # V read where K bytes are: unit-variance K rows are not RMS 1 after the 3x of make_kv
        with self.assertRaises(ks.SelfCheckError):
            ks.collect_stats(self.state(v=self.k), cfg_for())
        # random bytes read as F16 give inf and nan
        junk = np.random.default_rng(0).integers(0, 65535, (2, self.n, self.h * HD), dtype=np.uint16).view(np.float16)
        with self.assertRaises(ks.SelfCheckError):
            ks.collect_stats(self.state(k=junk), cfg_for())


# ---------------------------------------------------------------------------------------------------------------------
# the simulation and the whole run
# ---------------------------------------------------------------------------------------------------------------------

class Simulation(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        rng = np.random.default_rng(31)
        n_layer, self.n, self.h = 2, 400, 2
        kp = np.ones((n_layer, 1, self.h, HD))
        kp[..., 5] = 12.0  # one hot channel in every head
        self.k, self.v = make_kv(rng, n_layer, self.n, self.h, kp)
        self.st = ks.KVState(write(self.dir.name, "s.state", build_state(self.k, self.v)), HD)
        self.cfg = cfg_for("--sim-cells", "0")
        self.stats = ks.collect_stats(self.st, self.cfg)

    def test_simulation_matches_the_formats_applied_directly(self):
        sim = ks.simulate_state(self.st, self.stats, self.cfg)
        for kind, arr in (("K", self.k), ("V", self.v)):
            for li in range(2):
                x = arr[li].astype(F32).reshape(self.n, self.h, HD)
                rms = self.stats[li][kind]["rms"]
                s = ks.make_scale(rms)
                for f in ks.FORMATS:
                    for suffix, sc in (("", None), ("+chan", s)):
                        xin = x if sc is None else x / sc
                        y = f.fn(xin.reshape(-1, HD)).reshape(x.shape)
                        y = y if sc is None else y * sc
                        rel = np.linalg.norm(y - x, axis=2) / np.linalg.norm(x, axis=2)
                        m = sim[kind]["layers"][li]["variants"][f.name + suffix]
                        self.assertAlmostEqual(m["rel_l2_mean"], float(rel.mean()), places=6)
                        self.assertAlmostEqual(m["rel_l2_p99"], float(np.percentile(rel.astype(F32), 99)), places=6)
                        self.assertEqual(m["vectors"], self.n * self.h)
        self.assertEqual(sim["K"]["all"]["q8_0"]["vectors"], 2 * self.n * self.h)

    def test_hot_channel_scale_helps_q8_0_on_k_and_a_flat_v_gains_nothing(self):
        sim = ks.simulate_state(self.st, self.stats, self.cfg)
        k = sim["K"]["all"]
        self.assertLess(k["q8_0+chan"]["rel_l2_mean"], 0.9 * k["q8_0"]["rel_l2_mean"])
        self.assertLess(k["fp4_e2m1_b16+chan"]["rel_l2_mean"], k["fp4_e2m1_b16"]["rel_l2_mean"])
        v = sim["V"]["all"]
        self.assertAlmostEqual(v["q8_0+chan"]["rel_l2_mean"], v["q8_0"]["rel_l2_mean"], delta=0.0008)
        self.assertGreater(k["fp8_e4m3"]["rel_l2_mean"], 0.01)
        self.assertEqual(sum(L["variants"]["fp8_e4m3"]["saturated"] for L in sim["K"]["layers"]), 0)

    def test_qk_proxy_is_the_expectation_over_gaussian_q(self):
        sim = ks.simulate_state(self.st, self.stats, self.cfg)
        self.assertNotIn("qk_rel_rms_iso", sim["V"]["layers"][0]["variants"]["q8_0"])
        rng = np.random.default_rng(7)
        n_draws = 3000  # independent draws a head: the closed form has none of their noise
        for li in range(2):
            x = self.k[li].astype(F32).reshape(self.n, self.h, HD)
            rms = self.stats[li]["K"]["rms"]
            w = rms / np.sqrt((rms * rms).mean(axis=-1, keepdims=True))
            dl = ks.fmt_q8_0(x.reshape(-1, HD)).reshape(x.shape) - x
            m = sim["K"]["layers"][li]["variants"]["q8_0"]
            for key, scale in (("qk_rel_rms_iso", np.ones_like(w)), ("qk_rel_rms_prof", w)):
                q = rng.standard_normal((self.h, HD, n_draws)).astype(F32) * scale[:, :, None]
                err = np.matmul(dl.transpose(1, 0, 2), q)
                true = np.matmul(x.transpose(1, 0, 2), q)
                drawn = math.sqrt(float((err * err).sum()) / float((true * true).sum()))
                self.assertAlmostEqual(m[key], drawn, delta=0.06 * drawn, msg=key)
        # for unit variance in every channel the proxy is the relative error of the whole sample's energy
        x = self.k[0].astype(F32).reshape(self.n, self.h, HD)
        dl = ks.fmt_q8_0(x.reshape(-1, HD)).reshape(x.shape) - x
        energy = math.sqrt(float((dl * dl).sum()) / float((x * x).sum()))
        self.assertAlmostEqual(sim["K"]["layers"][0]["variants"]["q8_0"]["qk_rel_rms_iso"], energy, places=6)

    def test_the_sample_is_evenly_spread_and_capped(self):
        cfg = cfg_for("--sim-cells", "100")
        sim = ks.simulate_state(self.st, self.stats, cfg)
        used = sim["K"]["layers"][0]["cells_used"]
        self.assertTrue(80 <= used <= 100, used)  # stride 4 of 400 cells
        self.assertEqual(sim["K"]["layers"][0]["variants"]["q8_0"]["vectors"], used * self.h)

    def test_cross_state_scale_is_offered_from_the_second_state_on(self):
        sim = ks.simulate_state(self.st, self.stats, self.cfg, first_stats=self.stats)
        self.assertIn("q8_0+chan@first", sim["K"]["all"])
        # the first state's own profile is the same profile: the two variants agree
        a, b = sim["K"]["all"]["q8_0+chan"], sim["K"]["all"]["q8_0+chan@first"]
        self.assertAlmostEqual(a["rel_l2_mean"], b["rel_l2_mean"], places=9)

    def test_formats_option_limits_the_variants(self):
        sim = ks.simulate_state(self.st, self.stats, cfg_for("--sim-cells", "0", "--formats", "q8_0"))
        self.assertEqual(sorted(sim["K"]["all"]), ["q8_0", "q8_0+chan"])


class WholeRun(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        rng = np.random.default_rng(41)
        self.paths = []
        for i, sc in enumerate((1.0, 1.4)):  # two "prompts": the same channel profile, a little drifted
            prof = np.ones((2, 1, 2, HD))
            prof[..., 3] = 8.0
            prof *= 1.0 + 0.05 * rng.standard_normal(prof.shape) * i
            k, v = make_kv(rng, 2, 240, 2, prof)
            self.paths.append(str(write(self.dir.name, f"s{i}.state", build_state(k, v, rng.permutation(240)))))

    def run_main(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = ks.main([*args])
        return code, out.getvalue(), err.getvalue()

    def test_two_states_json_and_text(self):
        outdir = Path(self.dir.name) / "out"
        code, text, _ = self.run_main(*self.paths, "--head-dim", str(HD), "--out", str(outdir), "-q", "--sim-cells", "0")
        self.assertEqual(code, 0)
        res = json.loads((outdir / "kv_stats.json").read_text(), parse_constant=lambda c: self.fail("non-finite " + c))
        self.assertEqual((outdir / "kv_stats.txt").read_text().rstrip(), text.rstrip())
        self.assertEqual(len(res["states"]), 2)
        layer = res["states"][0]["layers"][0]
        for kind in "KV":
            self.assertEqual(np.array(layer[kind]["max_abs"]).shape, (2, HD))
            self.assertEqual(np.array(layer[kind]["rms_thirds"]).shape, (3, 2, HD))
            self.assertIn("r_log_min", layer[kind]["a2_thirds"])
            self.assertAlmostEqual(layer[kind]["a2_thirds"]["sampling_floor_log2"], 1.4427 / math.sqrt(2 * 80), places=4)
        self.assertEqual(layer["layer_id"], 5)
        self.assertIn("q8_0+chan@first", res["states"][1]["sim"]["K"]["all"])
        self.assertNotIn("q8_0+chan@first", res["states"][0]["sim"]["K"]["all"])
        cross = res["cross_states"][0]["K"]
        self.assertGreater(cross["r_lin_min"], 0.9)  # the hot channel dominates the linear correlation
        for part in ("M4", "A1", "A2  across depth", "A2  across prompts", "Format simulation, K", "Format simulation, V",
                     "self-check: passed", "q8_0+chan@first", "K vectors at the clip", "floor"):
            self.assertIn(part, text)
        text.encode("ascii")
        (outdir / "kv_stats.json").read_bytes().decode("ascii")

    def test_failing_self_check_exits_3_and_writes_nothing(self):
        bad = Path(self.dir.name) / "bad.state"
        rng = np.random.default_rng(1)
        k, v = make_kv(rng, 2, 40, 2)
        bad.write_bytes(build_state(k, np.zeros_like(v)))  # V rows of zeros: not RMS 1
        outdir = Path(self.dir.name) / "out2"
        code, _, err = self.run_main(str(bad), "--head-dim", str(HD), "--out", str(outdir), "-q")
        self.assertEqual(code, 3)
        self.assertIn("SELF-CHECK FAILED", err)
        self.assertFalse(outdir.exists())

    def test_garbage_values_in_warn_mode_still_render(self):
        rng = np.random.default_rng(2)
        junk = rng.integers(0, 65535, (2, 40, 2 * HD), dtype=np.uint16).view(np.float16)  # inf and nan among them
        path = write(self.dir.name, "junk.state", build_state(junk, junk))
        outdir = Path(self.dir.name) / "o5"
        code, text, _ = self.run_main(str(path), "--head-dim", str(HD), "--out", str(outdir), "-q", "--selfcheck", "warn",
                                      "--no-sim")
        self.assertEqual(code, 0)
        self.assertIn("self-check: FAILED", text)
        json.loads((outdir / "kv_stats.json").read_text(), parse_constant=lambda c: self.fail("non-finite " + c))

    def test_unreadable_state_exits_2(self):
        junk = Path(self.dir.name) / "junk.state"
        junk.write_bytes(b"not a state" * 10)
        code, _, err = self.run_main(str(junk), "-q", "--out", str(Path(self.dir.name) / "o3"))
        self.assertEqual(code, 2)
        self.assertIn("not a sequence state", err)

    def test_missing_file_exits_2(self):
        code, _, err = self.run_main(str(Path(self.dir.name) / "nope.state"), "-q", "--out", str(Path(self.dir.name) / "o6"))
        self.assertEqual(code, 2)
        self.assertIn("nope.state", err)

    def test_one_state_without_simulation(self):
        outdir = Path(self.dir.name) / "o4"
        code, text, _ = self.run_main(self.paths[0], "--head-dim", str(HD), "--out", str(outdir), "-q", "--no-sim")
        self.assertEqual(code, 0)
        res = json.loads((outdir / "kv_stats.json").read_text())
        self.assertNotIn("sim", res["states"][0])
        self.assertEqual(res["cross_states"], {})
        self.assertNotIn("Format simulation", text)


if __name__ == "__main__":
    unittest.main()
