"""The down projections' K-quants rewritten in blocks of 32 hold the values they held, to one f16 rounding of a scale."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import even_down
from even_down import T, gguf
from gguf.quants import Q4_K, dequantize


def k_blocks(qtype, n, rng):
    """n blocks of random quants, scales and mins under real f16 super-scales"""
    size = gguf.GGML_QUANT_SIZES[qtype][1]
    blocks = rng.integers(0, 256, size=(n, size), dtype=np.uint8)
    blocks[:, 0:2] = rng.uniform(1e-4, 2e-2, size=(n, 1)).astype(np.float16).view(np.uint8)
    blocks[:, 2:4] = rng.uniform(1e-4, 2e-2, size=(n, 1)).astype(np.float16).view(np.uint8)
    return blocks


class EvenDown(unittest.TestCase):
    def check(self, qtype, convert, top):
        blocks = k_blocks(qtype, 4096, np.random.default_rng(1234))
        before = dequantize(blocks, qtype).reshape(-1, 8, 32)
        after = dequantize(convert(blocks), even_down.TARGET[qtype]).reshape(-1, 8, 32)
        # the one rounding of d*sc and dmin*m to f16, half a unit in the last of its 11 bits
        d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32)
        dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32)
        sc, mn = Q4_K.get_scale_min(blocks[:, 4:16])
        bound = (top * d * sc + dmin * mn) * 2.0 ** -11
        self.assertTrue(np.all(np.abs(after - before) <= bound[..., None] + 1e-12))
        # on average a few thousandths of a quantization step
        self.assertLess(np.mean(np.abs(after - before)), 5e-3 * np.mean(d * sc))

    def test_q4_k(self):
        self.check(T.Q4_K, even_down.q4_k_to_q4_1, 15)

    def test_q5_k(self):
        self.check(T.Q5_K, even_down.q5_k_to_q5_1, 31)

    def test_pack(self):
        """a pack's down projections change type and value for value, and nothing else changes"""
        rng = np.random.default_rng(7)
        down = k_blocks(T.Q4_K, 3 * 64 * 2, rng).reshape(3, 64, 2 * 144)  # 3 experts, 64 rows, K = 512
        gate = k_blocks(T.Q4_K, 3 * 64 * 2, rng).reshape(3, 64, 2 * 144)
        norm = rng.standard_normal(512).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "pack.gguf", Path(tmp) / "even.gguf"
            writer = gguf.GGUFWriter(source, arch="inkling2")
            writer.add_uint32("inkling2.block_count", 1)
            writer.add_tensor("blk.0.ffn_down_exps.weight", down, raw_dtype=T.Q4_K)
            writer.add_tensor("blk.0.ffn_gate_up_exps.weight", gate, raw_dtype=T.Q4_K)
            writer.add_tensor("blk.0.ffn_norm.weight", norm)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            subprocess.run([sys.executable, str(Path(even_down.__file__)), str(source), str(output)], check=True,
                           capture_output=True)
            before, after = gguf.GGUFReader(source), gguf.GGUFReader(output)
            self.assertEqual(after.get_field("inkling2.block_count").contents(), 1)
            tensors = {t.name: t for t in after.tensors}
            self.assertEqual(tensors["blk.0.ffn_down_exps.weight"].tensor_type, T.Q4_1)
            self.assertEqual(list(tensors["blk.0.ffn_down_exps.weight"].shape), [512, 64, 3])
            for t in before.tensors:
                if t.name == "blk.0.ffn_down_exps.weight":
                    want = even_down.q4_k_to_q4_1(np.asarray(t.data).reshape(-1, 144)).reshape(-1)
                    self.assertTrue(np.array_equal(np.asarray(tensors[t.name].data).reshape(-1), want))
                else:
                    self.assertEqual(tensors[t.name].tensor_type, t.tensor_type)
                    self.assertTrue(np.array_equal(np.asarray(tensors[t.name].data), np.asarray(t.data)))


if __name__ == "__main__":
    unittest.main()
