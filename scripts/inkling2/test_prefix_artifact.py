"""Unit tests for scripts/inkling2/prefix_artifact.py (doc 19 shared-prefix KV artifact)."""

from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import prefix_artifact as pa


def _write_synthetic_ggsq(
    path: Path,
    *,
    tokens: list[int],
    global_cells: int,
    global_layers: int = 8,
    global_type: int = 1,  # f16
    global_row_bytes: int = 256,
    n_rot_k: int = 0,
    n_rot_v: int = 0,
    swa_cells: int = 16,
    swa_layers: int = 40,
    swa_type: int = 8,  # GGML_TYPE_Q8_0
    swa_row_bytes: int = 136,
) -> Path:
    buf = bytearray()
    # ggsq v4 header + server token list
    words = [pa.TOKEN_NULL, 1, len(tokens)] + list(tokens)
    buf += struct.pack("<3I", pa.STATE_MAGIC, pa.STATE_VERSION, len(words))
    buf += struct.pack(f"<{len(words)}I", *words)

    def append_section(n_cells: int, n_layers: int, t_id: int, row_bytes: int, rk: int, rv: int) -> None:
        buf.extend(struct.pack("<I", 1))  # n_stream = 1
        buf.extend(struct.pack("<I", n_cells))
        for pos in range(n_cells):
            buf.extend(struct.pack("<iIi", pos, 1, 0))
        buf.extend(struct.pack("<4I", 0, n_layers, rk, rv))  # v_trans=0
        for kind_idx in range(2):  # K then V
            for li in range(n_layers):
                buf.extend(struct.pack("<iQ", t_id, row_bytes))
                buf.extend(bytes([(kind_idx + li + j) & 0xFF for j in range(n_cells * row_bytes)]))

    append_section(global_cells, global_layers, global_type, global_row_bytes, n_rot_k, n_rot_v)
    if swa_cells > 0:
        append_section(swa_cells, swa_layers, swa_type, swa_row_bytes, 64, 64)
    path.write_bytes(bytes(buf))
    return path


class PrefixArtifactTest(unittest.TestCase):
    def test_pack_verify_and_pin_refusals(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tokens = [10, 20, 30, 40, 50, 60, 70, 80]
            state = _write_synthetic_ggsq(root / "slot0.ggsq", tokens=tokens, global_cells=len(tokens), swa_cells=8)
            bundle = root / "bundle"
            manifest = pa.pack_bundle(
                state,
                bundle,
                pack_name="Inkling-2-Small-abliterated-e4853745-Q3_K_M-imatrix.gguf",
                template_sha256="aa" * 32,
                model_revision="e4853745aa9cb4587e8959d3c80b92e1bef99a4a",
                engine_commit="2b914b819",
            )
            self.assertEqual(manifest["prefix"]["length"], 8)
            self.assertEqual(manifest["kv"]["global_type_k"], "f16")
            self.assertEqual(manifest["kv"]["swa_type_k"], "q8_0")
            expected_corpus = hashlib.sha256(struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest()
            self.assertEqual(manifest["prefix"]["corpus_sha256"], expected_corpus)

            # Exact match verifies clean
            _, reasons = pa.verify_bundle(
                bundle,
                pack_name="Inkling-2-Small-abliterated-e4853745-Q3_K_M-imatrix.gguf",
                template_sha256="aa" * 32,
                model_revision="e4853745aa9cb4587e8959d3c80b92e1bef99a4a",
                engine_commit="2b914b819",
                kv_type="f16",
                swa_kv_type="q8_0",
                corpus_sha256=expected_corpus,
                max_pool_cells=3014656,
            )
            self.assertEqual(reasons, [])

            # Pin mismatches must refuse loudly
            _, bad_tpl = pa.verify_bundle(bundle, template_sha256="bb" * 32)
            self.assertTrue(any("template_sha256" in r for r in bad_tpl))

            _, bad_kv = pa.verify_bundle(bundle, kv_type="q8_0")
            self.assertTrue(any("global kv type" in r for r in bad_kv))

            _, bad_swa = pa.verify_bundle(bundle, swa_kv_type="f16")
            self.assertTrue(any("swa kv type" in r for r in bad_swa))

            _, bad_rot = pa.verify_bundle(bundle, swa_n_rot_k=128)
            self.assertTrue(any("swa_n_rot_k" in r for r in bad_rot))

            # Verify enum IDs match ggml.h (Q4_0=2, Q8_0=8, F8_E4M3=43)
            self.assertEqual(pa.GGML_TYPES[2], "q4_0")
            self.assertEqual(pa.GGML_TYPES[8], "q8_0")
            self.assertEqual(pa.GGML_TYPES[43], "f8_e4m3")

            _, bad_pool = pa.verify_bundle(bundle, max_pool_cells=4)
            self.assertTrue(any("exceeding runtime pool cap" in r for r in bad_pool))

            _, bad_corpus = pa.verify_bundle(bundle, corpus_sha256="00" * 32)
            self.assertTrue(any("corpus_sha256" in r for r in bad_corpus))

            # Tampering with a single byte of payload must be caught by sha256
            payload = bundle / "state.ggsq"
            raw = bytearray(payload.read_bytes())
            raw[-1] ^= 0xFF
            payload.write_bytes(bytes(raw))
            _, bad_sha = pa.verify_bundle(bundle)
            self.assertTrue(any("sha256 mismatch" in r for r in bad_sha))

    def test_swa_window_overflow_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state = _write_synthetic_ggsq(root / "bigswa.ggsq", tokens=list(range(20)), global_cells=20, swa_cells=20)
            with self.assertRaises(pa.ArtifactError):
                pa.pack_bundle(state, root / "b", pack_name="p.gguf", template_sha256="11" * 32, swa_window=16)

    def test_restore_dry_run_and_http(self):
        received = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                received["path"] = self.path
                received["body"] = json.loads(self.rfile.read(length).decode("utf-8"))
                out = json.dumps({"id_slot": 0, "n_restored": 8}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *args):
                pass

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state = _write_synthetic_ggsq(root / "slot0.ggsq", tokens=[1, 2, 3, 4], global_cells=4, swa_cells=4)
            bundle = root / "bundle"
            pa.pack_bundle(state, bundle, pack_name="pack.gguf", template_sha256="cc" * 32)

            save_dir = root / "slots"
            plan = pa.restore_bundle(
                bundle, save_dir, "http://127.0.0.1:9999", dry_run=True, pack_name="pack.gguf", template_sha256="cc" * 32
            )
            self.assertEqual(plan["prefix_len"], 4)
            self.assertFalse(save_dir.exists())

            srv = HTTPServer(("127.0.0.1", 0), Handler)
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            try:
                res = pa.restore_bundle(
                    bundle,
                    save_dir,
                    f"http://127.0.0.1:{srv.server_port}",
                    pack_name="pack.gguf",
                    template_sha256="cc" * 32,
                    kv_type="f16",
                    swa_kv_type="q8_0",
                )
            finally:
                srv.shutdown()
                srv.server_close()
            self.assertEqual(received["path"], "/slots/0?action=restore")
            self.assertEqual(received["body"]["filename"], res["filename"])
            self.assertTrue((save_dir / res["filename"]).is_file())
            self.assertEqual(res["response"]["n_restored"], 8)

    def test_serve_make_plan_prefix_bundle(self):
        import serve

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state = _write_synthetic_ggsq(root / "slot0.ggsq", tokens=[1, 2, 3, 4], global_cells=4, swa_cells=4)
            bundle = root / "bundle"
            pa.pack_bundle(
                state,
                bundle,
                pack_name=serve.PROFILES["tensor-4"]["pack"],
                template_sha256=serve.TEMPLATE_SHA256,
            )
            args = serve.arguments(["--profile", "tensor-4", "--prefix-bundle", str(bundle)])
            argv, _ = serve.make_plan(args)
            self.assertIn("--slot-save-path", argv)
            self.assertEqual(argv[argv.index("--slot-save-path") + 1], str(bundle.resolve()))

            # Mismatched KV type must refuse cleanly
            bad_args = serve.arguments(["--profile", "tensor-4", "--kv-cache", "q8_0", "--prefix-bundle", str(bundle)])
            with self.assertRaises(ValueError):
                serve.make_plan(bad_args)


if __name__ == "__main__":
    unittest.main()
