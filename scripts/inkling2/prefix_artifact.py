#!/usr/bin/env python3
"""Content-addressed, manifest-pinned shared-prefix KV artifact tool (doc 19).

Packages a saved ggsq v4 slot state file from llama-server (/slots/0?action=save)
into a portable, sha256-verified bundle pinned to model pack, chat template hash,
engine commit, KV/SWA formats, Hadamard rotation, and decoded token provenance.

Subcommands:
  pack    <state_file> <bundle_dir> [pins...]   Inspect ggsq v4, hash in 64 MiB chunks, write manifest.json + state.ggsq
  verify  <bundle_dir> [expected pins...]       Verify manifest, shard sha256s, and every runtime pin (refuse on mismatch)
  restore <bundle_dir> --slot-save-path DIR ... Verify bundle, stage into slot-save-path, and POST /slots/0?action=restore
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

FORMAT_VERSION = 1
STATE_MAGIC = 0x67677371  # 'ggsq'
STATE_VERSION = 4
TOKEN_NULL = 0xFFFFFFFF
CHUNK_BYTES = 64 * 1024 * 1024  # 64 MiB streaming chunks (flat host RAM)

GGML_TYPES = {
    0: "f32",
    1: "f16",
    2: "q4_0",
    3: "q4_1",
    6: "q5_0",
    7: "q5_1",
    8: "q8_0",
    30: "bf16",
    43: "f8_e4m3",
}


class ArtifactError(Exception):
    """Raised when a ggsq state or prefix bundle fails parsing or verification."""


def _sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK_BYTES)
            if not chunk:
                break
            h.update(chunk)
            total += len(chunk)
    return h.hexdigest(), total


def _read_exact(f, n: int, label: str) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise ArtifactError(f"truncated {label} (wanted {n} bytes, got {len(data)})")
    return data


def _skip_stream_cells(f, cell_count: int) -> tuple[int, int]:
    """Skip cell metadata records {i32 pos, u32 n_seq_id, n_seq_id x i32}, returning (pos_min, pos_max)."""
    pos_min = 1 << 62
    pos_max = -(1 << 62)
    for idx in range(cell_count):
        pos, n_seq = struct.unpack("<iI", _read_exact(f, 8, f"cell {idx} header"))
        if n_seq > 4096:
            raise ArtifactError(f"implausible n_seq_id={n_seq} at cell {idx}")
        if n_seq:
            _read_exact(f, 4 * n_seq, f"cell {idx} seq ids")
        if pos < pos_min:
            pos_min = pos
        if pos > pos_max:
            pos_max = pos
    return (pos_min, pos_max) if cell_count > 0 else (-1, -1)


def _parse_kv_section(f, file_size: int, section_name: str) -> dict:
    """Parse one llama_kv_cache state_write section (global or SWA) without loading row payloads."""
    n_stream = struct.unpack("<I", _read_exact(f, 4, f"{section_name}.n_stream"))[0]
    if not 1 <= n_stream <= 4096:
        raise ArtifactError(f"{section_name}: implausible n_stream={n_stream}")

    active_stream = None
    cell_count = 0
    pos_min = -1
    pos_max = -1
    v_trans = 0
    n_layer = 0
    n_rot_k = 0
    n_rot_v = 0
    type_k = None
    type_v = None
    row_bytes_k = 0
    row_bytes_v = 0

    for s in range(n_stream):
        n = struct.unpack("<I", _read_exact(f, 4, f"{section_name}.stream[{s}].cell_count"))[0]
        if n == 0:
            continue
        if active_stream is not None:
            raise ArtifactError(f"{section_name}: cells found in multiple streams ({active_stream} and {s})")
        active_stream = s
        cell_count = n
        pos_min, pos_max = _skip_stream_cells(f, n)
        v_trans, n_layer, n_rot_k, n_rot_v = struct.unpack("<4I", _read_exact(f, 16, f"{section_name}.header"))
        if v_trans > 1 or not 1 <= n_layer <= 4096:
            raise ArtifactError(f"{section_name}: implausible header v_trans={v_trans} n_layer={n_layer}")
        if v_trans != 0:
            raise ArtifactError(f"{section_name}: transposed V is not supported (serve with -fa on)")

        for kind in ("K", "V"):
            seen = set()
            for _ in range(n_layer):
                t_id, row_bytes = struct.unpack("<iQ", _read_exact(f, 12, f"{section_name}.{kind} layer header"))
                seen.add((t_id, row_bytes))
                skip = cell_count * row_bytes
                cur = f.tell()
                if cur + skip > file_size:
                    raise ArtifactError(f"{section_name}.{kind}: truncated payload ({cur + skip} > {file_size})")
                f.seek(cur + skip)
            if len(seen) != 1:
                raise ArtifactError(f"{section_name}.{kind}: heterogeneous layer types/rows {sorted(seen)}")
            t_id, row_bytes = seen.pop()
            t_name = GGML_TYPES.get(t_id, f"type_{t_id}")
            if kind == "K":
                type_k, row_bytes_k = t_name, row_bytes
            else:
                type_v, row_bytes_v = t_name, row_bytes

    return {
        "n_stream": n_stream,
        "stream": active_stream,
        "cell_count": cell_count,
        "pos_min": pos_min,
        "pos_max": pos_max,
        "n_layer": n_layer,
        "n_rot_k": n_rot_k,
        "n_rot_v": n_rot_v,
        "type_k": type_k,
        "type_v": type_v,
        "row_bytes_k": row_bytes_k,
        "row_bytes_v": row_bytes_v,
    }


def inspect_ggsq(path: Path) -> dict:
    """Inspect a ggsq v4 state file in streaming fashion and extract geometry + token provenance."""
    file_size = path.stat().st_size
    with path.open("rb") as f:
        magic, version, n_words = struct.unpack("<3I", _read_exact(f, 12, "ggsq header"))
        if magic != STATE_MAGIC:
            raise ArtifactError(f"{path}: bad magic {magic:#010x} (expected {STATE_MAGIC:#010x})")
        if version != STATE_VERSION:
            raise ArtifactError(f"{path}: unsupported ggsq version {version} (expected {STATE_VERSION})")

        token_raw = _read_exact(f, 4 * n_words, "token table")
        words = struct.unpack(f"<{n_words}I", token_raw) if n_words else ()
        if len(words) >= 3 and words[0] == TOKEN_NULL and words[1] == 1 and 3 + words[2] <= len(words):
            n_tok = words[2]
            tok_bytes = token_raw[12 : 12 + 4 * n_tok]
        else:
            n_tok = len(words)
            tok_bytes = token_raw
        corpus_sha256 = hashlib.sha256(tok_bytes).hexdigest()

        global_sec = _parse_kv_section(f, file_size, "global")
        if global_sec["stream"] is None or global_sec["cell_count"] == 0:
            raise ArtifactError(f"{path}: global KV section contains zero cells")

        swa_sec = None
        if file_size - f.tell() >= 8:
            try:
                swa_sec = _parse_kv_section(f, file_size, "swa")
            except ArtifactError:
                swa_sec = None

    return {
        "file_bytes": file_size,
        "prefix_len": n_tok if n_tok > 0 else global_sec["cell_count"],
        "n_tokens": n_tok,
        "corpus_sha256": corpus_sha256,
        "global": global_sec,
        "swa": swa_sec,
    }


def _default_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return ""


def _canonical_manifest_bytes(manifest: dict) -> bytes:
    copy = {k: v for k, v in manifest.items() if k != "bundle_sha256"}
    return json.dumps(copy, sort_keys=True, separators=(",", ":")).encode("utf-8")


def pack_bundle(
    state_path: Path,
    bundle_dir: Path,
    *,
    pack_name: str,
    template_sha256: str,
    model_revision: str = "",
    tokenizer_sha256: str = "",
    engine_commit: str = "",
    swa_window: int = 1024,
    copy_mode: str = "copy",
) -> dict:
    """Create a content-addressed, manifest-pinned shared-prefix KV bundle."""
    state_path = state_path.resolve()
    bundle_dir = bundle_dir.resolve()
    info = inspect_ggsq(state_path)

    swa = info["swa"]
    if swa is not None and swa["cell_count"] > swa_window:
        raise ArtifactError(
            f"SWA section has {swa['cell_count']} cells, exceeding swa_window={swa_window}"
        )

    bundle_dir.mkdir(parents=True, exist_ok=True)
    payload_name = "state.ggsq"
    dst_payload = bundle_dir / payload_name
    if dst_payload.exists() or dst_payload.is_symlink():
        dst_payload.unlink()

    if copy_mode == "hardlink":
        os.link(state_path, dst_payload)
    elif copy_mode == "symlink":
        os.symlink(state_path, dst_payload)
    else:
        with state_path.open("rb") as src, dst_payload.open("wb") as dst:
            shutil.copyfileobj(src, dst, CHUNK_BYTES)

    payload_sha256, payload_bytes = _sha256_file(dst_payload)
    manifest = {
        "format_version": FORMAT_VERSION,
        "model": {
            "pack_name": pack_name,
            "revision": model_revision,
        },
        "template_sha256": template_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "engine_commit": engine_commit or _default_git_commit(),
        "prefix": {
            "length": info["prefix_len"],
            "n_tokens": info["n_tokens"],
            "corpus_sha256": info["corpus_sha256"],
        },
        "kv": {
            "global_type_k": info["global"]["type_k"],
            "global_type_v": info["global"]["type_v"],
            "global_cells": info["global"]["cell_count"],
            "global_layers": info["global"]["n_layer"],
            "n_rot_k": info["global"]["n_rot_k"],
            "n_rot_v": info["global"]["n_rot_v"],
            "swa_type_k": swa["type_k"] if swa else None,
            "swa_type_v": swa["type_v"] if swa else None,
            "swa_cells": swa["cell_count"] if swa else 0,
            "swa_layers": swa["n_layer"] if swa else 0,
            "swa_n_rot_k": swa["n_rot_k"] if swa else 0,
            "swa_n_rot_v": swa["n_rot_v"] if swa else 0,
        },
        "parts": [
            {
                "name": payload_name,
                "bytes": payload_bytes,
                "sha256": payload_sha256,
            }
        ],
    }
    manifest["bundle_sha256"] = hashlib.sha256(_canonical_manifest_bytes(manifest)).hexdigest()

    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify_bundle(
    bundle_dir: Path,
    *,
    pack_name: str | None = None,
    template_sha256: str | None = None,
    model_revision: str | None = None,
    tokenizer_sha256: str | None = None,
    engine_commit: str | None = None,
    kv_type: str | None = None,
    swa_kv_type: str | None = None,
    n_rot_k: int | None = None,
    n_rot_v: int | None = None,
    swa_n_rot_k: int | None = None,
    swa_n_rot_v: int | None = None,
    corpus_sha256: str | None = None,
    max_pool_cells: int | None = None,
    verify_payload_hashes: bool = True,
) -> tuple[dict, list[str]]:
    """Verify bundle integrity and every requested pin. Returns (manifest, reasons); empty reasons = valid."""
    bundle_dir = bundle_dir.resolve()
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ArtifactError(f"missing manifest.json in {bundle_dir}")

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"corrupt manifest.json: {exc}") from exc

    reasons: list[str] = []
    if manifest.get("format_version") != FORMAT_VERSION:
        reasons.append(f"format_version mismatch: got {manifest.get('format_version')}, expected {FORMAT_VERSION}")

    expected_bundle_sha = hashlib.sha256(_canonical_manifest_bytes(manifest)).hexdigest()
    if manifest.get("bundle_sha256") != expected_bundle_sha:
        reasons.append("bundle_sha256 mismatch (manifest modified)")

    parts = manifest.get("parts")
    if not isinstance(parts, list) or not parts:
        reasons.append("manifest lists no payload parts")
    else:
        for part in parts:
            p_path = bundle_dir / part["name"]
            if not p_path.is_file():
                reasons.append(f"missing part {part['name']}")
                continue
            actual_size = p_path.stat().st_size
            if actual_size != part["bytes"]:
                reasons.append(f"part {part['name']} size mismatch: got {actual_size}, expected {part['bytes']}")
                continue
            if verify_payload_hashes:
                actual_sha, _ = _sha256_file(p_path)
                if actual_sha != part["sha256"]:
                    reasons.append(f"part {part['name']} sha256 mismatch")

    model = manifest.get("model", {})
    kv = manifest.get("kv", {})
    prefix = manifest.get("prefix", {})

    if pack_name is not None and model.get("pack_name") != pack_name:
        reasons.append(f"pack_name mismatch: bundle={model.get('pack_name')!r}, runtime={pack_name!r}")
    if model_revision is not None and model.get("revision") != model_revision:
        reasons.append(f"model revision mismatch: bundle={model.get('revision')!r}, runtime={model_revision!r}")
    if template_sha256 is not None and manifest.get("template_sha256") != template_sha256:
        reasons.append(
            f"template_sha256 mismatch: bundle={manifest.get('template_sha256')}, runtime={template_sha256}"
        )
    if tokenizer_sha256 is not None and manifest.get("tokenizer_sha256") != tokenizer_sha256:
        reasons.append("tokenizer_sha256 mismatch")
    if engine_commit is not None and manifest.get("engine_commit") != engine_commit:
        reasons.append(f"engine_commit mismatch: bundle={manifest.get('engine_commit')}, runtime={engine_commit}")
    if kv_type is not None:
        if kv.get("global_type_k") != kv_type or kv.get("global_type_v") != kv_type:
            reasons.append(
                f"global kv type mismatch: bundle=({kv.get('global_type_k')},{kv.get('global_type_v')}), runtime={kv_type}"
            )
    if swa_kv_type is not None and kv.get("swa_type_k") is not None:
        if kv.get("swa_type_k") != swa_kv_type or kv.get("swa_type_v") != swa_kv_type:
            reasons.append(
                f"swa kv type mismatch: bundle=({kv.get('swa_type_k')},{kv.get('swa_type_v')}), runtime={swa_kv_type}"
            )
    for key, expected in (
        ("n_rot_k", n_rot_k),
        ("n_rot_v", n_rot_v),
        ("swa_n_rot_k", swa_n_rot_k),
        ("swa_n_rot_v", swa_n_rot_v),
    ):
        if expected is not None and kv.get(key) != expected:
            reasons.append(f"kv {key} mismatch: bundle={kv.get(key)}, runtime={expected}")
    if corpus_sha256 is not None and prefix.get("corpus_sha256") != corpus_sha256:
        reasons.append(
            f"corpus_sha256 mismatch: bundle={prefix.get('corpus_sha256')}, expected={corpus_sha256}"
        )
    if max_pool_cells is not None:
        p_len = int(prefix.get("length", 0))
        g_cells = int(kv.get("global_cells", 0))
        need = max(p_len, g_cells)
        if need > max_pool_cells:
            reasons.append(f"prefix needs {need} cells, exceeding runtime pool cap {max_pool_cells}")

    return manifest, reasons


def restore_bundle(
    bundle_dir: Path,
    slot_save_path: Path,
    server_url: str,
    *,
    slot_id: int = 0,
    timeout_s: float = 300.0,
    dry_run: bool = False,
    **verify_kwargs,
) -> dict:
    """Verify bundle pins, stage payload into slot_save_path, and trigger POST /slots/{id}?action=restore."""
    manifest, reasons = verify_bundle(bundle_dir, **verify_kwargs)
    if reasons:
        raise ArtifactError("refusing bundle restore:\n  - " + "\n  - ".join(reasons))

    part_name = manifest["parts"][0]["name"]
    src_payload = bundle_dir.resolve() / part_name
    stage_name = f"prefix-{manifest['bundle_sha256'][:16]}.ggsq"
    slot_save_path = slot_save_path.resolve()
    staged_path = slot_save_path / stage_name
    endpoint = f"{server_url.rstrip('/')}/slots/{slot_id}?action=restore"

    plan = {
        "bundle_sha256": manifest["bundle_sha256"],
        "prefix_len": manifest["prefix"]["length"],
        "staged_file": str(staged_path),
        "endpoint": endpoint,
        "filename": stage_name,
    }
    if dry_run:
        return plan

    slot_save_path.mkdir(parents=True, exist_ok=True)
    if staged_path.exists() or staged_path.is_symlink():
        staged_path.unlink()
    try:
        os.link(src_payload, staged_path)
    except OSError:
        with src_payload.open("rb") as src, staged_path.open("wb") as dst:
            shutil.copyfileobj(src, dst, CHUNK_BYTES)

    body = json.dumps({"filename": stage_name}).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            plan["response"] = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"restore request to {endpoint} failed: {exc}") from exc
    return plan


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Shared-prefix KV artifact packer, verifier, and restorer (doc 19).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pk = sub.add_parser("pack", help="Pack a saved ggsq v4 slot state into a pinned bundle directory")
    pk.add_argument("state_file", type=Path)
    pk.add_argument("bundle_dir", type=Path)
    pk.add_argument("--pack-name", required=True)
    pk.add_argument("--template-sha256", required=True)
    pk.add_argument("--model-revision", default="")
    pk.add_argument("--tokenizer-sha256", default="")
    pk.add_argument("--engine-commit", default="")
    pk.add_argument("--swa-window", type=int, default=1024)
    pk.add_argument("--copy-mode", choices=("copy", "hardlink", "symlink"), default="copy")

    for name in ("verify", "restore"):
        sp = sub.add_parser(name, help=f"{name.title()} a shared-prefix KV bundle")
        sp.add_argument("bundle_dir", type=Path)
        sp.add_argument("--pack-name")
        sp.add_argument("--template-sha256")
        sp.add_argument("--model-revision")
        sp.add_argument("--tokenizer-sha256")
        sp.add_argument("--engine-commit")
        sp.add_argument("--kv-type")
        sp.add_argument("--swa-kv-type")
        sp.add_argument("--corpus-sha256")
        sp.add_argument("--max-pool-cells", type=int)
        sp.add_argument("--skip-payload-hash", action="store_true")
        if name == "restore":
            sp.add_argument("--slot-save-path", type=Path, required=True)
            sp.add_argument("--server-url", default="http://127.0.0.1:8080")
            sp.add_argument("--slot-id", type=int, default=0)
            sp.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.cmd == "pack":
            m = pack_bundle(
                args.state_file,
                args.bundle_dir,
                pack_name=args.pack_name,
                template_sha256=args.template_sha256,
                model_revision=args.model_revision,
                tokenizer_sha256=args.tokenizer_sha256,
                engine_commit=args.engine_commit,
                swa_window=args.swa_window,
                copy_mode=args.copy_mode,
            )
            print(json.dumps({"bundle_sha256": m["bundle_sha256"], "prefix_len": m["prefix"]["length"]}))
            return 0
        verify_kwargs = {
            "pack_name": args.pack_name,
            "template_sha256": args.template_sha256,
            "model_revision": args.model_revision,
            "tokenizer_sha256": args.tokenizer_sha256,
            "engine_commit": args.engine_commit,
            "kv_type": args.kv_type,
            "swa_kv_type": args.swa_kv_type,
            "corpus_sha256": args.corpus_sha256,
            "max_pool_cells": args.max_pool_cells,
            "verify_payload_hashes": not args.skip_payload_hash,
        }
        if args.cmd == "verify":
            m, reasons = verify_bundle(args.bundle_dir, **verify_kwargs)
            if reasons:
                for r in reasons:
                    print(f"FAIL: {r}", file=sys.stderr)
                return 2
            print(json.dumps({"ok": True, "bundle_sha256": m["bundle_sha256"], "prefix_len": m["prefix"]["length"]}))
            return 0
        res = restore_bundle(
            args.bundle_dir,
            args.slot_save_path,
            args.server_url,
            slot_id=args.slot_id,
            dry_run=args.dry_run,
            **verify_kwargs,
        )
        print(json.dumps(res))
        return 0
    except ArtifactError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
