#!/usr/bin/env python3
"""Write a DFlash draft that carries its target's token embeddings and output head.

The public DFlash drafts ship neither tensor and read both from the target while they run. A target split across cards
by -sm tensor holds them in a buffer no draft context can run on, and the server aborts at load ("pre-allocated tensor
(output.weight) in a buffer (Meta(...)) that cannot run the operation"). With its own copies the draft references no
target tensor and runs whole on the card --spec-draft-device names.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402

EMBD, HEAD = "token_embd.weight", "output.weight"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draft", type=Path, required=True, help="The DFlash draft GGUF as published")
    parser.add_argument("--target", type=Path, nargs="+", required=True, help="The target GGUF, every shard of it")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    draft = gguf.GGUFReader(args.draft)
    own = {tensor.name for tensor in draft.tensors}
    shards = [gguf.GGUFReader(path) for path in args.target]
    found = {tensor.name: tensor for shard in shards for tensor in shard.tensors if tensor.name in (EMBD, HEAD)}
    if EMBD not in found:
        raise SystemExit(f"{EMBD} is in none of the target's files: give every shard")
    # a target without an output head ties it to its embeddings, and so does the draft
    carried = [(name, found.get(name, found[EMBD])) for name in (EMBD, HEAD) if name not in own]
    if not carried:
        raise SystemExit("the draft already carries both tensors")

    writer = gguf.GGUFWriter(args.out, arch=draft.fields[gguf.Keys.General.ARCHITECTURE].contents())
    alignment = draft.fields.get(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment.contents()
    for field in draft.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        kind = field.types[0]
        writer.add_key_value(field.name, field.contents(), kind,
                             sub_type=field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None)
    tensors = [(tensor.name, tensor) for tensor in draft.tensors] + carried
    for name, tensor in tensors:
        writer.add_tensor_info(name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for _, tensor in tensors:
        writer.write_tensor_data(tensor.data)
    writer.close()
    for name, tensor in carried:
        print(f"{name}: {tensor.tensor_type.name} {list(tensor.shape)}, {tensor.n_bytes / 2**20:.0f} MiB from the target")


if __name__ == "__main__":
    main()
