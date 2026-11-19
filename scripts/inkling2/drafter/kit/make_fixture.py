"""Build the rehearsal pair: the fork's small Inkling 2 fixture carrying the real Inkling tokenizer, and a DSpark
draft that fits it.

The small fixture (three blocks, 512 wide) has a 128-token test vocabulary, so nothing rendered from a real request
tokenizes on it. This writes the same model with the real tokenizer and a vocabulary of the real size, so the whole
recording path runs on it with the token counts, cuts and prefixes the real pack will see.

The end-of-message row of the output matrix is written larger than the rest, so a sampled continuation ends by
itself now and then and both ways a continuation can end are rehearsed.

Usage: make_fixture.py --gguf-py DIR --target fixture-target.gguf --tokenizer tokenizer.gguf --out DIR
"""
import argparse
import os
import sys

import numpy as np

SKIP = ("GGUF.", "general.architecture", "tokenizer.")


def copy_field(gguf, writer, key, field):
    kind = field.types[0]
    value = field.contents()
    if kind == gguf.GGUFValueType.ARRAY:
        if not value:
            return
        writer.add_key_value(key, value, kind, sub_type=field.types[-1])
    else:
        writer.add_key_value(key, value, kind)


def write_target(gguf, args, tokenizer, path):
    reader = gguf.GGUFReader(args.target)
    arch = reader.fields["general.architecture"].contents()
    if arch != "inkling2":
        raise SystemExit("the fixture is %s, not inkling2" % arch)
    tokens = tokenizer.fields["tokenizer.ggml.tokens"].contents()
    unpadded = len(tokens)
    padded = args.padded
    if padded < unpadded:
        raise SystemExit("the padded vocabulary is smaller than the tokenizer")
    # the token that ends a model turn, which is the vocabulary's end of generation
    stop = int(tokenizer.fields["tokenizer.ggml.eos_token_id"].contents())
    if tokens[stop] != "<|content_model_end_sampling|>":
        raise SystemExit("the end of generation is %s, not the end of a model turn" % tokens[stop])
    width = reader.fields["inkling2.embedding_length"].contents()
    override = {
        "inkling2.vocab_size": padded,
        "inkling2.vocab_size_unpadded": unpadded,
        "inkling2.context_length": args.context,
        "inkling2.rope.scaling.original_context_length": args.context,
    }
    writer = gguf.GGUFWriter(path, arch)
    for key, field in reader.fields.items():
        if key.startswith(SKIP):
            continue
        if key in override:
            writer.add_key_value(key, override.pop(key), field.types[0])
        else:
            copy_field(gguf, writer, key, field)
    if override:
        raise SystemExit("the fixture has no %s" % ", ".join(sorted(override)))
    for key, field in tokenizer.fields.items():
        if key.startswith("tokenizer."):
            copy_field(gguf, writer, key, field)

    by_name = {t.name: t for t in reader.tensors}
    embd_std = float(np.std(np.asarray(by_name["token_embd.weight"].data, dtype=np.float32)))
    norm_rms = float(np.sqrt(np.mean(np.square(np.asarray(by_name["output_norm.weight"].data, dtype=np.float32)))))
    # after the RMS norm a hidden row is about sqrt(width) * norm_rms long, so this spreads the logits by logit_std
    out_std = args.logit_std / (np.sqrt(width) * max(norm_rms, 1e-6))
    rng = np.random.default_rng(args.seed)
    for t in reader.tensors:
        if t.name == "token_embd.weight":
            data = rng.standard_normal((padded, width), dtype=np.float32) * np.float32(embd_std)
        elif t.name == "output.weight":
            data = rng.standard_normal((padded, width), dtype=np.float32) * np.float32(out_std)
            data[stop] *= np.float32(args.stop_gain)
        else:
            writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
            continue
        writer.add_tensor(t.name, data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("target: %s vocab %d (padded %d) width %d context %d embd_std %.4g out_std %.4g stop token %d" % (
        path, unpadded, padded, width, args.context, embd_std, out_std, stop), flush=True)


def write_draft(gguf, target_path, path):
    """A small draft laid out as the served one is, with the target's vocabulary tensors.

    The served draft has heads 128 wide and keeps its matrices in BF16. Both decide how the engine computes it: a
    head that wide has its quantized cache rotated, and a BF16 matrix is multiplied in BF16. The trainer copies
    those, so the rehearsal's draft has to take the same paths or the live check proves nothing about them.
    """
    reader = gguf.GGUFReader(target_path)
    writer = gguf.GGUFWriter(path, "dflash")
    width = reader.fields["inkling2.embedding_length"].contents()
    padded = reader.fields["inkling2.vocab_size"].contents()
    vocab = reader.fields["inkling2.vocab_size_unpadded"].contents()
    layers = reader.fields["inkling2.block_count"].contents()
    heads, kv_heads, dim, ff, rank = 4, 2, 128, 64, 8
    writer.add_context_length(reader.fields["inkling2.context_length"].contents())
    writer.add_embedding_length(width)
    writer.add_feed_forward_length(ff)
    writer.add_block_count(2)
    writer.add_head_count(heads)
    writer.add_head_count_kv(kv_heads)
    writer.add_key_length(dim)
    writer.add_value_length(dim)
    writer.add_rope_dimension_count(dim)
    writer.add_rope_freq_base(10000.0)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_vocab_size(padded)
    writer.add_uint32("dflash.vocab_size_unpadded", vocab)
    writer.add_block_size(8)
    writer.add_target_layers([0, layers])
    writer.add_uint32("dflash.dspark.markov_rank", rank)
    writer.add_sample_from_anchor(False)
    writer.add_has_confidence_head(False)
    writer.add_bool("dflash.attention.causal", False)
    writer.add_sliding_window(32)
    writer.add_sliding_window_pattern([True, False])
    writer.add_final_logit_softcapping(30)
    for key, field in reader.fields.items():
        if key.startswith("tokenizer.") and key != "tokenizer.ggml.mask_token_id":
            copy_field(gguf, writer, key, field)
    writer.add_mask_token_id(0)
    for t in reader.tensors:
        if t.name in ("token_embd.weight", "token_embd_norm.weight", "output.weight"):
            writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
    rng = np.random.default_rng(4558)

    def weight(name, shape, norm=False):
        data = np.ones(shape, dtype=np.float32) if norm else rng.normal(0.0, 0.035, shape).astype(np.float32)
        writer.add_tensor(name, data)

    def matrix(name, shape):
        data = rng.normal(0.0, 0.035, shape).astype(np.float32)
        stored = gguf.quants.quantize(data, gguf.GGMLQuantizationType.BF16)
        writer.add_tensor(name, stored, raw_dtype=gguf.GGMLQuantizationType.BF16)

    weight("dspark_fuse.weight", (3, 2))
    weight("dspark_norm.weight", width, True)
    weight("output_norm.weight", width, True)
    weight("mask_embd.weight", width)
    matrix("markov_w1.weight", (padded, rank))
    matrix("joint_proj.weight", (3*rank, 2*(rank + width)))
    weight("joint_proj.bias", 3*rank)
    matrix("dspark_hidden_out.weight", (width, rank))
    weight("dspark_hidden_out.bias", width)
    for layer in range(2):
        prefix = f"blk.{layer}."
        for name in ("attn_norm", "ffn_norm", "dflare_norm"):
            weight(prefix + name + ".weight", width, True)
        for name in ("attn_q_norm", "attn_k_norm", "attn_k_ctx_norm"):
            weight(prefix + name + ".weight", dim, True)
        for name in ("attn_k", "attn_v", "attn_k_ctx", "attn_v_ctx"):
            matrix(prefix + name + ".weight", (kv_heads*dim, width))
        matrix(prefix + "attn_q.weight", (heads*dim, width))
        matrix(prefix + "attn_output.weight", (width, heads*dim))
        matrix(prefix + "ffn_gate.weight", (ff, width))
        matrix(prefix + "ffn_up.weight", (ff, width))
        matrix(prefix + "ffn_down.weight", (width, ff))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("draft: %s taps 0,%d, %d heads %d wide, matrices in BF16" % (path, layers, heads, dim), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf-py", required=True, help="the engine source's gguf-py folder")
    parser.add_argument("--target", required=True, help="the fork's small inkling2 fixture")
    parser.add_argument("--tokenizer", required=True, help="a GGUF carrying the real tokenizer.* keys")
    parser.add_argument("--out", required=True, help="folder for target.gguf and draft.gguf")
    parser.add_argument("--context", type=int, default=524288)
    parser.add_argument("--padded", type=int, default=201024, help="padded vocabulary, as the real pack has it")
    parser.add_argument("--logit-std", type=float, default=2.0)
    parser.add_argument("--stop-gain", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--draft-only", action="store_true",
                        help="keep the target.gguf already in --out and write the draft again (recordings made from that target stay good)")
    args = parser.parse_args()
    sys.path.insert(0, args.gguf_py)
    import gguf

    os.makedirs(args.out, exist_ok=True)
    target = os.path.join(args.out, "target.gguf")
    if args.draft_only:
        if not os.path.isfile(target):
            raise SystemExit("no %s to keep" % target)
    else:
        tokenizer = gguf.GGUFReader(args.tokenizer)
        write_target(gguf, args, tokenizer, target + ".tmp")
        os.replace(target + ".tmp", target)
    draft = os.path.join(args.out, "draft.gguf")
    write_draft(gguf, target, draft + ".tmp")
    os.replace(draft + ".tmp", draft)


if __name__ == "__main__":
    main()
