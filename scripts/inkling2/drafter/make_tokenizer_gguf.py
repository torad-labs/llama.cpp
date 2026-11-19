"""Write a GGUF that holds only the tokenizer of a model GGUF (no tensors).

Usage: make_tokenizer_gguf.py <gguf-py folder> <source.gguf> <tokenizer.gguf>
"""
import sys

sys.path.insert(0, sys.argv[1])
import gguf  # noqa: E402

reader = gguf.GGUFReader(sys.argv[2])
writer = gguf.GGUFWriter(sys.argv[3], "tokenizer")
count = 0
for key, field in reader.fields.items():
    if not key.startswith("tokenizer.") or key == "tokenizer.ggml.mask_token_id":
        continue
    kind = field.types[0]
    value = field.contents()
    if kind == gguf.GGUFValueType.ARRAY and not value:
        continue
    writer.add_key_value(key, value, kind, sub_type=field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None)
    count += 1
    print(key, kind.name, len(value) if isinstance(value, (list, str)) else value)
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
print("wrote", count, "keys")
