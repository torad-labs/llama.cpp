# llama-gram

The full calibration Gram of every projection's input, the sum of x x^T over each token, every off-diagonal
included, gathered on the model's own forward. `llama-imatrix` keeps only the diagonal of that sum; GPTQ
chooses codes on the whole matrix.

```
llama-gram -m model.gguf --sequences list.txt -o out/ [--ubatch 2048] [--device 0]
```

`list.txt` holds one path per line; each file is one sequence of token ids as consecutive little-endian
4-byte words, decoded as given (no tokenizing, no BOS added). Sequences are packed into ubatches, one
sequence id each, with logits on for every token so the last layer and the head see every row.

Projections that read one input share one Gram: q, k and v (`attn_in`), the attention output projection
(`attn_out_in`), gate and up (`ffn_in`), down (`ffn_down_in`), and the head (`head_in`, in folder `output`).
In each ubatch every member of a group must read the same input tensor as the first one, or the run stops.

Each ubatch's x x^T is cuBLAS SSYRK in binary32 with pedantic math (no TF32) on the card, and is added into
a binary32 running sum on the host. Output, into a directory that must not exist:

- `<folder>/<group>.hessian`: the upper triangle row by row (i <= j < n), little-endian binary32.
- `gram.json`, written last: the model path, build, sequences, token and ubatch counts, and per group its
  width, token count, file and members (GGUF name, type, shape, tokens, and the byte range of the tensor's
  data in the model file, so a caller can hold it to another copy of the same weights).

The tool stops on a weight matmul under `blk.` of a kind it does not group, an indirect (expert) matmul, a
non-f32 input, a group whose members read different inputs, a group or member that missed a token, and a
non-finite sum. It builds only with `GGML_CUDA`.
