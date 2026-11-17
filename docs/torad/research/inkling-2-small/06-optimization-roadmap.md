# 06. Optimization roadmap

Order matters more than completeness here. The baseline is upstream's correctness-first
graph. Everything below is sequenced so each step is measured against a running server,
and so the one structural problem, relative attention, is fixed before micro-optimizing
around it. Expected-value claims cite vLLM and SGLang measurements on datacenter Blackwell
systems; they indicate where time went on those systems, not what the same change buys on
four RTX PRO 6000 cards.

## Step 0. Baseline census

Before changing kernels, record the shape of the problem the way the GLM work did: kernel
census and step budget at decode rows of 1, 3, 8, 23 and 69, prefill chunks of 128, 1,024
and 2,048, graph node counts, host time per step, and reduction time by route. Pay
specific attention to three costs that do not exist in GLM serving: host time filling the
distance matrices, memory traffic of the gathered per-head bias, and the copy count inside
the convolution helper. Every later step is justified or dropped by these numbers.

## Step 1. Relative attention without a materialized bias

This is the main kernel project. Today the graph builds, per attention layer, a per-head
F16 mask of size heads x KV cells x query rows. At a 2.36M-token pool and 69 decode rows
that tensor is not a rounding error; it is gigabytes of intermediate traffic per layer
shape, and the host also writes an I32 distance matrix of KV cells x rows for two caches
on every step. Both costs scale with pool size even though the information content is a
16-wide table and two position vectors.

The end state is what both production frameworks built: the attention kernel takes the
relative features R and the distance table as inputs and adds the bias where it computes
logits, deriving distance from query position and key cell position. SGLang's dedicated
decode kernel reports attention dropping from 25 to 9 microseconds per layer at batch 1.
vLLM's relative-bias decode and verify kernel reports a batch-1 decode step at 100K
context dropping from 15.1 to 4.8 ms, and its prefill path pre-shears the bias inside a
patched FlashAttention 4.

In ggml terms this means one of two moves, in order of ambition:

1. Intermediate: keep the gathered bias, but stop building distances on the host. Maintain
   cell positions as device data in the KV cache and compute the distance and gather on
   GPU. This removes host cost and the graph-reuse shape dependency, and keeps the
   per-head mask kernels from the port.
2. End state: extend the flash-attention path to accept R, the table, query positions and
   per-cell positions or row indices, and fold the sink in the kernel as upstream's graph
   algebra already prescribes. The sliding layers can go further: distance there is
   bounded by the window, so the kernel's existing window logic bounds the table access.

Correctness anchor: upstream's backend-ops per-head mask cases, plus logits agreement
between the mask path and the kernel path on the synthetic arch fixture at several
context depths, including distances beyond the 1,024 extent where the sentinel row must
be used.

## Step 2. Prologue and epilogue fusions

Per layer the baseline spends many small launches around the large matmuls: the residual
mix, three input norms in global layers, three clip operations, the value norm, the gate
sigmoid and multiply, and the router chain. vLLM fused all-reduce, short convolution,
residual handling, the attention-residual mix and RMSNorm into one kernel on NVLink
multicast hardware. The ggml equivalent is incremental and fits the fork's existing fusion
machinery:

- Fuse the mix into one kernel: normalized scoring over at most nine sources and the
  weighted sum, reading the summaries once.
- Fuse clip into the projection epilogues where the fork's matmul fusions allow, or one
  clip kernel for Q, K, V norm and R together per layer.
- Fuse gate sigmoid and multiply into the attention output side.
- Keep the router in F32 and confirm the fork's topk-moe fusion covers sigmoid scoring
  with selection bias and the sum-plus-epsilon normalization. SGLang fuses exactly this
  chain; if our fused kernel declines these semantics, extend it rather than falling back
  silently.

Each fusion is small. Together they decide whether small-row decode is launch-bound, which
is the regime agent sessions live in between tool calls.

## Step 3. Short convolution as one stateful kernel

Upstream's `convolve` is graph plumbing: views, transposes, a concat with history,
`ggml_ssm_conv`, a state store, and for dynamic layers a four-tap loop with contiguous
copies. It runs 192 times per forward. A fused kernel should read the three-position ring
and the current rows, apply static plus dynamic taps, add the residual term with its
scale, and write the new ring, in one launch per convolution site.

Constraints that shape the kernel: sequence boundaries inside a ubatch, the state store's
participation in rollback and prompt-cache restore through the existing recurrent memory
machinery, and chunked prefill, where the state crosses chunk boundaries 128 tokens at a
time in the proposed launcher. The fork's `build_conv_state_store` semantics are the
contract; the kernel replaces the arithmetic around it, not the contract.

## Step 4. MoE path at 384 experts, top-8

The expert shapes differ from GLM's in ways the kernel gates notice: 384 experts, 8 used,
intermediate 2,304, hidden 4,608, and a softcapped activation. Work items:

- Verify which existing paths actually take these shapes at decode rows and at prefill
  rows: the MMVQ routed ring (`ggml_cuda_mmvq_moe_usable` bounds tokens and token-expert
  pairs), the shared MMQ expert pair (`ggml_cuda_can_share_mmq_id` shape rules), and the
  fused gate/up MMQ (which declines data-split experts and shared pairs by design).
  Backend-ops cases at the real shapes come first; kernel changes only where a gate
  declines for a removable reason.
- Teach the chosen fused path the SITU softcaps, or prove the unfused activation cost is
  noise at served row counts.
- Expect wide expert spread even at modest row counts. At 23 rows and top-8, a layer
  touches on the order of 150 distinct experts if routing were uniform, so per-step expert
  weight traffic is closer to a large-batch regime than GLM's at the same session count.
  SGLang's MegaMoE switches on only at 64 or more tokens per forward and reported 20-27%
  time-to-first-token gains and 17% decode gains at batch 64. Our equivalent decision is
  where MMQ grouping beats the decode ring, measured per row count.
- Keep quantization of activations out of scope. The native checkpoint's NVFP4 activation
  scheme only matters on a native-precision path this study does not recommend.

## Step 5. Graph capture, shapes and host work

- Node budget: upstream already raises the arch to `32 * n_tensors`. Confirm the fork's
  CUDA graph node ceiling and kept-graph counts against a census of the real graph rather
  than importing GLM's environment values.
- Shape stability: after Step 1 removes the KV-sized distance input, graph shapes depend
  on row count and sequence count the way GLM's do, and the fork's any-seqs reuse,
  `--session-step` and `--prompt-shapes` machinery applies. Read them in the GLM order:
  defaults off, box leg on.
- Prompt path: `--prompt-step-ms` is model-agnostic and should work unchanged. Its cost
  model was tuned on GLM step shapes; the read-beside-a-stream leg re-reads it here.

## Step 6. KV cache precision

F16 first. Then Q8_0, which the fork's flash-attention already reads, with a quality leg
at depth: greedy agreement and a long-context retrieval probe at the slot limit, sliding
and global layers quantized together as the launcher would. Q4_0 only if Q8_0 reads clean
and capacity pressure demands it. Note the asymmetry: the sliding caches are small and
could stay F16 while the global cache quantizes, if the quality leg points that way and
the cache-type plumbing allows separate choices.

## Step 7. Vision, only if the workload asks

The encoder is a 292M ViT running the generic mtmd builder. For agent serving, image
encode is a per-request cost measured in tens of milliseconds against multi-second
generations. Optimize later if at all: batching images across simultaneous requests is
the only item with plausible value, and SGLang's encoder-disaggregation note warns that
vision outputs shift slightly with which images are encoded together, so batching is a
quality decision, not just a speed one. Qualification must pin single-image behavior
first either way.

## Step 8. DSpark, last

DSpark is where the published single-user rates come from: vLLM's roughly 630 tokens per
second per user includes an acceptance length of 3.5. It is also the largest single chunk
of new machinery. It should be started only with baseline Tensor 4 rates in hand, because
its economics invert at high session counts: verification multiplies target rows by eight,
so 23 decoding sessions become 184-row steps with 96 reductions each. The fork's adaptive
draft-depth idea from GLM MTP transfers as a concept; none of its code does. See the
phase plan in [07-implementation-plan.md](07-implementation-plan.md).
