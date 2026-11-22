# Addendum: six techniques checked against Inkling 2 Small

Addendum to `15-kernels-shaped-to-the-architecture.md`. Section numbers like 3.1 and M4 refer to that document. Same setting: four RTX PRO 6000, 23 sessions, Q3_K_M, F16 KV, no draft, 425.61 tok/s on the combined arm, a step of about 54 ms.

The list checked: AWQ's pipeline order, GPTQ's Hessian mechanics, SmoothQuant's migration, KIVI's key and value asymmetry, the speculative bound, llama.cpp's MMQ shape, L2 set-aside fractions, draft-length tables, and per-architecture tactic tables with labs' serving constants. They fall into six items, ordered by what they are worth here.

Marking as in the main document. Verified: read in code or in a measured report. External: another team's published result. Inferred: my reading.

| # | Item | Where it lands | Worth here |
|---|---|---|---|
| 1 | KIVI's asymmetry with SmoothQuant's migration | Far KV cells of the global layers | Largest. Read-once floor of global attention 7.7 ms toward about 2 ms |
| 2 | The speculative bound | Draft depth, and whether the draft path is worth fixing | A diagnostic. Says the measured loss is in the draft path |
| 3 | MMQ's shape on int8 tensor cores | Prefill | Aimed at most of the workload's wall time |
| 4 | GPTQ and AWQ | Expert pack | Uncertain. Pays only if the pack drops a size tier at equal KL |
| 5 | L2 set-aside | Hot expert shards | Small. About 0.5% of expert bytes |
| 6 | Tactic tables and serving constants | Nowhere | Nothing to add. Sweeps on the card cover it |

---

## 1. KIVI's asymmetry with SmoothQuant's migration, on the far cells

Extends 3.4 of the main document.

**The finding being borrowed (external).** KIVI: quantize keys per channel, quantize values per token, and keep the most recent tokens in full precision.

**How each part maps onto this model.**

- **Values per token.** Inkling 2 RMS-normalizes V per token and head before the store (verified). The per-token scale KIVI needs is fixed at 1 by the architecture. That is why a format with no scale works for V at all.
- **Keys per channel.** K is L2-clipped per token vector, which bounds the vector and says nothing about single channels. If some K channels are consistently large, a format with one fixed scale spends its levels on them.
- **Recent tokens in full precision.** That is this model's near region: the 1,024 cells within the bias extent. Near cells stay F16, far cells take the low-bit format. The split of 3.1 is already that split, and the near cells are 0.4% of what a query reads here.

**Migration.** For a static scale `s` per K channel:

`q.k = sum over channels of (q_c s_c)(k_c / s_c)`

- Store `k / s`. Multiply Q by `s` once a step in the attention prologue.
- Cells carry no scale. The per-cell decode does not change. Nothing is read per block.
- Q is not quantized, so the whole range moves to Q. SmoothQuant's exponent trades range between two quantized sides; here there is one, so there is nothing to tune.
- V the same way: divide by a per-channel scale on store, multiply the accumulated output once per channel after the merge of the partials.

**Measure first.**

- A1. For stored K and V in the global layers: largest value and RMS per channel, a KV head, a layer, over the parity capture's prompts. Is there a set of channels that is large on every prompt?
- A2. Whether those per-channel figures hold across prompts and across depth in the context. A static scale needs them to.
- A3. KL against F16 on the parity capture for four arms: far cells at 8 bits and at 4 bits, each with and without the channel scale.

**Build.**

- The scales as a small table a layer (128 a KV head), produced once by a calibration tool from A1 and stored with the model.
- Applied in the store path (divide) and in the attention prologue (multiply Q) and epilogue (multiply V's output).
- The far path's element decode is already a parameter of the tile kernel (3.4). Add the 4-bit decode beside the one-byte ones.
- If A1 shows K is even across channels, skip the scale and keep 3.4 as it is.
- If A2 shows the scales drift, fall back to a scale per channel per page of cells, stored with the page, and scale Q per page. That costs 128 multiplies per query, head and page.

**Exactness.** The migration is exact in real arithmetic and moves rounding only. The low-bit format is an approximation, gated by the KL capture with the budget taken from the drift two stock runs show against each other.

**Evidence.** ds4-dfm-rs runs this model's global-layer KV as NVFP4 (4 bits with a scale per 16 values) with top-1 equal on its short checks. KIVI reports 2 bits on Llama and Mistral with the residual. Neither is 23 sessions over a 262K context.

**Gain bound.** A cell is 1,024 bytes a global layer a card in F16. At 4 bits it is 256, or 288 with a block scale. The read-once floor goes from 7.7 ms to about 1.9 to 2.2 ms. The near cells kept in F16 add 193 MB a step, about 0.1 ms.

---

## 2. The speculative bound

**The bound.** With `a_k` the chance that the first k drafts are all accepted:

- tokens a step = `1 + a_1 + a_2 + ... + a_depth`
- gain = tokens a step, divided by (cost of a verify step + cost of the draft) / (cost of a plain step)

**Acceptance (external).** One published curve by position for this model's draft (meldavy, two reasoning benchmarks): 64, 50, 42, 37, 33, 29, 27%. That gives 1.64 tokens a step at depth 1, 2.14 at depth 2, 2.56 at depth 3.

**Cost at 23 sessions, by the byte model (inferred).**

- Attention bytes do not grow with rows when a session's rows share one read of its cells. The fork's query grouping already reads 2 to 4 rows of a sequence together.
- Expert bytes grow with distinct experts, not with rows. Under uniform routing: 147 distinct at 23 rows, 238 at 46, 294 at 69, 329 at 92. That is 1.62x, 2.0x and 2.23x the expert bytes for 2x, 3x and 4x the rows.
- So a depth-1 verify step is about 54 + 0.62 x 12.8 = 62 ms, before the draft's own cost and the wider reductions and sampling.

**What it says.** 62 ms for 1.64 tokens leaves room at depth 1. The fork measured -23% at depth 1 against no draft. The gap is in the draft path, not in the method. The fork's review already names one cost: the draft head reads the 1.85 GB LM head three times a draft step.

**Measure first.**

- A4. Acceptance by position on the fork's own coding replay at 23 sessions.
- A5. Cost of a verify step at 23, 46, 69 and 92 rows with the draft's cost separated, and distinct experts a layer at each (extends M1).

**Use.**

- Draft depth is the depth that maximizes measured tokens over measured cost at the current load. No table of draft lengths from other stacks is needed, and none would transfer: public sweeps land at 3 for one stream on small cards (meldavy, ds4-dfm-rs) and at 1 for 23 sessions here.
- If A4 and A5 put the bound above 1 at depth 1, the draft path is worth fixing before the draft is judged. If they put it below 1, drop the draft at this load.

---

## 3. MMQ's shape on int8 tensor cores, for prefill

Outside the main document's scope, which is decode.

- MMQ's shape: weight tiles unpacked to int8 in shared memory, activations quantized to q8_1, integer products.
- The fork already runs PQ2_0 weights against q8_1 activations on the int8 tensor cores (`mma.m16n8k32`, verified in `TORAD.md`).
- On this card the K-quants have no Blackwell MMQ entry and fall back to the Ampere table (verified, the fork's review).
- Taking the Q3_K and Q4_K expert tensors through the same int8 tensor-core shape is the prefill lever. Prefill is compute-bound where decode is byte-bound, and the review puts it at 14 of the workload's 15 minutes.

**Measure first.** A6. A trace of a prefill microbatch: the share of the K-quant expert products, and how far they sit from the tensor cores' rate.

---

## 4. GPTQ and AWQ on the experts

Expert cost follows bytes, so a better quantizer helps only if it lets the pack drop a size tier at the same KL. Each 10% fewer bits is about 1.3 ms of the 12.8 ms the experts take.

**What is already in the pack's method (external).** llama.cpp's importance matrix weights quantization error by the mean squared activation of each input channel, which is the diagonal of the Hessian. GPTQ adds compensation across columns through the inverse Hessian. AWQ adds a per-channel scale and a clip, both searched.

**How much the extra buys (external).** One published W4A16 comparison on Llama 3.1 8B, perplexity: RTN 6.96, GPTQ 6.92, AWQ 6.89, imatrix RTN 6.85, GPTQ with imatrix 6.83. Over an imatrix pack the gain at four bits is small. At three bits and below, where the fork serves, it is not known.

**What this model adds.**

- A token reaches 8 of 384 experts, so an expert sees about 2% of the calibration tokens against a Hessian 4,608 wide. Cold experts will not have enough samples and fall back to the diagonal.
- AWQ's scale folds exactly where a linear norm precedes the product: one scale per layer into `ffn_norm`'s gain, with the router's columns and every expert's gate and up columns divided by it. It cannot pass through the softcapped SwiGLU, so the down projection gets none.
- Order: start from the native NVFP4 values, not from a requantized pack. Calibrate each layer on the output of the layers already requantized. Scale, then clip, then quantize.

**Measure first.** A7. How the served Q3_K_M pack was made (with an importance matrix or not), and its KL against the native checkpoint.

**The bar.** Unsloth's published curve for this model: 285 GB at KLD 0.0176 against 226 GB at 0.0364. A new quantizer has to reach the first figure at something near the second size.

**When.** After M1 says the experts are byte-bound, and after attention is down, when the experts are the largest piece of the step again.

---

## 5. L2 set-aside

**What it is (external).** CUDA lets a process set aside up to the smaller of 75% of the L2 cache and a device maximum for persisting accesses, with an access window per stream or graph node. Persisting accesses get priority in that portion.

**What it would hold here.**

- One expert's share on a card is about 3.8 MB (inferred from the pack's types: 9.1 MB of gate and up plus 6.0 MB of down, a quarter a card).
- A set-aside near 96 MB holds about 25 such shards, of 18,432 a card. Even if every one is hit every step that is about 0.5% of the 18.6 GB of expert weights a card reads a step.
- A layer's shared KV prefix is 201 MB and does not fit.
- The small tensors read every step (bias tables, taps, queries, norms) are bound by launches, not by bytes.

**Verdict.** Small. The L2 lever that measured positive in qgre is warming early, which is already a campaign row. Read `persistingL2CacheMaxSize` on the card (A8) before spending anything here.

---

## 6. Per-architecture tactic tables and labs' serving constants

I have nothing reliable to add from memory, and this is an area where my training is thin. Nothing in this plan depends on such a constant: the fork chooses geometry by a sweep on the card and derives thresholds from measured distributions, and that is the right source for both.

---

## Order

1. A1 to A3 beside M4, then item 1 on top of 3.1 and 3.4.
2. A4 and A5 in the same rental window as the combined arm's trace. They cost two arms and decide item 2.
3. A6 with the first prefill trace.
4. A7 is a read. Item 4 waits for M1 and for attention to come down.
5. A8 is one call.

## Sources

- KIVI: https://arxiv.org/abs/2402.02750
- CUDA L2 cache control: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html
- llm-compressor, iMatrix importance-weighted quantization (comparison table): https://docs.vllm.ai/projects/llm-compressor/en/stable/examples/imatrix/
- Unsloth, Inkling-2-Small-GGUF (size and KLD table): https://huggingface.co/unsloth/Inkling-2-Small-GGUF
- meldavy, Inkling 2 Small on 3x RTX PRO 6000 (acceptance by position, draft sweeps): https://github.com/meldavy/inkling2-small-sm120-tp3
- ds4-dfm-rs, Inkling-2-Small VQ (NVFP4 KV, draft sweeps): https://github.com/Baekpica/ds4-dfm-rs/blob/main/docs/inkling2-vq.md
- Fork: `reports/inkling2-tensor4-inefficiency-review.md`, `TORAD.md` (PQ2_0 on int8 tensor cores; row 342), `src/models/inkling2.cpp`

---

## Update, 2026-11-22: item 3 checked against the prefill step (reports/inkling2-prefill-roofline.md)

Item 3 does not hold as written. A served prefill step is a 1,024-row pass (8 sessions x 128 rows, capped by
`--prompt-step 1024` while the donor decodes), 0.51 s to 0.63 s a step. At 1,024 rows an expert gets about 21
columns: 99 int8 ops a weight byte against the card's ridge near 631, so the expert products are bound by their
weight stream, not by the tensor cores, and no int8 MMQ shape beats the weight floor. Experts are about 17% of the
step; a perfect kernel gives about 1.08x. Tensor-core work at peak is 6% of the step. The estimated rest of the step
(no prefill trace exists yet): the 96 reductions and gathers about 80 ms, the pool-scaled mask and position build and
upload about 68 ms (the step grows about 70 ns per pool cell, which 3.3's run addressing also removes), global
attention about 62 ms, glue about 45 ms, about 27% unexplained. Cheapest lever inferred: `--prompt-step 2048`, about
10% at the cost of the decoding peers' gaps. The report holds the Nsight capture plan that settles these shares.

## Update, 2026-11-22: item 2 checked against the fork's data (reports/inkling2-spec-bound.md)

The draft path, not the method, is most of the measured loss: of the -23% at depth 1 (23 sessions), about 73% is the
draft path and 27% the wider verify rows. The bigger finding is that a drafted step does not start from the plain
step's fast paths. Four gates send it elsewhere: query-first requires exactly 23 rows; sequences with 2-4 rows leave
the tile kernel; the bounded mask class requires every session to hold at most 1/8 of the used pool, and a 262K
session holds 15.4%, so drafted steps go dense; compact greedy skips speculating slots and needs temperature 0. So a
drafted step starts near the stock 79 ms, and the fair plain baseline at temperature 1.0 is about 62.6 ms, not 54.0.
At an estimated 28 ms all-in draft overhead, depth 1 loses (0.88-0.94x against the temperature-1.0 plain step) and
depth 3 wins (1.13-1.39x). Order for the drafter: give drafted steps the fast attention path (the runs builder's
run-addressed kernel takes rows by sequence and is the natural home), then measure A4/A5, then pick depth by the
marginal rule C(d) - C(d-1) < a_d x plain. The launcher now applies the draft layout with `--draft`, and
`--draft-layout --no-draft` is the matched control (the kit's drivers updated).
