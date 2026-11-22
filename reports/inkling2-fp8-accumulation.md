# FP8 accumulation for the relative decode (row inkling-fp8-accumulation)

Kernel side (`inkling-quant`), meeting inkling-small's attention side on one contract. Reference design:
QGRE's lossless FP8 accumulator — `qgre-core/qgre-silicon/kernels/l1-hardware/reduce/silicon_l1_pair_kslice_sum.cuh`
and `silicon_l1_pair_partial_row_combine_two_term.cuh`, with the scalar pair and the block-scaled MMA headers
they include. Countersigned by the coordinator 2026-11-22 with three conditions, recorded on the row.

## What is measured, plainly (condition 2)

**The partial traffic stays 4 bytes a value.** A partial slot is one u32: `[hi code, lo code, hi sf, lo sf]`
— exactly the f32 slot it replaces (`fattn_fp8_slot_pack` in `ggml/src/ggml-cuda/fattn-rel-fp8-pair.cuh`).
Nothing about the win below is a byte saving on the partials. What the fold buys is:

- the **fp8 MMA rate**: the block-scaled QMMA (m16n8k32, `kind::mxf8f6f4`, ue8m0 scales) issues twice the
  MACs of an f16 m16n8k16 at the same rate, so a fold of terms costs half the instructions of f16
  multiply-adds per element;
- **one rounding per fold**: terms enter as exact e4m3 pairs under their own E8M0 scales, the sum lives in
  the tensor-core fp32 fragment, and the fold exits through a single RNE two-term encode
  (`fattn_fp8_encode_two`) — no software accumulation on the hot path, and the carried partial's `lo` holds
  exactly what its `hi` cannot.

**The falsifier is quality plus prefill time**: hi+lo pair quality against f16 on attention scores and the
arch fixture's greedy chains, and the prefill time of the integrated path. A fold that silently drops `lo`
reds on the quality half; a fold that keeps f16's rounding cost reds on the time half. The evidence file is
`reports/inkling2-fp8-accumulation.json`, judged by `scripts/inkling2/fp8_accumulation.py --check`.

## The contract (countersigned)

Files:

- NEW `ggml/src/ggml-cuda/fattn-rel-fp8-pair.cuh` — the pair algebra (e4m3 code + E8M0 scale; the two-term
  hi+lo encode; the 4-byte slot), the block-cooperative encode, the block-scaled MMA wrapper and its
  erratum-law LDS scale word, and the two folds: the 32-lane two-term block sum (two serial MMAs, one exit)
  and the one-hot per-term sum (up to 16 terms of one element, each under its own scale).
- MODIFY `fattn-rel-mma.cuh` (QK/PV fold), `fattn-rel-decode.cuh` / `fattn-rel-runs.cuh` (merge paths) —
  **held**: those files carry inkling-small's staged D2 hunks (condition 1), so they change only after its
  hunk commit lands or it confirms no overlap.
- `fattn-rel-elem.cuh` staging gains the shared scale tables when the fold lands there.
- Switch `GGML_CUDA_FATTN_FP8_ACCUM`, default off, declared in `tunables.toml` + `ggml-tunables.inc`; with it
  off every existing path is byte-identical.

Fold order: (1) QK chain — Q and K tiles as e4m3 pairs with per-K32-block scales, MMA per k-step under its
own scale, fp32 fragment, one exit encode into two-term partials; (2) PV chain — weights in the same pair
algebra against V, partials carried identically; (3) shared-prefix merge and one-cell merges — two-term
combines over the pair planes, one encode at their exit, never a widened value across a seam; (4) split
reductions inherit the wire.

Tests: `test-inkling-fp8-pair-cuda0` (landed with the header: the algebra and the folds on the device, held
against independent oracles — the host's double-precision e4m3 construction and CUDA's own
`__nv_cvt_float_to_fp8`; a single-lane fold must be bit-exact, and hi+lo must beat hi-alone error by 4x at
least). Then, on integration: `test-inkling-kv-fp8acc-tile-cuda0` (FLASH_ATTN_EXT, `type_KV=f8_e4m3`,
switch on, against the host) and the model-level `test-inkling-fp8acc-{served,restore}` arms with the
greedy-chain fixture.

Condition 3, recorded: the model-level greedy fixture runs with the switch ON and `type_KV=f8_e4m3`;
otherwise K is quantized on the fly and the comparison muddles.

## Geometry at our sizes

- Head 128 = four K32 blocks: the per-term fold carries four terms an element slot against QGRE's sixteen.
- The block sum folds 32 partials a trip (two K32 blocks: the block's hi codes then its lo codes, one shared
  E8M0 scale each); an element's partial bank of up to 256 folds in two rounds of eight group sums.
- The merge's partial slot count is the stream count the rel-mma kernel already writes (up to 256), so the
  fold capacity is the served ceiling.

## Status 2026-11-22

Landed: the header, the unit test, the switch declaration, and the CMake arm (all off by default; nothing
wired into the decode yet). The D2-held files are untouched. Box validation of the unit test runs on the
5090 buildbox; its verdict goes to the row. The rel-mma QK/PV folding, the merge rewiring and the ops/model
arms follow inkling-small's hunk commit.
