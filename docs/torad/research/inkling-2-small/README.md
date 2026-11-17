# Inkling 2 Small support study

Date: 2026-11-16. Scope: what it would take for this fork to serve Thinking Machines Lab's
Inkling 2 Small with optimization and serving functionality comparable to GLM-5.3 Flash on the
four-card tensor layout (Tensor 4) across four RTX PRO 6000 cards.

## The answer in one page

Support is feasible and most of the model work already exists upstream. Upstream llama.cpp
merged Inkling 2 text and still-image vision support on 2026-11-14 CT in commit `5c008e000`
(PR #32217), with a chat media follow-up in `dd099a6bf`. This fork at HEAD has no Inkling
support, so the first job is a hand port, not a design from scratch. The port is not a clean
cherry-pick: 33 of the 45 files that commit touches also differ between its parent and this
fork's HEAD, including the CUDA flash-attention files, the graph builder, the KV cache and
mtmd. Budget the port as careful re-application of a known-good implementation onto diverged
machinery, with upstream's own tests as the oracle.

The hard limit is memory, and it decides the shape of the project before any kernel work:

- The model is 594B parameters total and 17B active. The native checkpoint stores experts in
  NVFP4 for layers 0-39 and MXFP8 for layers 40-47 and is 393.5 GB on disk. Thinking Machines
  states it needs at least 500 GB of aggregate VRAM.
- Four RTX PRO 6000 cards hold 382.4 GiB in total, which is 410.6 GB. The native checkpoint
  alone would leave about 4 GiB per card for the KV cache, compute buffers, CUDA graphs, the
  vision tower and the driver. It does not fit as a serving target.
- Tensor 4 parity therefore means a requantized pack, the same way GLM Tensor 4 serves an
  IQ3_XXS pack rather than the reference checkpoint. Unsloth's dynamic GGUFs bracket the
  useful range: UD-IQ3_XXS at 226.0 GB leaves about 43 GiB per card after an even split, and
  UD-Q3_K_XL at 285.1 GB leaves about 29 GiB per card. Start from UD-Q3_K_XL for quality and
  keep UD-IQ3_XXS as the capacity fallback. Details and the arithmetic are in
  [05-capacity-and-tensor4.md](05-capacity-and-tensor4.md).

The second hard problem is attention. Inkling 2 has no RoPE. Position enters through a
learned relative bias per head, and upstream llama.cpp materializes that bias as a per-head
attention mask over the whole KV dimension for every layer. That is a correct bridge and the
wrong shape for a shared multi-million-token pool with 23 sessions decoding at once. vLLM and
SGLang both wrote dedicated relative-bias attention kernels for this model. We should plan
the same, after a correct baseline: first port upstream's per-head mask path so the model
runs, then move the bias into the flash-attention kernel where the distance can be computed
from positions instead of materialized. See [06-optimization-roadmap.md](06-optimization-roadmap.md).

Everything else maps onto machinery this fork already runs for GLM: hybrid attention plus
recurrent state in one memory object, tensor split through the meta backend with all-reduce,
kept CUDA graphs by shape, prompt chunking beside replies with `--prompt-step-ms`, prompt
cache save and restore, and a launcher with pinned dry-run plans. Two GLM features do not
transfer directly. The fork's coupled MTP is hard-gated to the GLM arch, and Inkling's
bundled speculator is DSpark, a separate five-layer draft model that upstream llama.cpp does
not convert or run yet. DSpark is its own phase at the end.

## Recommended route

1. Port upstream `5c008e000` and `dd099a6bf` by hand onto this fork. Keep upstream's tests.
2. Audit tensor split state for every Inkling tensor before any four-card load. The KV
   caches, the convolution state and the relative-attention tensors each need an explicit,
   tested split decision.
3. Build the launcher and qualification harness by analogy with `scripts/glm53/`, targeting
   Unsloth's UD-Q3_K_XL pack and the F16 mmproj first.
4. Run the first four-card box read in the GLM order: reduction route, load memory, rate
   sweep, short turns, a read beside a stream, depth, then quality spot checks.
5. Optimize in measured order: relative-attention kernel, prologue fusions, short-conv
   fusion, MoE path, graph shapes. Then DSpark if the acceptance economics still look good
   on this hardware.

## Files in this study

- [01-model-facts.md](01-model-facts.md) - release, license, checkpoints, ecosystem status.
- [02-architecture.md](02-architecture.md) - the text model, vision tower and DSpark head.
- [03-upstream-llama-port.md](03-upstream-llama-port.md) - what upstream shipped, file by file.
- [04-fork-gap-analysis.md](04-fork-gap-analysis.md) - what this fork has and what it lacks.
- [05-capacity-and-tensor4.md](05-capacity-and-tensor4.md) - memory arithmetic and serving design.
- [06-optimization-roadmap.md](06-optimization-roadmap.md) - kernel and graph work, in order.
- [07-implementation-plan.md](07-implementation-plan.md) - phases, tests and the box read.
- [08-risks-and-questions.md](08-risks-and-questions.md) - risks, unknowns and decisions.
- [data/model-facts.json](data/model-facts.json) - the numbers above in structured form.
- [data/sources.md](data/sources.md) - every source used, with what it was trusted for.

Nothing in this study was measured on GPU. Every performance statement is either a cited
measurement from vLLM or SGLang on datacenter Blackwell hardware, or an estimate derived
from this fork's GLM tensor-layout records. The two are labeled as such wherever they appear.
