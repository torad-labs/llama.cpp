# 08. Risks, unknowns and decisions

## Risks that can change the plan

1. **Real-checkpoint conversion is unproven upstream.** The merged converter was validated
   on synthetic packed weights against a pinned vLLM decode, and its own commit message
   states that real checkpoint conversion remains untested. Unsloth's stock GGUFs prove
   someone converted the real weights successfully, but not with this converter and not
   from the abliterated derivative. Mitigation: use a stock Unsloth pack only for engine
   bring-up; treat conversion of the pinned abliterated source as a verification event with
   sampled tensor comparisons, and do not block the baseline port on the full conversion.
2. **The packed-format metadata disagrees with itself.** `hf_quant_config.json` declares
   an MXFP8 block shape that the stored tensors do not have. Upstream trusts the stored
   shapes. A future checkpoint revision could fix the metadata, change the storage, or
   both. Pin checkpoint revisions and hashes wherever the launcher or converter is used.
3. **Per-head bias scaling.** The baseline attention materializes a heads x KV x rows
   bias and a host-built distance matrix. At small pools this is merely slow; at the
   target pool and row counts it can dominate memory traffic and host time, and its shape
   breaks graph reuse as the pool grows. This is the one risk that can make a correct
   port unservable, and it is why the kernel work is Step 1, not an afterthought.
4. **Split-state mistakes are silent.** A mirrored cache fails loudly by not loading, but
   a convolution state split on the wrong axis, or clip scales split against the wrong
   head mapping, produce a model that runs and answers slightly wrong. The Phase 2
   two-rank logits gates exist for exactly this. No four-card load happens before them.
5. **Pack quality at 3 bits on agent work.** Unsloth's KLD and top-token figures describe
   stock packs on wikitext and do not transfer to a locally quantized abliterated pack.
   Agent workloads stress long-horizon consistency, tool-call formatting and code
   correctness, none of which KLD sees. The quality leg must measure each target pack
   against a high-precision conversion of the abliterated source, then read agent behavior
   after the box read. Keep the quality profile runnable so the trade can be re-decided
   with evidence.
6. **Reduction-wire error enters persistent state.** GLM's 8-bit wire was validated where
   its error landed in one step's activations. Inkling writes reduced values into
   convolution state that every later step reads. The wire stays BF16 until its own leg
   reads clean at depth.
7. **Template and parser drift.** Tool calling and reasoning parsing in llama.cpp ride on
   the embedded template and the generic autoparser, while vLLM and SGLang wrote
   dedicated parsers matched to Thinking Machines' reference renderer. Edge cases in
   nested tool arguments, thinking retention across turns, and invalid reasoning effort
   handling are qualification items, and upstream already needed one chat fix after the
   initial merge.
8. **Upstream is moving.** The llama.cpp support merged two days before this study. Expect
   follow-up fixes, and expect the still-open DSpark work in other frameworks to settle
   semantics we would otherwise guess at. Re-check upstream at branch start and again
   before the box read.
9. **Interconnect variance across rentals.** GLM reads showed the reduction route and
   even layout viability depend on the host's topology. A four-card rental without peer
   mapping measures a different machine. Topology check first, always.
10. **The selected derivative is new and self-measured.** The abliterated repository was
   created on the day it was selected and had no downloads or independent evaluation when
   checked. Its card reports longer reasoning and more unfinished answers on long
   documents. Pin the revision, verify the files against stock, and make behavior
   preservation and long-session completion explicit pack gates.

## Open questions

- Does the fork's flash-attention already have a row-indexed KV path that the relative
  kernel can reuse for per-cell positions, or does the kernel need a new position input?
  Answered in Phase 2 by reading, before any kernel is designed.
- Can sliding and global KV caches take different cache types in one ISWA context through
  existing plumbing? This decides whether the F16 sliding plus Q8 global variant is free
  or a small project.
- What does the vision token limit of 4,096 do to a 4,096-pixel-wide screenshot in
  practice, and does the reference path resize, crop or refuse? Answered by the image leg
  against reference outputs.
- Is there a public reference for DSpark's Markov head semantics beyond the vLLM
  implementation? The vLLM PR is currently the de facto specification. Phase 6 starts by
  reading it as one.
- What acceptance length does DSpark get on agent traffic rather than framework
  benchmarks? Only measurable after Phase 6 exists; the phase is optional for that reason.

## Decisions

Made in this study, each reversible with evidence:

- Serve a locally converted and requantized GGUF pack from the pinned
  `benthecarman/Inkling-2-Small-abliterated` revision, starting in the Q3 size class.
  Stock Unsloth packs are bring-up and comparison inputs only. No native-checkpoint serving
  and no MXFP8 type work.
- Slot context 262,144 and 23 reserved sessions as the parity shape. No 1M-token product
  claim on this hardware.
- Port upstream's per-head mask path first for correctness, then replace it with a
  kernel-resident bias. Do not attempt the kernel first.
- BF16 reduction wire and F16 KV until their own quality legs read clean.
- DSpark last and optional, off by default until it pays at served concurrency.

For Marcos:

- Whether to run this at all, against the alternative of using the hosted API at published
  prices while GLM Tensor 4 remains the self-hosted workhorse.
- The four-card rental spend for Phase 4, when the port is green and ready to read.
- If both proceed, whether the first served target optimizes for session depth
  (capacity profile) or answer quality (quality profile) when the box read forces the
  trade.
