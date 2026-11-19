# The drafter for Inkling 2 Small: the scenario and its conditions

Written on 2026-11-19 for Inkling 2 Small itself. The founder asked that the model that will be served design the
pipeline that trains its own drafter, starting from the facts and deciding the method. This page holds the facts and
the conditions. It proposes no method. Where a README in this repository reads a number and draws a conclusion, that
conclusion is its author's and you may reject it. Every path is relative to the repository's root unless it says
otherwise, and `DATA-INDEX.md` beside this page says where each record is.

## What is asked of you

Design the pipeline that produces the drafter this service decodes fastest with. Your plan is committed as you
write it, word for word, and a fresh session builds from it. That session changes the plan only where a measurement
or the engine contradicts it, and says where it did.

The plan has to answer, with the reason for each answer:

- what data to generate, how much of it, and from what;
- on which architecture and machine each part runs;
- the draft's design and size;
- how it is trained;
- the rule by which the target accepts what the draft offers;
- how the result is measured, and what counts as better;
- what you would try first, and what result would make you change course.

Where a number you need is missing below, say which one and how to measure it. A guess is fine when it is marked as
one.

## The service

- The target is Inkling 2 Small, abliterated: `benthecarman/Inkling-2-Small-abliterated`, revision `e4853745`.
  48 layers, hidden width 4608, 384 experts with 8 routed per token, a vocabulary of 200,058. Facts and their
  sources: `docs/torad/research/inkling-2-small/01-model-facts.md`, `02-architecture.md`, `09-abliterated-target.md`.
- The weights are the 3-bit pack, Q3_K_M quantized with an importance matrix: a 283.5 GB GGUF. This target stays.
  The founder decided it, and a plan that needs another target is out of scope.
- It runs on four RTX PRO 6000 cards (Blackwell, 96 GB each) on a rented machine, split by tensor across the four
  (`-sm tensor -ts 1,1,1,1`), so the four cards work on the same token together. How that split works and what
  it costs: `12-four-card-serving.md`, `13-batched-serving-results.md` in the same folder as the model facts.
- It serves up to 23 Claude Code sessions at once. Sessions are long, and most of what the model writes in them is
  tool calls. The sessions share a prefix of about 17.8K tokens (the Claude Code system prompt and its tool
  schemas); contexts in today's runs ran from 18K to 200K tokens. A slot can hold 294,912 tokens.
- Sampling is temperature 1.0 with top-p 1.0, top-k 0 and min-p 0: the target draws from its whole distribution.
- The goal is the most decoded tokens a second for each session at that load, measured on the service itself, with
  what the target writes unchanged in distribution.

The exact launch, argument by argument and variable by variable, is recorded in
`data/drafter-20261119/g1-gate/launch.json` and `meta-heldout.json`, and `scripts/inkling2/serve.py` with its
`README.md` is the launcher (profile `tensor-4-draft`).

## The numeric path, as the engine has it

The request that reached us described the hidden state as dequantized to FP8 at the tensor boundaries and quantized
back. The engine at the served commit does not do that, so here is what it does do:

- The weights are 3-bit K-quant blocks. The teacher check of the recording found the output layer stored as Q6_K.
  The pack's other tensor types were not listed for this page.
- Global-attention layers read their key and value cache as Q8_0: 8-bit integers in blocks of 32 with one scale a
  block. Sliding-window layers read theirs as F16.
- A card's share of a reduction crosses to the other cards as BF16 (`--reduction-wire bf16`; `q8` exists).
- FP8 (E4M3) exists in this engine as an optional cache type and as a weight format. It is not on the served path.
  On another model of ours, attention with FP8 keys and values missed the quality bar
  (`docs/torad/bonsai-27b/README.md`).
- The founder has raised running the hidden-state math in int8 or FP8 on these cards. For Inkling nothing of that
  is decided or built. If your plan wants it, it is a proposal he decides.

The code: `src/models/inkling2.cpp`, `src/llama-kv-cache.cpp`, `ggml/src/ggml-backend-meta.cpp` (the split),
`ggml/src/ggml-cuda/allreduce.cu` (card to card), `ggml/src/ggml-cuda/fattn-rel-decode.cuh` and its siblings
(attention), `TORAD.md` (every switch the fork adds).

## The draft we have

- The model's own repository ships a draft head (`mtp.safetensors`). Converted, it is a 4.59 GB GGUF of the
  DSpark kind: 5 layers of width 4608, 1,711.7 million parameters of its own, stored in BF16, a window of 1024, a
  block of 8. It reads the target's hidden state at eight depths (after layers 5, 11, 17, 23, 29, 35, 41 and 48)
  and shares the target's embedding and output layer. How it is converted: `scripts/inkling2/README.md`, "Native
  draft". How the engine runs it: `src/models/dspark.cpp`, `common/speculative.cpp`.
- Served, it offers up to 3 tokens a round, runs on the fourth card, and offers its own first choices.
- The check: the target draws each token from its distribution, and a drafted token is kept only when the draw
  equals it. A round writes the kept drafted tokens and one token of the target's own. The engine refuses any
  other sampling for this draft today (`common/speculative.cpp`, near line 1485); a coupled check exists for
  another model's head (`LLAMA_SPEC_COUPLED` in `TORAD.md`).
- We own the fork. The draft's architecture, the check, where the draft runs and what a round does can all change
  if your plan needs them to.

## What was measured on 2026-11-19

Everything below ran at engine commit `7b262d27f`, on the rented machine, against the served pack. The fork has
moved since (`68c844088` at the time of writing); what the later commits change in a drafted round's cost was not
measured.

**Speed, 23 sessions of 18K to 200K tokens, 256 tokens a reply** (`data/drafter-g7/README.md`):

| tokens a second, each session | no draft | stock draft | tuned draft |
|---|---|---|---|
| 23 at once | 12.4 | 12.1 | 12.0 |
| 8 at once | 12.3 | 15.4 | 15.1 |
| 4 at once | 12.2 | 23.0 | 22.1 |
| 1 at once | 24.3 | 41.5 | 42.6 |

The stock draft kept 60.6% of what it offered over those runs, 1.82 tokens a round; the tuned one 61.1%, 1.83.

**What a round costs.** From those runs a plain round takes about 41 ms for one session and about 81 ms for 4 to
23; a drafted round about 67 ms for one session, 125 ms for 4, 182 ms for 8 and 240 ms for 23. A traced run of the
stock draft (`data/drafter-20261119/g4-trace/bench-stock-trace-split.txt`) splits a drafted round at 23 sessions
into 46.8 ms before the target runs (the draft's call is 45.5 ms of it), 173.9 ms of target evaluation over 92
rows, 10.4 ms of feeding and 73.5 ms of sampling 63 rows, 304.6 ms in all with tracing on. At one session the same
four parts are 34.4, 58.9, 0.5 and 4.2 ms. The target's evaluation by rows in the batch: 58.9 ms for 4 rows, 64.6
for 8, 52.7 for 16, 88.3 for 32, 138.0 for 64, 169.2 for 80. A commit after this trace samples the rows of
drafted slots side by side; its effect was not measured here.

**The first training round** (`data/drafter-g7/`). The stock draft was tuned for 398 steps, two passes over 42,580
tokens Inkling wrote as continuations of 73 recorded requests, at four learning rates. At 1e-5 its cross-entropy
against the target on held-out requests fell from 1.174 to 0.948, and the tokens it would keep a round at depth 3
went from 1.907 to 1.920, where a draft that always named the target's first choice would keep 2.178. At 3e-5,
1e-4 and 3e-4 it kept 1.863, 1.704 and 1.310. Served, the tuned draft was within noise of the stock one.

**Acceptance by context length**, walked offline over 46,147 held-out rounds
(`data/drafter-g7/sweep-lr1e-5-walk-after.json`, `data/drafter-20261119/g5-round/walk-stock.json`). Tokens kept a
round at depth 3, then what a draft naming the target's first choice every time would keep:

| context | stock draft | tuned draft | first choice every time |
|---|---|---|---|
| under 32K | 2.091 | 2.134 | 2.352 |
| 32K to 64K | 1.754 | 1.749 | 2.044 |
| 64K to 128K | 1.728 | 1.729 | 2.004 |
| 128K and over | 2.143 | 2.148 | 2.367 |

At depth 5 the tuned draft keeps 2.574 of a possible 3.100, and at depth 7, 2.946 of 3.761.

**A different check, walked offline only.** If a drafted token were kept with the chance the target gives it,
drawn by a draft that samples its own rows, the tuned draft would keep 2.184 a round at depth 3, 2.981 at depth 5
and 3.457 at depth 7. The engine does not do this for this draft, and the stock draft was not walked under it.

**Stopping early.** With the draft offering a token only while its own confidence stays above a floor, at depth 3
the tuned draft keeps 1.865 of 2.592 offered at a floor of 0.3, 1.678 of 2.023 at 0.5, 1.425 of 1.547 at 0.7 and
1.131 of 1.155 at 0.9.

**Long sessions.** The founder has seen the stock draft lose almost all its acceptance late in long sessions, in
earlier tests of the service. Today's replay did not reproduce a collapse, and it is one run: 23 sessions of 1 to 8
turns grown to as much as 199,967 tokens kept 68.4% overall (`data/drafter-20261119/g5-round/replay-stock.txt`),
by turn 70.4, 67.2, 69.4, 69.0, 79.1, 77.9, 61.6 and 52.3%, with single turns as low as 26%. Turns 7 and 8 are 13
and 10 turns of evidence. What makes acceptance fall late in a session is not established.

**The trainer against the server.** The trainer's copy of the draft names the same token as the served draft in
98.8% of 333 live positions (`data/drafter-20261119/g2-sweep/live-stock.txt`).

## What exists to work with

- A recording of Inkling's own states. For 66 held-out requests: 1,178,396 prompt tokens and 135,293 tokens the
  model wrote in 845 continuations, each row holding the hidden state at the eight depths (73,728 bytes a row, in
  BF16) and the target's 64 likeliest tokens with their probabilities. For training, 73 requests so far: 42,580
  written tokens, and 299,440 more tokens in the model turns of the prompts, which another model wrote and Inkling
  only read. The labels at prompt positions are Inkling's own likeliest tokens there, never the token the prompt
  holds. Recording the held-out set took 5,352 seconds and 98 GB.
- The requests come from six public sets of real Claude Code sessions, rebuilt behind the real Claude Code system
  prompt and tool schemas: 1,155 session groups for training and 118 held out, about 56.9 million tokens
  (`data/drafter-20261119/g1-build/SOURCES.md`, `g1-gate/plan.txt`). Only public data may be used.
- The kit that did all of the above, `scripts/inkling2/drafter/`: the recorder, a PyTorch copy of the draft that
  matches the engine, the trainer, the offline walks, the served bench and replay.
- The rented machine holds the pack, the recordings, the build and the drafts on 1.5 TB of disk. It costs $7.77 an
  hour running and $0.83 an hour stopped. The founder adds credit in small amounts and decides every spend, so
  the plan should say what each stage costs before it runs.

## Conditions the founder set

- The served target stays as it is: the 3-bit pack, four cards, up to 23 sessions.
- A mistake has to show early and cheaply. A long run that turns out wrong at its end is the outcome to design
  against, so each stage needs a check that fails before the next stage is paid for.
- A small round comes before a large one, and the small round has to be able to show whether the large one is
  worth running.
- What the trainer sees should be what the served model computes, at the served precision.

## What was not measured

- The stock draft under the probability-aware check.
- What the commits after `7b262d27f` change in a drafted round (pull requests #274 and #275, and `68c844088`).
- Hardware-counter profiles of the kernels: the rented machine's container refuses them, so none exist.
- Acceptance split by what is being written (tool call, prose, reasoning).
- Acceptance in live Claude Code sessions, as against replays of recorded ones.
