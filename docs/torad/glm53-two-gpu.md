# GLM-5.3 Flash on two RTX PRO 6000 cards

Read this before starting any two-card GLM-5.3 work. The four-card goal and its numbers are in
[glm53-goal.md](glm53-goal.md); this page is the same engine and the same pack on half the cards.

## The goal

GLM-5.3 Flash, the BoldingBuilds IQ3_XXS pack, served by this fork on two RTX PRO 6000 cards to as many
concurrent agent sessions as the two cards hold, each able to reach 307,200 tokens at once (Marcos, November 15,
2026: the four-card optimizations carried to two cards; no need for 23 sessions; as many sessions at 300K as fit).
There is no per-session rate target yet: the first box read sets the number, as the first four-card reads did.

## Where it stands

The launcher is ready and nothing is read on cards. `scripts/glm53/serve.py --profile tensor-2` (the default now
sizes for this goal) and an experimental `--profile dp-2` both launch the pack with the settings the four-card
work measured, sized for two cards. Their plans are pinned by `scripts/glm53/test_qualification.py`, and the
tensor-4 and dp-4 plans are byte-identical to what they were before this change. The previous two-card recipe
(three slots, a 524,288-token pool, Q8 latent caches, none of the settings below) is replaced by it; its
November 2 sweep stays in [GLM53-SERVING.md](../backend/GLM53-SERVING.md) as history.

What the two tensor profiles now share, because the engine work behind them never counted ranks: the 4-bit
latent cache, prompts read in chunks apart from replies, graphs kept by shape for any set of sessions
(`LLAMA_GRAPH_ANY_SEQS` with kept graphs and held CUDA executables), coupled MTP drafts, prompt-cache saves and
restores beside the decode steps, and every default-on host and kernel change of the four-card period. Two
settings deliberately differ:

- **The reduction route.** Four cards force reductions onto the copy engines through host memory, because the
  engine's peer routes read three times slower on the rented hosts (glm53-goal.md, November 11). No pair of
  cards has been read either way, and the goal doc's own finding is that the right route is the host's, so two
  cards keep the engine's routes and `--reduction-route host` is the launcher option to read against them. The
  8-bit wire is on for both, with `--reduction-wire bf16` as its off switch; its quality read (two chunks at
  8,192 tokens) was at four ranks.
- **Kept graphs and their reserve.** Two cards run a session limit by default, so they keep 32 graphs a context
  by the same rule as a limited four-card pool, and a context hands a kept graph's buffers back with 1 GiB
  still free (`LLAMA_GRAPH_DEVICE_RESERVE_MIB=1024` against the engine's 256 MiB) because the cards run fuller.

## Capacity, as arithmetic

Every number here is sums from the four-card reads and the pack's tensor table
([glm53-decode-roofline.json](glm53-decode-roofline.json)), not a load. A card has 97,887 MiB (95.6 GiB).

Tensor split, per card. The pack is 125.1 GiB of tensors: 115.7 GiB of routed experts, which split, and 9.4 GiB
of the rest, which split by heads except small tensors. A card therefore carries between 62.5 GiB (everything
split) and 67.2 GiB (every non-expert byte mirrored) of weights; the tensor table's count with attention
mirrored was 66.15 GiB (TORAD.md, row `833d067e0`), and the current split divides attention by heads too. The
mirrored caches cost 9,088 bytes a pool token (the November 11 load's 25.5 GiB of caches over 3,014,656
tokens), the two compute buffers about 6.5 GiB at a 307,200 session limit (4.8 to 5.2 GiB read at 262,144 on
four cards, widened for two ranks' larger shares), the pool-map bank 14 bytes a cell a slot, state rows about
0.1 GiB a session, and 2 GiB of driver and allocator room.

| Sessions of 307,200 | Pool tokens | Caches a card | Used a card, weights at 66.2 GiB | Free |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 1,843,200 | 15.6 GiB | about 91 GiB | about 4.6 GiB |
| 7 | 2,150,400 | 18.2 GiB | about 93.8 GiB | about 1.8 GiB |

Six is the default: it keeps the DP qualification's 4,096 MiB free even at the heavy end of the weight range,
with the vision projector's 1.05 GiB on the first card inside that room. Seven fits only if the weights land
near the light end, which is why it is the box's second read and not the default. `--slots 7` asks for it;
`--slots N` always sizes the pool as N whole sessions of the `--slot-context` limit.

Data-parallel split, per card. Routed experts halve (57.9 GiB) and everything else is whole (about 9.4 GiB), so
about 67 GiB of weights a card. A slot's private Q8 caches are 11,424 bytes a token on its owner's card only
(`scripts/glm53/capacity.py`'s geometry) plus about 0.13 GiB of recurrent state at MTP depth 2: 3.4 GiB a
307,200-cell slot. With about 8 GiB of buffers, driver room and the 4 GiB kept free, five slots a card fit and
six do not, so `dp-2` defaults to 10 slots of 307,200 cells. `capacity.py --devices 2 --slot-cells 307200`
turns two boot snapshots on the box into the measured count; its claims stay estimates, as on four cards.

The tensor layout is the default candidate: one pass a step whichever slots are busy, a shared pool sessions
draw from as they grow, and the layout every November 9 to 15 read ran on. The DP layout holds more 300K
sessions by these sums, at private caches that reserve their cells whether a session fills them or not, dense
weights read whole on both cards every pass, and a step that still splits on idle slots between busy ones
(glm53-goal.md, cause 2). The box reads both and the numbers choose.

## Speed, as a hypothesis

At a fixed row count each card reads twice the expert and dense bytes it read on four cards, so the
memory-bound part of a step roughly doubles, while host work, launches and the mirrored attention stay near
their four-card costs and a reduction crosses to one peer instead of three. From the November 11 step budget
(experts 16.2 ms of a 40.2 ms target decode at 69 rows) a two-card 69-row target decode lands near 55 to 70 ms.
At six sessions the steps are 18 rows, where launches and host work weigh more than weight reads, so the
per-session rate will sit closer to the four-card rate than half of it. This paragraph is a hypothesis with
its sources named; the four-card record shows box reads correcting this kind of prediction in both directions,
and the variant of RTX PRO 6000 (Server at 1,597 GB/s, Workstation at 1,792, Max-Q below both) moves every
memory-bound term.

## The box read, in order

One rental, two RTX PRO 6000 with peer access between the pair (`nvidia-smi topo -m` first: a pair that cannot
map each other's memory reads a different engine). Nothing here needs a second box; stop the rental the moment
a leg disproves its layout rather than tuning on rented time.

1. **The route, before any serving.** Build `test-backend-meta-allreduce-copy` and run `--bench 2`: p50 of a
   69 x 4,096 reduction on each route. This chooses tensor-2's `--reduction-route` default for the rest of the
   read; run step 3's sweep on the other route only if the bench is close.
2. **The load.** `serve.py --profile tensor-2 --dry-run`, then launch. Record free MiB a card at load from the
   server log. Under 4,096 MiB free, relaunch `--slots 5` and record that the default sizing missed, in this
   file, before any rate is read.
3. **The rate.** `rate.py --sessions 1 2 4 6`, two passes after the load (the first captures graphs), then the
   same at histories of about 4,000 tokens if time allows. Per-session tok/s at each count is the number.
4. **The depth.** `dp_longctx.py --streams 6 --context-tokens 295000 --turns 2` against the same server: all
   six sessions populate near the cap at once, no cache lost, VRAM peak per card from its sampler. This is the
   leg the goal is made of; a short-context rate without it is not the goal read.
5. **The seventh session.** Relaunch `--slots 7`, read free memory and `rate.py --sessions 7` only. Keep
   whichever of 6 or 7 the free-memory rule allows as the recorded capacity; do not tune kept graphs to squeeze
   the seventh in on rented time.
6. **The other layout.** `serve.py --profile dp-2`: boot at 8 and 12 slots for `capacity.py --devices 2
   --slot-cells 307200`, serve at the estimated count under the same 4,096 MiB rule, `rate.py` at the same
   session counts it fits, one `dp_longctx.py` leg at its full slot count and 100,000 tokens (its 300K fill is
   hours at DP prefill rates; the reservation is what the boot already proved).
7. **Quality spot checks, if the clock allows.** Greedy 96-token continuations of one prompt set under
   `--reduction-wire q8` and `bf16`, and under both routes: report same-token agreement, no more. A KLD read
   of the wire and the 4-bit cache at two ranks and at depth is its own leg, not squeezed into this one.

Write what each leg read into this file, with the box's card variant, host CPU and topology at the top, then
destroy the box. Not carried to two cards, by design: `estimate_decode_pcie.py`'s DP4 PCIe model,
`compare_decode.py`'s fixed four-GPU comparison case, the four-GPU simulation page, and the decode lab's
four-card ideal. They are models and fixtures of the four-card question; `capacity.py` and
`simulate_decode.py` take `--devices 2` for the two-card one. `measure.py`'s hand-built tensor-2 control
command is unchanged, so its historical legs stay comparable; its `--recipe` takes `dp-2` through the launcher
like `dp-4`.
