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
sizes for this goal) launches the pack with the settings the four-card work measured, sized for two cards. Its
plan is pinned by `scripts/glm53/test_qualification.py`, and the tensor-4 plan is what it was before this change.
The previous two-card recipe (three slots, a 524,288-token pool, Q8 latent caches, none of the settings below)
is replaced by it; its November 2 sweep stays in [GLM53-SERVING.md](../backend/GLM53-SERVING.md) as history.
The DP profiles (`dp-2`, `dp-4`) and their tooling were removed on November 16, 2026: production runs the
four-card tensor layout with 23 sessions on a shared pool, and the decode cost follows the requests decoding at
once, not the slots reserved, so the data-parallel split has no case left (Marcos, November 16, 2026). The
engine still takes `-sm dp`; nothing in the tree selects it.

What the two tensor profiles now share, because the engine work behind them never counted ranks: the 4-bit
latent cache, prompts read in chunks apart from replies, graphs kept by shape for any set of sessions
(`LLAMA_GRAPH_ANY_SEQS` with kept graphs and held CUDA executables), coupled MTP drafts, prompt-cache saves and
restores beside the decode steps, and every default-on host and kernel change of the four-card period. Two more
of that period's asks landed after this page's sums and are live here too: `kpool_indices`, some 14 launches in
every round, no longer sorts its padding (TORAD.md, row `eeb9dde73`), and a prompt step beside replies can be
bounded by time, `--prompt-step-ms` (row `ddd16b73c`), which is an opt-in on both tensor profiles like
`--session-step` with `--prompt-shapes` and `--prefix-copy`; the box read below decides those three for two
cards. Two settings deliberately differ:

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

Six whole sessions is the default pool: it keeps the qualification's 4,096 MiB free even at the heavy end of
the weight range, with the vision projector's 1.05 GiB on the first card inside that room. Seven fits only if
the weights land near the light end, which is why it is the box's second read and not the default;
`--context 2150400` asks for it. The pool holds six whole sessions and the launcher reserves twelve, as the
four-card production server reserves 23 slots against a pool far below 23 whole sessions: the sessions that
cost time are the ones decoding at once, not the ones reserved (Marcos, November 16, 2026). `--slots` sets the
reserved count and never resizes the pool; `--context` sizes the pool and `--slot-context` a session's limit in
it.

The tensor layout is the only one the launcher serves: one pass a step whichever slots are busy, a shared pool
sessions draw from as they grow, the layout every November 9 to 15 read ran on, and the layout production runs.
The DP split held more simultaneously-full sessions by these sums, at private caches that reserve their cells
whether a session fills them or not, dense weights read whole on both cards every pass, and a step that still
splits on idle slots between busy ones (glm53-goal.md, cause 2). It was removed, with its tooling, on
November 16, 2026.

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
   read; run step 3's sweep on the other route only if the bench is close. The same bench reads where the
   pair's own low-latency limits cross on its link: the 16 KB host limit was crossed on a Gen5 x4 link between
   two desktop cards, and a 3-row verify's 24 KB is over it, so those reductions take the card-to-card path to
   128 KB where the pair maps each other's memory and the chunked kernel where it cannot (TORAD.md, rows
   `1cbc66f49`, `3bef5edc2`). Pin what the pair reads in the launcher.
2. **The load.** `serve.py --profile tensor-2 --dry-run`, then launch. Record free MiB a card at load from the
   server log. Under 4,096 MiB free, relaunch with a pool one session smaller, `--context 1536000 --slot-context 307200`, and record
   that the default sizing missed, in this file, before any rate is read. Keep `--slot-context 307200` with any explicit pool size so that the limit and the 32 kept graphs stay unchanged; `--context` alone removes the limit and selects 128 kept graphs.
3. **The rate.** `rate.py --sessions 1 2 4 6 12`, two passes after the load (the first captures graphs), then
   the same at histories of about 4,000 tokens if time allows. Per-session tok/s at each count is the number;
   the 12-session pass reads what twelve reserved sessions decoding together cost against six.
4. **Short turns.** Agent turns of tens of tokens are where graph shapes churn: with a dozen sessions a
   session's graphs change shape every 2,048 tokens it grows, against 32 kept graphs. Relaunch with `--session-step
   65536 --prompt-shapes 8` and repeat step 3's sweep, with `prompt_evals.py` on a traced pass for the counts
   of graphs built and replayed. On the two-rank fixture a turn of 1 to 32 new tokens read in 45.9 ms against
   92.8 and sessions decoding beside a reader ran 39.0 against 36.0 tok/s each, while prompts of hundreds of
   tokens read slower in several passes (TORAD.md, rows `c9e6bbc4b`, `052fa4c50`): the pair says which side
   wins at six sessions. If it agrees, the two become tensor-2's defaults in the launcher, as the async save
   and restore did after their four-card read.
5. **A read beside a stream.** One session streams a long reply while another reads about 200,000 tokens; the
   streamer's longest wait for a chunk is the number, as #238's box read it. Then relaunch with
   `--prompt-step-ms` at about twice the step time step 3 read and read it again, with the server's `prompt
   step beside replies` lines at `-lv 4` saying where the count settled. Two cards' steps run longer than
   four-card ones, so a fixed count costs more here than #255's capture read there; the time, not the count,
   is what carries over.
6. **The depth.** `dp_longctx.py --streams 6 --context-tokens 295000 --turns 2` against the same server: six
   sessions populate the whole pool near the cap at once, no cache lost, VRAM peak per card from its sampler.
   This is the leg the goal is made of; a short-context rate without it is not the goal read.
7. **The seventh session.** Relaunch `--context 2150400 --slot-context 307200`, a seventh whole session in the pool, read free
   memory and `rate.py --sessions 7` only. Keep whichever pool the free-memory rule allows as the recorded
   capacity; do not tune kept graphs to squeeze the seventh in on rented time.
8. **Quality spot checks, if the clock allows.** Greedy 96-token continuations of one prompt set under
   `--reduction-wire q8` and `bf16`, and under both routes: report same-token agreement, no more. A KLD read
   of the wire and the 4-bit cache at two ranks and at depth is its own leg, not squeezed into this one.

Write what each leg read into this file, with the box's card variant, host CPU and topology at the top, then
destroy the box. Removed with the DP profiles on November 16, 2026: `capacity.py`, `simulate_decode.py`,
`compare_decode.py`, `server_matrix.py`, `control_matrix.py` and `workload.py`; git has them. The decode lab
and `estimate_decode_pcie.py` stay: they read recorded captures of the four-card question, and nothing launches
through them. `measure.py`'s hand-built tensor-2 control command is unchanged, so its historical legs stay
comparable.
