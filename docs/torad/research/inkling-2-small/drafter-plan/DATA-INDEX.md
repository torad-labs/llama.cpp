# The past-run data for the drafter plan, and where each piece is

For the session in which Inkling 2 Small designs its drafter pipeline, and for the session that builds from its
plan. `SCENARIO.md` beside this page states the task. Paths are relative to the repository's root unless they start
with `~` (this machine, outside the repository) or `/workspace` (the rented machine, reachable only while it runs).
Everything of 2026-11-19 ran at engine commit `7b262d27f1c55a0f96842a2163823dbb620013a7`.

## Start here

| what | where |
|---|---|
| The first training round, read and tabled | `docs/torad/research/inkling-2-small/data/drafter-g7/README.md` |
| The day's other records, one folder a job | `docs/torad/research/inkling-2-small/data/drafter-20261119/` |
| The model, its architecture, the four-card service | `docs/torad/research/inkling-2-small/01-model-facts.md` to `13-batched-serving-results.md` |
| Every switch the fork adds | `TORAD.md` |
| How the service and the draft are launched | `scripts/inkling2/README.md`, `scripts/inkling2/serve.py` |
| The kit that recorded, trained and measured | `scripts/inkling2/drafter/README.md` |

## The drafter jobs of 2026-11-19

Each job's small records are in the repository under `docs/torad/research/inkling-2-small/data/`. Its full results,
with the logs, are on this machine under `~/.local/share/rig/local/jobs/dspark-<job>/results/results/`, and rig's
own log of the run is `…/results/.rig-job/log`.

| job | what it did | in the repository | only in the full results |
|---|---|---|---|
| `g1-build` | built the engine and the tool on the rented machine; lists the request sources | `drafter-20261119/g1-build/`: `SOURCES.md`, `box.txt`, `gpu.csv`, `engine-bin.txt`, `torch.txt` | `tool-compile.log`, `tool-syntax.log` |
| `g1-gate` | recorded the held-out set and checked the recording, the trainer's copy of the draft against the engine, and the stock draft offline | `drafter-20261119/g1-gate/`: `launch.json` and `meta-heldout.json` (the launch), `summary-heldout.json` (time and bytes), `check-heldout.txt`, `parity-real.txt`, `parity-fixture.txt`, `eval-stock.json`, `plan.txt` and `plan-stats.json` (the request plans and their sizes), `pack.txt` | `capture-*.log` (the server's log, head and tail, of each recording), `prepared.jsonl`, the smoke runs' `meta-` and `summary-` files |
| `g2-sweep` | the trainer's copy against the live server; the stock draft served | `drafter-20261119/g2-sweep/`: `live-stock.txt`, `live-stock.json`, `live-stock-rounds.json`, `bench-stock-server-acceptance.txt`, `selftest.txt`, `trainsmoke.txt`; the bench itself is `drafter-g7/bench-stock.txt` | `bench-stock.json`, `capture-live-stock.*.log` |
| `g3-none` | no draft served; the stock draft offering 7 | `drafter-g7/bench-none.txt`, `drafter-g7/bench-stock-d7.txt`, `drafter-20261119/g3-none/bench-stock-d7-server-acceptance.txt` | `bench-none.json`, `bench-stock-d7.json` |
| `g4-trace` | the stock draft served with tracing on: where a drafted round's time goes | `drafter-20261119/g4-trace/`: `bench-stock-trace-split.txt` and `.json` (the split by sessions and by rows), `bench-stock-trace.txt`, `bench-stock-trace-trace.txt`, `bench-stock-trace-server-draft-time.txt`, `bench-stock-trace-server-acceptance.txt` | `bench-stock-trace-marks.log.gz` (every mark) |
| `g5-round` | 23 recorded sessions replayed turn by turn with the stock draft; the stock draft walked offline | `drafter-20261119/g5-round/`: `replay-stock.txt` (by turn, by context, a line a session), `replay-stock-server-acceptance.txt`, `replay-stock-server-slots.txt`, `walk-stock.json`, `eval-stock.json` | `replay-stock.json`, `baseline.tail.log` |
| `g6-round` | the stock draft walked offline, request by request | `drafter-20261119/g6-round/walk-stock.json` | `baseline.tail.log` |
| `g7-round` | four learning rates, the pick, the tuned draft served | `drafter-g7/`: `README.md`, `sweep.txt`, `pick.txt`, `summary-lr*.json`, `train-lr*.txt` (the training logs), `sweep-lr*-eval-*.json`, `sweep-lr1e-5-walk-after.json`, `bench-tuned.txt`, `bench-tuned-server-acceptance.txt`, `draft-tuned.sha256` | `draft-tuned.gguf` (4.59 GB, the tuned draft), `sweep-lr*-train.jsonl` (every step), `sweep-lr*-walk-after.json` for the three other rates, `bench-tuned.json` |

Jobs `dspark-g0-*` in the same folder are the kit's own bring-up (syntax, smoke, diagnostics); their records carry
no measurement the later jobs don't.

How to read the walks: `walked` is tokens kept a round by the draft under the first-choice check, `ceiling` what a
draft naming the target's first choice every time would keep, `sampled` and `sampled_most` what a sampling draft
would keep under a check that keeps a token with the chance the target gives it. Keys `d3`, `d5`, `d7` are the
depth offered. `eval-*.json` holds the cross-entropy and the first-choice agreement by position.

## Drafter logs

- Training, a line a logged step: `docs/torad/research/inkling-2-small/data/drafter-g7/train-lr*.txt`. Every step:
  `~/.local/share/rig/local/jobs/dspark-g7-round/results/results/sweep-lr*-train.jsonl`, and the same runs are in
  MLflow, experiment `inkling2-drafter` (the address and login are in this machine's environment as
  `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`).
- The server's side of a served run: the `*-server-acceptance.txt` files (kept and offered, a line a slot a reply),
  `*-server-slots.txt`, `*-server-draft-time.txt`.
- The whole logs of each stage stayed on the rented machine: `/workspace/rig/dspark/logs/`.
- Yesterday's work on serving the stock draft (2026-11-18), including its conversion log:
  `~/Documents/dev/infra/rig/local-nvme/research/inkling2-dspark-20261118/` (33 GB; `convert.log`,
  `measure-draft.py`, `paid-01` to `paid-03`, `replacement/`).

## nsys and ncu profiles

- **ncu: none exist.** The rented machine's container refuses hardware counters (`ERR_NVGPUCTRPERM`), so no kernel
  was profiled with it.
- **nsys timelines of 2026-11-19**, taken by the engine seat (`llama-speed`) at the served profile, with and
  without the draft: timelines named `prodplain`, `nodesplain`, `engineplain`, `proddraft` and `nodesdraft`. On
  the rented machine they are under `/workspace/rig/jobs/speed-data1/out/` and `/workspace/rig/jobs/speed-data2/out/`
  (`<name>-summary.txt`, `<name>-window.json`, `<name>-server.log`, `<name>.nsys-rep`, `<name>.sqlite.zst`, and the
  setting searches' `trials.jsonl` and `best.json`). The engine seat collects them to
  `~/.local/share/rig/local/jobs/speed-data1/results/` and `…/speed-data2/results/` and puts the text under
  `docs/torad/research/inkling-2-small/`. They arrived at 5:43 and 5:59 PM CT, with a `speed-data4` to follow,
  and what they show is chapter 14 with its two data files; the `padplain` timeline, taken after the shared
  pool's padding was fixed, is the one to read for the host route as it now stands.
- **nsys of the four-card engine work of 2026-11-18**:
  `~/Documents/dev/infra/rig/local-nvme/research/inkling2-tp4-20261118/` (21 GB): `nsys-direct-native/`,
  `nsys-direct-work/`, `nsys-derivative-native/`, `nsys-derivative-work/` and `nsys-window2-production/` each hold
  a `service.nsys-rep`; `remote-results/gather-trace.nsys-rep`; read as text in `full-model-nsys-analysis.json` and
  `remote-results/nsys-stats.log`.
- Older four-card profiles (2026-11-09 to 11-11): `~/Documents/dev/infra/rig/local-nvme/research/tensor4-read-2026-11-09/`
  (`cen-*.nsys-rep`, `wsum-*.nsys-rep`, `box-logs-1111*/`). Which model each one profiles was not checked for this
  page.
- The traced bench of `g4-trace` above is the only split of a *drafted* round by its parts that this seat took.

## Bench outputs

| what was served | file |
|---|---|
| no draft, 23 / 8 / 4 / 1 sessions | `docs/torad/research/inkling-2-small/data/drafter-g7/bench-none.txt` |
| stock draft, depth 3 | `…/drafter-g7/bench-stock.txt` |
| stock draft, depth 7 | `…/drafter-g7/bench-stock-d7.txt` |
| tuned draft, depth 3 | `…/drafter-g7/bench-tuned.txt` |
| stock draft, traced | `…/drafter-20261119/g4-trace/bench-stock-trace.txt` |
| stock draft, 23 sessions replayed over 134 turns | `…/drafter-20261119/g5-round/replay-stock.txt` |
| the four-card service without a draft, earlier runs | `docs/torad/research/inkling-2-small/13-batched-serving-results.md`, `12-four-card-serving.md` |
| the settings search (`scripts/inkling2/search.py`) | the engine seat's `speed-*` and `knobs-*` jobs under `~/.local/share/rig/local/jobs/` |

## The engine, for the questions the plan will ask of it

| question | files |
|---|---|
| how the model is built and split over four cards | `src/models/inkling2.cpp`, `ggml/src/ggml-backend-meta.cpp`, `ggml/src/ggml-cuda/allreduce.cu`, `allreduce.cuh`, `allreduce-q8.cuh` |
| attention and its caches | `ggml/src/ggml-cuda/fattn-rel-decode.cuh`, `fattn-rel-mma.cuh`, `fattn-rel-prefill.cuh`, `attn-res.cu`, `src/llama-kv-cache.cpp` |
| the draft and its round | `src/models/dspark.cpp`, `src/models/dflash.cpp`, `common/speculative.cpp`, `tools/server/` |
| converting the shipped draft head | `scripts/inkling2/README.md` ("Native draft"), `convert_hf_to_gguf.py`, `scripts/glm53/dflash_own_head.py` |
| the chat template and its reasoning levels | `models/templates/thinkingmachines-Inkling-2-Small.jinja` |
| commits since the measured build | `git log 7b262d27f..HEAD` |

## On the rented machine

All under `/workspace/rig/dspark/`, and gone if the machine's disk is released:

- `pack/model/`: the served pack (283 GB), the projector and the stock draft.
- `rec-pack/heldout/` (92 GB): the held-out recording. `rec-pack/a/` (57 GB): the training recording so far, 84
  requests read and 73 continued.
- `g7-pack/lr1e-5/draft-tuned.gguf`: the tuned draft (also on this machine, above).
- `baseline-pack/`: the stock draft's offline walk and evaluation.
- `build-7b262d27…/bin/`: `llama-server` and `dspark_tool`. `src-7b262d27…/`: the engine source it was built from.
- `venv/`: the trainer's Python. `logs/`: every stage's log.

The machine's stop is set for about 7 pm CT on 2026-11-19 and was left as set. Stopped, it keeps its disk at
$0.83 an hour; the recordings are not copied anywhere else.

## For whoever arranges the Inkling session

- Whatever that session reads becomes a prompt sent to the rented machine, so rig's rule on what leaves this
  machine applies to it. `scripts/inkling2/drafter/material.py` builds a folder for it: the fork's tracked files
  without the seats' instruction files, and the text records of the run folders named above, with this machine's
  home path replaced. Run rig's credential scan on that folder before the session opens it. On the copy built on
  2026-11-19 the scan flagged 11 files of the four-card folder and one engine document
  (`docs/torad/glm53-dp-qualification-20261104.json`); they were left out, not judged.
- Inkling is reached through the Splice gateway as provider `inkling`. On 2026-11-19 the gateway served only models
  that belong to a head, and Splice's `gateway-providers` campaign is what lets it serve this one.
- The session needs to read, search the web and write its plan. It needs no credentials, no `rig` and no `vastai`.

## Open when this seat closed

- The acceptance rule is part of the plan, so no campaign row was added for a probability-aware check.
- The trainer can also train at the tokens of the prompts' model turns (`train_dspark.py --prompt-anchors`). That
  round was not run.
- `.gitleaksignore` names `TORAD.md:generic-api-key:664`; that line is now 666. The fork's lead decides the entry.
- Leftovers on the rented machine from a smoke test of the kit: `/tmp/kit-smoke`, `/tmp/smoke.log`,
  `/tmp/smoke-out`.
