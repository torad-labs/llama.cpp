# The drafter kit for Inkling 2 Small

The scripts that recorded Inkling 2 Small's own states on the rented four-card machine, tuned its stock draft on
them and measured the result, on 2026-11-19. What they found is in
`docs/torad/research/inkling-2-small/data/drafter-g7/README.md`, and
`docs/torad/research/inkling-2-small/drafter-plan/` holds the scenario for the next step and an index of the data.

The kit ran as it stands here. It was written in a day, for one machine and one model, so read it as a record of
what was run and a place to start from, and expect to change it.

## What is here

| | |
|---|---|
| `kit/` | the folder a job ships to the rented machine: `run.sh` and everything it calls |
| `mkjob.sh` | makes one job's folder from `kit/` with its own mode, stages and time limit |
| `upload-parts.sh` | carries a large input to the machine in parts of at most 1 GiB |
| `make_tokenizer_gguf.py` | writes `kit/tokenizer.gguf` from a model GGUF |
| `relay/` | copies a running sweep's training lines from the machine into MLflow |
| `material.py` | builds the folder a session served by the rented machine may read |

Inside `kit/`:

| | |
|---|---|
| `run.sh` | the driver: runs the stages `stages.txt` names, in order, and stops at the first that fails |
| `dspark_tool.cpp` | compiled on the machine against the engine build: reads requests through the served model and writes each token's hidden state at the draft's eight depths with the target's 64 likeliest tokens |
| `build_requests_v5.py`, `fetch_sources.py`, `make_plan.py` | the requests: public Claude Code sessions rebuilt behind the real system prompt and tools, and the plans that pick which are recorded |
| `launch.py`, `record.args` | the server's launch, the same as the served profile |
| `check_record.py` | checks a recording against the model: the states, the labels, the continuations |
| `dspark_model.py`, `dspark_data.py`, `dspark_step.py` | the draft in PyTorch, matching the engine's; the recording's loader; one training step |
| `train_dspark.py`, `train.args`, `sweep.args`, `pick.py` | the trainer, the sweep's trials, and the pick among them |
| `selftest.py`, `make_fixture.py` | the PyTorch draft against the engine's on a small fixture and on the real draft |
| `live_check.py`, `live_bench.py`, `live_replay.py`, `make_replay.py`, `trace_split.py` | against the live server: the trainer's copy token for token, the bench at 23 / 8 / 4 / 1 sessions, the replay of recorded sessions turn by turn, and a traced round split into its parts |
| `prefetch.py`, `prefetch/`, `restore-staged.py`, `restore-xet.py` | rebuild the served pack on the machine from its pinned public sources |
| `engine.sha`, `*.sha256`, `parts.sizes` | the engine commit to build, and the sums and sizes the inputs are held to |

Each stage is a function `stage_<name>` in `run.sh`. A job's `stages.txt` lists the ones it runs and `mode.txt`
says whether it works on the fixture or the pack.

## Running a job

```sh
./mkjob.sh g8-round pack "sweep pick bench_tuned" 240
cd <rig's checkout> && rig job run <this folder>/jobs/g8-round --name dspark-g8-round --box <the box's id>
```

`rig job status`, `rig job collect` and `rig job stop` follow it by that name; results land in
`~/.local/share/rig/local/jobs/dspark-g8-round/results/`. On the machine everything slow to make again (the engine
build, the pack, the datasets, the recordings) lives under `/workspace/rig/dspark`, so a later job finds it.

The relay runs on this machine while a sweep trains: `DSPARK_BOX` is rig's id for the machine and `DSPARK_BOX_SSH`
its ssh address as `rig vast status` prints it (`user@host:port`). MLflow's address and login come from
`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` in the environment.

## What the kit needs that is not in the repository

They are binaries or large inputs. On the machine they were made on they are in
`~/Documents/dev/infra/rig/local-nvme/research/inkling2-dspark-train-20261119/`.

| file | what it is | how to get it again |
|---|---|---|
| `kit/uv` | the `uv` binary for Linux x86-64, which makes the machine's Python | astral's release |
| `kit/tokenizer.gguf` | the target's tokenizer alone | `make_tokenizer_gguf.py <gguf-py> <the pack's GGUF> kit/tokenizer.gguf` |
| `kit/fixture-target.gguf` | the fork's small inkling2 fixture model (14 MB), which `make_fixture.py` builds the fixture's target and draft on | `kit/` in the folder above |
| `kit/parity/` | a small draft and the engine's outputs on it, which `selftest.py` holds the PyTorch draft to | `kit/parity/` in the folder above |
| `release.tar` in parts | the sealed restore of the served pack (9.3 GB), held to `kit/release.sha256` | `kit-pack/pack-job/` in the folder above |
| the stock draft in parts | 4.59 GB, held to `kit/draft.sha256` | `scripts/inkling2/README.md`, "Native draft"; `kit-pack/draft-job/` |
| `kit/sweep.name` | one word naming a sweep's folder on the machine (`g7` for the first round) | write it |

`jobs/` and these files are ignored by git here.
