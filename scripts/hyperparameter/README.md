# The readers we tune by, kept in the repo

These seven scripts are the box-side analysis readers written during the November GLM-5.3 campaign on the rented
four-card boxes. They lived only on those boxes' disks and were lost when the rentals were destroyed; they were
recovered from the campaign sessions' transcripts on 2026-11-22, so the lessons of one model's tuning are reusable
by the other (Inkling 2 Small and GLM-5.3 Flash) and by every later run. The measurements they produce are the ones
the search and the engine work are judged by, and the engine's own ledger (`ggml-ledger.h`, `scripts/ledger.py`)
now carries part of the same readings by itself.

Each script reads a log or trace its own docstring says how to produce (`serve.py --trace-decode`, the graph
trace at `-lv 4`, or an Nsight export at GGML_CUDA_NVTX=1); the usage line is the first line of each file.

| Script | Reads | Answers |
| --- | --- | --- |
| `agents.py` | a running server's HTTP API | drives N agent-shaped sessions (short tool-result prompts, short replies, arrivals overlapping), prints tokens a second in all and a session |
| `idle-census.py` | Nsight SQLite trace | inside the steady target evals of one row count, per device: before the first kernel, kernel-union time, gaps by copy size, bare gaps by length, after the last kernel |
| `passcount.py` | `-lv 4` graph-trace log | each target eval that carried prompt rows: its passes (graph builds and slot reuses between begin and end), by rows and sequences, and its time |
| `evalwhy.py` | `serve.py --trace-decode` log | every target eval by rows, and what the engine did for it: kept-graph reuse or a build, subgraph launches replayed, run node by node, or captured |
| `step-passes.py` | `-lv 4` graph-trace log | a step by passes: which shapes got their own build, which were reused, and the pass boundaries' times |
| `itersplit.py` | `serve.py --trace-decode` log | an iteration's time split by stage (target eval, post, host wait), per row count |
| `q8dense-census.py` | Nsight SQLite trace | the dense Q8_0 products of a decode stage on one rank: which MUL_MAT launches ran mul_mat_q, at what shapes, and their share of the stage |

`scripts/ledger.py` beside them reads the engine's own ledger instead of a profiler: `ledger.py watch URL` prints,
for a running server of any model, where a step's time went.
