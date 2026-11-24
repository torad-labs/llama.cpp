# Inkling 2 integrated verification

Source build commit: `c41aacd3063766f72fb2cf8d5699642d2943a31d`. The full Release
build passed on CloudRift `buildbox` (NVIDIA GeForce RTX 5090, compute capability
12.0, driver 580.159.03, 32607 MiB reported by nvidia-smi; CUDA toolkit 13.3.73,
GNU 13.3.0). The build directory remains on the box for reuse.

## Build

From the synchronized campaign source root, via `on-box`:

```sh
CUDACXX=/usr/local/cuda/bin/nvcc cmake -S . -B /home/riftuser/build-campaign-inkling2 -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON
flock /tmp/buildbox-build.lock cmake --build /home/riftuser/build-campaign-inkling2 -j 3
```

The first configure could not find nvcc on PATH. With CUDACXX set, compilation
started. The local task session stopped waiting during that build, so the
build was resumed in the same directory. CUDA 13.3 emitted references to
`cublasGetStatusString` and `cuGetErrorString` from the shared header, exposing
missing direct link dependencies in CUDA-compiled test fixtures.
`a8718ef0b` and `c41aacd30` fixed the links; no kernel or assertion changed.
The final incremental build linked every target and exited 0.

## Test results

Commands ran from the source root unless a different directory is shown.
These variables expand to the actual paths and CTest filter used:

```sh
B=/home/riftuser/build-campaign-inkling2
V=$B/server-tests-venv
G=/tmp/buildbox-gpu.lock
CTEST_RE='inkling-seam|inkling-route-gate|inkling-router-bake|inkling-fuse-chain|inkling-kvg|test-inkling-fp8-pair|test-inkling-kv-fp8acc|inkling-kquant-jrule'
```

| Check | Invocation | Result | Key result |
| --- | --- | --- | --- |
| Tunables registry | `python3 -B scripts/tunables.py check` | PASS | 562 parameters; 487 available, 75 not yet |
| Inkling Python tests | `cd scripts/inkling2 && PATH="$V/bin:$PATH" python3 -B -m unittest discover -p 'test_*.py'` | PASS | 281 tests, 13.634 s |
| Wait audit | `python3 -B scripts/inkling2/wait_audit.py --check --evidence reports/inkling2-wait-audit.json` | PASS | 39 sites: 5 counter, 15 kept-host-data, 19 out-of-path, 0 barriers |
| FP8 accumulation evidence | `python3 -B scripts/inkling2/fp8_accumulation.py --check --evidence reports/inkling2-fp8-accumulation.json` | PASS | 47.2x pair error reduction, fold32 32/32 exact, 8/8 tile cases, 116/116 argmax for served and restore |
| Route histogram evidence | `python3 -B scripts/inkling2/route_histogram.py --check --evidence reports/inkling2-route-histogram.json` | PASS | Exit 0; saved evidence remains branch c, 649224192 slot draws, no measured top-1 share |
| Tunables binary | `"$B/bin/test-tunables"` | PASS | 444 listed, 378 checked |
| Targeted CUDA CTest | `flock "$G" ctest --test-dir "$B" --output-on-failure --no-tests=error -R "$CTEST_RE"` | PASS | 37/37, 1534.76 s; no skips |
| AllReduce copy default | `flock "$G" "$B/bin/test-backend-meta-allreduce-copy"` | PASS | 8 s; 0 of 8396800 values differ on the largest buffer parity check |
| AllReduce copy, four ranks | `flock "$G" "$B/bin/test-backend-meta-allreduce-copy" --logical 4` | PASS | 312 OK cases, 127 s; logical ranks on one GPU |
| Reduction queue | `flock "$G" "$B/bin/test-backend-meta-split" --reduction-queue` | PASS | 16 cases, 576 reductions per case, all same=1, 7 s |
| AllReduce capture fallback | `flock "$G" "$B/bin/test-backend-meta-split" --ar-fallback` | PASS | 2 ranks, 2 captures, 4 replays; bit-exact across 6 evaluations, 2 s |
| 23-row fixture, warm off | `flock "$G" env GGML_CUDA_L2_EXPERT_WARM=0 "$B/bin/test-inkling-kquant-tile" --tma-23` | PASS | 9/9 shapes; worst NMSE 2.539e-05, default dispatch |
| 23-row fixture, warm on | `flock "$G" env GGML_CUDA_L2_EXPERT_WARM=1 "$B/bin/test-inkling-kquant-tile" --tma-23` | PASS | 9/9 shapes; worst NMSE 2.539e-05, default dispatch |
| TMA enabled, warm off | `flock "$G" env GGML_CUDA_MMQ_TMA=1 GGML_CUDA_MMVQ_KQUANT_MMID_MAX=23 GGML_CUDA_L2_EXPERT_WARM=0 "$B/bin/test-inkling-kquant-tile" --tma-23` | PASS | 9/9 shapes; worst NMSE 2.512e-05 |
| TMA enabled, warm on | `flock "$G" env GGML_CUDA_MMQ_TMA=1 GGML_CUDA_MMVQ_KQUANT_MMID_MAX=23 GGML_CUDA_L2_EXPERT_WARM=1 "$B/bin/test-inkling-kquant-tile" --tma-23` | PASS | 9/9 shapes; worst NMSE 2.512e-05 |
| Inkling 2 defaults | `flock "$G" "$B/bin/test-llama-archs" -b CUDA0 -a inkling2` | PASS | 2/2, CUDA0 NMSE 3.05e-07 |
| Inkling 2 opt-ins on | `flock "$G" env GGML_CUDA_MOE_ROUTE_GATE=1 LLAMA_INKLING2_ROUTER_BAKE=1 GGML_CUDA_L2_EXPERT_WARM=1 GGML_CUDA_MMQ_TMA=1 "$B/bin/test-llama-archs" -b CUDA0 -a inkling2` | PASS | 2/2, CUDA0 NMSE 3.62e-07 |
| Inkling 2 legacy sync | `flock "$G" env GGML_CUDA_AR_COUNTER_SYNC_LEGACY=1 GGML_CUDA_COUNTER_SYNC_LEGACY=1 GGML_CUDA_DISABLE_GRAPHS=1 "$B/bin/test-llama-archs" -b CUDA0 -a inkling2` | PASS | 2/2, CUDA0 NMSE 4.01e-07 |
| GLM 5.3 regression | `flock "$G" "$B/bin/test-llama-archs" -b CUDA0 -a glm5next` | PASS | 4/4, 51 s; tensor regression 0 differing top tokens |
| Speculative batch capacity | `cd tools/server/tests && flock "$G" env LLAMA_SERVER_BIN_PATH="$B/bin/llama-server" LLAMA_SERVER_TEST_PORT_BASE=18085 N_GPU_LAYERS=99 "$V/bin/pytest" unit/test_speculative.py -k test_draft_respects_batch_capacity` | PASS | 1 passed, 17 deselected, 91.25 s; fixture asserts both slots hit draft cap 15 with ubatch 8 and preserve tokens |

The switch-on architecture run used only `GGML_CUDA_MOE_ROUTE_GATE`,
`LLAMA_INKLING2_ROUTER_BAKE`, `GGML_CUDA_L2_EXPERT_WARM` and
`GGML_CUDA_MMQ_TMA`. Route histogram and engine ledger diagnostics were off.
Device arrival counters are the default; the legacy arm disabled CUDA graph
capture so the uncaptured sync path is eligible. The synthetic architecture
fixture checks compatibility with these settings; the targeted CTests cover
served-shape route gating, baked fusion and the unconditional f8_e4m3 dispatch
change (including the separate one-byte and FP8 accumulation switches).
The two additional TMA arms use the CTest fixture's dispatch settings so the
warm comparison does not rely on TMA being on by default.

The initial Python discovery run lacked NumPy and found stale wait source
locations. `69635ddb6` realigned the existing wait ranges after WS-2 and WS-4,
without changing the checker, classifications or measurements. No rebuild was
needed for that evidence-only change. The Python environment was prepared with
`python3 -m venv "$V"`,
`"$V/bin/python" -m pip install -r tools/server/tests/requirements.txt` and
`"$V/bin/python" -m pip install 'pyyaml>=5.1' 'tqdm>=4.27'`.
The wait audit and full Python discovery passed on rerun.

Logs on CloudRift under `$B`: `verification-job.Izk3Izkq` (build and suite),
`integrated-verification.1YIxEskm` (individual checks),
`integrated-tma.nrSc35JU` (TMA-enabled arms),
`integrated-python.y9QJpT4t` (passing wait audit) and
`integrated-python.8H2S3mVd` (passing full Python discovery).

OVERALL: PASS
