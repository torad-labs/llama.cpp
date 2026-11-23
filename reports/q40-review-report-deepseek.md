# Adversarial review — Q4_0 KV decode work (branch draft-speed)

Reviewed tree: `/home/marcos/Documents/dev/infra/torad-llama.cpp` at the working-tree state of 2026-11-22.
The work is in commits `3280492de` (rel kernels), `4c4c8e0e6` (kv_stats rotation), `1dc3b3da8` (doc 18, serve,
kv1b runner), `69b948642` (relocation) and `847e4c638` (kv_stats usage text), plus the dirty working tree. The
brief's paths were stale: the note is now `reports/kv-q40-dispatch-pending.md`, the runner is
`scripts/inkling2/kv1b_run.sh`, and the `kv_stats` changes are committed rather than in the working tree. The
uncommitted `BEST_FATTN_KERNEL_REL_ROWS` hunk in `fattn.cu` and `fattn-rel-rows.cuh` is another seat's rider and
was excluded, as were the ledger/tunables/decode-trace/server riders the brief names.

Verdict: four findings above nit, two of them wrong behaviour or irreversible loss.

---

## Findings, most severe first

### 1. IMPORTANT — the Q4_0 one-cell decode reads the wrong nibble for blocks 1–3, and runs past the row

`ggml/src/ggml-cuda/fattn-rel-decode.cuh:99`

```c
const block_q4_0 & block = ((const block_q4_0 *) row)[col/QK4_0];
const int q = col < QK4_0/2 ? (block.qs[col] & 0x0F) : (block.qs[col - QK4_0/2] >> 4);
```

`col` is `32*i + lane` with `i = 0..3`, i.e. 0..127. The block index takes `col/QK4_0`, but the within-block
position never takes `col % QK4_0`. For every `col >= 32` the expression tests against 16, then indexes
`block.qs[col-16]`: for block 1 that is `qs[16..47]` and for block 3 `qs[80..111]`, all outside the 16-byte `qs`
array of a `block_q4_0`. The right test and index are `const int c = col % QK4_0;` then
`block.qs[c]` / `block.qs[c - QK4_0/2]`.

Failure scenario: serve with `GGML_CUDA_FATTN_REL_DECODE_Q4=1`, `LLAMA_INKLING2_KV_RUNS` unset (the served
profile sets no such variable; `scripts/inkling2/serve.py` never exports it), and the profile's
`--mean-mask-decode` on. A low-mean decode step sets `sparse_mean` and passes the gate at `fattn.cu:1391-1399`;
the dispatch at `fattn.cu:1476-1486` finds `fattn_rel_runs_takes` false (no runs tensor) and
`fattn_rel_mma_shapes` false for Q4_0 (`fattn-rel-mma.cuh:38-42` admits only F16 and the one-byte types), so it
takes `ggml_cuda_fattn_rel_decode` → `fattn_rel_decode_rows<Q4_0>` for both K and V. A host emulation with C
semantics gives 88 of 128 head components wrong on a random block row (columns 32..127; the exact count varies
with coincidental nibble matches), and the deepest read is row byte 167 of a 72-byte row for `col = 127`, 95
bytes past its block's end. For the last cell of the pool that is a read past the tensor, not only into the next
cell.

The same branch is what serves query-grouped drafted rows (`group` 2–4) and every 33..128-row verify step, since
`fattn_rel_runs_takes` requires `group == 1` and `q->ne[1] <= 32`. Nothing in `tests/test-backend-ops.cpp`
exercises it: there is no Q4_0 arm behind the switch, and the existing relative Q4_0 cases run with the switch
off on the dense deq path. This is the only kernel the feature's fast path uses unless `LLAMA_INKLING2_KV_RUNS`
is turned on.

### 2. IMPORTANT — `kv1b_run.sh` can restore the whole tree, and continues silently when a restore fails

`scripts/inkling2/kv1b_run.sh:18-31`

- A trailing or doubled comma in `RESTORE_COMMA_LIST` yields an empty `$p`; the loop then runs
  `git checkout "$BASE" -- "./"`. In this repo `git ls-files -- ./` matches all 5,041 tracked files, so a typo
  such as `ggml/src/ggml-cuda/fattn.cu,` restores every tracked path to the base commit — working tree and
  index. On any checkout shared with another seat that is exactly the loss the house law forbids, and the staged
  base versions are then picked up by the next seat's named `git add`, the 847e4c638 incident recorded at
  `reports/kv-q40-dispatch-pending.md:125-133`.
- `git reset --hard "$BASE"` + `git clean -fdq` for `RESTORE=all` is unconditional and irreversible: it discards
  every uncommitted edit and untracked file (including the seat's own `reports/`), with no guard that the script
  is on a disposable box copy.
- The checkout has no error check. `git checkout` failing (a path the base does not contain, a rename, a
  mistyped path) prints to stderr and the script proceeds to configure, build and test the *unrestored* tree,
  tagging the log with the restored arm's name. The whole purpose of the runner is attribution, so that is a
  silent wrong result. Add `|| exit 1`, reject empty list entries, and keep the index out of it (for example
  `git show "$BASE:$p" > "$p"` after a copy-out, or a warning plus a hard stop when the tree is not disposable).

### 3. IMPORTANT — the prepared Q4_0 arm also quantizes the sliding-window cache, unlike the profile it is meant to measure

`docs/torad/research/inkling-2-small/18-global-kv-off-f16.md:140` vs `scripts/inkling2/serve.py:232-233,245-246`

The arm is written as `D --kv-cache q4_0` (and `C --kv-cache q8_0`). The launcher sets
`kv = args.kv_cache` and only adds `--cache-type-k/v-swa` when `--swa-kv-cache` (or the draft layout) supplies
one (`serve.py:232-233`), and the engine then uses `type_k_swa = type_k` when no SWA type was given
(`src/llama-context.cpp:950`). So arm D as written serves Q4_0 in the sliding layers too, while section 3.3 of
the same document defines the target as "global KV in the chosen format, sliding KV F16"
(`18-global-kv-off-f16.md:192`). A seat running the arm exactly as listed measures a different cache mix and
attributes a full-cache effect to a global-only change. The launcher behaves as documented; the arm needs
`--swa-kv-cache f16`.

### 4. IMPORTANT — the rotated-on-store acceptance is inert for every state the reader accepts, and the usage text states the opposite

`scripts/inkling2/kv_stats.py:13-14`, `:231-236`, `:250-251`; `scripts/inkling2/test_kv_stats.py:353-358`

The new comment says a state from a rotated cache is read as-is, and the usage block says "a state saved from a
quantized cache parses as well: such a cache is rotated on store". Neither holds:

1. A quantized cache writes its own type id and raw row bytes (`src/llama-kv-cache.cpp:3133-3170`), and the
   parser still rejects any type outside `{0: f32, 1: f16, 30: bf16}` at `kv_stats.py:250-251` before the `n_rot`
   fields can matter. `test_kv_stats.py:365` pins that refusal for type 8. Following the usage text with
   `--kv-cache q4_0` ends at "cache type id 2 is not F32, F16 or BF16", exit 2.
2. The caches that do parse are F16/F32/BF16. Rotation requires `ggml_is_quantized(type_k|type_v)` or
   `lid_k_hdmd()` (`src/llama-kv-cache.cpp:402-435`); Inkling 2 does not override `lid_k_hdmd()`
   (`src/llama-model.h:1086`), so an Inkling 2 state with accepted types has `n_rot_k = n_rot_v = 0`. The
   acceptance path is unreachable in the documented flow.

The statistical claim also has a domain gap for the states that do parse. The simulation quantizes the F16
values it reads, but a deployed `--kv-cache q4_0/q8_0` cache is rotated on store first, so the Q4_0/Q8_0 errors
and the `+chan` scale arm are computed in the unrotated basis, not "exactly what the kernel sees". That matters
for K2: the reported Q4_0 error is not the deployed kernel's error. What does survive rotation mathematically is
the per-vector check: the Hadamard is orthonormal, so K's L2 bound and V's RMS bound hold exactly, and the
existing 0.2%/0.01 slack covers the single f16 rounding. What is reinterpreted is every per-channel figure
(channel RMS, hot-channel count, per-channel max, `+chan`): for K at head 128 the rotation is
`n_rot_k = 128`, so all 128 channels of a head mix; for V `n_rot_v = 64`, so each half mixes. The v_trans refusal
is still in place (`kv_stats.py:229-230`).

---

## Nits

- `ggml/src/ggml-cuda/fattn-rel-elem.cuh:25` still points at `research_notes/kv-q40-dispatch-pending.md`; the
  note moved to `reports/` in 69b948642 and only doc 18's pointers were updated. A reader following the code
  comment finds no file.
- `scripts/inkling2/kv1b_run.sh:3,7` still show the old invocation `on-box sh research_notes/kv1b-run.sh dirty`;
  the file is now `scripts/inkling2/kv1b_run.sh`, so the documented command is a "No such file" failure.
- `GGML_CUDA_FATTN_REL_DECODE_Q4` is read through `ggml_env_switch` (`fattn-rel-elem.cuh:27`) but is not declared
  in `tunables.toml`; `scripts/tunables.py` scans `.cuh` files (`SOURCE_EXTS`, line 163) and reports
  "GGML_CUDA_FATTN_REL_DECODE_Q4 is read in ... and tunables.toml does not declare it"
  (`scripts/tunables.py:190-191`). The `repo-tunables-clean` row's check (`python3 scripts/tunables.py check`)
  therefore gains a third problem after its two existing ones are fixed. The note at
  `reports/kv-q40-dispatch-pending.md:67-72` acknowledges the missing entry but justifies it with "both are
  frozen"; they are no longer frozen in this tree.
- `docs/torad/research/inkling-2-small/18-global-kv-off-f16.md:60` says the Walsh-Hadamard rotation has `n_rot`
  64. That is true for V only. `n_rot_k` starts at 64 and doubles while the head size is divisible by `2*n_rot`
  (`src/llama-kv-cache.cpp:419-427`), so at head 128 `n_rot_k = 128`. Any "channels mix within groups of 64"
  reasoning is wrong for K.
- `18-global-kv-off-f16.md:69-70` says the tile kernel's element decode has "F16, Q8_0 and E4M3 instances" and
  "does not take Q4_0 yet". The rel-mma kernel still declines Q4_0 (true), but `fattn_rel_elem<Q4_0>` now exists,
  so the instance list is stale against this same work.
- `reports/kv-q40-dispatch-pending.md:1,8,34-37,93-95` is stale about its own state: the dispatcher patch is
  applied (`fattn.cu:1397-1398`), the "uncommitted, editable files only" section is committed in 3280492de, and
  the "rebase this working-tree state as its own commit" step is done. A reader could re-apply Patch 1 or hunt
  for uncommitted work that is not there.
- `scripts/inkling2/test_kv_stats.py:353-358` only parses and reads a rotated state; it never runs
  `collect_stats`, the self-check or the format simulation on one, so the comment's guarantee that the
  self-check bounds stand under rotation has no test. A future change to the statistics that assumes the
  unrotated basis would pass every test. (The removed refusal did not cover that either, so this is a
  strengthening gap rather than a regression.)

---

## Claims checked clean

- **Q1, the Q4_0 tile store.** `fattn_rel_elem<Q4_0>::store` (`fattn-rel-elem.cuh:124-162`) is a correct
  transcription of `flash_attn_ext_raw_dequant_pair<Q4_0>` (`fattn-mma-f16.cuh:790-823`): the nine words start
  at load word `part`, block scales come from `w[0] & 0xFFFF` and `w[4] >> 16` (block 1 at row bytes 18-19 / block
  3 at 54-55), the `0x5432`/`0x4140`/`0x4342` selectors give qs indices `4t+2h`, and lows land in values 0-15,
  highs in 16-31, exactly as `dequantize_row_q4_0` reads them (`ggml/src/ggml-quants.c:524-534`). I emulated the
  full load/word-extract/byte-perm/nibble/f16-bias path and compared it with `dequantize_row_q4_0` over 2,000
  random rows across scales `{0, 1e-7, 3e-7, 1e-3, 0.02, 0.5, 3.0, 255, 6e4}`: bit-identical, subnormals
  included. The bias trick is exact (`0x6400|q` is 1024+q, the subtraction 1032 gives q-8 with no rounding) and
  the single `__hmul2` rounds once, the same as the dense reference; any FTZ caveat is shared with the dense path
  the tile is required to match. `fattn_rel_elem_align(Q4_0) = 8` is right for 72-byte rows and the `int2` loads.
- **Q2, staging reachability.** No path reaches a type it cannot stage. The rel-mma chain declines Q4_0 by
  construction (`fattn_rel_mma_type` / `fattn_rel_mma_elem`, `fattn-rel-mma.cuh:38-52`, so `elem` and `q8` are
  both false in `fattn_rel_mma_shapes`), the runs dispatch has a Q4_0 case (`fattn-rel-runs.cuh:587-589`) and
  `fattn_rel_runs_type` admits it only under its own switch (`:25-28`), and the one-cell pick has a Q4_0 case
  (`fattn-rel-decode.cuh:283`). The one-cell read is the broken one (finding 1), not an unstageable type. The
  read window is safe: part 0 covers row bytes 0..39, part 1 covers 32..71, both inside the 72-byte row, and
  `k->nb[1]%8`, `k->nb[2]%8` and `data%8` are enforced by `fattn_rel_runs_takes` / `fattn_rel_runs_aligned`.
- **Q3, the gate.** `fattn.cu:1397-1398` matches the other type gates: prefill and short-rows stay F16-only
  (`:1385`, `:1347`), rel-mma declines Q4_0, and the runs kernel keeps its own type test. The mean-mask class
  reaches the one-cell kernel with the same packed-I16 shape and is where finding 1 bites. There is no q4 path
  that assumes 16-byte global copies: the q4 stage is register-load/store at 8 bytes, and the f16/one-byte
  cp.async sizing is unchanged.
- **Q4, rotation.** Orthonormal Hadamard preserves each vector's L2 norm, so the K ≤ 22.6274 and V RMS
  [0.9, 1.01] self-check bounds survive rotation mathematically, with the existing single-rounding slack. Dot
  products are preserved because `llama-graph.cpp:4427-4430,4546-4549` rotates Q with the same matrix the store
  uses. Per-channel statistics do not survive: they become storage-basis mixtures (K over 128, V over 64). The
  v_trans refusal remains. The false part of the commit is the reachability/usage claim (finding 4).
- **Q4, block layout.** The simulation quantizes `x.reshape(-1, 128)` in blocks of 32 (`kv_stats.py:517-530`),
  i.e. 32 consecutive storage values, which is the kernel's block order; that part of the claim is right. The
  basis of those values is the caveat in finding 4.
- **Q5.** The positive parse test does what it says (header, values unchanged, `info()`); the residual coverage
  gap is the untested stats/self-check-on-rotated path (nit above).
- **Q6, launcher.** The flag pass-through is sound: explicit `--kv-cache q4_0` wins over the draft layout's
  default q8_0, the draft's own KV stays `--spec-draft-type-k/v q8_0` (`serve.py:259-265`), and `--swa-kv-cache`
  still overrides per layer. The measurement arm is what is miswritten (finding 3), not the launcher.
- **Q7, the note's bit-for-bit argument.** The tile-store argument is step-for-step correct against
  `fattn-mma-f16.cuh:790-823` and `ggml-quants.c:524-534`; the read-window claim holds ([0..39] and [32..71] in
  a 72-byte row, no over-read); `fattn_rel_mma_shapes` really does never pass for Q4_0, so the optional Patch 2
  is correctly described as changing nothing on its own. What the note does not cover is the one-cell
  `fattn_rel_decode_value` branch (finding 1) and it is stale about its own committed state (nit).
- **Q8, tree-verifiable facts of doc 18.** Capacity arithmetic is exact: `block_q8_0` 34 B / 32 values and
  `block_q4_0` 18 B / 32 values; 4,096 values per cell per card per 8 global layers gives 8,192 / 4,352 / 4,096 /
  2,304 B; times 3,014,656 cells gives 24.70 / 13.12 / 12.35 / 6.95 GB and freed 11.58 / 12.35 / 17.75 GB.
  The launcher accepts `--kv-cache q4_0` (`serve.py:96`), `--swa-kv-cache`, `--mean-mask-decode`,
  `--reduction-wire q8`; `GGML_CUDA_FATTN_REL_DECODE_Q4`, `GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE`,
  `LLAMA_ATTN_ROT_DISABLE`, `LLAMA_INKLING2_BUCKET_PREFILL_KV` and `LLAMA_INKLING2_MEAN_MASK_DECODE` all exist
  as written. The draft-layout sentence (`:59`) matches `serve.py:232-233`. The `n_rot 64` and stale-instance
  nits are above.
- **Q9, shell.** Apart from the restore defects and stale paths: `set -u` is safe with `${1:-}`, `${2:-}`,
  `${3:-}` and the `${RESTORE:-}` expansion; `IFS=,` is saved and restored; the quoted `"./$p"` prevents globbing;
  the `||` build handlers and the final `ctest` propagate the right exit codes. The fix should still validate
  `TAG` (documented `a-z0-9-`, unvalidated) and refuse empty restore entries.

---

## Excluded from this review

The uncommitted `BEST_FATTN_KERNEL_REL_ROWS` hunk (`fattn.cu:1401-1405`, `1435-1436`, `1473-1475`) and
`fattn-rel-rows.cuh` are another seat's live work. Note only that `fattn_rel_rows_takes` accepts Q4_0/Q8_0 and is
checked after the REL_DECODE gate, so on this dirty tree a q4_0 decode with the Q4 switch off may route there
instead of the dense path; that changes the meaning of "dense-path decode today" in `serve.py:97` for anyone
measuring before that rider lands or is reverted.
