# Decode comparison

A local comparison page for GLM-5.3 Flash on four RTX PRO 6000 Blackwell Server Edition GPUs.

## Open it

From the repository root:

```sh
python3 scripts/glm53/decode_lab.py serve --port 8768
```

Open **http://127.0.0.1:8768**. The server binds only to loopback, serves this analysis folder, and recalculates the ideal ledger using the existing Python estimator. Stop it with Ctrl-C. No inference server, GPU or external service is required.

The default is **one active session at 400k context on four GPUs**. The left shows the ideal budget; **Future capture** on the right mirrors its GPUs, phases and layers with empty measurement slots. No historical capture loads automatically. Empty means not measured, never zero. The page does not manufacture a speedup ratio.

The server sends `Cache-Control: no-store` for local responses. If a browser still retains the older page, open **http://127.0.0.1:8768/?view=future**.

## Matched-comparison preparation

The page now has a 1:1 visual structure, but no matched measurement yet. The [CPU-only preparation package](../glm53-comparison-preparation.md) encodes the real one-active-session/four-GPU recipe, adds opt-in host phase markers, recovers scheduling records omitted from the compact export, and tests matched logical-work comparison. Its readiness report keeps missing device counters and complete ideal costs explicit. No new GPU measurement has been made.

The real launcher uses adaptive MTP2. The ideal reference shows a cycle with two proposals and perfect acceptance, not an adaptive sequence or a variation distribution. A future match must identify that same work. The page does not yet render `glm53-matched-comparison-v1` results into the pending slots.

## Visual comparison

The default view contains no tables. Both sides have four GPUs, the same Draft 1 / Draft 2 / Verify / Catch up phases, and 45 target layers:

- **Blue:** ideal GDDR work.
- **Hatched orange:** ideal transfer/wait budgets.
- **Empty outlines:** future measurements, all still pending.
- **Hatched '?' on the ideal:** a per-rank timing the model cannot supply, distinct from an unrecorded future measurement.

A filled GPU mosaic has 100 squares: one square is 1% of modeled time, not a core or a memory-capacity unit. The ideal uses the busiest-owner model. Balanced owners share that assumption; timings for less-busy owners remain unknown when ownership is uneven. With one active session, GPU 0 owns its sequence and the other three still participate in routed-expert work; their full per-rank times are not modeled.

Click a GPU, category, PCIe path, phase or layer to inspect it. The ideal's four phase buttons and 45 layer bars open underlying budgets; the future counterparts explicitly say **Awaiting capture**. These are budgets and empty measurement slots, not aligned CUDA traces.

**Adjust model** changes the ideal and expected workload. **Open raw evidence** reveals the ledger and the historical-capture selector. Imported profiles also stay in raw evidence; even matching batch metadata does not fill a future cycle. Close an inspector with its Close button or Escape. Import remains local and retains the previous evidence on failure.

Historical activity uses blue for local kernels/device copies, orange for collective/host-KV kernels and copies other than known local D2D copies, and gray for no exported kernel/copy interval. Gray is not proof of idle hardware. Overlapping intervals count once, with transfer/wait taking priority. Summary shares use per-GPU interval totals, not a claimed critical-path duration. Timeline windows retain hover, focus, arrow-key navigation and exact-value inspection.

## What can be inspected

- **Layer budgets:** all 45 target layers, output/host boundaries, both draft proposals and the hypothetical cache-only catch-up. Open any group to inspect its substeps.
- **Substeps:** exact tensor read sets, expected routed-expert reuse, index scans, selected latent rows, KDA matrix/convolution reads and speculative outputs, individual gather/reduction payloads, host-KV stages, hidden-state handoffs and explicit latency budgets.
- **Operations:** named NVTX operations, shape-specific logical read/write estimates, GPU invocation means/percentiles and their underlying kernel events.
- **Kernels:** exact signature and GPU, invocation count, mean/P50/P95/max, summed duration, and every individual captured kernel event.
- **Memory copies:** actual CUDA copy payload and duration, a byte-matched reference at the selected directional link rate, measured payload GB/s and every copy event.
- **CUDA APIs:** host call count and duration, including graph creation/instantiation and synchronization. These are not added to GPU totals.
- **Activity:** per-GPU local/transfer/unrecorded time over 96 equal capture windows, with hover/focus inspection and an on-demand numeric table. These are disjoint interval unions, not summed kernel durations.

Both bundled captures retain **all** kernel, explicit-copy and runtime-API records, at the nanosecond units exported by Nsight. Event pagination does not sample or discard records. The source SQLite files also contain OS-runtime and other profiler tables that this CUDA-focused page does not export.

## Actual evidence

| Capture | Active / initial context | Kernels | Explicit copies | CUDA APIs | Kernels assigned to named operations |
| --- | --- | ---: | ---: | ---: | ---: |
| 4K | 1 / 4096 | 167,321 | 12,190 | 98,600 | 147,897 (88.4%) |
| 512 | 1 / 512 | 146,568 | 10,103 | 58,434 | 100,924 (68.9%) |

Both are historical Nsight Systems captures from runtime `82ad56cb8`, with 40 configured slots, one active session, host latent KV and adaptive MTP2. The recorded request includes a four-token prompt tail, not exclusively steady decode. Context grows during generation. Configured slots do not mean active sessions.

The 4K capture spans 1,735.8791 ms of GPU activity timestamps and contains 3,682.224325 ms of kernel time summed over four GPUs. Their union across GPUs is 992.603754 ms. These are three different measurements; none is interchangeable with request decode time.

The traces predate the later owner-local sampling and repeated-copy follow-ups. They contain sampling-related gathers as well as expert gathers. Historical kernel-family totals are not evidence of today's exact execution path.

Source files:

- `.Codex/tmp/dp-profile-20261104/receipts/results/timeline-4k.sqlite`
- `.Codex/tmp/dp-profile-20261104/early/timeline-512.sqlite`
- `.Codex/tmp/dp-profile-20261104/receipts/results/measurements.json`
- `docs/torad/glm53-dp-profile-20261104.json`
- `docs/torad/glm53-decode-pcie.md`
- `docs/torad/glm53-decode-roofline.json`

The exported profiles retain the source filename and complete SQLite SHA-256, but omit process environment, command lines, prompt text and absolute machine paths.

## Timing and byte definitions

### Ideal detail

The default one-session ideal ledger has **1,955 entries**: 1,850 target entries, 44 for proposal 1, 42 for proposal 2 and 19 for catch-up. Its leaf times sum to the parent PCIe-aware estimate, **9.0958282554 ms/cycle**, under the displayed ideal assumptions. This is not a measured or complete compute-cost prediction. The underlying pack includes all 1,412 exact GGUF tensor records; token embeddings remain row-lookups rather than a full table read.

- Memory budgets use the busiest sequence owner and balanced expert work.
- Routed expert reads use the expected distinct-expert union, not eight entirely new experts for every row.
- PCIe payload and synchronization assumptions are separate entries.
- A scatter that only changes tensor layout does not acquire a fictional transfer.
- Host staging writes and extra reads are distinct byte quantities.
- Controls change the existing Python model, not a separately reimplemented browser formula.
- The detail view currently supports fixed MTP2, cache-only catch-up and no overlap credit. The original estimator still provides its other scenarios.

Tensor byte budgets are not CUDA instruction-level predictions. Arithmetic/dequantization, most intermediate activations and software overhead remain omissions, rather than being assigned invented microsecond values.

### Recorded operations

`ggml/src/ggml-cuda/nvtx.cuh:31-71` emits operation names and logical byte estimates. They are **not measured DRAM or PCIe counters**. MoE annotations can count selected logical work differently from physical cache-reused reads.

A kernel is attributed through either:

1. its launch API's enclosing operation range on the same host thread/process; or
2. its graph node's creation range, following original-node relationships through graph clones.

Fused child markers do not receive separate invented execution times. Multiple kernels within one attributed invocation count its logical byte estimate once. Captured kernels without a supported link stay unattributed. Unnamed operations are retained as unnamed operations, not guessed from neighboring timestamps.

Operation invocation duration is its earliest attributed kernel start to latest end, including gaps. Kernel-duration sum and interval union are retained separately. Percentiles use nearest-rank selection. Means can be pooled by invocation count; percentiles of different shapes are not averaged.

### Transfers

Explicit CUDA copies have recorded bytes and durations, so payload GB/s can be calculated. Kernel-driven peer pushes, numeric collectives and mapped-host KV reads usually do not have explicit memcpy records. Their durations are retained, but their measured bandwidth remains **Not measured**.

Source-derived estimates and a selected link bandwidth can supply a reference, never missing hardware counters. CPU API time may overlap GPU execution. Summing all four GPUs' kernel time is not a critical-path latency.

## Import another run

Export an Nsight Systems report to SQLite using the profiler's `nsys export` command on a machine with the matching Nsight version. Then:

```sh
python3 scripts/glm53/decode_lab.py profile \
  --sqlite /path/to/capture.sqlite \
  --label run-name --active 1 --context 400000 \
  --runtime ACTUAL_COMMIT --slots 1 --depth-max 2 \
  --depth-policy adaptive --kv resident --scope request \
  --output .Codex/analysis/run-name/profile.json.gz
```

Set active count and context from the actual workload, not the server's slot reservation. Optional `--measurements /path/to/measurements.json` attaches the matching label's request timings.

Choose **Import profile** on the page and select that JSON or gzip file, then **Open raw evidence** to inspect it. Parsing happens in the browser. Nothing is uploaded. Invalid imports leave the previous evidence in place. Expanded files are limited to 80 MB; larger traces should be exported as shorter captures. Future-cycle slots remain empty after any raw-profile import.

Unspecified slot counts, depth policy and KV placement remain unknown in new exports. Supply the flags only from the actual run; do not set `--depth-policy fixed-2` unless established. The bundled historical captures use `--slots 40 --depth-max 2 --depth-policy adaptive --kv host --scope request`. Matching session/context/depth metadata does not establish cycle alignment. The page explicitly retains that distinction and does not display a cycle speedup ratio.

For genuine matched-cycle deltas, a future capture needs trustworthy proposal/verification/catch-up boundaries and actual row counts/acceptance for each cycle. Existing NVTX operation labels cannot establish every one of those facts.

## Rebuild the bundled data

```sh
python3 scripts/glm53/decode_lab.py ideal \
  --manifest .Codex/analysis/glm53-ceiling-20261104.RgdyZj/headers/manifest.json \
  --output docs/torad/decode-lab/data/ideal.json

python3 scripts/glm53/decode_lab.py profile \
  --sqlite .Codex/tmp/dp-profile-20261104/receipts/results/timeline-4k.sqlite \
  --label timeline-4k --active 1 --context 4096 --runtime 82ad56cb8 --slots 40 --depth-max 2 --depth-policy adaptive --kv host --scope request \
  --measurements .Codex/tmp/dp-profile-20261104/receipts/results/measurements.json \
  --output docs/torad/decode-lab/data/timeline-4k.json.gz

python3 scripts/glm53/decode_lab.py profile \
  --sqlite .Codex/tmp/dp-profile-20261104/early/timeline-512.sqlite \
  --label timeline-512 --active 1 --context 512 --runtime 82ad56cb8 --slots 40 --depth-max 2 --depth-policy adaptive --kv host --scope request \
  --measurements .Codex/tmp/dp-profile-20261104/receipts/results/measurements.json \
  --output docs/torad/decode-lab/data/timeline-512.json.gz
```

Gzip output is deterministic. The page has no frontend build step. Font requests use Google Fonts; all measurements and code remain local. Serif/sans/monospace fallbacks keep the interface usable without font-network access.

## Validation performed

- All 130 tests in the GLM Python tooling suite passed. After correcting unspecified export metadata to remain unknown, all nine new exporter/server/detail tests passed again.
- All seven browser-data tests passed in the original build; those helpers are unchanged.
- Six new visual-accounting tests passed, including generated overlapping event sequences and area-rounding properties. They failed on missing behavior before implementation.
- The revised browser journey passed: no default tables, GPU inspection, uneven ownership, layer drill-down, keyboard timeline navigation, all 96 numeric windows, model settings, evidence tables and responsive bounds. Separate import-safety and first-viewport phone-comparison journeys passed.
- The inspector-overlap and absent phone-comparison cases were reproduced in failing browser tests, then fixed. The desk inspector now reserves space instead of covering the recorded GPUs.
- Playwright exercised assumption changes, kernel search, raw-event inspection, copy rates, capture switching, valid/invalid local imports, untrusted labels, and 3840/1440/390 layout bounds.
- The browser test initially inherited a phone-sized context before navigation. Its setup now sets the desk viewport before loading the page; the full journey passed after that correction.
- Light and dark 3840 x 2050 captures, plus laptop and phone captures, are retained under `.Codex/analysis/glm53-decode-lab/screens/`.
- The earlier table-first page passed an independent visual check but was rejected by the user as too text-heavy and busy. It has been replaced with the visual-first comparison; the earlier scores do not qualify this redesign.
- The previous historical-comparison layout scored 9.6/10 in its fresh 4K light/dark visual check and 9.7/10 with the inspector. Those scores do not qualify the newer future-capture layout. Its captures remain under `.Codex/analysis/glm53-decode-visual/screens/r2-*.png`.
- The future-capture journey failed first on the old 40-session default, then passed with one session, four pending GPUs, four matching phases, 45 pending layers and no visible historical timeline or tables. The main journey, separate local-import safety journey and first-viewport phone journey passed. Raw imports leave the future side empty.
- The local-server freshness assertion failed before `Cache-Control: no-store` was added; all nine exporter/server/detail tests then passed. Browser tests now create a fresh context instead of inheriting older cached pages from the automation session.
- The page build made no inference-engine changes. The later CPU-only preparation adds opt-in host tracing, not inference-algorithm changes. No new GPU run, paid hardware, deployment or matched 40-session/400k measurement was performed.

## Checks

```sh
python3 -m unittest discover -s scripts/glm53 -p 'test_decode_lab.py'
node --test scripts/glm53/test_decode_lab_core.mjs scripts/glm53/test_decode_visual.mjs
```

The lasting browser test uses Playwright over an isolated agent-browser CDP connection:

```sh
npm install --prefix /tmp/glm53-browser-tools --no-audit --no-fund playwright-core
agent-browser --session glm53-lab-test --executable-path /usr/bin/google-chrome-stable open http://127.0.0.1:8768
agent-browser --session glm53-lab-test get cdp-url
PLAYWRIGHT_NODE_PATH=/tmp/glm53-browser-tools \
  DECODE_LAB_CDP=THE_PRINTED_WEBSOCKET_URL \
  node scripts/glm53/test_decode_lab_browser.cjs
PLAYWRIGHT_NODE_PATH=/tmp/glm53-browser-tools \
  DECODE_LAB_CDP=THE_PRINTED_WEBSOCKET_URL \
  node scripts/glm53/test_decode_lab_browser.cjs --import-only
PLAYWRIGHT_NODE_PATH=/tmp/glm53-browser-tools \
  DECODE_LAB_CDP=THE_PRINTED_WEBSOCKET_URL \
  node scripts/glm53/test_decode_lab_browser.cjs --future-only
PLAYWRIGHT_NODE_PATH=/tmp/glm53-browser-tools \
  DECODE_LAB_CDP=THE_PRINTED_WEBSOCKET_URL \
  node scripts/glm53/test_decode_lab_browser.cjs --phone-only
agent-browser --session glm53-lab-test close
```

Use an installed Chrome with its sandbox intact. No disabling of the browser sandbox is required.
