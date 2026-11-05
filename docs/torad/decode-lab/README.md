# Decode comparison

A local comparison page for GLM-5.3 Flash on four RTX PRO 6000 Blackwell Server Edition GPUs.

## Open it

From the repository root:

```sh
python3 scripts/glm53/decode_lab.py serve --port 8768
```

Open **http://127.0.0.1:8768**. The server binds only to loopback, serves this analysis folder, and recalculates the ideal ledger using the existing Python estimator. Stop it with Ctrl-C. No inference server, GPU or external service is required.

The default comparison is **40 active sessions at 400k context** versus the recorded **one-session 4096-context** capture. These are intentionally labeled different workloads. This page does not manufacture a speedup by dividing their totals.

## What can be inspected

- **Layer budgets:** all 45 target layers, output/host boundaries, both draft proposals and the hypothetical cache-only catch-up. Open any group to inspect its substeps.
- **Substeps:** exact tensor read sets, expected routed-expert reuse, index scans, selected latent rows, KDA matrix/convolution reads and speculative outputs, individual gather/reduction payloads, host-KV stages, hidden-state handoffs and explicit latency budgets.
- **Operations:** named NVTX operations, shape-specific logical read/write estimates, GPU invocation means/percentiles and their underlying kernel events.
- **Kernels:** exact signature and GPU, invocation count, mean/P50/P95/max, summed duration, and every individual captured kernel event.
- **Memory copies:** actual CUDA copy payload and duration, a byte-matched reference at the selected directional link rate, measured payload GB/s and every copy event.
- **CUDA APIs:** host call count and duration, including graph creation/instantiation and synchronization. These are not added to GPU totals.
- **Activity:** per-GPU kernel occupancy over 96 equal capture windows, with hover/focus details and an accessible data table.

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

The default ideal ledger has **1,955 entries**. Its leaf times sum to the parent PCIe-aware estimate, 30.5645796776 ms/cycle in the default independent-routing scenario. The underlying pack includes all 1,412 exact GGUF tensor records; token embeddings remain row-lookups rather than a full table read.

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
  --label run-name --active 40 --context 400000 \
  --runtime ACTUAL_COMMIT --slots 40 --depth-max 2 \
  --depth-policy fixed-2 --kv host --scope request \
  --output /tmp/run-name.json.gz
```

Set active count and context from the actual workload, not the server's slot reservation. Optional `--measurements /path/to/measurements.json` attaches the matching label's request timings.

Choose **Import profile** on the page and select that JSON or gzip file. Parsing happens in the browser. Nothing is uploaded. Invalid imports leave the previous evidence in place. Expanded files are limited to 80 MB; larger traces should be exported as shorter captures.

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
- All seven browser-data tests passed.
- Playwright exercised assumption changes, kernel search, raw-event inspection, copy rates, capture switching, valid/invalid local imports, untrusted labels, and 3840/1440/390 layout bounds.
- The browser test initially inherited a phone-sized context before navigation. Its setup now sets the desk viewport before loading the page; the full journey passed after that correction.
- Light and dark 3840 x 2050 captures, plus laptop and phone captures, are retained under `.Codex/analysis/glm53-decode-lab/screens/`.
- The fresh final visual judge scored the owner-screen light/dark captures **8.5/10 each**, with every criterion at least 7: the binding desk-screen check passes. The phone scored 8.8; the laptop scored 7.9 with a readability advisory. The judge also could not see the below-fold pagination: the implementation bounds lists to 24 rows per page. No third polishing round was run.
- No inference-engine change, new GPU run, paid hardware, deployment or matched 40-session/400k measurement was performed.

## Checks

```sh
python3 -m unittest discover -s scripts/glm53 -p 'test_decode_lab.py'
node --test scripts/glm53/test_decode_lab_core.mjs
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
agent-browser --session glm53-lab-test close
```

Use an installed Chrome with its sandbox intact. No disabling of the browser sandbox is required.
