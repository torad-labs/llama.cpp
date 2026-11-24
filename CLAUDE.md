IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.
This codebase is yours: what it's for and how it's built are your calls, and you'll keep building on it.

<!-- campaign:start -->
## Campaign

This repo uses the new `campaign` CLI (`campaign help`) configured in `.campaign/settings.toml` (inspect with
`campaign setup --show --json`).

- **Unattended worktree campaigns (default for features, fixes, integrations):** write a concise Markdown goal
  file with executable checks and launch `campaign hand <name> <goal.md>`. Track with `campaign status [<name> | --all]`
  and `campaign queue`, reply to prompts with `campaign origin reply <name> <ask-id> "<answer>"`, pause/resume with
  `campaign hold` / `campaign resume`, and tear down superseded runs with `campaign cancel <name> "<reason>" --force`.
- **Interactive Live Mode:** follow the `campaign` skill (`blocks/live.md` and `references/host-routing.md`). Keep
  the coordinator thin, enforce every allowlist/effort rule in `campaign setup --show --json`, and archive finished
  child sessions once collected.
- **Cross-lab verification bar (critical & blocker only):** every non-trivial change is verified by a fresh
  cross-lab verifier from the kernel allowlist below. Re-loop to the builder ONLY for critical or blocker bugs
  (wrong math/numerical divergence, memory corruption/OOB/crash/deadlock/real race, build break, GLM 5.3 regression,
  or broken core feature contract) - log minor concerns/nits without blocking merge.
- **Remote builds & GPU tests:** `[remote]` in `.campaign/settings.toml` routes builds/tests to CloudRift RTX 5090
  via `on-box` (`buildbox`). Always use a dedicated build directory per checkout and per-command GPU locks (`flock`).
- **Legacy ledger:** `.campaign/ledger.toml` holds historical row tracking from the previous workflow and is reconciled
  directly as rows land (the old `campaign next`/`claim`/`done`/`gate`/`feature` subcommands are retired).
<!-- campaign:end -->

## Changing shared code (Marcos, 2026-11-22)

The goal is throughput and better engine management, not upstream compatibility. llama.cpp keeps its decode
paths generic so every model and backend fits; this fork does not owe that.

- Change anything the work needs: drop compatibility machinery, restructure how decode and the engine are run,
  and write code made specifically for Inkling 2 Small where that makes it serve better.
- The primary serving target is Inkling 2 Small. Inkling 2 Mini is served too and has its own lane; a kernel
  or path shared by both architectures serves both, and the row note says which targets a change serves.
- Two limits: GLM 5.3 serving must keep working, and other models must not be completely wrecked.
- Prefer a model-specific path or a `..._LEGACY` switch when the shared version would hurt another served
  model; when the change helps everyone and GLM stays green, land it shared without ceremony. Bit-identical
  is the strongest proof when it comes for free, but a measured divergence is allowed when it is worth it.
- Say what you changed, and the measurement behind it, in the commit message or row note.

## Boxes and GPUs (Marcos, 2026-11-22)

- All builds, tests and dev runs go to the rented CloudRift box(s) with the RTX 5090s (`on-box`, ssh host
  `buildbox`). Rent another 5090 if the work needs one. Nothing new runs on this machine's local cards.
- The four-card RTX PRO 6000 Inkling box tests a batch of changes after they are made: one test and profile
  window per batch on that box, then stop the box, and every seat iterates against its recorded numbers from
  the 5090s. Profile once per window; do not re-profile piecemeal.
- Never touch or stop the other RTX PRO 6000 box: it serves GLM and Marcos works through it.

## Host-side tensor memory

- Host-side code (tests, tools, loaders, fixture writers) never holds a full f32 copy of a large tensor.
  Generate, convert and upload in chunks of whole rows or blocks, so peak host memory stays near the
  tensor's stored size plus one chunk. On 2026-11-17 the MUL_MAT_ID case in test-backend-ops at 384 experts
  built its whole 384x2304x4608 weight tensor as one f32 array, about 16 GB, before quantizing it, and hung
  for 22 minutes under memory pressure; the same case in chunks of about 16M values peaks near its quantized
  size, about 2.3 GB.

## Models allowed for kernel work (Marcos, 2026-11-24)

Only these models are allowed to build, debug, or review kernel work (CUDA, PTX, TMA, attention, MoE, quantize, allreduce, backend ops):

- Gemini: Carbon (`claude-antigravity--gem-4-carbon`), Neon (`claude-antigravity--gem-4-neon`) - use Neon for difficult kernel tasks
- GPT: Astra 6.2 (`claude-oai--6.2-astra`) or Sol 6.2 (`claude-oai--6.2-sol`)
- Claude: Opus 5.5 (`claude-opus-5-5`)
- Muse: Muse Wonder Full 1.0 (`claude-muse--muse-wonder-full-1.0`)

NO OTHER MODEL IS EVER ALLOWED TO WORK ON KERNELS (never Fable 5.2, never Grok 4.7, never Sonnet/Haiku/Flash/etc.).
