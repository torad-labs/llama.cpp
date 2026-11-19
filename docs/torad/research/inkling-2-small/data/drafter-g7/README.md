# Drafter round g7: the stock draft tuned on Inkling's own continuations

2026-11-19, on a rented box of 4x RTX PRO 6000. Target: Inkling 2 Small abliterated, Q3_K_M
(`benthecarman/Inkling-2-Small-abliterated` at `e4853745`). Engine `7b262d27`, launcher profile `tensor-4-draft`:
23 slots, the draft offers 3 tokens a round, the target keeps a drafted token when it is its own first choice.

## What was trained

The stock draft (5 layers, 1711.7M parameters, BF16) for 398 steps, two passes over 42,580 anchors in 73 requests.
A request is a public Claude Code session as the prompt; Inkling wrote the continuations, and its hidden states and
the 64 most likely tokens of each row were recorded. Four learning rates ran at once, one a card, 10 minutes each,
and each was scored on requests kept out of training. Every step is in `train-<rate>.txt` and in MLflow
(experiment `inkling2-drafter`, https://mlflow.torad.ai/#/experiments/21).

| learning rate | loss, first 20 steps to last 20 | accepted a round at anchors evenly apart (2.234 before, 2.419 at most) | kept a round over the server's own rounds (1.907 before, 2.178 at most) |
|---|---|---|---|
| 1e-5 | 1.052 to 0.870 | 2.258 | 1.920 |
| 3e-5 | 0.943 to 0.892 | 2.213 | 1.863 |
| 1e-4 | 0.945 to 1.056 | 2.104 | 1.704 |
| 3e-4 | 1.255 to 1.369 | 1.815 | 1.310 |

1e-5 was kept (`draft-tuned.sha256`). Its cross-entropy against the target on the requests kept out fell from 1.174
to 0.948, and the rounds it would keep rose by 0.7%. Anything faster than 1e-5 made the draft worse within two passes.

## Served: 23 sessions of 18K to 200K tokens, 256 tokens a reply

| | no draft | stock draft | tuned draft |
|---|---|---|---|
| tokens/s a session, 23 at once | 12.4 (120.3 in all) | 12.1 (117.3 in all, least 4.2) | 12.0 (120.1 in all, least 8.1) |
| 8 at once | 12.3 | 15.4 | 15.1 |
| 4 at once | 12.2 | 23.0 | 22.1 |
| 1 at once | 24.3 | 41.5 | 42.6 |
| kept of offered, the four levels together | | 13209 of 21789, 60.6% (1.82 a round) | 12845 of 21030, 61.1% (1.83 a round) |

Sources: `bench-none.txt`, `bench-stock.txt`, `bench-tuned.txt`; `bench-stock-d7.txt` is the stock draft offering 7.

## What it says

- The served numbers agree with the offline ones: this round moved acceptance by under 1%, and speed not at all.
- At 23 sessions a draft of this quality buys nothing over no draft. A drafted round there takes about 240 ms
  against 81 ms for a plain one, so 1.8 kept tokens a round only breaks even. At one session a drafted round takes
  67 ms against 41 ms, and the draft is worth 1.7x.
- Under the first-choice check the most a draft can keep at depth 3 is 2.18 a round, so training alone can add about
  12% at 23 sessions. The larger levers are in the engine: what a drafted round costs at 23 sessions, and a check
  that keeps a drafted token with the chance the target gives it. Walked offline, the tuned draft would keep 2.18 a
  round at depth 3 under that check against 1.92 under the first-choice one, and 3.46 against 2.95 at depth 7. The
  stock draft was not walked under it.
- 42,580 anchors is a small set for 1.7 billion parameters. The same recording holds 299,440 more tokens in the
  model turns of the prompts, written by the sessions' own model, with Inkling's state at each. The trainer can
  train on those too (`--prompt-anchors`); that round was not run.

## What comes next

The founder decided on 2026-11-19 that Inkling 2 Small designs the pipeline that trains its own drafter, and that a
fresh session builds from its plan. The scenario it starts from and an index of every record are in
`../../drafter-plan/`. The other records of the day are in `../drafter-20261119/`, and the kit that made them all is
`scripts/inkling2/drafter/`.
