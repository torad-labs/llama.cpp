IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.
This codebase is yours: what it's for and how it's built are your calls, and you'll keep building on it.

<!-- campaign:start -->
## Campaign

Open work lives in `.campaign/ledger.toml`. Run `campaign help` for the verbs.

- Work on a row: `campaign next`, then `campaign claim <id> --as <your seat>`. Every pull request names the row
  it moves forward (or `upkeep`).
- A row is done only when `campaign done <id>` passes. It runs the row's checks as the base branch has them.
  Saying it is done, a test count or a commit message is not done.
- Checks change only in a commit of their own that touches nothing but `.campaign/` and judged paths, written by
  the lead. They judge once they reach the base branch.
  If a check is wrong, `campaign note` why and `campaign status <id> blocked`.
- New work starts as a row whose checks fail today (`campaign add` refuses checks that already pass).
- A migration row's check proves the old path is gone, not only that the new one exists.
- The reviewer runs `campaign gate --title "<pr title>"` on every pull request; it merges only when the gate passes.
- Nobody outside the seats approves or reviews anything. What a row needs judged, its checks judge, including
  review checks that run a fresh agent against the work.
- A feature is cut into items by its lead (`campaign feature`, `campaign add --feature`) and run by the tool:
  the committed items are the plan, and no session writes or reviews it. `campaign start <feature>` builds
  its items in the feature's worktree and has them reviewed, `campaign finish <feature>` merges it. An item of a
  feature is done only when its checks, its feature review and its code review pass in one round. A session given
  items builds them and ends; it never commits, closes an item or reviews it.
<!-- campaign:end -->

## Host-side tensor memory

- Host-side code (tests, tools, loaders, fixture writers) never holds a full f32 copy of a large tensor.
  Generate, convert and upload in chunks of whole rows or blocks, so peak host memory stays near the
  tensor's stored size plus one chunk. On 2026-11-17 the MUL_MAT_ID case in test-backend-ops at 384 experts
  built its whole 384x2304x4608 weight tensor as one f32 array, about 16 GB, before quantizing it, and hung
  for 22 minutes under memory pressure; the same case in chunks of about 16M values peaks near its quantized
  size, about 2.3 GB.
