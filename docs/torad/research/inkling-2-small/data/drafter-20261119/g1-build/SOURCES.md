# Request set for the Inkling 2 draft-head recording

Written by build_requests_v5.py. Every line of the two files is one request: an id, its source, the
group it was split by, and `body`, an Anthropic Messages body (system, tools, messages) as Claude
Code sends it. The model's turns in these bodies are context only: the recording lets Inkling
generate its own continuation at the start of every model turn and at the end of the request.

| Source | Revision | Train requests | Train model turns | Train session text (M chars) | Held-out requests | Held-out model turns | Held-out session text (M chars) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AlinCiocan/fable-5-claude-code-traces | e33ebbca230a | 18 | 1068 | 1.7 | 1 | 129 | 0.2 |
| armand0e/claude-fable-5-claude-code | 7388ef96f317 | 48 | 3462 | 8.5 | 4 | 325 | 0.9 |
| choucsan/mimo-claude-code-traces-1k | 39cc3fc3ed60 | 939 | 2998 | 11.5 | 78 | 261 | 1.0 |
| crispwisp/wisp-claude-code-sessions | c2c90b591743 | 39 | 1719 | 2.4 | 25 | 136 | 0.2 |
| liangzhidanta/claude-code-glm53-swesmith-trajectories | 73041e495806 | 917 | 9605 | 27.7 | 86 | 885 | 2.5 |
| trace-commons/agent-traces | 112ebd4d03ce | 34 | 3943 | 8.2 | 4 | 321 | 0.9 |

The split is by session for the raw logs (files that share lines, and the context windows of one
file, stay together) and by repository for the Claude Code 2.1.258 set: 1155 groups train, 118 held out,
none on both sides. Per source, groups are held out in the order of sha256("inkling2-dspark-requests-v5:" + group)
until they hold 8% of the source's session text, passing over a group that would take it past 13%.

## How a raw session log becomes requests

- One request per context window: a log is cut where the client compacted, and each part is followed
  from its last turn back to its start by parent links. Subagent turns and rewound branches are out.
- Lines the client marks as API errors, transcript-only or synthetic are skipped. Attachments and
  hook output are not replayed.
- A model reply written as several lines is one assistant message. Thinking is kept where the log
  kept its text. Images and documents become the note the server writes for a block it drops.
  A tool_reference result part becomes text, because the logs do not hold the deferred tool list.
- A request stops before a tool call that never got its result, and ends on a user message.
- System prompt: the real Claude Code 2.1.258 prompt from the liangzhidanta set up to its Environment
  section, then the Environment lines the log can fill (folder, git, platform, model) and the
  Context management section. Tools: the 25 real schemas of that set, then, for other tools the
  session calls, the short schemas the trace-commons rows carry, sorted by name.

## Counts while building

- alin-fable: 19 context windows, 0 dropped as copies of a longer one, 0 with nothing left to send; tools called but not declared (no public schema): SendUserMessage (3), headroom_retrieve (1), mcp__headroom__headroom_retrieve (1), mcp__plugin_claude-mem_mcp-search__search (1), mcp__plugin_claude-mem_mcp-search__get_observations (1), mcp__perplexity-ask__perplexity_ask (1), mcp__plugin_claude-mem_mcp-search__save_memory (1)
- armand0e-fable: 52 context windows, 3 dropped as copies of a longer one, 0 with nothing left to send; tools called but not declared (no public schema): mcp__Claude_Preview__preview_start (3), mcp__Claude_Preview__preview_screenshot (3), mcp__Claude_Preview__preview_eval (3), mcp__Claude_Preview__preview_console_logs (3), SendUserFile (2), mcp__Claude_Preview__preview_stop (2), mcp__Claude_Preview__preview_click (1), mcp__Claude_Preview__preview_snapshot (1), mcp__Claude_Preview__preview_resize (1), mcp__huggingface__paper_search (1)
- crispwisp: 64 context windows, 2 dropped as copies of a longer one, 0 with nothing left to send
- glm53: 1003 context windows, 0 dropped as copies of a longer one, 0 with nothing left to send
- mimo: 1017 context windows, 0 dropped as copies of a longer one, 0 with nothing left to send
- trace-commons: 39 context windows, 0 dropped as copies of a longer one, 1 with nothing left to send

## Files

- requests-train.jsonl sha256 28e5513dbf91d90ded6ddfb7b0871de72876a7c499031b245c9372c366042c14
- requests-heldout.jsonl sha256 5df871cb74f420deb56a447b2fdf7df4282bdd4ca0eeed20c7bee068948724e8
