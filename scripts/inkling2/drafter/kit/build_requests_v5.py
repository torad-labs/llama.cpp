#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyarrow>=15"]
# ///
"""Build the request set for the Inkling 2 draft-head recording.

Every request is an Anthropic Messages body (system, tools, messages), the shape Claude Code sends to
the server's /v1/messages route, so the recording tool can render and tokenize it with the server's
own code. Contexts come from public Claude Code session datasets; the assistant text in them is
context only, because Inkling generates its own continuation at every assistant turn.

Sources (all public, fetched anonymously into a Hugging Face cache by fetch_sources.py):
  raw Claude Code session logs   choucsan/mimo-claude-code-traces-1k, armand0e/claude-fable-5-claude-code,
                                 AlinCiocan/fable-5-claude-code-traces, trace-commons/agent-traces,
                                 crispwisp/wisp-claude-code-sessions
  Claude Code 2.1.258 requests   liangzhidanta/claude-code-glm53-swesmith-trajectories (also the source
                                 of the real system prompt and the 25 real tool schemas)

Usage: build_requests_v5.py --hub <hf-cache/hub> --out <empty or existing data folder>
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import sys

RAW_SOURCES = [
    # name, hub folder, repository
    ("mimo", "datasets--choucsan--mimo-claude-code-traces-1k", "choucsan/mimo-claude-code-traces-1k"),
    ("armand0e-fable", "datasets--armand0e--claude-fable-5-claude-code", "armand0e/claude-fable-5-claude-code"),
    ("alin-fable", "datasets--AlinCiocan--fable-5-claude-code-traces", "AlinCiocan/fable-5-claude-code-traces"),
    ("trace-commons", "datasets--trace-commons--agent-traces", "trace-commons/agent-traces"),
    ("crispwisp", "datasets--crispwisp--wisp-claude-code-sessions", "crispwisp/wisp-claude-code-sessions"),
]
GLM_FOLDER = "datasets--liangzhidanta--claude-code-glm53-swesmith-trajectories"
GLM_REPOSITORY = "liangzhidanta/claude-code-glm53-swesmith-trajectories"
GLM_FILE = "data/sft/teacher_v2_candidates.jsonl"
ENVIRONMENT_MARK = "\n# Environment\n"
HELDOUT_SHARE = 0.08
HELDOUT_CEILING = 0.13
SPLIT_SALT = "inkling2-dspark-requests-v5:"


def snapshot(hub: str, folder: str) -> str:
    found = sorted(glob.glob(os.path.join(hub, folder, "snapshots", "*")))
    if len(found) != 1:
        raise SystemExit("expected one snapshot of %s under %s, found %d" % (folder, hub, len(found)))
    return found[0]


def split_order(key: str) -> str:
    return hashlib.sha256((SPLIT_SALT + key).encode()).hexdigest()


def heldout_groups(requests: list[dict]) -> set[str]:
    """Per source, the split groups held out: groups are taken in the order of their hash until they
    hold HELDOUT_SHARE of the source's session text, passing over a group that would take the share
    past HELDOUT_CEILING. A plain hash threshold leaves a source of a few long sessions with none."""
    sizes: dict[str, dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
    for request in requests:
        sizes[request["source"]][request["split_key"]] += request["session_chars"]
    out: set[str] = set()
    for source in sorted(sizes):
        total = sum(sizes[source].values())
        held = 0
        for key in sorted(sizes[source], key=split_order):
            if held >= HELDOUT_SHARE * total:
                break
            if held + sizes[source][key] <= HELDOUT_CEILING * total:
                held += sizes[source][key]
                out.add(key)
    return out


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def omitted(block: dict) -> dict:
    """An image or document cannot reach the served pack (it is served without a projector), so a
    client's block of that kind is replaced by the note the server itself writes for blocks it drops."""
    source = block.get("source") if isinstance(block.get("source"), dict) else {}
    media = source.get("media_type") if isinstance(source.get("media_type"), str) else ""
    return text_block("[%s omitted%s]" % (block.get("type") or "block", ": " + media if media else ""))


# ---------------------------------------------------------------------------------------------------
# The real system prompt and tool schemas


def load_glm_rows(hub: str) -> tuple[list[dict], str]:
    root = snapshot(hub, GLM_FOLDER)
    rows = []
    with open(os.path.join(root, GLM_FILE), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows, os.path.basename(root)


def real_tools(rows: list[dict]) -> list[dict]:
    """The 25 tool schemas Claude Code 2.1.258 sent, as Anthropic tool definitions, in the order sent."""
    first = json.dumps(rows[0]["metadata"]["tools"], sort_keys=True)
    for row in rows:
        if json.dumps(row["metadata"]["tools"], sort_keys=True) != first:
            raise SystemExit("the tool schemas differ between rows of the Claude Code 2.1.258 set")
    out = []
    for tool in rows[0]["metadata"]["tools"]:
        fn = tool["function"]
        out.append({"name": fn["name"], "description": fn.get("description") or "",
                    "input_schema": fn.get("parameters") or {"type": "object", "properties": {}}})
    return out


def system_parts(rows: list[dict]) -> tuple[str, list[str], str]:
    """The system prompt up to its Environment section (the same in every row), the Environment lines
    that do not describe one machine, and the Context management section."""
    system = rows[0]["messages"][0]["content"]
    cut = system.index(ENVIRONMENT_MARK)
    prefix = system[:cut]
    for row in rows:
        if row["messages"][0]["content"][:cut] != prefix:
            raise SystemExit("the system prompt differs before its Environment section")
    rest = system[cut + len(ENVIRONMENT_MARK):]
    environment, _, after = rest.partition("\n\n# Context management\n")
    static = [line for line in environment.split("\n")
              if line.startswith(" - ") and not line.startswith((
                  " - Primary working directory:", " - Is a git repository:", " - Platform:", " - Shell:",
                  " - OS Version:", " - You are powered by the model"))]
    context = after.split("\n\n<total_tokens>")[0]
    return prefix, static, context


def session_system(prefix: str, static: list[str], context: str, facts: dict) -> str:
    """The system prompt for a session from a raw log: the real prompt, with the Environment lines the
    log can fill (its folder, whether it is a git repository, its platform, its model)."""
    lines = ["You have been invoked in the following environment: "]
    cwd = facts.get("cwd")
    if cwd:
        lines.append(" - Primary working directory: " + cwd)
        lines.append(" - Is a git repository: " + ("true" if facts.get("git") else "false"))
        if len(cwd) > 2 and cwd[1] == ":" and cwd[2] in "\\/":
            lines.append(" - Platform: win32")
        elif cwd.startswith("/Users/"):
            lines.append(" - Platform: darwin")
        elif cwd.startswith("/"):
            lines.append(" - Platform: linux")
    if facts.get("model"):
        lines.append(" - You are powered by the model %s." % facts["model"])
    lines += static
    return prefix + ENVIRONMENT_MARK + "\n".join(lines) + "\n\n# Context management\n" + context


def abbreviated_tools(hub: str) -> dict[str, dict]:
    """Schemas of tools outside the 25, as the trace-commons rows carry them (short descriptions)."""
    import pyarrow.parquet as pq

    root = snapshot(hub, "datasets--trace-commons--agent-traces")
    out: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(root, "data", "*.parquet"))):
        table = pq.read_table(path, columns=["session_id", "tools"]).to_pylist()
        for row in sorted(table, key=lambda r: r["session_id"] or ""):
            for tool in row["tools"] or []:
                tool = json.loads(tool) if isinstance(tool, str) else tool
                fn = tool.get("function") if isinstance(tool, dict) else None
                if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                    continue
                params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else None
                out.setdefault(fn["name"], {
                    "name": fn["name"], "description": fn.get("description") or "",
                    "input_schema": params or {"type": "object", "properties": {}}})
    return out


# ---------------------------------------------------------------------------------------------------
# Messages


def tool_result_content(content) -> list[dict] | str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(text_block(part))
        elif not isinstance(part, dict):
            continue
        elif part.get("type") == "text":
            if isinstance(part.get("text"), str):
                parts.append(text_block(part["text"]))
        elif part.get("type") == "tool_reference":
            # The server accepts a reference only to a tool the request declares as deferred; these
            # sessions' deferred tool lists are not in their logs, so the reference is kept as text.
            parts.append(text_block("[tool_reference: %s]" % part.get("tool_name", "")))
        else:
            parts.append(omitted(part))
    return parts


def user_blocks(content) -> list[dict]:
    if isinstance(content, str):
        return [text_block(content)] if content else []
    blocks = []
    for block in content if isinstance(content, list) else []:
        if isinstance(block, str):
            if block:
                blocks.append(text_block(block))
        elif not isinstance(block, dict):
            continue
        elif block.get("type") == "text":
            if isinstance(block.get("text"), str) and block["text"]:
                blocks.append(text_block(block["text"]))
        elif block.get("type") == "tool_result":
            if isinstance(block.get("tool_use_id"), str):
                blocks.append({"type": "tool_result", "tool_use_id": block["tool_use_id"],
                               "content": tool_result_content(block.get("content"))})
        else:
            blocks.append(omitted(block))
    return blocks


def assistant_blocks(content) -> list[dict]:
    if isinstance(content, str):
        return [text_block(content)] if content else []
    blocks = []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            if isinstance(block.get("text"), str) and block["text"]:
                blocks.append(text_block(block["text"]))
        elif kind == "thinking":
            # Logs keep the signature and often drop the text; only thinking with text is sent back.
            if isinstance(block.get("thinking"), str) and block["thinking"]:
                blocks.append({"type": "thinking", "thinking": block["thinking"]})
        elif kind == "tool_use":
            if isinstance(block.get("id"), str) and isinstance(block.get("name"), str) and block["name"]:
                args = block.get("input") if isinstance(block.get("input"), dict) else {}
                blocks.append({"type": "tool_use", "id": block["id"], "name": block["name"], "input": args})
    return blocks


def well_formed(messages: list[dict], spare: dict[str, dict] | None = None) -> list[dict]:
    """Merge neighbours of one role, keep only tool results that answer the turn before them (in the
    order of its calls, ahead of the user's text, as the client sends them), stop before a tool call
    that never got its result, and end on a user message: the request a client would send next, with
    the model's turn still to come. The spare results, by call id, are ones the caller found outside
    the messages; a call with no result among the messages takes its result from there."""
    spare = spare or {}
    merged: list[dict] = []
    for message in messages:
        if not message["content"]:
            continue
        if merged and merged[-1]["role"] == message["role"]:
            merged[-1]["content"] += message["content"]
        else:
            merged.append({"role": message["role"], "content": list(message["content"])})
    out: list[dict] = []
    calls: list[str] = []
    index = 0
    while index < len(merged):
        message = merged[index]
        index += 1
        if message["role"] == "assistant":
            calls = []
            for block in message["content"]:
                if block["type"] == "tool_use" and block["id"] not in calls:
                    calls.append(block["id"])
            if calls:
                if index == len(merged):
                    merged.append({"role": "user", "content": []})
                answer = merged[index]
                answered = {b["tool_use_id"] for b in answer["content"] if b["type"] == "tool_result"}
                missing = [c for c in calls if c not in answered]
                if any(c not in spare for c in missing):
                    break
                answer["content"] = [spare[c] for c in missing] + answer["content"]
            out.append(message)
        else:
            results: dict[str, dict] = {}
            rest = []
            for block in message["content"]:
                if block["type"] != "tool_result":
                    rest.append(block)
                elif block["tool_use_id"] in calls:
                    results.setdefault(block["tool_use_id"], block)
            kept = [results[c] for c in calls if c in results] + rest
            calls = []
            if kept:
                if out and out[-1]["role"] == "user":
                    out[-1]["content"] += kept
                else:
                    out.append({"role": "user", "content": kept})
    while out and out[-1]["role"] == "assistant":
        out.pop()
    # A window that opens on the model's turn has lost what came before it: start at the first user
    # message, without the results of calls that are no longer there.
    while out and out[0]["role"] == "assistant":
        out.pop(0)
        if out:
            out[0]["content"] = [b for b in out[0]["content"] if b["type"] != "tool_result"]
            if not out[0]["content"]:
                out.pop(0)
    return out


def measure(body: dict) -> dict:
    chars = len(body["system"]) + sum(len(json.dumps(t, ensure_ascii=False)) for t in body["tools"])
    session = 0
    turns = calls = 0
    for message in body["messages"]:
        if message["role"] == "assistant":
            turns += 1
        for block in message["content"]:
            if block["type"] == "tool_use":
                calls += 1
                session += len(block["name"]) + len(json.dumps(block["input"], ensure_ascii=False))
            elif block["type"] == "tool_result":
                content = block["content"]
                session += len(content) if isinstance(content, str) else sum(len(p["text"]) for p in content)
            else:
                session += len(block.get("text") or block.get("thinking") or "")
    return {"messages": len(body["messages"]), "assistant_turns": turns, "tool_calls": calls,
            "preamble_chars": chars, "session_chars": session}


# ---------------------------------------------------------------------------------------------------
# Raw session logs


def read_log(path: str) -> list[dict]:
    lines = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                lines.append(value)
    return lines


def is_turn(line: dict) -> bool:
    return line.get("type") in ("user", "assistant") and not line.get("isSidechain") and isinstance(line.get("message"), dict)


def log_segments(lines: list[dict]) -> list[tuple[list[dict], dict[str, dict]]]:
    """The conversation a log holds, one list of lines per context window: the log is cut where the
    client compacted, and in each part the lines from its last turn back to its start are followed by
    their parent links, which leaves out subagents' turns and branches the user rewound."""
    parts: list[list[dict]] = [[]]
    for line in lines:
        if line.get("type") == "system" and line.get("subtype") == "compact_boundary":
            parts.append([])
        else:
            parts[-1].append(line)
    segments = []
    for part in parts:
        by_uuid = {line["uuid"]: line for line in part if isinstance(line.get("uuid"), str)}
        turns = [line for line in part if is_turn(line)]
        if not turns:
            continue
        chain = []
        seen: set[int] = set()
        current = turns[-1]
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            chain.append(current)
            parent = current.get("parentUuid")
            current = by_uuid.get(parent) if isinstance(parent, str) else None
        chain.reverse()
        # A reply that calls several tools at once is logged as a tree: each result hangs off its own
        # call, so only one of them is on the way back from the last turn. The others are picked up
        # here by call id.
        on_chain = {id(line) for line in chain}
        spare: dict[str, dict] = {}
        for line in turns:
            if line["type"] == "user" and id(line) not in on_chain:
                for block in user_blocks(line["message"].get("content")):
                    if block["type"] == "tool_result":
                        spare.setdefault(block["tool_use_id"], block)
        segments.append(([line for line in chain if is_turn(line)], spare))
    return segments


def segment_messages(segment: list[dict], spare: dict[str, dict]) -> list[dict]:
    messages: list[dict] = []
    last_id = None
    for line in segment:
        message = line["message"]
        if line.get("isApiErrorMessage") or line.get("isVisibleInTranscriptOnly"):
            continue
        if line["type"] == "assistant":
            if message.get("model") == "<synthetic>":
                continue
            blocks = assistant_blocks(message.get("content"))
            if messages and messages[-1]["role"] == "assistant" and last_id is not None and message.get("id") == last_id:
                messages[-1]["content"] += blocks
            else:
                messages.append({"role": "assistant", "content": blocks})
            last_id = message.get("id")
        else:
            messages.append({"role": "user", "content": user_blocks(message.get("content"))})
            last_id = None
    return well_formed(messages, spare)


def raw_requests(hub: str, tools25: list[dict], extra: dict[str, dict], system: tuple, stats: dict) -> list[dict]:
    names25 = {tool["name"] for tool in tools25}
    out = []
    for name, folder, repository in RAW_SOURCES:
        root = snapshot(hub, folder)
        files = sorted(glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True))
        found = []
        for path in files:
            lines = read_log(path)
            relative = os.path.relpath(path, root)
            stem = os.path.splitext(relative)[0]
            for index, (segment, spare) in enumerate(log_segments(lines)):
                found.append((stem, index, segment, spare))
        # A resumed session copies its earlier lines into a new file. Keep the longest copy of each
        # line run, and put files that share lines in one split group.
        found.sort(key=lambda item: (-len(item[2]), item[0], item[1]))
        owner: dict[str, int] = {}
        group = list(range(len(found)))

        def root_of(i: int) -> int:
            while group[i] != i:
                group[i] = group[group[i]]
                i = group[i]
            return i

        kept = []
        for position, (stem, index, segment, spare) in enumerate(found):
            uuids = [line["uuid"] for line in segment if isinstance(line.get("uuid"), str)]
            shared = [owner[u] for u in uuids if u in owner]
            for other in shared:
                group[root_of(position)] = root_of(other)
            if uuids and len(shared) * 2 > len(uuids):
                stats[name]["duplicate_segments"] += 1
                continue
            for u in uuids:
                owner.setdefault(u, position)
            kept.append(position)
        # A file's context windows belong to one session, so they share a split group too.
        first_of_file: dict[str, int] = {}
        for position, (stem, index, segment, spare) in enumerate(found):
            if stem in first_of_file:
                group[root_of(position)] = root_of(first_of_file[stem])
            else:
                first_of_file[stem] = position
        group_key: dict[int, str] = {}
        for position, (stem, index, segment, spare) in enumerate(found):
            r = root_of(position)
            if r not in group_key or stem < group_key[r]:
                group_key[r] = stem
        for position in kept:
            stem, index, segment, spare = found[position]
            messages = segment_messages(segment, spare)
            stats[name]["segments"] += 1
            if not messages:
                stats[name]["empty"] += 1
                continue
            used = []
            for message in messages:
                for block in message["content"]:
                    if block["type"] == "tool_use" and block["name"] not in names25 and block["name"] not in used:
                        used.append(block["name"])
            declared = [extra[n] for n in sorted(used) if n in extra]
            for n in used:
                if n not in extra:
                    stats[name]["undeclared_tools"][n] += 1
            facts = {}
            for line in segment:
                if "cwd" not in facts and isinstance(line.get("cwd"), str) and line["cwd"]:
                    facts["cwd"] = line["cwd"]
                    facts["git"] = bool(line.get("gitBranch"))
                model = line["message"].get("model") if line["type"] == "assistant" else None
                if "model" not in facts and isinstance(model, str) and model and model != "<synthetic>":
                    facts["model"] = model
            body = {"system": session_system(system[0], system[1], system[2], facts), "tools": tools25 + declared, "messages": messages}
            key = "%s:%s" % (name, group_key[root_of(position)])
            out.append({"id": "%s:%s#%d" % (name, stem, index), "source": repository, "split_key": key,
                        "prefix": "cc25" if declared else "cc25-system", **measure(body), "body": body})
    return out


# ---------------------------------------------------------------------------------------------------
# The Claude Code 2.1.258 set


def glm_requests(rows: list[dict], tools25: list[dict], stats: dict) -> list[dict]:
    out = []
    for number, row in enumerate(rows):
        source = row["messages"]
        if source[0]["role"] != "system":
            raise SystemExit("row %d of the Claude Code 2.1.258 set does not start with its system prompt" % number)
        messages: list[dict] = []
        waiting: list[str] = []
        calls = 0
        for message in source[1:]:
            role = message["role"]
            if role == "assistant":
                blocks = []
                if message.get("reasoning_content"):
                    blocks.append({"type": "thinking", "thinking": message["reasoning_content"]})
                if message.get("content"):
                    blocks.append(text_block(message["content"]))
                waiting = []
                for call in message.get("tool_calls") or []:
                    fn = call["function"]
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        args = json.loads(args) if args.strip() else {}
                    if not isinstance(args, dict):
                        args = {}
                    identifier = "toolu_%05d_%05d" % (number, calls)
                    calls += 1
                    waiting.append(identifier)
                    blocks.append({"type": "tool_use", "id": identifier, "name": fn["name"], "input": args})
                messages.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                if waiting:
                    messages.append({"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": waiting.pop(0), "content": message.get("content") or ""}]})
            elif role == "user":
                waiting = []
                messages.append({"role": "user", "content": user_blocks(message.get("content"))})
            else:
                raise SystemExit("row %d has a message of role %r after its system prompt" % (number, role))
        messages = well_formed(messages)
        stats["glm53"]["segments"] += 1
        if not messages:
            stats["glm53"]["empty"] += 1
            continue
        body = {"system": source[0]["content"], "tools": tools25, "messages": messages}
        task = row["metadata"].get("task_id") or "row"
        out.append({"id": "glm53:%05d:%s" % (number, task), "source": GLM_REPOSITORY,
                    "split_key": "glm53:" + str(row["metadata"]["repo"]), "prefix": "cc25-system",
                    **measure(body), "body": body})
    return out


# ---------------------------------------------------------------------------------------------------


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub", required=True, help="the Hugging Face cache's hub folder holding the datasets")
    parser.add_argument("--out", required=True, help="folder to write requests-train.jsonl, requests-heldout.jsonl and SOURCES.md")
    args = parser.parse_args()

    stats: dict = collections.defaultdict(lambda: {
        "segments": 0, "duplicate_segments": 0, "empty": 0,
        "undeclared_tools": collections.Counter()})
    rows, glm_revision = load_glm_rows(args.hub)
    tools25 = real_tools(rows)
    system = system_parts(rows)
    extra = abbreviated_tools(args.hub)
    requests = raw_requests(args.hub, tools25, extra, system, stats) + glm_requests(rows, tools25, stats)

    identifiers = [r["id"] for r in requests]
    if len(set(identifiers)) != len(identifiers):
        raise SystemExit("two requests share an id")
    requests.sort(key=lambda r: r["id"])
    os.makedirs(args.out, exist_ok=True)
    paths = {False: os.path.join(args.out, "requests-train.jsonl"), True: os.path.join(args.out, "requests-heldout.jsonl")}
    table: dict = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.Counter()))
    keys = {False: set(), True: set()}
    held_keys = heldout_groups(requests)
    with open(paths[False], "w", encoding="utf-8") as train, open(paths[True], "w", encoding="utf-8") as held:
        for request in requests:
            is_heldout = request["split_key"] in held_keys
            keys[is_heldout].add(request["split_key"])
            (held if is_heldout else train).write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
            cell = table[request["source"]][is_heldout]
            cell["requests"] += 1
            for field in ("messages", "assistant_turns", "tool_calls", "session_chars"):
                cell[field] += request[field]
    if keys[False] & keys[True]:
        raise SystemExit("a split group is on both sides")

    revisions = {repository: os.path.basename(snapshot(args.hub, folder)) for _, folder, repository in RAW_SOURCES}
    revisions[GLM_REPOSITORY] = glm_revision
    lines = [
        "# Request set for the Inkling 2 draft-head recording",
        "",
        "Written by build_requests_v5.py. Every line of the two files is one request: an id, its source, the",
        "group it was split by, and `body`, an Anthropic Messages body (system, tools, messages) as Claude",
        "Code sends it. The model's turns in these bodies are context only: the recording lets Inkling",
        "generate its own continuation at the start of every model turn and at the end of the request.",
        "",
        "| Source | Revision | Train requests | Train model turns | Train session text (M chars) | Held-out requests | Held-out model turns | Held-out session text (M chars) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for repository in sorted(table):
        a, b = table[repository][False], table[repository][True]
        lines.append("| %s | %s | %d | %d | %.1f | %d | %d | %.1f |" % (
            repository, revisions[repository][:12], a["requests"], a["assistant_turns"], a["session_chars"] / 1e6,
            b["requests"], b["assistant_turns"], b["session_chars"] / 1e6))
    lines += [
        "",
        "The split is by session for the raw logs (files that share lines, and the context windows of one",
        "file, stay together) and by repository for the Claude Code 2.1.258 set: %d groups train, %d held out," % (
            len(keys[False]), len(keys[True])),
        "none on both sides. Per source, groups are held out in the order of sha256(\"%s\" + group)" % SPLIT_SALT,
        "until they hold %d%% of the source's session text, passing over a group that would take it past %d%%." % (
            round(HELDOUT_SHARE * 100), round(HELDOUT_CEILING * 100)),
        "",
        "## How a raw session log becomes requests",
        "",
        "- One request per context window: a log is cut where the client compacted, and each part is followed",
        "  from its last turn back to its start by parent links. Subagent turns and rewound branches are out.",
        "- Lines the client marks as API errors, transcript-only or synthetic are skipped. Attachments and",
        "  hook output are not replayed.",
        "- A model reply written as several lines is one assistant message. Thinking is kept where the log",
        "  kept its text. Images and documents become the note the server writes for a block it drops.",
        "  A tool_reference result part becomes text, because the logs do not hold the deferred tool list.",
        "- A request stops before a tool call that never got its result, and ends on a user message.",
        "- System prompt: the real Claude Code 2.1.258 prompt from the liangzhidanta set up to its Environment",
        "  section, then the Environment lines the log can fill (folder, git, platform, model) and the",
        "  Context management section. Tools: the 25 real schemas of that set, then, for other tools the",
        "  session calls, the short schemas the trace-commons rows carry, sorted by name.",
        "",
        "## Counts while building",
        "",
    ]
    for name in sorted(stats):
        s = stats[name]
        undeclared = ", ".join("%s (%d)" % item for item in s["undeclared_tools"].most_common(12))
        lines.append("- %s: %d context windows, %d dropped as copies of a longer one, %d with nothing left to send%s" % (
            name, s["segments"], s["duplicate_segments"], s["empty"],
            "; tools called but not declared (no public schema): " + undeclared if undeclared else ""))
    lines += ["", "## Files", ""]
    for is_heldout in (False, True):
        lines.append("- %s sha256 %s" % (os.path.basename(paths[is_heldout]), sha256_of(paths[is_heldout])))
    with open(os.path.join(args.out, "SOURCES.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
