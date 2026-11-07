#!/usr/bin/env python3
"""Read-only NVTX/CUPTI round analysis for a GLM 5.3 Nsight SQLite export.

Run with Python isolation:
  python3 -I scripts/glm53/decode_report.py --db TRACE.sqlite --run-json run.json --launch-json launch.json --output report.txt
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Iterable

ITERATION_RE = re.compile(r"\biteration=(\d+)")
FIELD_RE = re.compile(r"([A-Za-z_]+)=([^ ]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="Nsight SQLite export")
    parser.add_argument("--run-json", type=Path, required=True, help="Sibling workload metadata")
    parser.add_argument("--launch-json", type=Path, required=True, help="Actual server launch record")
    parser.add_argument("--output", type=Path, required=True, help="Plain-text report to write")
    return parser.parse_args()


def sqlite_read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def union_ns(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return 0
    total = 0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def percentile(values: list[int], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def ms(value_ns: float) -> str:
    return f"{value_ns / 1_000_000:.3f}"


def us(value_ns: float) -> str:
    return f"{value_ns / 1_000:.1f} us"


def field_map(text: str) -> dict[str, str]:
    return dict(FIELD_RE.findall(text))


def geometry_text(geometry: tuple[int, ...]) -> str:
    gx, gy, gz, bx, by, bz, registers, static_smem, dynamic_smem = geometry
    return (
        f"grid {gx}x{gy}x{gz}; block {bx}x{by}x{bz}; "
        f"regs {registers}; smem {static_smem}+{dynamic_smem} B"
    )


def associate_launch(con, run, launch_path, markers):
    raw = launch_path.read_bytes()
    launch = json.loads(raw)
    notes = []
    if run.get("launch_sha256") is None:
        notes.append("launch digest: unavailable (legacy run has no link to the launch record)")
    elif run["launch_sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("launch digest does not match the workload")
    else:
        notes.append("launch digest: matched")
    binaries = [arg for arg in launch["argv"] if Path(arg).name == "llama-server"]
    if len(binaries) != 1:
        raise ValueError("launch must identify exactly one server executable")
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    marker_pids = {pid for _, _, _, pid in markers}
    if "PROCESSES" in tables:
        matching = {pid for pid, name in con.execute("SELECT globalPid,name FROM PROCESSES")
                    if name == binaries[0]} & marker_pids
        if len(matching) != 1:
            raise ValueError("trace process does not uniquely match the launched server")
        pid = matching.pop()
        notes.append("trace process: matched launched executable and NVTX process")
    else:
        if len(marker_pids) != 1:
            raise ValueError("trace process is ambiguous without process metadata")
        pid = marker_pids.pop()
        notes.append("trace process: unavailable executable identity (NVTX process only)")
    source = launch.get("source", "")
    notes.append("source recorded by launcher: " + (source if re.fullmatch(r"[0-9a-f]{40}", source) else "unavailable"))
    settings = launch.get("env", {})
    route = settings.get("GGML_CUDA_ALLREDUCE")
    if route not in ("internal", "internal-nccl-gather", "nccl"):
        route = "unavailable"
    notes.append("collective route recorded by launcher: " + route)
    notes.append("Source and route are launch declarations, not independently attested by the trace.")
    notes.append("association: process/time-correlated, not unique workload identity (no per-workload NVTX ID)")
    visible = settings.get("CUDA_VISIBLE_DEVICES", "")
    devices = list(range(len(visible.split(",")))) if visible else None
    return pid, notes, devices


def associate_time(con, run, rounds):
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "TARGET_INFO_SESSION_START_TIME" not in tables:
        return "trace time: unavailable (no session epoch)"
    epochs = list(con.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME"))
    if len(epochs) != 1 or epochs[0][0] is None:
        return "trace time: unavailable (ambiguous session epoch)"
    if "decode_started_epoch" in run and "decode_finished_epoch" in run:
        start, end = run["decode_started_epoch"], run["decode_finished_epoch"]
        scope = "measured decode"
    else:
        start, end = run.get("started_epoch"), run.get("finished_epoch")
        scope = "whole workload only (legacy run has no decode interval)"
    if start is None or end is None:
        return "trace time: unavailable (no workload interval)"
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("invalid workload clock interval")
    first = epochs[0][0] + rounds[0]["start"]
    last = epochs[0][0] + rounds[-1]["end"]
    if first < round(start*1_000_000_000) or last > round(end*1_000_000_000):
        raise ValueError("trace clock lies outside the workload interval")
    return "trace time: matched " + scope


def collective_kind(name):
    if name.startswith("ggml_cuda_ar_"):
        return "internal"
    if name.startswith(("ncclDevKernel_", "ncclKernel_")):
        return "native NCCL"
    return None


def main() -> None:
    args = parse_args()
    run = json.loads(args.run_json.read_text())

    active_sessions = int(run["active_sessions"])
    initial_tokens = run.get("initial_prompt_tokens")
    initial_text = (f"{initial_tokens} initial prompt tokens" if initial_tokens is not None
                    else f"initial prompt: unavailable; legacy reported context: {run.get('context_tokens', 'unavailable')}")
    effective = run.get("decode_prompt_tokens")
    if effective is not None:
        if len(effective) != active_sessions or any(type(n) is not int or n <= 0 for n in effective):
            raise ValueError("effective prompt lengths must cover every active session")
        effective_text = f"{min(effective)}..{max(effective)} tokens"
    else:
        effective_text = "unavailable (legacy metadata labels only the initial prompt)"
    warmup_text = "unavailable (legacy metadata)"
    if initial_tokens is not None:
        warmup = run.get("warmup_tokens_per_session")
        total_warmup = run.get("warmup_generated_tokens")
        if (type(initial_tokens) is not int or initial_tokens <= 0 or not isinstance(warmup, list)
                or len(warmup) != active_sessions or any(type(n) is not int or n != 16 for n in warmup)
                or type(total_warmup) is not int or total_warmup != sum(warmup)
                or effective != [initial_tokens+n for n in warmup]):
            raise ValueError("warmup evidence must account for every initial-to-decode prompt handoff")
        warmup_text = f"{min(warmup)}..{max(warmup)} tokens/session; {total_warmup} generated tokens"
    decode_tokens = int(run["decode_tokens"])
    draft_n = int(run["draft_n"])
    draft_n_accepted = int(run["draft_n_accepted"])
    if active_sessions <= 0 or run.get("passed") is False:
        raise ValueError("workload must have active sessions and must not have failed")

    con = sqlite_read_only(args.db)
    strings = dict(con.execute("SELECT id,value FROM StringIds"))
    markers = [(start, end, label or strings.get(label_id, ""), (tid >> 24) << 24)
               for start, end, tid, label, label_id in
               con.execute("SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS ORDER BY start")
               if tid is not None]
    iterations = [m for m in markers if m[2].startswith("glm53.v1 stage=iteration ")]
    pid, association, expected_devices = associate_launch(con, run, args.launch_json, iterations)
    markers = [m for m in markers if m[3] == pid]
    iteration_rows = []
    for start, end, text, _ in iterations:
        if _ != pid:
            continue
        match = ITERATION_RE.search(text)
        if match is None or start is None or end is None or end <= start:
            raise RuntimeError("Iteration marker is missing its ID or a complete positive interval")
        iteration_rows.append((int(match.group(1)), start, end))

    if len(iteration_rows) < 3:
        raise RuntimeError(f"Need at least three iteration markers, found {len(iteration_rows)}")

    # The requested invariant excludes the first and last observed marker ranges.
    selected_markers = iteration_rows[1:-1]
    rounds: list[dict[str, object]] = []
    for iteration, start, end in selected_markers:
        acceptance = []
        for marker_start, _, text, _pid in markers:
            if start <= marker_start < end and text.startswith("glm53.v1 stage=accept "):
                fields = field_map(text)
                accepted = int(fields["accepted"])
                if accepted < 0:
                    raise ValueError("negative accepted-token count")
                acceptance.append((int(fields["sequence"]), accepted))
        rounds.append(
            {
                "iteration": iteration,
                "start": start,
                "end": end,
                "duration": end - start,
                "acceptance": acceptance,
                "kernel_intervals": collections.defaultdict(list),
                "activity_intervals": collections.defaultdict(list),
                "all_activity_intervals": [],
                "signature_duration": collections.defaultdict(int),
                "signature_launches": collections.defaultdict(int),
                "collectives": collections.defaultdict(list),
            }
        )

    if any(len(round_["acceptance"]) != active_sessions for round_ in rounds):
        bad = [
            f"{round_['iteration']}={len(round_['acceptance'])}"
            for round_ in rounds
            if len(round_["acceptance"]) != active_sessions
        ]
        raise RuntimeError(
            "Selected rounds do not all have the requested concurrent-session "
            f"acceptance count ({active_sessions}): {', '.join(bad)}"
        )
    expected_sequences = set(range(active_sessions))
    if any({sequence for sequence, _ in round_["acceptance"]} != expected_sequences for round_ in rounds):
        raise RuntimeError("At least one selected round does not have exactly one acceptance marker per sequence")

    association.append(associate_time(con, run, rounds))
    devices = sorted(
        row[0]
        for row in con.execute(
            """
            SELECT DISTINCT deviceId
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE end > ? AND start < ? AND globalPid = ?
            ORDER BY deviceId
            """,
            (rounds[0]["start"], rounds[-1]["end"], pid),
        )
    )
    if not devices or (expected_devices is not None and devices != expected_devices):
        raise ValueError("trace devices do not match the launch")
    if expected_devices is None:
        association.append("expected devices: unavailable (report uses observed devices only)")
    kernel_columns = {r[1] for r in con.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    graph_column = next((c for c in ("graphNodeId", "graphId") if c in kernel_columns), None)
    graph_kernels = total_kernels = 0
    gpu_names = {
        device_id: name
        for device_id, name in con.execute(
            "SELECT id, name FROM TARGET_INFO_GPU WHERE id IN ({})".format(
                ",".join("?" for _ in devices)
            ),
            devices,
        )
    }

    # Query each NVTX range independently so all event intervals are clipped to
    # the actual marker range, not to a guessed cadence.
    for round_ in rounds:
        start = int(round_["start"])
        end = int(round_["end"])
        kernel_rows = con.execute(
            """
            SELECT
                k.start, k.end, k.deviceId, strings.value,
                k.gridX, k.gridY, k.gridZ,
                k.blockX, k.blockY, k.blockZ,
                k.registersPerThread, k.staticSharedMemory, k.dynamicSharedMemory
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            JOIN StringIds AS strings ON strings.id = k.shortName
            WHERE k.end > ? AND k.start < ? AND k.globalPid = ?
            ORDER BY k.start
            """,
            (start, end, pid),
        ).fetchall()
        if {row[2] for row in kernel_rows} != set(devices):
            raise ValueError("selected round lacks kernels on expected devices")
        total_kernels += len(kernel_rows)
        if graph_column:
            graph_kernels += con.execute(
                f"SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE end>? AND start<? AND globalPid=? AND {graph_column}>0",
                (start, end, pid)).fetchone()[0]
        for (
            event_start,
            event_end,
            device,
            name,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            registers,
            static_smem,
            dynamic_smem,
        ) in kernel_rows:
            clipped = (max(start, event_start), min(end, event_end))
            geometry = (
                grid_x,
                grid_y,
                grid_z,
                block_x,
                block_y,
                block_z,
                registers or 0,
                static_smem or 0,
                dynamic_smem or 0,
            )
            signature = (name, geometry)
            round_["kernel_intervals"][device].append(clipped)
            round_["activity_intervals"][device].append(clipped)
            round_["all_activity_intervals"].append(clipped)
            round_["signature_duration"][(signature, device)] += clipped[1] - clipped[0]
            round_["signature_launches"][(signature, device)] += 1
            if collective_kind(name):
                round_["collectives"][signature].append((event_start, event_end, device))

        for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
            for event_start, event_end, device in con.execute(
                f"""
                SELECT start, end, deviceId
                FROM {table}
                WHERE end > ? AND start < ? AND globalPid = ?
                """,
                (start, end, pid),
            ):
                clipped = (max(start, event_start), min(end, event_end))
                round_["activity_intervals"][device].append(clipped)
                round_["all_activity_intervals"].append(clipped)

    round_count = len(rounds)
    total_duration = sum(int(round_["duration"]) for round_ in rounds)
    # server-context.cpp emits accepted.size() - 1, then sends every ID,
    # including the verifier token (lines 4698-4704 and 4741).
    tokens_by_round = [
        sum(accepted + 1 for _, accepted in round_["acceptance"]) for round_ in rounds
    ]
    total_tokens = sum(tokens_by_round)
    mean_tokens = total_tokens / round_count
    aggregate_tok_s = total_tokens / (total_duration / 1_000_000_000)
    per_session_tok_s = aggregate_tok_s / active_sessions
    round_ms_50 = mean_tokens / (active_sessions * 50) * 1_000
    round_ms_70 = mean_tokens / (active_sessions * 70) * 1_000

    all_union = [union_ns(round_["all_activity_intervals"]) for round_ in rounds]
    kernel_union = [
        union_ns(
            interval
            for intervals in round_["kernel_intervals"].values()
            for interval in intervals
        )
        for round_ in rounds
    ]
    mean_round_ns = total_duration / round_count
    mean_all_union = sum(all_union) / round_count
    mean_kernel_union = sum(kernel_union) / round_count

    signature_duration: dict[tuple[object, object], dict[int, int]] = collections.defaultdict(
        lambda: collections.defaultdict(int)
    )
    signature_launches: dict[tuple[object, object], dict[int, int]] = collections.defaultdict(
        lambda: collections.defaultdict(int)
    )
    for round_ in rounds:
        for (signature, device), duration in round_["signature_duration"].items():
            signature_duration[signature][device] += duration
        for (signature, device), launches in round_["signature_launches"].items():
            signature_launches[signature][device] += launches

    top_signatures = sorted(
        signature_duration.items(), key=lambda item: sum(item[1].values()), reverse=True
    )[:8]

    collective_metrics: dict[str, dict[str, object]] = {}
    for round_ in rounds:
        for signature, events in round_["collectives"].items():
            name, geometry = signature
            metric = collective_metrics.setdefault(
                name,
                {
                    "geometries": set(),
                    "operations": [],
                    "kernel_events": 0,
                    "skipped_groups": 0,
                },
            )
            metric["geometries"].add(geometry)
            metric["kernel_events"] += len(events)
            per_device: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
            for event_start, event_end, device in events:
                per_device[device].append((event_start, event_end))
            counts = {device: len(per_device[device]) for device in devices}
            if set(per_device) != set(devices) or len(set(counts.values())) != 1:
                metric["skipped_groups"] += 1
                continue
            for intervals in per_device.values():
                intervals.sort()
            for index in range(next(iter(counts.values()))):
                operation = [per_device[device][index] for device in devices]
                starts = [event[0] for event in operation]
                ends = [event[1] for event in operation]
                metric["operations"].append(
                    (
                        max(starts) - min(starts),
                        max(ends) - max(starts),
                        max(ends) - min(starts),
                    )
                )

    lines = [
        f"GLM 5.3, {active_sessions}-session decode trace",
        "==================================",
        "",
        "Scope",
        "-----",
        f"- Nsight SQLite: {args.db}",
        f"- Workload metadata: {args.run_json}",
        f"- Workload: {active_sessions} active sessions; {initial_text}; effective prompt: {effective_text}; "
        f"{decode_tokens}-token generation; draft accepted {draft_n_accepted}/{draft_n}.",
        f"- warmup evidence: {warmup_text}.",
        f"- Observed NVTX iteration markers: {iteration_rows[0][0]}..{iteration_rows[-1][0]} "
        f"({len(iteration_rows)} ranges). Per request, first ({iteration_rows[0][0]}) and "
        f"last ({iteration_rows[-1][0]}) are excluded.",
        f"- Measured complete ranges: {rounds[0]['iteration']}..{rounds[-1]['iteration']} "
        f"({round_count} rounds). Every range has exactly {active_sessions} acceptance markers "
        f"for sequences 0..{active_sessions - 1}.",
        *("- " + note for note in association),
        "- graph-node kernels: " + (f"{graph_kernels}/{total_kernels}" if graph_column else "unavailable"),
        "- Graph identifiers show trace-visible captured kernels only; they do not prove full mixed-graph capture or its benefit.",
        "",
        "Delivered throughput",
        "--------------------",
        f"- {total_tokens} delivered tokens (accepted drafts plus one verifier token per sequence) in {ms(total_duration)} ms: "
        f"{aggregate_tok_s:.2f} aggregate tok/s, {per_session_tok_s:.2f} tok/s/session.",
        f"- Mean round: {ms(mean_round_ns)} ms; mean delivered: {mean_tokens:.2f} tokens/round "
        f"({min(tokens_by_round)}..{max(tokens_by_round)}).",
        "- Delivered tokens/round: "
        + ", ".join(
            f"{round_['iteration']}={tokens}"
            for round_, tokens in zip(rounds, tokens_by_round)
        )
        + ".",
        f"- Objective: 50..70 tok/s/session. Holding the observed delivered-token volume fixed "
        f"would require {round_ms_50:.2f}..{round_ms_70:.2f} ms/round.",
        "",
        "GPU timeline",
        "------------",
        "- GPU union means the union of CUPTI kernel, memcpy, and memset intervals, across all "
        f"{len(devices)} devices in the server process, clipped to each NVTX iteration range. It is elapsed activity, not the sum "
        "of device time.",
        f"- All-device activity union: {ms(mean_all_union)} ms/round "
        f"({mean_all_union / mean_round_ns * 100:.1f}%); idle: "
        f"{ms(mean_round_ns - mean_all_union)} ms/round.",
        f"- Kernel-only union: {ms(mean_kernel_union)} ms/round. Memcpy/memset extend the all-device "
        f"union by {us(mean_all_union - mean_kernel_union)} per round.",
        "- Activity and idle intervals are observations, not removable costs or a hardware ceiling.",
        "",
        "Per-device activity union",
        "-------------------------",
        "| device | model | active ms/round | idle ms/round |",
        "|---:|---|---:|---:|",
    ]
    for device in devices:
        device_union = [
            union_ns(round_["activity_intervals"][device]) for round_ in rounds
        ]
        mean_device_union = sum(device_union) / round_count
        lines.append(
            f"| {device} | {gpu_names.get(device, 'unknown')} | {ms(mean_device_union)} | "
            f"{ms(mean_round_ns - mean_device_union)} |"
        )

    lines.extend(
        [
            "",
            "Largest kernel signatures",
            "-------------------------",
            f"GPU-ms/round is summed execution time divided by the {round_count} rounds. It is useful "
            "attribution, not additive elapsed-time savings: kernels can overlap and the "
            "devices run concurrently.",
            "",
            f"| kernel | GPU-ms/round, device {'/'.join(map(str, devices))} | launches/GPU/round | geometry |",
            "|---|---:|---:|---|",
        ]
    )
    for signature, per_device_duration in top_signatures:
        name, geometry = signature
        per_device_ms = [
            per_device_duration.get(device, 0) / round_count / 1_000_000
            for device in devices
        ]
        launch_values = [
            signature_launches[signature].get(device, 0) / round_count
            for device in devices
        ]
        lines.append(
            f"| {name} | {' / '.join(f'{value:.3f}' for value in per_device_ms)} | "
            f"{' / '.join(f'{value:.0f}' for value in launch_values)} | {geometry_text(geometry)} |"
        )

    lines.extend(
        [
            "",
            "Collectives",
            "-----------",
            "- Includes internal ggml_cuda_ar_* and native ncclDevKernel_*/ncclKernel_* signatures. "
            "A native kernel name alone does not establish gather versus numeric reduction.",
            "- Groups are ordinal matches of equal per-device signature counts within each round, "
            "not proven logical operations. Multiple streams and NCCL protocol stages can break that correspondence. "
            "Skew is last start minus first start; post-last-start is last completion minus last start. "
            "These inferred envelopes are not additive optimization budgets.",
            "",
            "| collective kernel | transport | kernel events | ordinal groups | groups/round | start skew p50/p95 | "
            "post-last-start p50/p95 | unmatched signature-rounds |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, metric in sorted(collective_metrics.items()):
        operations = metric["operations"]
        arrival = [operation[0] for operation in operations]
        post_last = [operation[1] for operation in operations]
        lines.append(
            f"| {name} | {collective_kind(name)} | {metric['kernel_events']} | {len(operations)} | {len(operations) / round_count:.1f} | "
            f"{us(percentile(arrival, 0.50))} / {us(percentile(arrival, 0.95))} | "
            f"{us(percentile(post_last, 0.50))} / {us(percentile(post_last, 0.95))} | "
            f"{metric['skipped_groups']} |"
        )

    lines.extend([
        "",
        "Limits",
        "------",
        "This report reads an existing capture; it does not run a workload. Source and route declarations "
        "must not be confused with trace observations. Kernel durations overlap, and ordinal collective "
        "groups do not establish a critical path or a causal speedup.",
        "",
    ])
    con.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
