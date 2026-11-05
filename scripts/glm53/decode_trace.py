"""Whitelisted Nsight scheduling evidence. No model, CUDA runtime or GPU is used."""
from typing import Any
import re

TABLES = {
    'kernels': ('CUPTI_ACTIVITY_KIND_KERNEL', (
        'start', 'end', 'deviceId', 'contextId', 'streamId', 'globalPid', 'correlationId',
        'graphId', 'graphNodeId', 'demangledName', 'shortName', 'launchType',
        'gridX', 'gridY', 'gridZ', 'blockX', 'blockY', 'blockZ',
        'registersPerThread', 'staticSharedMemory', 'dynamicSharedMemory')),
    'copies': ('CUPTI_ACTIVITY_KIND_MEMCPY', (
        'start', 'end', 'deviceId', 'contextId', 'streamId', 'globalPid', 'correlationId',
        'graphNodeId', 'bytes', 'copyKind', 'srcKind', 'dstKind',
        'srcDeviceId', 'srcContextId', 'dstDeviceId', 'dstContextId', 'copyCount')),
    'memsets': ('CUPTI_ACTIVITY_KIND_MEMSET', (
        'start', 'end', 'deviceId', 'contextId', 'streamId', 'globalPid', 'correlationId',
        'graphNodeId', 'bytes', 'memKind')),
    'runtime': ('CUPTI_ACTIVITY_KIND_RUNTIME', (
        'start', 'end', 'globalTid', 'correlationId', 'nameId', 'returnValue')),
    'cuda_events': ('CUPTI_ACTIVITY_KIND_CUDA_EVENT', (
        'timestamp', 'deviceId', 'contextId', 'streamId', 'globalPid', 'correlationId',
        'eventId', 'eventSyncId')),
    'synchronizations': ('CUPTI_ACTIVITY_KIND_SYNCHRONIZATION', (
        'start', 'end', 'deviceId', 'contextId', 'streamId', 'globalPid', 'correlationId',
        'syncType', 'eventId', 'eventSyncId')),
    'graph_nodes': ('CUDA_GRAPH_NODE_EVENTS', (
        'start', 'end', 'globalTid', 'eventClass', 'nameId', 'graphNodeId', 'originalGraphNodeId')),
}


def capture_readiness(db, devices):
    """Reject startup/probe exports without model phase markers and kernels from the same process on every card."""
    expected = set(devices)
    if not expected or any(type(d) is not int or d < 0 for d in expected):
        raise ValueError('expected devices must be nonnegative device IDs')
    required = {
        'CUPTI_ACTIVITY_KIND_KERNEL': {'start', 'end', 'deviceId', 'globalPid'},
        'NVTX_EVENTS': {'start', 'end', 'globalTid', 'text', 'textId'},
    }
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    result = dict(usable=False, expected_devices=sorted(expected), kernel_counts={}, phase_markers=0,
                  missing_tables=sorted(set(required)-tables), missing_columns={},
                  note='Capture integrity only; does not qualify model correctness, throughput or capacity.')
    for table, fields in required.items():
        if table in tables:
            available = {r[1] for r in db.execute('PRAGMA table_info("'+table+'")')}
            if fields-available:
                result['missing_columns'][table] = sorted(fields-available)
    if result['missing_tables'] or result['missing_columns']:
        return result
    strings = dict(db.execute('SELECT id,value FROM StringIds')) if 'StringIds' in tables else {}
    processes = {}
    for start, end, tid, label, label_id in db.execute('SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS'):
        fields = parse_marker(label or strings.get(label_id, ''))
        if fields and fields['stage'] in ('proposal', 'target_eval', 'catchup') and tid is not None and start is not None:
            result['phase_markers'] += 1
            pid = (int(tid) >> 24) << 24
            processes[pid] = min(start, processes.get(pid, start))
    # Startup work and separate preflight processes cannot supply missing decode devices.
    # GPU work may finish after a host scope, so this is only a lower bound, not phase attribution.
    by_process = {}
    for pid, first_phase in processes.items():
        by_process[pid] = dict(db.execute('SELECT deviceId,COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL '
                                         'WHERE globalPid=? AND start>=? AND end>start GROUP BY deviceId',
                                         (pid, first_phase)))
    result['processes'] = {str(pid): {str(d): n for d, n in counts.items()} for pid, counts in by_process.items()}
    for device in expected:
        result['kernel_counts'][str(device)] = sum(counts.get(device, 0) for counts in by_process.values())
    result['usable'] = any(expected <= set(counts) for counts in by_process.values())
    return result


def export_dependencies(db, origin, strings):
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    result: dict[str, Any] = dict(schema='glm53-cuda-trace-v1', clock='relative-nanoseconds',
                  origin_ns=origin, missing_tables=[], missing_columns={})
    for key, (table, wanted) in TABLES.items():
        result[key] = []
        if table not in tables:
            result['missing_tables'].append(table)
            continue
        available = {r[1] for r in db.execute('PRAGMA table_info("' + table + '")')}
        selected = [field for field in wanted if field in available]
        result['missing_columns'][key] = [field for field in wanted if field not in available]
        clock = 'timestamp' if 'timestamp' in selected else 'start' if 'start' in selected else None
        query = 'SELECT ' + ','.join('"' + f + '"' for f in selected) + ' FROM "' + table + '"'
        if clock:
            query += ' ORDER BY "' + clock + '"'
        for row in db.execute(query):
            item = {field: row[field] if field in selected else None for field in wanted}
            for field in ('start', 'end', 'timestamp'):
                if item.get(field) is not None:
                    item[field] -= origin
            if key == 'kernels':
                item['name'] = strings.get(item['demangledName'], strings.get(item['shortName'], 'Unknown kernel'))
            elif 'nameId' in item:
                item['name'] = strings.get(item['nameId'], 'Unknown CUDA event')
            result[key].append(item)
    markers, evaluations, coverage = semantic_evaluations(db, tables, strings, origin, result)
    result.update(semantic_markers=markers, evaluations=evaluations, semantic_coverage=coverage)
    result['identity_notes'] = [
        'Correlation IDs require process identity; event IDs require device/context identity.',
        'A CUDA graph node ID is not a decode cycle or phase ID.',
        'Host event timestamps are not GPU completion timestamps.',
        'Synchronization records alone do not reveal intra-kernel collective arrival skew.',
        'Graph dependency edges and semantic cycle/workload records are not inferred.',
    ]
    return result


MARKER_FIELDS = {'iteration', 'proposal', 'head', 'rows', 'offset', 'sequence',
                 'prompt_rows', 'position_min', 'position_max', 'drafted', 'accepted',
                 'verified_rows', 'rollback_rows', 'replay'}
MARKER_STAGES = {'iteration', 'target_eval', 'proposal', 'catchup', 'batch',
                 'accept_candidate', 'accept'}


def parse_marker(label):
    if not isinstance(label,str) or not label.startswith('glm53.v1 '):
        return None
    fields = {}
    for part in label[len('glm53.v1 '):].split():
        key, separator, value = part.partition('=')
        if not separator or key in fields:
            return None
        if key == 'stage':
            if value not in MARKER_STAGES:
                return None
            fields[key] = value
        elif key in MARKER_FIELDS and re.fullmatch(r'-?\d+',value):
            fields[key] = int(value)
        else:
            # Unknown/free text is deliberately not exported (including prompts).
            return None
    return fields if 'stage' in fields else None


def semantic_evaluations(db, tables, strings, origin, trace):
    markers = []
    if 'NVTX_EVENTS' in tables:
        available = {r[1] for r in db.execute('PRAGMA table_info(NVTX_EVENTS)')}
        if {'start','end','globalTid','text','textId'} <= available:
            for row in db.execute('SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS ORDER BY start'):
                fields = parse_marker(row['text'] or strings.get(row['textId'],''))
                if fields:
                    markers.append(dict(index=len(markers), start=row['start']-origin,
                                        end=row['end']-origin if row['end'] is not None else None,
                                        globalTid=row['globalTid'], **fields))

    def containing(items, tid, time):
        matches = [m for m in items if m['globalTid']==tid and m['end'] is not None
                   and m['start'] <= time <= m['end']]
        return min(matches,key=lambda m:m['end']-m['start']) if matches else None

    iterations = [m for m in markers if m['stage']=='iteration']
    phases = [m for m in markers if m['stage'] in ('proposal','target_eval','catchup') and m['end'] is not None]
    evaluations = []
    by_marker = {}
    for marker in phases:
        parent = containing(iterations,marker['globalTid'],marker['start'])
        item = dict(marker_index=marker['index'], stage=marker['stage'],
                    iteration=marker.get('iteration',parent.get('iteration') if parent else None),
                    proposal=marker.get('proposal'), rows=marker.get('rows'),
                    host_start_ns=marker['start'],host_end_ns=marker['end'],
                    kernel_indices=[],copy_indices=[])
        evaluations.append(item);by_marker[marker['index']]=item
    launch_phase = {}
    for call in trace['runtime']:
        if call.get('globalTid') is None or call.get('start') is None:
            continue
        marker = containing(phases,call['globalTid'],call['start'])
        if marker is not None:
            pid = (int(call['globalTid']) >> 24) << 24
            launch_phase[(pid,call['correlationId'])] = by_marker[marker['index']]
    assigned = {'kernels':0,'copies':0}
    for kind, field in [('kernels','kernel_indices'),('copies','copy_indices')]:
        for index,event in enumerate(trace[kind]):
            match = launch_phase.get((event.get('globalPid'),event.get('correlationId')))
            if match is not None:
                match[field].append(index);assigned[kind]+=1
    coverage = dict(assigned_kernels=assigned['kernels'],unassigned_kernels=len(trace['kernels'])-assigned['kernels'],
                    assigned_copies=assigned['copies'],unassigned_copies=len(trace['copies'])-assigned['copies'])
    return markers,evaluations,coverage


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    import sqlite3

    parser = argparse.ArgumentParser(description='Check an exported Nsight SQLite capture locally; no GPU required.')
    parser.add_argument('sqlite', type=Path)
    parser.add_argument('--devices', nargs='+', type=int, required=True)
    args = parser.parse_args()
    with sqlite3.connect(args.sqlite.resolve().as_uri()+'?mode=ro', uri=True) as db:
        result = capture_readiness(db, args.devices)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result['usable'] else 1)
