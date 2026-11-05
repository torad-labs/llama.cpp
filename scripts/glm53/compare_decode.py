#!/usr/bin/env python3
"""CPU-only case preparation, dependency scheduling and logical-work comparison.

This tool never launches a server, runs CUDA, rents hardware or treats a synthetic
fixture as a measurement. Known durations define a feasible schedule, not a proof
that an optimal implementation can attain it.
"""
import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any

import serve

ROOT = Path(__file__).resolve().parents[2]
PHASES = {'prefill', 'draft-1', 'draft-2', 'verify', 'catchup', 'accept', 'replay', 'output'}
IDENTITY = ('cycle', 'phase', 'layer', 'operation', 'device', 'work')


def prepare_case(*, active=1, context=400000, slots=None, model=None, server=None,
                 mmproj=None, kv='resident', host_budget_mib=None, trace_decode=False):
    slots = active if slots is None else slots
    if type(active) is not int or type(slots) is not int or not 1 <= active <= slots <= 40:
        raise ValueError('active sessions must be 1..reserved slots, with at most 40 slots')
    if type(context) is not int or not 1 <= context <= serve.DP_SLOT_CONTEXT:
        raise ValueError('context must fit the configured per-slot context')
    if kv not in ('resident', 'host'):
        raise ValueError('kv must be resident or host')
    unresolved = ['model'] if model is None else []
    if kv == 'host' and host_budget_mib is None:
        unresolved.append('host_budget_mib')
    options = ['--profile', 'dp-4', '--slots', str(slots), '--mtp-depth', '2']
    if kv == 'host':
        # Minimum requirement is not a measurement of available RAM.
        budget = serve.host_kv_mib(slots) if host_budget_mib is None else host_budget_mib
        options += ['--kv-host', '--host-budget-mib', str(budget)]
    if trace_decode:
        options += ['--trace-decode']
    if server:
        options += ['--server', str(server)]
    if mmproj:
        options += ['--mmproj', str(mmproj)]
    # The sentinel is only used to ask the real launcher for its other settings.
    # It is never emitted as a model path or an executable launch.
    args = serve.arguments(['--model', str(model or '__UNBOUND_MODEL__'), *options, '--dry-run'])
    argv, env = serve.make_plan(args)
    serve.verify_assets()
    option = lambda name: argv[argv.index(name)+1]
    pack = json.loads((ROOT/'docs/torad/glm53-decode-roofline.json').read_text())['inputs']
    return dict(
        schema='glm53-comparison-case-v1',
        workload=dict(active_sessions=active, context_tokens=context,
                      sequence_ids=list(range(active)), phase='steady-decode',
                      prompt_tail_in_timing=False),
        configuration=dict(
            profile='dp-4', gpu_count=len(env['CUDA_VISIBLE_DEVICES'].split(',')),
            reserved_slots=int(option('-np')), reserved_context_per_slot=serve.DP_SLOT_CONTEXT,
            owners=[len(range(rank,active,4)) for rank in range(4)], kv=kv,
            max_draft_depth=int(option('--spec-draft-n-max')), depth_policy='adaptive',
            adaptive_threshold=float(option('--spec-draft-mtp-adaptive')),
            target_batch=int(option('-b')), target_microbatch=int(option('-ub')),
            draft_microbatch=int(option('-ubd')), prompt_step=int(option('--prompt-step')),
            cache_k=option('--cache-type-k'), cache_v=option('--cache-type-v'),
            indexer_cache=option('-ctki'), recurrent_state=option('-cts'),
            indexer_query=env['LLAMA_INDEXER_Q8_QUERY'], projector_loaded=mmproj is not None,
            transport='current', cache_only_catchup=False,
            minimum_host_kv_mib=serve.host_kv_mib(slots) if kv=='host' else 0),
        model=dict(repository='BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF',
                   revision=pack['model_revision'], quantization=pack['quantization'],
                   artifact_hashes='Verify against the pinned serving head before the GPU run'),
        launcher=dict(entrypoint='scripts/glm53/serve.py', options=options,
                      required_bindings=unresolved),
        resolved_launch=dict(argv=argv,env=env) if not unresolved else None,
        unresolved_bindings=unresolved,
        capture=dict(tool='Nsight Systems', options=['--trace=cuda,nvtx', '--cuda-graph-trace=node'],
                     semantic_schema='glm53-work-journal-v1',
                     required_observations=['cycle and phase identity', 'actual rows and sequence contexts',
                                            'expert routing per layer/rank', 'KV selections and reuse',
                                            'collective arrival/release', 'payload vs hardware traffic',
                                            'clock, topology and simultaneous link calibration']),
        diagnostics_enabled=trace_decode, gpu_validated=False)


def number(value):
    return not isinstance(value, bool) and isinstance(value, (int,float)) and math.isfinite(value)


def ordered_steps(steps):
    if not isinstance(steps,list) or not steps:
        raise ValueError('A nonempty list of logical steps is required')
    nodes = {}
    for row in steps:
        if not isinstance(row,dict) or not isinstance(row.get('id'),str) or not row['id']:
            raise ValueError('Each step requires a nonempty string id')
        if row['id'] in nodes:
            raise ValueError('Duplicate step id: '+row['id'])
        if type(row.get('cycle')) is not int or row['cycle'] < 0:
            raise ValueError('A nonnegative cycle ID is required')
        if row.get('phase') not in PHASES:
            raise ValueError('Unknown semantic phase')
        deps, resources = row.get('depends_on'), row.get('resources')
        if not isinstance(deps,list) or any(not isinstance(x,str) for x in deps):
            raise ValueError('Dependencies must be explicit step IDs')
        if not isinstance(resources,list) or not resources or any(not isinstance(x,str) or not x for x in resources):
            raise ValueError('Exclusive resources must be explicit nonempty names')
        duration = row.get('duration_ns')
        if duration is not None and (not number(duration) or duration < 0):
            raise ValueError('Duration must be nonnegative finite nanoseconds, or null')
        nodes[row['id']] = row
    if any(dep not in nodes for row in steps for dep in row['depends_on']):
        raise ValueError('Unknown dependency')
    remaining, done, result = set(nodes), set(), []
    while remaining:
        ready = sorted(key for key in remaining if set(nodes[key]['depends_on']) <= done)
        if not ready:
            raise ValueError('Dependency cycle')
        # A deterministic feasible list order. It does not claim globally optimal scheduling.
        key = ready[0]
        result.append(nodes[key]); done.add(key); remaining.remove(key)
    return result


def schedule(steps):
    ordered = ordered_steps(steps)
    unknown = [row['id'] for row in ordered if row.get('duration_ns') is None]
    result: dict[str, Any] = dict(complete=not unknown, unknown_costs=unknown, intervals=[], makespan_ns=None,
                  basis='Dependency-respecting exclusive-resource list schedule; not proven optimal')
    if unknown:
        return result
    ends, resource_ends = {}, {}
    for row in ordered:
        start = max([0]+[ends[dep] for dep in row['depends_on']]
                    +[resource_ends.get(resource,0) for resource in row['resources']])
        end = start + row['duration_ns']
        result['intervals'].append(dict(id=row['id'], start_ns=start, end_ns=end,
                                        device=row['device'], cycle=row['cycle'], phase=row['phase']))
        ends[row['id']] = end
        for resource in row['resources']:
            resource_ends[resource] = end
    result['makespan_ns'] = max(ends.values())
    return result


def compare_journals(ideal, recorded):
    gaps = []
    for name,journal in [('ideal',ideal),('recorded',recorded)]:
        if journal.get('schema') != 'glm53-work-journal-v1':
            raise ValueError(name+': expected a semantic work journal, not a raw capture')
        if journal.get('evidence') not in ('synthetic-fixture', 'captured', 'source-model'):
            raise ValueError(name+': evidence origin is required')
        if not isinstance(journal.get('case'),dict):
            raise ValueError(name+': case metadata is required')
    if ideal['case'].get('workload') != recorded['case'].get('workload'):
        gaps.append('Workloads differ: active sequences, contexts or timing scope')
    if ideal['case'].get('model') != recorded['case'].get('model'):
        gaps.append('Model identity differs')
    if not ideal['case'].get('workload') or not ideal['case'].get('model'):
        gaps.append('Workload and model identity must be known')
    configs = [j['case'].get('configuration',{}) for j in (ideal,recorded)]
    differences = {key:dict(ideal=configs[0].get(key), recorded=configs[1].get(key))
                   for key in sorted(set(configs[0])|set(configs[1]))
                   if configs[0].get(key) != configs[1].get(key)}
    if ideal.get('cost_coverage') != 'complete':
        gaps.append('Ideal model is not a complete execution-cost model')
    for journal in (ideal,recorded):
        ordered_steps(journal['steps'])
        if len({row['cycle'] for row in journal['steps']}) != 1:
            raise ValueError('Compare exactly one cycle at a time; do not relabel a whole request')
    plan = schedule(ideal['steps'])
    if not plan['complete']:
        gaps.append('Unknown ideal costs: '+', '.join(plan['unknown_costs']))
    actual = {row['id']:row for row in ordered_steps(recorded['steps'])}
    expected = {row['id']:row for row in ideal['steps']}
    if set(actual) != set(expected):
        gaps.append('Logical step coverage differs')
    pairs, spans = [], []
    for key in sorted(set(actual)&set(expected)):
        a,b = expected[key],actual[key]
        if any(field not in a or field not in b or a[field]!=b[field] for field in IDENTITY) or not a.get('work'):
            gaps.append('Logical workload or identity differs: '+key)
            continue
        start,end = b.get('start_ns'),b.get('end_ns')
        if not number(start) or not number(end) or end < start:
            gaps.append('Missing or invalid recorded interval: '+key)
            continue
        spans.append((start,end))
        pairs.append(dict(id=key, cycle=a['cycle'], phase=a['phase'], layer=a['layer'],
                          operation=a['operation'], device=a['device'],
                          ideal_ns=a.get('duration_ns'), recorded_ns=end-start))
    for row in actual.values():
        for dep in row['depends_on']:
            parent = actual[dep]
            if number(row.get('start_ns')) and number(parent.get('end_ns')) and row['start_ns'] < parent['end_ns']:
                gaps.append('Recorded dependency is violated: '+row['id']+' <- '+dep)
    total = max(b for _,b in spans)-min(a for a,_ in spans) if spans else None
    comparable = not gaps
    synthetic = any(j['evidence']=='synthetic-fixture' for j in (ideal,recorded))
    return dict(schema='glm53-matched-comparison-v1', comparable=comparable, gaps=gaps,
                configuration_differences=differences, pairs=pairs, ideal_schedule=plan,
                ideal_cycle_ns=plan['makespan_ns'], recorded_cycle_ns=total,
                speedup=total/plan['makespan_ns'] if comparable and plan['makespan_ns'] else None,
                evidence='synthetic-fixture' if synthetic else 'captured-vs-source-model',
                gpu_validated=False,
                caveat='Semantic work matching is not numerical, capacity or performance qualification.')


def read_json(path):
    path=Path(path)
    with gzip.open(path,'rt') if path.suffix=='.gz' else path.open() as stream:
        return json.load(stream)


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise ValueError('Output exists; choose a new output path: '+str(path))
    data=json.dumps(value,indent=2,allow_nan=False)+'\n'
    if path.suffix=='.gz':
        with path.open('xb') as stream:
            stream.write(gzip.compress(data.encode(),mtime=0))
    else:
        with path.open('x') as stream:
            stream.write(data)


def inspect_profile(profile):
    trace=profile.get('trace') or {}
    return dict(schema='glm53-comparison-readiness-v1', metadata=profile.get('metadata',{}),
                source=profile.get('source',{}), counts=profile.get('counts',{}),
                scheduling_counts={key:len(trace.get(key,[])) for key in
                                   ('kernels','copies','memsets','runtime','cuda_events','synchronizations','graph_nodes')},
                semantic_marker_count=len(trace.get('semantic_markers',[])),
                semantic_coverage=trace.get('semantic_coverage',{}), full_comparison_ready=False,
                gaps=['No trusted cycle/phase-to-logical-step journal has been supplied',
                      'Actual routing, selected-KV reuse and per-rank collective arrivals are not established',
                      'Kernel-driven traffic counters and calibrated complete ideal costs are not supplied'],
                note='Recovered scheduling records are usable evidence, not a substitute for semantic workload identity.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare',help='Write a one-session case; never launch inference')
    prep.add_argument('--active',type=int,default=1)
    prep.add_argument('--context',type=int,default=400000)
    prep.add_argument('--slots',type=int)
    prep.add_argument('--model')
    prep.add_argument('--server')
    prep.add_argument('--mmproj')
    prep.add_argument('--kv',choices=('resident','host'),default='resident')
    prep.add_argument('--host-budget-mib',type=float)
    prep.add_argument('--trace-decode',action='store_true')
    prep.add_argument('--output',required=True,type=Path)
    compare=sub.add_parser('compare')
    compare.add_argument('--ideal',required=True,type=Path)
    compare.add_argument('--recorded',required=True,type=Path)
    compare.add_argument('--output',required=True,type=Path)
    inspect=sub.add_parser('inspect')
    inspect.add_argument('--profile',required=True,type=Path)
    inspect.add_argument('--output',required=True,type=Path)
    args=vars(parser.parse_args());command=args.pop('command');output=args.pop('output')
    try:
        if command=='prepare':
            value=prepare_case(**args)
        elif command=='compare':
            value=compare_journals(read_json(args['ideal']),read_json(args['recorded']))
        else:
            value=inspect_profile(read_json(args['profile']))
        write_json(output,value)
        print(json.dumps(dict(output=str(output),schema=value['schema'],gpu_validated=False)))
    except (ValueError,KeyError,TypeError,OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
