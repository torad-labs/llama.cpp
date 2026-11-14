#!/usr/bin/env python3
"""What a prompt read costs and where the time goes, from the log of a server started with serve.py --trace-decode.

    prompt_evals.py LOG [--list]

The log's trace lines carry row counts, sizes and times, and no prompt text or token. Two streams are matched in order:

  GLM53_TRACE spans (LLAMA_DECODE_TRACE=1): a server iteration, each target eval in it with its rows and, a sequence,
  its prompt rows, and the MTP head's proposals and catch-ups;

  graph_trace lines (GGML_GRAPH_TRACE=1 at -lv 4): one "layer=llama event=build|reuse" a decode (a ubatch) with the host
  microseconds of its slot, measure, build and alloc, the scheduler's alloc line before it, and the CUDA subgraphs it
  ran after it: "direct" (launched kernel by kernel), "capture" (run once more to record) and "replay" (a recorded
  graph launched whole).

A target eval's "run" is its span less the host time of its decodes: inputs, the cards' work and the wait for it.
Printed: the evals that carry prompt rows, by the eval's prompt rows (how many, how many built a graph, medians, and
the same for those that built and those that replayed); the evals of replies alone, by rows; what the iterations
with a prompt spent outside their target evals (the head's catch-up, context checkpoints, the rest); and with --list
every eval that carried prompt rows. A graph built for a row count seen once shows as "built" with every CUDA subgraph
direct; the same eval from a kept graph shows as "replayed".
"""
import collections
import re
import statistics
import sys

EVENT = re.compile(r'GLM53_TRACE event=(\w+) clock=host-monotonic host_ns=(\d+) thread=\w+ nvtx=\d (.*)')
GRAPH = re.compile(r'graph_trace layer=(\w+) event=(\w+) (.*)')
CHECKPOINT = re.compile(r'created context checkpoint .*?, (new|kept) buffers, ([0-9.]+) ms')
CHECKPOINT_PLAIN = re.compile(r'created context checkpoint')
BUCKETS = ((1, 32), (33, 128), (129, 512), (513, 2048), (2049, 1 << 30))
ORDER = ['1-32', '33-128', '129-512', '513-2048', '2049+']


def fields(text):
    return dict(pair.split('=', 1) for pair in text.split() if '=' in pair)


def bucket(rows):
    for low, high in BUCKETS:
        if low <= rows <= high:
            return '%d-%d' % (low, high) if high < 1 << 30 else '2049+'
    return None


def median(values):
    return statistics.median(values) if values else float('nan')


def read(lines):
    """The log's iterations, decodes and checkpoints, matched: (prompt evals by bucket, reply evals by rows, what the
    iterations with a prompt spent outside their target evals by bucket, counts)."""
    iterations = []
    current = None
    open_span = None
    decodes = {'target': [], 'draft': []}
    pending_alloc = None
    last_decode = None
    checkpoints = []  # (ms, new or kept buffers, the target decodes seen before it)
    plain_checkpoints = 0
    builds = 0
    builds_untimed = 0  # build lines of an engine that does not print host_us: their host time is inside the run
    for line in lines:
        match = EVENT.search(line)
        if match:
            kind, ns, f = match.group(1), int(match.group(2)), fields(match.group(3))
            stage = f.get('stage')
            if stage == 'iteration':
                if kind == 'begin':
                    current = {'t0': ns, 'spans': [], 'id': f.get('iteration')}
                elif kind == 'end' and current is not None:
                    current['t1'] = ns
                    iterations.append(current)
                    current = None
            elif current is not None:
                if kind == 'begin':
                    open_span = {'stage': stage, 'b': ns, 'f': f, 'marks': []}
                elif kind == 'end' and open_span is not None and open_span['stage'] == stage:
                    open_span['e'] = ns
                    current['spans'].append(open_span)
                    open_span = None
                elif kind == 'mark' and stage == 'batch' and open_span is not None:
                    open_span['marks'].append(f)
            continue
        match = GRAPH.search(line)
        if match:
            layer, event, f = match.group(1), match.group(2), fields(match.group(3))
            if layer == 'sched' and event == 'alloc':
                pending_alloc = f
            elif layer == 'llama' and event in ('build', 'reuse'):
                host = dict(part.split(':') for part in f.get('host_us', 'slot:0,measure:0,build:0,alloc:0').split(','))
                last_decode = {'event': event, 'rows': int(f.get('rows', 0)),
                               'host': {key: int(value) for key, value in host.items()},
                               'sched': pending_alloc if event == 'build' else None, 'cuda': collections.Counter()}
                pending_alloc = None
                builds += event == 'build'
                builds_untimed += event == 'build' and 'host_us' not in f
                decodes['target' if f.get('phase') == 'target' else 'draft'].append(last_decode)
            elif layer == 'cuda' and last_decode is not None:
                last_decode['cuda'][event] += 1
            continue
        match = CHECKPOINT.search(line)
        if match:
            checkpoints.append((float(match.group(2)), match.group(1), len(decodes['target'])))
        elif CHECKPOINT_PLAIN.search(line):
            plain_checkpoints += 1

    # a span takes decodes of its stream, in order, until their rows are its rows
    position = {'target': 0, 'draft': 0}
    unmatched = 0
    for iteration in iterations:
        iteration['first_target'] = position['target']
        for span in iteration['spans']:
            if span['stage'] not in ('target_eval', 'proposal', 'catchup'):
                continue
            phase = 'target' if span['stage'] == 'target_eval' else 'draft'
            rows, got, taken = int(span['f'].get('rows', 0)), 0, []
            while got < rows and position[phase] < len(decodes[phase]):
                taken.append(decodes[phase][position[phase]])
                got += taken[-1]['rows']
                position[phase] += 1
            span['decodes'] = taken
            unmatched += got != rows
        iteration['last_target'] = position['target']
    by_decode = collections.defaultdict(list)
    for checkpoint in checkpoints:
        by_decode[checkpoint[2]].append(checkpoint)
    for iteration in iterations:
        # a checkpoint is made in the iteration whose target decodes it sits among
        iteration['checkpoints'] = [checkpoint for index in range(iteration['first_target'], iteration['last_target'] + 1)
                                    for checkpoint in by_decode.get(index, [])
                                    if index > iteration['first_target'] or iteration['last_target'] > iteration['first_target']]

    prompt_evals = collections.defaultdict(list)
    reply_evals = collections.defaultdict(list)
    outside = collections.defaultdict(list)
    for iteration in iterations:
        prompt_rows = 0
        for span in iteration['spans']:
            if span['stage'] != 'target_eval':
                continue
            prompt = sum(int(mark.get('prompt_rows', 0)) for mark in span['marks'])
            rows = int(span['f'].get('rows', 0))
            ms = (span['e'] - span['b']) / 1e6
            taken = span.get('decodes', [])
            host = {key: sum(d['host'].get(key, 0) for d in taken) / 1e3 for key in ('slot', 'measure', 'build', 'alloc')}
            record = {'ms': ms, 'rows': rows, 'prompt': prompt, 'passes': len(taken),
                      'built': sum(d['event'] == 'build' for d in taken), 'host': host, 'host_ms': sum(host.values()),
                      'run_ms': ms - sum(host.values()),
                      'direct': sum(d['cuda']['direct'] for d in taken), 'capture': sum(d['cuda']['capture'] for d in taken),
                      'replay': sum(d['cuda']['replay'] for d in taken),
                      'place_new': sum(int((d['sched'] or {}).get('place_new_us', 0)) for d in taken) / 1e3,
                      'plan': sum(int((d['sched'] or {}).get('plan_us', 0)) + int((d['sched'] or {}).get('split_us', 0))
                                  for d in taken) / 1e3,
                      'iteration': iteration['id']}
            if prompt > 0:
                prompt_evals[bucket(prompt)].append(record)
                prompt_rows += prompt
            else:
                reply_evals[rows].append(record)
        if prompt_rows:
            target = sum(span['e'] - span['b'] for span in iteration['spans'] if span['stage'] == 'target_eval') / 1e6
            catchups = [span for span in iteration['spans'] if span['stage'] == 'catchup']
            proposals = [span for span in iteration['spans'] if span['stage'] == 'proposal']
            outside[bucket(prompt_rows)].append({
                'iteration_ms': (iteration['t1'] - iteration['t0']) / 1e6, 'target_ms': target,
                'catchup_ms': sum(span['e'] - span['b'] for span in catchups) / 1e6,
                'catchup_built': sum(d['event'] == 'build' for span in catchups for d in span.get('decodes', [])),
                'proposal_ms': sum(span['e'] - span['b'] for span in proposals) / 1e6,
                'checkpoint_ms': sum(checkpoint[0] for checkpoint in iteration['checkpoints']),
                'checkpoints': len(iteration['checkpoints'])})
    counts = {'iterations': len(iterations), 'target_decodes': len(decodes['target']), 'draft_decodes': len(decodes['draft']),
              'unmatched': unmatched, 'checkpoints': checkpoints, 'plain_checkpoints': plain_checkpoints,
              'builds': builds, 'builds_untimed': builds_untimed}
    return prompt_evals, reply_evals, outside, counts


def table(title, groups, keys, out):
    print(title, file=out)
    print('  %-9s %5s %5s %6s | %8s %8s = %6s %6s %6s %6s | %8s | %7s %7s %7s' % (
        'rows', 'evals', 'built', 'passes', 'eval ms', 'host ms', 'slot', 'measr', 'build', 'alloc', 'run ms',
        'direct', 'capture', 'replay'), file=out)
    for key in keys:
        evals = groups.get(key)
        if not evals:
            continue
        for name, chosen in (('all', evals), ('built', [x for x in evals if x['built']]),
                             ('replayed', [x for x in evals if not x['built']])):
            if not chosen or (name != 'all' and len(chosen) == len(evals)):
                continue
            print('  %-9s %5d %5d %6.1f | %8.2f %8.2f = %6.2f %6.2f %6.2f %6.2f | %8.2f | %7.0f %7.0f %7.0f   %s' % (
                key if name == 'all' else '', len(chosen), sum(1 for x in chosen if x['built']),
                statistics.fmean(x['passes'] for x in chosen), median([x['ms'] for x in chosen]),
                median([x['host_ms'] for x in chosen]), median([x['host']['slot'] for x in chosen]),
                median([x['host']['measure'] for x in chosen]), median([x['host']['build'] for x in chosen]),
                median([x['host']['alloc'] for x in chosen]), median([x['run_ms'] for x in chosen]),
                median([x['direct'] for x in chosen]), median([x['capture'] for x in chosen]),
                median([x['replay'] for x in chosen]), '' if name == 'all' else name), file=out)


def report(name, data, show, out):
    prompt_evals, reply_evals, outside, counts = data
    print('%s: %d iterations, %d target decodes and %d of the head in the graph trace%s' % (
        name, counts['iterations'], counts['target_decodes'], counts['draft_decodes'],
        '' if not counts['unmatched'] else '; %d spans whose rows the graph trace did not match' % counts['unmatched']),
        file=out)
    if not counts['target_decodes']:
        print('  no graph_trace line: host and run are not split (GGML_GRAPH_TRACE=1 at -lv 4)', file=out)
    elif counts['builds_untimed']:
        print('  %d of %d build lines carry no host_us (an engine before the field): the host time of those builds is '
              'inside "run", and "host" reads 0 for them' % (counts['builds_untimed'], counts['builds']), file=out)
    table('target evals that carry prompt rows, by the eval\'s prompt rows (medians, ms; CUDA subgraphs an eval):',
          prompt_evals, ORDER, out)
    commonest = sorted(reply_evals, key=lambda rows: -len(reply_evals[rows]))[:8]
    table('target evals of replies alone, by rows (the eight commonest):',
          {str(rows): reply_evals[rows] for rows in commonest}, [str(rows) for rows in sorted(commonest)], out)
    print('iterations with a prompt, by their prompt rows: outside the target evals (medians, ms):', file=out)
    print('  %-9s %5s | %8s %8s | %8s %7s | %8s | %6s %8s | %8s' % (
        'rows', 'iters', 'iter ms', 'target', 'catch-up', 'built', 'proposal', 'ckpts', 'ckpt ms', 'the rest'), file=out)
    for key in ORDER:
        rows = outside.get(key)
        if not rows:
            continue
        print('  %-9s %5d | %8.2f %8.2f | %8.2f %7.1f | %8.2f | %6.2f %8.2f | %8.2f' % (
            key, len(rows), median([x['iteration_ms'] for x in rows]), median([x['target_ms'] for x in rows]),
            median([x['catchup_ms'] for x in rows]), statistics.fmean(x['catchup_built'] for x in rows),
            median([x['proposal_ms'] for x in rows]), statistics.fmean(x['checkpoints'] for x in rows),
            statistics.fmean(x['checkpoint_ms'] for x in rows),
            median([x['iteration_ms'] - x['target_ms'] - x['catchup_ms'] - x['proposal_ms'] - x['checkpoint_ms']
                    for x in rows])), file=out)
    checkpoints = counts['checkpoints']
    if checkpoints:
        ms = [checkpoint[0] for checkpoint in checkpoints]
        print('context checkpoints: %d with a time (median %.2f ms, max %.2f; %d into new buffers), %d lines without one' % (
            len(ms), median(ms), max(ms), sum(1 for c in checkpoints if c[1] == 'new'), counts['plain_checkpoints']), file=out)
    elif counts['plain_checkpoints']:
        print('context checkpoints: %d lines, none with a time (an engine that does not print it): what a checkpoint took '
              'is inside "the rest" of its iteration' % counts['plain_checkpoints'], file=out)
    if show:
        print('every eval with prompt rows: iteration, prompt rows, rows, passes, built, eval ms, host ms, run ms, '
              'direct/capture/replay', file=out)
        for key in ORDER:
            for x in prompt_evals.get(key, []):
                print('  it %-6s p %5d rows %5d passes %2d built %2d  %8.2f = host %7.2f (alloc %6.2f, new place %6.2f, '
                      'plan %5.2f) + run %8.2f   %d/%d/%d' % (
                          x['iteration'], x['prompt'], x['rows'], x['passes'], x['built'], x['ms'], x['host_ms'],
                          x['host']['alloc'], x['place_new'], x['plan'], x['run_ms'], x['direct'], x['capture'],
                          x['replay']), file=out)


def main(argv=None, out=sys.stdout):
    argv = sys.argv[1:] if argv is None else argv
    paths = [arg for arg in argv if not arg.startswith('--')]
    if len(paths) != 1 or any(arg != '--list' for arg in argv if arg.startswith('--')):
        print('usage: prompt_evals.py LOG [--list]', file=sys.stderr)
        return 2
    with open(paths[0], errors='replace') as log:
        data = read(log)
    report(paths[0].rsplit('/', 1)[-1], data, '--list' in argv, out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
