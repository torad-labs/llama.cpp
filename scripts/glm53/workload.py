"""Concurrent sessions against the running DP server on this box: fill, then a measured decode. No server control."""
import argparse
import concurrent.futures
import hashlib
import http.client
import json
from pathlib import Path
import statistics
import sys
import threading
import time
import urllib.request
import urllib.error
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--depth', choices=['1', '2', 'off'], required=True)
    parser.add_argument('--port', type=int, default=8102)
    parser.add_argument('--slots', type=int, required=True)
    parser.add_argument('--active', type=int, required=True)
    parser.add_argument('--context', type=int, required=True)
    parser.add_argument('--tokens', type=int, default=256)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--launch-json', type=Path, help='Actual server launch record')
    args = parser.parse_args()
    assert 1 <= args.active <= args.slots <= 40
    assert 128 < args.context <= 400000
    assert 32 <= args.tokens <= 1024
    assert args.deadline > time.time()
    sys.path.insert(0, str(args.root / 'scripts/glm53'))
    from measure import ServingLeg, completion_payload, read_completion_stream, retained_lines, common_decode_window, write_json

    config = SimpleNamespace(root=args.root, run=args.name, port=args.port,
        deadline=args.deadline, binary=args.binary, mode='timing',
        recipe='dp-4', profile=None, ubatch=512, context=args.slots*442368,
        slots=args.slots, adaptive=0.18, model=args.root/'model/GLM-5.3-Flash-Uncensored-IQ3_XXS-00001-of-00003.gguf',
        depth=args.depth, kv_host=False, host_budget_mib=None, mmproj=None)
    client = ServingLeg(config)
    client.out.mkdir(parents=True, exist_ok=False)
    report = dict(started_epoch=time.time(), arguments=vars(args),
        reserved_slots=args.slots, active_sessions=args.active, initial_prompt_tokens=args.context, decode_tokens=args.tokens,
        workload='Public corpus (the fork docs), repeated to exact token depth; distinct session prefix; native completion API',
        sampling='temperature 1, top_p .95, repeat penalty 1.1 over 256, ignore_eos, seed 530 + slot',
        passed=False)
    if args.launch_json:
        report['launch_sha256'] = hashlib.sha256(args.launch_json.read_bytes()).hexdigest()
    write_json(client.out/'run.json', report)
    corpus = (args.root/'public-corpus.txt').read_text()
    report['corpus_sha256'] = hashlib.sha256(corpus.encode()).hexdigest()
    prompts = []
    for slot in range(args.active):
        fragment = corpus[:max(10000, args.context*6)]
        copies = max(2, (args.context*6)//len(fragment)+2)
        while True:
            text = f'Session {slot}: inspect these public source excerpts.\n' + (fragment+'\n')*copies
            messages = [dict(role='system', content='You are a software engineering assistant.'),
                        dict(role='user', content=text+'\nExplain the implementation in detail, including several examples. Continue until the explanation is complete.')]
            rendered = client.request('/apply-template', dict(messages=messages))['prompt']
            ids = client.request('/tokenize', dict(content=rendered, add_special=True))['tokens']
            if len(ids) >= args.context:
                break
            copies *= 2
        ids = ids[:args.context-128] + ids[-128:]
        assert len(ids) == args.context
        prompts.append(ids)
    report['prompt_sha256'] = [hashlib.sha256(json.dumps(p).encode()).hexdigest() for p in prompts]
    write_json(client.out/'run.json', report)

    def metrics(tag):
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/metrics', timeout=client.timeout(10)) as r:
                data = r.read()
        except (OSError, urllib.error.URLError, http.client.HTTPException) as exc:
            status = dict(available=False, error_kind=type(exc).__name__)
            if isinstance(exc, urllib.error.HTTPError):
                status['status'] = exc.code
        else:
            (client.out/f'metrics-{tag}.txt').write_bytes(data)
            status = dict(available=True)
        report.setdefault('metrics', {})[tag] = status


    def complete(slot, count, phase, barrier, cached):
        payload = completion_payload(prompts[slot], count, slot)
        payload.update(stream=True, return_tokens=True, cache_prompt=cached, seed=530+slot)
        record = client.response_record()
        barrier.wait(timeout=60)
        started = time.monotonic()
        with client.capture_response('/completion', payload, maximum=7200, record=record,
                context=dict(phase=phase, slot=slot, prompt_tokens=len(prompts[slot]))) as (response, saved):
            with record.with_suffix('.events.jsonl').open('x') as events:
                def keep_event(event):
                    events.write(json.dumps(event)+'\n')
                    events.flush()
                value = read_completion_stream(retained_lines(response, saved, client.timeout), count,
                    None if cached else len(prompts[slot]), on_event=keep_event)
        if value['response']['id_slot'] != slot:
            raise ValueError('Response came from a different reserved slot')
        if value['response'].get('truncated'):
            raise ValueError('Context was truncated')
        value.update(slot=slot, owner_device=slot%4, started=started, ended=time.monotonic(),
            response_record=str(record.relative_to(client.out)))
        write_json(record.with_suffix('.stream.json'), value)
        prompts[slot] += value['token_ids']
        return value

    def group(count, phase, cached):
        barrier = threading.Barrier(args.active)
        start, mono = time.time(), time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.active) as pool:
            futures = [pool.submit(complete, slot, count, phase, barrier, cached) for slot in range(args.active)]
            rows = [future.result() for future in futures]
        ended = time.monotonic()
        first = min(r['bursts'][0]['at'] for r in rows)
        last = max(r['bursts'][-1]['at'] for r in rows)
        wall = dict(seconds=ended-mono, tokens=args.active*count,
            tokens_per_second=args.active*count/(ended-mono),
            first_token_to_last_seconds=last-first,
            first_token_to_last_tokens_per_second=(args.active*count)/(last-first) if last > first else None,
            per_stream_completion_seconds=sorted(r['bursts'][-1]['at']-r['started'] for r in rows))
        value = dict(phase=phase, started_epoch=start, ended_epoch=time.time(), streams=rows,
            common_decode=common_decode_window(rows), wall=wall)
        write_json(client.out/(phase+'.json'), value)
        return value

    try:
        filled = group(16, 'warmup', False)
        report['warmup_seconds'] = filled['wall']['seconds']
        report['warmup_tokens_per_session'] = [len(row['token_ids']) for row in filled['streams']]
        report['warmup_generated_tokens'] = sum(report['warmup_tokens_per_session'])
        report['populated_prompt_tokens'] = args.active*args.context
        report['decode_prompt_tokens'] = [len(prompt) for prompt in prompts]
        report['decode_prompt_sha256'] = [hashlib.sha256(json.dumps(p).encode()).hexdigest() for p in prompts]
        write_json(client.out/'run.json', report)
        metrics('before')
        report['decode_started_epoch'] = time.time()
        write_json(client.out/'run.json', report)
        measured = group(args.tokens, 'decode', True)
        report['decode_finished_epoch'] = measured['ended_epoch']
        metrics('after')
        overlap = measured['common_decode']
        if not overlap['available'] or overlap['seconds'] <= 0:
            raise ValueError('No common decode interval; concurrent throughput is unavailable')
        cache = [r['response']['timings'].get('cache_n') for r in measured['streams']]
        timings = [r['response']['timings'] for r in measured['streams']]
        drafted = sum(t.get('draft_n', 0) or 0 for t in timings)
        accepted = sum(t.get('draft_n_accepted', 0) or 0 for t in timings)
        report.update(common_decode=overlap, wall=measured['wall'], cache_n=cache,
            cache_retained=all(isinstance(n, int) and n >= 0.90*args.context for n in cache),
            server_timings=timings, draft_n=drafted, draft_n_accepted=accepted,
            median_stream_tokens_per_second=statistics.median(s['tokens_per_second'] for s in overlap['streams']),
            passed=True)
        if not report['cache_retained']:
            report['passed'] = False
            raise ValueError('Decode continuation did not retain at least 90 percent of each prompt')
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['finished_epoch'] = time.time()
        write_json(client.out/'run.json', report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('arguments','server_timings','prompt_sha256','decode_prompt_sha256')},default=str), flush=True)


if __name__ == '__main__':
    main()
