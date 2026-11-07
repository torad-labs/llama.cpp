"""Run an explicit serving plan on an already approved box; no rental or download.
One server at a time, guarded at 4096 MiB free per card. Completion markers request teardown, never certify success."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--deadline', type=float, required=True)
p.add_argument('--plan', required=True, help='JSON list of [engine, depth, [[active, context], ...], [[active, context], ...]]')
p.add_argument('--tag', default='p1', help='prefix of this pass\'s folders and its done marker')
a = p.parse_args()
root = a.root
out = root/'results'
selected_cards = json.loads((out/'selected-gpus.json').read_text())['selected']
assert len(selected_cards) == 4
CARD_INDICES = [card['index'] for card in selected_cards]
CARD_UUIDS = [card['uuid'] for card in selected_cards]
sys.path.insert(0, str(root/'scripts/glm53'))
import serve
from qualification import matrix_succeeded, profile_evidence, serving_window_seconds
sys.path.insert(0, str(root))
import engines

SLOTS = 36
PORT = 8102
MODEL = root/'model/GLM-5.3-Flash-Uncensored-IQ3_XXS-00001-of-00003.gguf'
PLAN = [(e, d, [tuple(x) for x in s_], [tuple(x) for x in l_], *rest) for e, d, s_, l_, *rest in json.loads(a.plan)]
report = dict(started_epoch=time.time(), servers=[], slots=SLOTS, tag=a.tag, plan=PLAN)


def write(path, data):
    path.write_text(json.dumps(data, indent=2, default=str) + '\n')


def event(kind, **values):
    value = dict(epoch=time.time(), event=kind, **values)
    write(out/'progress.json', value)
    print(json.dumps(value, default=str), flush=True)
    with (out/'progress.jsonl').open('a') as f:
        f.write(json.dumps(value, default=str) + '\n')


def request(path, payload=None, timeout=30):
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}' + path,
        data=None if payload is None else json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=min(timeout, max(1, a.deadline-time.time()))) as response:
        return json.load(response)


def memory():
    fields = ['index', 'memory.total', 'memory.used', 'memory.free', 'power.draw', 'utilization.gpu',
              'clocks.sm', 'clocks.mem', 'temperature.gpu']
    raw = subprocess.check_output(['nvidia-smi', '-i', ','.join(CARD_UUIDS), '--query-gpu='+','.join(fields), '--format=csv,noheader,nounits'],
                                  text=True, timeout=10)
    rows = []
    for line in raw.splitlines():
        v = [s.strip() for s in line.split(',')]
        rows.append(dict(index=CARD_INDICES.index(int(v[0])), physical_index=int(v[0]), total_mib=int(v[1]), used_mib=int(v[2]), free_mib=int(v[3]), power_w=v[4],
                         utilization_pct=v[5], sm_mhz=v[6], memory_mhz=v[7], temperature_c=v[8]))
    return rows


class Server:
    def __init__(self, engine, depth, ordinal, extra=None):
        self.engine, self.depth = engine, depth
        self.label = f'{a.tag}-{ordinal:02d}-{engine}-mtp{depth}' + (f"-{extra['label']}" if extra else '')
        self.legs = 0
        self.folder = out/self.label
        self.folder.mkdir(exist_ok=False)
        self.binary = root/'engines'/engine/'bin'
        self.done = threading.Event()
        self.failure = None
        self.minimum = [None]*4
        params = ['--profile', 'dp-4', '--slots', str(SLOTS), '--mtp-depth', depth, '--port', str(PORT),
                  '--model', str(MODEL), '--server', str(self.binary/'llama-server')] + ((extra or {}).get('serve') or [])
        command, settings = serve.make_plan(serve.arguments(params))
        settings['CUDA_VISIBLE_DEVICES'] = ','.join(CARD_UUIDS)
        collective = (extra or {}).get('allreduce', 'internal')
        if collective not in ('internal', 'internal-nccl-gather'):
            raise ValueError('serving qualification preserves internal numeric reductions')
        settings['GGML_CUDA_ALLREDUCE'] = collective
        command = command + ((extra or {}).get('args') or [])
        # A profiled server runs under an Nsight session whose collection starts only in the decode window (workload).
        self.session = self.label if (extra or {}).get('profile') else None
        if self.session:
            command = ['nsys', 'launch', '--session-new='+self.session, '--wait=all', '--trace=cuda,nvtx',
                       '--cuda-graph-trace=node', '--cuda-flush-interval=1000'] + command
        write(self.folder/'launch.json', dict(argv=command, env=settings, engine=engine, depth=depth, extra=extra,
              source=(root/'engines'/engine/'SOURCE').read_text().strip()))
        self.log = (self.folder/'server.log').open('x')
        self.started = time.time()
        self.proc = subprocess.Popen(command, env=serve.execution_environment(settings, {key: os.environ[key] for key in ("PATH", "HOME", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER") if key in os.environ}), stdin=subprocess.DEVNULL,
            stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True, cwd=root)
        self.guard = threading.Thread(target=self.watch, daemon=True)
        self.guard.start()

    def kill(self, sig=signal.SIGTERM):
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, sig)
            except ProcessLookupError:
                pass

    def watch(self):
        try:
            with (self.folder/'memory.jsonl').open('x') as f:
                while not self.done.is_set():
                    rows = memory()
                    for row in rows:
                        d = row['index']
                        self.minimum[d] = row['free_mib'] if self.minimum[d] is None else min(self.minimum[d], row['free_mib'])
                    f.write(json.dumps(dict(epoch=time.time(), gpus=rows)) + '\n')
                    f.flush()
                    if min(r['free_mib'] for r in rows) < 4096:
                        self.failure = 'memory_reserve'
                        break
                    if time.time() >= a.deadline:
                        self.failure = 'deadline'
                        break
                    self.done.wait(1)
        except Exception as exc:
            self.failure = 'guard_error: ' + str(exc)
        finally:
            if self.failure:
                self.kill()
                event('guard_stopped_server', server=self.label, reason=self.failure, minimum_free_mib=self.minimum)

    def ready(self):
        end = min(a.deadline, time.time()+900)
        last_error = 'no_health_response'
        while time.time() < end:
            if self.failure or self.proc.poll() is not None:
                raise RuntimeError(self.failure or f'server_exited_before_ready ({self.proc.returncode})')
            try:
                if request('/health', timeout=2).get('status') == 'ok':
                    break
                last_error = 'health_status_not_ok'
            except (OSError, ValueError) as exc:
                last_error = type(exc).__name__ + (f' status={exc.code}' if isinstance(exc, urllib.error.HTTPError) else '')
            time.sleep(1)
        else:
            raise TimeoutError('Server readiness deadline; last error: '+last_error)
        if (self.folder/'launch.json').is_file():
            transport = json.loads((self.folder/'launch.json').read_text())['env']['GGML_CUDA_ALLREDUCE']
            if transport == 'internal-nccl-gather' and 'falling back' in (self.folder/'server.log').read_text():
                raise RuntimeError('native gather transport fell back during initialization')
        self.ready_seconds = time.time() - self.started
        write(self.folder/'ready-memory.json', memory())
        short = request('/v1/chat/completions', dict(model='GLM-5.3-Flash-Uncensored-IQ3_XXS',
            messages=[dict(role='user', content='What is 2 + 2? Answer with only the number.')],
            temperature=0, max_tokens=128, stream=False, chat_template_kwargs={'enable_thinking': False}), timeout=300)
        write(self.folder/'short-response.json', short)
        self.short_ok = (short['choices'][0]['message'].get('content') or '').strip() in ('4', '4.')
        if not self.short_ok:
            raise RuntimeError('short-answer quality check failed; see short-response.json')
        # One greedy single-session completion: its tokens should match across engines that claim bit-equal logits.
        greedy = request('/completion', dict(prompt='The three laws of thermodynamics, stated briefly:\n1.',
            n_predict=96, temperature=0, top_k=1, seed=1, cache_prompt=False, return_tokens=True), timeout=300)
        write(self.folder/'greedy.json', greedy)
        if self.failure:
            raise RuntimeError(self.failure)
        event('server_ready', server=self.label, ready_seconds=self.ready_seconds, short_ok=self.short_ok,
              minimum_free_mib=self.minimum)

    def close(self):
        self.kill(signal.SIGINT)
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.kill(signal.SIGKILL)
            self.proc.wait(timeout=20)
        self.done.set()
        self.guard.join(timeout=15)
        self.log.close()
        write(self.folder/'memory-summary.json', dict(minimum_free_mib=self.minimum, failure=self.failure,
              exit=self.proc.returncode, measurement='1 s nvidia-smi samples, not continuous minima'))


def processing_slots():
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/metrics', timeout=2) as r:
            for line in r.read().decode().splitlines():
                if line.startswith('llamacpp:requests_processing'):
                    return float(line.split()[-1])
    except (OSError, ValueError):
        pass
    return None


def profile_decode(server, name, active, cancelled, window=4.0):
    """Capture `window` seconds of steady decode: after the prefill group, once every session is generating."""
    from measure import finish_nsys_capture
    record = dict(name=name)
    output = server.folder/f'{name}-decode'
    try:
        end = min(a.deadline, time.time()+300)
        while not (out/name/'warmup.json').exists() and time.time() < end and server.proc.poll() is None and not cancelled.is_set():
            cancelled.wait(0.2)
        seen = None
        while time.time() < end and server.proc.poll() is None and not cancelled.is_set():
            seen = processing_slots()
            if seen is not None and seen >= active:
                break
            cancelled.wait(0.25)
        record['processing_slots_at_start'] = seen
        if cancelled.wait(3) or time.time() >= end or seen is None or seen < active:
            raise RuntimeError('active_decode_not_observed')
        seen = processing_slots()
        if cancelled.is_set() or server.proc.poll() is not None or seen is None or seen < active:
            raise RuntimeError('active_decode_ended_before_capture')
        command = ['nsys', 'start', '--session='+server.session, '--sample=none', '--cpuctxsw=none',
                   '--output='+str(output), '--force-overwrite=false']
        r = subprocess.run(command, capture_output=True, text=True, timeout=min(30, max(1, a.deadline-time.time())))
        record.update(start_epoch=time.time(), start=dict(argv=command, exit=r.returncode, stdout=r.stdout, stderr=r.stderr))
        if r.returncode == 0:
            cancelled.wait(min(window, max(0, a.deadline-time.time())))
            record['stop_epoch'] = time.time()
            record['finalize'] = finish_nsys_capture(server.proc, server.session, output, min(a.deadline, time.time()+240))
    except Exception as exc:
        record['error_kind'] = type(exc).__name__
        record['error'] = str(exc)
    finally:
        record['evidence'] = profile_evidence(True, False, record)
        write(server.folder/f'{name}-profile.json', record)
        event('profile_finished', server=server.label, name=name, **record['evidence'])


def workload(server, active, context, tokens=256):
    server.legs += 1
    name = f'{server.label}-{server.legs:02d}-active{active}-ctx{context}'
    event('workload_start', server=server.label, active=active, context=context)
    command = ['python3', str(root/'scripts/glm53/workload.py'), '--root', str(root), '--name', name, '--binary', str(server.binary),
        '--launch-json', str(server.folder/'launch.json'),
        '--depth', server.depth, '--port', str(PORT), '--slots', str(SLOTS), '--active', str(active),
        '--context', str(context), '--tokens', str(tokens), '--deadline', str(a.deadline)]
    limit = 600 if context <= 4096 else 2400
    capture = None
    cancelled = threading.Event()
    if server.session:
        capture = threading.Thread(target=profile_decode, args=(server, name, active, cancelled), daemon=True)
        capture.start()
    with (server.folder/f'{name}.out').open('x') as f:
        try:
            r = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, timeout=max(1, min(limit, a.deadline-time.time())))
        except subprocess.TimeoutExpired:
            event('workload_timeout', server=server.label, active=active, context=context, limit=limit)
            raise RuntimeError(f'{name} exceeded {limit} s; the server is treated as wedged')
        finally:
            cancelled.set()
            if capture:
                capture.join(timeout=max(1, min(300, a.deadline-time.time())))
    path = out/name/'run.json'
    result = json.loads(path.read_text()) if path.exists() else {}
    profile_path = server.folder/f'{name}-profile.json'
    profile_record = json.loads(profile_path.read_text()) if profile_path.exists() and not (capture and capture.is_alive()) else None
    evidence = profile_evidence(bool(server.session), bool(capture and capture.is_alive()), profile_record)
    row = dict(name=name, active=active, context=context, exit=r.returncode, passed=r.returncode == 0 and result.get('passed') is True,
        profile=evidence,
        error=result.get('error'), per_stream_tps=result.get('median_stream_tokens_per_second'),
        aggregate_tps=(result.get('common_decode') or {}).get('tokens_per_second'),
        wall_tps=(result.get('wall') or {}).get('tokens_per_second'),
        initial_prompt_tokens=result.get('initial_prompt_tokens'), decode_prompt_tokens=result.get('decode_prompt_tokens'),
        warmup_seconds=result.get('warmup_seconds'), warmup_generated_tokens=result.get('warmup_generated_tokens'),
        drafted=result.get('draft_n'), accepted=result.get('draft_n_accepted'), minimum_free_mib=list(server.minimum))
    if server.failure or server.proc.poll() is not None:
        row.update(passed=False, error=server.failure or f'server exited ({server.proc.returncode})')
    write(server.folder/f'{name}-summary.json', row)
    event('workload_done', server=server.label, **row)
    return row


try:
    for ordinal, (engine, depth, short, long, *rest) in enumerate(PLAN):
        extra = rest[0] if rest else None
        done = engines.install_pending(root)
        if done:
            event('engines_installed', labels=done)
        if not engines.installed(root, engine):
            report['servers'].append(dict(engine=engine, depth=depth, skipped='engine not installed'))
            continue
        required_seconds = serving_window_seconds(bool((extra or {}).get('profile')))
        remaining_seconds = a.deadline - time.time()
        if remaining_seconds < required_seconds:
            report['servers'].append(dict(engine=engine, depth=depth, skipped='window',
                                          required_seconds=required_seconds, remaining_seconds=remaining_seconds))
            continue
        server = Server(engine, depth, ordinal, extra)
        entry = dict(label=server.label, engine=engine, depth=depth, runs=[])
        report['servers'].append(entry)
        try:
            server.ready()
            entry.update(ready_seconds=server.ready_seconds, short_ok=server.short_ok)
            legs = [(active, context, 256, 0) for active, context in short]
            legs += [(active, context, 128, 1500) for active, context in long]
            for active, context, tokens, reserve in legs:
                if a.deadline - time.time() < reserve:
                    entry['runs'].append(dict(active=active, context=context, skipped='window'))
                    continue
                row = workload(server, active, context, tokens=tokens)
                entry['runs'].append(row)
                if not row['passed'] or row['profile']['state'] in ('start_failed', 'invalid'):
                    raise RuntimeError(f'{row["name"]} failed; see workload summary and profile state')
        except Exception as exc:
            entry['error'] = f'{type(exc).__name__}: {exc}'
            event('server_failed', server=server.label, error=entry['error'])
        finally:
            server.close()
            entry['minimum_free_mib'] = server.minimum
            entry['guard'] = server.failure
            entry['exit'] = server.proc.returncode
            write(out/f'matrix-{a.tag}.json', report)
except Exception as exc:
    report['error'] = f'{type(exc).__name__}: {exc}'
finally:
    report['passed'] = matrix_succeeded(report)
    report['finished_epoch'] = time.time()
    write(out/f'matrix-{a.tag}.json', report)
    (out/f'gpu-work-complete-{a.tag}').write_text('Measurement ended; collect and destroy this rental.\n')
    event('gpu_work_complete', passed=report['passed'], error=report.get('error'))

raise SystemExit(0 if report['passed'] else 1)
