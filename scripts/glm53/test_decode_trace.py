"""Goal: recover real scheduling evidence without changing its identity or clock."""
import json
import os
import contextlib
import sqlite3
import tempfile
import random
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import decode_trace
import measure
import unittest

import decode_lab
import test_decode_lab as fixtures


def decoded_capture_database(rows=36, batch_markers=True):
    db = sqlite3.connect(':memory:')
    db.executescript(f'''
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER,correlationId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER);
        CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER);
        INSERT INTO NVTX_EVENTS VALUES (100,200,16777217,'glm53.v1 stage=target_eval rows={rows}',NULL);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (120,121,16777217,7);
    ''')
    for device in range(4):
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (130,160,?,16777216,7)', (device,))
    if batch_markers:
        add_decode_batch_markers(db, rows)
    return db


def add_decode_batch_markers(db, count):
    for sequence in range(count):
        label = f'glm53.v1 stage=batch sequence={sequence} rows=1 prompt_rows=0'
        db.execute('INSERT INTO NVTX_EVENTS VALUES (105,NULL,16777217,?,NULL)', (label,))


def write_decoded_capture_sqlite(path):
    with contextlib.closing(decoded_capture_database()) as source:
        source.commit()
        with sqlite3.connect(path) as target:
            source.backup(target)


class TraceRecoveryTest(unittest.TestCase):
    def test_real_cuda_graph_updates_remain_traceable(self):
        binary = os.environ.get("GLM53_PROFILE_GRAPH_BINARY")
        if not binary:
            self.skipTest("set GLM53_PROFILE_GRAPH_BINARY and GLM53_PROFILE_TEST_OUTPUT for the GPU/Nsight integration")
        output = Path(os.environ["GLM53_PROFILE_TEST_OUTPUT"]).resolve()
        output.mkdir(parents=True, exist_ok=False)
        capture = output / "graph-updates"
        env = {key: os.environ[key] for key in ("PATH", "HOME", "LD_LIBRARY_PATH",
               "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER") if key in os.environ}
        with (output / "profile.log").open("x") as log:
            run = subprocess.run(["nsys", "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node",
                                  "--sample=none", "--cpuctxsw=none", "--output=" + str(capture), binary],
                                 env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        self.assertEqual(run.returncode, 0, "graph numerical check or profiling failed; see profile.log")
        with (output / "export.log").open("x") as log:
            exported = subprocess.run(["nsys", "export", "--type", "sqlite", "--output",
                                       str(capture) + ".sqlite", str(capture) + ".nsys-rep"],
                                      env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        self.assertEqual(exported.returncode, 0, "see export.log")
        with contextlib.closing(sqlite3.connect(str(capture) + ".sqlite")) as db:
            self.assertGreater(db.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0], 0)
            errors = db.execute("SELECT text,COUNT(*) FROM DIAGNOSTIC_EVENT WHERE severity>=3 GROUP BY text").fetchall()
        self.assertEqual(errors, [], "actual graph updates must remain traceable without collector errors")

    def test_decode_window_follows_same_process_worker_launches(self):
        rng = random.Random(59921522)
        for trial in range(24):
            with self.subTest(trial=trial), contextlib.closing(decoded_capture_database()) as db:
                pid = (rng.randrange(1, 65536) << 24) | (rng.randrange(2) << 48)
                main_tid = pid + 1
                db.execute('UPDATE NVTX_EVENTS SET globalTid=?', (main_tid,))
                db.execute('DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME')
                db.execute('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL')
                correlations = rng.sample(range(10, 10000), 4)
                intervals = []
                for device, correlation in enumerate(correlations):
                    tid = main_tid if device == trial % 5 else pid + rng.randrange(2, 100000)
                    launch = rng.randrange(110, 125)
                    start, end = rng.randrange(130, 150), rng.randrange(160, 240)
                    intervals.append((start, end))
                    db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)',
                               (launch, launch + 1, tid, correlation))
                    db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?)',
                               (start, end, device, pid, correlation))
                result = decode_trace.capture_readiness(db, range(4), expected_sequences=36)
                self.assertTrue(result['usable'], result)
                self.assertEqual(result['actual_sequences'], 36)
                start, end = min(x[0] for x in intervals), max(x[1] for x in intervals)
                self.assertEqual(result['gpu_window'], dict(start_ns=start, end_ns=end,
                                                           duration_ns=end-start, kernel_count=4))
                correlation = correlations[rng.randrange(4)]
                invalid_updates = [
                    ('UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET globalTid=? WHERE correlationId=?',
                     (main_tid + (1 << 24), correlation)),
                    ('UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET start=90,end=91 WHERE correlationId=?', (correlation,)),
                    ('UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET start=201,end=202 WHERE correlationId=?', (correlation,)),
                    ('UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET start=190,end=201 WHERE correlationId=?', (correlation,)),
                    ('UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET end=start WHERE correlationId=?', (correlation,)),
                    ('DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=?', (correlation,)),
                    ('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start=105 WHERE correlationId=?', (correlation,)),
                    ('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET globalPid=? WHERE correlationId=?', (pid + (1 << 24), correlation)),
                    ('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=start WHERE correlationId=?', (correlation,)),
                ]
                for sql, params in invalid_updates:
                    with self.subTest(sql=sql):
                        db.execute('SAVEPOINT invalid_launch')
                        db.execute(sql, params)
                        self.assertFalse(decode_trace.capture_readiness(db, range(4), expected_sequences=36)['usable'])
                        db.execute('ROLLBACK TO invalid_launch')
                        db.execute('RELEASE invalid_launch')

    def test_capture_counts_decoding_sequences_not_tensor_rows(self):
        rng = random.Random(2408)
        for widths in ([2] * 36, [3] * 36, [1] * 36,
                       [rng.randrange(1, 4) for _ in range(36)]):
            with self.subTest(widths=widths):
                db = decoded_capture_database(rows=sum(widths), batch_markers=False)
                identities = rng.sample(range(400), 36)
                for sequence, width in zip(identities, widths):
                    label = f'glm53.v1 stage=batch sequence={sequence} rows={width} prompt_rows=0 position_min=528 position_max={527 + width}'
                    db.execute('INSERT INTO NVTX_EVENTS VALUES (105,NULL,16777217,?,NULL)', (label,))
                result = decode_trace.capture_readiness(db, range(4), expected_sequences=36)
                self.assertTrue(result['usable'], result)
                self.assertEqual(result['actual_sequences'], 36)
                self.assertEqual(result['actual_rows'], sum(widths))
                db.close()

    def test_capture_rejects_profiler_failures_for_the_measured_process(self):
        rng = random.Random(9134)
        diagnostics = [
            (3, "GetGraphId returned CUPTI_ERROR_INVALID_PARAMETER"),
            (2, "Installed CUDA driver version (13.4) is not supported by this build of Nsight Systems."),
            (2, "Not all CUDA events might have been collected."),
            (2, "Not all NVTX events might have been collected."),
        ]
        for _ in range(24):
            severity, message = rng.choice(diagnostics)
            pid = rng.choice([16777216, 33554432, None])
            with self.subTest(severity=severity, message=message, pid=pid):
                with contextlib.closing(decoded_capture_database()) as db:
                    db.execute("CREATE TABLE DIAGNOSTIC_EVENT(timestamp INTEGER,severity INTEGER,text TEXT,globalPid INTEGER)")
                    db.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES (?,?,?,?)",
                               (rng.randrange(-1000, 1000), severity, message, pid))
                    result = decode_trace.capture_readiness(db, range(4), expected_sequences=36)
                    self.assertEqual(result["usable"], pid == 33554432, result)

    def test_capture_rejects_unproved_or_prefill_cohorts(self):
        for case in ('missing', 'duplicate', 'prefill', 'outside-phase', 'wrong-thread', 'incomplete-rows'):
            with self.subTest(case=case):
                db = decoded_capture_database(batch_markers=False)
                if case != 'missing':
                    for sequence in range(36):
                        identity = 0 if case == 'duplicate' else sequence
                        prompt = 1 if case == 'prefill' and sequence == 0 else 0
                        at = 90 if case == 'outside-phase' else 105
                        tid = 16777218 if case == 'wrong-thread' else 16777217
                        if case == 'incomplete-rows' and sequence == 0:
                            continue
                        label = f'glm53.v1 stage=batch sequence={identity} rows=1 prompt_rows={prompt}'
                        db.execute('INSERT INTO NVTX_EVENTS VALUES (?,NULL,?,?,NULL)', (at, tid, label))
                self.assertFalse(decode_trace.capture_readiness(db, range(4), expected_sequences=36)['usable'])
                db.close()

    def test_capture_requires_an_actual_36_session_gpu_decode_window(self):
        db = decoded_capture_database()
        result = decode_trace.capture_readiness(db, range(4), expected_sequences=36)
        self.assertTrue(result['usable'])
        self.assertEqual(result['actual_rows'], 36)
        self.assertEqual(result['gpu_window'], {'start_ns': 130, 'end_ns': 160, 'duration_ns': 30, 'kernel_count': 4})
        self.assertNotIn('host_start_ns', result['gpu_window'])
        for name in ('wrong-active-count', 'wrong-device', 'wrong-process', 'wrong-launch-time'):
            with self.subTest(name=name):
                probe = decoded_capture_database()
                if name == 'wrong-active-count':
                    probe.execute("UPDATE NVTX_EVENTS SET text='glm53.v1 stage=target_eval rows=35'")
                elif name == 'wrong-device':
                    probe.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET deviceId=4 WHERE deviceId=3')
                elif name == 'wrong-process':
                    probe.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET globalPid=33554432 WHERE deviceId=3')
                else:
                    probe.execute('UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET start=90')
                failed = decode_trace.capture_readiness(probe, range(4), expected_sequences=36)
                self.assertFalse(failed['usable'])
                self.assertIsNone(failed['gpu_window'])
                probe.close()
        db.close()

    def test_runtime_only_export_is_not_a_model_capture(self):
        db = sqlite3.connect(':memory:')
        db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER)')
        result = decode_trace.capture_readiness(db, range(4))
        self.assertFalse(result['usable'])
        self.assertIn('CUPTI_ACTIVITY_KIND_KERNEL', result['missing_tables'])
        db.close()

    def test_startup_kernels_do_not_fill_a_missing_decode_device(self):
        db = sqlite3.connect(':memory:')
        db.executescript('''
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER,correlationId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER);
            CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER);
            INSERT INTO NVTX_EVENTS VALUES (100,190,16777217,'glm53.v1 stage=target_eval rows=36',NULL);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (110,111,16777217,7);
        ''')
        add_decode_batch_markers(db, 36)
        for device in range(4):
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (10,90,?,16777216,7)', (device,))
        for device in range(3):
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (120,200,?,16777216,7)', (device,))
        self.assertFalse(decode_trace.capture_readiness(db, range(4), expected_sequences=36)['usable'])
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (120,200,3,16777216,7)')
        self.assertTrue(decode_trace.capture_readiness(db, range(4), expected_sequences=36)['usable'])
        db.close()

    def test_aborted_application_can_finish_export_before_profiler_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'concurrency.v2'
            report = output.with_name(output.name+'.nsys-rep')
            report.write_bytes(b'partial')
            # Stand-in exporter: the application has failed, but the profiler is still writing its report.
            child = subprocess.Popen([sys.executable, '-c',
                'import pathlib,sys,time; time.sleep(0.1); pathlib.Path(sys.argv[1]).write_bytes(b"report")',
                str(report)])
            try:
                with patch.object(measure.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1,
                        stdout='', stderr='Collection stop is not allowed in this state.')) as stop:
                    result = measure.finish_nsys_capture(child, 'owned-test', output, time.time()+5)
                self.assertEqual(child.poll(), 0)
                self.assertTrue(result['report_present'])
                self.assertFalse(result['capture_verified'])
                self.assertEqual(stop.call_count, 1)
                self.assertEqual(stop.call_args.args[0], ['nsys','stop','--session=owned-test'])
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait()

    def test_profile_finalization_never_waits_beyond_the_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'concurrency'
            with patch.object(measure.subprocess, 'run') as stop:
                result = measure.finish_nsys_capture(None, 'owned-test', output, time.time()-1)
            stop.assert_not_called()
            self.assertFalse(result['report_present'])
            self.assertFalse(result['capture_verified'])
            self.assertTrue(result['deadline_reached'])

    def test_export_retains_raw_report_versions_and_verified_gpu_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'decode'
            report = output.with_name(output.name + '.nsys-rep')
            report.write_bytes(b'raw-nsys-report')
            sqlite = output.with_name(output.name + '-export.sqlite')
            request = {'active_sessions': 36, 'launch_sha256': 'launch',
                       'capture_nsys': {'exit': 0, 'version': 'NVIDIA Nsight Systems version 2026.1.3'}}
            def run(argv, **kwargs):
                if argv[1:] == ['--version']:
                    return subprocess.CompletedProcess(argv, 0, 'NVIDIA Nsight Systems version 2026.1.3\n', '')
                self.assertEqual(argv[:4], ['nsys', 'export', '--type', 'sqlite'])
                write_decoded_capture_sqlite(sqlite)
                return subprocess.CompletedProcess(argv, 0, 'exported', '')
            with patch.object(measure.subprocess, 'run', side_effect=run):
                result = measure.export_nsys_capture(output, range(4), request, time.time() + 10)
            self.assertTrue(result['capture_verified'])
            self.assertEqual(result['launch_sha256'], 'launch')
            self.assertEqual(result['readiness']['gpu_window']['duration_ns'], 30)
            self.assertEqual(result['exporter']['compatibility']['state'], 'exact-version')
            self.assertEqual(Path(result['report']).read_bytes(), b'raw-nsys-report')
            self.assertTrue(Path(result['sqlite']).is_file())
            receipt = json.loads(output.with_name(output.name + '-export.json').read_text())
            self.assertEqual(receipt['export']['stdout'], 'exported')

    def test_export_failure_keeps_the_raw_report_and_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'decode'
            report = output.with_name(output.name + '.nsys-rep')
            report.write_bytes(b'raw-nsys-report')
            request = {'active_sessions': 36, 'launch_sha256': 'launch',
                       'capture_nsys': {'exit': 0, 'version': 'NVIDIA Nsight Systems version 2026.1.3'}}
            with patch.object(measure.subprocess, 'run', side_effect=[
                    subprocess.CompletedProcess([], 0, 'NVIDIA Nsight Systems version 2026.1.3\n', ''),
                    subprocess.CompletedProcess([], 7, '', 'cannot export')]):
                result = measure.export_nsys_capture(output, range(4), request, time.time() + 10)
            self.assertFalse(result['capture_verified'])
            self.assertEqual(result['failure'], 'sqlite_export_failed')
            self.assertEqual(report.read_bytes(), b'raw-nsys-report')
            receipt = json.loads(output.with_name(output.name + '-export.json').read_text())
            self.assertEqual(receipt['export']['stderr'], 'cannot export')

    def test_generated_marker_metadata_round_trips_but_free_text_never_does(self):
        rng=random.Random(53)
        fields=sorted(decode_trace.MARKER_FIELDS)
        stages=sorted(decode_trace.MARKER_STAGES)
        for _ in range(200):
            expected={'stage':rng.choice(stages)}
            expected.update({key:rng.randrange(1000000) for key in rng.sample(fields,rng.randrange(5))})
            label='glm53.v1 '+' '.join(str(k)+'='+str(v) for k,v in expected.items())
            self.assertEqual(decode_trace.parse_marker(label),expected)
            self.assertIsNone(decode_trace.parse_marker(label+' private=not-exported'))
            self.assertIsNone(decode_trace.parse_marker(label+' stage=proposal'))

    def test_rich_trace_retains_streams_endpoints_events_and_process_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = fixtures.ProfileEvidenceTest().fixture(tmp)
            db = sqlite3.connect(path)
            for table in ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_MEMCPY'):
                for column in ('contextId', 'streamId'):
                    db.execute(f'ALTER TABLE {table} ADD COLUMN {column} INTEGER')
            db.execute('ALTER TABLE CUPTI_ACTIVITY_KIND_MEMCPY ADD COLUMN srcDeviceId INTEGER')
            db.execute('ALTER TABLE CUPTI_ACTIVITY_KIND_MEMCPY ADD COLUMN dstDeviceId INTEGER')
            db.execute('ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN graphId INTEGER')
            db.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET contextId=4, streamId=9, graphId=88')
            db.execute('UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET contextId=4, streamId=10, srcDeviceId=0, dstDeviceId=1')
            db.executescript("""
                CREATE TABLE CUPTI_ACTIVITY_KIND_CUDA_EVENT(timestamp INTEGER,deviceId INTEGER,
                    contextId INTEGER,streamId INTEGER,globalPid INTEGER,correlationId INTEGER,eventId INTEGER,eventSyncId INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION(start INTEGER,end INTEGER,deviceId INTEGER,
                    contextId INTEGER,streamId INTEGER,globalPid INTEGER,correlationId INTEGER,eventId INTEGER,eventSyncId INTEGER,syncType INTEGER);
                INSERT INTO CUPTI_ACTIVITY_KIND_CUDA_EVENT VALUES (390,0,4,9,16777216,7,15,16);
                INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION VALUES (410,450,1,4,10,33554432,7,15,16,1);
                CREATE TABLE PRIVATE_ENVIRONMENT(secret TEXT);
                INSERT INTO PRIVATE_ENVIRONMENT VALUES ('do-not-export');
            """)
            db.commit()
            db.close()
            profile = decode_lab.export_trace(path, {}, include_trace=True)
        trace = profile['trace']
        self.assertEqual(trace['schema'], 'glm53-cuda-trace-v1')
        self.assertEqual(trace['origin_ns'], 400)
        self.assertEqual(trace['kernels'][0]['start'], 0)
        self.assertEqual(trace['kernels'][0]['streamId'], 9)
        self.assertEqual(trace['kernels'][0]['graphId'], 88)
        self.assertEqual(trace['copies'][0]['srcDeviceId'], 0)
        self.assertEqual(trace['copies'][0]['dstDeviceId'], 1)
        self.assertEqual(trace['cuda_events'][0]['timestamp'], -10)
        self.assertEqual(trace['synchronizations'][0]['start'], 10)
        self.assertNotEqual(trace['cuda_events'][0]['globalPid'], trace['synchronizations'][0]['globalPid'])
        self.assertEqual(trace['cuda_events'][0]['correlationId'], trace['synchronizations'][0]['correlationId'])
        self.assertEqual(len(trace['kernels']), profile['counts']['kernels'])
        self.assertEqual(len(trace['copies']), profile['counts']['copies'])
        self.assertNotIn('do-not-export', json.dumps(profile))
        self.assertNotIn('PRIVATE_ENVIRONMENT', json.dumps(profile))

    def test_semantic_markers_identify_launches_not_gpu_timestamp_containment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=fixtures.ProfileEvidenceTest().fixture(tmp)
            db=sqlite3.connect(path)
            db.executescript("""
                INSERT INTO NVTX_EVENTS VALUES
                  (280,350,16777217,'glm53.v1 stage=iteration iteration=7',NULL),
                  (290,330,16777217,'glm53.v1 stage=proposal proposal=1 rows=1',NULL),
                  (295,NULL,16777217,'glm53.v1 stage=batch sequence=0 rows=1 prompt_rows=0 position_min=399999 position_max=399999',NULL),
                  (300,NULL,16777217,'glm53.v1 stage=batch prompt=private-text',NULL);
            """)
            db.commit();db.close()
            profile=decode_lab.export_trace(path,{},include_trace=True)
        trace=profile['trace']
        self.assertEqual(len(trace['semantic_markers']),3)
        self.assertEqual(trace['evaluations'][0]['iteration'],7)
        self.assertEqual(trace['evaluations'][0]['stage'],'proposal')
        # GPU events begin after the host range ended; their launch is inside it.
        self.assertEqual(trace['evaluations'][0]['kernel_indices'],[0,1])
        self.assertEqual(trace['semantic_coverage']['assigned_kernels'],2)
        self.assertNotIn('private-text',json.dumps(trace))

    def test_legacy_missing_fields_remain_unknown_not_fabricated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = fixtures.ProfileEvidenceTest().fixture(tmp)
            profile = decode_lab.export_trace(path, {}, include_trace=True)
        trace = profile['trace']
        self.assertEqual(len(trace['kernels']), 3)
        self.assertIsNone(trace['kernels'][0]['streamId'])
        self.assertIsNone(trace['copies'][0]['dstDeviceId'])
        self.assertEqual(trace['cuda_events'], [])
        self.assertIn('CUPTI_ACTIVITY_KIND_CUDA_EVENT', trace['missing_tables'])
        self.assertEqual(trace['clock'], 'relative-nanoseconds')


class DecodeReportTest(unittest.TestCase):
    def fixture(self, folder):
        import hashlib
        root = Path(folder)
        launch = root/"launch.json"
        launch.write_text(json.dumps(dict(
            argv=["/workspace/engines/B/bin/llama-server"],
            env={"GGML_CUDA_ALLREDUCE": "internal-nccl-gather", "CUDA_VISIBLE_DEVICES": "0,1",
                 "PRIVATE_SECRET": "do-not-export"},
            source="a"*40)))
        run = root/"run.json"
        run.write_text(json.dumps(dict(active_sessions=2, context_tokens=512,
            initial_prompt_tokens=512, decode_prompt_tokens=[528, 528],
            warmup_tokens_per_session=[16, 16], warmup_generated_tokens=32,
            decode_tokens=256, draft_n=0, draft_n_accepted=0,
            started_epoch=1000, finished_epoch=1001,
            decode_started_epoch=1000.01, decode_finished_epoch=1000.02,
            launch_sha256=hashlib.sha256(launch.read_bytes()).hexdigest())))
        db = root/"trace.sqlite"
        with contextlib.closing(sqlite3.connect(db)) as con, con:
            con.executescript("""
                CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER);
                CREATE TABLE StringIds(id INTEGER,value TEXT);
                CREATE TABLE PROCESSES(globalPid INTEGER,pid INTEGER,name TEXT);
                INSERT INTO PROCESSES VALUES (16777216,1,'/workspace/engines/B/bin/llama-server');
                INSERT INTO PROCESSES VALUES (33554432,2,'/workspace/other/bin/llama-server');
                CREATE TABLE TARGET_INFO_SESSION_START_TIME(utcEpochNs INTEGER);
                INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (1000000000000);
                CREATE TABLE TARGET_INFO_GPU(id INTEGER,name TEXT);
                INSERT INTO TARGET_INFO_GPU VALUES (0,'fixture'),(1,'fixture');
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,
                    globalPid INTEGER,shortName INTEGER,gridX INTEGER,gridY INTEGER,gridZ INTEGER,
                    blockX INTEGER,blockY INTEGER,blockZ INTEGER,registersPerThread INTEGER,
                    staticSharedMemory INTEGER,dynamicSharedMemory INTEGER,graphNodeId INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET(start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER);
            """)
            for pid in (16777216, 33554432):
                for iteration in range(3):
                    con.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,NULL)",
                        (iteration*10000000, (iteration+1)*10000000, pid+1,
                         f"glm53.v1 stage=iteration iteration={iteration}"))
                for seq in range(2):
                    con.execute("INSERT INTO NVTX_EVENTS VALUES (?,NULL,?,?,NULL)",
                        (19000000, pid+1, f"glm53.v1 stage=accept sequence={seq} accepted=0"))
            con.execute("INSERT INTO NVTX_EVENTS VALUES (11000000,18000000,16777217,?,NULL)",
                        ("glm53.v1 stage=target_eval rows=2",))
            for idx, name in enumerate(("ncclDevKernel_AllReduce_u32_RING",
                                       "ncclDevKernel_AllReduce_bf16_RING",
                                       "ggml_cuda_ar_owner_reduce_kernel_n", "foreign_kernel"), 1):
                con.execute("INSERT INTO StringIds VALUES (?,?)", (idx,name))
                for device in range(2):
                    con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (12000000, 14000000, device, 33554432 if idx==4 else 16777216,
                         idx,1,1,1,128,1,1,32,0,0,idx))
        return root, db, run, launch

    def report(self, root, db, run, launch):
        output = root/"report.txt"
        result = subprocess.run([sys.executable, "-I", str(Path(__file__).with_name("decode_report.py")),
            "--db", str(db), "--run-json", str(run), "--launch-json", str(launch),
            "--output", str(output)], capture_output=True, text=True, timeout=15)
        return result, output

    def test_report_binds_process_and_time_and_includes_native_collectives_without_claiming_operations(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(tmp)
            result, output = self.report(*paths)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = output.read_text()
        self.assertIn("2-session", report)
        self.assertIn("512 initial prompt tokens", report)
        self.assertIn("528..528", report)
        self.assertIn("warmup evidence: 16..16 tokens/session; 32 generated tokens", report)
        self.assertIn("process/time-correlated, not unique workload identity", report)
        self.assertIn("launch digest: matched", report)
        self.assertIn("trace process: matched", report)
        self.assertIn("trace time: matched measured decode", report)
        self.assertIn("graph-node kernels: 6/6", report)
        collectives = report.split("Collectives\n")[1]
        self.assertIn("ncclDevKernel_AllReduce_u32_RING", collectives)
        self.assertIn("ncclDevKernel_AllReduce_bf16_RING", collectives)
        self.assertIn("ggml_cuda_ar_owner_reduce_kernel_n", collectives)
        self.assertIn("ordinal", collectives)
        self.assertIn("not proven logical operations", collectives)
        self.assertNotIn("foreign_kernel", report)
        self.assertNotIn("do-not-export", report)
        self.assertNotIn("below", report)
        self.assertNotIn("logical 4-GPU ops", report)

    def test_mismatched_launch_process_or_clock_cannot_produce_a_report(self):
        for mismatch in ("launch", "process", "clock", "devices"):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as tmp:
                root, db, run, launch = self.fixture(tmp)
                if mismatch == "launch":
                    launch.write_text(launch.read_text()+" ")
                else:
                    with contextlib.closing(sqlite3.connect(db)) as con, con:
                        if mismatch == "process":
                            con.execute("UPDATE PROCESSES SET name='/workspace/wrong/llama-server'")
                        elif mismatch == "clock":
                            con.execute("UPDATE TARGET_INFO_SESSION_START_TIME SET utcEpochNs=2000000000000")
                        else:
                            con.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId=1")
                result, output = self.report(root, db, run, launch)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(mismatch, result.stderr.lower())
                self.assertFalse(output.exists())

    def test_current_prompt_handoff_requires_complete_consistent_warmup(self):
        for bad in (None, [16], [16, 15], [16, 17]):
            with self.subTest(warmup=bad), tempfile.TemporaryDirectory() as tmp:
                root, db, run, launch = self.fixture(tmp)
                metadata = json.loads(run.read_text())
                if bad is None:
                    del metadata["warmup_tokens_per_session"]
                else:
                    metadata["warmup_tokens_per_session"] = bad
                run.write_text(json.dumps(metadata))
                result, output = self.report(root, db, run, launch)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("warmup", result.stderr.lower())
                self.assertFalse(output.exists())

    def test_legacy_missing_evidence_is_explicit_and_never_invented(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, db, run, launch = self.fixture(tmp)
            metadata = json.loads(run.read_text())
            for key in ("launch_sha256", "initial_prompt_tokens", "decode_prompt_tokens",
                        "decode_started_epoch", "decode_finished_epoch",
                        "warmup_tokens_per_session", "warmup_generated_tokens"):
                del metadata[key]
            run.write_text(json.dumps(metadata))
            with contextlib.closing(sqlite3.connect(db)) as con, con:
                con.execute("DROP TABLE TARGET_INFO_SESSION_START_TIME")
                con.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL DROP COLUMN graphNodeId")
            result, output = self.report(root, db, run, launch)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = output.read_text()
        self.assertIn("launch digest: unavailable", report)
        self.assertIn("trace time: unavailable", report)
        self.assertIn("warmup evidence: unavailable", report)
        self.assertIn("initial prompt: unavailable", report)
        self.assertIn("legacy reported context: 512", report)
        self.assertNotIn("512 initial prompt tokens", report)
        self.assertIn("effective prompt: unavailable", report)
        self.assertIn("graph-node kernels: unavailable", report)
        self.assertNotIn("528", report)


if __name__ == '__main__':
    unittest.main()
