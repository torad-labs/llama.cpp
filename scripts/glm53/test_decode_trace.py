"""Goal: recover real scheduling evidence without changing its identity or clock."""
import json
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


class TraceRecoveryTest(unittest.TestCase):
    def test_capture_requires_workload_markers_and_kernels_on_every_expected_device(self):
        db = sqlite3.connect(':memory:')
        db.executescript('''
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER);
            CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER);
            CREATE TABLE StringIds(id INTEGER,value TEXT);
            INSERT INTO StringIds VALUES (1,'glm53.v1 stage=target_eval rows=36');
            INSERT INTO NVTX_EVENTS VALUES (1,90,16777217,NULL,1);
        ''')
        for device in range(4):
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (10,100,?,16777216)',(device,))
        self.assertTrue(decode_trace.capture_readiness(db, range(4))['usable'])
        db.execute('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId=3')
        self.assertFalse(decode_trace.capture_readiness(db, range(4))['usable'])
        # An unrelated preflight kernel on device 3 does not complete the model's capture.
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (10,100,3,33554432)')
        self.assertFalse(decode_trace.capture_readiness(db, range(4))['usable'])
        db.execute('DELETE FROM NVTX_EVENTS')
        self.assertFalse(decode_trace.capture_readiness(db, range(4))['usable'])
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
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER);
            CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER);
            INSERT INTO NVTX_EVENTS VALUES (100,190,16777217,'glm53.v1 stage=target_eval rows=36',NULL);
        ''')
        for device in range(4):
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (10,90,?,16777216)', (device,))
        for device in range(3):
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (110,200,?,16777216)', (device,))
        self.assertFalse(decode_trace.capture_readiness(db, range(4))['usable'])
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (110,200,3,16777216)')
        self.assertTrue(decode_trace.capture_readiness(db, range(4))['usable'])
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
