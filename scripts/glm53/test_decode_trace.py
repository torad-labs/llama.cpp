"""Goal: recover real scheduling evidence without changing its identity or clock."""
import json
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


if __name__ == '__main__':
    unittest.main()
