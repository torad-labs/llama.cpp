"""Goal: recover real scheduling evidence without changing its identity or clock."""
import json
import sqlite3
import tempfile
import random
import decode_trace
import unittest

import decode_lab
import test_decode_lab as fixtures


class TraceRecoveryTest(unittest.TestCase):
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
