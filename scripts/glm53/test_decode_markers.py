"""Goal: opt-in host markers preserve nesting and never expose token contents."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import serve

ROOT = Path(__file__).resolve().parents[2]


class HostMarkersTest(unittest.TestCase):
    def test_actual_cpp_emitter_is_quiet_by_default_and_balanced_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'probe.cpp'
            binary=Path(tmp)/'probe'
            source.write_text(r'''
#include "common/decode-trace.h"
int main() {
    common_decode_trace::range cycle("stage=iteration iteration=7");
    { common_decode_trace::range phase("stage=proposal proposal=1 rows=1"); }
    common_decode_trace::mark("stage=accept sequence=0 drafted=2 accepted=1");
}
''')
            built=subprocess.run(['c++','-std=c++17','-I',str(ROOT),str(source),'-o',str(binary)],
                                 capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            env=dict(os.environ);env.pop('LLAMA_DECODE_TRACE',None)
            quiet=subprocess.run([str(binary)],env=env,capture_output=True,text=True)
            self.assertEqual(quiet.stderr,'')
            env['LLAMA_DECODE_TRACE']='1'
            traced=subprocess.run([str(binary)],env=env,capture_output=True,text=True)
            records=[line for line in traced.stderr.splitlines() if line.startswith('GLM53_TRACE ')]
            self.assertEqual(len(records),5)
            self.assertIn('event=begin',records[0]);self.assertIn('iteration=7',records[0])
            self.assertIn('stage=proposal',records[1]);self.assertIn('event=end',records[2])
            self.assertIn('event=mark',records[3]);self.assertIn('accepted=1',records[3])
            self.assertIn('event=end',records[4])
            self.assertTrue(all('clock=host-monotonic' in line and 'nvtx=' in line for line in records))

    def test_launcher_explicitly_preserves_requested_diagnostics_only(self):
        normal=serve.arguments(['--model','/tmp/not-loaded.gguf','--profile','dp-4','--slots','1'])
        argv,base=serve.make_plan(normal)
        self.assertNotIn('LLAMA_DECODE_TRACE',base)
        self.assertNotIn('-lv',argv)
        traced=serve.arguments(['--model','/tmp/not-loaded.gguf','--profile','dp-4','--slots','1','--trace-decode'])
        argv,env=serve.make_plan(traced)
        self.assertEqual(env['LLAMA_DECODE_TRACE'],'1')
        self.assertEqual(env['GGML_CUDA_NVTX'],'1')
        self.assertEqual(env['GGML_GRAPH_TRACE'],'1')
        self.assertEqual(env['LLAMA_TIME_DECODE'],'1')
        # ggml INFO is routed through common's TRACE level (4), above the server default (3).
        self.assertIn('-lv',argv)
        self.assertEqual(argv[argv.index('-lv')+1],'4')


if __name__ == '__main__':
    unittest.main()
