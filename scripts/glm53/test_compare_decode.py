"""Goal: vary load on one fixed 40-slot/four-GPU setup, then compare the same work."""
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from typing import Any

import compare_decode as comparison
import serve

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / 'scripts/glm53/compare_decode.py'


def step(identity, device, duration, deps=(), **extra) -> dict[str, Any]:
    return dict(id=identity, cycle=0, phase='verify', layer=3, operation=identity, device=device,
                depends_on=list(deps), resources=['gpu:'+str(device)],
                work=dict(rows=1, sequence_ids=[0], context_tokens=[400000]),
                duration_ns=duration, **extra)


class PreparationTest(unittest.TestCase):
    def test_launcher_runs_legacy_and_current_engines_with_matching_dry_run_arguments(self):
        rng = random.Random(53)
        for supported in (False, True):
            for spelling in ("--prompt-chunk", "-pc, --prompt-chunk", "--prompt-chunk-extra"):
                enabled = supported and spelling != "--prompt-chunk-extra"
                with self.subTest(supported=enabled, spelling=spelling), tempfile.TemporaryDirectory() as tmp:
                    folder = Path(tmp)
                    model = folder / "model.gguf"
                    model.touch()
                    server = folder / "llama-server"
                    help_text = "  --prompt-step N\n" + (
                        " " * rng.randrange(1, 8) + spelling + " N\n" if supported else ""
                    )
                    server.write_text(
                        f"#!{sys.executable}\n"
                        "import json,sys\n"
                        f"if '--help' in sys.argv: print({help_text!r}); sys.exit(0)\n"
                        f"if '--prompt-chunk' in sys.argv and not {enabled!r}: sys.exit(2)\n"
                        "print(json.dumps(sys.argv))\n"
                    )
                    server.chmod(0o755)
                    command = [sys.executable, str(ROOT / "scripts/glm53/serve.py"),
                               "--model", str(model), "--server", str(server), "--profile", "dp-4"]
                    actual = subprocess.run(command, capture_output=True, text=True, timeout=30)
                    self.assertEqual(actual.returncode, 0, actual.stderr)
                    argv = json.loads(actual.stdout)
                    self.assertEqual("--prompt-chunk" in argv, enabled)
                    if enabled:
                        self.assertEqual(argv[argv.index("--prompt-chunk") + 1], "128")
                    for flag, value in (("--prompt-step", "512"), ("-np", "36"), ("-c", "15925248")):
                        self.assertEqual(argv[argv.index(flag) + 1], value)
                    planned = subprocess.run(command + ["--dry-run"], capture_output=True, text=True, timeout=30)
                    self.assertEqual(planned.returncode, 0, planned.stderr)
                    self.assertEqual(json.loads(planned.stdout)["argv"], argv)

    def test_launcher_refuses_failed_capability_detection_without_starting_the_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            model = folder / "model.gguf"
            model.touch()
            marker = folder / "started"
            server = folder / "llama-server"
            server.write_text(
                f"#!{sys.executable}\nimport pathlib,sys\n"
                "if '--help' in sys.argv: sys.exit(7)\n"
                f"pathlib.Path({str(marker)!r}).touch()\n"
            )
            server.chmod(0o755)
            result = subprocess.run([sys.executable, str(ROOT / "scripts/glm53/serve.py"),
                "--model", str(model), "--server", str(server), "--profile", "dp-4"],
                capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(marker.exists())
            self.assertIn("help", result.stderr)

    def test_one_session_is_still_four_gpus_and_uses_real_launcher(self):
        case = comparison.prepare_case(model='/tmp/not-loaded.gguf', server='/tmp/not-executed-server')
        self.assertEqual(case['workload']['active_sessions'], 1)
        self.assertEqual(case['workload']['context_tokens'], 400000)
        self.assertEqual(case['configuration']['gpu_count'], 4)
        self.assertEqual(case['configuration']['reserved_slots'], 40)
        self.assertEqual(case['configuration']['reserved_context_per_slot'], serve.DP_SLOT_CONTEXT)
        self.assertEqual(case['configuration']['depth_policy'], 'adaptive')
        args = serve.arguments(['--model', '/tmp/not-loaded.gguf', '--server', '/tmp/not-executed-server',
                                '--profile', 'dp-4', '--slots', '40', '--dry-run'])
        argv, env = serve.make_plan(args)
        self.assertEqual(case['resolved_launch']['argv'], argv)
        self.assertEqual(case['resolved_launch']['env'], env)
        self.assertFalse(case['gpu_validated'])
        for transport in ('internal', 'internal-nccl-gather'):
            with self.subTest(transport=transport):
                case = comparison.prepare_case(model='/tmp/not-loaded.gguf', transport=transport)
                self.assertEqual(case['configuration']['transport_requested'], transport)
                self.assertIsNone(case['configuration']['transport_actual'])
                self.assertEqual(case['resolved_launch']['env']['GGML_CUDA_ALLREDUCE'], transport)
                self.assertIn(transport, case['launcher']['options'])

    def test_active_load_does_not_change_the_serving_configuration(self):
        bindings = dict(model='/tmp/not-loaded.gguf', server='/tmp/not-executed-server')
        target = comparison.prepare_case(active=40, **bindings)
        self.assertEqual(target['configuration']['owners'], [10, 10, 10, 10])
        expected = {key: value for key, value in target['configuration'].items() if key != 'owners'}
        for active in range(1, 41):
            with self.subTest(active=active):
                case = comparison.prepare_case(active=active, **bindings)
                self.assertEqual(case['resolved_launch'], target['resolved_launch'])
                self.assertEqual(case['model'], target['model'])
                config = {key: value for key, value in case['configuration'].items() if key != 'owners'}
                self.assertEqual(config, expected)
                self.assertEqual(case['workload']['active_sessions'], active)
                owners = case['configuration']['owners']
                self.assertEqual(sum(owners), active)
                self.assertLessEqual(max(owners) - min(owners), 1)

    def test_unbound_model_is_explicit_and_reserved_is_not_active(self):
        case = comparison.prepare_case(slots=40)
        self.assertEqual(case['workload']['active_sessions'], 1)
        self.assertEqual(case['configuration']['reserved_slots'], 40)
        self.assertIsNone(case['resolved_launch'])
        self.assertIn('model', case['unresolved_bindings'])
        with self.assertRaisesRegex(ValueError, 'active'):
            comparison.prepare_case(active=4, slots=1)
        with self.assertRaisesRegex(ValueError, 'context'):
            comparison.prepare_case(context=serve.DP_SLOT_CONTEXT+1)

    def test_host_memory_budget_is_a_binding_not_an_invented_measurement(self):
        case=comparison.prepare_case(model='/tmp/not-loaded.gguf',kv='host')
        self.assertIsNone(case['resolved_launch'])
        self.assertIn('host_budget_mib',case['unresolved_bindings'])
        self.assertGreater(case['configuration']['minimum_host_kv_mib'],0)

    def test_real_cli_prepares_without_server_or_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'case.json'
            run = subprocess.run([sys.executable, str(CLI), 'prepare', '--output', str(out)],
                                 capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stderr)
            case = json.loads(out.read_text())
            self.assertEqual(case['workload']['active_sessions'], 1)
            self.assertEqual(case['configuration']['reserved_slots'], 40)
            self.assertEqual(case['schema'], 'glm53-comparison-case-v1')
            self.assertFalse(case['gpu_validated'])


class ScheduleTest(unittest.TestCase):
    def test_slowest_rank_releases_the_next_layer_not_the_average(self):
        tasks = [step('rank0', 0, 100), step('rank1', 1, 300),
                 step('join', 0, 50, ['rank0', 'rank1'])]
        result = comparison.schedule(tasks)
        self.assertTrue(result['complete'])
        self.assertEqual(result['makespan_ns'], 350)
        join = next(row for row in result['intervals'] if row['id'] == 'join')
        self.assertEqual(join['start_ns'], 300)

    def test_unknown_cost_is_not_zero_and_cycles_are_rejected(self):
        result = comparison.schedule([step('unknown', 0, None), step('next', 1, 3, ['unknown'])])
        self.assertFalse(result['complete'])
        self.assertIsNone(result['makespan_ns'])
        self.assertIn('unknown', result['unknown_costs'])
        for tasks in ([step('a', 0, 1, ['missing'])],
                      [step('a', 0, 1, ['b']), step('b', 1, 1, ['a'])],
                      [step('a', 0, 1), step('a', 1, 1)]):
            with self.assertRaises(ValueError):
                comparison.schedule(tasks)

    def test_generated_dags_respect_dependencies_and_exclusive_resources(self):
        rng = random.Random(53)
        for _ in range(80):
            tasks = [step(str(i), rng.randrange(4), rng.randrange(1, 100),
                          [str(j) for j in range(i) if rng.random() < .12]) for i in range(25)]
            result = comparison.schedule(tasks)
            by_id = {row['id']: row for row in result['intervals']}
            for task in tasks:
                item = by_id[task['id']]
                self.assertEqual(item['end_ns']-item['start_ns'], task['duration_ns'])
                for dep in task['depends_on']:
                    self.assertGreaterEqual(item['start_ns'], by_id[dep]['end_ns'])
            for gpu in range(4):
                items = sorted((by_id[t['id']] for t in tasks if t['device'] == gpu), key=lambda r:r['start_ns'])
                for a,b in zip(items,items[1:]):
                    self.assertLessEqual(a['end_ns'], b['start_ns'])
            rng.shuffle(tasks)
            self.assertEqual(comparison.schedule(tasks), result)


class MatchedWorkTest(unittest.TestCase):
    def journals(self):
        case = comparison.prepare_case()
        tasks = [step('rank0', 0, 100), step('rank1', 1, 300), step('join', 0, 50, ['rank0','rank1'])]
        ideal: dict[str, Any] = dict(schema='glm53-work-journal-v1', case=case, evidence='synthetic-fixture',
                     cost_coverage='complete', steps=tasks)
        measured = copy.deepcopy(ideal)
        for task,start,end in zip(measured['steps'], [0,0,400], [200,400,500]):
            task['start_ns'],task['end_ns'] = start,end
        return ideal,measured

    def test_matched_cycle_uses_wall_interval_and_reports_each_logical_step(self):
        ideal,recorded = self.journals()
        result = comparison.compare_journals(ideal,recorded)
        self.assertTrue(result['comparable'])
        self.assertEqual(len(result['pairs']), 3)
        self.assertEqual(result['ideal_cycle_ns'], 350)
        self.assertEqual(result['recorded_cycle_ns'], 500)
        self.assertFalse(result['gpu_validated'])
        self.assertEqual(result['evidence'], 'synthetic-fixture')
        self.assertEqual(result['transport']['recorded']['requested'], 'internal')
        self.assertIsNone(result['transport']['recorded']['actual'])

    def test_captured_comparison_requires_actual_requested_transport(self):
        ideal, recorded = self.journals()
        ideal['evidence'], recorded['evidence'] = 'source-model', 'captured'
        for actual in (None, 'internal'):
            recorded['case']['configuration'].update(transport_requested='internal-nccl-gather', transport_actual=actual)
            result = comparison.compare_journals(ideal, recorded)
            self.assertFalse(result['comparable'])
            self.assertIsNone(result['speedup'])
            self.assertTrue(any('transport' in gap.lower() for gap in result['gaps']))
        recorded['case']['configuration']['transport_actual'] = 'internal-nccl-gather'
        self.assertTrue(comparison.compare_journals(ideal, recorded)['comparable'])
        ideal['evidence'] = 'captured'
        result = comparison.compare_journals(ideal, recorded)
        self.assertFalse(result['comparable'])
        self.assertIsNone(result['speedup'])
        self.assertTrue(any('transport' in gap.lower() for gap in result['gaps']))

    def test_workload_missing_steps_or_memory_only_model_cannot_claim_full_comparison(self):
        ideal,recorded = self.journals()
        for mutate in [
            lambda x: x['case']['workload'].update(context_tokens=4096),
            lambda x: x['steps'].pop(),
            lambda x: x['steps'][0]['work'].update(rows=3),
            lambda x: x['steps'][0].update(phase='draft-1'),
        ]:
            bad=copy.deepcopy(recorded);mutate(bad)
            result=comparison.compare_journals(ideal,bad)
            self.assertFalse(result['comparable'])
            self.assertIsNone(result['speedup'])
            self.assertTrue(result['gaps'])
        ideal['cost_coverage']='memory-only'
        result=comparison.compare_journals(ideal,recorded)
        self.assertFalse(result['comparable'])
        self.assertIsNone(result['speedup'])

    def test_real_cli_compares_a_named_cycle_and_labels_synthetic_evidence(self):
        ideal,recorded=self.journals()
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            (directory/'ideal.json').write_text(json.dumps(ideal))
            (directory/'recorded.json').write_text(json.dumps(recorded))
            run=subprocess.run([sys.executable,str(CLI),'compare','--ideal',str(directory/'ideal.json'),
                                '--recorded',str(directory/'recorded.json'),'--output',str(directory/'result.json')],
                               text=True,capture_output=True,timeout=10)
            self.assertEqual(run.returncode,0,run.stderr)
            result=json.loads((directory/'result.json').read_text())
            self.assertEqual(result['recorded_cycle_ns'],500)
            self.assertEqual(result['evidence'],'synthetic-fixture')
            self.assertFalse(result['gpu_validated'])

    def test_multiple_cycles_cannot_be_called_one_cycle(self):
        ideal,recorded=self.journals()
        for journal in (ideal,recorded):
            journal['steps'].append(step('later',0,10,cycle_override=True))
            journal['steps'][-1].update(cycle=1,start_ns=500,end_ns=510)
        with self.assertRaisesRegex(ValueError, 'one cycle'):
            comparison.compare_journals(ideal,recorded)

    def test_unknown_workload_identity_is_not_a_match(self):
        ideal,recorded=self.journals()
        for journal in (ideal,recorded):
            journal['steps'][0]['cycle']=None
        with self.assertRaisesRegex(ValueError, 'cycle'):
            comparison.compare_journals(ideal,recorded)

    def test_different_implementation_is_named_not_confused_with_different_work(self):
        ideal,recorded=self.journals()
        recorded['case']['configuration']['kv']='host'
        result=comparison.compare_journals(ideal,recorded)
        self.assertTrue(result['comparable'])
        self.assertIn('kv', result['configuration_differences'])


if __name__ == '__main__':
    unittest.main()
