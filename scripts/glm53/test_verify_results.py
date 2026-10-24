"""The 200 tok/s goal cannot pass with missing requests or an averaged rate."""
import copy
import math
import random
import unittest
import verify_results
from verify_results import evaluate, validate_routes

HASHES = {name: name + "-pinned" for name in ("english", "chinese", "code", "tools")}

def arm(rate):
    return [dict(pair=pair, category=category, repeat=repeat, prompt_sha256=prompt_hash,
                 response=dict(content="generated", timings=dict(predicted_n=384,
                     predicted_ms=383000/rate, predicted_per_second=rate)))
            for pair in (0, 1) for category, prompt_hash in HASHES.items() for repeat in range(3)]

class Goal(unittest.TestCase):
    def test_exact_goal_and_weighted_denominator(self):
        result = evaluate(arm(140), arm(200), HASHES)
        self.assertTrue(result["passed"])
        self.assertEqual(result["candidate"]["steady_tokens"], 24*383)
        self.assertAlmostEqual(result["candidate"]["tok_s"], 200)
        rows = arm(300)
        for i, row in enumerate(rows):
            row["response"]["timings"]["predicted_ms"] = 383000/(150 if i%2 else 300)
            row["response"]["timings"]["predicted_per_second"] = 9999  # never the rate denominator
        result = evaluate(arm(140), rows, HASHES)
        self.assertAlmostEqual(result["candidate"]["tok_s"], 200)

    def test_target_pairs_and_categories_each_required(self):
        result = evaluate(arm(140), arm(199.99), HASHES)
        self.assertIn("combined rate below 200 tok/s", result["failures"])
        rows = arm(300)
        for row in rows:
            if row["pair"] == 1: row["response"]["timings"]["predicted_ms"] = 383000/139
        self.assertIn("pair 1 did not improve", evaluate(arm(140), rows, HASHES)["failures"])
        rows = arm(300)
        for row in rows:
            if row["category"] == "tools": row["response"]["timings"]["predicted_ms"] = 383000/137
        self.assertIn("tools regressed more than 2%", evaluate(arm(140), rows, HASHES)["failures"])

    def test_bad_or_incomplete_workload_cannot_pass(self):
        valid = arm(210)
        variants = [valid[:-1], valid + [valid[0]]]
        for key, value in [("predicted_n",383),("predicted_n",385),("predicted_ms",0),
                           ("predicted_ms",-1),("predicted_ms",math.nan),("predicted_ms",math.inf)]:
            rows = copy.deepcopy(valid); rows[0]["response"]["timings"][key] = value; variants.append(rows)
        for key, value in [("category","unknown"),("repeat",4),("pair",2),("prompt_sha256","different")]:
            rows=copy.deepcopy(valid); rows[0][key]=value; variants.append(rows)
        for rows in variants:
            with self.subTest(rows=rows[0]):
                with self.assertRaises(ValueError): evaluate(arm(140), rows, HASHES)
        with self.assertRaises(ValueError): evaluate(arm(140), valid, {"english":HASHES["english"]})

    def test_route_histograms_conserve_work(self):
        rng=random.Random(530)
        for _ in range(128):
            hist=[0]*64; rows=[0]*8; pairs=unique=reused=0
            for _ in range(17):
                width=rng.randint(1,8); rows[width-1]+=1
                counts={}
                for token in range(width):
                    for expert in rng.sample(range(24),8): counts[expert]=counts.get(expert,0)+1
                for count in counts.values(): hist[count-1]+=1; pairs+=count; unique+=1; reused+=count>=3
            record=dict(calls=17,pairs=pairs,unique=unique,reused=reused,rows=rows,multiplicity=hist)
            self.assertTrue(validate_routes(record))
            for key in ("calls","pairs","unique","reused"):
                wrong=copy.deepcopy(record); wrong[key]+=1
                self.assertFalse(validate_routes(wrong))

    def test_idle_contexts_preserve_required_device_observations(self):
        idle = dict(calls=0, pairs=0, unique=0, reused=0, rows=[0]*8, multiplicity=[0]*64)
        self.assertTrue(validate_routes(idle))
        rng = random.Random(531)
        for _ in range(128):
            records = []
            for device in (0, 1):
                calls = rng.randint(1, 5)
                records.append(dict(device=device, calls=calls, pairs=2*calls, unique=2*calls, reused=0,
                                    rows=[calls] + [0]*7, multiplicity=[2*calls] + [0]*63))
                records += [dict(copy.deepcopy(idle), device=device) for _ in range(rng.randint(1, 5))]
            rng.shuffle(records)
            original = copy.deepcopy(records)
            self.assertTrue(verify_results.validate_route_observations(records))
            self.assertEqual(records, original)
            for missing in (0, 1):
                no_work = [r for r in records if r['device'] != missing or r['calls'] == 0]
                self.assertFalse(verify_results.validate_route_observations(no_work))
            wrong = copy.deepcopy(records)
            next(r for r in wrong if r['calls'] == 0)['pairs'] = 1
            self.assertFalse(verify_results.validate_route_observations(wrong))
            self.assertFalse(verify_results.validate_route_observations(records + [dict(idle, device=2)]))
        self.assertFalse(verify_results.validate_route_observations([]))
        self.assertFalse(verify_results.validate_route_observations([dict(idle, device=0), dict(idle, device=1)]))

if __name__ == "__main__":
    unittest.main()
