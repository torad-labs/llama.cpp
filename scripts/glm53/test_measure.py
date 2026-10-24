import unittest
from types import SimpleNamespace

from measure import completion_payload, server_command, validate_response


class ServedContract(unittest.TestCase):
    def test_sampling_is_the_frozen_served_distribution(self):
        payload = completion_payload("public prompt", 384, 0)
        for key, value in dict(temperature=1.0, top_p=0.95, repeat_penalty=1.1,
                               repeat_last_n=256, seed=530, ignore_eos=True,
                               cache_prompt=False, n_predict=384, id_slot=0).items():
            self.assertEqual(payload[key], value)
        self.assertNotIn("top_k", payload)  # Preserve the qualified server default.
        self.assertEqual(payload["prompt"], "public prompt")

    def test_baseline_and_candidate_use_same_model_and_native_mtp(self):
        args = SimpleNamespace(binary="/bundle with spaces/baseline", model="/models/part1.gguf",
                               fixtures="/fixtures", context=1048576, ubatch=512, profile="",
                               port=8102, slots=3, depth=2, adaptive=0.18)
        baseline = server_command(args)
        args.profile = "sm120-balanced"
        candidate = server_command(args)
        self.assertEqual(candidate[:-2], baseline)
        self.assertEqual(candidate[-2:], ["--glm53-profile", "sm120-balanced"])
        self.assertEqual(baseline[0], "/bundle with spaces/baseline/llama-server")
        for option, value in (("-sm", "tensor"), ("-ts", "1,1"), ("-ctk", "q8_0"),
                              ("-ctv", "q8_0"), ("-ctki", "q8_0"), ("-b", "4096"),
                              ("-np", "3"), ("--spec-type", "draft-mtp"),
                              ("--spec-draft-mtp-adaptive", "0.18")):
            self.assertEqual(baseline[baseline.index(option)+1], value)
        self.assertIn("/fixtures/glm-mtp-public-98304.i32", baseline)

    def test_real_response_requires_delivered_output_and_complete_prompt(self):
        row = {"content": "answer", "timings": {"predicted_n": 384, "prompt_n": 32768}}
        self.assertTrue(validate_response(row, 384, 32768))
        for bad in ({"content": "", "timings": row["timings"]},
                    {"content": "answer", "timings": {"predicted_n": 64, "prompt_n": 32768}},
                    {"content": "answer", "timings": {"predicted_n": 384, "prompt_n": 127}},
                    {"error": "request failed"}):
            with self.assertRaises(ValueError):
                validate_response(bad, 384, 32768)


if __name__ == "__main__":
    unittest.main()
