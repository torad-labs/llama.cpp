import math
import pytest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from utils import *
import inkling_fixture as fx

# We use a F16 MOE gguf as main model, and q4_0 as draft model

server = ServerPreset.stories15m_moe()

MODEL_DRAFT_FILE_URL = "https://huggingface.co/ggml-org-dev/tiny-llamas/resolve/main/stories15M-q4_0.gguf"

def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    # set default values
    server.model_draft = download_file(MODEL_DRAFT_FILE_URL)
    server.spec_type = "draft-simple"
    server.spec_draft_n_min = 4
    server.spec_draft_n_max = 8
    server.fa = "off"


@pytest.fixture(autouse=True)
def fixture_create_server():
    return create_server()


def test_with_and_without_draft():
    global server
    request = {
        "prompt": "I believe the meaning of life is",
        "temperature": 0.2,
        "top_k": 5,
        "seed": 4242,
        "n_predict": 16,
        "return_tokens": True,
    }

    server.model_draft = None  # disable draft model
    server.spec_type = None
    server.start()
    res = server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    tokens_no_draft = res.body["tokens"]
    server.stop()

    # create new server with draft model
    create_server()
    server.start()
    res = server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    tokens_draft = res.body["tokens"]

    assert tokens_no_draft == tokens_draft

    server.stop()
    create_server()
    assert server.spec_draft_n_max is not None
    server.spec_synth_rates = [0.0] * server.spec_draft_n_max
    server.start()
    res = server.make_request("POST", "/completion", data=request)

    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    assert res.body["timings"]["draft_n_accepted"] == 0
    assert res.body["tokens"] == tokens_no_draft


def test_draft_respects_batch_capacity(tmp_path):
    global server
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = server.model_draft
    server.n_slots = 2
    server.n_ctx = 2048
    server.n_batch = 32
    # Keep ubatch within CUDA MMVQ_MAX_BATCH_SIZE (8) so batched verification and plain 1-token decoding use the same matmul kernel; n_batch=32 still trims the 48-token draft across both slots.
    server.n_ubatch = 8
    server.spec_draft_n_min = 1
    server.spec_draft_n_max = 48
    server.server_continuous_batching = True
    # The tokens are held to the plain run's. Outside flash attention the host computes an attention row by one kernel
    # for a batch of one column and by another for more, so a column's last bits follow its batch's width, and this
    # model's 8-bit activations can turn one of them into another token (on eight prompts of 90 tokens: 3 before
    # upstream v1.0.0, 1 after). The host's flash attention computes every column the same way
    server.fa = "on"

    prompts = ["Once upon a time", "A little girl found a"]
    base_req = {
        "temperature": 0.0,
        "n_predict": 96,
        "ignore_eos": True,
        "cache_prompt": False,
        "return_tokens": True,
        "n_probs": 2,
    }

    draft_model = server.model_draft
    server.model_draft = None
    server.spec_type = None
    server.start()
    plain = []
    for slot, prompt in enumerate(prompts):
        result = server.make_request("POST", "/completion", data={**base_req, "prompt": prompt, "id_slot": slot})
        assert result.status_code == 200
        plain.append(result.body)
    server.stop()

    log_path = tmp_path / "drafted.log"
    server.model_draft = draft_model
    server.spec_type = "draft-simple"
    server.debug = True
    server.log_path = str(log_path)
    server.start()
    # One multi-prompt request enqueues both slots together so continuous batching shares n_batch=32 across them.
    result = server.make_request("POST", "/completion", data={**base_req, "prompt": prompts})
    assert result.status_code == 200
    drafted = result.body
    server.stop()

    assert len(drafted) == len(prompts)
    log_text = log_path.read_text(errors="replace")
    shared_cap = server.n_batch // server.n_slots - 1
    for slot, (expected, actual) in enumerate(zip(plain, drafted)):
        assert actual["timings"]["draft_n"] > 0
        assert actual["tokens"] == expected["tokens"]
        assert len(actual["completion_probabilities"]) == len(actual["tokens"])
        assert re.search(rf"id\s+{slot}\s+\|.*max possible draft:\s+{shared_cap}\b", log_text), (
            f"slot {slot} did not hit shared draft cap {shared_cap}"
        )


def test_draft_acceptance_floor():
    # The other tests check that drafting happens, not that drafts are accepted.
    # This drafter/target pair is mismatched on purpose, so accept is only 15-17%.
    global server
    server.start()
    for prompt in [
        "I believe the meaning of life is",
        "Once upon a time there was a little girl who",
    ]:
        res = server.make_request("POST", "/completion", data={
            "prompt": prompt,
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 64,
        })
        assert res.status_code == 200
        draft_n = res.body["timings"]["draft_n"]
        draft_n_accepted = res.body["timings"]["draft_n_accepted"]
        assert draft_n > 0, f"nothing was drafted for {prompt!r}"
        accept_rate = draft_n_accepted / draft_n
        assert accept_rate > 0.05, (
            f"draft acceptance collapsed to {100 * accept_rate:.2f}% "
            f"({draft_n_accepted}/{draft_n}) for {prompt!r}"
        )


def test_backend_sampling_lazy_grammar_mid_draft():
    # The model drafts for itself, so every draft is accepted and the token that triggers the lazy grammar sits
    # mid-verification: the rows after it were sampled on the backend without the grammar, so the verification must
    # end on that token and the slot continue on the CPU. The tokens match a CPU-sampled run.
    global server
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = server.model_draft
    server.backend_sampling = True
    server.start()

    def run(backend_sampling):
        res = server.make_request("POST", "/completion", data={
            "prompt": "Once upon a time",
            "temperature": 0.0,
            "n_predict": 48,
            "return_tokens": True,
            "backend_sampling": backend_sampling,
            "grammar": 'root ::= "girl named Zo" [a-z]* "."',
            "grammar_lazy": True,
            "grammar_triggers": [{"type": 1, "value": "girl named"}],
        })
        assert res.status_code == 200
        return res.body

    on, off = run(True), run(False)
    assert on["timings"]["draft_n"] > 0
    assert on["generation_settings"]["backend_sampling"] is True
    assert "girl named Zo" in on["content"]
    assert on["tokens"] == off["tokens"]


@pytest.mark.parametrize("post_sampling", [False, True])
def test_draft_token_probs(post_sampling: bool):
    # The model drafts for itself, so after the first token every token comes out of a verification: each carries the
    # probabilities plain decoding gives it, up to rounding. A verification computes its tokens as one batch and plain
    # decoding one at a time, and the two round apart: up to 0.10 in logprob and 0.11 post-sampling over seed 4242 and
    # 0 to 63 on the CPU, where a prompt batch of the same tokens gives the drafted values; the tolerance is about twice
    # that. The runs may also sample apart at temperature 0.2, so the tokens are compared until they part
    global server
    request = {
        "prompt": "I believe the meaning of life is",
        "temperature": 0.2,
        "top_k": 5,
        "seed": 4242,
        "n_predict": 16,
        "n_probs": 3,
        "post_sampling_probs": post_sampling,
    }

    def run(draft: bool):
        global server
        create_server()
        server.model_hf_repo = None
        server.model_hf_file = None
        server.model_file = server.model_draft
        if not draft:
            server.model_draft = None
            server.spec_type = None
        server.start()
        res = server.make_request("POST", "/completion", data=request)
        server.stop()
        assert res.status_code == 200
        return res.body

    plain, drafted = run(False), run(True)
    assert drafted["timings"]["draft_n_accepted"] > 0

    key, top, one = ("prob", "top_probs", 1.0) if post_sampling else ("logprob", "top_logprobs", 0.0)
    probs_plain, probs_drafted = plain["completion_probabilities"], drafted["completion_probabilities"]
    n = next((i for i, (a, b) in enumerate(zip(probs_drafted, probs_plain)) if a["id"] != b["id"]),
             min(len(probs_drafted), len(probs_plain)))
    assert n >= 4
    for a, b in zip(probs_drafted[:n], probs_plain[:n]):
        assert a[key] == pytest.approx(b[key], abs=0.25)
        # a probability the server never set goes out as exactly 1 (logprob 0), which the tolerance takes where plain
        # decoding gives 0.75 or more. A real exact 1 (the other tokens' share rounds to nothing; post-sampling also
        # top-p or min-p leaving one candidate, which cut before the temperature) has plain decoding at 0.999 or more:
        # parting them takes 1.5 nats of logit or more, where the runs differ by 0.1
        assert a[key] != one or b[key] == pytest.approx(one, abs=1e-3)
        assert len(a[top]) == len(b[top]) > 0


def test_different_draft_min_draft_max():
    global server
    test_values = [
        (1, 2),
        (1, 4),
        (4, 8),
        (4, 12),
        (8, 16),
    ]
    last_content = None
    for draft_min, draft_max in test_values:
        server.stop()
        server.spec_draft_n_min = draft_min
        server.spec_draft_n_max = draft_max
        server.start()
        res = server.make_request("POST", "/completion", data={
            "prompt": "I believe the meaning of life is",
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 16,
        })
        assert res.status_code == 200
        if last_content is not None:
            assert last_content == res.body["content"]
        last_content = res.body["content"]


def test_synth_is_deterministic():
    global server
    assert server.spec_draft_n_max is not None
    server.spec_synth_rates = [0.75 ** (i + 1) for i in range(server.spec_draft_n_max)]
    server.start()

    request = {
        "prompt": "I believe the meaning of life is",
        "temperature": 0.2,
        "top_k": 5,
        "seed": 4242,
        "n_predict": 32,
    }
    responses = [server.make_request("POST", "/completion", data=request) for _ in range(2)]

    for res in responses:
        assert res.status_code == 200
        assert res.body["timings"]["draft_n"] > 0
    assert responses[0].body["timings"]["draft_n"] == responses[1].body["timings"]["draft_n"]
    assert responses[0].body["timings"]["draft_n_accepted"] == responses[1].body["timings"]["draft_n_accepted"]


def test_synth_ignores_target_tokens():
    global server
    assert server.spec_draft_n_max is not None
    server.spec_synth_rates = [1.0] * server.spec_draft_n_max
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "seed": 4242,
        "n_predict": 32,
    })

    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    assert res.body["timings"]["draft_n_accepted"] == res.body["timings"]["draft_n"]

    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "seed": 4242,
        "n_predict": 6,
        "grammar": 'root ::= "a"{5,5}',
    })
    assert res.status_code == 200, res.body

    res = server.make_request("POST", "/completion", data={
        "prompt": "Respond with only: OK",
        "temperature": 0.0,
        "seed": 4242,
        "n_predict": 64,
        "ignore_eos": True,
    })
    assert res.status_code == 200, res.body
    assert res.body["tokens_predicted"] == 64
    assert res.body["stop_type"] == "limit"


def test_draft_n_ubatch_smaller_than_target():
    global server
    request = {
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 16,
    }

    # draft with the default ubatch size of the draft context
    server.start()
    res_default = server.make_request("POST", "/completion", data=request)
    server.stop()
    assert res_default.status_code == 200
    assert res_default.body["timings"]["draft_n"] > 0

    # a smaller draft ubatch splits the draft batches into more ubatches, but must not change the result
    server.spec_draft_n_ubatch = 8
    server.start()
    res_small = server.make_request("POST", "/completion", data=request)
    server.stop()
    assert res_small.status_code == 200
    assert res_small.body["timings"]["draft_n"] > 0
    assert res_small.body["tokens"] == res_default.body["tokens"]


def test_slot_ctx_not_exceeded():
    global server
    server.n_ctx = 256
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello " * 248,
        "temperature": 0.0,
        "top_k": 1,
        "speculative.p_min": 0.0,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0


def test_with_ctx_shift():
    global server
    server.n_ctx = 256
    server.enable_ctx_shift = True
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello " * 248,
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 256,
        "speculative.p_min": 0.0,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0
    assert res.body["tokens_predicted"] == 256
    assert res.body["truncated"] == True


@pytest.mark.parametrize("n_slots,n_requests", [
    (1, 2),
    (2, 2),
])
def test_multi_requests_parallel(n_slots: int, n_requests: int):
    global server
    server.n_slots = n_slots
    server.start()
    tasks = []
    for _ in range(n_requests):
        tasks.append((server.make_request, ("POST", "/completion", {
            "prompt": "I believe the meaning of life is",
            "temperature": 0.0,
            "top_k": 1,
        })))
    results = parallel_function_calls(tasks)
    for res in results:
        assert res.status_code == 200
        assert match_regex("(wise|kind|owl|answer)+", res.body["content"])


def test_mtp_vocab_without_draft_mtp_is_refused(tmp_path):
    # the draft-mtp options are read only when the MTP head drafts: without --spec-type draft-mtp no draft context
    # is built and the file would never be opened, so the server refuses them at load instead of serving without them
    global server
    server.model_draft = None
    server.spec_type = None
    vocab = tmp_path / "vocab.i32"
    vocab.write_bytes(b"\x00\x00\x00\x00")
    server.spec_draft_mtp_vocab = str(vocab)
    with pytest.raises(RuntimeError, match="Server process died"):
        server.start()


# The synthetic Inkling 2 target (128 tokens) and a DSpark draft of random weights for it, small enough to sample
# thousands of replies
INKLING_PROMPT = [5, 17, 42, 9, 77, 3, 61, 28]


@pytest.fixture(scope="module")
def inkling_pair(tmp_path_factory):
    folder = tmp_path_factory.mktemp("inkling-dspark")
    target = fx.write_model(str(folder))
    draft = str(folder / "draft.gguf")
    fx.write_dspark_draft(draft, target)
    return target, draft


def create_inkling_server(pair: tuple[str, str], sampling: str) -> ServerProcess:
    s = ServerProcess()
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file, s.model_draft = pair
    s.spec_type = "draft-dspark"
    s.spec_draft_n_max = 3
    s.spec_draft_n_min = 1
    s.spec_draft_p_min = 0
    s.spec_draft_sampling = sampling
    s.n_ctx = 256
    s.n_slots = 1
    s.n_gpu_layer = int(os.environ.get("N_GPU_LAYERS", 99))
    s.spec_draft_n_gpu_layer = s.n_gpu_layer
    s.server_metrics = True
    return s


def inkling_reply(s: ServerProcess, seed: int, temperature: float, n_probs: int = 0) -> dict:
    res = s.make_request("POST", "/completion", data={
        "prompt": INKLING_PROMPT,
        "temperature": temperature,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "seed": seed,
        "n_predict": 4,
        "n_probs": n_probs,
        "return_tokens": True,
    })
    assert res.status_code == 200
    return res.body


def spec_counts(s: ServerProcess) -> tuple[float, float, float]:
    body = s.make_request("GET", "/metrics").body
    values = []
    for name in ("draft_tokens", "accepted_tokens", "drafts"):
        m = re.search(rf"^llamacpp:spec_decode_num_{name}_total (\S+)$", body, re.MULTILINE)
        assert m is not None
        values.append(float(m.group(1)))
    return values[0], values[1], values[2]


def spec_reject_rows(s: ServerProcess) -> float:
    # draft rows verified by rejection: 0 where a sampled draft fell back to sample-and-match
    body = s.make_request("GET", "/metrics").body
    m = re.search(r"^llamacpp:spec_decode_num_reject_rows_total (\S+)$", body, re.MULTILINE)
    assert m is not None
    return float(m.group(1))


def test_inkling_dspark_sampled_draft_keeps_the_target_distribution(inkling_pair):
    # Every reply token is a sample of the target's distribution given the tokens before it, whether the draft proposed
    # it or not: over many replies, the count of each token at a position matches the sum of the probabilities the
    # target gave it there (each reply's n_probs rows, read from the verified rows), within a chi-square bound
    s = create_inkling_server(inkling_pair, "probabilistic")
    s.start()
    n_replies = int(os.environ.get("INKLING_DSPARK_REPLIES", 4000))
    expected = [{} for _ in range(4)]
    observed = [{} for _ in range(4)]
    try:
        for seed in range(n_replies):
            body = inkling_reply(s, seed, 1.0, n_probs=128)
            for i, tok in enumerate(body["completion_probabilities"]):
                observed[i][tok["id"]] = observed[i].get(tok["id"], 0) + 1
                for cand in tok["top_logprobs"]:
                    expected[i][cand["id"]] = expected[i].get(cand["id"], 0.0) + math.exp(cand["logprob"])
        drafted, accepted, _ = spec_counts(s)
        reject_rows = spec_reject_rows(s)
    finally:
        s.stop()
    assert drafted > 0 and accepted > 0 and reject_rows > 0
    for i in range(4):
        # cells under 5 expected are pooled into one
        chi2, cells, pooled_e, pooled_o = 0.0, 0, 0.0, 0
        for tok, e in expected[i].items():
            o = observed[i].get(tok, 0)
            if e < 5.0:
                pooled_e += e
                pooled_o += o
                continue
            chi2 += (o - e)**2/e
            cells += 1
        if pooled_e >= 5.0:
            chi2 += (pooled_o - pooled_e)**2/pooled_e
            cells += 1
        dof = cells - 1
        bound = dof + 6.0*math.sqrt(2.0*dof)
        print(f"INKLING_DSPARK_EXACT position {i}: chi-square {chi2:.1f} over {dof} dof (bound {bound:.1f}), "
              f"drafted {drafted:.0f} accepted {accepted:.0f}")
        assert chi2 < bound, f"position {i}: chi-square {chi2:.1f} over {dof} dof"


def test_inkling_dspark_sampled_draft_acceptance(inkling_pair):
    # the same replies at temperature 1, drafted greedily and verified by matching, then sampled and verified by
    # rejection; at temperature 0 the sampled draft's graph keeps the greedy chain, so both draft and keep alike
    n_replies = int(os.environ.get("INKLING_DSPARK_REPLIES", 4000)) // 4
    rates = {}
    at_zero = {}
    reject_rows = {}
    for sampling in ("greedy", "probabilistic"):
        s = create_inkling_server(inkling_pair, sampling)
        s.start()
        try:
            for seed in range(n_replies):
                inkling_reply(s, seed, 1.0)
            drafted, accepted, drafts = spec_counts(s)
            reject_rows[sampling] = spec_reject_rows(s)
            for seed in range(16):
                inkling_reply(s, seed, 0.0)
            after = spec_counts(s)
        finally:
            s.stop()
        rates[sampling] = (accepted/drafted, 1.0 + accepted/drafts)
        at_zero[sampling] = (after[0] - drafted, after[1] - accepted)
        print(f"INKLING_DSPARK_ACCEPT {sampling}: drafted {drafted:.0f} accepted {accepted:.0f} rounds {drafts:.0f} "
              f"accept rate {rates[sampling][0]:.4f} tokens a round {rates[sampling][1]:.4f}; "
              f"at temperature 0 drafted {at_zero[sampling][0]:.0f} accepted {at_zero[sampling][1]:.0f}")
    assert rates["greedy"][0] > 0 and rates["probabilistic"][0] > 0
    assert at_zero["greedy"] == at_zero["probabilistic"]
    assert reject_rows["greedy"] == 0 and reject_rows["probabilistic"] > 0


def test_inkling_dspark_sampled_drafts_of_different_lengths(inkling_pair):
    # Sessions at temperature 1 verify sampled drafts in one step: a draft cut at an end-of-generation token and padded
    # to the step's rows keeps one distribution a row, and every request finishes. The fixture's vocabulary has no
    # end token, so one is named; the requests ignore it, so the target never samples it and the draft proposes it
    s = create_inkling_server(inkling_pair, "probabilistic")
    s.n_slots = 4
    s.n_ctx = 4*256
    s.override_kv = "tokenizer.ggml.eos_token_id=int:7"
    s.start()
    try:
        for wave in range(int(os.environ.get("INKLING_DSPARK_WAVES", 8))):
            tasks = [(s.make_request, ("POST", "/completion", {
                "prompt": INKLING_PROMPT[:4 + i],
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
                "min_p": 0.0,
                "seed": 100*wave + i,
                "n_predict": 32 + 7*i,
                "ignore_eos": True,
            })) for i in range(4)]
            for res in parallel_function_calls(tasks):
                assert res.status_code == 200
        drafted, _, _ = spec_counts(s)
        assert drafted > 0
        assert spec_reject_rows(s) > 0
    finally:
        s.stop()


@pytest.mark.skipif(os.environ.get("INKLING_DSPARK_TP4") != "1", reason="requires four logical CUDA devices")
def test_inkling_dspark_tp4_rollback_and_restore(tmp_path, monkeypatch):
    target = fx.write_model(str(tmp_path))
    draft = str(tmp_path / "draft.gguf")
    fx.write_dspark_draft(draft, target, tp4=True)
    for name, value in {
        "LLAMA_ARG_DEVICE": "CUDA0,CUDA1,CUDA2,CUDA3",
        "LLAMA_ARG_SPLIT_MODE": "tensor",
        "LLAMA_TOK_EMBD_GPU": "1",
        "LLAMA_ARG_FIT": "off",
    }.items():
        monkeypatch.setenv(name, value)

    def config():
        s = create_inkling_server((target, draft), "greedy")
        s.n_slots = 23
        s.n_ctx = 23 * 256
        s.n_batch = s.n_ubatch = 256
        s.fa = "on"
        s.kv_unified = True
        s.server_continuous_batching = True
        return s

    def complete(s, seq, *, slot=None):
        res = s.make_request("POST", "/completion", data={
            "prompt": INKLING_PROMPT * 4 + [31 + seq],
            "id_slot": seq if slot is None else slot,
            "temperature": 0.0,
            "n_predict": 12 + seq % 4,
            "ignore_eos": True,
            "cache_prompt": True,
            "return_tokens": True,
        })
        assert res.status_code == 200
        return res.body["tokens"]

    plain = config()
    plain.model_draft = plain.spec_type = None
    plain.start()
    try:
        expected = [complete(plain, i) for i in range(23)]
    finally:
        plain.stop()

    monkeypatch.setenv("LLAMA_ARG_SPEC_DRAFT_SPLIT_MODE", "tensor")
    s = config()
    s.spec_synth_rates = [0.0] * 3
    s.server_slots = True
    s.slot_save_path = str(tmp_path)
    s.start()
    try:
        for n in range(1, 24):
            ready = Barrier(n)

            def send(seq):
                ready.wait()
                return complete(s, seq)

            with ThreadPoolExecutor(max_workers=n) as pool:
                assert list(pool.map(send, range(n))) == expected[:n]

        path = {"filename": "dspark-state.bin"}
        saved = s.make_request("POST", "/slots/0?action=save", data=path)
        assert saved.status_code == 200 and saved.body["n_saved"] > 0
        restored = s.make_request("POST", "/slots/22?action=restore", data=path)
        assert restored.status_code == 200 and restored.body["n_restored"] == saved.body["n_saved"]
        assert complete(s, 0, slot=22) == expected[0]
        drafted, accepted, _ = spec_counts(s)
        assert drafted > 0 and accepted == 0
    finally:
        s.stop()

