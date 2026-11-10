# Goal: while one slot streams a reply, a long prompt read in another slot does not hold that reply back for a whole
# n_batch step. With --prompt-step N, the server adds at most N prompt tokens to a step in which another slot is
# generating, so the reply waits at most one short step between its tokens. A prompt read while no other slot
# generates still fills whole n_batch steps, and without --prompt-step nothing changes.
#
# A step's prompt tokens are what the read's streamed prompt_progress advances by between two reports.
import random
import threading
from collections.abc import Generator

import pytest
from utils import *

server: ServerProcess

N_BATCH = 256
STEP = 32
PROMPT = "Once upon a time there was a little girl who liked to read long stories. " * 60


@pytest.fixture(scope="session")
def load_server_presets():
    # This module needs only this small model, not every server preset. Offline runs use its existing cache.
    preset = ServerPreset.tinyllama2()
    preset.offline = os.environ.get("LLAMA_ARG_OFFLINE") == "1"
    preset.start()
    preset.stop()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_ctx = 8192
    server.n_batch = N_BATCH
    server.kv_unified = True
    server.server_continuous_batching = True


class Reply:
    """slot 1 streams a long reply in the background; `started` is set at its first token, `done` once it ends"""

    def __init__(self):
        self.started = threading.Event()
        self.done = threading.Event()
        self.cancel = threading.Event()
        self.errors = []
        self.thread = threading.Thread(target=self.run)

    def run(self):
        res = server.make_stream_request("POST", "/completion", data={
            "prompt": "Once upon a time",
            "id_slot": 1,
            "n_predict": 4000,
            "ignore_eos": True,
            "cache_prompt": False,
            "stream": True,
        })
        assert isinstance(res, Generator)
        try:
            for data in res:
                if data.get("content") or data.get("tokens"):
                    self.started.set()
                if self.cancel.is_set():
                    break
        except Exception as error:
            self.errors.append(error)
        finally:
            res.close()
            self.started.set()
            self.done.set()


def read_steps():
    """slot 0 reads PROMPT; the prompt tokens of each of its steps, and the prompt's length"""
    res = server.make_stream_request("POST", "/completion", data={
        "prompt": PROMPT,
        "id_slot": 0,
        "n_predict": 1,
        "cache_prompt": False,
        "stream": True,
        "return_progress": True,
    })
    steps = []
    last = None
    for data in res:
        p = data.get("prompt_progress")
        if p is None:
            continue
        if last is not None:
            steps.append(p["processed"] - last["processed"])
        last = p
    assert last is not None, "the read streamed no prompt progress"
    return steps, last["total"] - last["cache"]


def read_beside_a_reply():
    reply = Reply()
    reply.thread.start()
    try:
        assert reply.started.wait(timeout=60), "the reply in slot 1 never streamed a token"
        assert not reply.done.is_set(), "the reply in slot 1 ended before the read began"
        steps, n_read = read_steps()
        still_generating = not reply.done.is_set()
    finally:
        # The assertion needs a live reply during the read, not 4,000 tokens after it.
        # Cancel from the reader itself and join before the fixture stops its server.
        reply.cancel.set()
        reply.thread.join(timeout=60)
        assert not reply.thread.is_alive(), "the reply reader did not stop"
    assert not reply.errors, reply.errors
    return steps, n_read, still_generating


@pytest.mark.parametrize("chunk", [0, 128])
def test_prompt_step_bounds_a_read_beside_a_reply(monkeypatch, chunk):
    monkeypatch.setenv("LLAMA_ARG_PROMPT_CHUNK", str(chunk))
    global server
    server.prompt_step = STEP
    server.start()
    steps, n_read, still_generating = read_beside_a_reply()
    assert still_generating, "the reply in slot 1 ended during the read, so the read was not beside a reply"
    assert n_read > 4 * STEP
    assert sum(steps) == n_read
    assert max(steps) <= STEP, f"a step read {max(steps)} prompt tokens beside a reply, over --prompt-step {STEP}: {steps}"


def test_prompt_step_reads_whole_batches_alone():
    global server
    server.prompt_step = STEP
    server.start()
    steps, n_read = read_steps()
    assert n_read > N_BATCH
    assert sum(steps) == n_read
    assert steps[0] == N_BATCH, f"alone, the read's first step took {steps[0]} tokens, not a whole batch of {N_BATCH}: {steps}"


def test_a_prompt_step_above_the_batch_reads_whole_batches_beside_a_reply():
    """any step the parser accepts, the largest int included, bounds a read by the batch and never stops it"""
    global server
    server.prompt_step = 2147483647
    server.start()
    steps, n_read, still_generating = read_beside_a_reply()
    assert still_generating, "the read waited for the reply in slot 1 to end"
    assert sum(steps) == n_read
    assert max(steps) >= N_BATCH - 2, f"with a step above the batch, a step beside a reply read only {max(steps)}: {steps}"


def test_without_prompt_step_a_read_beside_a_reply_fills_its_batches():
    global server
    server.start()
    steps, n_read, still_generating = read_beside_a_reply()
    assert still_generating, "the reply in slot 1 ended during the read, so the read was not beside a reply"
    assert sum(steps) == n_read
    # the reply's token takes one place in the batch
    assert max(steps) >= N_BATCH - 2, f"without --prompt-step a step beside a reply read only {max(steps)}: {steps}"


@pytest.mark.parametrize("chunk", [0, 128])
def test_short_prompt_progresses_before_long_prompt_finishes(monkeypatch, chunk):
    monkeypatch.setenv("LLAMA_ARG_PROMPT_CHUNK", str(chunk))
    server.n_batch = 32
    server.n_threads = 1
    server.start()
    started = threading.Event()
    prompt_done = threading.Event()
    errors = []

    def read_long_prompt():
        try:
            for data in server.make_stream_request("POST", "/completion", data={
                "prompt": PROMPT,
                "id_slot": 0,
                "n_predict": 1,
                "cache_prompt": False,
                "stream": True,
                "return_progress": True,
            }):
                progress = data.get("prompt_progress")
                if progress:
                    if progress["processed"] > 0:
                        started.set()
                    if progress["processed"] == progress["total"]:
                        prompt_done.set()
        except Exception as error:
            errors.append(error)
            started.set()

    reader = threading.Thread(target=read_long_prompt)
    reader.start()
    try:
        assert started.wait(timeout=60), "long prompt never made progress"
        assert not prompt_done.is_set(), "long prompt ended before the competing request"
        response = server.make_request("POST", "/completion", data={
            "prompt": "Once upon a time",
            "id_slot": 1,
            "n_predict": 1,
            "cache_prompt": False,
        })
        assert response.status_code == 200
        assert not prompt_done.is_set(), "short prompt waited for the long prompt to finish"
    finally:
        reader.join(timeout=60)
    assert not reader.is_alive()
    assert not errors


@pytest.mark.parametrize("chunk", [32, 128])
def test_prompt_chunks_are_useful_and_rotate_fairly(monkeypatch, chunk):
    # Goal: concurrent long reads get whole useful chunks, every read progresses before
    # any long read finishes, and short tails consume only their own remaining tokens.
    monkeypatch.setenv("LLAMA_ARG_PROMPT_CHUNK", str(chunk))
    server.n_slots = 8
    server.n_ctx = 16384
    server.n_batch = 4 * chunk
    server.n_ubatch = 2 * chunk
    server.n_threads = 1
    server.start()
    rng = random.Random(53)
    lengths = [8 * chunk + rng.randrange(1, chunk) for _ in range(server.n_slots)]
    # Token arrays fix lengths independently of tokenizer behavior.
    prompts = [[1] + [20 + i] * (length - 1) for i, length in enumerate(lengths)]
    last = {}
    increments = {}
    first_steps = {}
    completed = set()
    for data in server.make_stream_request("POST", "/completion", data={
        "prompt": prompts, "n_predict": 1, "ignore_eos": True,
        "cache_prompt": False, "stream": True, "return_progress": True,
    }):
        if data.get("stop"):
            completed.add(data["index"])
            assert data["tokens_predicted"] == 1
        progress = data.get("prompt_progress")
        if progress is None:
            continue
        idx = data["index"]
        processed = progress["processed"]
        delta = processed - last.get(idx, 0)
        last[idx] = processed
        if delta <= 0:
            continue
        increments.setdefault(idx, []).append(delta)
        first_steps.setdefault(idx, delta)
        if processed < progress["total"]:
            assert delta >= chunk, f"prompt {idx} got a thin {delta}-token slice: {increments}"
        if processed == progress["total"]:
            assert len(first_steps) == len(prompts), "a long prompt finished before all readers progressed"
    assert completed == set(range(len(prompts)))
    for idx, length in enumerate(lengths):
        assert sum(increments[idx]) == length
    assert max(first_steps.values()) <= 2 * chunk


@pytest.mark.parametrize("chunk,n_tokens", [(300, 400), (129, 400), (511, 512)])
def test_prompt_chunks_leave_pooled_embeddings_whole(monkeypatch, chunk, n_tokens):
    # An indivisible prompt that fits one physical batch must have the same pooled
    # embedding with chunk scheduling enabled as when the entire input is decoded.
    server.n_slots = 1
    server.n_ctx = 2048
    server.n_batch = server.n_ubatch = 512
    server.n_threads = 1
    server.server_embeddings = True
    server.pooling = "mean"
    tokens = [1] + [20 + (i * 7) % 19 for i in range(n_tokens - 1)]

    def embedding(size):
        monkeypatch.setenv("LLAMA_ARG_PROMPT_CHUNK", str(size))
        server.start()
        try:
            response = server.make_request("POST", "/v1/embeddings", data={"input": tokens})
            assert response.status_code == 200
            assert len(response.body["data"]) == 1
            return response.body["data"][0]["embedding"]
        finally:
            server.stop()

    expected = embedding(0)
    assert len(expected) > 1
    assert embedding(chunk) == expected, "chunk scheduling split an indivisible pooled embedding"
