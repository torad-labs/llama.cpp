# Goal: while one slot streams a reply, a long prompt read in another slot does not hold that reply back for a whole
# n_batch step. With --prompt-step N, the server adds at most N prompt tokens to a step in which another slot is
# generating, so the reply waits at most one short step between its tokens. A prompt read while no other slot
# generates still fills whole n_batch steps, and without --prompt-step nothing changes.
#
# A step's prompt tokens are what the read's streamed prompt_progress advances by between two reports.
import threading

import pytest
from utils import *

server: ServerProcess

N_BATCH = 256
STEP = 32
PROMPT = "Once upon a time there was a little girl who liked to read long stories. " * 60


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
        for data in res:
            if data.get("content") or data.get("tokens"):
                self.started.set()
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
    assert reply.started.wait(timeout=60), "the reply in slot 1 never streamed a token"
    assert not reply.done.is_set(), "the reply in slot 1 ended before the read began"
    steps, n_read = read_steps()
    still_generating = not reply.done.is_set()
    reply.thread.join(timeout=120)
    return steps, n_read, still_generating


def test_prompt_step_bounds_a_read_beside_a_reply():
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


def test_without_prompt_step_a_read_beside_a_reply_fills_its_batches():
    global server
    server.start()
    steps, n_read, still_generating = read_beside_a_reply()
    assert still_generating, "the reply in slot 1 ended during the read, so the read was not beside a reply"
    assert sum(steps) == n_read
    # the reply's token takes one place in the batch
    assert max(steps) >= N_BATCH - 2, f"without --prompt-step a step beside a reply read only {max(steps)}: {steps}"
