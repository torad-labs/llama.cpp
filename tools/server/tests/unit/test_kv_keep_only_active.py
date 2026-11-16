import os
import re
import tempfile
import threading
import pytest
import requests
from utils import *

server = ServerPreset.tinyllama2()

class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0
    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content
    def wait_for(self, text):
        content = ""
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            content += self.drain()
            if text in content:
                return content
            time.sleep(0.01)
        return content

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_predict = 4
    server.temperature = 0.0
    server.server_slots = True
    server.cache_ram = 100
    server.kv_unified = True
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    yield


LONG_PROMPT = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers. "
    "He met many creatures along the way including dragons and fairies "
    "and wizards who helped him on his noble quest to save the kingdom."
)


# idle slot cleared on launch should restore from cache-ram
def test_clear_and_restore():
    global server
    server.start()
    log = LogReader(server.log_path)

    # verify feature is enabled
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" in log.wait_for("__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__")

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    original_prompt_n = res.body["timings"]["prompt_n"]

    # Slot 0 is the only slot with KV — should NOT be cleared
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()

    # Launching slot 1 clears idle slot 0
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" in log.wait_for("__TEST_TAG_CACHE_IDLE_SLOT__")

    # Re-send same prompt — should restore from cache-ram
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "updating prompt cache" in log.wait_for("updating prompt cache")
    assert res.body["timings"]["cache_n"] > 0
    assert res.body["timings"]["prompt_n"] < original_prompt_n

    # Follow-up — slot 0 kept its KV, no clearing needed
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT + " The knight finally reached the castle gates.",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()


# the same, with the returning request naming the slot it was on: the slot is empty, and its prompt comes from cache-ram
def test_clear_and_restore_named_slot():
    global server
    server.start()
    log = LogReader(server.log_path)

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    original_prompt_n = res.body["timings"]["prompt_n"]

    # Launching slot 1 clears idle slot 0
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" in log.wait_for("__TEST_TAG_CACHE_IDLE_SLOT__")

    # Re-send the same prompt to slot 0 by id: it restores from cache-ram
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "updating prompt cache" in log.wait_for("updating prompt cache")
    assert res.body["timings"]["cache_n"] > 0
    assert res.body["timings"]["prompt_n"] < original_prompt_n


# LLAMA_STATE_RESTORE_ASYNC=1: the restore's rows are copied on a thread of the context while the server goes on. The
# returning request reuses as much of its prompt and replies as it does after a blocking restore
def test_clear_and_restore_async(monkeypatch):
    global server
    replies = []
    for switch in (None, "1"):
        if switch is None:
            monkeypatch.delenv("LLAMA_STATE_RESTORE_ASYNC", raising=False)
        else:
            monkeypatch.setenv("LLAMA_STATE_RESTORE_ASYNC", switch)
        server.start()
        log = LogReader(server.log_path)

        res = server.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT,
            "id_slot": 0,
            "cache_prompt": True,
        })
        assert res.status_code == 200

        # Launching slot 1 clears idle slot 0
        res = server.make_request("POST", "/completion", data={
            "prompt": "The quick brown fox",
            "id_slot": 1,
            "cache_prompt": True,
        })
        assert res.status_code == 200
        assert "__TEST_TAG_CACHE_IDLE_SLOT__" in log.wait_for("__TEST_TAG_CACHE_IDLE_SLOT__")

        # Slot 0 restores from cache-ram
        res = server.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT,
            "id_slot": 0,
            "cache_prompt": True,
        })
        assert res.status_code == 200
        assert res.body["timings"]["cache_n"] > 0
        restored = log.wait_for("cache_restore reason=restored")
        assert ("async=1" in restored) == (switch == "1")
        # the line gives the entry's size: its tokens, its bytes and the checkpoints among them, and the copies the restore makes
        assert re.search(r"cache_restore reason=restored slot=0 tokens=\d+ bytes=\d+ checkpoints=\d+ copies=[1-9]\d* ", restored)
        replies.append((res.body["content"], res.body["timings"]["cache_n"]))
        server.stop()

    assert replies[0] == replies[1]


OTHER_PROMPT = (
    "The lighthouse keeper counted the ships each evening and wrote their "
    "names in a ledger bound in green cloth, and on the night of the great "
    "storm he counted one more ship than had ever sailed from the harbour, "
    "a vessel with no lanterns that moved against the wind toward the rocks."
)


# LLAMA_STATE_SAVE_ASYNC=1: a conversation that takes the one slot from another has the slot's state copied to the prompt
# cache on a thread of the context, its task deferred until the copy is done. Both conversations come back from the cache
# with as many tokens reused and the same replies as after blocking saves
def test_exchange_save_async(monkeypatch):
    global server
    replies = []
    for switch in (None, "1"):
        if switch is None:
            monkeypatch.delenv("LLAMA_STATE_SAVE_ASYNC", raising=False)
        else:
            monkeypatch.setenv("LLAMA_STATE_SAVE_ASYNC", switch)
        server.n_slots = 1
        server.start()
        log = LogReader(server.log_path)

        turns = []
        for prompt in (LONG_PROMPT, OTHER_PROMPT, LONG_PROMPT, OTHER_PROMPT):
            res = server.make_request("POST", "/completion", data={
                "prompt": prompt,
                "cache_prompt": True,
            })
            assert res.status_code == 200
            turns.append((res.body["content"], res.body["timings"]["cache_n"]))
        # the third and the fourth come back from the cache
        assert turns[2][1] > 0 and turns[3][1] > 0
        assert turns[0][0] == turns[2][0] and turns[1][0] == turns[3][0]
        content = log.wait_for("cache_restore reason=restored")
        assert ("cache_save reason=saving" in content) == (switch == "1")
        assert ("save=saving" in content) == (switch == "1")
        replies.append(turns)
        server.stop()

    assert replies[0] == replies[1]


# LLAMA_PROMPT_CACHE_DIR: the sessions of a server that is stopped are files in the directory, and the server started next
# with the same model takes them into its prompt cache. Both conversations come back with as many tokens reused and the
# same replies as on a server that never stopped
def test_sessions_survive_a_stop(monkeypatch, tmp_path):
    global server
    prompts = (LONG_PROMPT, OTHER_PROMPT)

    def turn(prompt):
        res = server.make_request("POST", "/completion", data={"prompt": prompt, "cache_prompt": True})
        assert res.status_code == 200
        return res.body["content"], res.body["timings"]["cache_n"]

    def files():
        return list(tmp_path.glob("*/*.state"))

    # without the directory: each conversation's second turn on the server that had its first
    monkeypatch.delenv("LLAMA_PROMPT_CACHE_DIR", raising=False)
    server.start()
    first = [turn(prompt) for prompt in prompts]
    again = [turn(prompt) for prompt in prompts]
    server.stop()
    assert all(cache_n > 0 for _, cache_n in again)
    assert not files()

    monkeypatch.setenv("LLAMA_PROMPT_CACHE_DIR", str(tmp_path))
    server.start()
    assert [turn(prompt) for prompt in prompts] == first
    server.stop()
    assert len(files()) == 2
    assert all(path.stat().st_mode & 0o077 == 0 for path in files())

    # the second turns, on a server started after the stop
    server.start()
    log = LogReader(server.log_path)
    assert re.search(r"prompt cache: 2 snapshots, .* taken from", log.wait_for("taken from"))
    assert not files()
    assert [turn(prompt) for prompt in prompts] == again
    assert "cache_restore reason=restored" in log.wait_for("cache_restore reason=restored")
    server.stop()
    # and that server left them again, for the one after it
    assert len(files()) == 2


# a request for a busy slot waits for it: the prompt cache does not save or replace the slot's state under the running request
def test_named_busy_slot_left_alone():
    global server
    server.start()
    log = LogReader(server.log_path)

    other = {"prompt": "The little dog ran to the park with a red ball.", "id_slot": 0, "cache_prompt": True}
    res = server.make_request("POST", "/completion", data=other)
    assert res.status_code == 200

    # Launching slot 1 saves idle slot 0 to cache-ram and clears it
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    long_run = {"prompt": LONG_PROMPT, "id_slot": 0, "n_predict": 160, "ignore_eos": True, "cache_prompt": False}
    res = server.make_request("POST", "/completion", data=long_run)
    assert res.status_code == 200
    expected = res.body["content"]
    log.drain()

    # the same run streamed, and once slot 0 generates, the other prompt asks for slot 0: its cached state is the best match
    other_res = []
    thread = None
    content = ""
    for data in server.make_stream_request("POST", "/completion", data={**long_run, "stream": True}):
        if thread is None:
            thread = threading.Thread(target=lambda: other_res.append(server.make_request("POST", "/completion", data=other)))
            thread.start()
        if not data["stop"]:
            content += data["content"]
    assert thread is not None
    thread.join()
    assert other_res[0].status_code == 200

    if "requested slot is unavailable" not in log.drain():
        pytest.skip("the other request did not reach the server while slot 0 was generating")  # ty: ignore[too-many-positional-arguments]
    assert content == expected


def test_disabled_with_flag():
    global server
    server.no_cache_idle_slots = True
    server.start()
    log = LogReader(server.log_path)

    # Feature should not be enabled
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" not in log.drain()

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    # Request on different slot — should NOT trigger clearing
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()


def test_ram_prefix_beats_short_resident_prefix():
    server.start()
    long_prompt = LONG_PROMPT * 2
    short_prompt = LONG_PROMPT[:90]
    first = server.make_request("POST", "/completion", data={
        "prompt": long_prompt, "id_slot": 0, "cache_prompt": True,
    })
    assert first.status_code == 200
    # Admission of slot 1 happens before idle slot 0 is saved. This leaves the
    # long prefix in RAM and the short exact prefix resident in slot 1.
    short = server.make_request("POST", "/completion", data={
        "prompt": short_prompt, "id_slot": 1, "cache_prompt": True, "n_predict": 1,
    })
    assert short.status_code == 200
    resumed = server.make_request("POST", "/completion", data={
        "prompt": long_prompt + " The tool returned the location of the sword.",
        "cache_prompt": True,
    })
    assert resumed.status_code == 200
    assert resumed.body["timings"]["cache_n"] >= first.body["timings"]["prompt_n"] - 1


def test_idle_slot_kept_when_snapshot_exceeds_ram_limit():
    server.cache_ram = 1
    server.n_ctx = 4096
    server.start()
    log = LogReader(server.log_path)
    first = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT * 10, "id_slot": 0, "cache_prompt": True, "n_predict": 1,
    })
    assert first.status_code == 200
    other = server.make_request("POST", "/completion", data={
        "prompt": "A different conversation", "id_slot": 1, "cache_prompt": True,
    })
    assert other.status_code == 200
    assert "exceeds cache size limit" in log.wait_for("exceeds cache size limit")
    for _ in range(2):
        other = server.make_request("POST", "/completion", data={
            "prompt": "A different conversation", "id_slot": 1, "cache_prompt": True,
        })
        assert other.status_code == 200
        unchanged = log.wait_for("cache_save reason=unchanged_refusal")
        assert "cache_save reason=unchanged_refusal" in unchanged
        assert "exceeds cache size limit" not in unchanged

    resumed = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT * 10 + " The tool returned successfully.",
        "id_slot": 0, "cache_prompt": True,
    })
    assert resumed.status_code == 200
    assert resumed.body["timings"]["cache_n"] >= first.body["timings"]["prompt_n"] - 1

    # Reusing the slot changes its target, draft and checkpoint state, even when the cache budget does not change.
    shortened = server.make_request("POST", "/completion", data={
        "prompt": "A small replacement", "id_slot": 0, "cache_prompt": True, "n_predict": 1,
    })
    assert shortened.status_code == 200
    log.drain()
    other = server.make_request("POST", "/completion", data={
        "prompt": "A different conversation", "id_slot": 1, "cache_prompt": True,
    })
    assert other.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" in log.wait_for("__TEST_TAG_CACHE_IDLE_SLOT__")


def test_kv_pressure_preserves_idle_tool_continuation():
    server.no_cache_idle_slots = True
    server.n_ctx = 512
    server.n_batch = 128
    server.n_ubatch = 128
    server.start()
    log = LogReader(server.log_path)
    tokenized = server.make_request("POST", "/tokenize", data={"content": LONG_PROMPT * 5})
    assert tokenized.status_code == 200
    tokens = tokenized.body["tokens"]
    assert len(tokens) > 400
    first = server.make_request("POST", "/completion", data={
        "prompt": tokens[:350], "id_slot": 0, "cache_prompt": True, "n_predict": 1,
    })
    assert first.status_code == 200
    other = server.make_request("POST", "/completion", data={
        "prompt": list(reversed(tokens[:350])), "id_slot": 1, "cache_prompt": True, "n_predict": 1,
    })
    assert other.status_code == 200
    assert "purging slot 0" in log.wait_for("purging slot 0")
    resumed = server.make_request("POST", "/completion", data={
        "prompt": tokens[:370], "id_slot": 0, "cache_prompt": True, "n_predict": 1,
    })
    assert resumed.status_code == 200
    assert resumed.body["timings"]["cache_n"] >= 349


# A conversation parked in RAM stays recoverable whatever takes its slot next: restoring it for a few shared tokens
# and then cutting it to that prefix would leave it in neither KV nor RAM.
def _park_long_prompt_in_ram(log):
    res = server.make_request("POST", "/completion", data={"prompt": LONG_PROMPT, "id_slot": 0, "cache_prompt": True})
    assert res.status_code == 200
    prompt_n = res.body["timings"]["prompt_n"]
    res = server.make_request("POST", "/completion", data={"prompt": "The quick brown fox", "id_slot": 1, "cache_prompt": True})
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" in log.wait_for("__TEST_TAG_CACHE_IDLE_SLOT__")
    return prompt_n


def test_parked_conversation_survives_a_short_shared_prefix():
    global server
    server.start()
    log = LogReader(server.log_path)
    original_prompt_n = _park_long_prompt_in_ram(log)
    # a new conversation shares only its first few tokens with the parked one
    res = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time there was a small grey cat who liked fish and slept in the sun all day long",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] < original_prompt_n // 4
    # the parked conversation returns whole
    res = server.make_request("POST", "/completion", data={"prompt": LONG_PROMPT, "cache_prompt": True})
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] >= original_prompt_n - 2, res.body["timings"]


def test_parked_conversation_survives_a_partial_restore():
    global server
    server.start()
    log = LogReader(server.log_path)
    original_prompt_n = _park_long_prompt_in_ram(log)
    # a new conversation shares its first half with the parked one: the half is restored, the rest cut
    half = " ".join(LONG_PROMPT.split()[:28]) + " and then everything changed for the better and nothing was the same again"
    res = server.make_request("POST", "/completion", data={"prompt": half, "cache_prompt": True})
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] > original_prompt_n // 4, res.body["timings"]
    # the parked conversation returns whole
    res = server.make_request("POST", "/completion", data={"prompt": LONG_PROMPT, "cache_prompt": True})
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] >= original_prompt_n - 2, res.body["timings"]


# With no byte limit the context size caps the parked tokens. A save that cannot fit beside the protected incoming
# snapshot is refused before it evicts anyone: other conversations never pay for a save that stores nothing.
def test_unlimited_byte_budget_refuses_a_save_that_cannot_fit_before_evicting():
    global server
    server.cache_ram = -1
    server.n_ctx = 256  # the smallest context llama.cpp allows (n_ctx is padded to a multiple of 256)
    server.start()
    x = " ".join(["alpha beta gamma delta"] * 8)
    y = " ".join(["red green blue"] * 3)
    i = " ".join(["north south east west"] * 12)
    rx = server.make_request("POST", "/completion", data={"prompt": x, "id_slot": 0, "cache_prompt": True})
    ry = server.make_request("POST", "/completion", data={"prompt": y, "id_slot": 1, "cache_prompt": True})
    ri = server.make_request("POST", "/completion", data={"prompt": i, "id_slot": 0, "cache_prompt": True})
    assert rx.status_code == 200 and ry.status_code == 200 and ri.status_code == 200
    nx, ny, ni = (r.body["timings"]["prompt_n"] for r in (rx, ry, ri))
    assert nx + ni > 256, (nx, ny, ni)        # X and I cannot be parked together
    assert nx + ny <= 256 - 16, (nx, ny, ni)  # X and Y can
    # X returns to the slot holding I: I cannot be parked beside the protected X; Y must not pay for the attempt
    res = server.make_request("POST", "/completion", data={"prompt": x, "id_slot": 0, "cache_prompt": True})
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] >= nx - 2, res.body["timings"]
    res = server.make_request("POST", "/completion", data={"prompt": y, "cache_prompt": True})
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] >= ny - 2, res.body["timings"]


def test_consumed_restore_does_not_inflate_future_token_evictions():
    global server
    server.cache_ram = -1
    server.n_ctx = 512
    server.n_slots = 2
    server.start()

    a = " ".join(["alpha beta gamma delta"] * 12)
    b = " ".join(["red green blue"] * 15)
    c = " ".join(["north south east west"] * 10)

    def complete(prompt):
        res = server.make_request("POST", "/completion", data={"prompt": prompt, "cache_prompt": True, "n_predict": 1})
        assert res.status_code == 200
        return res.body["timings"]

    cold_a = complete(a)
    complete(b)
    # A consumes its RAM snapshot on return. Its tokens must leave the cache-wide total before the prompt moves out.
    assert complete(a)["cache_n"] >= cold_a["prompt_n"] - 2
    complete(c)
    # A and B still fit beside one another, so B must not be evicted by A's already-consumed tokens.
    assert complete(b)["cache_n"] > 0


def test_consumed_restore_retries_idle_cache_refusal():
    server.cache_ram = -1
    server.n_ctx = 512
    server.n_slots = 3
    server.start()
    log = LogReader(server.log_path)

    def complete(tokens, slot):
        res = server.make_request("POST", "/completion", data={
            "prompt": tokens, "id_slot": slot, "cache_prompt": True, "n_predict": 1,
        })
        assert res.status_code == 200
        return res.body["timings"]

    # Two RAM snapshots nearly fill the 512-token budget; each fits beside the idle resident in KV.
    complete([12] * 300, 2)
    complete([11] * 200, 1)
    complete([13] * 200, 0)
    log.drain()
    complete([14], 2)
    refused = log.drain()
    assert "cache_idle action=retain reason=no_space" in refused

    # Restoring into empty slot 1 consumes its snapshot. Slot 0 is checked before slot 2 can change the cache.
    restored = complete([11] * 200 + [15], 1)
    assert restored["cache_n"] >= 199
    retried = log.wait_for("cache_save reason=saved slot=0")
    assert "consumed=1" in retried
    assert "cache_save reason=saved slot=0" in retried
    assert "cache_save reason=unchanged_refusal slot=0" not in retried


def test_admission_eviction_preserves_recent_snapshot():
    global server
    server.cache_ram = 2
    server.n_slots = 1
    server.n_ctx = 2048
    server.start()
    log = LogReader(server.log_path)

    prompts = [
        "alpha beta gamma delta " * 70,
        "red green blue " * 60,
        "north south east west " * 40,
        "orange purple silver gold " * 50,
        "one two three four " * 60,
        "silver bronze copper iron " * 50,
        "black white gray " * 40,
    ]
    for prompt in prompts:
        res = server.make_request("POST", "/completion", data={"prompt": prompt, "cache_prompt": True, "n_predict": 1})
        assert res.status_code == 200
    assert "cache_evict" in log.drain(), log.path

    # The previous prompt is no longer resident. It must survive bounded update() eviction in RAM.
    res = server.make_request("POST", "/completion", data={"prompt": prompts[-2], "cache_prompt": True, "n_predict": 1})
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] > 0


# Conversations that start with the same system prompt.
# A request that shares only that prefix with a slot does not continue the slot's conversation: it keeps under half of what the slot holds.
SHARED_PREFIX = LONG_PROMPT
TAIL_A = " The knight rode north for many days and many nights." * 12
TAIL_C = " A wizard read old books by the fire every evening." * 12


def _shared_prefix_server(n_slots):
    global server
    server.no_cache_idle_slots = True
    server.n_slots = n_slots
    server.n_ctx = 2048
    server.start()
    return LogReader(server.log_path)


def _one_token(prompt):
    res = server.make_request("POST", "/completion", data={"prompt": prompt, "cache_prompt": True, "n_predict": 1})
    assert res.status_code == 200
    return res


# With an empty slot free, the request takes it: the conversation in the other slot is not saved and cut.
def test_shared_prefix_takes_an_empty_slot():
    log = _shared_prefix_server(3)
    first = _one_token(SHARED_PREFIX + TAIL_A)
    log.drain()
    other = _one_token(SHARED_PREFIX + " A small grey cat slept in the sun all day.")
    content = log.wait_for("selected slot by LRU")
    assert other.body["id_slot"] != first.body["id_slot"]
    assert "an empty slot is free" in content
    assert "selected slot by LCP similarity" not in content
    assert "cache_save reason=saved" not in content
    # the first conversation continues in its slot, from the cards
    again = _one_token(SHARED_PREFIX + TAIL_A + " Then he saw the castle.")
    content = log.wait_for("selected slot by LCP similarity")
    assert again.body["id_slot"] == first.body["id_slot"]
    assert "cache_restore" not in content
    assert again.body["timings"]["cache_n"] >= first.body["timings"]["prompt_n"] - 1


TAIL_E = " A small grey cat slept in the sun all day." * 12


def _held(res):
    return res.body["timings"]["cache_n"] + res.body["timings"]["prompt_n"]


# Cost includes both resident tokens and idle age. The legacy switch keeps LRU.
# The age counts by the chance of a return, which fades over LLAMA_SLOT_VICTIM_RETURN_S seconds (90 by default): half a second here, so a few seconds age a conversation.
def test_shared_prefix_takes_the_slot_that_costs_least_to_cut(monkeypatch):
    monkeypatch.setenv("LLAMA_SLOT_VICTIM_RETURN_S", "0.5")
    for legacy in (None, "1"):
        if legacy is None:
            monkeypatch.delenv("LLAMA_SLOT_VICTIM_LRU_LEGACY", raising=False)
        else:
            monkeypatch.setenv("LLAMA_SLOT_VICTIM_LRU_LEGACY", legacy)
        log = _shared_prefix_server(2)
        long = _one_token(SHARED_PREFIX + TAIL_A * 3)
        short = _one_token(SHARED_PREFIX + TAIL_C)
        assert short.body["id_slot"] != long.body["id_slot"]
        # Refresh both cached prompts in one batch so model load and prefill time do not separate their last uses.
        refreshed = server.make_request("POST", "/completion", data={
            "prompt": [SHARED_PREFIX + TAIL_A * 3, SHARED_PREFIX + TAIL_C],
            "cache_prompt": True, "n_predict": 1,
        })
        assert refreshed.status_code == 200
        assert [item["id_slot"] for item in refreshed.body] == [long.body["id_slot"], short.body["id_slot"]]
        log.drain()
        # Both are recent: the short conversation costs less, but the long one is LRU.
        cut = long if legacy else short
        other = _one_token(SHARED_PREFIX + TAIL_E)
        content = log.wait_for("cache_save reason=saved")
        assert other.body["id_slot"] == cut.body["id_slot"]
        assert ("selected slot by LRU" if legacy else "selected slot by cost") in content
        assert ("selected slot by cost" if legacy else "selected slot by LRU") not in content
        assert re.search(rf"cache_save reason=saved slot={cut.body['id_slot']} tokens=\d+ bytes=\d+ checkpoints=\d+ copies=[1-9]\d*\n", content)
        if legacy is None:
            # Age the long conversation enough to outweigh its size, then refresh the short one: after 6 s its chance of a
            # return is 0.17 + 0.83 / 13 = 0.23, and the refreshed one's is over 0.76 for its first 200 ms, so the long one
            # costs less while it holds under 3.2 times the tokens.
            assert _held(long) < 3 * _held(other)
            time.sleep(6)
            more = _one_token(SHARED_PREFIX + TAIL_E + " Then it woke up.")
            assert more.body["id_slot"] == other.body["id_slot"]
            log.drain()
            third = _one_token(SHARED_PREFIX + " A wizard read a letter and then went to sleep in the tower.")
            content = log.wait_for("cache_save reason=saved")
            assert third.body["id_slot"] == long.body["id_slot"]
            assert "selected slot by cost" in content
        back = _one_token(SHARED_PREFIX + (TAIL_A * 3 if legacy else TAIL_C) + " Then the fire went out.")
        assert back.body["timings"]["cache_n"] >= _held(cut) - 1
        server.stop()


# A few seconds away take little from a conversation's chance of a return (0.97 of it is left after 3 s at the default 90 s),
# so the conversation of twice the size stays and the one just used is cut. LLAMA_SLOT_VICTIM_IDLE_LEGACY=1 divides the tokens
# by the seconds idle instead, and those seconds outweigh the size.
def test_shared_prefix_weighs_idle_seconds_by_the_chance_of_a_return(monkeypatch):
    monkeypatch.delenv("LLAMA_SLOT_VICTIM_LRU_LEGACY", raising=False)
    monkeypatch.delenv("LLAMA_SLOT_VICTIM_RETURN_S", raising=False)
    for legacy in (None, "1"):
        if legacy is None:
            monkeypatch.delenv("LLAMA_SLOT_VICTIM_IDLE_LEGACY", raising=False)
        else:
            monkeypatch.setenv("LLAMA_SLOT_VICTIM_IDLE_LEGACY", legacy)
        log = _shared_prefix_server(2)
        long = _one_token(SHARED_PREFIX + TAIL_A * 3)
        short = _one_token(SHARED_PREFIX + TAIL_C)
        assert short.body["id_slot"] != long.body["id_slot"]
        assert _held(long) > 1.5 * _held(short)
        time.sleep(_held(long) / _held(short) + 0.5)
        more = _one_token(SHARED_PREFIX + TAIL_C + " Then the fire went out.")
        assert more.body["id_slot"] == short.body["id_slot"]
        log.drain()
        third = _one_token(SHARED_PREFIX + TAIL_E)
        content = log.wait_for("cache_save reason=saved")
        assert third.body["id_slot"] == (long if legacy else short).body["id_slot"]
        assert "selected slot by cost" in content
        server.stop()


def _reply(prompt):
    res = server.make_request("POST", "/completion", data={"prompt": prompt, "cache_prompt": True, "n_predict": 8})
    assert res.status_code == 200
    return res


# LLAMA_SLOT_PREFIX_COPY=1: the request that takes an empty slot starts from a copy of the slot it shares the prefix with, and reads only what follows the prefix.
# Both conversations then continue in their slots, and every reply is the one the server gives without the copy.
def test_shared_prefix_is_copied_to_the_empty_slot(monkeypatch):
    replies = []
    for switch in (None, "1"):
        if switch is None:
            monkeypatch.delenv("LLAMA_SLOT_PREFIX_COPY", raising=False)
        else:
            monkeypatch.setenv("LLAMA_SLOT_PREFIX_COPY", switch)
        log = _shared_prefix_server(3)
        n_shared = len(server.make_request("POST", "/tokenize", data={"content": SHARED_PREFIX}).body["tokens"])
        first = _reply(SHARED_PREFIX + TAIL_A)
        log.drain()
        other = _reply(SHARED_PREFIX + TAIL_C)
        content = log.wait_for("selected slot by LRU")
        assert other.body["id_slot"] != first.body["id_slot"]
        assert "an empty slot is free" in content
        assert "cache_save reason=saved" not in content
        assert ("copied the prompt of slot %d," % first.body["id_slot"] in content) == (switch == "1")
        if switch == "1":
            assert other.body["timings"]["cache_n"] >= n_shared - 2
        else:
            assert other.body["timings"]["cache_n"] == 0
        n_other = other.body["timings"]["cache_n"] + other.body["timings"]["prompt_n"]
        # the slot copied from keeps everything, and the copy keeps what its own request read
        again = _reply(SHARED_PREFIX + TAIL_A + " Then he saw the castle.")
        assert again.body["id_slot"] == first.body["id_slot"]
        assert again.body["timings"]["cache_n"] >= first.body["timings"]["prompt_n"] - 1
        more = _reply(SHARED_PREFIX + TAIL_C + " Then the fire went out.")
        assert more.body["id_slot"] == other.body["id_slot"]
        assert more.body["timings"]["cache_n"] >= n_other - 1
        replies.append([res.body["content"] for res in (first, other, again, more)])
        server.stop()
    assert replies[0] == replies[1]


def test_young_branches_keep_separate_slots(monkeypatch):
    monkeypatch.setenv("LLAMA_SLOT_PREFIX_COPY", "1")
    log = _shared_prefix_server(3)
    prefix = SHARED_PREFIX * 4
    first = _reply(prefix + " The red bird sings.")
    log.drain()
    sibling = _reply(prefix + " The blue cat sleeps.")
    assert sibling.body["id_slot"] != first.body["id_slot"]
    assert "cache_save reason=saved" not in log.drain()
    again = _reply(prefix + " The red bird sings. What happens next?")
    assert again.body["id_slot"] == first.body["id_slot"]
    assert again.body["timings"]["cache_n"] >= first.body["timings"]["prompt_n"] - 1
    assert sibling.body["timings"]["cache_n"] > first.body["timings"]["prompt_n"] // 2


@pytest.mark.parametrize("copy_prefix", [False, True])
def test_editing_the_last_prompt_keeps_its_prefix(monkeypatch, copy_prefix):
    if copy_prefix:
        monkeypatch.setenv("LLAMA_SLOT_PREFIX_COPY", "1")
    else:
        monkeypatch.delenv("LLAMA_SLOT_PREFIX_COPY", raising=False)
    log = _shared_prefix_server(2)
    prefix = SHARED_PREFIX * 6
    first = _reply(prefix + " The red bird sings.")
    log.drain()
    edited = _reply(prefix + " The blue cat sleeps.")
    assert edited.body["timings"]["cache_n"] > first.body["timings"]["prompt_n"] // 2
    assert (edited.body["id_slot"] != first.body["id_slot"]) == copy_prefix
    assert "cache_save reason=saved" not in log.drain()


@pytest.mark.parametrize("async_state", [False, True])
def test_prefix_donor_does_not_choose_the_victim(monkeypatch, async_state):
    monkeypatch.setenv("LLAMA_SLOT_PREFIX_COPY", "1")
    if async_state:
        monkeypatch.setenv("LLAMA_STATE_SAVE_ASYNC", "1")
        monkeypatch.setenv("LLAMA_STATE_RESTORE_ASYNC", "1")
    log = _shared_prefix_server(3)
    prefix = SHARED_PREFIX * 4
    prompts = [prefix + TAIL_A * 2, prefix + TAIL_C, "Once upon a time there was a small cat."]
    held = [_one_token(prompt) for prompt in prompts]
    assert len({res.body["id_slot"] for res in held}) == 3
    refreshed = server.make_request("POST", "/completion", data={
        "prompt": prompts, "cache_prompt": True, "n_predict": 1,
    })
    assert refreshed.status_code == 200
    log.drain()
    branch = _reply(prefix + TAIL_E)
    assert branch.body["id_slot"] == held[2].body["id_slot"]
    n_shared = len(server.make_request("POST", "/tokenize", data={"content": prefix}).body["tokens"])
    assert branch.body["timings"]["cache_n"] >= n_shared - 2
    assert "cache_save reason=saved" in log.drain()
    for prompt, previous in zip(prompts[:2], held[:2]):
        continued = _reply(prompt + " What happens next?")
        assert continued.body["id_slot"] == previous.body["id_slot"]
        assert continued.body["timings"]["cache_n"] >= _held(previous) - 1


def test_restored_input_boundary_uses_the_existing_reuse_rule(monkeypatch, tmp_path):
    monkeypatch.setenv("LLAMA_SLOT_PREFIX_COPY", "1")
    server.slot_save_path = str(tmp_path)
    log = _shared_prefix_server(2)
    prefix = SHARED_PREFIX * 6
    first = _reply(prefix + " The red bird sings.")
    slot = first.body["id_slot"]
    for action in ("save", "restore"):
        res = server.make_request("POST", f"/slots/{slot}?action={action}", data={"filename": "boundary.bin"})
        assert res.status_code == 200
    log.drain()
    edited = _reply(prefix + " The blue cat sleeps.")
    assert edited.body["id_slot"] == slot
    assert edited.body["timings"]["cache_n"] > first.body["timings"]["prompt_n"] // 2
    assert "cache_save reason=saved" not in log.drain()


def test_young_branch_is_saved_when_every_slot_is_occupied(monkeypatch):
    monkeypatch.setenv("LLAMA_SLOT_PREFIX_COPY", "1")
    log = _shared_prefix_server(1)
    prefix = SHARED_PREFIX * 4
    first = _reply(prefix + " The red bird sings.")
    log.drain()
    _reply(prefix + " The blue cat sleeps.")
    assert "cache_save reason=saved" in log.drain()
    again = _reply(prefix + " The red bird sings. What happens next?")
    assert again.body["timings"]["cache_n"] >= first.body["timings"]["prompt_n"] - 1


@pytest.mark.parametrize("cache_prompt", [False, True])
@pytest.mark.parametrize("identical", [False, True])
def test_simultaneous_requests_read_the_shared_prefix_once(monkeypatch, cache_prompt, identical):
    replies = []
    for copy_prefix in (False, True):
        if copy_prefix:
            monkeypatch.setenv("LLAMA_SLOT_PREFIX_COPY", "1")
        else:
            monkeypatch.delenv("LLAMA_SLOT_PREFIX_COPY", raising=False)
        server.no_cache_idle_slots = True
        server.n_slots = 4
        server.n_ctx = 8192
        server.n_batch = 128
        server.n_ubatch = 128
        server.start()
        prefix = SHARED_PREFIX * 6
        n_shared = len(server.make_request("POST", "/tokenize", data={"content": prefix}).body["tokens"])
        tails = [" The red bird sings.", " The blue cat sleeps.", " The green frog jumps."]
        prompts = [prefix + (tails[0] if identical else tail) for tail in tails]
        res = server.make_request("POST", "/completion", data={
            "prompt": prompts, "cache_prompt": cache_prompt, "n_predict": 8,
        })
        assert res.status_code == 200
        assert len(res.body) == len(prompts)
        assert len({item["id_slot"] for item in res.body}) == len(prompts)
        cached = sorted(item["timings"]["cache_n"] for item in res.body)
        if copy_prefix and cache_prompt:
            assert cached[0] == 0
            assert cached[1] >= n_shared - 2
            assert cached[2] >= n_shared - 2
        else:
            assert cached == [0, 0, 0]
        replies.append([item["content"] for item in res.body])
        server.stop()
    assert replies[0] == replies[1]


def test_cancelled_prefix_source_releases_its_waiter(monkeypatch):
    monkeypatch.setenv("LLAMA_SLOT_PREFIX_COPY", "1")
    server.no_cache_idle_slots = True
    server.n_slots = 2
    server.n_ctx = 4096
    server.n_batch = 2
    server.n_ubatch = 2
    server.n_threads = 1
    server.start()
    log = LogReader(server.log_path)
    prefix = SHARED_PREFIX * 8
    result = []
    errors = []

    def finish_follower():
        try:
            result.append(_reply(prefix + " The blue cat sleeps."))
        except Exception as error:
            errors.append(error)

    follower = threading.Thread(target=finish_follower)
    try:
        with requests.post(server.make_url("/completion"), json={
            "prompt": prefix + " The red bird sings.",
            "cache_prompt": True, "n_predict": 256,
            "stream": True, "return_progress": True,
        }, stream=True, timeout=DEFAULT_REQUEST_TIMEOUT) as source:
            assert source.status_code == 200, source.text
            follower.start()
            assert "prefix tokens from slot" in log.wait_for("prefix tokens from slot")
        assert "prefix source ended" in log.wait_for("prefix source ended")
    finally:
        if follower.ident is not None:
            follower.join(timeout=DEFAULT_REQUEST_TIMEOUT)
    assert not follower.is_alive()
    assert not errors
    assert len(result) == 1
    assert result[0].body["timings"]["cache_n"] == 0


# A slot that holds cells another slot holds is never shifted: with the switch a request's n_cache_reuse is ignored.
# The request below repeats the first conversation's tail a few tokens later, which is what cache reuse shifts, and the first conversation then replies as it does without the copy.
def test_copied_prefix_is_not_shifted(monkeypatch):
    replies = []
    for switch in (None, "1"):
        if switch is None:
            monkeypatch.delenv("LLAMA_SLOT_PREFIX_COPY", raising=False)
        else:
            monkeypatch.setenv("LLAMA_SLOT_PREFIX_COPY", switch)
        log = _shared_prefix_server(3)
        first = _reply(SHARED_PREFIX + TAIL_A)
        log.drain()
        other = server.make_request("POST", "/completion", data={
            "prompt": SHARED_PREFIX + " A wizard read old books by the fire." + TAIL_A,
            "cache_prompt": True,
            "n_predict": 8,
            "n_cache_reuse": 4,
        })
        assert other.status_code == 200
        content = log.wait_for("ignoring n_cache_reuse")
        assert other.body["id_slot"] != first.body["id_slot"]
        assert ("copied the prompt of slot %d," % first.body["id_slot"] in content) == (switch == "1")
        assert ("ignoring n_cache_reuse" in content) == (switch == "1")
        again = _reply(SHARED_PREFIX + TAIL_A + " Then he saw the castle.")
        assert again.body["id_slot"] == first.body["id_slot"]
        assert again.body["timings"]["cache_n"] >= first.body["timings"]["prompt_n"] - 1
        replies.append([res.body["content"] for res in (first, other, again)])
        server.stop()
    assert replies[0] == replies[1]
