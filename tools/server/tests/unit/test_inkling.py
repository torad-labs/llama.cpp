# Inkling 2 through this server: the synthetic inkling2 model test-llama-archs writes, the official template and a
# synthetic inkling2v projector. Its tokens are "tok_<id>" and decode to hex ids, so a reply's text means nothing;
# what the tests read is the prompt the server renders, the status it answers with and the ids it samples.
import base64
import json
import random
from concurrent.futures import ThreadPoolExecutor

import pytest

import inkling_fixture as fx
from utils import *

MARKER = "<__media_inkling__>"
# the stand-in for <|content_model_end_sampling|>, the end token a converted pack declares as its eos (and eot)
END_ID = 100
SYSTEM = "You answer briefly."
TURNS = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "Name a colour."},
    {"role": "assistant", "content": "Blue."},
    {"role": "user", "content": "Another one."},
]
EFFORT = {"none": "0", "minimal": "0.1", "low": "0.2", "medium": "0.7", "high": "0.9", "xhigh": "0.99", "max": "0.99"}


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    folder = tmp_path_factory.mktemp("inkling")
    model = fx.write_model(str(folder))
    mmproj = str(folder / "mmproj-inkling2v.gguf")
    image_tokens = fx.write_mmproj(mmproj, fx.read_u32(model, "inkling2.embedding_length"))
    return {"model": model, "mmproj": mmproj, "image_tokens": image_tokens,
            "vocab": fx.read_u32(model, "inkling2.vocab_size"), "unpadded": fx.read_u32(model, "inkling2.vocab_size_unpadded")}


@pytest.fixture
def server(files, monkeypatch):
    monkeypatch.setenv("LLAMA_MEDIA_MARKER", MARKER)
    s = ServerProcess()
    s.offline = True
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = files["model"]
    s.mmproj_file = files["mmproj"]
    s.model_alias = "inkling2"
    s.jinja = True
    s.chat_template_file = fx.TEMPLATE
    s.n_ctx = 4096
    s.n_slots = 2
    s.n_gpu_layer = 99
    s.override_kv = f"tokenizer.ggml.eos_token_id=int:{END_ID}"
    s.start()
    return s


def render(server, body):
    res = server.make_request("POST", "/apply-template", data=body)
    assert res.status_code == 200, res.body
    return res.body["prompt"]


def reply(server, body):
    res = server.make_request("POST", "/chat/completions", data={"max_tokens": 4, "temperature": 1.0, "seed": 7, **body})
    assert res.status_code == 200, res.body
    choice = res.body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    return res.body


def assert_in_order(prompt, *parts):
    pos = 0
    for part in parts:
        found = prompt.find(part, pos)
        assert found >= 0, f"{part!r} is not in the prompt after position {pos}: {prompt!r}"
        pos = found + len(part)


def test_inkling_text_turns_and_system_prompt(server):
    prompt = render(server, {"messages": TURNS})
    assert prompt.startswith("<|message_system|><|content_text|>" + SYSTEM + "<|end_message|>")
    assert_in_order(prompt, "<|message_user|><|content_text|>Name a colour.<|end_message|>",
                    "<|message_model|><|content_text|>Blue.<|end_message|><|content_model_end_sampling|>",
                    "<|message_user|><|content_text|>Another one.<|end_message|>")
    body = reply(server, {"messages": TURNS})
    # the reply was read from the rendered prompt, not from another one
    tokens = server.make_request("POST", "/tokenize", data={"content": prompt, "parse_special": True}).body["tokens"]
    assert body["usage"]["prompt_tokens"] >= len(tokens)


@pytest.mark.parametrize("level", list(EFFORT))
def test_inkling_reasoning_effort(server, level):
    prompt = render(server, {"messages": TURNS, "reasoning_effort": level})
    assert prompt.count("Thinking effort level: ") == 1
    assert "<|message_system|><|content_text|>Thinking effort level: " + EFFORT[level] + "<|end_message|>" in prompt
    reply(server, {"messages": TURNS, "reasoning_effort": level})


def test_inkling_reasoning_effort_defaults_to_high(server):
    assert "Thinking effort level: 0.9<|end_message|>" in render(server, {"messages": TURNS})
    assert render(server, {"messages": TURNS, "reasoning_effort": None}) == render(server, {"messages": TURNS})


def test_inkling_web_chat_off_sends_zero_effort(server):
    off = {"chat_template_kwargs": {"enable_thinking": False}}
    assert render(server, {"messages": TURNS, **off}) == render(server, {"messages": TURNS, "reasoning_effort": "none"})


def test_inkling_reasoning_levels_across_clients(files, monkeypatch):
    monkeypatch.setenv("LLAMA_SERVER_SLOTS_DEBUG", "1")
    s = ServerProcess()
    s.model_hf_repo = s.model_hf_file = None
    s.model_file = files["model"]
    s.jinja = s.offline = s.server_slots = True
    s.chat_template_file = fx.TEMPLATE
    s.n_ctx, s.n_slots = 1024, 2
    s.start()
    messages = [{"role": "user", "content": "Which part needs a test?"}]

    def processed_prompt(path, fields, slot=0):
        res = s.make_request("POST", path, data={"model": "inkling", "id_slot": slot, **fields})
        assert res.status_code == 200, res.body
        return next(row["prompt"] for row in s.make_request("GET", "/slots").body if row["id"] == slot)

    prompts = {}
    for level, value in EFFORT.items():
        canonical = render(s, {"messages": messages, "reasoning_effort": level})
        assert f"Thinking effort level: {value}<|end_message|>" in canonical
        ids = s.make_request("POST", "/tokenize", data={"content": canonical, "parse_special": True, "add_special": True}).body["tokens"]
        prompts[level] = s.make_request("POST", "/detokenize", data={"tokens": ids}).body["content"]
        assert processed_prompt("/v1/chat/completions", {"messages": messages, "reasoning_effort": level, "max_tokens": 1}) == prompts[level]
        assert processed_prompt("/v1/responses", {"input": messages, "reasoning": {"effort": level}, "max_output_tokens": 1}) == prompts[level]
        assert processed_prompt("/v1/messages", {"messages": messages, "output_config": {"effort": level}, "max_tokens": 1}) == prompts[level]
    assert len(set(prompts.values())) == len(set(EFFORT.values()))
    assert processed_prompt("/v1/messages", {"messages": messages, "thinking": {"type": "disabled"}, "max_tokens": 1}) == prompts["none"]

    for route, body, wrapper in (("/v1/responses", {"input": messages, "max_output_tokens": 1}, "reasoning"),
                                 ("/v1/messages", {"messages": messages, "max_tokens": 1}, "output_config")):
        for malformed in ("low", [], {"effort": "turbo"}, {"effort": 0.5}, {"effort": ""}):
            res = s.make_request("POST", route, data={"model": "inkling", **body, wrapper: malformed})
            assert res.status_code == 400 and res.body["error"]["type"] == "invalid_request_error", res.body
        for omitted in ({}, {wrapper: None}, {wrapper: {"effort": None}}):
            assert processed_prompt(route, {**body, **omitted}) == prompts["high"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        for left, right in (("none", "max"), ("high", "low")):
            oai = pool.submit(processed_prompt, "/v1/chat/completions", {"messages": messages, "reasoning_effort": left, "max_tokens": 1}, 0)
            anthropic = pool.submit(processed_prompt, "/v1/messages", {"messages": messages, "thinking": {"type": "adaptive"},
                                                                      "output_config": {"effort": right}, "max_tokens": 1}, 1)
            assert oai.result() == prompts[left]
            assert anthropic.result() == prompts[right]
    assert processed_prompt("/v1/chat/completions", {"messages": messages, "max_tokens": 1}) == prompts["high"]
    s.stop()
    s.reasoning = "off"
    s.start()
    assert "Thinking effort level: 0<|end_message|>" in render(s, {"messages": messages})
    assert "Thinking effort level: 0.9<|end_message|>" in render(s, {"messages": messages, "reasoning_effort": "high"})


# an invalid effort however the request carries it: the OAI field, or the template's own input in chat_template_kwargs
INVALID_EFFORT = {"field-unknown": {"reasoning_effort": "turbo"}, "field-empty": {"reasoning_effort": ""},
                  # the field names a level; a number there is refused, in range or not, rather than dropped for the default
                  "field-number": {"reasoning_effort": 2}, "field-number-in-range": {"reasoning_effort": 0.5},
                  "kwargs-unknown": {"chat_template_kwargs": {"reasoning_effort": "extreme"}},
                  "kwargs-out-of-range": {"chat_template_kwargs": {"reasoning_effort": 1.5}}}
ENDPOINTS = {"chat": ("/chat/completions", {"max_tokens": 4}), "streaming-chat": ("/chat/completions", {"max_tokens": 4, "stream": True}),
             "apply-template": ("/apply-template", {})}


@pytest.mark.parametrize("endpoint", list(ENDPOINTS))
@pytest.mark.parametrize("effort", list(INVALID_EFFORT))
def test_inkling_invalid_reasoning_effort_is_a_client_error(server, effort, endpoint):
    path, extra = ENDPOINTS[endpoint]
    res = server.make_request("POST", path, data={"messages": TURNS, **INVALID_EFFORT[effort], **extra})
    assert res.status_code == 400, res.body
    assert res.body["error"]["type"] == "invalid_request_error"
    if effort in ("field-unknown", "field-empty"):
        levels = res.body["error"]["message"].split("supported levels: ")[1]
        assert set(levels.split(", ")) == set(EFFORT)
    # the server is unharmed and still answers
    reply(server, {"messages": TURNS})


def test_inkling_reasoning_effort_in_template_kwargs(server):
    # the template's own input takes a level or a number in its range
    for effort, level in [("low", "0.2"), (0.5, "0.5"), (0, "0")]:
        prompt = render(server, {"messages": TURNS, "chat_template_kwargs": {"reasoning_effort": effort}})
        assert "<|message_system|><|content_text|>Thinking effort level: " + level + "<|end_message|>" in prompt


TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Weather for a place",
    "parameters": {"type": "object", "properties": {
        "location": {"type": "object", "properties": {"city": {"type": "string"}, "country": {"type": "string"}}},
        "units": {"type": "array", "items": {"type": "string"}}},
        "required": ["location"]}}}]
NESTED = {"units": ["celsius"], "location": {"country": "FR", "city": "Paris"}}
TOOL_HISTORY = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "Weather in Paris?"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
        "function": {"name": "get_weather", "arguments": json.dumps(NESTED)}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "21 C and sunny"},
    {"role": "user", "content": "Thanks. Tomorrow?"},
]


def test_inkling_tool_declarations(server):
    prompt = render(server, {"messages": TURNS, "tools": TOOLS})
    start = prompt.index("<|message_system|>tool_declare<|content_xml|>")
    block = prompt[start:prompt.index("<|end_message|>", start)]
    assert '"name":"get_weather"' in block and '"city"' in block and '"units"' in block
    # the synthetic model has no way to finish a tool call, so the reply is held to text
    reply(server, {"messages": TURNS, "tools": TOOLS, "tool_choice": "none"})


def test_inkling_tool_call_with_nested_arguments_and_its_result(server):
    prompt = render(server, {"messages": TOOL_HISTORY, "tools": TOOLS})
    # the arguments arrive as a JSON string and are rendered as one object: its keys sorted, nested objects as sent
    assert_in_order(prompt, "<|message_user|><|content_text|>Weather in Paris?<|end_message|>",
                    '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":'
                    '{"location":{"country":"FR","city":"Paris"},"units":["celsius"]}}<|end_message|>',
                    "<|content_model_end_sampling|>",
                    "<|message_tool|>get_weather<|content_text|>21 C and sunny<|end_message|>",
                    "<|message_user|><|content_text|>Thanks. Tomorrow?<|end_message|>")
    reply(server, {"messages": TOOL_HISTORY, "tools": TOOLS, "tool_choice": "none"})


def test_inkling_thinking_kept_across_turns(server):
    history = [
        {"role": "user", "content": "First question."},
        {"role": "assistant", "reasoning_content": "PLAN-ALPHA think first", "content": "First answer."},
        {"role": "user", "content": "Second question."},
        {"role": "assistant", "reasoning_content": "PLAN-BETA think again", "content": "Second answer."},
        {"role": "user", "content": "Third question."},
    ]
    prompt = render(server, {"messages": history})
    assert_in_order(prompt, "<|message_model|><|content_thinking|>PLAN-ALPHA think first<|end_message|>",
                    "<|message_model|><|content_text|>First answer.<|end_message|>",
                    "<|message_model|><|content_thinking|>PLAN-BETA think again<|end_message|>",
                    "<|message_model|><|content_text|>Second answer.<|end_message|>",
                    "<|message_user|><|content_text|>Third question.<|end_message|>")
    reply(server, {"messages": history})


def image_part(seed):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(fx.png(seed=seed)).decode()}}


@pytest.mark.parametrize("where", ["before", "between", "after"])
def test_inkling_image_parts_around_text(server, files, where):
    look, what = {"type": "text", "text": "Look at this."}, {"type": "text", "text": "What changed?"}
    content = {"before": [image_part(1), look], "between": [look, image_part(1), what], "after": [look, image_part(1)]}[where]
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
    prompt = render(server, {"messages": messages})
    # the image is a user block of its own, where mtmd puts <|content_image|> and the image's embeddings
    blocks = {"image": "<|message_user|>" + MARKER + "<|end_message|>",
              "look": "<|message_user|><|content_text|>Look at this.<|end_message|>",
              "what": "<|message_user|><|content_text|>What changed?<|end_message|>"}
    order = {"before": ["image", "look"], "between": ["look", "image", "what"], "after": ["look", "image"]}[where]
    assert_in_order(prompt, *[blocks[b] for b in order])
    assert prompt.count(MARKER) == 1
    with_image = reply(server, {"messages": messages})["usage"]["prompt_tokens"]
    text_only = [part for part in content if part["type"] == "text"]
    without = reply(server, {"messages": [messages[0], {"role": "user", "content": text_only}]})["usage"]["prompt_tokens"]
    # the image took its own tokens in the prompt the model read
    assert with_image >= without + files["image_tokens"]


def test_inkling_several_images_keep_their_order(server, files):
    content = [image_part(1), {"type": "text", "text": "A"}, image_part(2), {"type": "text", "text": "B"}, image_part(3)]
    messages = [{"role": "user", "content": content}]
    prompt = render(server, {"messages": messages})
    image = "<|message_user|>" + MARKER + "<|end_message|>"
    assert_in_order(prompt, image, "<|content_text|>A<|end_message|>", image, "<|content_text|>B<|end_message|>", image)
    body = reply(server, {"messages": messages})
    assert body["usage"]["prompt_tokens"] >= 3 * files["image_tokens"]


def test_inkling_sampling_stays_below_the_unpadded_vocabulary(server, files):
    assert files["vocab"] > files["unpadded"], "the fixture's vocabulary is not padded"
    sampled = 0
    for seed in range(1, 9):
        res = server.make_request("POST", "/completion", data={
            "prompt": "Count with me: one two three", "n_predict": 48, "seed": seed, "temperature": 1.0,
            "top_k": 0, "top_p": 1.0, "min_p": 0.0, "ignore_eos": True, "return_tokens": True,
            "n_probs": files["vocab"], "cache_prompt": False})
        assert res.status_code == 200, res.body
        ids = list(res.body["tokens"])
        for step in res.body["completion_probabilities"]:
            alternatives = [alt["id"] for alt in step["top_logprobs"]]
            # the padded rows are not candidates at all
            assert len(alternatives) <= files["unpadded"]
            ids += [step["id"], *alternatives]
        assert all(0 <= i < files["unpadded"] for i in ids), sorted(set(i for i in ids if i >= files["unpadded"]))
        sampled += len(res.body["tokens"])
    assert sampled == 8 * 48


def test_inkling_generation_at_context_limit(files):
    s = ServerProcess()
    s.model_hf_repo = s.model_hf_file = None
    s.model_file = files["model"]
    s.offline = True
    s.n_ctx, s.n_slots = 256, 1
    s.override_kv = f"tokenizer.ggml.eos_token_id=int:{END_ID}"
    s.start()
    for requested in (7, 8, 9):
        res = s.make_request("POST", "/completion", data={
            "prompt": [3] * 248, "n_predict": requested, "ignore_eos": True, "cache_prompt": False})
        assert res.status_code == 200, res.body
        assert res.body["timings"]["predicted_n"] == min(requested, 8)
        assert res.body["truncated"] == (requested > 8)

    eos = s.make_request("POST", "/completion", data={
        "prompt": [3] * 255, "n_predict": 8, "temperature": 0, "logit_bias": [[END_ID, 1000]]})
    assert eos.status_code == 200, eos.body
    assert eos.body["stop_type"] == "eos" and not eos.body["truncated"]


@pytest.mark.parametrize("shared,extend,expected_cache", [(100, True, 100), (80, True, 80), (80, False, 64)])
def test_inkling_appends_after_async_ram_restore(files, tmp_path, monkeypatch, shared, extend, expected_cache):
    monkeypatch.setenv("LLAMA_STATE_SAVE_ASYNC", "1")
    monkeypatch.setenv("LLAMA_STATE_RESTORE_ASYNC", "1")
    if shared < 100:
        monkeypatch.setenv("LLAMA_ARG_CHECKPOINT_EVERY", "16")
    s = ServerProcess()
    s.model_hf_repo = s.model_hf_file = None
    s.model_file = files["model"]
    s.offline = s.kv_unified = s.no_cache_idle_slots = True
    s.n_ctx, s.n_slots, s.cache_ram = 512, 2, 128
    s.log_path = str(tmp_path / "server.log")
    s.start()
    rng = random.Random(17)

    def complete(tokens, slot, cache=True):
        res = s.make_request("POST", "/completion", data={
            "prompt": tokens, "id_slot": slot, "n_predict": 1, "cache_prompt": cache, "temperature": 0,
            "ignore_eos": True, "return_tokens": True, "n_probs": files["unpadded"]})
        assert res.status_code == 200, res.body
        return res.body

    def logprobs(body):
        return {p["id"]: p["logprob"] for p in body["completion_probabilities"][0]["top_logprobs"]}

    saved = rng.choices(range(3, files["unpadded"]), k=100)
    first = complete(saved, 0)
    complete(rng.choices(range(3, files["unpadded"]), k=36), 0)
    prompt = saved[:shared]
    if extend:
        next_token = first["tokens"][0] if shared == len(saved) else (4 if saved[shared] == 3 else 3)
        prompt += [next_token] + rng.choices(range(3, files["unpadded"]), k=47)
    restored = complete(prompt, 1)
    assert restored["timings"]["cache_n"] == expected_cache
    cold = complete(prompt, 0, cache=False)
    assert cold["timings"]["cache_n"] == 0
    s.stop()
    log = (tmp_path / "server.log").read_text()
    assert "cache_save reason=saved slot=0" in log
    assert any("cache_restore reason=restored slot=1" in line and "async=1" in line for line in log.splitlines())
    assert logprobs(restored) == pytest.approx(logprobs(cold), abs=2e-3)


def test_inkling_ram_restore_matches_file_from_same_prefix(files, tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "scripts/inkling2"))
    import qualify

    monkeypatch.setenv("LLAMA_STATE_SAVE_ASYNC", "1")
    monkeypatch.setenv("LLAMA_STATE_RESTORE_ASYNC", "1")
    server = ServerProcess()
    server.model_hf_repo = server.model_hf_file = None
    server.model_file = files["model"]
    server.offline = server.kv_unified = server.no_cache_idle_slots = True
    server.n_ctx, server.n_slots, server.cache_ram = 2048, 4, 128
    server.slot_save_path = str(tmp_path)
    server.override_kv = "inkling2.context_length=int:1024"
    server.start()
    options = SimpleNamespace(port=server.server_port, reader_tokens=352, timeout=30,
                              restore_reference_dir=str(tmp_path), logprob_tolerance=1e-4)
    result = qualify.workload(options, tmp_path, "restore")
    assert result["cached_tokens"] == {"cold": 0, "resident": 352, "restored": 352}
    reference = result["same_slot_file_control"]
    assert reference["passed"], reference
    assert reference["restored_vs_file"]["common_top_tokens"] == 32
    assert result["independent_prefixes"]["comparison"]["common_top_tokens"] > 0
    assert not list(tmp_path.glob("inkling-reference-*.bin"))


def test_inkling_active_prefix_with_projector(files, monkeypatch):
    monkeypatch.setenv("LLAMA_MEDIA_MARKER", MARKER)
    pictures = [base64.b64encode(fx.png(seed=29)).decode()]
    replies = []
    for enabled in (False, True):
        monkeypatch.setenv("LLAMA_SLOT_PREFIX_COPY", str(int(enabled)))
        s = ServerProcess()
        s.model_hf_repo = s.model_hf_file = None
        s.model_file, s.mmproj_file = files["model"], files["mmproj"]
        s.offline = s.kv_unified = s.no_cache_idle_slots = True
        s.n_ctx, s.n_slots, s.n_batch, s.n_ubatch = 3072, 2, 32, 32
        s.override_kv = "inkling2.context_length=int:1024"
        s.start()
        for slot in range(s.n_slots):
            warm = s.make_request("POST", "/completion", data={"id_slot": slot, "prompt": f"Previous task {slot}.", "n_predict": 1})
            assert warm.status_code == 200, warm.body
        answers = []
        for branch, media in (("same", False), ("split", False), ("image", True)):
            prefix = branch + (MARKER if media else "") + "Public inventory record of this shipment. " * 44
            tails = [" The red truck.", " The blue truck." if branch != "same" else " The red truck."]
            prompts = [{"prompt_string": prefix + t, "multimodal_data": pictures if media else []} for t in tails]
            res = s.make_request("POST", "/completion", data={"prompt": prompts, "temperature": 0,
                                 "cache_prompt": True, "n_predict": 4, "return_tokens": True})
            assert res.status_code == 200, res.body
            first, second = res.body
            assert first["id_slot"] != second["id_slot"]
            cached = sorted(item["timings"]["cache_n"] for item in res.body)
            if enabled and not media:
                assert cached[0] == 0 and cached[1] >= 256
            else:
                assert cached == [0, 0]
            answers.append([item["tokens"] for item in res.body])
        s.stop()
        replies.append(answers)
    assert replies[0] == replies[1]


def test_inkling_generation_ends_at_the_end_token(server):
    forced = {"prompt": "Say something", "n_predict": 16, "logit_bias": [[END_ID, 100.0]], "return_tokens": True}
    res = server.make_request("POST", "/completion", data=forced)
    assert res.status_code == 200, res.body
    assert res.body["stop_type"] == "eos"
    assert res.body["tokens_predicted"] == 1
    # the same request told to go past the end token runs to its length: the stop came from the end token
    res = server.make_request("POST", "/completion", data={**forced, "ignore_eos": True})
    assert res.body["stop_type"] == "limit" and res.body["tokens_predicted"] == 16
    res = server.make_request("POST", "/chat/completions", data={"messages": TURNS, "max_tokens": 16, "logit_bias": {str(END_ID): 100.0}})
    assert res.status_code == 200, res.body
    assert res.body["choices"][0]["finish_reason"] == "stop"
    assert res.body["usage"]["completion_tokens"] == 1
