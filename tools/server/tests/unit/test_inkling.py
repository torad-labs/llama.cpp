# Inkling 2 through this server: the synthetic inkling2 model test-llama-archs writes, the official template and a
# synthetic inkling2v projector. Its tokens are "tok_<id>" and decode to hex ids, so a reply's text means nothing;
# what the tests read is the prompt the server renders, the status it answers with and the ids it samples.
import base64
import json

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


# an invalid effort however the request carries it: the OAI field, or the template's own input in chat_template_kwargs
INVALID_EFFORT = {"field-unknown": {"reasoning_effort": "turbo"},
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
