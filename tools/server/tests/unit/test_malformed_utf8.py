import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


# stories260K's byte tokens: <0xE4>, the lead byte of a 3-byte character, and <0xBD>, a continuation byte
LEAD = 231
CONT = 192


# A model can sample a byte that cannot continue the character before it, or a continuation byte with no lead. The
# final parse of the generated text (common_chat_peg_parse) refused malformed UTF-8, so the request answered 500 after
# the whole generation, or a stream ended on an error event. A malformed byte now reads U+FFFD, as the JSON writer
# shows it. An incomplete tail at the limit is as it was: a stream holds it back, /completion shows it as U+FFFD and
# the chat message may drop it, so it is allowed as one more U+FFFD or none.
#   stray: one continuation byte;
#   lead-lead: E4 E4 E4, two leads each followed by a byte that cannot continue it, then an incomplete tail;
#   cut: the limit stops inside a character, an incomplete tail alone (it answered 200 before too).
@pytest.mark.parametrize("token,n_predict,n_malformed,tail", [
    (CONT, 1, 1, False),
    (LEAD, 3, 2, True),
    (LEAD, 1, 0, True),
], ids=["stray", "lead-lead", "cut"])
def test_malformed_utf8_output(token: int, n_predict: int, n_malformed: int, tail: bool):
    global server
    server.start()
    head = "\ufffd" * n_malformed
    allowed = {head, head + "\ufffd"} if tail else {head}
    bias = {"logit_bias": [[token, 100]], "temperature": 0}
    prompt = {"prompt": "Once upon a time", "n_predict": n_predict}
    chat = {"messages": [{"role": "user", "content": "Once upon a time"}], "max_tokens": n_predict}
    res = server.make_request("POST", "/completion", data={**prompt, **bias})
    assert res.status_code == 200, res.body
    assert res.body["content"] in allowed
    res = server.make_request("POST", "/chat/completions", data={**chat, **bias})
    assert res.status_code == 200, res.body
    assert res.body["choices"][0]["message"]["content"] in allowed
    # a stream sends nothing after its last send while the text ends in an incomplete tail, and the limit ends it there:
    # with a tail, any run of the U+FFFD up to the content's may have been sent
    if tail:
        allowed = {"�" * j for j in range(n_malformed + 2)}
    for path, data in (("/completion", prompt), ("/chat/completions", chat)):
        text = ""
        for chunk in server.make_stream_request("POST", path, data={**data, **bias, "stream": True}):
            assert "error" not in chunk, chunk
            if "content" in chunk:
                text += chunk["content"]
            elif chunk.get("choices"):
                text += chunk["choices"][0]["delta"].get("content") or ""
        assert text in allowed, path
