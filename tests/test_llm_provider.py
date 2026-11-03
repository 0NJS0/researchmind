"""OpenAI-compatible provider behaviour under OpenRouter free-tier conditions (HTTP mocked)."""
import json

import httpx
import pytest

from src.llm import llm
from src.llm.llm import LLMError, OpenAICompatProvider, QuotaExhausted, RateLimited, strip_reasoning
from src.llm.structured import extract_json


class FakeHTTP:
    """Replaces httpx.post: returns scripted responses and records the payloads sent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.payloads.append(json)
        status, body, resp_headers = self.responses.pop(0)
        return httpx.Response(status, json=body, headers=resp_headers or {},
                              request=httpx.Request("POST", url))


def ok(text, finish="stop"):
    return 200, {"choices": [{"message": {"content": text}, "finish_reason": finish}]}, None


def err(status, message, headers=None):
    return status, {"error": {"code": status, "message": message}}, headers


@pytest.fixture
def http(monkeypatch):
    def install(*responses):
        fake = FakeHTTP(responses)
        monkeypatch.setattr(llm.httpx, "post", fake)
        return fake
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setattr(llm, "caps", llm._ModelCaps())
    return install


def provider(fallbacks=None):
    return OpenAICompatProvider("model-a", fallbacks or [], base_url="http://llm.test/v1", api_key="k",
                                name="openrouter")


def test_strip_reasoning_blocks():
    assert strip_reasoning("<think>cite [P1] maybe</think>\n{\"a\": 1}") == '{"a": 1}'
    assert strip_reasoning("reasoning without opening tag</think>Answer") == "Answer"
    assert strip_reasoning("<think>never closed") == ""


def test_extract_json_skips_brackets_in_prose_and_think_blocks():
    assert extract_json('<think>see [P1] and {x}</think>Here: {"queries": ["a"]}') == {"queries": ["a"]}
    assert extract_json('Evidence from [P2] shows: {"ok": true}') == {"ok": True}


def test_rate_limit_is_retried_after_retry_after(http):
    fake = http(err(429, "Rate limit exceeded: free-models-per-min", {"Retry-After": "2"}), ok("hello"))
    assert provider().complete("hi") == "hello"
    assert len(fake.payloads) == 2


def test_daily_cap_raises_quota_exhausted_without_retry(http):
    fake = http(err(429, "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000"))
    with pytest.raises(QuotaExhausted):
        provider().complete("hi")
    assert len(fake.payloads) == 1


def test_json_mode_disabled_when_openrouter_rejects_parameters(http):
    fake = http(err(404, "No endpoints found that can handle the requested parameters."), ok('{"a": 1}'))
    assert provider().complete("hi", json_mode=True) == '{"a": 1}'
    assert "response_format" in fake.payloads[0] and "response_format" not in fake.payloads[1]


def test_system_role_folded_into_user_message_when_rejected(http):
    fake = http(err(400, "Developer instruction is not enabled for this model"), ok("done"))
    assert provider().complete("question", system="be precise") == "done"
    assert fake.payloads[0]["messages"][0]["role"] == "system"
    assert fake.payloads[1]["messages"] == [{"role": "user", "content": "be precise\n\nquestion"}]


def test_fallback_model_used_when_primary_is_rate_limited(http):
    limited = err(429, "model-a is temporarily rate-limited upstream")
    fake = http(limited, ok("from b"))
    p = provider(fallbacks=["model-b"])
    assert p.complete("hi") == "from b"
    # A provider-side limit goes straight to the next model instead of burning retries on the busy one.
    assert [x["model"] for x in fake.payloads] == ["model-a", "model-b"] and p.last_model == "model-b"


def test_account_wide_rate_limit_waits_instead_of_switching_models(http, monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    limited = err(429, "Rate limit exceeded: free-models-per-min")
    fake = http(limited, ok("after waiting"))
    assert provider(fallbacks=["model-b"]).complete("hi") == "after waiting"
    assert [x["model"] for x in fake.payloads] == ["model-a", "model-a"]


def test_truncation_retries_are_bounded_and_keep_the_last_reply(http):
    cut = ok('{"a": 1, "b": [1, 2', finish="length")
    fake = http(cut, cut, cut, ok("never reached"))
    assert provider(fallbacks=["model-b"]).complete("hi", max_tokens=400, json_mode=True) == '{"a": 1, "b": [1, 2'
    assert len(fake.payloads) == 3 and all(x["model"] == "model-a" for x in fake.payloads)


def test_malformed_replies_raise_llm_errors(http):
    for body in ({"choices": [None]}, {"choices": "x"}, {"choices": [{"message": "x"}]}):
        http((200, body, None))
        with pytest.raises(LLMError):
            provider().complete("hi")


def test_outage_classification():
    from src.llm.llm import LLMHTTPError, ServiceUnavailable, is_outage
    assert is_outage(RateLimited("x")) and is_outage(ServiceUnavailable("x"))
    assert is_outage(LLMHTTPError(401, "User not found"))              # expired / revoked key
    assert not is_outage(LLMHTTPError(403, "model is only available on agentic harnesses"))
    assert not is_outage(LLMHTTPError(400, "bad request")) and not is_outage(LLMError("empty reply"))


def test_all_models_down_raises_service_unavailable(http, monkeypatch):
    from src.llm.llm import ServiceUnavailable
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    http(*([err(503, "down")] * 20))
    with pytest.raises(ServiceUnavailable):
        provider(fallbacks=["model-b"]).complete("hi")


def test_all_models_rate_limited_raises_rate_limited(http):
    http(*([err(429, "slow down")] * (llm.LLM_MAX_RETRIES + 1)))
    with pytest.raises(RateLimited):
        provider().complete("hi")


def test_empty_reply_after_reasoning_gets_bigger_budget(http):
    fake = http(ok("", finish="length"), ok("<think>long</think>final"))
    assert provider().complete("hi", max_tokens=500) == "final"
    assert fake.payloads[1]["max_tokens"] == 2048    # straight to a budget a reasoning model can work with


def test_client_errors_are_not_retried(http):
    fake = http(err(401, "No auth credentials found"))
    with pytest.raises(LLMError):
        provider().complete("hi")
    assert len(fake.payloads) == 1


def test_server_errors_are_retried(http):
    fake = http(err(502, "bad gateway"), ok("recovered"))
    assert provider().complete("hi") == "recovered"
    assert len(fake.payloads) == 2


def test_restricted_model_falls_through_to_next(http):
    fake = http(err(403, "thinkingmachines/inkling:free is only available on agentic harnesses."), ok("from b"))
    assert provider(fallbacks=["model-b"]).complete("hi") == "from b"
    assert [p["model"] for p in fake.payloads] == ["model-a", "model-b"]


def test_free_only_guard(monkeypatch):
    import src.config as config
    monkeypatch.setattr(config, "LLM_FREE_ONLY", True)
    assert config.is_allowed_model("nvidia/nemotron-3-super-120b-a12b:free")
    assert not config.is_allowed_model("openai/gpt-4o")
    monkeypatch.setattr(llm, "LLM_PROVIDER", "openrouter")
    with pytest.raises(LLMError, match="not a free model"):
        llm._build_provider("openai/gpt-4o", with_fallbacks=False)


def test_truncated_json_reply_is_retried_with_bigger_budget(http):
    fake = http(ok('{"verdict": "crowded", "neighbours": [{"index": 1', finish="length"), ok('{"verdict": "crowded", "reason": "x"}'))
    assert provider().complete("hi", max_tokens=800, json_mode=True) == '{"verdict": "crowded", "reason": "x"}'
    assert fake.payloads[1]["max_tokens"] == 2048
