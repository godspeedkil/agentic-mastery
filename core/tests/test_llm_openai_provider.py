import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import openai
import pytest
from openai.types.responses import Response as OpenAIResponse
from pydantic import BaseModel

from core.llm import Message, Request, Response, TextDelta
from core.llm import openai_provider as provider_module
from core.llm.errors import ProviderUnavailableError, RateLimitedError
from core.llm.openai_provider import OpenAIProvider, from_openai_response, to_openai_params
from core.settings import Settings

FIXTURES = Path(__file__).parent / "fixtures" / "openai"
REQUEST_HI = Request(model="m", messages=[Message.user("hi")])


def load_response(name: str, **overrides) -> OpenAIResponse:
    data = json.loads((FIXTURES / name).read_text())
    return OpenAIResponse.model_validate(data | overrides)


def test_params_map_system_to_instructions_and_disable_storage():
    request = Request(model="m", messages=[Message.user("hi")], system="Be brief.")
    assert to_openai_params(request) == {
        "model": "m",
        "max_output_tokens": 1024,
        "input": [{"role": "user", "content": "hi"}],
        "store": False,
        "instructions": "Be brief.",
    }


def test_stop_sequences_are_rejected():
    request = Request(model="m", messages=[Message.user("hi")], stop_sequences=["END"])
    with pytest.raises(ValueError, match="stop sequences"):
        to_openai_params(request)


def test_response_keeps_text_and_skips_reasoning():
    response = from_openai_response(load_response("response_with_reasoning.json"))
    assert response.text == "The capital of France is Paris."
    assert response.stop_reason == "end_turn"
    assert response.raw_stop_reason == "completed"


def test_usage_splits_cached_tokens_out_of_input():
    usage = from_openai_response(load_response("response_with_reasoning.json")).usage
    assert usage.input_tokens == 18
    assert usage.cache_read_tokens == 1200
    assert usage.cache_write_tokens == 300
    assert usage.total_input_tokens == 1518
    assert usage.reasoning_tokens == 80


def test_inconsistent_cache_counts_never_go_negative():
    data = json.loads((FIXTURES / "response_with_reasoning.json").read_text())
    data["usage"]["input_tokens"] = 1000
    usage = from_openai_response(OpenAIResponse.model_validate(data)).usage
    assert usage.input_tokens == 0


@pytest.mark.parametrize(
    ("reason", "expected"),
    [("max_output_tokens", "max_tokens"), ("content_filter", "refusal"), ("steered", "other")],
)
def test_incomplete_responses_are_normalized(reason, expected):
    response = from_openai_response(
        load_response(
            "response_with_reasoning.json",
            status="incomplete",
            incomplete_details={"reason": reason},
        )
    )
    assert response.stop_reason == expected
    assert response.raw_stop_reason == f"incomplete:{reason}"


def test_refusal_content_sets_stop_reason():
    refusal = {
        "type": "message",
        "id": "msg_02",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "refusal", "refusal": "I can't help with that."}],
    }
    response = from_openai_response(load_response("response_with_reasoning.json", output=[refusal]))
    assert response.stop_reason == "refusal"
    assert response.text == "I can't help with that."


def test_unsupported_output_item_raises():
    call = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "search",
        "arguments": "{}",
    }
    with pytest.raises(ValueError, match="function_call"):
        from_openai_response(load_response("response_with_reasoning.json", output=[call]))


def test_complete_sends_params_and_parses_reply():
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return load_response("response_with_reasoning.json")

    fake_client = SimpleNamespace(responses=SimpleNamespace(create=create))
    response = OpenAIProvider(client=fake_client).complete(
        Request(model="m", messages=[Message.user("Capital?")])
    )
    assert sent["input"] == [{"role": "user", "content": "Capital?"}]
    assert response.text == "The capital of France is Paris."


def test_missing_api_key_fails_loudly(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(provider_module, "get_settings", lambda: Settings(_env_file=None))
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIProvider()


def failing_client(exc: Exception) -> SimpleNamespace:
    def create(**kwargs):
        raise exc

    return SimpleNamespace(responses=SimpleNamespace(create=create))


def test_sdk_status_errors_become_core_errors():
    response = SimpleNamespace(status_code=429, headers={"retry-after": "2"}, request=None)
    sdk_error = openai.RateLimitError("slow down", response=response, body=None)
    provider = OpenAIProvider(client=failing_client(sdk_error))
    with pytest.raises(RateLimitedError) as caught:
        provider.complete(Request(model="m", messages=[Message.user("hi")]))
    assert caught.value.retry_after == 2.0
    assert caught.value.__cause__ is sdk_error


def test_sdk_timeouts_become_provider_unavailable():
    provider = OpenAIProvider(client=failing_client(openai.APITimeoutError(request=None)))
    with pytest.raises(ProviderUnavailableError):
        provider.complete(Request(model="m", messages=[Message.user("hi")]))


def streaming_client(events: list) -> SimpleNamespace:
    def create(**kwargs):
        assert kwargs["stream"] is True
        return nullcontext(events)

    return SimpleNamespace(responses=SimpleNamespace(create=create))


def text_delta(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="response.output_text.delta", delta=text)


def test_stream_yields_text_deltas_then_the_final_response():
    final = SimpleNamespace(
        type="response.completed", response=load_response("response_with_reasoning.json")
    )
    events = [SimpleNamespace(type="response.created"), text_delta("Par"), text_delta("is."), final]
    streamed = list(OpenAIProvider(client=streaming_client(events)).stream(REQUEST_HI))

    assert streamed[:2] == [TextDelta(text="Par"), TextDelta(text="is.")]
    assert isinstance(streamed[-1], Response)
    assert streamed[-1].usage.cache_read_tokens == 1200


def test_stream_error_event_raises_retryable_error():
    events = [text_delta("Par"), SimpleNamespace(type="error", code="server_error", message="boom")]
    with pytest.raises(ProviderUnavailableError, match="server_error"):
        list(OpenAIProvider(client=streaming_client(events)).stream(REQUEST_HI))


def test_stream_without_final_response_raises():
    with pytest.raises(ProviderUnavailableError, match="without a response"):
        list(OpenAIProvider(client=streaming_client([text_delta("Par")])).stream(REQUEST_HI))


class Verdict(BaseModel):
    sentiment: Literal["positive", "neutral", "negative", "mixed"]
    due: str | None = None


def test_output_type_sends_a_strict_json_schema():
    request = Request(model="m", messages=[Message.user("hi")], output_type=Verdict)
    output_format = to_openai_params(request)["text"]["format"]
    assert output_format["name"] == "Verdict"
    assert output_format["strict"] is True
    assert output_format["schema"]["additionalProperties"] is False
    # Strict mode: every field is required; optional ones are nullable instead.
    assert output_format["schema"]["required"] == ["sentiment", "due"]


def test_no_output_type_sends_no_schema():
    assert "text" not in to_openai_params(REQUEST_HI)
