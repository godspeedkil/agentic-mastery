import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import anthropic
import pytest
from anthropic.types import Message as AnthropicMessage
from pydantic import BaseModel

from core.llm import Message, Request, Response, TextDelta
from core.llm import anthropic_provider as provider_module
from core.llm.anthropic_provider import (
    AnthropicProvider,
    from_anthropic_message,
    to_anthropic_params,
)
from core.llm.errors import ProviderUnavailableError, RateLimitedError
from core.settings import Settings

FIXTURES = Path(__file__).parent / "fixtures" / "anthropic"
REQUEST_HI = Request(model="m", messages=[Message.user("hi")])


def load_message(name: str, **overrides) -> AnthropicMessage:
    data = json.loads((FIXTURES / name).read_text())
    return AnthropicMessage.model_validate(data | overrides)


def test_params_include_only_what_was_set():
    request = Request(model="m", messages=[Message.user("hi")])
    assert to_anthropic_params(request) == {
        "model": "m",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }


def test_params_carry_system_and_stop_sequences():
    request = Request(
        model="m", messages=[Message.user("hi")], system="Be brief.", stop_sequences=["END"]
    )
    params = to_anthropic_params(request)
    assert params["system"] == "Be brief."
    assert params["stop_sequences"] == ["END"]


def test_response_keeps_text_and_skips_thinking():
    response = from_anthropic_message(load_message("message_with_thinking.json"))
    assert response.text == "The capital of France is Paris."
    assert response.model == "claude-haiku-5-5"
    assert response.stop_reason == "end_turn"


def test_usage_maps_cache_and_thinking_tokens():
    usage = from_anthropic_message(load_message("message_with_thinking.json")).usage
    assert usage.input_tokens == 18
    assert usage.cache_read_tokens == 1200
    assert usage.cache_write_tokens == 300
    assert usage.output_tokens == 95
    assert usage.reasoning_tokens == 80
    assert usage.total_input_tokens == 1518


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("max_tokens", "max_tokens"),
        ("model_context_window_exceeded", "max_tokens"),
        ("refusal", "refusal"),
        ("pause_turn", "other"),
    ],
)
def test_stop_reasons_are_normalized_and_raw_value_kept(raw, expected):
    response = from_anthropic_message(load_message("message_with_thinking.json", stop_reason=raw))
    assert response.stop_reason == expected
    assert response.raw_stop_reason == raw


def test_unsupported_block_type_raises():
    tool_use = {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {}}
    message = load_message("message_with_thinking.json", content=[tool_use])
    with pytest.raises(ValueError, match="tool_use"):
        from_anthropic_message(message)


def test_complete_sends_params_and_parses_reply():
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return load_message("message_with_thinking.json")

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=create))
    provider = AnthropicProvider(client=fake_client)
    response = provider.complete(Request(model="m", messages=[Message.user("Capital?")]))
    assert sent["model"] == "m"
    assert response.text == "The capital of France is Paris."


def test_missing_api_key_fails_loudly(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(provider_module, "get_settings", lambda: Settings(_env_file=None))
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider()


def failing_client(exc: Exception) -> SimpleNamespace:
    def create(**kwargs):
        raise exc

    return SimpleNamespace(messages=SimpleNamespace(create=create))


def test_sdk_status_errors_become_core_errors():
    response = SimpleNamespace(status_code=429, headers={"retry-after": "2"}, request=None)
    sdk_error = anthropic.RateLimitError("slow down", response=response, body=None)
    provider = AnthropicProvider(client=failing_client(sdk_error))
    with pytest.raises(RateLimitedError) as caught:
        provider.complete(Request(model="m", messages=[Message.user("hi")]))
    assert caught.value.retry_after == 2.0
    assert caught.value.__cause__ is sdk_error


def test_sdk_timeouts_become_provider_unavailable():
    provider = AnthropicProvider(client=failing_client(anthropic.APITimeoutError(request=None)))
    with pytest.raises(ProviderUnavailableError):
        provider.complete(Request(model="m", messages=[Message.user("hi")]))


class FakeMessageStream:
    """Stands in for the SDK's stream manager: a context manager with text and a final message."""

    def __init__(self, chunks: list[str], final: AnthropicMessage) -> None:
        self.text_stream = iter(chunks)
        self.final = final
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.closed = True

    def get_final_message(self) -> AnthropicMessage:
        return self.final


def test_stream_yields_text_deltas_then_the_final_response():
    final = load_message("message_with_thinking.json")
    fake = FakeMessageStream(["The capital ", "is Paris."], final)
    client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kwargs: fake))
    events = list(AnthropicProvider(client=client).stream(REQUEST_HI))

    assert events[:2] == [TextDelta(text="The capital "), TextDelta(text="is Paris.")]
    assert isinstance(events[-1], Response)
    assert events[-1].usage.reasoning_tokens == 80
    assert fake.closed


def test_stream_errors_are_translated():
    def stream(**kwargs):
        raise anthropic.APITimeoutError(request=None)

    client = SimpleNamespace(messages=SimpleNamespace(stream=stream))
    with pytest.raises(ProviderUnavailableError):
        list(AnthropicProvider(client=client).stream(REQUEST_HI))


class Verdict(BaseModel):
    sentiment: Literal["positive", "neutral", "negative", "mixed"]
    due: str | None = None


def test_output_type_sends_a_json_schema():
    request = Request(model="m", messages=[Message.user("hi")], output_type=Verdict)
    output_format = to_anthropic_params(request)["output_config"]["format"]
    assert output_format["type"] == "json_schema"
    assert output_format["schema"]["additionalProperties"] is False
    assert output_format["schema"]["required"] == ["sentiment"]


def test_no_output_type_sends_no_schema():
    assert "output_config" not in to_anthropic_params(REQUEST_HI)
