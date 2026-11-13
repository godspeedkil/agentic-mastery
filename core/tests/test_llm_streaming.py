import random
from collections.abc import Generator
from itertools import count

import pytest

from core.llm import (
    Ledger,
    LLMClient,
    Message,
    Request,
    Response,
    StreamEvent,
    TextBlock,
    TextDelta,
    Usage,
    read_records,
)
from core.llm.errors import ProviderUnavailableError

REQUEST = Request(model="m", messages=[Message.user("hi")])


def final_response(text: str) -> Response:
    return Response(
        model="m", content=[TextBlock(text=text)], stop_reason="end_turn", usage=Usage()
    )


class StreamingProvider:
    """Streams scripted chunks; each entry in `failures` makes one attempt fail.

    A failure is (number of chunks to stream first, error to raise).
    """

    name = "anthropic"

    def __init__(self, chunks: list[str], *failures: tuple[int, Exception]) -> None:
        self.chunks = chunks
        self.failures = list(failures)
        self.attempts = 0

    def complete(self, request: Request) -> Response:
        raise AssertionError("a streaming call must use stream()")

    def stream(self, request: Request) -> Generator[StreamEvent, None, None]:
        self.attempts += 1
        if self.failures:
            chunks_before_error, error = self.failures.pop(0)
            for chunk in self.chunks[:chunks_before_error]:
                yield TextDelta(text=chunk)
            raise error
        for chunk in self.chunks:
            yield TextDelta(text=chunk)
        yield final_response("".join(self.chunks))


def ticking_clock():
    ticks = count()
    return lambda: next(ticks) * 0.1


def make_client(provider, tmp_path) -> LLMClient:
    return LLMClient(
        {"anthropic": provider},
        ledger=Ledger(tmp_path / "run.jsonl"),
        clock=ticking_clock(),
        sleep=lambda seconds: None,
        rng=random.Random(0),
    )


def unavailable() -> ProviderUnavailableError:
    return ProviderUnavailableError("overloaded", provider="anthropic")


def test_on_text_receives_chunks_in_order(tmp_path):
    received: list[str] = []
    completion = make_client(StreamingProvider(["Hel", "lo"]), tmp_path).complete(
        "anthropic", REQUEST, on_text=received.append
    )
    assert received == ["Hel", "lo"]
    assert completion.response.text == "Hello"


def test_streaming_records_time_to_first_token(tmp_path):
    # Clock reads: start 0.0, first chunk 0.1, end 0.2.
    completion = make_client(StreamingProvider(["Hel", "lo"]), tmp_path).complete(
        "anthropic", REQUEST, stream=True
    )
    assert completion.record.ttft_ms == pytest.approx(100.0)
    assert completion.record.latency_ms == pytest.approx(200.0)


def test_plain_calls_have_no_time_to_first_token(tmp_path):
    class PlainProvider(StreamingProvider):
        def complete(self, request: Request) -> Response:
            return final_response("hi")

    completion = make_client(PlainProvider([]), tmp_path).complete("anthropic", REQUEST)
    assert completion.record.ttft_ms is None


def test_failure_before_any_text_is_retried(tmp_path):
    provider = StreamingProvider(["Hel", "lo"], (0, unavailable()))
    received: list[str] = []
    make_client(provider, tmp_path).complete("anthropic", REQUEST, on_text=received.append)

    assert provider.attempts == 2
    assert received == ["Hel", "lo"]
    assert [r.attempt for r in read_records(tmp_path / "run.jsonl")] == [1, 2]


def test_failure_after_text_was_delivered_is_not_retried(tmp_path):
    provider = StreamingProvider(["Hel", "lo"], (1, unavailable()))
    received: list[str] = []
    with pytest.raises(ProviderUnavailableError):
        make_client(provider, tmp_path).complete("anthropic", REQUEST, on_text=received.append)

    assert provider.attempts == 1
    assert received == ["Hel"]
    [record] = read_records(tmp_path / "run.jsonl")
    assert record.error is not None


def test_stream_without_final_response_is_an_error(tmp_path):
    class TruncatedProvider(StreamingProvider):
        def stream(self, request: Request) -> Generator[StreamEvent, None, None]:
            yield TextDelta(text="Hel")

    with pytest.raises(RuntimeError, match="without a final Response"):
        make_client(TruncatedProvider([]), tmp_path).complete("anthropic", REQUEST, stream=True)
