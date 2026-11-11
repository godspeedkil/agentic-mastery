from datetime import UTC, datetime
from itertools import count

import pytest

from core.llm import (
    Ledger,
    LLMClient,
    Message,
    ModelPrice,
    PriceTable,
    Request,
    Response,
    TextBlock,
    Usage,
    read_records,
)

REQUEST = Request(model="model-a", messages=[Message.user("hi")])
PRICES = PriceTable(prices={"model-a": ModelPrice(input=3.0, output=15.0)})


class FakeProvider:
    name = "anthropic"

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def complete(self, request: Request) -> Response:
        if self.error:
            raise self.error
        return Response(
            model="model-a-2026-10-01",
            content=[TextBlock(text="hello")],
            stop_reason="end_turn",
            usage=Usage(input_tokens=1_000_000, output_tokens=100_000),
        )


def ticking_clock(step_seconds: float = 0.25):
    ticks = count()
    return lambda: next(ticks) * step_seconds


def test_successful_call_is_priced_timed_and_recorded(tmp_path):
    ledger = Ledger(tmp_path / "run.jsonl")
    client = LLMClient(
        {"anthropic": FakeProvider()},
        prices=PRICES,
        ledger=ledger,
        run_id="run-1",
        clock=ticking_clock(),
    )
    completion = client.complete("anthropic", REQUEST)

    assert completion.response.text == "hello"
    record = completion.record
    assert record.model == "model-a"
    assert record.run_id == "run-1"
    assert record.cost_usd == pytest.approx(3.0 + 1.5)
    assert record.latency_ms == pytest.approx(250.0)
    assert record.stop_reason == "end_turn"
    assert read_records(ledger.path) == [record]


def test_unpriced_model_records_unknown_cost():
    completion = LLMClient({"anthropic": FakeProvider()}).complete("anthropic", REQUEST)
    assert completion.record.cost_usd is None


def test_failed_call_is_recorded_then_reraised(tmp_path):
    ledger = Ledger(tmp_path / "run.jsonl")
    client = LLMClient({"anthropic": FakeProvider(error=TimeoutError("slow"))}, ledger=ledger)
    with pytest.raises(TimeoutError):
        client.complete("anthropic", REQUEST)

    [record] = read_records(ledger.path)
    assert record.error == "TimeoutError: slow"
    assert record.cost_usd is None
    assert record.latency_ms is not None


def test_started_at_is_taken_before_the_call():
    called_at = []

    class RecordingProvider(FakeProvider):
        def complete(self, request: Request) -> Response:
            called_at.append(datetime.now(UTC))
            return super().complete(request)

    completion = LLMClient({"anthropic": RecordingProvider()}).complete("anthropic", REQUEST)
    assert completion.record.started_at <= called_at[0]


def test_unknown_provider_is_rejected_before_any_call():
    with pytest.raises(ValueError, match="have \\['anthropic'\\]"):
        LLMClient({"anthropic": FakeProvider()}).complete("openai", REQUEST)
