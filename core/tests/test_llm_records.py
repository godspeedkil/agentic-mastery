import pytest
from pydantic import ValidationError

from core.llm import CallRecord, Usage


def test_total_input_sums_disjoint_fields():
    usage = Usage(input_tokens=100, cache_read_tokens=900, cache_write_tokens=50)
    assert usage.total_input_tokens == 1050


def test_reasoning_cannot_exceed_output():
    with pytest.raises(ValidationError, match="reasoning_tokens"):
        Usage(output_tokens=10, reasoning_tokens=11)


def test_negative_counts_rejected():
    with pytest.raises(ValidationError):
        Usage(input_tokens=-1)


def test_record_round_trips_through_json():
    record = CallRecord(
        provider="anthropic",
        model="claude-haiku-5-5",
        usage=Usage(input_tokens=12, output_tokens=34),
        cost_usd=0.0001,
        latency_ms=850.0,
        stop_reason="end_turn",
    )
    line = record.model_dump_json()
    assert "\n" not in line
    assert CallRecord.model_validate_json(line) == record


def test_records_are_immutable():
    record = CallRecord(provider="openai", model="gpt-x")
    with pytest.raises(ValidationError):
        record.model = "other"
