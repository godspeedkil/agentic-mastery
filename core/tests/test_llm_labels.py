import pytest
from pydantic import BaseModel

from core.llm import (
    CallRecord,
    Ledger,
    LLMClient,
    Message,
    Request,
    Response,
    RetryPolicy,
    TextBlock,
    Usage,
    read_records,
)
from core.llm.errors import RequestRejectedError

REQUEST = Request(model="m", messages=[Message.user("hi")])
LABELS = {"case": "inv-001", "repeat": "2"}


class Answer(BaseModel):
    value: int


class ScriptedProvider:
    name = "anthropic"

    def __init__(self, *outcomes: str | Exception) -> None:
        self.outcomes = list(outcomes)

    def complete(self, request: Request) -> Response:
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return Response(
            model="m", content=[TextBlock(text=outcome)], stop_reason="end_turn", usage=Usage()
        )


def client_for(provider, tmp_path) -> LLMClient:
    return LLMClient(
        {"anthropic": provider},
        ledger=Ledger(tmp_path / "run.jsonl"),
        retry=RetryPolicy(max_attempts=1),
    )


def test_records_have_no_labels_by_default():
    assert CallRecord(provider="openai", model="m").labels == {}


def test_labels_are_recorded_on_successful_calls(tmp_path):
    completion = client_for(ScriptedProvider("ok"), tmp_path).complete(
        "anthropic", REQUEST, labels=LABELS
    )
    assert completion.record.labels == LABELS
    assert read_records(tmp_path / "run.jsonl")[0].labels == LABELS


def test_labels_are_recorded_on_failed_calls(tmp_path):
    provider = ScriptedProvider(RequestRejectedError("bad", provider="anthropic"))
    with pytest.raises(RequestRejectedError):
        client_for(provider, tmp_path).complete("anthropic", REQUEST, labels=LABELS)
    [record] = read_records(tmp_path / "run.jsonl")
    assert record.labels == LABELS
    assert record.error is not None


def test_parse_labels_every_call_including_repairs(tmp_path):
    provider = ScriptedProvider('{"value": "not a number"}', '{"value": 3}')
    parsed = client_for(provider, tmp_path).parse("anthropic", REQUEST, Answer, labels=LABELS)
    assert parsed.value.value == 3
    records = read_records(tmp_path / "run.jsonl")
    assert len(records) == 2
    assert all(record.labels == LABELS for record in records)
