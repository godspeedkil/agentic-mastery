from typing import Literal, Self

import pytest
from pydantic import BaseModel, model_validator

from core.llm import LLMClient, Message, Request, Response, StopReason, TextBlock, Usage
from core.llm.errors import StructuredOutputError

REQUEST = Request(model="m", messages=[Message.user("Review: great product, late delivery.")])


class Verdict(BaseModel):
    sentiment: Literal["positive", "neutral", "negative", "mixed"]
    score: int

    @model_validator(mode="after")
    def _score_matches_sentiment(self) -> Self:
        if self.sentiment == "mixed" and self.score in (1, 5):
            raise ValueError("a mixed review cannot have an extreme score")
        return self


class ScriptedProvider:
    """Answers with the given texts in order and keeps every request it received."""

    name = "anthropic"

    def __init__(self, *answers: str, stop_reason: StopReason = "end_turn") -> None:
        self.answers = list(answers)
        self.stop_reason = stop_reason
        self.requests: list[Request] = []

    def complete(self, request: Request) -> Response:
        self.requests.append(request)
        text = self.answers.pop(0)
        content = [TextBlock(text=text)] if text else []
        return Response(model="m", content=content, stop_reason=self.stop_reason, usage=Usage())


def parse(provider: ScriptedProvider, **kwargs):
    return LLMClient({"anthropic": provider}).parse("anthropic", REQUEST, Verdict, **kwargs)


def test_valid_answer_is_parsed_and_schema_is_requested():
    provider = ScriptedProvider('{"sentiment": "mixed", "score": 3}')
    parsed = parse(provider)

    assert parsed.value == Verdict(sentiment="mixed", score=3)
    assert len(parsed.completions) == 1
    assert provider.requests[0].output_type is Verdict


def test_invalid_answer_is_repaired_with_the_error_as_feedback():
    provider = ScriptedProvider(
        '{"sentiment": "mixed", "score": 5}',
        '{"sentiment": "mixed", "score": 3}',
    )
    parsed = parse(provider)

    assert parsed.value.score == 3
    assert len(parsed.completions) == 2
    repair_request = provider.requests[1]
    assert [m.role for m in repair_request.messages] == ["user", "assistant", "user"]
    assert "extreme score" in repair_request.messages[-1].content[0].text


def test_gives_up_after_max_repairs():
    provider = ScriptedProvider(*['{"sentiment": "mixed", "score": 5}'] * 3)
    with pytest.raises(StructuredOutputError, match="after 2 repair") as caught:
        parse(provider, max_repairs=2)
    assert len(provider.requests) == 3
    assert "extreme score" in str(caught.value.__cause__)


def test_zero_repairs_means_a_single_call():
    provider = ScriptedProvider('{"sentiment": "mixed", "score": 5}')
    with pytest.raises(StructuredOutputError):
        parse(provider, max_repairs=0)
    assert len(provider.requests) == 1


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal"])
def test_cut_off_or_refused_answers_are_not_parsed(stop_reason):
    provider = ScriptedProvider('{"sentiment": "mi', stop_reason=stop_reason)
    with pytest.raises(StructuredOutputError, match=stop_reason):
        parse(provider)
    assert len(provider.requests) == 1


def test_empty_answer_is_an_error_not_a_repair():
    provider = ScriptedProvider("")
    with pytest.raises(StructuredOutputError, match="no text"):
        parse(provider)
    assert len(provider.requests) == 1


def test_negative_max_repairs_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        parse(ScriptedProvider(), max_repairs=-1)
