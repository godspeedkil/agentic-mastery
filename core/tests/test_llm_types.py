import pytest
from pydantic import ValidationError

from core.llm import Message, Request, Response, TextBlock, Usage


def test_message_shortcuts_build_single_text_block():
    message = Message.user("hello")
    assert message.role == "user"
    assert message.content == [TextBlock(text="hello")]
    assert Message.assistant("hi").role == "assistant"


def test_request_requires_at_least_one_message():
    with pytest.raises(ValidationError):
        Request(model="m", messages=[])


def test_request_rejects_unknown_fields():
    with pytest.raises(ValidationError, match="temperature"):
        Request(model="m", messages=[Message.user("x")], temperature=0.0)


def test_message_rejects_empty_content():
    with pytest.raises(ValidationError):
        Message(role="user", content=[])


def test_response_text_joins_text_blocks():
    response = Response(
        model="m",
        content=[TextBlock(text="Hello, "), TextBlock(text="world")],
        stop_reason="end_turn",
        usage=Usage(),
    )
    assert response.text == "Hello, world"


def test_response_rejects_unknown_stop_reason():
    with pytest.raises(ValidationError):
        Response(model="m", content=[], stop_reason="stop", usage=Usage())
