import pytest
from pydantic import BaseModel, ValidationError

from core.llm import (
    Message,
    Reasoning,
    Request,
    Response,
    TextBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)


class Lookup(BaseModel):
    vendor_id: str


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


CALL = ToolCall(id="call_1", name="lookup", arguments='{"vendor_id": "V-1"}')


def test_response_lists_tool_calls_in_order():
    second = ToolCall(id="call_2", name="lookup", arguments="{}")
    response = Response(
        model="m",
        content=[TextBlock(text="Checking."), CALL, second],
        stop_reason="tool_use",
        usage=Usage(),
    )
    assert response.tool_calls == [CALL, second]
    assert response.text == "Checking."


def test_content_blocks_parse_by_type():
    message = Message.model_validate(
        {"role": "user", "content": [{"type": "tool_result", "call_id": "call_1", "content": "ok"}]}
    )
    assert message.content == [ToolResult(call_id="call_1", content="ok")]


@pytest.mark.parametrize(
    ("role", "block"),
    [
        ("user", CALL),
        ("user", Reasoning(provider="anthropic", data={})),
        ("assistant", ToolResult(call_id="call_1", content="ok")),
    ],
)
def test_message_rejects_blocks_its_role_cannot_carry(role, block):
    with pytest.raises(ValidationError, match="cannot contain"):
        Message(role=role, content=[block])


def test_tool_spec_rejects_names_providers_refuse():
    with pytest.raises(ValidationError):
        ToolSpec(name="look up", description="d", input_type=Lookup)
    with pytest.raises(ValidationError):
        ToolSpec(name="x" * 65, description="d", input_type=Lookup)


def test_request_carries_tools():
    spec = ToolSpec(name="lookup", description="Find a vendor.", input_type=Lookup)
    request = Request(model="m", messages=[Message.user("x")], tools=[spec])
    assert request.tools == [spec]
