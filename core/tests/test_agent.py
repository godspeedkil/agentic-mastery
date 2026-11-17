import json

import pytest
from pydantic import BaseModel

from core.agent import Tool, run_agent, run_tool
from core.llm import (
    LLMClient,
    Message,
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


class Decision(BaseModel):
    approve: bool


def lookup_vendor(args: Lookup) -> str:
    if args.vendor_id == "V-404":
        raise KeyError(args.vendor_id)
    return json.dumps({"id": args.vendor_id, "name": "Acme Ltd"})


LOOKUP = Tool(
    ToolSpec(name="lookup_vendor", description="Find a vendor.", input_type=Lookup),
    lookup_vendor,
)
SUBMIT = ToolSpec(name="submit_decision", description="Give the decision.", input_type=Decision)
TOOLS = {"lookup_vendor": LOOKUP}
REQUEST = Request(model="m", messages=[Message.user("Check invoice 7.")])


def call(name: str, arguments: dict, id: str = "call_1") -> ToolCall:
    return ToolCall(id=id, name=name, arguments=json.dumps(arguments))


def reply(*blocks) -> Response:
    calls = any(isinstance(block, ToolCall) for block in blocks)
    return Response(
        model="m",
        content=list(blocks),
        stop_reason="tool_use" if calls else "end_turn",
        usage=Usage(),
    )


class ScriptedProvider:
    """Returns the scripted responses in order and keeps every request it received."""

    name = "anthropic"

    def __init__(self, *responses: Response) -> None:
        self.responses = list(responses)
        self.requests: list[Request] = []

    def complete(self, request: Request) -> Response:
        self.requests.append(request)
        return self.responses.pop(0)


def run(provider: ScriptedProvider, **kwargs):
    return run_agent(LLMClient({"anthropic": provider}), "anthropic", REQUEST, [LOOKUP], **kwargs)


def test_tool_result_carries_the_output():
    result = run_tool(call("lookup_vendor", {"vendor_id": "V-1"}), TOOLS)
    assert result == ToolResult(call_id="call_1", content='{"id": "V-1", "name": "Acme Ltd"}')


@pytest.mark.parametrize(
    ("tool_call", "expected"),
    [
        (call("find_vendor", {}), "unknown tool 'find_vendor'. Available: lookup_vendor."),
        (call("lookup_vendor", {"vendor": "V-1"}), "invalid arguments for lookup_vendor"),
        (call("lookup_vendor", {"vendor_id": "V-404"}), "lookup_vendor failed: KeyError"),
    ],
)
def test_every_failure_becomes_an_error_result(tool_call, expected):
    result = run_tool(tool_call, TOOLS)
    assert result.is_error
    assert result.content.startswith("Error:")
    assert expected in result.content


def test_loop_runs_tools_until_the_model_answers():
    provider = ScriptedProvider(
        reply(call("lookup_vendor", {"vendor_id": "V-1"})),
        reply(TextBlock(text="Vendor is Acme Ltd.")),
    )
    result = run(provider)
    assert result.text == "Vendor is Acme Ltd."
    assert [m.role for m in result.messages] == ["user", "assistant", "user", "assistant"]
    second = provider.requests[1]
    assert second.messages[-1].content == [
        ToolResult(call_id="call_1", content='{"id": "V-1", "name": "Acme Ltd"}')
    ]
    assert [spec.name for spec in second.tools] == ["lookup_vendor"]


def test_parallel_calls_get_their_results_in_one_turn():
    provider = ScriptedProvider(
        reply(
            call("lookup_vendor", {"vendor_id": "V-1"}, id="a"),
            call("lookup_vendor", {"vendor_id": "V-2"}, id="b"),
        ),
        reply(TextBlock(text="Done.")),
    )
    result = run(provider)
    results = result.messages[2].content
    assert [r.call_id for r in results] == ["a", "b"]


def test_answer_tool_ends_the_run_with_a_validated_answer():
    provider = ScriptedProvider(reply(call("submit_decision", {"approve": True})))
    result = run(provider, answer=SUBMIT)
    assert result.answer == Decision(approve=True)
    assert [spec.name for spec in provider.requests[0].tools] == [
        "lookup_vendor",
        "submit_decision",
    ]


def test_invalid_answer_goes_back_as_an_error_and_the_loop_continues():
    provider = ScriptedProvider(
        reply(call("submit_decision", {"approve": "maybe"})),
        reply(call("submit_decision", {"approve": False}, id="call_2")),
    )
    result = run(provider, answer=SUBMIT)
    assert result.answer == Decision(approve=False)
    [error] = provider.requests[1].messages[-1].content
    assert error.is_error
    assert "invalid answer" in error.content


def test_each_call_is_labeled_with_its_step():
    provider = ScriptedProvider(
        reply(call("lookup_vendor", {"vendor_id": "V-1"})),
        reply(TextBlock(text="Done.")),
    )
    result = run(provider, labels={"case": "inv-7"})
    assert [c.record.labels for c in result.completions] == [
        {"case": "inv-7", "step": "1"},
        {"case": "inv-7", "step": "2"},
    ]


def test_earlier_requests_are_not_changed_by_later_steps():
    provider = ScriptedProvider(
        reply(call("lookup_vendor", {"vendor_id": "V-1"})),
        reply(TextBlock(text="Done.")),
    )
    run(provider)
    assert len(provider.requests[0].messages) == 1
