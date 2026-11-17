import json

import pytest
from pydantic import BaseModel

from core.agent import Limits, Tool, call_key, run_agent, run_tool
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


def reply(*blocks, stop_reason=None, tokens: int = 0) -> Response:
    calls = any(isinstance(block, ToolCall) for block in blocks)
    return Response(
        model="m",
        content=list(blocks),
        stop_reason=stop_reason or ("tool_use" if calls else "end_turn"),
        usage=Usage(input_tokens=tokens),
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
    assert result.stop_reason == "replied"
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
    assert result.stop_reason == "answered"
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


def lookups(n: int, same: bool = False) -> list[Response]:
    return [
        reply(call("lookup_vendor", {"vendor_id": "V-1" if same else f"V-{i}"}, id=f"c{i}"))
        for i in range(n)
    ]


def test_cut_off_reply_is_a_provider_stop():
    provider = ScriptedProvider(reply(TextBlock(text="The vend"), stop_reason="max_tokens"))
    assert run(provider).stop_reason == "provider_stop"


def test_step_limit_stops_before_the_next_call():
    provider = ScriptedProvider(*lookups(3))
    result = run(provider, limits=Limits(max_steps=2))
    assert result.stop_reason == "max_steps"
    assert len(result.completions) == 2


def test_token_budget_counts_every_step():
    provider = ScriptedProvider(reply(call("lookup_vendor", {"vendor_id": "V-1"}), tokens=600))
    result = run(provider, limits=Limits(max_tokens=500))
    assert result.stop_reason == "token_budget"
    assert result.tokens == 600


def test_repeating_the_same_call_stops_the_run():
    provider = ScriptedProvider(*lookups(5, same=True))
    result = run(provider, limits=Limits(max_repeats=3))
    assert result.stop_reason == "repeated_call"
    assert len(result.completions) == 3


def test_call_key_ignores_key_order_and_spacing():
    first = ToolCall(id="a", name="t", arguments='{"x": 1, "y": 2}')
    second = ToolCall(id="b", name="t", arguments='{"y":2,"x":1}')
    assert call_key(first) == call_key(second)


def test_error_streak_stops_the_run():
    failing = [reply(call("find_vendor", {}, id=f"c{i}")) for i in range(5)]
    result = run(ScriptedProvider(*failing), limits=Limits(max_error_streak=2))
    assert result.stop_reason == "error_streak"
    assert len(result.completions) == 2


def test_a_success_resets_the_error_streak():
    script = [
        reply(call("find_vendor", {}, id="a")),
        reply(call("lookup_vendor", {"vendor_id": "V-1"}, id="b")),
        reply(call("find_vendor", {}, id="c")),
        reply(TextBlock(text="Done.")),
    ]
    result = run(ScriptedProvider(*script), limits=Limits(max_error_streak=2))
    assert result.stop_reason == "replied"


def test_graceful_last_turn_offers_only_the_answer_tool():
    provider = ScriptedProvider(*lookups(2), reply(call("submit_decision", {"approve": False})))
    result = run(provider, answer=SUBMIT, limits=Limits(max_steps=2))
    assert result.stop_reason == "max_steps"
    assert result.answer == Decision(approve=False)
    last = provider.requests[-1]
    assert [spec.name for spec in last.tools] == ["submit_decision"]
    assert last.messages[-1] == Message.user(
        "Stop: max_steps. Call submit_decision now with your best answer."
    )


def test_no_graceful_turn_when_disabled():
    provider = ScriptedProvider(*lookups(3))
    result = run(provider, answer=SUBMIT, limits=Limits(max_steps=2, graceful=False))
    assert len(result.completions) == 2
    assert result.answer is None
