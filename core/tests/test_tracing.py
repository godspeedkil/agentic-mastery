import json

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import BaseModel

from core.agent import Tool, run_agent
from core.llm import (
    LLMClient,
    Message,
    Request,
    Response,
    TextBlock,
    ToolCall,
    ToolSpec,
    Usage,
)

EXPORTER = InMemorySpanExporter()


@pytest.fixture(scope="module", autouse=True)
def tracer_provider():
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
    trace.set_tracer_provider(provider)


@pytest.fixture
def spans():
    EXPORTER.clear()
    yield lambda: {span.name: span for span in EXPORTER.get_finished_spans()}


class Lookup(BaseModel):
    vendor_id: str


class Decision(BaseModel):
    approve: bool


LOOKUP = Tool(
    ToolSpec(name="lookup_vendor", description="Find a vendor.", input_type=Lookup),
    lambda args: json.dumps({"id": args.vendor_id}),
)
SUBMIT = ToolSpec(name="submit_decision", description="Decide.", input_type=Decision)


class ScriptedProvider:
    name = "anthropic"

    def __init__(self, *responses: Response) -> None:
        self.responses = list(responses)

    def complete(self, request: Request) -> Response:
        return self.responses.pop(0)


def reply(*blocks) -> Response:
    calls = any(isinstance(block, ToolCall) for block in blocks)
    return Response(
        model="model-2026",
        content=list(blocks),
        stop_reason="tool_use" if calls else "end_turn",
        usage=Usage(input_tokens=100, output_tokens=20),
    )


def run(*responses: Response, **kwargs):
    client = LLMClient({"anthropic": ScriptedProvider(*responses)})
    request = Request(model="model", messages=[Message.user("Check invoice 7.")])
    return run_agent(client, "anthropic", request, [LOOKUP], name="approver", **kwargs)


LOOKUP_CALL = ToolCall(id="call_1", name="lookup_vendor", arguments='{"vendor_id": "V-1"}')


def test_a_run_is_one_trace_shaped_like_the_loop(spans):
    run(reply(LOOKUP_CALL), reply(TextBlock(text="Done.")))
    finished = EXPORTER.get_finished_spans()
    names = [span.name for span in finished]
    assert names.count("chat model") == 2
    by_name = spans()
    root = by_name["invoke_agent approver"]
    assert root.parent is None
    assert {span.context.trace_id for span in finished} == {root.context.trace_id}
    assert by_name["execute_tool lookup_vendor"].parent.span_id == root.context.span_id


def test_chat_spans_carry_genai_attributes(spans):
    run(reply(TextBlock(text="Done.")))
    chat = spans()["chat model"].attributes
    assert chat["gen_ai.operation.name"] == "chat"
    assert chat["gen_ai.provider.name"] == "anthropic"
    assert chat["gen_ai.request.model"] == "model"
    assert chat["gen_ai.response.model"] == "model-2026"
    assert chat["gen_ai.usage.input_tokens"] == 100
    assert chat["gen_ai.usage.output_tokens"] == 20
    assert tuple(chat["gen_ai.response.finish_reasons"]) == ("end_turn",)


def test_the_agent_span_records_how_the_run_ended(spans):
    run(reply(LOOKUP_CALL), reply(TextBlock(text="Done.")))
    agent = spans()["invoke_agent approver"].attributes
    assert agent["gen_ai.operation.name"] == "invoke_agent"
    assert agent["gen_ai.agent.name"] == "approver"
    assert agent["agent.stop_reason"] == "replied"
    assert agent["agent.steps"] == 2
    assert agent["gen_ai.usage.input_tokens"] == 200


def test_a_failed_tool_call_is_an_error_span(spans):
    bad = ToolCall(id="call_9", name="find_vendor", arguments="{}")
    run(reply(bad), reply(TextBlock(text="Done.")))
    tool = spans()["execute_tool find_vendor"]
    assert tool.attributes["error.type"] == "unknown_tool"
    assert tool.attributes["gen_ai.tool.call.id"] == "call_9"
    assert tool.status.status_code is StatusCode.ERROR


def test_tool_content_is_left_out_unless_allowed(spans):
    run(reply(LOOKUP_CALL), reply(TextBlock(text="Done.")))
    assert "gen_ai.tool.call.arguments" not in spans()["execute_tool lookup_vendor"].attributes


def test_tool_content_is_recorded_when_allowed(spans):
    run(reply(LOOKUP_CALL), reply(TextBlock(text="Done.")), capture_content=True)
    tool = spans()["execute_tool lookup_vendor"].attributes
    assert tool["gen_ai.tool.call.arguments"] == '{"vendor_id": "V-1"}'
    assert tool["gen_ai.tool.call.result"] == '{"id": "V-1"}'


def test_the_answer_tool_gets_a_span_too(spans):
    answer = ToolCall(id="call_2", name="submit_decision", arguments='{"approve": true}')
    run(reply(answer), answer=SUBMIT)
    tool = spans()["execute_tool submit_decision"]
    assert tool.attributes["gen_ai.tool.name"] == "submit_decision"
    assert "error.type" not in tool.attributes
