"""A framework-free agent loop: call the model, run the tools it asks for, repeat."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from core.llm import Completion, LLMClient, Message, Request, ToolCall, ToolResult, ToolSpec


@dataclass(frozen=True)
class Tool[A: BaseModel]:
    """A ToolSpec plus the function that carries it out."""

    spec: ToolSpec
    run: Callable[[A], str]
    """Receives the validated arguments; returns text for the model to read."""


@dataclass
class AgentRun:
    """Everything a run produced: the full conversation, every call, and the answer."""

    messages: list[Message]
    completions: list[Completion] = field(default_factory=list)
    answer: BaseModel | None = None
    """The validated arguments of the answer tool, if the model called it."""

    @property
    def text(self) -> str:
        """The text of the last response; empty before the first one."""
        return self.completions[-1].response.text if self.completions else ""


def error_result(call: ToolCall, message: str) -> ToolResult:
    return ToolResult(call_id=call.id, content=message, is_error=True)


def run_tool(call: ToolCall, tools: Mapping[str, Tool]) -> ToolResult:
    """Run one tool call. Every failure becomes an error result the model can read."""
    tool = tools.get(call.name)
    if tool is None:
        available = ", ".join(sorted(tools))
        return error_result(call, f"Error: unknown tool {call.name!r}. Available: {available}.")
    try:
        arguments = tool.spec.input_type.model_validate_json(call.arguments)
    except ValidationError as exc:
        return error_result(call, f"Error: invalid arguments for {call.name}.\n{exc}")
    try:
        return ToolResult(call_id=call.id, content=tool.run(arguments))
    except Exception as exc:  # any failure inside a tool is reported, never raised
        return error_result(call, f"Error: {call.name} failed: {type(exc).__name__}: {exc}")


def run_agent(
    client: LLMClient,
    provider: str,
    request: Request,
    tools: Sequence[Tool],
    *,
    answer: ToolSpec | None = None,
    labels: Mapping[str, str] | None = None,
) -> AgentRun:
    """Loop until the model answers without calling a tool, or calls the answer tool.

    `request` supplies the model, system prompt, first messages, and limits; its tools are
    replaced by `tools` plus `answer`. Each call's record is labeled with its step number.
    A valid answer ends the run at once: calls after it in the same response are not run.
    """
    by_name = {tool.spec.name: tool for tool in tools}
    specs = [tool.spec for tool in tools] + ([answer] if answer else [])
    run = AgentRun(messages=list(request.messages))
    step = 0
    while True:
        step += 1
        step_request = request.model_copy(update={"messages": list(run.messages), "tools": specs})
        completion = client.complete(
            provider, step_request, labels={**(labels or {}), "step": str(step)}
        )
        run.completions.append(completion)
        response = completion.response
        if response.content:
            run.messages.append(Message(role="assistant", content=response.content))
        if not response.tool_calls:
            return run
        results = []
        for call in response.tool_calls:
            if answer is not None and call.name == answer.name:
                try:
                    run.answer = answer.input_type.model_validate_json(call.arguments)
                    return run
                except ValidationError as exc:
                    results.append(error_result(call, f"Error: invalid answer.\n{exc}"))
                    continue
            results.append(run_tool(call, by_name))
        run.messages.append(Message(role="user", content=results))
