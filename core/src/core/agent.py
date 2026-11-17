"""A framework-free agent loop: call the model, run the tools it asks for, repeat."""

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ValidationError

from core.llm import (
    Completion,
    LLMClient,
    Message,
    Request,
    ToolCall,
    ToolResult,
    ToolSpec,
)

AgentStopReason = Literal[
    "answered",  # the answer tool was called with valid arguments
    "replied",  # the model ended without calling a tool
    "provider_stop",  # the response was cut off or refused and requested nothing
    "max_steps",
    "token_budget",
    "repeated_call",
    "error_streak",
]
CAPS: set[AgentStopReason] = {"max_steps", "token_budget", "repeated_call", "error_streak"}
"""Stop reasons that end a run before the model finished; they get a graceful last turn."""

PROVIDER_STOPS = {"max_tokens", "refusal"}


@dataclass(frozen=True)
class Tool[A: BaseModel]:
    """A ToolSpec plus the function that carries it out."""

    spec: ToolSpec
    run: Callable[[A], str]
    """Receives the validated arguments; returns text for the model to read."""


@dataclass(frozen=True)
class Limits:
    """When a run is stopped even though the model hasn't finished."""

    max_steps: int = 20
    """Model calls allowed, not counting the graceful last turn."""
    max_tokens: int = 200_000
    """Input plus output tokens across all steps."""
    max_repeats: int = 3
    """Stop once one call, same tool and same arguments, has run this many times."""
    max_error_streak: int = 3
    """Stop after this many error results in a row."""
    graceful: bool = True
    """On a cap, give the model one last turn with only the answer tool."""


DEFAULT_LIMITS = Limits()


@dataclass
class AgentRun:
    """Everything a run produced: the full conversation, every call, and how it ended."""

    messages: list[Message]
    completions: list[Completion] = field(default_factory=list)
    answer: BaseModel | None = None
    """The validated arguments of the answer tool, if the model called it."""
    stop_reason: AgentStopReason | None = None

    @property
    def text(self) -> str:
        """The text of the last response; empty before the first one."""
        return self.completions[-1].response.text if self.completions else ""

    @property
    def tokens(self) -> int:
        """Input plus output tokens across all calls so far."""
        return sum(
            c.record.usage.total_input_tokens + c.record.usage.output_tokens
            for c in self.completions
        )


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


def call_key(call: ToolCall) -> tuple[str, str]:
    """Identifies a call by tool and arguments, ignoring key order and spacing."""
    try:
        arguments = json.dumps(json.loads(call.arguments), sort_keys=True)
    except json.JSONDecodeError:
        arguments = call.arguments
    return call.name, arguments


def run_agent(
    client: LLMClient,
    provider: str,
    request: Request,
    tools: Sequence[Tool],
    *,
    answer: ToolSpec | None = None,
    limits: Limits = DEFAULT_LIMITS,
    labels: Mapping[str, str] | None = None,
) -> AgentRun:
    """Loop until the model finishes or a limit stops it; `stop_reason` says which.

    `request` supplies the model, system prompt, first messages, and output budget; its tools
    are replaced by `tools` plus `answer`. Each call's record is labeled with its step number.
    A valid answer ends the run at once: calls after it in the same response are not run.
    """
    by_name = {tool.spec.name: tool for tool in tools}
    specs = [tool.spec for tool in tools] + ([answer] if answer else [])
    run = AgentRun(messages=list(request.messages))
    calls_seen: Counter[tuple[str, str]] = Counter()
    error_streak = 0

    def step(offered: list[ToolSpec]) -> list[ToolCall]:
        step_request = request.model_copy(update={"messages": list(run.messages), "tools": offered})
        completion = client.complete(
            provider, step_request, labels={**(labels or {}), "step": str(len(run.completions) + 1)}
        )
        run.completions.append(completion)
        response = completion.response
        if response.content:
            run.messages.append(Message(role="assistant", content=response.content))
        return response.tool_calls

    def take_answer(call: ToolCall) -> ToolResult | None:
        """Store a valid answer and return None; return an error result for an invalid one."""
        assert answer is not None
        try:
            run.answer = answer.input_type.model_validate_json(call.arguments)
        except ValidationError as exc:
            return error_result(call, f"Error: invalid answer.\n{exc}")
        return None

    def stop(reason: AgentStopReason) -> AgentRun:
        run.stop_reason = reason
        if reason in CAPS and limits.graceful and answer is not None:
            note = f"Stop: {reason}. Call {answer.name} now with your best answer."
            run.messages.append(Message.user(note))
            for call in step([answer]):
                if call.name == answer.name and take_answer(call) is None:
                    break
        return run

    while True:
        if len(run.completions) >= limits.max_steps:
            return stop("max_steps")
        if run.tokens >= limits.max_tokens:
            return stop("token_budget")
        calls = step(specs)
        if not calls:
            last = run.completions[-1].response
            return stop("provider_stop" if last.stop_reason in PROVIDER_STOPS else "replied")
        results = []
        for call in calls:
            if answer is not None and call.name == answer.name:
                if (error := take_answer(call)) is None:
                    return stop("answered")
                results.append(error)
                continue
            calls_seen[call_key(call)] += 1
            results.append(run_tool(call, by_name))
        run.messages.append(Message(role="user", content=results))
        for result in results:
            error_streak = error_streak + 1 if result.is_error else 0
        if max(calls_seen.values(), default=0) >= limits.max_repeats:
            return stop("repeated_call")
        if error_streak >= limits.max_error_streak:
            return stop("error_streak")
