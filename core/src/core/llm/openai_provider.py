"""Adapter between core.llm types and the OpenAI Responses API."""

from collections.abc import Generator
from contextlib import AbstractContextManager
from typing import Any

import openai
from openai.lib._pydantic import to_strict_json_schema
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import ResponseUsage as OpenAIUsage

from core.llm.errors import ProviderUnavailableError, translate_sdk_errors
from core.llm.records import Usage
from core.llm.types import (
    ContentBlock,
    Message,
    Reasoning,
    Request,
    Response,
    StopReason,
    StreamEvent,
    TextBlock,
    TextDelta,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from core.settings import get_settings

FINAL_EVENTS = {"response.completed", "response.incomplete", "response.failed"}
"""Stream events that carry the finished response."""

INCOMPLETE_REASONS: dict[str, StopReason] = {
    "max_output_tokens": "max_tokens",
    "content_filter": "refusal",
}
"""Why a response ended with status "incomplete"; anything else becomes "other"."""


REASONING_FIELDS = {"type", "id", "summary", "encrypted_content"}
"""The parts of a reasoning item the API accepts back as input."""


def to_openai_items(message: Message) -> list[dict[str, Any]]:
    """One message as Responses input items, in order.

    Text becomes a message item; tool calls, tool results, and reasoning are items of their
    own. Consecutive text blocks share one message item.
    """
    items: list[dict[str, Any]] = []
    text: list[str] = []

    def flush_text() -> None:
        if text:
            items.append({"role": message.role, "content": "".join(text)})
            text.clear()

    for block in message.content:
        if isinstance(block, TextBlock):
            text.append(block.text)
            continue
        flush_text()
        match block:
            case ToolCall():
                items.append(
                    {
                        "type": "function_call",
                        "call_id": block.id,
                        "name": block.name,
                        "arguments": block.arguments,
                    }
                )
            case ToolResult():
                # No error flag on this API: the content itself must say the tool failed.
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": block.call_id,
                        "output": block.content,
                    }
                )
            case Reasoning() if block.provider == "openai":
                items.append(block.data)
            case Reasoning():
                pass  # another provider's reasoning means nothing here
    flush_text()
    return items


def to_openai_tool(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "name": spec.name,
        "description": spec.description,
        "parameters": to_strict_json_schema(spec.input_type),
        "strict": True,
    }


def to_openai_params(request: Request) -> dict[str, Any]:
    """Build keyword arguments for `client.responses.create`."""
    if request.stop_sequences:
        raise ValueError("the OpenAI Responses API does not support stop sequences")
    params: dict[str, Any] = {
        "model": request.model,
        "max_output_tokens": request.max_tokens,
        "input": [item for message in request.messages for item in to_openai_items(message)],
        "store": False,
    }
    if request.system is not None:
        params["instructions"] = request.system
    if request.output_type is not None:
        params["text"] = {
            "format": {
                "type": "json_schema",
                "name": request.output_type.__name__,
                "schema": to_strict_json_schema(request.output_type),
                "strict": True,
            }
        }
    if request.tools:
        params["tools"] = [to_openai_tool(spec) for spec in request.tools]
        # Without storage, reasoning survives between calls only if we carry it ourselves.
        params["include"] = ["reasoning.encrypted_content"]
    return params


def to_usage(usage: OpenAIUsage) -> Usage:
    """OpenAI's input_tokens includes cache reads and writes; split them out."""
    details = usage.input_tokens_details
    uncached = usage.input_tokens - details.cached_tokens - details.cache_write_tokens
    return Usage(
        # Clamp: reads + writes have been reported above the total; never go negative.
        input_tokens=max(uncached, 0),
        cache_read_tokens=details.cached_tokens,
        cache_write_tokens=details.cache_write_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
    )


def stop_reason_of(response: OpenAIResponse, refused: bool, called: bool) -> tuple[StopReason, str]:
    """Normalized stop reason plus a raw description, from status and incomplete details."""
    if response.status == "incomplete":
        reason = response.incomplete_details.reason if response.incomplete_details else None
        return INCOMPLETE_REASONS.get(reason or "", "other"), f"incomplete:{reason}"
    raw = response.status or "unknown"
    if response.status != "completed":
        return "other", raw
    if refused:
        return "refusal", raw
    return ("tool_use" if called else "end_turn"), raw


def from_openai_response(response: OpenAIResponse) -> Response:
    content: list[ContentBlock] = []
    refused = False
    for item in response.output:
        if item.type == "reasoning":
            data = item.model_dump(include=REASONING_FIELDS, exclude_none=True)
            content.append(Reasoning(provider="openai", data=data))
            continue
        if item.type == "function_call":
            content.append(ToolCall(id=item.call_id, name=item.name, arguments=item.arguments))
            continue
        if item.type != "message":
            raise ValueError(f"unsupported OpenAI output item: {item.type!r}")
        for part in item.content:
            if part.type == "output_text":
                content.append(TextBlock(text=part.text))
            else:  # "refusal"
                refused = True
                content.append(TextBlock(text=part.refusal))

    if response.usage is None:
        raise ValueError("OpenAI response has no usage")
    called = any(isinstance(block, ToolCall) for block in content)
    stop_reason, raw = stop_reason_of(response, refused, called)
    return Response(
        model=response.model,
        content=content,
        stop_reason=stop_reason,
        raw_stop_reason=raw,
        usage=to_usage(response.usage),
    )


class OpenAIProvider:
    """Sends a Request through the OpenAI SDK and returns a Response."""

    name = "openai"

    def __init__(self, client: openai.OpenAI | None = None) -> None:
        # max_retries=0: retries belong to our client, which records every attempt.
        self._client = client or openai.OpenAI(
            api_key=get_settings().require("openai_api_key"), max_retries=0
        )

    def complete(self, request: Request) -> Response:
        with self._translated_errors():
            response = self._client.responses.create(**to_openai_params(request))
        return from_openai_response(response)

    def stream(self, request: Request) -> Generator[StreamEvent]:
        final = None
        with (
            self._translated_errors(),
            self._client.responses.create(**to_openai_params(request), stream=True) as events,
        ):
            for event in events:
                if event.type == "response.output_text.delta":
                    yield TextDelta(text=event.delta)
                elif event.type in FINAL_EVENTS:
                    final = event.response
                elif event.type == "error":
                    raise ProviderUnavailableError(
                        f"stream error {event.code}: {event.message}", provider="openai"
                    )
        if final is None:
            raise ProviderUnavailableError("stream ended without a response", provider="openai")
        yield from_openai_response(final)

    def _translated_errors(self) -> AbstractContextManager[None]:
        return translate_sdk_errors("openai", openai.APIStatusError, openai.APIConnectionError)
