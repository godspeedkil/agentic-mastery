"""Adapter between core.llm types and the Anthropic Messages API."""

import json
from collections.abc import Generator
from contextlib import AbstractContextManager
from typing import Any

import anthropic
from anthropic.types import Message as AnthropicMessage
from anthropic.types import Usage as AnthropicUsage

from core.llm.errors import translate_sdk_errors
from core.llm.records import Usage
from core.llm.types import (
    ContentBlock,
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

STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "end_turn",
    "max_tokens": "max_tokens",
    "model_context_window_exceeded": "max_tokens",
    "stop_sequence": "stop_sequence",
    "tool_use": "tool_use",
    "refusal": "refusal",
}
"""Anthropic stop reasons we map explicitly; anything else becomes "other"."""

REASONING_BLOCK_TYPES = {"thinking", "redacted_thinking"}
"""Reasoning blocks: kept whole, because they must go back unchanged in a tool loop."""


def to_anthropic_block(block: ContentBlock) -> dict[str, Any] | None:
    """One content block in Anthropic's shape; None for another provider's reasoning."""
    match block:
        case TextBlock():
            return {"type": "text", "text": block.text}
        case ToolCall():
            return {
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": json.loads(block.arguments),
            }
        case Reasoning():
            return block.data if block.provider == "anthropic" else None
        case ToolResult():
            return {
                "type": "tool_result",
                "tool_use_id": block.call_id,
                "content": block.content,
                "is_error": block.is_error,
            }


def to_anthropic_tool(spec: ToolSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": anthropic.transform_schema(spec.input_type),
        "strict": True,
    }


def to_anthropic_params(request: Request) -> dict[str, Any]:
    """Build keyword arguments for `client.messages.create`."""
    messages = []
    for message in request.messages:
        blocks = [b for b in map(to_anthropic_block, message.content) if b is not None]
        # Anthropic requires tool results before any other content in a user message.
        blocks.sort(key=lambda b: b["type"] != "tool_result")
        messages.append({"role": message.role, "content": blocks})
    params: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": messages,
    }
    if request.system is not None:
        params["system"] = request.system
    if request.stop_sequences:
        params["stop_sequences"] = request.stop_sequences
    if request.output_type is not None:
        schema = anthropic.transform_schema(request.output_type)
        params["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
    if request.tools:
        params["tools"] = [to_anthropic_tool(spec) for spec in request.tools]
    return params


def to_usage(usage: AnthropicUsage) -> Usage:
    """Anthropic reports cache reads and writes separately from input_tokens."""
    details = usage.output_tokens_details
    return Usage(
        input_tokens=usage.input_tokens,
        cache_read_tokens=usage.cache_read_input_tokens or 0,
        cache_write_tokens=usage.cache_creation_input_tokens or 0,
        output_tokens=usage.output_tokens,
        reasoning_tokens=details.thinking_tokens if details else 0,
    )


def from_anthropic_message(message: AnthropicMessage) -> Response:
    content: list[ContentBlock] = []
    for block in message.content:
        if block.type == "text":
            content.append(TextBlock(text=block.text))
        elif block.type == "tool_use":
            content.append(
                ToolCall(id=block.id, name=block.name, arguments=json.dumps(block.input))
            )
        elif block.type in REASONING_BLOCK_TYPES:
            data = block.model_dump(include={"type", "thinking", "signature", "data"})
            content.append(Reasoning(provider="anthropic", data=data))
        else:
            raise ValueError(f"unsupported Anthropic content block: {block.type!r}")
    raw = message.stop_reason
    return Response(
        model=message.model,
        content=content,
        stop_reason=STOP_REASONS.get(raw, "other") if raw else "other",
        raw_stop_reason=raw,
        usage=to_usage(message.usage),
    )


class AnthropicProvider:
    """Sends a Request through the Anthropic SDK and returns a Response."""

    name = "anthropic"

    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        # max_retries=0: retries belong to our client, which records every attempt.
        self._client = client or anthropic.Anthropic(
            api_key=get_settings().require("anthropic_api_key"), max_retries=0
        )

    def complete(self, request: Request) -> Response:
        with self._translated_errors():
            message = self._client.messages.create(**to_anthropic_params(request))
        return from_anthropic_message(message)

    def stream(self, request: Request) -> Generator[StreamEvent]:
        with (
            self._translated_errors(),
            self._client.messages.stream(**to_anthropic_params(request)) as stream,
        ):
            for text in stream.text_stream:
                yield TextDelta(text=text)
            message = stream.get_final_message()
        yield from_anthropic_message(message)

    def _translated_errors(self) -> AbstractContextManager[None]:
        return translate_sdk_errors(
            "anthropic", anthropic.APIStatusError, anthropic.APIConnectionError
        )
