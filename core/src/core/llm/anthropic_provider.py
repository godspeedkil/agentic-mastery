"""Adapter between core.llm types and the Anthropic Messages API."""

from typing import Any

import anthropic
from anthropic.types import Message as AnthropicMessage
from anthropic.types import Usage as AnthropicUsage

from core.llm.errors import ProviderUnavailableError, error_from_status
from core.llm.records import Usage
from core.llm.types import Request, Response, StopReason, TextBlock
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

SKIPPED_BLOCK_TYPES = {"thinking", "redacted_thinking"}
"""Reasoning blocks: billed through usage, not part of the answer text."""


def to_anthropic_params(request: Request) -> dict[str, Any]:
    """Build keyword arguments for `client.messages.create`."""
    params: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": [
            {
                "role": message.role,
                "content": [{"type": "text", "text": block.text} for block in message.content],
            }
            for message in request.messages
        ],
    }
    if request.system is not None:
        params["system"] = request.system
    if request.stop_sequences:
        params["stop_sequences"] = request.stop_sequences
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
    content = []
    for block in message.content:
        if block.type == "text":
            content.append(TextBlock(text=block.text))
        elif block.type not in SKIPPED_BLOCK_TYPES:
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
        try:
            message = self._client.messages.create(**to_anthropic_params(request))
        except anthropic.APIStatusError as exc:
            raise error_from_status(
                "anthropic", exc.status_code, exc.message, exc.response.headers
            ) from exc
        except anthropic.APIConnectionError as exc:  # includes timeouts
            raise ProviderUnavailableError(str(exc), provider="anthropic") from exc
        return from_anthropic_message(message)
