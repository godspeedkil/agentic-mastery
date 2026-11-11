"""Adapter between core.llm types and the OpenAI Responses API."""

from contextlib import AbstractContextManager
from typing import Any

import openai
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import ResponseUsage as OpenAIUsage

from core.llm.errors import translate_sdk_errors
from core.llm.records import Usage
from core.llm.types import Request, Response, StopReason, TextBlock
from core.settings import get_settings

INCOMPLETE_REASONS: dict[str, StopReason] = {
    "max_output_tokens": "max_tokens",
    "content_filter": "refusal",
}
"""Why a response ended with status "incomplete"; anything else becomes "other"."""


def to_openai_params(request: Request) -> dict[str, Any]:
    """Build keyword arguments for `client.responses.create`."""
    if request.stop_sequences:
        raise ValueError("the OpenAI Responses API does not support stop sequences")
    params: dict[str, Any] = {
        "model": request.model,
        "max_output_tokens": request.max_tokens,
        "input": [
            {"role": message.role, "content": "".join(block.text for block in message.content)}
            for message in request.messages
        ],
        "store": False,
    }
    if request.system is not None:
        params["instructions"] = request.system
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


def stop_reason_of(response: OpenAIResponse, refused: bool) -> tuple[StopReason, str]:
    """Normalized stop reason plus a raw description, from status and incomplete details."""
    if response.status == "incomplete":
        reason = response.incomplete_details.reason if response.incomplete_details else None
        return INCOMPLETE_REASONS.get(reason or "", "other"), f"incomplete:{reason}"
    raw = response.status or "unknown"
    if response.status != "completed":
        return "other", raw
    return ("refusal" if refused else "end_turn"), raw


def from_openai_response(response: OpenAIResponse) -> Response:
    content = []
    refused = False
    for item in response.output:
        if item.type == "reasoning":
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
    stop_reason, raw = stop_reason_of(response, refused)
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

    def _translated_errors(self) -> AbstractContextManager[None]:
        return translate_sdk_errors("openai", openai.APIStatusError, openai.APIConnectionError)
