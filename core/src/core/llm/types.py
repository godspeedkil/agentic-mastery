"""Provider-independent request, message and response types.

Application code builds a Request and receives a Response; provider adapters translate
between these types and each SDK's own format.
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from core.llm.records import Usage

Role = Literal["user", "assistant"]

StopReason = Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "refusal", "other"]
"""Why generation ended, normalized across providers."""


class TextBlock(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["text"] = "text"
    text: str


ContentBlock = TextBlock
"""Becomes a union of block types when tool calls arrive in Phase 2."""


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Role
    content: list[ContentBlock] = Field(min_length=1)

    @classmethod
    def user(cls, text: str) -> Self:
        return cls(role="user", content=[TextBlock(text=text)])

    @classmethod
    def assistant(cls, text: str) -> Self:
        return cls(role="assistant", content=[TextBlock(text=text)])


class Request(BaseModel):
    """What to send. Sampling parameters are deliberately absent: current models reject them."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_attribute_docstrings=True)

    model: str
    """Exact model ID; prefer dated snapshots over aliases."""
    messages: list[Message] = Field(min_length=1)
    system: str | None = None
    max_tokens: PositiveInt = 1024
    stop_sequences: list[str] = Field(default_factory=list)


class Response(BaseModel):
    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    model: str
    """Model ID reported by the provider, which may differ from the requested alias."""
    content: list[ContentBlock]
    stop_reason: StopReason
    raw_stop_reason: str | None = None
    """The provider's original value, kept for debugging."""
    usage: Usage

    @property
    def text(self) -> str:
        """All text blocks joined; empty if there are none."""
        return "".join(block.text for block in self.content if isinstance(block, TextBlock))
