"""Provider-independent request, message and response types.

Application code builds a Request and receives a Response; provider adapters translate
between these types and each SDK's own format.
"""

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from core.llm.records import Usage

Role = Literal["user", "assistant"]

StopReason = Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "refusal", "other"]
"""Why generation ended, normalized across providers."""


class TextBlock(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["text"] = "text"
    text: str


class ToolCall(BaseModel):
    """The model asking to run a tool. Appears in assistant messages."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    type: Literal["tool_call"] = "tool_call"
    id: str
    """Provider-assigned; the matching ToolResult must carry it."""
    name: str
    arguments: str
    """The arguments as a JSON string, exactly as the model wrote them."""


class ToolResult(BaseModel):
    """What running a tool produced. Appears in user messages."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    type: Literal["tool_result"] = "tool_result"
    call_id: str
    """The `id` of the ToolCall this answers."""
    content: str
    is_error: bool = False
    """True when the tool failed; `content` then explains why, for the model to read."""


class Reasoning(BaseModel):
    """Reasoning returned by a provider, kept so it can be sent back unchanged."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    type: Literal["reasoning"] = "reasoning"
    provider: str
    """Only this provider's adapter can send it back; others skip it."""
    data: dict[str, Any]
    """The provider's block or item, exactly as received."""


ContentBlock = Annotated[TextBlock | ToolCall | ToolResult | Reasoning, Field(discriminator="type")]

ROLE_BLOCKS: dict[Role, set[str]] = {
    "user": {"text", "tool_result"},
    "assistant": {"text", "tool_call", "reasoning"},
}
"""Which block types each role may carry."""


class ToolSpec(BaseModel):
    """A tool the model may call: its name, what it does, and the arguments it takes."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    """Letters, digits, `_` and `-`, at most 64: a name both providers accept."""
    description: str
    """When and how to use the tool. The model reads it to decide."""
    input_type: type[BaseModel]
    """The arguments as a Pydantic model; its JSON Schema is sent to the provider."""


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Role
    content: list[ContentBlock] = Field(min_length=1)

    @model_validator(mode="after")
    def _blocks_fit_role(self) -> Self:
        allowed = ROLE_BLOCKS[self.role]
        for block in self.content:
            if block.type not in allowed:
                raise ValueError(f"a {self.role} message cannot contain {block.type!r} blocks")
        return self

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
    output_type: type[BaseModel] | None = None
    """When set, the provider constrains the answer to this model's JSON Schema."""
    tools: list[ToolSpec] = Field(default_factory=list)
    """Tools the model may call; empty means a plain completion."""


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

    @property
    def tool_calls(self) -> list[ToolCall]:
        """The tools the model asked to run, in order; empty if it answered directly."""
        return [block for block in self.content if isinstance(block, ToolCall)]


class TextDelta(BaseModel):
    """A piece of answer text, as it arrives during streaming."""

    model_config = ConfigDict(frozen=True)

    text: str


StreamEvent = TextDelta | Response
"""Adapters stream any number of TextDeltas, then exactly one final Response."""
