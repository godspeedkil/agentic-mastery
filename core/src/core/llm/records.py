"""One normalized record per LLM call, written as a JSON line for later analysis."""

from datetime import UTC, datetime
from typing import Literal, Self
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    computed_field,
    model_validator,
)

Provider = Literal["anthropic", "openai"]


class Usage(BaseModel):
    """Token counts normalized across providers.

    The three input fields are disjoint and sum to the total input. Each provider reports
    usage in its own shape; its adapter converts that shape into this one.
    """

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    input_tokens: NonNegativeInt = 0
    """Input billed at the base rate: neither read from nor written to the cache."""
    cache_read_tokens: NonNegativeInt = 0
    """Input read from the prompt cache."""
    cache_write_tokens: NonNegativeInt = 0
    """Input written to the prompt cache."""
    output_tokens: NonNegativeInt = 0
    """All generated tokens, including reasoning."""
    reasoning_tokens: NonNegativeInt = 0
    """The part of output_tokens spent on reasoning, when the provider reports it."""

    @computed_field
    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @model_validator(mode="after")
    def _reasoning_within_output(self) -> Self:
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError("reasoning_tokens cannot exceed output_tokens")
        return self


class CallRecord(BaseModel):
    """Everything worth analyzing about one LLM call, except prompt and response content."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    id: str = Field(default_factory=lambda: uuid4().hex)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_id: str | None = None
    """Groups the calls of one benchmark or agent run."""
    labels: dict[str, str] = Field(default_factory=dict)
    """Caller-chosen tags, such as a case ID, for joining records with other results."""
    provider: Provider
    model: str
    """The exact model ID sent to the API."""
    attempt: PositiveInt = 1
    """1 for the first try; retries of the same request count up from there."""
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float | None = None
    """None when the model's price is unknown."""
    latency_ms: float | None = None
    """Wall-clock duration of the whole call."""
    ttft_ms: float | None = None
    """Time to first token; streaming calls only."""
    stop_reason: str | None = None
    error: str | None = None
    """Exception type and message if the call failed."""
