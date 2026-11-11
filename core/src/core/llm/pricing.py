"""Turn token usage into dollars using a per-model price table."""

import tomllib
from datetime import date
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, NonNegativeFloat

from core.llm.records import Usage

TOKENS_PER_UNIT = 1_000_000


class ModelPrice(BaseModel):
    """Prices for one model, in USD per million tokens."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    input: NonNegativeFloat
    output: NonNegativeFloat
    """Applies to all output, including reasoning."""
    cache_read: NonNegativeFloat | None = None
    """None if the model has no cache pricing."""
    cache_write: NonNegativeFloat | None = None
    """None if cache writes are not billed separately."""
    as_of: date | None = None
    """When the price was checked."""
    source: str | None = None
    """URL of the pricing page the numbers came from."""

    def cost(self, usage: Usage) -> float | None:
        """USD for one call, or None if the usage needs a rate this price doesn't have."""
        total = usage.input_tokens * self.input + usage.output_tokens * self.output
        for tokens, rate in [
            (usage.cache_read_tokens, self.cache_read),
            (usage.cache_write_tokens, self.cache_write),
        ]:
            if tokens == 0:
                continue
            if rate is None:
                return None
            total += tokens * rate
        return total / TOKENS_PER_UNIT


class PriceTable(BaseModel):
    """Prices keyed by exact model ID."""

    model_config = ConfigDict(frozen=True)

    prices: dict[str, ModelPrice]

    @classmethod
    def from_toml(cls, path: Path) -> Self:
        with path.open("rb") as f:
            return cls.model_validate(tomllib.load(f))

    def cost(self, model: str, usage: Usage) -> float | None:
        """USD for one call, or None if the model is not in the table."""
        price = self.prices.get(model)
        return price.cost(usage) if price else None
