"""The models the benchmark compares, and where their prices live."""

from pathlib import Path
from typing import NamedTuple

from core.llm import PriceTable, Provider

PRICES = Path(__file__).parents[2] / "data" / "prices.toml"


class Model(NamedTuple):
    provider: Provider
    id: str


MODELS = [
    Model("anthropic", "claude-opus-5-5"),
    Model("anthropic", "claude-sonnet-5-5"),
    Model("anthropic", "claude-haiku-5-5"),
    Model("openai", "gpt-6.1-sol"),
    Model("openai", "gpt-6.1-terra"),
    Model("openai", "gpt-6-luna"),
]


def load_prices(path: Path = PRICES) -> PriceTable:
    return PriceTable.from_toml(path)
