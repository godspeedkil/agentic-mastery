from datetime import date

import pytest

from core.llm import ModelPrice, PriceTable, Usage

PRICE = ModelPrice(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75)


def test_cost_sums_each_token_type_at_its_rate():
    usage = Usage(
        input_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert PRICE.cost(usage) == pytest.approx(3.0 + 0.3 + 3.75 + 15.0)


def test_lesson_example_costs_one_eighty():
    # The 25-step agent from section 1.1 of the Phase 1 guide.
    usage = Usage(input_tokens=562_500, output_tokens=7_500)
    assert PRICE.cost(usage) == pytest.approx(1.80)


def test_cache_tokens_without_a_cache_rate_give_unknown_cost():
    no_cache = ModelPrice(input=3.0, output=15.0)
    assert no_cache.cost(Usage(input_tokens=10, cache_read_tokens=5)) is None
    assert no_cache.cost(Usage(input_tokens=10, output_tokens=5)) is not None


def test_unknown_model_gives_unknown_cost():
    table = PriceTable(prices={"known": PRICE})
    assert table.cost("unknown", Usage(input_tokens=1)) is None


def test_loads_from_toml(tmp_path):
    path = tmp_path / "prices.toml"
    path.write_text(
        '[prices."model-a"]\n'
        "input = 1.0\n"
        "output = 2.0\n"
        "as_of = 2026-11-10\n"
        'source = "https://example.com/pricing"\n'
    )
    table = PriceTable.from_toml(path)
    assert table.prices["model-a"].as_of == date(2026, 11, 10)
    assert table.cost("model-a", Usage(input_tokens=1_000_000)) == pytest.approx(1.0)
