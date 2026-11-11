from core.llm import Usage
from p01.models import MODELS, load_prices


def test_every_benchmarked_model_has_a_sourced_price():
    prices = load_prices().prices
    for model in MODELS:
        price = prices[model.id]
        assert price.as_of is not None, model.id
        assert price.source and price.source.startswith("https://"), model.id
        assert price.cache_read is not None and price.cache_write is not None, model.id


def test_model_ids_are_unique():
    assert len({model.id for model in MODELS}) == len(MODELS)


def test_a_typical_call_costs_a_fraction_of_a_cent_to_a_few_cents():
    # About what one benchmark call uses: a short prompt and a few hundred output tokens.
    usage = Usage(input_tokens=600, output_tokens=400)
    prices = load_prices()
    for model in MODELS:
        cost = prices.cost(model.id, usage)
        assert cost is not None and 0 < cost < 0.05, model.id
