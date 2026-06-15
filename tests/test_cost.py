import logging

from rag_eval.evaluation.cost import PRICING, query_cost_usd


def test_known_model_computes_expected_cost() -> None:
    price = PRICING["claude-sonnet-4-6"]
    cost = query_cost_usd(
        "claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=1_000_000
    )

    assert cost == price.input_per_million + price.output_per_million


def test_unknown_model_returns_zero_and_warns(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        cost = query_cost_usd(
            "some-future-model", input_tokens=1000, output_tokens=1000
        )

    assert cost == 0.0
    assert "some-future-model" in caplog.text
