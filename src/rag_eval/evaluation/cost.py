"""Token -> USD costing for generation models.

Reference data only (not a tuning knob) — the single sanctioned place for
provider pricing. Rates as of 2026-06, USD per 1M tokens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: float
    output_per_million: float


PRICING: dict[str, ModelPrice] = {
    "claude-sonnet-4-6": ModelPrice(input_per_million=3.0, output_per_million=15.0),
    "gpt-4o": ModelPrice(input_per_million=2.5, output_per_million=10.0),
}


def query_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price = PRICING.get(model)
    if price is None:
        logger.warning("No pricing data for model %r; treating cost as 0.0", model)
        return 0.0
    return (
        input_tokens / 1_000_000 * price.input_per_million
        + output_tokens / 1_000_000 * price.output_per_million
    )
