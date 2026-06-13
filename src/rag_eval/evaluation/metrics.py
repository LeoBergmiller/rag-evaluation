"""Framework-free retrieval metrics and bootstrap confidence intervals."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved = set(ranked_ids[:k])
    return len(retrieved & relevant_ids) / len(relevant_ids)


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    retrieved = set(ranked_ids[:k])
    return len(retrieved & relevant_ids) / k


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: set[str]) -> float:
    for i, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """NDCG@k with binary relevance gains."""
    if not relevant_ids:
        return 0.0

    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, chunk_id in enumerate(ranked_ids[:k], start=1)
        if chunk_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int,
    seed: int,
    ci: float = 0.95,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI. Returns (point estimate, lower, upper)."""
    data = np.asarray(values, dtype=float)
    point = float(statistic(data))

    rng = np.random.default_rng(seed)
    resampled = rng.choice(data, size=(n_resamples, len(data)), replace=True)
    boot_stats = statistic(resampled, axis=1)

    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boot_stats, alpha))
    hi = float(np.quantile(boot_stats, 1.0 - alpha))
    return point, lo, hi
