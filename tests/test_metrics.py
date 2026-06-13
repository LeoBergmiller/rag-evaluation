import numpy as np

from rag_eval.evaluation.metrics import (
    bootstrap_ci,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k() -> None:
    ranked = ["a", "b", "c", "d"]
    relevant = {"a", "c"}

    assert recall_at_k(ranked, relevant, k=4) == 1.0
    assert recall_at_k(ranked, relevant, k=1) == 0.5
    assert recall_at_k(ranked, {"z"}, k=4) == 0.0
    assert recall_at_k(ranked, set(), k=4) == 0.0


def test_precision_at_k() -> None:
    ranked = ["a", "b", "c", "d"]
    relevant = {"a", "c"}

    assert precision_at_k(ranked, relevant, k=4) == 0.5
    assert precision_at_k(ranked, relevant, k=2) == 0.5
    assert precision_at_k(ranked, {"z"}, k=4) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_ndcg_at_k_relevant_first() -> None:
    ranked = ["a", "b", "c"]
    relevant = {"a"}

    assert ndcg_at_k(ranked, relevant, k=3) == 1.0


def test_ndcg_at_k_relevant_absent() -> None:
    ranked = ["a", "b", "c"]
    relevant = {"z"}

    assert ndcg_at_k(ranked, relevant, k=3) == 0.0


def test_ndcg_at_k_relevant_not_first() -> None:
    ranked = ["a", "b", "c"]
    relevant = {"b"}

    expected = (1.0 / np.log2(3)) / (1.0 / np.log2(2))
    assert ndcg_at_k(ranked, relevant, k=3) == expected


def test_bootstrap_ci_deterministic() -> None:
    values = [0.1, 0.5, 0.9, 0.3, 0.7]

    point1, lo1, hi1 = bootstrap_ci(values, n_resamples=200, seed=42)
    point2, lo2, hi2 = bootstrap_ci(values, n_resamples=200, seed=42)

    assert (point1, lo1, hi1) == (point2, lo2, hi2)
    assert lo1 <= point1 <= hi1


def test_bootstrap_ci_constant() -> None:
    values = [0.5, 0.5, 0.5, 0.5]

    point, lo, hi = bootstrap_ci(values, n_resamples=100, seed=0)

    assert point == lo == hi == 0.5
