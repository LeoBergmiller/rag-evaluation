"""Measure run-to-run noise of the dense baseline (N runs) to calibrate the
regression gate's tolerances, floors, and operational ceilings.

Usage: python scripts/measure_baseline.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from rag_eval.config import load_config
from rag_eval.evaluation.harness import default_eval_examples, evaluate_strategy
from rag_eval.evaluation.report import StrategyReport, save
from rag_eval.retrieval.registry import load_resources

logging.basicConfig(level=logging.INFO)

N_RUNS = 5

POINT_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
]
SCALAR_METRICS = ["p95_latency_ms", "cost_per_query_usd", "abstention_accuracy"]


def metric_values(report: StrategyReport, name: str) -> float:
    if name in SCALAR_METRICS:
        return float(getattr(report, name))
    return float(getattr(report, name).point)


def main() -> None:
    cfg = load_config()
    resources = load_resources(cfg)
    examples = default_eval_examples(cfg)

    reports: list[StrategyReport] = []
    for i in range(N_RUNS):
        report = evaluate_strategy("dense", cfg, resources, examples)
        path = save(report, Path("results"))
        print(f"Run {i + 1}/{N_RUNS}: wrote {path}")
        reports.append(report)

    print("\n--- Per-metric mean / std across runs ---")
    stats: dict[str, tuple[float, float]] = {}
    for name in POINT_METRICS + SCALAR_METRICS:
        values = np.array([metric_values(r, name) for r in reports])
        mean = float(values.mean())
        std = float(values.std(ddof=1))
        stats[name] = (mean, std)
        print(f"{name}: mean={mean:.4f} std={std:.4f}")

    print("\n--- Suggested gate values ---")
    print("tolerance:")
    for name in ["faithfulness", "answer_relevancy", "context_recall", "ndcg_at_k"]:
        mean, std = stats[name]
        tolerance = max(2 * std, 0.01)
        print(f"  {name}: {tolerance:.4f}")

    print("floors:")
    for name in ["faithfulness", "context_recall"]:
        mean, std = stats[name]
        floor = round(mean - 2 * std, 2)
        print(f"  {name}: {floor}")

    print("operational_ceilings:")
    p95_mean, _ = stats["p95_latency_ms"]
    cost_mean, _ = stats["cost_per_query_usd"]
    print(f"  p95_latency_ms: {round(p95_mean * 5, 2)}")
    print(f"  cost_per_query_usd: {round(cost_mean * 3, 4)}")


if __name__ == "__main__":
    main()
