"""One-off: run the dense baseline over the eval set's test split and save a report.

Usage: python scripts/run_baseline_dense.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from rag_eval.config import load_config
from rag_eval.evaluation.harness import default_eval_examples, evaluate_strategy
from rag_eval.evaluation.report import save
from rag_eval.retrieval.registry import load_resources

logging.basicConfig(level=logging.INFO)

cfg = load_config()
resources = load_resources(cfg)
examples = default_eval_examples(cfg)

report = evaluate_strategy("dense", cfg, resources, examples)

path = save(report, Path("results"))
print(f"Wrote {path}")
print(f"n_examples: {report.n_examples}")
print(f"faithfulness: {report.faithfulness}")
print(f"answer_relevancy: {report.answer_relevancy}")
print(f"context_precision: {report.context_precision}")
print(f"context_recall: {report.context_recall}")
print(f"recall_at_k: {report.recall_at_k}")
print(f"precision_at_k: {report.precision_at_k}")
print(f"mrr: {report.mrr}")
print(f"ndcg_at_k: {report.ndcg_at_k}")
print(f"p95_latency_ms: {report.p95_latency_ms}")
print(f"cost_per_query_usd: {report.cost_per_query_usd}")
print(f"abstention_accuracy: {report.abstention_accuracy}")
print(f"abstention_rate: {report.abstention_rate}")
