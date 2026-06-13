"""Structured per-strategy evaluation report: schema + IO."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel


class MetricCI(BaseModel):
    point: float
    lo: float
    hi: float


class ExampleRecord(BaseModel):
    id: str
    answerable: bool
    question: str
    answer: str
    abstained: bool
    retrieved_chunk_ids: list[str]
    reference_chunk_ids: list[str] | None
    latency_ms: float
    retrieval_metrics: dict[str, float]
    ragas_metrics: dict[str, float]


class StrategyReport(BaseModel):
    run_id: str
    timestamp: str
    strategy: str
    n_examples: int
    config_fingerprint: str
    eval_set_hash: str
    prompt_template_hash: str

    faithfulness: MetricCI
    answer_relevancy: MetricCI
    context_precision: MetricCI
    context_recall: MetricCI

    recall_at_k: MetricCI
    precision_at_k: MetricCI
    mrr: MetricCI
    ndcg_at_k: MetricCI

    p95_latency_ms: float
    cost_per_query_usd: float
    abstention_accuracy: float
    abstention_rate: float

    examples: list[ExampleRecord]

    @classmethod
    def new_run_id(cls) -> str:
        return uuid.uuid4().hex

    @classmethod
    def now_timestamp(cls) -> str:
        return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def save(report: StrategyReport, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = (
        results_dir / f"{report.timestamp}_{report.strategy}_{report.run_id[:8]}.json"
    )
    path.write_text(report.model_dump_json(indent=2))
    return path
