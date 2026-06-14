"""Strategy-agnostic evaluation harness: retrieval + generation + RAGAS + bootstrap CIs.

The single entry point (`evaluate_strategy`) every retrieval strategy is run through,
producing a `StrategyReport` consumed by the regression gate (step 9) and the ablation
report (step 12).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from langchain_core.callbacks import get_usage_metadata_callback
from ragas.dataset_schema import SingleTurnSample

from rag_eval.config import Config
from rag_eval.evaluation.cost import query_cost_usd
from rag_eval.evaluation.dataset import EvalDataset, EvalExample
from rag_eval.evaluation.judge import run_ragas
from rag_eval.evaluation.metrics import (
    bootstrap_ci,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag_eval.evaluation.report import ExampleRecord, MetricCI, StrategyReport
from rag_eval.generation.generator import Generator
from rag_eval.generation.prompts import prompt_template_hash
from rag_eval.retrieval.registry import RetrieverResources, build_retriever

logger = logging.getLogger(__name__)

JudgeFn = Callable[[list[SingleTurnSample], Config], list[dict[str, float]]]


def default_eval_examples(cfg: Config) -> list[EvalExample]:
    """The frozen test split — what the regression gate evaluates against."""
    return EvalDataset.load(cfg.evaluation.eval_set).test


def evaluate_strategy(
    strategy: str,
    cfg: Config,
    resources: RetrieverResources,
    examples: list[EvalExample],
    *,
    generator: Generator | None = None,
    judge_fn: JudgeFn = run_ragas,
) -> StrategyReport:
    retriever = build_retriever(strategy, cfg, resources)
    generator = generator or Generator(cfg.generation)

    example_records: list[ExampleRecord] = []
    ragas_inputs: list[SingleTurnSample] = []
    ragas_record_indices: list[int] = []

    recall_vals: list[float] = []
    precision_vals: list[float] = []
    mrr_vals: list[float] = []
    ndcg_vals: list[float] = []
    abstention_correct: list[bool] = []
    costs: list[float] = []
    latencies: list[float] = []

    for example in examples:
        result = retriever.retrieve(example.question, cfg.retrieval.top_k)
        with get_usage_metadata_callback() as cb:
            gen_result = generator.generate(example.question, result.chunks)

        latency_ms = result.latency_ms
        latencies.append(latency_ms)

        retrieved_ids = [chunk.chunk_id for chunk in result.chunks]

        retrieval_metrics: dict[str, float] = {}
        if example.answerable and example.reference_chunk_ids:
            relevant = set(example.reference_chunk_ids)
            k = cfg.retrieval.top_k
            r = recall_at_k(retrieved_ids, relevant, k)
            p = precision_at_k(retrieved_ids, relevant, k)
            mrr = reciprocal_rank(retrieved_ids, relevant)
            ndcg = ndcg_at_k(retrieved_ids, relevant, k)
            retrieval_metrics = {
                "recall_at_k": r,
                "precision_at_k": p,
                "mrr": mrr,
                "ndcg_at_k": ndcg,
            }
            recall_vals.append(r)
            precision_vals.append(p)
            mrr_vals.append(mrr)
            ndcg_vals.append(ndcg)

        abstention_correct.append(gen_result.abstained == (not example.answerable))

        for model_name, usage in cb.usage_metadata.items():
            costs.append(
                query_cost_usd(
                    model_name, usage["input_tokens"], usage["output_tokens"]
                )
            )

        record = ExampleRecord(
            id=example.id,
            answerable=example.answerable,
            question=example.question,
            answer=gen_result.answer,
            abstained=gen_result.abstained,
            retrieved_chunk_ids=retrieved_ids,
            reference_chunk_ids=example.reference_chunk_ids,
            latency_ms=latency_ms,
            retrieval_metrics=retrieval_metrics,
            ragas_metrics={},
        )
        example_records.append(record)

        if example.answerable and result.chunks:
            ragas_inputs.append(
                SingleTurnSample(
                    user_input=example.question,
                    retrieved_contexts=[chunk.text for chunk in result.chunks],
                    response=gen_result.answer,
                    reference=example.reference_answer,
                )
            )
            ragas_record_indices.append(len(example_records) - 1)

    ragas_results = judge_fn(ragas_inputs, cfg)
    ragas_vals: dict[str, list[float]] = {
        name: [] for name in cfg.evaluation.ragas_metrics
    }
    for record_index, scores in zip(ragas_record_indices, ragas_results, strict=True):
        example_records[record_index].ragas_metrics = {
            name: (value if np.isfinite(value) else None)
            for name, value in scores.items()
        }
        for name, value in scores.items():
            ragas_vals[name].append(value)

    # RAGAS marks a row NaN if its judge calls fail outright (e.g. exhausted
    # retries); drop those rather than poisoning the aggregate with NaN.
    for name, values in ragas_vals.items():
        finite = [v for v in values if np.isfinite(v)]
        if len(finite) < len(values):
            logger.warning(
                "Dropped %d/%d NaN %s scores from RAGAS judge",
                len(values) - len(finite),
                len(values),
                name,
            )
        ragas_vals[name] = finite

    n_resamples = cfg.evaluation.bootstrap_resamples

    def ci(values: list[float], seed: int) -> MetricCI:
        if not values:
            return MetricCI(point=0.0, lo=0.0, hi=0.0)
        point, lo, hi = bootstrap_ci(values, n_resamples=n_resamples, seed=seed)
        return MetricCI(point=point, lo=lo, hi=hi)

    p95_latency_ms = float(np.percentile(latencies, 95)) if latencies else 0.0
    cost_per_query_usd = float(np.mean(costs)) if costs else 0.0
    abstention_accuracy = (
        float(np.mean(abstention_correct)) if abstention_correct else 0.0
    )
    abstention_rate = (
        sum(1 for example in examples if not example.answerable) / len(examples)
        if examples
        else 0.0
    )

    return StrategyReport(
        run_id=StrategyReport.new_run_id(),
        timestamp=StrategyReport.now_timestamp(),
        strategy=strategy,
        n_examples=len(examples),
        config_fingerprint=cfg.fingerprint(),
        eval_set_hash=EvalDataset.load(cfg.evaluation.eval_set).hash(),
        prompt_template_hash=prompt_template_hash(),
        faithfulness=ci(ragas_vals.get("faithfulness", []), seed=1),
        answer_relevancy=ci(ragas_vals.get("answer_relevancy", []), seed=2),
        context_precision=ci(ragas_vals.get("context_precision", []), seed=3),
        context_recall=ci(ragas_vals.get("context_recall", []), seed=4),
        recall_at_k=ci(recall_vals, seed=5),
        precision_at_k=ci(precision_vals, seed=6),
        mrr=ci(mrr_vals, seed=7),
        ndcg_at_k=ci(ndcg_vals, seed=8),
        p95_latency_ms=p95_latency_ms,
        cost_per_query_usd=cost_per_query_usd,
        abstention_accuracy=abstention_accuracy,
        abstention_rate=abstention_rate,
        examples=example_records,
    )
