"""RAGAS judge wiring: a different model family from the generator (anthropic vs openai)
to avoid self-preference bias.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.llms.base import BaseRagasLLM
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.metrics.base import Metric
from ragas.run_config import RunConfig

from rag_eval.config import Config

# Low concurrency + generous backoff: avoids OpenAI 429s during a full eval run
# silently turning into NaN metric scores (ragas marks a row NaN if its judge
# calls exhaust retries).
_JUDGE_RUN_CONFIG = RunConfig(max_workers=2, max_wait=120, max_retries=15)

RAGAS_METRIC_REGISTRY: dict[str, Metric] = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}


def build_judge(cfg: Config) -> tuple[BaseRagasLLM, BaseRagasEmbeddings]:
    """Build the judge LLM + embeddings used by RAGAS metrics."""
    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=cfg.evaluation.judge_model,
            temperature=cfg.evaluation.judge_temperature,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    return llm, embeddings


def run_ragas(samples: list[SingleTurnSample], cfg: Config) -> list[dict[str, float]]:
    """Run the configured RAGAS metrics over `samples`, one result dict per sample."""
    if not samples:
        return []

    metrics = [RAGAS_METRIC_REGISTRY[name] for name in cfg.evaluation.ragas_metrics]
    llm, embeddings = build_judge(cfg)

    dataset = EvaluationDataset(samples)
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        show_progress=False,
        run_config=_JUDGE_RUN_CONFIG,
    )

    df = result.to_pandas()
    metric_names = list(cfg.evaluation.ragas_metrics)
    return [
        {name: float(row[name]) for name in metric_names} for _, row in df.iterrows()
    ]
