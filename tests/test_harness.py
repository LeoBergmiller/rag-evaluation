from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from ragas.dataset_schema import SingleTurnSample

from rag_eval.config import Config, load_config
from rag_eval.evaluation.dataset import EvalExample
from rag_eval.evaluation.harness import evaluate_strategy
from rag_eval.evaluation.report import StrategyReport, save
from rag_eval.evaluation.rubric import RubricScore
from rag_eval.generation.generator import Generator
from tests.test_retrieval import FIXTURE_CHUNKS, FakeEmbedder

from rag_eval.ingest.index import BM25Index, DenseIndex
from rag_eval.retrieval.registry import RetrieverResources

EXAMPLES = [
    EvalExample(
        id="answerable::0",
        question="transformer attention mechanism scaling laws",
        reference_answer="The paper discusses transformer attention scaling.",
        reference_chunk_ids=["paper::0"],
        answerable=True,
        split="test",
    ),
    EvalExample(
        id="answerable::1",
        question="reinforcement learning policy gradient reward",
        reference_answer="The paper discusses policy gradient methods.",
        reference_chunk_ids=["paper::3"],
        answerable=True,
        split="test",
    ),
    EvalExample(
        id="unanswerable::0",
        question="What is the capital of France?",
        reference_answer=None,
        reference_chunk_ids=None,
        answerable=False,
        split="test",
    ),
]


def _stub_judge(samples: list[SingleTurnSample], cfg: Config) -> list[dict[str, float]]:
    return [{name: 0.8 for name in cfg.evaluation.ragas_metrics} for _ in samples]


def _stub_rubric(**kwargs: object) -> RubricScore | None:
    if kwargs.get("reference_answer") is None:
        return None
    return RubricScore(correctness=2, completeness=2, citation_valid=True, rationale="stub")


@pytest.fixture
def resources() -> RetrieverResources:
    embedder = FakeEmbedder()
    texts = [c.text for c in FIXTURE_CHUNKS]
    chunk_ids = [c.chunk_id for c in FIXTURE_CHUNKS]
    chunks_by_id = {c.chunk_id: c for c in FIXTURE_CHUNKS}

    dense_index = DenseIndex.build(embedder.embed_passages(texts), chunk_ids)
    bm25_index = BM25Index.build(texts, chunk_ids)

    return RetrieverResources(
        embedder=embedder,
        dense_index=dense_index,
        bm25_index=bm25_index,
        chunks_by_id=chunks_by_id,
    )


def test_evaluate_strategy_report_shape(resources: RetrieverResources) -> None:
    cfg = load_config()
    fake_model = FakeListChatModel(
        responses=[
            "Attention scales [paper::0].",
            "Policy gradients update parameters [paper::3].",
            "I don't know.",
        ]
    )
    generator = Generator(cfg.generation, chat_model=fake_model)

    report = evaluate_strategy(
        "dense",
        cfg,
        resources,
        EXAMPLES,
        generator=generator,
        judge_fn=_stub_judge,
        rubric_fn=_stub_rubric,
    )

    assert isinstance(report, StrategyReport)
    assert report.strategy == "dense"
    assert report.n_examples == 3
    assert report.config_fingerprint
    assert report.eval_set_hash
    assert report.prompt_template_hash

    for metric_ci in (
        report.faithfulness,
        report.answer_relevancy,
        report.context_precision,
        report.context_recall,
        report.recall_at_k,
        report.precision_at_k,
        report.mrr,
        report.ndcg_at_k,
    ):
        assert metric_ci.lo <= metric_ci.point <= metric_ci.hi

    assert report.p95_latency_ms >= 0
    assert report.abstention_accuracy == 1.0
    assert report.abstention_rate == pytest.approx(1 / 3)

    assert report.correctness is not None
    assert report.completeness is not None
    assert report.citation_valid_rate is not None
    for rubric_ci in (report.correctness, report.completeness, report.citation_valid_rate):
        assert rubric_ci.lo <= rubric_ci.point <= rubric_ci.hi


def test_evaluate_strategy_report_roundtrip(
    resources: RetrieverResources, tmp_path: Path
) -> None:
    cfg = load_config()
    fake_model = FakeListChatModel(
        responses=[
            "Attention scales [paper::0].",
            "Policy gradients update parameters [paper::3].",
            "I don't know.",
        ]
    )
    generator = Generator(cfg.generation, chat_model=fake_model)

    report = evaluate_strategy(
        "dense",
        cfg,
        resources,
        EXAMPLES,
        generator=generator,
        judge_fn=_stub_judge,
        rubric_fn=_stub_rubric,
    )

    path = save(report, tmp_path)
    reloaded = StrategyReport.model_validate_json(path.read_text())

    assert reloaded == report
