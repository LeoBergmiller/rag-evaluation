from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_eval import api
from rag_eval.evaluation.ablation import build_ablation, save_ablation
from rag_eval.evaluation.report import MetricCI, StrategyReport
from rag_eval.generation.generator import GenerationResult
from rag_eval.retrieval.base import RetrievalResult, ScoredChunk

FIXTURE_CHUNKS = [
    ScoredChunk(chunk_id="paper::0", text="some passage", score=0.9, parent_id="paper"),
    ScoredChunk(
        chunk_id="paper::1", text="another passage", score=0.8, parent_id="paper"
    ),
]


class FakeRetriever:
    def __init__(self, name: str) -> None:
        self.name = name

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        return RetrievalResult(
            query=query, chunks=FIXTURE_CHUNKS[:k], latency_ms=12.3, strategy=self.name
        )


class FakeGenerator:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def generate(self, question, chunks):
        return GenerationResult(
            answer="The answer is 42. [paper::0]",
            cited_chunk_ids=["paper::0"],
            abstained=False,
        )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "load_resources", lambda cfg: object())
    monkeypatch.setattr(
        api, "build_retriever", lambda strategy, cfg, resources: FakeRetriever(strategy)
    )
    monkeypatch.setattr(api, "Generator", FakeGenerator)

    with TestClient(api.app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["config_fingerprint"] == "defa37d5071c7049"
    assert set(body["strategies"]) == {"dense", "bm25", "rerank", "hybrid", "hyde"}


def test_query_happy_path(client: TestClient) -> None:
    response = client.post("/query", json={"question": "what is the answer?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The answer is 42. [paper::0]"
    assert body["cited_chunk_ids"] == ["paper::0"]
    assert body["abstained"] is False
    assert body["strategy"] == "dense"
    assert [c["chunk_id"] for c in body["chunks"]] == ["paper::0", "paper::1"]


def test_query_unknown_strategy(client: TestClient) -> None:
    response = client.post(
        "/query", json={"question": "what is the answer?", "strategy": "does-not-exist"}
    )

    assert response.status_code == 400


def test_query_empty_question(client: TestClient) -> None:
    response = client.post("/query", json={"question": ""})

    assert response.status_code == 422


def test_query_invalid_k(client: TestClient) -> None:
    response = client.post("/query", json={"question": "what is the answer?", "k": 0})

    assert response.status_code == 422


def _ci(point: float) -> MetricCI:
    return MetricCI(point=point, lo=point, hi=point)


def _report(strategy: str = "dense") -> StrategyReport:
    return StrategyReport(
        run_id="candidate",
        timestamp="20260101T000000Z",
        strategy=strategy,
        n_examples=59,
        config_fingerprint="defa37d5071c7049",
        eval_set_hash="403ff26e0eff39a0",
        prompt_template_hash="de4b5a5833ff80ce",
        faithfulness=_ci(0.93),
        answer_relevancy=_ci(0.84),
        context_precision=_ci(0.71),
        context_recall=_ci(0.86),
        recall_at_k=_ci(0.75),
        precision_at_k=_ci(0.15),
        mrr=_ci(0.47),
        ndcg_at_k=_ci(0.54),
        p95_latency_ms=37.0,
        cost_per_query_usd=0.0118,
        abstention_accuracy=0.97,
        abstention_rate=0.12,
        examples=[],
    )


def test_latest_ablation_path_returns_none_on_empty_dir(tmp_path: Path) -> None:
    assert api._latest_ablation_path(tmp_path) is None


def test_latest_ablation_path_picks_newest_by_name(tmp_path: Path) -> None:
    older = tmp_path / "20260101T000000Z_ablation_aaaaaaaa.json"
    newer = tmp_path / "20260201T000000Z_ablation_bbbbbbbb.json"
    older.write_text("{}")
    newer.write_text("{}")

    assert api._latest_ablation_path(tmp_path) == newer


def test_ablation_endpoint_returns_report(
    client: TestClient, tmp_path: Path, monkeypatch
) -> None:
    report = build_ablation([_report()], baseline_strategy="dense")
    json_path, _ = save_ablation(report, tmp_path)
    monkeypatch.setattr(api, "_latest_ablation_path", lambda: json_path)

    response = client.get("/ablation")

    assert response.status_code == 200
    body = response.json()
    assert body["baseline_strategy"] == "dense"
    assert [row["strategy"] for row in body["rows"]] == ["dense"]


def test_ablation_endpoint_404_when_missing(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(api, "_latest_ablation_path", lambda: None)

    response = client.get("/ablation")

    assert response.status_code == 404
