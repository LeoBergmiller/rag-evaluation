import pytest
from fastapi.testclient import TestClient

from rag_eval import api
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
