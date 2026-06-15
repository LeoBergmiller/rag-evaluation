import dataclasses

import numpy as np
import pytest

from rag_eval.config import load_config
from rag_eval.ingest.chunk import Chunk
from rag_eval.ingest.index import BM25Index, DenseIndex
from rag_eval.retrieval.base import RetrievalResult, Retriever, ScoredChunk
from rag_eval.retrieval.bm25 import BM25Retriever
from rag_eval.retrieval.dense import DenseRetriever
from rag_eval.retrieval.hybrid import HybridRetriever
from rag_eval.retrieval.hyde import HydeRetriever
from rag_eval.retrieval.registry import (
    RetrieverResources,
    build_retriever,
)
from rag_eval.retrieval.rerank import RerankRetriever

FIXTURE_CHUNKS = [
    Chunk(
        chunk_id="paper::0",
        text="transformer attention mechanism scaling laws",
        parent_id="paper",
        metadata={"parent_id": "paper", "chunk_index": 0},
    ),
    Chunk(
        chunk_id="paper::1",
        text="convolutional neural network image classification",
        parent_id="paper",
        metadata={"parent_id": "paper", "chunk_index": 1},
    ),
    Chunk(
        chunk_id="paper::2",
        text="recurrent network long short term memory sequence",
        parent_id="paper",
        metadata={"parent_id": "paper", "chunk_index": 2},
    ),
    Chunk(
        chunk_id="paper::3",
        text="reinforcement learning policy gradient reward",
        parent_id="paper",
        metadata={"parent_id": "paper", "chunk_index": 3},
    ),
]


class FakeEmbedder:
    """Deterministic hashing bag-of-words embedder, no model download."""

    _DIM = 64

    @property
    def dimension(self) -> int:
        return self._DIM

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self._DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                vectors[i, hash(word) % self._DIM] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)


@pytest.fixture
def chunks_by_id() -> dict[str, Chunk]:
    return {c.chunk_id: c for c in FIXTURE_CHUNKS}


@pytest.fixture
def resources(chunks_by_id: dict[str, Chunk]) -> RetrieverResources:
    embedder = FakeEmbedder()
    texts = [c.text for c in FIXTURE_CHUNKS]
    chunk_ids = [c.chunk_id for c in FIXTURE_CHUNKS]

    dense_index = DenseIndex.build(embedder.embed_passages(texts), chunk_ids)
    bm25_index = BM25Index.build(texts, chunk_ids)

    return RetrieverResources(
        embedder=embedder,
        dense_index=dense_index,
        bm25_index=bm25_index,
        chunks_by_id=chunks_by_id,
    )


def test_dense_returns_topk_scored_chunks(resources: RetrieverResources) -> None:
    retriever = DenseRetriever(
        resources.embedder, resources.dense_index, resources.chunks_by_id
    )

    result = retriever.retrieve("transformer attention", k=2)

    assert isinstance(result, RetrievalResult)
    assert result.strategy == "dense"
    assert len(result.chunks) == 2
    assert result.latency_ms >= 0
    assert result.diagnostics["index_type"] == "flat"

    scores = [c.score for c in result.chunks]
    assert scores == sorted(scores, reverse=True)

    top = result.chunks[0]
    assert isinstance(top, ScoredChunk)
    assert top.chunk_id == "paper::0"
    assert top.text == FIXTURE_CHUNKS[0].text
    assert top.parent_id == "paper"


def test_dense_query_retrieves_matching_chunk(resources: RetrieverResources) -> None:
    retriever = DenseRetriever(
        resources.embedder, resources.dense_index, resources.chunks_by_id
    )

    result = retriever.retrieve("reinforcement learning policy reward", k=1)

    assert result.chunks[0].chunk_id == "paper::3"


def test_dense_respects_k(resources: RetrieverResources) -> None:
    retriever = DenseRetriever(
        resources.embedder, resources.dense_index, resources.chunks_by_id
    )

    result = retriever.retrieve("network", k=3)

    assert len(result.chunks) == 3


def test_bm25_retriever(resources: RetrieverResources) -> None:
    retriever = BM25Retriever(resources.bm25_index, resources.chunks_by_id)

    result = retriever.retrieve("convolutional image classification", k=1)

    assert result.strategy == "bm25"
    assert result.chunks[0].chunk_id == "paper::1"


class FakeReranker:
    """Deterministic reranker: scores passages by ascending input order, so the
    final order is the reverse of whatever the base retriever returned."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [float(i) for i in range(len(passages))]


def test_rerank_retriever(resources: RetrieverResources) -> None:
    base = DenseRetriever(
        resources.embedder, resources.dense_index, resources.chunks_by_id
    )
    base_order = [c.chunk_id for c in base.retrieve("network", k=4).chunks]

    retriever = RerankRetriever(base, FakeReranker(), candidate_k=4)
    result = retriever.retrieve("network", k=2)

    assert result.strategy == "rerank"
    assert len(result.chunks) == 2
    assert result.diagnostics["base_strategy"] == "dense"
    assert result.diagnostics["candidate_k"] == 4

    reranked_order = [c.chunk_id for c in result.chunks]
    assert reranked_order == list(reversed(base_order))[:2]


def test_hybrid_retriever(resources: RetrieverResources) -> None:
    dense = DenseRetriever(
        resources.embedder, resources.dense_index, resources.chunks_by_id
    )
    bm25 = BM25Retriever(resources.bm25_index, resources.chunks_by_id)

    retriever = HybridRetriever(dense, bm25, candidate_k=4, rrf_k=60)
    result = retriever.retrieve("network", k=2)

    assert result.strategy == "hybrid"
    assert len(result.chunks) == 2
    assert result.diagnostics["fusion"] == "rrf"
    assert result.diagnostics["candidate_k"] == 4
    assert result.diagnostics["rrf_k"] == 60
    assert result.diagnostics["components"] == ["dense", "bm25"]

    dense_top = {c.chunk_id for c in dense.retrieve("network", k=4).chunks}
    bm25_top = {c.chunk_id for c in bm25.retrieve("network", k=4).chunks}
    assert result.chunks[0].chunk_id in dense_top | bm25_top


class FakeExpander:
    """Deterministic expander: always returns the same hypothetical document."""

    def __init__(self, document: str) -> None:
        self._document = document

    def expand(self, query: str) -> str:
        return self._document


def test_hyde_retriever(resources: RetrieverResources) -> None:
    dense = DenseRetriever(
        resources.embedder, resources.dense_index, resources.chunks_by_id
    )
    hypothetical_document = "reinforcement learning policy gradient reward"

    retriever = HydeRetriever(dense, FakeExpander(hypothetical_document))
    result = retriever.retrieve("totally unrelated query text", k=2)

    expected = dense.retrieve(hypothetical_document, k=2)

    assert result.strategy == "hyde"
    assert [c.chunk_id for c in result.chunks] == [c.chunk_id for c in expected.chunks]
    assert result.diagnostics["base_strategy"] == "dense"
    assert result.diagnostics["hypothetical_document"] == hypothetical_document


def test_registry_builds_base_strategies(resources: RetrieverResources) -> None:
    cfg = load_config()

    for strategy in ("dense", "bm25", "hybrid"):
        retriever = build_retriever(strategy, cfg, resources)
        assert isinstance(retriever, Retriever)
        assert retriever.name == strategy


def test_registry_unknown_strategy_raises(resources: RetrieverResources) -> None:
    cfg = load_config()

    with pytest.raises(ValueError, match="Unknown retrieval strategy"):
        build_retriever("does-not-exist", cfg, resources)


def test_scored_chunk_and_result_frozen() -> None:
    chunk = ScoredChunk(chunk_id="a", text="hello", score=1.0)
    result = RetrievalResult(
        query="q", chunks=[chunk], latency_ms=0.0, strategy="dense"
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.score = 2.0  # type: ignore[misc]

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.strategy = "bm25"  # type: ignore[misc]
