"""Cross-encoder reranking: wraps a base retriever, never reimplements retrieval."""

from __future__ import annotations

import time
from typing import Protocol

from sentence_transformers import CrossEncoder

from rag_eval.retrieval.base import RetrievalResult, Retriever, ScoredChunk


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Cross-encoder relevance scorer (e.g. bge-reranker-base)."""

    def __init__(self, model_name: str) -> None:
        self._model = CrossEncoder(model_name)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, passage) for passage in passages]
        return self._model.predict(pairs).tolist()


class RerankRetriever:
    """Retrieves `candidate_k` from a base retriever, cross-encodes, returns top-k."""

    name = "rerank"

    def __init__(self, base: Retriever, reranker: Reranker, candidate_k: int) -> None:
        self._base = base
        self._reranker = reranker
        self._candidate_k = candidate_k

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        start = time.perf_counter()

        candidates = self._base.retrieve(query, self._candidate_k).chunks
        scores = self._reranker.score(query, [c.text for c in candidates])

        reranked = [
            ScoredChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=score,
                parent_id=chunk.parent_id,
                metadata=chunk.metadata,
            )
            for chunk, score in zip(candidates, scores, strict=True)
        ]
        reranked.sort(key=lambda c: c.score, reverse=True)

        latency_ms = (time.perf_counter() - start) * 1000
        return RetrievalResult(
            query=query,
            chunks=reranked[:k],
            latency_ms=latency_ms,
            strategy=self.name,
            diagnostics={
                "base_strategy": self._base.name,
                "candidate_k": self._candidate_k,
            },
        )
