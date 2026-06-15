"""Hybrid retrieval: fuse dense + BM25 rankings via Reciprocal Rank Fusion."""

from __future__ import annotations

import time

from rag_eval.retrieval.base import RetrievalResult, Retriever, ScoredChunk


class HybridRetriever:
    """Fuses a dense and a BM25 retriever's rankings with RRF."""

    name = "hybrid"

    def __init__(
        self, dense: Retriever, bm25: Retriever, candidate_k: int, rrf_k: int
    ) -> None:
        self._dense = dense
        self._bm25 = bm25
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        start = time.perf_counter()

        dense_chunks = self._dense.retrieve(query, self._candidate_k).chunks
        bm25_chunks = self._bm25.retrieve(query, self._candidate_k).chunks

        fused_scores: dict[str, float] = {}
        chunks_by_id: dict[str, ScoredChunk] = {}
        for ranked in (dense_chunks, bm25_chunks):
            for rank, chunk in enumerate(ranked, start=1):
                fused_scores[chunk.chunk_id] = fused_scores.get(
                    chunk.chunk_id, 0.0
                ) + 1.0 / (self._rrf_k + rank)
                chunks_by_id.setdefault(chunk.chunk_id, chunk)

        fused = [
            ScoredChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=fused_scores[chunk.chunk_id],
                parent_id=chunk.parent_id,
                metadata=chunk.metadata,
            )
            for chunk in chunks_by_id.values()
        ]
        fused.sort(key=lambda c: c.score, reverse=True)

        latency_ms = (time.perf_counter() - start) * 1000
        return RetrievalResult(
            query=query,
            chunks=fused[:k],
            latency_ms=latency_ms,
            strategy=self.name,
            diagnostics={
                "fusion": "rrf",
                "candidate_k": self._candidate_k,
                "rrf_k": self._rrf_k,
                "components": ["dense", "bm25"],
            },
        )
