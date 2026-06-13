"""Sparse retrieval over the BM25 index."""

from __future__ import annotations

import time

from rag_eval.ingest.chunk import Chunk
from rag_eval.ingest.index import BM25Index
from rag_eval.retrieval.base import RetrievalResult, to_scored_chunks


class BM25Retriever:
    """Lexical retrieval via BM25Okapi."""

    name = "bm25"

    def __init__(self, index: BM25Index, chunks_by_id: dict[str, Chunk]) -> None:
        self._index = index
        self._chunks_by_id = chunks_by_id

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        start = time.perf_counter()

        pairs = self._index.search(query, k)
        chunks = to_scored_chunks(pairs, self._chunks_by_id)

        latency_ms = (time.perf_counter() - start) * 1000
        return RetrievalResult(
            query=query,
            chunks=chunks,
            latency_ms=latency_ms,
            strategy=self.name,
            diagnostics={},
        )
