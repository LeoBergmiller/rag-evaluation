"""Dense retrieval over the exact FAISS index."""

from __future__ import annotations

import time

from rag_eval.ingest.chunk import Chunk
from rag_eval.ingest.embed import Embedder
from rag_eval.ingest.index import DenseIndex
from rag_eval.retrieval.base import RetrievalResult, to_scored_chunks


class DenseRetriever:
    """Embeds the query (bge query prefix applied by the embedder) and searches
    the exact FAISS IndexFlatIP."""

    name = "dense"

    def __init__(
        self,
        embedder: Embedder,
        index: DenseIndex,
        chunks_by_id: dict[str, Chunk],
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._chunks_by_id = chunks_by_id

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        start = time.perf_counter()

        query_vector = self._embedder.embed_queries([query])[0]
        pairs = self._index.search(query_vector, k)
        chunks = to_scored_chunks(pairs, self._chunks_by_id)

        latency_ms = (time.perf_counter() - start) * 1000
        return RetrievalResult(
            query=query,
            chunks=chunks,
            latency_ms=latency_ms,
            strategy=self.name,
            diagnostics={"index_type": "flat"},
        )
