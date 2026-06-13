"""The apples-to-apples retrieval contract.

Every retrieval strategy implements `Retriever`. Composite strategies (rerank,
hybrid, HyDE) WRAP a base retriever rather than reimplementing retrieval, so the
benchmark loop never branches on strategy type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from rag_eval.ingest.chunk import Chunk


@dataclass(frozen=True)
class ScoredChunk:
    chunk_id: str
    text: str
    score: float
    parent_id: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    chunks: list[ScoredChunk]  # ordered, best first
    latency_ms: float
    strategy: str
    diagnostics: dict = field(default_factory=dict)


@runtime_checkable
class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, k: int) -> RetrievalResult: ...


def to_scored_chunks(
    pairs: list[tuple[str, float]], chunks_by_id: dict[str, Chunk]
) -> list[ScoredChunk]:
    """Map (chunk_id, score) pairs from an index search to ScoredChunks."""
    scored = []
    for chunk_id, score in pairs:
        chunk = chunks_by_id[chunk_id]
        scored.append(
            ScoredChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=score,
                parent_id=chunk.parent_id,
                metadata=chunk.metadata,
            )
        )
    return scored
