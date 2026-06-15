"""Exact FAISS dense index and BM25 sparse index, plus an ingest manifest.

The dense index is deliberately `IndexFlatIP` (exact, brute-force inner product
over L2-normalized vectors == cosine similarity) — NOT IVF/HNSW. Approximate
search introduces its own recall loss, which would contaminate the retrieval
strategy comparison.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_DENSE_INDEX_FILE = "dense.faiss"
_DENSE_IDS_FILE = "dense_chunk_ids.json"
_BM25_FILE = "bm25.pkl"

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class DenseIndex:
    """Exact FAISS IndexFlatIP over L2-normalized embeddings."""

    def __init__(self, index: faiss.IndexFlatIP, chunk_ids: list[str]) -> None:
        self.index = index
        self.chunk_ids = chunk_ids

    @classmethod
    def build(cls, embeddings: np.ndarray, chunk_ids: list[str]) -> "DenseIndex":
        if embeddings.shape[0] != len(chunk_ids):
            raise ValueError("embeddings and chunk_ids must have the same length")

        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index=index, chunk_ids=list(chunk_ids))

    def search(self, query_embedding: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return up to k (chunk_id, score) pairs, best first."""
        vector = np.ascontiguousarray(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vector)

        scores, indices = self.index.search(vector, k)
        results: list[tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0], strict=True):
            if idx == -1:
                continue
            results.append((self.chunk_ids[idx], float(score)))
        return results

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / _DENSE_INDEX_FILE))
        (directory / _DENSE_IDS_FILE).write_text(json.dumps(self.chunk_ids))

    @classmethod
    def load(cls, directory: Path) -> "DenseIndex":
        # faiss.read_index returns the base Index type; we know it's an
        # IndexFlatIP because that's the only index_type this project writes.
        index = faiss.read_index(str(directory / _DENSE_INDEX_FILE))
        chunk_ids = json.loads((directory / _DENSE_IDS_FILE).read_text())
        return cls(index=index, chunk_ids=chunk_ids)  # type: ignore[arg-type]


class BM25Index:
    """BM25 sparse index over chunk text."""

    def __init__(self, bm25: BM25Okapi, chunk_ids: list[str]) -> None:
        self._bm25 = bm25
        self.chunk_ids = chunk_ids

    @classmethod
    def build(cls, texts: list[str], chunk_ids: list[str]) -> "BM25Index":
        if len(texts) != len(chunk_ids):
            raise ValueError("texts and chunk_ids must have the same length")

        tokenized = [_tokenize(text) for text in texts]
        bm25 = BM25Okapi(tokenized)
        return cls(bm25=bm25, chunk_ids=list(chunk_ids))

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return up to k (chunk_id, score) pairs, best first."""
        scores = self._bm25.get_scores(_tokenize(query))
        top_indices = np.argsort(scores)[::-1][:k]
        return [(self.chunk_ids[i], float(scores[i])) for i in top_indices]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / _BM25_FILE).open("wb") as f:
            pickle.dump({"bm25": self._bm25, "chunk_ids": self.chunk_ids}, f)

    @classmethod
    def load(cls, directory: Path) -> "BM25Index":
        with (directory / _BM25_FILE).open("rb") as f:
            data = pickle.load(f)
        return cls(bm25=data["bm25"], chunk_ids=data["chunk_ids"])


class IngestManifest(BaseModel):
    """Metadata tying a built index back to the config that produced it."""

    config_fingerprint: str
    embedding_model: str
    embedding_dimension: int
    n_papers: int
    n_chunks: int
    index_type: str
    built_at: datetime

    @classmethod
    def create(
        cls,
        *,
        config_fingerprint: str,
        embedding_model: str,
        embedding_dimension: int,
        n_papers: int,
        n_chunks: int,
        index_type: str = "flat",
    ) -> "IngestManifest":
        return cls(
            config_fingerprint=config_fingerprint,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            n_papers=n_papers,
            n_chunks=n_chunks,
            index_type=index_type,
            built_at=datetime.now(timezone.utc),
        )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, directory: Path) -> "IngestManifest":
        return cls.model_validate_json((directory / "manifest.json").read_text())
