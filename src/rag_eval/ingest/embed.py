"""bge embedder: owns the query-prefix / passage asymmetry.

The bge family expects an instruction prefix on QUERIES ONLY
("Represent this sentence for searching relevant passages: "). Passages are
embedded with no prefix. Getting this backwards silently tanks recall, so this
module is the single place the prefix is applied and is unit-tested for it.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_eval.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    def embed_queries(self, texts: list[str]) -> np.ndarray: ...

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...

    @property
    def dimension(self) -> int: ...


class BGEEmbedder:
    """SentenceTransformer-backed embedder for the bge model family."""

    def __init__(self, embedding_cfg: EmbeddingConfig) -> None:
        self._cfg = embedding_cfg
        self._model = SentenceTransformer(
            embedding_cfg.model, device=embedding_cfg.device
        )

    @property
    def dimension(self) -> int:
        return self._model.get_embedding_dimension()

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """Embed queries WITH the bge query-instruction prefix."""
        return self._model.encode(
            texts,
            prompt=self._cfg.query_prefix,
            normalize_embeddings=True,
        )

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Embed passages with NO prefix."""
        return self._model.encode(
            texts,
            normalize_embeddings=True,
        )
