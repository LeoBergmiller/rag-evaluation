"""Config-selected, recursion-ready retriever registry.

Composite strategies (rerank, hybrid, HyDE — later steps) register builders that
call `build_retriever` for their base strategy, so the benchmark loop never
branches on strategy type.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from rag_eval.config import Config
from rag_eval.ingest.chunk import Chunk
from rag_eval.ingest.embed import BGEEmbedder, Embedder
from rag_eval.ingest.index import BM25Index, DenseIndex
from rag_eval.ingest.pipeline import load_chunks
from rag_eval.retrieval.base import Retriever
from rag_eval.retrieval.bm25 import BM25Retriever
from rag_eval.retrieval.dense import DenseRetriever


@dataclass
class RetrieverResources:
    """Shared substrate that retriever builders draw from."""

    embedder: Embedder
    dense_index: DenseIndex
    bm25_index: BM25Index
    chunks_by_id: dict[str, Chunk]


RetrieverBuilder = Callable[[Config, RetrieverResources], Retriever]

_REGISTRY: dict[str, RetrieverBuilder] = {}


def register(name: str) -> Callable[[RetrieverBuilder], RetrieverBuilder]:
    def decorator(builder: RetrieverBuilder) -> RetrieverBuilder:
        _REGISTRY[name] = builder
        return builder

    return decorator


def build_retriever(
    strategy: str, cfg: Config, resources: RetrieverResources
) -> Retriever:
    try:
        builder = _REGISTRY[strategy]
    except KeyError:
        raise ValueError(
            f"Unknown retrieval strategy: {strategy!r} (registered: "
            f"{sorted(_REGISTRY)})"
        ) from None
    return builder(cfg, resources)


def load_resources(cfg: Config) -> RetrieverResources:
    """Load the shared substrate (real embedder + indexes) from cfg.corpus.index_dir."""
    index_dir = cfg.corpus.index_dir
    chunks = load_chunks(index_dir)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    return RetrieverResources(
        embedder=BGEEmbedder(cfg.embedding),
        dense_index=DenseIndex.load(index_dir),
        bm25_index=BM25Index.load(index_dir),
        chunks_by_id=chunks_by_id,
    )


@register("dense")
def _build_dense(cfg: Config, resources: RetrieverResources) -> Retriever:
    return DenseRetriever(
        resources.embedder, resources.dense_index, resources.chunks_by_id
    )


@register("bm25")
def _build_bm25(cfg: Config, resources: RetrieverResources) -> Retriever:
    return BM25Retriever(resources.bm25_index, resources.chunks_by_id)
