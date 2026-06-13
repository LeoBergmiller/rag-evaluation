import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from rag_eval.config import load_config
from rag_eval.ingest.chunk import chunk_text
from rag_eval.ingest.embed import BGEEmbedder
from rag_eval.ingest.index import BM25Index, DenseIndex, IngestManifest
from rag_eval.ingest.parse import _strip_references


def test_parse_strips_references() -> None:
    text = (
        "Introduction\nThis paper studies things.\n\n"
        "References\n[1] Author, Title, 2020.\n[2] Other, Paper, 2021."
    )

    stripped = _strip_references(text)

    assert "References" not in stripped
    assert "[1] Author" not in stripped
    assert "This paper studies things." in stripped


def test_parse_no_references_returns_text_unchanged() -> None:
    text = "Introduction\nNo references section here."

    assert _strip_references(text) == text


def test_chunker_count_ids_and_parent() -> None:
    text = " ".join(f"word{i}" for i in range(2000))

    chunks = chunk_text(
        text,
        paper_id="2401.00001",
        length_function=lambda s: len(s.split()),
        chunk_size=512,
        chunk_overlap=64,
    )

    assert len(chunks) > 1

    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))

    for chunk in chunks:
        assert chunk.parent_id == "2401.00001"
        assert chunk.metadata["parent_id"] == "2401.00001"
        assert len(chunk.text.split()) <= 512


def test_query_prefix_applied_to_queries_only(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config()

    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros((1, 768), dtype=np.float32)
    monkeypatch.setattr(
        "rag_eval.ingest.embed.SentenceTransformer", lambda *a, **k: mock_model
    )

    embedder = BGEEmbedder(cfg.embedding)

    embedder.embed_queries(["what is attention?"])
    query_kwargs = mock_model.encode.call_args.kwargs
    assert query_kwargs["prompt"] == cfg.embedding.query_prefix
    assert query_kwargs["normalize_embeddings"] is True

    mock_model.encode.reset_mock()
    embedder.embed_passages(["attention is all you need"])
    passage_kwargs = mock_model.encode.call_args.kwargs
    assert "prompt" not in passage_kwargs
    assert passage_kwargs["normalize_embeddings"] is True


def test_dense_index_is_exact_flat() -> None:
    embeddings = np.random.default_rng(0).random((10, 16)).astype(np.float32)
    chunk_ids = [f"paper::{i}" for i in range(10)]

    index = DenseIndex.build(embeddings, chunk_ids)

    import faiss

    assert isinstance(index.index, faiss.IndexFlatIP)
    assert index.index.ntotal == len(chunk_ids)


def test_dense_roundtrip_and_persist(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    embeddings = rng.random((5, 8)).astype(np.float32)
    chunk_ids = [f"paper::{i}" for i in range(5)]

    index = DenseIndex.build(embeddings, chunk_ids)

    # querying with chunk 2's own (normalized) embedding should rank it first
    query_vec = embeddings[2]
    results = index.search(query_vec, k=1)
    assert results[0][0] == "paper::2"

    index.save(tmp_path)
    loaded = DenseIndex.load(tmp_path)
    loaded_results = loaded.search(query_vec, k=1)
    assert loaded_results == results


def test_bm25_search() -> None:
    texts = [
        "attention is all you need transformer architecture",
        "convolutional neural networks for image classification",
        "recurrent networks and long short term memory",
    ]
    chunk_ids = ["a", "b", "c"]

    index = BM25Index.build(texts, chunk_ids)
    results = index.search("transformer attention", k=2)

    assert results[0][0] == "a"
    assert len(results) == 2


def test_bm25_persist(tmp_path: Path) -> None:
    texts = ["alpha beta gamma", "delta epsilon zeta"]
    chunk_ids = ["x", "y"]

    index = BM25Index.build(texts, chunk_ids)
    index.save(tmp_path)
    loaded = BM25Index.load(tmp_path)

    assert loaded.search("alpha", k=1) == index.search("alpha", k=1)


def test_manifest_roundtrip(tmp_path: Path) -> None:
    manifest = IngestManifest.create(
        config_fingerprint="abc123",
        embedding_model="BAAI/bge-base-en-v1.5",
        embedding_dimension=768,
        n_papers=5,
        n_chunks=42,
    )

    manifest.save(tmp_path)
    loaded = IngestManifest.load(tmp_path)

    assert loaded.config_fingerprint == "abc123"
    assert loaded.n_chunks == 42
    assert loaded.index_type == "flat"


@pytest.mark.skipif(
    os.environ.get("RAG_EVAL_RUN_INGEST") != "1",
    reason="set RAG_EVAL_RUN_INGEST=1 to run the real networked ingest",
)
def test_ingest_5_papers(tmp_path: Path) -> None:
    import dataclasses

    from rag_eval.ingest.pipeline import run_ingest

    cfg = load_config()
    cfg = dataclasses.replace(
        cfg,
        corpus=dataclasses.replace(
            cfg.corpus,
            max_papers=5,
            raw_dir=tmp_path / "raw",
            index_dir=tmp_path / "index",
        ),
    )

    manifest = run_ingest(cfg)

    dense_index = DenseIndex.load(cfg.corpus.index_dir)
    bm25_index = BM25Index.load(cfg.corpus.index_dir)

    assert dense_index.index.ntotal == manifest.n_chunks
    assert len(bm25_index.chunk_ids) == manifest.n_chunks
