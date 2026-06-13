"""End-to-end ingest: download -> parse -> chunk -> embed -> index."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from transformers import AutoTokenizer

from rag_eval.config import Config
from rag_eval.ingest.chunk import Chunk, chunk_text
from rag_eval.ingest.download import download_papers
from rag_eval.ingest.embed import BGEEmbedder, Embedder
from rag_eval.ingest.index import BM25Index, DenseIndex, IngestManifest
from rag_eval.ingest.parse import extract_text

logger = logging.getLogger(__name__)

_CHUNKS_FILE = "chunks.jsonl"


def run_ingest(cfg: Config, *, embedder: Embedder | None = None) -> IngestManifest:
    """Run the full ingest pipeline and persist indexes to `cfg.corpus.index_dir`.

    `embedder` can be injected (e.g. in tests) to avoid loading the real bge model.
    """
    papers = download_papers(cfg.corpus)

    tokenizer = AutoTokenizer.from_pretrained(cfg.embedding.model)

    def length_function(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    chunks: list[Chunk] = []
    for paper in papers:
        text = extract_text(
            paper.pdf_path, strip_references=cfg.corpus.strip_references
        )
        chunks.extend(
            chunk_text(
                text,
                paper.arxiv_id,
                length_function=length_function,
                chunk_size=cfg.chunking.chunk_size,
                chunk_overlap=cfg.chunking.chunk_overlap,
            )
        )

    if embedder is None:
        embedder = BGEEmbedder(cfg.embedding)

    chunk_ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]
    embeddings = embedder.embed_passages(texts)

    dense_index = DenseIndex.build(embeddings, chunk_ids)
    bm25_index = BM25Index.build(texts, chunk_ids)

    index_dir = cfg.corpus.index_dir
    dense_index.save(index_dir)
    bm25_index.save(index_dir)
    _save_chunks(chunks, index_dir)

    manifest = IngestManifest.create(
        config_fingerprint=cfg.fingerprint(),
        embedding_model=cfg.embedding.model,
        embedding_dimension=embedder.dimension,
        n_papers=len(papers),
        n_chunks=len(chunks),
        index_type=cfg.retrieval.index_type,
    )
    manifest.save(index_dir)

    logger.info(
        "Ingest complete: %d papers, %d chunks -> %s",
        len(papers),
        len(chunks),
        index_dir,
    )
    return manifest


def _save_chunks(chunks: list[Chunk], index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / _CHUNKS_FILE).open("w") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")


def load_chunks(index_dir: Path) -> list[Chunk]:
    chunks = []
    with (index_dir / _CHUNKS_FILE).open("r") as f:
        for line in f:
            chunks.append(Chunk.model_validate(json.loads(line)))
    return chunks
