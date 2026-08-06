"""End-to-end ingest: download -> parse -> chunk -> embed -> index."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from transformers import AutoTokenizer

from rag_eval.config import Config
from rag_eval.ingest.chunk import Chunk, chunk_text
from rag_eval.ingest.embed import BGEEmbedder, Embedder
from rag_eval.ingest.index import BM25Index, DenseIndex, IngestManifest

logger = logging.getLogger(__name__)

_CHUNKS_FILE = "chunks.jsonl"


def _load_source_documents(cfg: Config) -> list[tuple[str, str]]:
    """(doc_id, cleaned_text) per source document, dispatched on the corpus source.

    All sources feed the SAME downstream chunk -> embed -> index pipeline; only the
    fetch + parse differ. arXiv yields PDF text keyed by arXiv id; PMC yields JATS
    full-text keyed by PMCID; `local` reads authored Markdown/text already on disk,
    keyed by filename stem.

    The per-source fetchers are imported INSIDE their branch rather than at module
    scope. arXiv needs `arxiv` + `pypdf` and PMC needs `requests` — all `[full]`
    extras — so a top-level import would make a base install unable to run ingest for
    the one source that needs no downloader at all (D13).
    """
    if cfg.corpus.source == "local":
        from rag_eval.ingest.local_source import load_local_documents

        return load_local_documents(cfg.corpus)

    if cfg.corpus.source == "pmc":
        from rag_eval.ingest.jats_parse import extract_jats_text
        from rag_eval.ingest.pmc_download import download_pmc_articles

        articles = download_pmc_articles(cfg.corpus)
        return [
            (
                article.pmcid,
                extract_jats_text(
                    article.xml_path, strip_references=cfg.corpus.strip_references
                ),
            )
            for article in articles
        ]

    from rag_eval.ingest.download import download_papers
    from rag_eval.ingest.parse import extract_text

    papers = download_papers(cfg.corpus)
    return [
        (
            paper.arxiv_id,
            extract_text(paper.pdf_path, strip_references=cfg.corpus.strip_references),
        )
        for paper in papers
    ]


def run_ingest(cfg: Config, *, embedder: Embedder | None = None) -> IngestManifest:
    """Run the full ingest pipeline and persist indexes to `cfg.corpus.index_dir`.

    `embedder` can be injected (e.g. in tests) to avoid loading the real bge model.
    """
    documents = _load_source_documents(cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg.embedding.model)

    def length_function(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    chunks: list[Chunk] = []
    for doc_id, text in documents:
        chunks.extend(
            chunk_text(
                text,
                doc_id,
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
        n_papers=len(documents),
        n_chunks=len(chunks),
        index_type=cfg.retrieval.index_type,
    )
    manifest.save(index_dir)

    logger.info(
        "Ingest complete: %d documents, %d chunks -> %s",
        len(documents),
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
