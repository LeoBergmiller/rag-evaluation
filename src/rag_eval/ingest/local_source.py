"""Local authored corpus: Markdown / plain-text documents already on disk.

The third corpus source alongside arXiv (PDF) and PMC (JATS XML). There is nothing
to fetch — the documents *are* the corpus — so this adapter only reads and orders
them, and everything downstream (chunk -> embed -> index -> retrieve) is unchanged.
That is the same config + ingest-adapter shape D12 established for PMC.

`doc_id` is the filename stem, which makes document ids stable, readable, and
citable by a downstream consumer rather than an opaque accession number.

Every failure mode here raises. A local corpus is the one source where the
plausible mistakes — an empty directory, a typo'd path, two files sharing a stem —
all produce a *working* index that silently retrieves nothing or drops a document.
"""

from __future__ import annotations

import logging
from collections import Counter

from rag_eval.config import CorpusConfig

logger = logging.getLogger(__name__)

#: Authored text formats. Deliberately not PDF/XML — those have their own adapters.
SUFFIXES = (".md", ".txt")


def load_local_documents(corpus_cfg: CorpusConfig) -> list[tuple[str, str]]:
    """(doc_id, text) for every document under `raw_dir`, ordered by path.

    Searched recursively, so a corpus may be organised into subdirectories; because
    `doc_id` is the stem alone, that flattening is checked for collisions.
    """
    raw_dir = corpus_cfg.raw_dir
    if not raw_dir.is_dir():
        raise ValueError(
            f"corpus.raw_dir does not exist: {raw_dir}. A local corpus is read from "
            f"disk rather than downloaded — create the directory and add "
            f"{' / '.join(SUFFIXES)} files."
        )

    paths = sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in SUFFIXES)
    if not paths:
        raise ValueError(
            f"No {' / '.join(SUFFIXES)} documents found under {raw_dir}. Ingesting an "
            "empty corpus would build a valid index that retrieves nothing."
        )

    duplicates = sorted(
        stem for stem, n in Counter(p.stem for p in paths).items() if n > 1
    )
    if duplicates:
        raise ValueError(
            f"Duplicate document ids under {raw_dir}: {duplicates}. doc_id is the "
            "filename stem, so two files sharing one (in different subdirectories, or "
            "as .md and .txt) would collide on chunk_id and silently drop a document."
        )

    documents = [(path.stem, path.read_text(encoding="utf-8")) for path in paths]

    empty = [doc_id for doc_id, text in documents if not text.strip()]
    if empty:
        raise ValueError(
            f"Empty documents under {raw_dir}: {empty}. An empty document produces "
            "zero chunks and would silently shrink the corpus."
        )

    logger.info("Loaded %d local documents from %s", len(documents), raw_dir)
    return documents
