"""Download arXiv papers (full text PDFs) into the immutable data/raw/ directory."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import arxiv
import requests

from rag_eval.config import CorpusConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PaperRef:
    """Reference to a downloaded paper."""

    arxiv_id: str
    title: str
    pdf_path: Path


def download_papers(corpus_cfg: CorpusConfig) -> list[PaperRef]:
    """Download up to `max_papers` full-text PDFs for the configured categories.

    Idempotent: a paper whose PDF already exists in raw_dir is skipped, since
    data/raw is immutable once written.
    """
    corpus_cfg.raw_dir.mkdir(parents=True, exist_ok=True)

    category_query = " OR ".join(f"cat:{cat}" for cat in corpus_cfg.categories)
    search = arxiv.Search(
        query=category_query,
        max_results=corpus_cfg.max_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    client = arxiv.Client()
    refs: list[PaperRef] = []
    for result in client.results(search):
        arxiv_id = result.get_short_id()
        pdf_path = corpus_cfg.raw_dir / f"{arxiv_id}.pdf"

        if not pdf_path.exists():
            _download_pdf(result.pdf_url, pdf_path)

        refs.append(PaperRef(arxiv_id=arxiv_id, title=result.title, pdf_path=pdf_path))

    logger.info("Downloaded/verified %d papers in %s", len(refs), corpus_cfg.raw_dir)
    return refs


def _download_pdf(pdf_url: str | None, dest: Path) -> None:
    if pdf_url is None:
        raise ValueError(f"No PDF URL available for {dest.stem}")

    response = requests.get(pdf_url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    logger.debug("Downloaded %s -> %s", pdf_url, dest)
