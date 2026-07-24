"""Download open-access PubMed Central full-text articles into data/raw_med/.

The medical-corpus counterpart to `download.py` (arXiv). Uses the NCBI E-utilities
per-article query APIs (esearch + efetch, db=pmc) rather than the bulk OA packages
(NCBI is sunsetting bulk FTP). For a narrow slice (<=~500 articles) this stays well
under the 10,000-record esearch cap introduced in the Feb 2026 PMC E-utilities update.

Idempotent (an article whose XML already exists is skipped, since data/raw* is
immutable once written). NCBI_API_KEY is read from the environment when present to
raise the rate limit from 3 to 10 requests/second; it is never hardcoded.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from rag_eval.config import CorpusConfig

logger = logging.getLogger(__name__)

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_TIMEOUT = 60.0
# Open-access subset filter — guarantees efetch returns full-text JATS, not just an abstract.
_OA_FILTER = "open access[filter]"

HttpGet = Callable[..., requests.Response]


@dataclass(frozen=True)
class ArticleRef:
    """Reference to a downloaded PMC article (JATS XML on disk)."""

    pmcid: str
    xml_path: Path


def _api_key() -> str | None:
    return os.environ.get("NCBI_API_KEY")


def _rate_limit_delay() -> float:
    """Seconds to wait between requests: ~9/s with an API key, ~3/s without."""
    return 0.11 if _api_key() else 0.34


def _with_key(params: dict[str, Any]) -> dict[str, Any]:
    key = _api_key()
    if key:
        params = {**params, "api_key": key}
    return params


def build_search_term(query: str) -> str:
    """Combine the corpus query with the open-access filter."""
    return f"({query}) AND {_OA_FILTER}"


def search_open_access(
    query: str, max_results: int, *, http_get: HttpGet = requests.get
) -> list[str]:
    """Return up to `max_results` open-access PMC UIDs matching `query`."""
    params = _with_key(
        {
            "db": "pmc",
            "term": build_search_term(query),
            "retmax": max_results,
            "retmode": "json",
        }
    )
    response = http_get(f"{_EUTILS}/esearch.fcgi", params=params, timeout=_TIMEOUT)
    response.raise_for_status()
    idlist: list[str] = response.json()["esearchresult"]["idlist"]
    logger.info("esearch matched %d open-access PMC articles", len(idlist))
    return idlist


def fetch_article_xml(uid: str, *, http_get: HttpGet = requests.get) -> str:
    """Fetch one article's JATS XML full text by PMC UID."""
    params = _with_key({"db": "pmc", "id": uid, "retmode": "xml"})
    response = http_get(f"{_EUTILS}/efetch.fcgi", params=params, timeout=_TIMEOUT)
    response.raise_for_status()
    return response.text


def download_pmc_articles(
    corpus_cfg: CorpusConfig, *, http_get: HttpGet = requests.get
) -> list[ArticleRef]:
    """Download up to `max_papers` open-access PMC articles for the configured query."""
    if not corpus_cfg.query:
        raise ValueError("corpus.query must be set for source='pmc'")

    corpus_cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    uids = search_open_access(
        corpus_cfg.query, corpus_cfg.max_papers, http_get=http_get
    )

    delay = _rate_limit_delay()
    refs: list[ArticleRef] = []
    failures = 0
    for uid in uids:
        pmcid = f"PMC{uid}"
        xml_path = corpus_cfg.raw_dir / f"{pmcid}.xml"

        if not xml_path.exists():
            try:
                xml = fetch_article_xml(uid, http_get=http_get)
            except Exception as exc:
                # One flaky article shouldn't abort a 500-article ingest.
                logger.warning("efetch failed for %s; skipping: %s", pmcid, exc)
                failures += 1
                continue
            xml_path.write_text(xml)
            logger.debug("Fetched %s -> %s", pmcid, xml_path)
            time.sleep(delay)

        refs.append(ArticleRef(pmcid=pmcid, xml_path=xml_path))

    logger.info(
        "Downloaded/verified %d PMC articles in %s (%d fetch failures skipped)",
        len(refs),
        corpus_cfg.raw_dir,
        failures,
    )
    return refs
