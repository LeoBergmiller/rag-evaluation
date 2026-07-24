from pathlib import Path

import pytest

from rag_eval.config import CorpusConfig
from rag_eval.ingest import pmc_download
from rag_eval.ingest.jats_parse import parse_jats
from rag_eval.ingest.pmc_download import (
    build_search_term,
    download_pmc_articles,
    search_open_access,
)

_JATS = """<article>
  <front><article-meta>
    <title-group><article-title>Checkpoint Inhibitors in Melanoma</article-title></title-group>
    <abstract><p>We review PD-1 blockade.</p></abstract>
  </article-meta></front>
  <body>
    <sec><title>Introduction</title><p>Nivolumab targets PD-1.</p></sec>
    <sec><title>Methods</title><p>We analyzed patient cohorts.</p></sec>
  </body>
  <back><ref-list><ref><mixed-citation>Smith 2020 UNIQUEREFTOKEN</mixed-citation></ref></ref-list></back>
</article>"""

_JATS_NAMESPACED = """<article xmlns="https://jats.nlm.nih.gov/ns">
  <front><article-meta>
    <title-group><article-title>Namespaced Title</article-title></title-group>
  </article-meta></front>
  <body><sec><p>Body content here.</p></sec></body>
</article>"""


def test_parse_jats_extracts_title_abstract_body_and_strips_references() -> None:
    text = parse_jats(_JATS, strip_references=True)

    assert "Checkpoint Inhibitors in Melanoma" in text
    assert "We review PD-1 blockade." in text
    assert "Nivolumab targets PD-1." in text
    assert "We analyzed patient cohorts." in text
    assert "UNIQUEREFTOKEN" not in text  # reference list dropped


def test_parse_jats_includes_back_matter_when_not_stripping() -> None:
    text = parse_jats(_JATS, strip_references=False)

    assert "UNIQUEREFTOKEN" in text


def test_parse_jats_is_namespace_robust() -> None:
    text = parse_jats(_JATS_NAMESPACED)

    assert "Namespaced Title" in text
    assert "Body content here." in text


def test_parse_jats_bad_xml_returns_empty_string() -> None:
    assert parse_jats("<article><body><p>unclosed") == ""


def test_build_search_term_adds_open_access_filter() -> None:
    term = build_search_term("immune checkpoint inhibitors")

    assert "immune checkpoint inhibitors" in term
    assert "open access[filter]" in term


class _FakeResponse:
    def __init__(self, *, json_data: dict | None = None, text: str = "") -> None:
        self._json = json_data
        self.text = text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        assert self._json is not None
        return self._json


def _make_http_get(idlist: list[str], xml_by_uid: dict[str, str]):
    calls: list[tuple[str, dict]] = []

    def http_get(url: str, params: dict, timeout: float) -> _FakeResponse:
        calls.append((url, params))
        if url.endswith("esearch.fcgi"):
            return _FakeResponse(json_data={"esearchresult": {"idlist": idlist}})
        if url.endswith("efetch.fcgi"):
            return _FakeResponse(text=xml_by_uid[params["id"]])
        raise AssertionError(f"unexpected url {url}")

    http_get.calls = calls  # type: ignore[attr-defined]
    return http_get


def test_search_open_access_parses_idlist_and_passes_params() -> None:
    http_get = _make_http_get(["111", "222"], {})

    uids = search_open_access("checkpoint", max_results=10, http_get=http_get)

    assert uids == ["111", "222"]
    url, params = http_get.calls[0]  # type: ignore[attr-defined]
    assert url.endswith("esearch.fcgi")
    assert params["db"] == "pmc"
    assert params["retmax"] == 10
    assert "open access[filter]" in params["term"]


def test_download_pmc_articles_writes_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pmc_download.time, "sleep", lambda _s: None)
    cfg = CorpusConfig(
        source="pmc",
        categories=(),
        max_papers=5,
        fulltext=True,
        strip_references=True,
        raw_dir=tmp_path / "raw_med",
        index_dir=tmp_path / "index_med",
        query="checkpoint",
    )
    xml_by_uid = {"111": "<article/>", "222": "<article/>"}
    http_get = _make_http_get(["111", "222"], xml_by_uid)

    refs = download_pmc_articles(cfg, http_get=http_get)

    assert [r.pmcid for r in refs] == ["PMC111", "PMC222"]
    assert all(r.xml_path.exists() for r in refs)
    efetch_calls = sum(1 for url, _ in http_get.calls if url.endswith("efetch.fcgi"))  # type: ignore[attr-defined]
    assert efetch_calls == 2

    # Second run: files already present, so no new efetch calls (immutable raw dir).
    http_get2 = _make_http_get(["111", "222"], xml_by_uid)
    download_pmc_articles(cfg, http_get=http_get2)
    efetch_calls2 = sum(1 for url, _ in http_get2.calls if url.endswith("efetch.fcgi"))  # type: ignore[attr-defined]
    assert efetch_calls2 == 0


def test_download_pmc_requires_query(tmp_path: Path) -> None:
    cfg = CorpusConfig(
        source="pmc",
        categories=(),
        max_papers=5,
        fulltext=True,
        strip_references=True,
        raw_dir=tmp_path,
        index_dir=tmp_path,
        query=None,
    )
    with pytest.raises(ValueError, match="corpus.query"):
        download_pmc_articles(cfg, http_get=_make_http_get([], {}))
