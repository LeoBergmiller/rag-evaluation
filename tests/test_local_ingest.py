"""The `local` corpus source: authored Markdown/text already on disk (D13)."""

from pathlib import Path

import pytest

from rag_eval.config import CorpusConfig
from rag_eval.ingest.local_source import load_local_documents


def _corpus(raw_dir: Path) -> CorpusConfig:
    return CorpusConfig(source="local", raw_dir=raw_dir, index_dir=raw_dir / "index")


def test_load_local_documents_uses_stem_as_doc_id_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "readmission_30day.md").write_text(
        "Readmission is anchored on discharge."
    )
    (tmp_path / "length_of_stay.md").write_text("LOS counts midnights.")
    (tmp_path / "admission.txt").write_text("Admission excludes wellness visits.")

    documents = load_local_documents(_corpus(tmp_path))

    assert [doc_id for doc_id, _ in documents] == [
        "admission",
        "length_of_stay",
        "readmission_30day",
    ]
    assert documents[0][1] == "Admission excludes wellness visits."


def test_load_local_documents_recurses_into_subdirectories(tmp_path: Path) -> None:
    (tmp_path / "load_bearing").mkdir()
    (tmp_path / "distractors").mkdir()
    (tmp_path / "load_bearing" / "readmission_30day.md").write_text(
        "anchored on discharge"
    )
    (tmp_path / "distractors" / "readmission_90day.md").write_text("ninety day window")

    documents = load_local_documents(_corpus(tmp_path))

    assert sorted(doc_id for doc_id, _ in documents) == [
        "readmission_30day",
        "readmission_90day",
    ]


def test_load_local_documents_ignores_other_suffixes(tmp_path: Path) -> None:
    (tmp_path / "kept.md").write_text("kept")
    (tmp_path / "ignored.pdf").write_text("ignored")
    (tmp_path / "ignored.json").write_text("{}")

    documents = load_local_documents(_corpus(tmp_path))

    assert [doc_id for doc_id, _ in documents] == ["kept"]


def test_missing_raw_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        load_local_documents(_corpus(tmp_path / "nope"))


def test_empty_corpus_raises_rather_than_building_an_empty_index(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="No .* documents found"):
        load_local_documents(_corpus(tmp_path))


def test_colliding_stems_raise(tmp_path: Path) -> None:
    """doc_id is the stem, so a collision would silently drop a document."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "readmission.md").write_text("one")
    (tmp_path / "sub" / "readmission.txt").write_text("two")

    with pytest.raises(ValueError, match="Duplicate document ids"):
        load_local_documents(_corpus(tmp_path))


def test_empty_document_raises(tmp_path: Path) -> None:
    (tmp_path / "good.md").write_text("real content")
    (tmp_path / "blank.md").write_text("   \n\n")

    with pytest.raises(ValueError, match="Empty documents"):
        load_local_documents(_corpus(tmp_path))
