import dataclasses
from pathlib import Path

import pytest

from rag_eval.config import _build_corpus_config, load_config


def test_load_config_fields_and_types() -> None:
    cfg = load_config()

    assert cfg.embedding.model == "BAAI/bge-base-en-v1.5"
    assert cfg.embedding.query_prefix.startswith("Represent this sentence")

    assert cfg.chunking.strategy == "recursive"
    assert cfg.chunking.units == "tokens"
    assert cfg.chunking.chunk_size == 512
    assert cfg.chunking.chunk_overlap == 64

    assert cfg.retrieval.index_type == "flat"
    assert cfg.retrieval.top_k == 5
    assert cfg.retrieval.candidate_k == 50
    assert cfg.retrieval.rrf_k == 60

    assert cfg.generation.provider == "anthropic"
    assert cfg.generation.temperature == 0

    assert isinstance(cfg.evaluation.gate.floors, dict)
    assert cfg.evaluation.gate.floors["faithfulness"] == 0.92
    assert cfg.evaluation.gate.operational_ceilings["p95_latency_ms"] == 185.71


def test_fingerprint_stable() -> None:
    fp1 = load_config().fingerprint()
    fp2 = load_config().fingerprint()

    assert fp1 == fp2
    assert len(fp1) == 16


def test_fingerprint_sensitive_to_controlled_variables() -> None:
    cfg = load_config()
    baseline = cfg.fingerprint()

    changed_embedding = dataclasses.replace(cfg.embedding, model="some-other-model")
    changed_cfg = dataclasses.replace(cfg, embedding=changed_embedding)
    assert changed_cfg.fingerprint() != baseline


def test_fingerprint_ignores_non_controlled_variables() -> None:
    cfg = load_config()
    baseline = cfg.fingerprint()

    changed_eval = dataclasses.replace(cfg.evaluation, judge_model="some-other-judge")
    changed_cfg = dataclasses.replace(cfg, evaluation=changed_eval)
    assert changed_cfg.fingerprint() == baseline


def test_config_is_frozen() -> None:
    cfg = load_config()

    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.retrieval.top_k = 10  # type: ignore[misc]


# --- per-source corpus schema (D13) ---


def test_local_corpus_needs_no_arxiv_fetch_fields() -> None:
    corpus = _build_corpus_config(
        {"source": "local", "raw_dir": "data/raw_ops", "index_dir": "data/index_ops"}
    )

    assert corpus.source == "local"
    assert corpus.raw_dir == Path("data/raw_ops")
    assert corpus.categories == ()
    assert corpus.query is None


def test_arxiv_corpus_still_requires_its_fetch_fields() -> None:
    with pytest.raises(ValueError, match="max_papers"):
        _build_corpus_config(
            {
                "source": "arxiv",
                "raw_dir": "data/raw",
                "index_dir": "data/index",
                "categories": ["cs.CL"],
                "fulltext": True,
                "strip_references": True,
            }
        )


def test_pmc_corpus_still_requires_its_query() -> None:
    with pytest.raises(ValueError, match="query"):
        _build_corpus_config(
            {
                "source": "pmc",
                "raw_dir": "data/raw_med",
                "index_dir": "data/index_med",
                "max_papers": 500,
                "strip_references": True,
            }
        )


def test_every_source_requires_the_universal_directories() -> None:
    with pytest.raises(ValueError, match="index_dir"):
        _build_corpus_config({"source": "local", "raw_dir": "data/raw_ops"})


def test_unknown_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown corpus.source"):
        _build_corpus_config(
            {"source": "s3", "raw_dir": "data/raw", "index_dir": "data/index"}
        )


def test_committed_configs_still_load_and_share_a_fingerprint() -> None:
    """D12's cross-domain claim rests on both corpora hashing identically."""
    arxiv = load_config(Path("configs/config.yaml"))
    med = load_config(Path("configs/config_med.yaml"))

    assert arxiv.corpus.source == "arxiv"
    assert med.corpus.source == "pmc"
    assert arxiv.fingerprint() == med.fingerprint()


def test_missing_config_path_is_actionable() -> None:
    with pytest.raises(FileNotFoundError, match="RAG_CONFIG"):
        load_config(Path("configs/does_not_exist.yaml"))
