import dataclasses

import pytest

from rag_eval.config import load_config


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
