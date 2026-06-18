import logging
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from rag_eval.evaluation.rubric import RubricScore, _format_cited_chunks, score_example


def _cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.evaluation.judge_model = "gpt-4o"
    return cfg


def _good_score() -> RubricScore:
    return RubricScore(correctness=2, completeness=1, citation_valid=True, rationale="looks good")


# --- RubricScore model ---


def test_rubric_score_valid() -> None:
    s = _good_score()
    assert s.correctness == 2
    assert s.completeness == 1
    assert s.citation_valid is True


def test_rubric_score_rejects_out_of_range_correctness() -> None:
    with pytest.raises(ValidationError):
        RubricScore(correctness=3, completeness=1, citation_valid=True, rationale="x")


def test_rubric_score_rejects_negative_completeness() -> None:
    with pytest.raises(ValidationError):
        RubricScore(correctness=1, completeness=-1, citation_valid=True, rationale="x")


# --- _format_cited_chunks ---


def test_format_cited_chunks_empty() -> None:
    assert _format_cited_chunks([], {}) == "(none cited)"


def test_format_cited_chunks_with_ids() -> None:
    result = _format_cited_chunks(["p::1", "p::2"], {"p::1": "alpha", "p::2": "beta"})
    assert "[p::1]: alpha" in result
    assert "[p::2]: beta" in result


def test_format_cited_chunks_missing_id() -> None:
    result = _format_cited_chunks(["p::99"], {})
    assert "[p::99]" in result
    assert "not available" in result


# --- score_example ---


def test_score_example_returns_valid_rubric() -> None:
    chain = MagicMock()
    chain.invoke.return_value = _good_score()

    result = score_example(
        question="What is attention?",
        reference_answer="Attention is a mechanism...",
        model_answer="Attention allows [p::1]",
        cited_chunk_ids=["p::1"],
        chunks_by_id={"p::1": "attention text"},
        cfg=_cfg(),
        _chain=chain,
    )

    assert isinstance(result, RubricScore)
    assert result.correctness == 2
    chain.invoke.assert_called_once()


def test_score_example_retries_on_first_failure(caplog: pytest.LogCaptureFixture) -> None:
    chain = MagicMock()
    chain.invoke.side_effect = [Exception("parse error"), _good_score()]

    with caplog.at_level(logging.WARNING):
        result = score_example(
            question="q",
            reference_answer="ref",
            model_answer="answer",
            cited_chunk_ids=[],
            chunks_by_id={},
            cfg=_cfg(),
            _chain=chain,
        )

    assert result is not None
    assert result.correctness == 2
    assert chain.invoke.call_count == 2
    assert "retrying" in caplog.text


def test_score_example_returns_none_on_double_failure(caplog: pytest.LogCaptureFixture) -> None:
    chain = MagicMock()
    chain.invoke.side_effect = [Exception("fail 1"), Exception("fail 2")]

    with caplog.at_level(logging.WARNING):
        result = score_example(
            question="q",
            reference_answer="ref",
            model_answer="answer",
            cited_chunk_ids=[],
            chunks_by_id={},
            cfg=_cfg(),
            _chain=chain,
        )

    assert result is None
    assert chain.invoke.call_count == 2
    assert "twice" in caplog.text


def test_score_example_skips_abstained_answer() -> None:
    chain = MagicMock()

    result = score_example(
        question="q",
        reference_answer="ref",
        model_answer="I don't know",
        cited_chunk_ids=[],
        chunks_by_id={},
        cfg=_cfg(),
        _chain=chain,
    )

    assert result is None
    chain.invoke.assert_not_called()


def test_score_example_skips_when_no_reference() -> None:
    chain = MagicMock()

    result = score_example(
        question="q",
        reference_answer=None,
        model_answer="some answer",
        cited_chunk_ids=[],
        chunks_by_id={},
        cfg=_cfg(),
        _chain=chain,
    )

    assert result is None
    chain.invoke.assert_not_called()


def test_score_example_abstention_case_insensitive() -> None:
    chain = MagicMock()

    result = score_example(
        question="q",
        reference_answer="ref",
        model_answer="  I DON'T KNOW the answer",
        cited_chunk_ids=[],
        chunks_by_id={},
        cfg=_cfg(),
        _chain=chain,
    )

    assert result is None
    chain.invoke.assert_not_called()
