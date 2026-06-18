"""Custom LLM-as-judge rubric: correctness, completeness, and citation validity.

Scores three dimensions RAGAS does not cover:
  - correctness vs reference_answer (0-2)
  - completeness vs reference_answer (0-2)
  - citation_valid: do cited chunk ids actually support the claims? (bool)

Grounding (faithfulness) is deliberately omitted — it duplicates RAGAS faithfulness.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, field_validator

from rag_eval.config import Config
from rag_eval.generation.llm import build_chat_model
from rag_eval.generation.prompts import build_rubric_prompt

logger = logging.getLogger(__name__)

_ABSTENTION_PREFIX = "i don't know"
_RUBRIC_MAX_TOKENS = 256


class RubricScore(BaseModel):
    correctness: int
    completeness: int
    citation_valid: bool
    rationale: str

    @field_validator("correctness", "completeness")
    @classmethod
    def must_be_0_1_2(cls, v: int) -> int:
        if v not in (0, 1, 2):
            raise ValueError(f"must be 0, 1, or 2; got {v}")
        return v


def _format_cited_chunks(cited_chunk_ids: list[str], chunks_by_id: dict[str, str]) -> str:
    if not cited_chunk_ids:
        return "(none cited)"
    return "\n\n".join(
        f"[{cid}]: {chunks_by_id.get(cid, '(text not available)')}"
        for cid in cited_chunk_ids
    )


def _build_rubric_chain(cfg: Config) -> Any:
    model = build_chat_model("openai", cfg.evaluation.judge_model, 0, _RUBRIC_MAX_TOKENS)
    return build_rubric_prompt() | model.with_structured_output(RubricScore)


def score_example(
    *,
    question: str,
    reference_answer: str | None,
    model_answer: str,
    cited_chunk_ids: list[str],
    chunks_by_id: dict[str, str],
    cfg: Config,
    _chain: Any | None = None,
) -> RubricScore | None:
    """Score one example; returns None for unanswerable/abstained examples or on double judge failure."""
    if reference_answer is None:
        return None
    if model_answer.strip().lower().startswith(_ABSTENTION_PREFIX):
        return None

    chain = _chain or _build_rubric_chain(cfg)
    inputs = {
        "question": question,
        "reference_answer": reference_answer,
        "answer": model_answer,
        "cited_chunk_texts": _format_cited_chunks(cited_chunk_ids, chunks_by_id),
    }

    for attempt in range(2):
        try:
            result = chain.invoke(inputs)
            return result
        except Exception as exc:
            if attempt == 0:
                logger.warning("Rubric judge failed (attempt 1): %s — retrying", exc)
            else:
                logger.warning("Rubric judge failed twice; recording None scores: %s", exc)
    return None
