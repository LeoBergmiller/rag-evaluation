"""One-time LLM synthesis of the gold eval set from the ingested corpus.

Offline tooling — not imported by the eval harness. Run via:
    python -m rag_eval.evaluation.synth
"""

from __future__ import annotations

import logging
import random

from langchain_core.runnables import Runnable
from pydantic import BaseModel

from rag_eval.config import Config, load_config
from rag_eval.evaluation.dataset import EvalDataset, EvalExample
from rag_eval.generation.llm import build_chat_model
from rag_eval.generation.prompts import load_prompt
from rag_eval.ingest.chunk import Chunk
from rag_eval.ingest.pipeline import load_chunks

logger = logging.getLogger(__name__)

_MIN_CHUNK_CHARS = 500


class QAItem(BaseModel):
    question: str
    reference_answer: str


def build_eval_set(
    cfg: Config, *, n_answerable: int = 90, n_unanswerable: int = 10, seed: int = 0
) -> EvalDataset:
    chunks = load_chunks(cfg.corpus.index_dir)
    substantive = [c for c in chunks if len(c.text) >= _MIN_CHUNK_CHARS]

    rng = random.Random(seed)
    answerable_chunks = rng.sample(substantive, min(n_answerable, len(substantive)))
    unanswerable_chunks = rng.sample(substantive, min(n_unanswerable, len(substantive)))

    model = build_chat_model(
        cfg.generation.provider,
        cfg.generation.model,
        cfg.generation.temperature,
        cfg.generation.max_tokens,
    ).with_structured_output(QAItem)

    answerable_template = load_prompt("synth_answerable.txt")
    unanswerable_template = load_prompt("synth_unanswerable.txt")

    examples: list[EvalExample] = []

    for i, chunk in enumerate(answerable_chunks):
        item = _synthesize(model, answerable_template, chunk, label="answerable")
        if item is None:
            continue
        examples.append(
            EvalExample(
                id=f"answerable::{i}::{chunk.chunk_id}",
                question=item.question,
                reference_answer=item.reference_answer,
                reference_chunk_ids=[chunk.chunk_id],
                answerable=True,
                split="test",
            )
        )

    for i, chunk in enumerate(unanswerable_chunks):
        item = _synthesize(model, unanswerable_template, chunk, label="unanswerable")
        if item is None:
            continue
        examples.append(
            EvalExample(
                id=f"unanswerable::{i}::{chunk.chunk_id}",
                question=item.question,
                reference_answer=None,
                reference_chunk_ids=None,
                answerable=False,
                split="test",
            )
        )

    dev_frac = cfg.evaluation.dev_test_split[0]
    examples = EvalDataset.assign_splits(examples, dev_frac=dev_frac, seed=seed)
    return EvalDataset(examples=examples)


def _synthesize(
    model: Runnable, template: str, chunk: Chunk, *, label: str
) -> QAItem | None:
    prompt = template.format(passage=chunk.text)
    for attempt in range(2):
        try:
            result = model.invoke(prompt)
            if isinstance(result, QAItem):
                return result
            return QAItem.model_validate(result)
        except Exception:
            logger.warning(
                "Synthesis attempt %d failed for %s chunk %s",
                attempt + 1,
                label,
                chunk.chunk_id,
                exc_info=True,
            )
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    dataset = build_eval_set(cfg)
    dataset.save(cfg.evaluation.eval_set)

    logger.info(
        "Wrote %d examples (%d dev, %d test) to %s",
        len(dataset.examples),
        len(dataset.dev),
        len(dataset.test),
        cfg.evaluation.eval_set,
    )

    logger.info("Sample examples:")
    for example in dataset.examples[:5]:
        logger.info(
            "[%s] (split=%s, answerable=%s)\n  Q: %s\n  A: %s\n  refs: %s",
            example.id,
            example.split,
            example.answerable,
            example.question,
            example.reference_answer,
            example.reference_chunk_ids,
        )
