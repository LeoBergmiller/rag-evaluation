"""Grounded answer generation with citation extraction and abstention detection."""

from __future__ import annotations

import re

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel

from rag_eval.config import GenerationConfig
from rag_eval.generation.llm import build_chat_model
from rag_eval.generation.prompts import build_answer_prompt
from rag_eval.retrieval.base import ScoredChunk

_CITATION_RE = re.compile(r"\[([^\]]+)\]")
_ABSTENTION_PHRASE = "i don't know"


class GenerationResult(BaseModel):
    answer: str
    cited_chunk_ids: list[str]
    abstained: bool


def format_context(chunks: list[ScoredChunk]) -> str:
    """Render retrieved chunks as `[chunk_id] text` lines so the model can cite ids."""
    return "\n".join(f"[{chunk.chunk_id}] {chunk.text}" for chunk in chunks)


class Generator:
    """LCEL prompt -> chat model -> string chain for grounded answer generation."""

    def __init__(
        self, generation_cfg: GenerationConfig, chat_model: BaseChatModel | None = None
    ) -> None:
        model = chat_model or build_chat_model(
            generation_cfg.provider,
            generation_cfg.model,
            generation_cfg.temperature,
            generation_cfg.max_tokens,
        )
        self._chain = build_answer_prompt() | model | StrOutputParser()

    def generate(self, question: str, chunks: list[ScoredChunk]) -> GenerationResult:
        context = format_context(chunks)
        answer = self._chain.invoke({"context": context, "question": question})

        retrieved_ids = {chunk.chunk_id for chunk in chunks}
        cited_chunk_ids = [
            cid for cid in _CITATION_RE.findall(answer) if cid in retrieved_ids
        ]
        # de-duplicate while preserving order
        cited_chunk_ids = list(dict.fromkeys(cited_chunk_ids))

        abstained = answer.strip().lower().startswith(_ABSTENTION_PHRASE)
        if abstained:
            cited_chunk_ids = []

        return GenerationResult(
            answer=answer, cited_chunk_ids=cited_chunk_ids, abstained=abstained
        )
