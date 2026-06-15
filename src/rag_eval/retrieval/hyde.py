"""HyDE retrieval: embed an LLM-generated hypothetical document instead of the
raw query, then dense-retrieve with it.
"""

from __future__ import annotations

import time
from typing import Protocol

from langchain_core.output_parsers import StrOutputParser

from rag_eval.config import Config
from rag_eval.generation.llm import build_chat_model
from rag_eval.generation.prompts import build_hyde_prompt
from rag_eval.retrieval.base import RetrievalResult, Retriever


class QueryExpander(Protocol):
    def expand(self, query: str) -> str: ...


class LLMQueryExpander:
    """Generates a hypothetical document that plausibly answers the query."""

    def __init__(self, cfg: Config) -> None:
        model = build_chat_model(
            cfg.generation.provider,
            cfg.generation.model,
            cfg.generation.temperature,
            cfg.generation.max_tokens,
        )
        self._chain = build_hyde_prompt() | model | StrOutputParser()

    def expand(self, query: str) -> str:
        return self._chain.invoke({"question": query})


class HydeRetriever:
    """Wraps a base retriever, retrieving with a hypothetical document in place
    of the raw query."""

    name = "hyde"

    def __init__(self, base: Retriever, expander: QueryExpander) -> None:
        self._base = base
        self._expander = expander

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        start = time.perf_counter()

        hypothetical_document = self._expander.expand(query)
        inner = self._base.retrieve(hypothetical_document, k)

        latency_ms = (time.perf_counter() - start) * 1000
        return RetrievalResult(
            query=query,
            chunks=inner.chunks,
            latency_ms=latency_ms,
            strategy=self.name,
            diagnostics={
                "base_strategy": self._base.name,
                "hypothetical_document": hypothetical_document[:300],
            },
        )
