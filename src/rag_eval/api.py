"""HTTP serving layer: `uvicorn rag_eval.api:app --reload`."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_eval.config import load_config
from rag_eval.generation.generator import Generator
from rag_eval.retrieval.registry import (
    build_retriever,
    load_resources,
    registered_strategies,
)

logging.basicConfig(level=logging.INFO)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    strategy: str | None = None
    k: int | None = Field(default=None, gt=0)


class ChunkOut(BaseModel):
    chunk_id: str
    score: float
    text: str
    parent_id: str | None


class QueryResponse(BaseModel):
    answer: str
    abstained: bool
    cited_chunk_ids: list[str]
    strategy: str
    latency_ms: float
    chunks: list[ChunkOut]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    cfg = load_config()
    app.state.cfg = cfg
    app.state.resources = load_resources(cfg)
    app.state.generator = Generator(cfg.generation)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="rag-evaluation", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        cfg = app.state.cfg
        return {
            "status": "ok",
            "strategies": registered_strategies(),
            "config_fingerprint": cfg.fingerprint(),
        }

    @app.post("/query")
    def query(request: QueryRequest) -> QueryResponse:
        cfg = app.state.cfg
        strategies = registered_strategies()
        if request.strategy is not None and request.strategy not in strategies:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown strategy: {request.strategy!r} (registered: {strategies})",
            )

        strategy = request.strategy or cfg.retrieval.strategy
        k = request.k or cfg.retrieval.top_k

        retriever = build_retriever(strategy, cfg, app.state.resources)
        result = retriever.retrieve(request.question, k)
        gen_result = app.state.generator.generate(request.question, result.chunks)

        return QueryResponse(
            answer=gen_result.answer,
            abstained=gen_result.abstained,
            cited_chunk_ids=gen_result.cited_chunk_ids,
            strategy=result.strategy,
            latency_ms=result.latency_ms,
            chunks=[
                ChunkOut(
                    chunk_id=chunk.chunk_id,
                    score=chunk.score,
                    text=chunk.text,
                    parent_id=chunk.parent_id,
                )
                for chunk in result.chunks
            ],
        )

    return app


app = create_app()
