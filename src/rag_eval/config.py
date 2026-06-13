"""Typed configuration loaded from configs/config.yaml.

Loads .env at import time so secrets (ANTHROPIC_API_KEY, OPENAI_API_KEY) are
available to downstream modules at runtime without being exported in the shell
or hardcoded anywhere.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


@dataclass(frozen=True)
class CorpusConfig:
    source: str
    categories: tuple[str, ...]
    max_papers: int
    fulltext: bool
    strip_references: bool
    raw_dir: Path
    index_dir: Path


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str
    units: str
    chunk_size: int
    chunk_overlap: int
    store_parent: bool


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    device: str
    query_prefix: str


@dataclass(frozen=True)
class RetrievalConfig:
    strategy: str
    index_type: str
    top_k: int
    candidate_k: int
    reranker: str
    fusion: str


@dataclass(frozen=True)
class GenerationConfig:
    provider: str
    model: str
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class GateConfig:
    floors: dict[str, float] = field(default_factory=dict)
    tolerance: dict[str, float] = field(default_factory=dict)
    operational_ceilings: dict[str, float | None] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationConfig:
    judge_provider: str
    judge_model: str
    judge_temperature: float
    ragas_metrics: tuple[str, ...]
    retrieval_metrics: tuple[str, ...]
    abstention: bool
    bootstrap_resamples: int
    eval_set: Path
    dev_test_split: tuple[float, ...]
    gate: GateConfig


@dataclass(frozen=True)
class Config:
    corpus: CorpusConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig

    def fingerprint(self) -> str:
        """Hash of the controlled variables held fixed across strategies.

        Excludes judge settings (not a controlled variable of the strategy
        comparison) and the eval set (tracked separately via eval_set_hash).
        TODO: fold prompt-template content in once generation/prompts/ exists (step 4).
        """
        canon: dict[str, Any] = {
            "embedding_model": self.embedding.model,
            "embedding_query_prefix": self.embedding.query_prefix,
            "chunking_strategy": self.chunking.strategy,
            "chunking_units": self.chunking.units,
            "chunk_size": self.chunking.chunk_size,
            "chunk_overlap": self.chunking.chunk_overlap,
            "top_k": self.retrieval.top_k,
            "generation_provider": self.generation.provider,
            "generation_model": self.generation.model,
            "generation_temperature": self.generation.temperature,
            "generation_max_tokens": self.generation.max_tokens,
        }
        digest = hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()
        return digest[:16]


def load_config(path: Path | None = None) -> Config:
    """Load and validate configuration from a YAML file."""
    config_path = path or _DEFAULT_CONFIG_PATH
    with config_path.open("r") as f:
        raw = yaml.safe_load(f)

    corpus_raw = raw["corpus"]
    corpus = CorpusConfig(
        source=corpus_raw["source"],
        categories=tuple(corpus_raw["categories"]),
        max_papers=corpus_raw["max_papers"],
        fulltext=corpus_raw["fulltext"],
        strip_references=corpus_raw["strip_references"],
        raw_dir=Path(corpus_raw["raw_dir"]),
        index_dir=Path(corpus_raw["index_dir"]),
    )

    chunking_raw = raw["chunking"]
    chunking = ChunkingConfig(
        strategy=chunking_raw["strategy"],
        units=chunking_raw["units"],
        chunk_size=chunking_raw["chunk_size"],
        chunk_overlap=chunking_raw["chunk_overlap"],
        store_parent=chunking_raw["store_parent"],
    )

    embedding_raw = raw["embedding"]
    embedding = EmbeddingConfig(
        model=embedding_raw["model"],
        device=embedding_raw["device"],
        query_prefix=embedding_raw["query_prefix"],
    )

    retrieval_raw = raw["retrieval"]
    retrieval = RetrievalConfig(
        strategy=retrieval_raw["strategy"],
        index_type=retrieval_raw["index_type"],
        top_k=retrieval_raw["top_k"],
        candidate_k=retrieval_raw["candidate_k"],
        reranker=retrieval_raw["reranker"],
        fusion=retrieval_raw["fusion"],
    )

    generation_raw = raw["generation"]
    generation = GenerationConfig(
        provider=generation_raw["provider"],
        model=generation_raw["model"],
        temperature=generation_raw["temperature"],
        max_tokens=generation_raw["max_tokens"],
    )

    evaluation_raw = raw["evaluation"]
    gate_raw = evaluation_raw["gate"]
    gate = GateConfig(
        floors=dict(gate_raw["floors"]),
        tolerance=dict(gate_raw["tolerance"]),
        operational_ceilings=dict(gate_raw["operational_ceilings"]),
    )
    evaluation = EvaluationConfig(
        judge_provider=evaluation_raw["judge_provider"],
        judge_model=evaluation_raw["judge_model"],
        judge_temperature=evaluation_raw["judge_temperature"],
        ragas_metrics=tuple(evaluation_raw["ragas_metrics"]),
        retrieval_metrics=tuple(evaluation_raw["retrieval_metrics"]),
        abstention=evaluation_raw["abstention"],
        bootstrap_resamples=evaluation_raw["bootstrap_resamples"],
        eval_set=Path(evaluation_raw["eval_set"]),
        dev_test_split=tuple(evaluation_raw["dev_test_split"]),
        gate=gate,
    )

    config = Config(
        corpus=corpus,
        chunking=chunking,
        embedding=embedding,
        retrieval=retrieval,
        generation=generation,
        evaluation=evaluation,
    )
    logger.debug(
        "Loaded config from %s (fingerprint=%s)", config_path, config.fingerprint()
    )
    return config
