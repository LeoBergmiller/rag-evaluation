"""Typed configuration loaded from configs/config.yaml.

Loads .env at import time so secrets (ANTHROPIC_API_KEY, OPENAI_API_KEY) are
available to downstream modules at runtime without being exported in the shell
or hardcoded anywhere.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
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
    """Corpus location and fetch parameters.

    Only `source`, `raw_dir` and `index_dir` are universal. The rest describe how to
    FETCH a corpus, and each source needs a different subset: arXiv searches by
    `categories`, PMC by `query`, and `local` fetches nothing at all. They are
    therefore optional on the dataclass and required per-source at load time by
    `_SOURCE_REQUIRED_KEYS` — a config that omits a key its own source needs still
    fails loudly, but a local corpus does not have to carry meaningless arXiv fields
    (D13). No field here feeds `Config.fingerprint()`, so none of this perturbs the
    controlled-variable identity that D12 relies on.
    """

    source: str
    raw_dir: Path
    index_dir: Path
    # arXiv (source="arxiv") search categories.
    categories: tuple[str, ...] = ()
    max_papers: int = 0
    fulltext: bool = True
    strip_references: bool = True
    # PMC (source="pmc") search term; unused by the arXiv source (which uses `categories`).
    query: str | None = None


#: Fetch parameters each source genuinely requires, beyond the universal three.
_SOURCE_REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "arxiv": ("categories", "max_papers", "fulltext", "strip_references"),
    "pmc": ("query", "max_papers", "strip_references"),
    "local": (),
}


def _build_corpus_config(raw: dict[str, Any]) -> CorpusConfig:
    source = raw.get("source")
    if source not in _SOURCE_REQUIRED_KEYS:
        raise ValueError(
            f"Unknown corpus.source: {source!r} "
            f"(known: {sorted(_SOURCE_REQUIRED_KEYS)})"
        )

    required = ("raw_dir", "index_dir", *_SOURCE_REQUIRED_KEYS[source])
    missing = [key for key in required if raw.get(key) is None]
    if missing:
        raise ValueError(
            f"corpus.source={source!r} requires these keys, which are missing or "
            f"null: {missing}"
        )

    return CorpusConfig(
        source=source,
        raw_dir=Path(raw["raw_dir"]),
        index_dir=Path(raw["index_dir"]),
        categories=tuple(raw.get("categories") or ()),
        max_papers=int(raw.get("max_papers", 0)),
        fulltext=bool(raw.get("fulltext", True)),
        strip_references=bool(raw.get("strip_references", True)),
        query=raw.get("query"),
    )


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
    rrf_k: int


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
    # Committed baseline the regression gate compares against for this corpus.
    baseline_path: str = "results/baseline.json"


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
        comparison), the eval set, and the prompt template -- those are
        tracked separately as eval_set_hash and prompt_template_hash on
        StrategyReport and checked independently by the regression gate.
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
    """Load and validate configuration from a YAML file.

    When no explicit path is given, the RAG_CONFIG environment variable selects the
    config file (used to point any entry point at a second corpus, e.g.
    RAG_CONFIG=configs/config_med.yaml); if unset, the arXiv default is used.
    """
    if path is not None:
        config_path = path
    else:
        env_path = os.environ.get("RAG_CONFIG")
        config_path = Path(env_path) if env_path else _DEFAULT_CONFIG_PATH

    if not config_path.is_file():
        raise FileNotFoundError(
            f"No config file at {config_path}. The default path is resolved relative "
            "to this package's source tree, so it only exists in a checkout or an "
            "editable install. A consumer that installed rag-eval as a dependency "
            "must pass load_config(path=...) explicitly or set RAG_CONFIG."
        )

    with config_path.open("r") as f:
        raw = yaml.safe_load(f)

    corpus = _build_corpus_config(raw["corpus"])

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
        rrf_k=retrieval_raw["rrf_k"],
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
        baseline_path=evaluation_raw.get("baseline_path", "results/baseline.json"),
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
