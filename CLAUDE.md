# CLAUDE.md — rag-evaluation

## Project Overview
Production RAG **evaluation harness** benchmarking multiple retrieval strategies
(dense, hybrid dense+BM25, cross-encoder rerank, HyDE) behind one swappable Retriever
interface, evaluated with RAGAS + a custom LLM-as-judge harness and a regression gate.
Runs over **two independent corpora** — arXiv ML/AI papers (default) and open-access
PubMed Central medical literature (`configs/config_med.yaml`) — to demonstrate that the
same harness, unchanged, benchmarks a second domain (cross-domain generalization). Both
corpora share the same `config_fingerprint`; they differ only by index, eval set, and
committed baseline.
Stack: Python 3.11, LangChain, FAISS, rank_bm25, bge embeddings + reranker, RAGAS,
FastAPI, Streamlit, Docker. (LlamaIndex intentionally not used.)

## Commands
```bash
# The BASE install is the framework-free retrieval + indexing core only. Every command
# below needs the `full` extra: generation, judges, gate, fetchers, CLI, API, UI (D13).
pip install -e ".[full,dev]"
python -m rag_eval.cli ingest          # download → chunk → embed → index (arXiv)
python -m rag_eval.cli query "..."     # single query against a strategy
python -m rag_eval.cli evaluate        # run benchmark across all strategies
python scripts/run_benchmark.py        # full 5-strategy sweep + ablation report
uvicorn rag_eval.api:app --reload
streamlit run app.py
pytest tests/ -v --tb=short
ruff check . && ruff format .

# Second corpus: any entry point selects a corpus via RAG_CONFIG (default = arXiv).
RAG_CONFIG=configs/config_med.yaml python -m rag_eval.cli ingest   # PubMed Central
python scripts/run_benchmark.py --config configs/config_med.yaml
python scripts/cross_domain_report.py --out results/cross_domain.md
```

### Docker
Requires `data/index/` to already exist (run `cli ingest` on the host first -- the
index is gitignored and not baked into the image) and a `.env` with
`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` (see `.env.example`).
```bash
docker compose up --build   # api on :8000, ui (Streamlit) on :8501
```

## Architecture rules
- Every retrieval strategy implements the same `Retriever` interface in
  src/rag_eval/retrieval/base.py. Strategies are selected by config, never hardcoded.
- All config via YAML + dataclasses (src/rag_eval/config.py). No magic numbers in logic.
- Prompt templates live in src/rag_eval/generation/prompts/ as .txt — no f-strings buried in logic.
- Eval outputs are structured Pydantic models written to results/ with a run id + timestamp.

## What Claude must never do
- Never write to or edit the raw corpus dirs (data/raw/, data/raw_med/) — immutable once fetched.
- Never hardcode API keys — read from environment.
- Never add a retrieval strategy without a matching test and a metrics entry.
- Never "clean up" working code unless I ask.