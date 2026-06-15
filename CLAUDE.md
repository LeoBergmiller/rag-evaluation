# CLAUDE.md — rag-evaluation

## Project Overview
Production RAG system over an arXiv ML/AI paper corpus, implementing and
benchmarking multiple retrieval strategies (dense, hybrid dense+BM25,
cross-encoder rerank, HyDE) behind one swappable Retriever interface,
evaluated with RAGAS + a custom LLM-as-judge harness and a regression gate.
Stack: Python 3.11, LangChain/LlamaIndex, FAISS, rank_bm25, bge embeddings +
reranker, RAGAS, FastAPI, Streamlit, Docker.

## Commands
```bash
pip install -e ".[dev]"
python -m rag_eval.cli ingest          # download → chunk → embed → index
python -m rag_eval.cli query "..."     # single query against a strategy
python -m rag_eval.cli evaluate        # run benchmark across all strategies
uvicorn rag_eval.api:app --reload
streamlit run app.py
pytest tests/ -v --tb=short
ruff check . && ruff format .
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
- Never write to or edit data/raw/ (source corpus is immutable).
- Never hardcode API keys — read from environment.
- Never add a retrieval strategy without a matching test and a metrics entry.
- Never "clean up" working code unless I ask.