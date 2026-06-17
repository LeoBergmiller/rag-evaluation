# rag-evaluation

[![CI](https://github.com/LeoBergmiller/rag-evaluation/actions/workflows/ci.yml/badge.svg)](https://github.com/LeoBergmiller/rag-evaluation/actions/workflows/ci.yml)
[![Docker Hub](https://img.shields.io/docker/v/leobergmiller/rag-evaluation?label=Docker%20Hub)](https://hub.docker.com/r/leobergmiller/rag-evaluation)

A production-style RAG system over an arXiv ML/AI paper corpus that implements and
**benchmarks five retrieval strategies** behind one swappable interface, evaluated with
[RAGAS](https://github.com/explodinggradients/ragas) + a custom retrieval-metrics harness,
gated by an automated regression check, and served via a FastAPI + Streamlit demo.

The goal isn't "a RAG demo" — it's the harness around it: a controlled experiment that lets
you swap retrieval strategies via config, measure quality/latency/cost with confidence
intervals, and fail CI if a change regresses quality below a calibrated tolerance.

## Architecture

```
                 ┌──────────────┐
 question ──────▶│  Retriever   │  dense | bm25 | hybrid (RRF) | rerank | hyde
                 │  (registry)  │  -- one Protocol, selected by config --
                 └──────┬───────┘
                        │ ScoredChunk[]
                        ▼
                 ┌──────────────┐
                 │  Generator   │  Claude (generation family A)
                 │ (LCEL chain) │  grounded answer + citations + abstention
                 └──────┬───────┘
                        │
        ┌───────────────┼────────────────────┐
        ▼                                     ▼
┌────────────────┐                  ┌──────────────────┐
│ Eval harness    │   RAGAS judge:   │ Regression gate   │
│ retrieval +     │   GPT-4o         │ provenance + floor│
│ RAGAS + bootstrap CIs│ (family B)  │ + tolerance +      │
└────────┬────────┘                  │ ceiling checks    │
         │                            └──────────────────┘
         ▼
  StrategyReport (JSON) ──▶ AblationReport ──▶ FastAPI (/query, /health, /ablation)
                                                   │
                                                   ▼
                                            Streamlit UI (Ask / Benchmark)
```

- **Every retrieval strategy implements the same `Retriever` protocol**
  ([src/rag_eval/retrieval/base.py](src/rag_eval/retrieval/base.py)) and is selected purely by
  config (`configs/config.yaml: retrieval.strategy`) — never hardcoded.
- **Generator (family A) is Claude**; the **RAGAS judge (family B) is GPT-4o** — a different
  model family from the generator, specifically to avoid self-preference bias in the judge.
- **`Config.fingerprint()`** hashes the controlled variables (embedding model, chunking,
  top_k, generation settings) so every report and the regression gate can verify they're
  comparing apples to apples.
- **The regression gate** ([src/rag_eval/gate/regression.py](src/rag_eval/gate/regression.py))
  checks four things against a committed baseline: provenance (fingerprint / eval set /
  prompt template all match), quality floors, no-regression tolerances, and operational
  ceilings (p95 latency, cost per query).

## Retrieval strategies

| Strategy | Description |
| --- | --- |
| `dense` | BGE (`bge-base-en-v1.5`) embeddings + exact FAISS `IndexFlatIP` |
| `bm25` | Sparse lexical retrieval (`rank_bm25`) |
| `hybrid` | Dense + BM25 fused via Reciprocal Rank Fusion |
| `rerank` | Dense candidates re-scored by a cross-encoder (`bge-reranker-base`) |
| `hyde` | LLM-generated hypothetical document, embedded and used as the dense query |

## Latest benchmark results

59-example gold eval set, `dense` as baseline (see
[results/20260615T005622Z_ablation_48a41165.md](results/20260615T005622Z_ablation_48a41165.md)
for the full report):

| strategy | faithfulness (Δ) | context_recall (Δ) | recall_at_k (Δ) | ndcg_at_k (Δ) | p95_latency_ms | cost_per_query_usd | gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | 0.9410 (+0.0092) | 0.9327 (+0.0865) | 0.9231 (+0.1731) | 0.8129 (+0.2709) | 5.40 | 0.0118 | ✓ |
| dense (baseline) | 0.9318 | 0.8462 | 0.7500 | 0.5420 | 39.91 | 0.0117 | ✓ |
| hybrid | 0.9408 (+0.0090) | 0.9135 (+0.0673) | 0.8269 (+0.0769) | 0.6831 (+0.1411) | 40.64 | 0.0118 | ✓ |
| hyde | 0.9267 (-0.0052) | 0.8494 (+0.0032) | 0.7692 (+0.0192) | 0.6182 (+0.0762) | 11444.68 | 0.0118 | ✗ |
| rerank | 0.9473 (+0.0154) | 0.9231 (+0.0769) | 0.9038 (+0.1538) | 0.8254 (+0.2835) | 1104.74 | 0.0118 | ✗ |

`hyde` and `rerank` fail the gate on latency/operational ceilings (not quality) — both add a
second LLM/model pass per query, which the gate is specifically designed to surface.

## Quickstart

```bash
git clone https://github.com/LeoBergmiller/rag-evaluation.git
cd rag-evaluation
pip install -e ".[dev]"
cp .env.example .env   # fill in ANTHROPIC_API_KEY and OPENAI_API_KEY
```

```bash
python -m rag_eval.cli ingest                 # download → chunk → embed → build FAISS + BM25 index
python -m rag_eval.cli query "What is attention?" --strategy hybrid
python -m rag_eval.cli evaluate --strategy dense --no-gate   # writes results/*_dense_*.json
python scripts/run_benchmark.py               # full 5-strategy sweep + ablation report
```

```bash
uvicorn rag_eval.api:app --reload   # http://localhost:8000 (docs at /docs, /health, /ablation)
streamlit run app.py                # http://localhost:8501 — Ask / Benchmark UI
```

### Docker

A pre-built image is available on Docker Hub — no local build required:

```bash
docker pull leobergmiller/rag-evaluation:latest
docker compose up          # api on :8000, Streamlit ui on :8501
```

Or build from source:

```bash
docker compose up --build
```

Requires `data/index/` to already exist (run `cli ingest` on the host first — the index is
gitignored and not baked into the image) and a `.env` with the two API keys.

## Testing & CI

```bash
pytest tests/ -v --tb=short
ruff check . && ruff format --check .
mypy src/rag_eval
python -m rag_eval.gate   # self-check: candidate == committed baseline
```

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs lint, type checks, the full
test suite, the gate self-check, builds the Docker image on every push, and publishes it to
Docker Hub on every push to `main`.

## Project layout

```
src/rag_eval/
  config.py            typed config (frozen dataclasses) + Config.fingerprint()
  ingest/              download → parse → chunk → embed → FAISS/BM25 index
  retrieval/           Retriever protocol + dense/bm25/hybrid/rerank/hyde + registry
  generation/          prompts + LCEL generation chain (citations, abstention)
  evaluation/          eval harness, RAGAS judge, retrieval metrics, ablation report
  gate/                regression gate (provenance/floor/tolerance/ceiling)
  api.py               FastAPI service (/query, /health, /ablation)
  apiclient.py         thin httpx client used by the Streamlit UI
  cli.py               operator CLI (ingest / query / evaluate)
app.py                 Streamlit UI (Ask / Benchmark) — pure HTTP client of the API
scripts/               run_benchmark.py (5-strategy sweep), measure_baseline.py
configs/config.yaml    single source of truth for all tunable parameters
data/eval/eval_set.jsonl  committed gold eval set (59 examples, dev/test split)
results/baseline.json     committed dense-strategy baseline for the regression gate
```

## License

[MIT](LICENSE)
