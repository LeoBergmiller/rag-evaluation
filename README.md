# rag-evaluation

[![CI](https://github.com/LeoBergmiller/rag-evaluation/actions/workflows/ci.yml/badge.svg)](https://github.com/LeoBergmiller/rag-evaluation/actions/workflows/ci.yml)
[![Docker Hub](https://img.shields.io/docker/v/leobergmiller/rag-evaluation?label=Docker%20Hub)](https://hub.docker.com/r/leobergmiller/rag-evaluation)

A production-style RAG evaluation harness that implements and **benchmarks five retrieval
strategies** behind one swappable interface, evaluated with
[RAGAS](https://github.com/explodinggradients/ragas) + a custom retrieval-metrics harness,
gated by an automated regression check, and served via a FastAPI + Streamlit demo. It runs
over **two independent corpora** — an arXiv ML/AI paper corpus and an open-access
[PubMed Central](https://www.ncbi.nlm.nih.gov/pmc/) medical-literature corpus — to show the
*same harness, unchanged, benchmarks a second domain and the strategy ranking reproduces*.

The goal isn't "a RAG demo" — it's the harness around it: a controlled experiment that lets
you swap retrieval strategies via config, measure quality/latency/cost with confidence
intervals, fail CI if a change regresses quality below a calibrated tolerance, and drop in a
whole new domain by adding a config + an ingest adapter — with **zero changes** to retrieval,
generation, evaluation, or the gate. (The medical corpus is a retrieval-**evaluation** testbed
over published *literature* — not a clinical tool; abstention on unanswerable questions is
measured as a first-class safety property.)

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

59-example gold eval set of single-document-grounded questions (plus unanswerable abstention
probes; multi-hop / multi-document QA is out of scope by design — see decision D10), `dense` as
baseline (see
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

**Finding — the fancier strategies don't earn their cost on this workload.** `rerank` does post
the single best quality (faithfulness +0.0154, nDCG@k +0.2835 vs the dense baseline), but at
~28× the latency (1105 ms vs 40 ms) — and plain `bm25` captures nearly all of that retrieval
gain (nDCG@k 0.8129 vs rerank's 0.8254) while being the *cheapest* strategy in the table
(~5 ms p95) and passing the gate. `hyde`'s extra LLM pass buys no quality at all here (faithfulness
−0.0052) for ~310× the latency. On this corpus — specific, single-document-grounded QA over ML
papers — a cross-encoder and a hypothetical-document step are complexity the results don't
justify: the cheap lexical/hybrid retrievers are on the Pareto front, and the expensive layers
sit behind it. The value of building all five is being *able to show that*, rather than assuming
the sophisticated strategy is better.

## Cross-domain generalization (arXiv → medical)

The headline claim — *the harness, not the demo* — is made concrete by running the **identical,
unchanged** pipeline over a second, unrelated domain: ~500 open-access PubMed Central full-text
articles on immune-checkpoint inhibitors in cancer (JATS XML, not PDF), with its own 100-example
gold set. Both corpora share the **same `config_fingerprint` (`defa37d5071c7049`)** — the
controlled variables (embedding, chunking, generation) are byte-identical; only the corpus and
eval set differ. See [results/cross_domain.md](results/cross_domain.md).

| corpus | strategy | context_recall | recall@k | nDCG@k | p95 (ms) | gate |
| --- | --- | --- | --- | --- | --- | --- |
| medical | dense (baseline) | 0.803 | 0.692 | 0.584 | 42 | ✓ |
| medical | bm25 | 0.885 | 0.885 | 0.821 | 108 | ✓ |
| medical | hybrid | 0.910 | 0.865 | 0.734 | 101 | ✓ |
| medical | rerank | 0.962 | 0.962 | 0.907 | 1316 | ✗ |
| medical | hyde | 0.644 | 0.481 | 0.374 | 12301 | ✗ |

**What reproduces:**
- **The leading 3 by nDCG@k are identical across domains** (`rerank > bm25 > hybrid`) — and cheap
  `bm25` holds a top spot in both, so the "cheap retrieval stays competitive" finding generalizes.
- **`rerank` and `hyde` fail the gate on the latency ceiling in *both* domains** — the
  earn-the-complexity result holds: the fancier strategies don't justify their cost on either corpus.

**What diverges (the interesting part):** **HyDE** collapses on medical — nDCG@k 0.62 (arXiv) →
0.37 (medical), the worst strategy on the corpus. Query transformation that helps modestly on one
domain can actively hurt on another; generalization is not free, and the harness is what lets you
*catch* that rather than assume it.

> Scope notes (documented in [decision D12](docs/architecture-decisions.md)): to fit an API budget,
> the medical run judges `faithfulness` + `context_recall` (the RAGAS metrics in this comparison),
> and its gate tolerances are reused from the arXiv calibration (judge noise is a harness property,
> identical across corpora). Batch-1 judge timeouts reduced faithfulness sample coverage on two
> strategies, so faithfulness is reported but excluded from the medical gate's fine tolerance.

## Quickstart

```bash
git clone https://github.com/LeoBergmiller/rag-evaluation.git
cd rag-evaluation
pip install -e ".[dev]"
cp .env.example .env   # ANTHROPIC_API_KEY, OPENAI_API_KEY (+ NCBI_API_KEY for the medical corpus)
```

**arXiv corpus (default):**
```bash
python -m rag_eval.cli ingest                 # download → chunk → embed → build FAISS + BM25 index
python -m rag_eval.cli query "What is attention?" --strategy hybrid
python -m rag_eval.cli evaluate --strategy dense --no-gate   # writes results/*_dense_*.json
python scripts/run_benchmark.py               # full 5-strategy sweep + ablation report
```

**Medical corpus (PubMed Central):** every entry point selects a corpus via the `RAG_CONFIG`
env var (default = arXiv). The medical config uses a separate index, eval set, and baseline:
```bash
RAG_CONFIG=configs/config_med.yaml python -m rag_eval.cli ingest        # fetch PMC + build index
RAG_CONFIG=configs/config_med.yaml python -m rag_eval.cli query "How do PD-1 inhibitors work?" -s hybrid
python scripts/run_benchmark.py --config configs/config_med.yaml        # medical 5-strategy sweep
python scripts/cross_domain_report.py --out results/cross_domain.md     # arXiv vs medical table
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
test suite, the gate self-check, and builds the Docker image on every push.

### Judge-vs-human agreement

The custom rubric judge (GPT-4o, [`evaluation/rubric.py`](src/rag_eval/evaluation/rubric.py))
was validated against 40 blind human labels on the same three-dimension rubric (dense strategy run):

| dimension | Cohen κ | % agreement |
| --- | --- | --- |
| correctness (0–2) | 0.70 | 85% |
| completeness (0–2) | 0.74 | 90% |
| citation_valid (bool) | 0.33 | 77.5% |

Correctness and completeness show substantial agreement (κ ≥ 0.70); the judge skews slightly lenient
on correctness — it accepts answers with correct key facts but loose framing that a human marked
partial. Citation_valid agreement is lower (κ = 0.33) not because the judge is unreliable but because
the rubric criterion is underspecified: "plausibly supports" conflates "topically related" with "the
cited chunk actually contains the specific claim." On 5 of 10 citation_valid disagreements the judge
was *stricter* than the human — it correctly flagged factually-right answers whose cited chunks do not
contain the specific claim, a real grounding failure the human missed. The ambiguous rubric wording is
a documented limitation; the κ reflects genuine interpretive ambiguity, not unreliable scoring.

**Update — criterion sharpened.** The low κ has two compounding causes. (1) The rubric prompt scored a
citation valid if the chunk *"plausibly supports"* the claim, which conflates *topically related* with
*actually contains the specific claim*. The [rubric prompt](src/rag_eval/generation/prompts/rubric_system.txt)
has since been sharpened to require the cited chunk to **state or directly entail** the specific claim,
with worked valid/invalid examples. (2) With only 8 negative labels in 40, Cohen κ is highly
base-rate-sensitive, so κ = 0.33 overstates unreliability relative to the 77.5% raw agreement. The figures
in the table were measured under the *old* wording. Because the sharpened criterion **moves the bar** (it
would reclassify the ~5 lenient human "valid" labels on genuine grounding failures), those 40 labels
cannot honestly re-validate the new wording — a faithful re-measurement requires re-labeling the citation
judgments under the sharpened definition. No new κ is claimed until that re-label is done; the committed
figures stand as the pre-sharpening measurement.

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

## Security

[docs/security.md](docs/security.md) maps the OWASP LLM Top 10 (2025) to this project — which
risks bind on the design (LLM01 indirect prompt injection, LLM05 output handling), which are the
core of what the eval harness measures (LLM09 misinformation), and which are N/A or low by scope
(static trusted arXiv corpus, text-only output, no agency).

## License

[MIT](LICENSE)
