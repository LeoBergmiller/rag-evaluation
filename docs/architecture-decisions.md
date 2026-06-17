# Architecture Decisions

### D1 — Vector store: FAISS IndexFlatIP (exact) over ANN / pgvector

- **Options:** FAISS IndexFlatIP (exact brute-force inner product); FAISS IVF/HNSW (approximate); pgvector (Postgres extension); no separate index (embed at query time)
- **Choice:** FAISS IndexFlatIP
- **Rationale:** Approximate search introduces its own recall error, which would be an uncontrolled variable in a retrieval strategy comparison. The benchmark isolates strategy quality differences — ANN approximation error would contaminate that signal. FAISS runs in-process (no server), and exact search over a 300-paper corpus adds negligible latency.

### D2 — Embedding model: BAAI/bge-base-en-v1.5, fixed as a controlled variable

- **Options:** bge-small-en-v1.5 (faster, lower quality); bge-base-en-v1.5; bge-large-en-v1.5 (slow on CPU); OpenAI text-embedding-3 (API cost); per-strategy embeddings
- **Choice:** BAAI/bge-base-en-v1.5, identical across all five strategies
- **Rationale:** The embedding model must not vary across the ablation — it is a controlled variable, not a treatment. Differences in strategy scores then reflect only retrieval logic. bge-base balances quality and CPU-only inference speed at this corpus scale.

### D3 — Chunking: token-recursive 512/64, paper-level parent pointers

- **Options:** Character-based splitting; fixed token windows; token-based recursive (LangChain RecursiveCharacterTextSplitter); semantic chunking; sentence-level
- **Choice:** Token-recursive, 512 tokens / 64 overlap, with paper-level `parent_id` stored on every chunk
- **Rationale:** Token count is the natural unit for embedding model context windows. Recursive splitting preserves paragraph/sentence boundaries within the budget. Paper-level parent pointers enable parent-document retrieval as a future enhancement without re-indexing. Chunk size and overlap are part of `Config.fingerprint()`, so the provenance chain catches accidental changes.

### D4 — Retriever interface: `@runtime_checkable Protocol` with composition

- **Options:** Abstract base class with subclasses; plain Protocol; plain functions; dataclass dispatch on strategy name
- **Choice:** `@runtime_checkable Protocol` with composition — rerank, hybrid, and HyDE wrap a base `Retriever` rather than re-implementing retrieval
- **Rationale:** Protocol allows structural typing without inheritance coupling. Composition for composites means the benchmark loop never branches on strategy type — every strategy satisfies the same `retrieve(query, k) → RetrievalResult` contract. Runtime checkability enables `isinstance` checks in tests without a concrete base class.

### D5 — Generator/judge model separation: two different families, temperature 0

- **Options:** Same model for both generation and judgment; different providers; human evaluation only
- **Choice:** Claude / Anthropic (family A) generates answers; GPT-4o / OpenAI (family B) judges via RAGAS; both at temperature 0
- **Rationale:** Self-preference bias is a documented LLM-as-judge failure mode — a model tends to rate its own outputs higher. A different model family for the judge removes this confounder. Temperature 0 on both sides ensures determinism across eval runs so metric variance reflects only RAGAS sampling, not temperature noise.

### D6 — Regression gate: provenance + floor + tolerance + operational ceiling

- **Options:** Manual review; simple threshold on one metric; multi-metric threshold; multi-check gate with provenance
- **Choice:** Four check types in sequence — provenance (config fingerprint / eval-set hash / prompt-template hash), quality floors (absolute minima), no-regression tolerances (Δ from committed baseline), operational ceilings (p95 latency, cost per query); tolerances calibrated from measured run-to-run noise (≈ 2 × std)
- **Rationale:** Provenance checks prevent comparing runs produced under different configs. Floors catch absolute quality collapse. Tolerances catch regressions while absorbing RAGAS judge stochasticity. Operational ceilings surface the latency/cost penalty of strategies that add a second model pass (hyde, rerank) — separating "passes quality" from "operationally viable" rather than masking the trade-off.

### D7 — Framework boundary: retrieval/indexing core is framework-free; LangChain for generation, chunking, and evaluation

- **Options:** Everything in LangChain; everything in LlamaIndex; LangChain for generation only; fully framework-free
- **Choice:** FAISS / rank-bm25 / sentence-transformers for indexing and retrieval search (no framework); LangChain for generation chain, chunking (text splitters), HyDE LLM call, and evaluation harness; LlamaIndex not used
- **Rationale:** The retrieval scoring core is what the benchmark measures — keeping it framework-free ensures results reflect the retrieval algorithm, not LangChain's retriever abstractions. LangChain is appropriate as a thin utility layer for generation (prompt templates, LCEL chains) and chunking (text splitters), where its abstractions don't interfere with measurement.

### D8 — Dependency pin: LangChain 0.3.x + ragas 0.2.x exact versions

- **Options:** Unpinned (latest); minor-version ranges; exact pins; separate virtual envs per experiment
- **Choice:** Exact-version pins for the entire LangChain 0.3 stack and ragas 0.2.15
- **Rationale:** LangChain 1.x + ragas 0.4.x is currently broken — ragas 0.4.3 hard-imports `langchain_community.chat_models.vertexai`, a path the 1.x-era community package removed, causing `import ragas` to fail on any provider. The 0.3 line + ragas 0.2.x is the mature, mutually-compatible combination. Exact pins ensure reproducibility across machines. Revisit when ragas ships LangChain 1.x support (ragas issue #2741 / PR #2739).

### D9 — CI scope: build-only; Docker Hub image published manually

- **Options:** Auto-push to Docker Hub on every merge to main (requires write-scoped token as repo secret); build-only CI + manual publish; no Docker in CI
- **Choice:** Build-only CI (`docker-build` job, `push: false`); Docker Hub image published manually
- **Rationale:** This repo has no deploy pipeline consuming the registry, so auto-publish adds a long-lived write-scoped token as a repo secret, a standing CI failure mode (red badge on registry or secret issues unrelated to code quality), and a second image build per merge for a convenience that won't be used. The manually-published public image at `leobergmiller/rag-evaluation:latest` and the README `docker pull` quickstart give a portfolio reviewer everything they need. Supersedes the `docker-push` CI job added in the initial Docker integration.
