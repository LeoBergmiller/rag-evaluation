# Security Threat Model — OWASP LLM Top 10 (2025)

This document maps the [OWASP Top 10 for LLM Applications (2025)](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
to **this project** and states, per risk, its relevance given the actual design. The
goal is to reason about the threat model explicitly — not to bolt guardrails onto a
system whose scope does not need them.

## System context (what determines the risk surface)

- **The corpus is trusted and static.** Content is arXiv ML/AI papers (`cs.CL`, `cs.LG`,
  `cs.AI`), fetched once into `data/raw/`, which is immutable by project rule. There is
  no user-supplied or web-crawled content in the index.
- **Output is text, and it stops at text.** The generated answer is returned as JSON by
  the FastAPI `/query` endpoint ([`api.py`](../src/rag_eval/api.py)) and rendered in the
  Streamlit UI via `st.markdown(...)` (answer) and `st.write(...)` (chunk text)
  ([`app.py`](../app.py)). It is **not** interpolated into SQL, a shell, `eval`, a
  templating engine with raw-HTML enabled, or any tool/function call.
- **No agency.** There are no tools, function-calling, code execution, or side-effecting
  actions available to the model. The system retrieves, generates one grounded answer,
  and returns it.
- **This is an offline benchmark harness with a demo surface**, not a hardened
  multi-tenant production deployment. Several mitigations below are stated as "would
  matter if …" precisely because the current scope does not exercise them.

## Risk-by-risk mapping

### LLM01 — Prompt Injection · *Relevant (indirect), low residual risk on this corpus*
Retrieved chunk text is untrusted-by-principle content that reaches the model. Current
defenses are **prompt-level, not enforced**: the system prompt
([`system.txt`](../src/rag_eval/generation/prompts/system.txt)) instructs the model to
answer *only* from context and to reply exactly `I don't know` when context is
insufficient, and retrieved passages are rendered as `[chunk_id] text` data lines
([`generator.py` `format_context`](../src/rag_eval/generation/generator.py)) so context
is presented as data, not as instructions.
- **Why low here:** the corpus is trusted arXiv papers, so an adversarial "ignore your
  instructions" payload sitting inside a chunk is not part of the threat model — no
  attacker controls the indexed content.
- **What would change it:** if the corpus became untrusted (user uploads, web crawl,
  wiki), indirect prompt injection (LLM01) becomes a first-class risk and prompt
  instruction alone would be insufficient — it would warrant an input-side guardrail
  (injection/anomaly detection on retrieved text) and stronger data/instruction
  separation. That is deliberately **not** built now; see the DO-NOT-ADD scope decisions
  in the project audit.

### LLM02 — Sensitive Information Disclosure · *Low / N/A for this corpus*
The index contains only already-public, published arXiv papers. There is no PII, no
secrets, and no private user data in retrievable content, so there is nothing sensitive
for retrieval or generation to leak. Author names in papers are public bibliographic
data, not PII in the disclosure sense.
- **What would change it:** an enterprise/internal corpus (support tickets, HR docs,
  customer data) would make PII redaction and access control on retrieval a real
  requirement. N/A here by virtue of the corpus, not by mitigation.

### LLM03 — Supply Chain · *Partially addressed*
The LLM/eval stack is pinned to exact versions (LangChain 0.3.x, `ragas==0.2.15`,
`faiss-cpu`, `sentence-transformers`, etc.) in [`pyproject.toml`](../pyproject.toml),
which bounds dependency drift and makes builds reproducible (see decision **D8**). Model
weights (BGE embed/rerank) are pulled from HuggingFace by name; the generator/judge are
hosted APIs (Anthropic, OpenAI). Residual: no hash-pinning of transitive deps or model
artifacts — acceptable for a portfolio benchmark, would be tightened (lockfile with
hashes, vendored/attested models) for a regulated production deployment.

### LLM04 — Data & Model Poisoning · *Low / N/A*
The corpus is fetched once from arXiv and then immutable; there is no continuous
ingestion, user-contributed data, or fine-tuning loop that an attacker could poison. The
embedding and reranker models are stock published checkpoints used as-is. No training or
model updates occur in this system.

### LLM05 — Improper Output Handling · *Relevant, low residual risk as designed*
This is the second risk that genuinely applies. The mitigation is **architectural, not
incidental**: model output never crosses a trust boundary into an interpreter. It is
returned as a JSON string field and rendered as Markdown with raw HTML disabled
(Streamlit `st.markdown` does not execute HTML/script unless `unsafe_allow_html=True`,
which this code does not set) and as plain text via `st.write`. There is no SQL, shell,
`eval`, filesystem, or downstream-tool sink for the output to be injected into.
- **What would change it:** if a downstream consumer rendered the answer as raw HTML, fed
  it to a SQL/code executor, or passed it to a tool-calling agent, output encoding /
  validation at that boundary would become mandatory. The current text-only sink is why
  no output sanitizer is warranted today.

### LLM06 — Excessive Agency · *N/A*
The model has no tools, no function-calling, no code execution, and no ability to take
actions in any external system. Its only capability is producing one text answer from
retrieved context. There is no agency to be excessive.

### LLM07 — System Prompt Leakage · *Low impact*
The system prompt could in principle be surfaced by a crafted question, but it contains
no secrets — only the answer-only-from-context / cite / abstain rules, which are
documented publicly in this repo. Leakage carries no confidentiality cost here.

### LLM08 — Vector & Embedding Weaknesses · *Low / N/A*
The FAISS index is single-tenant, local, and built from the trusted corpus; there is no
cross-user embedding store, no per-user data partitioning to violate, and no
user-controlled content that could be embedded to poison neighbors. Embedding-inversion
concerns (recovering source text from vectors) are moot because the source text is
already public.

### LLM09 — Misinformation · *Relevant — this is what the whole project measures*
Ungrounded/hallucinated answers are the core failure mode the system is built to detect
and suppress: grounded-only generation with mandatory citations and an explicit
`I don't know` abstention path ([`system.txt`](../src/rag_eval/generation/prompts/system.txt)),
plus an evaluation harness that scores **faithfulness** (RAGAS) and **citation validity**
(custom rubric) and a **regression gate with a hard faithfulness floor (0.92)** that fails
CI on hallucination regressions. Misinformation risk is not eliminated (LLMs can still err
within grounded context) but it is measured, bounded, and gated rather than ignored.

### LLM10 — Unbounded Consumption · *Low, partially bounded*
Generation is capped (`max_tokens=512`), retrieval breadth is bounded (`candidate_k=50`,
`top_k=5`), and per-query cost/latency are measured and enforced by the gate's operational
ceilings. The demo `/query` API has **no authentication or rate limiting** — acceptable
for a local/portfolio demo, but it would need request limits and auth before any public
exposure. Called out honestly as a demo-scope gap, not a design claim.

## Summary

| Risk | Relevance here | Basis |
| --- | --- | --- |
| LLM01 Prompt Injection | Relevant (indirect), low residual | Trusted static corpus; prompt-level defense + `[id]` data-line separation |
| LLM02 Sensitive Info | N/A | Public arXiv content only; no PII/secrets indexed |
| LLM03 Supply Chain | Partially addressed | Exact-pinned deps (D8); no artifact hash-pinning |
| LLM04 Data/Model Poisoning | N/A | Immutable one-shot corpus; stock model checkpoints |
| LLM05 Improper Output Handling | Relevant, low residual | Text-only sink; Markdown w/ HTML off; no SQL/code/tool boundary |
| LLM06 Excessive Agency | N/A | No tools, no actions |
| LLM07 System Prompt Leakage | Low impact | Prompt holds no secrets |
| LLM08 Vector/Embedding | N/A | Single-tenant local index; public source text |
| LLM09 Misinformation | Core, measured & gated | Grounding + citations + abstention + faithfulness floor in CI |
| LLM10 Unbounded Consumption | Low, partial | Token/retrieval caps + cost ceilings; demo API lacks auth/rate-limit |

**Bottom line:** two risks genuinely bind on this design — LLM01 (indirect injection) and
LLM05 (output handling) — and both are held low by the trusted-corpus + text-only-output
architecture rather than by dedicated guardrail code. LLM09 (misinformation) is the risk
the project actively engineers against and measures. The remainder are N/A or low by
virtue of scope (static public corpus, no agency, single-tenant), and are documented here
so those are visible as reasoned scope decisions, not oversights.

## Red-team probes

[`scripts/redteam.py`](../scripts/redteam.py) is a small **demonstration** battery (not a
production guardrail) that turns this threat model into runnable probes. Against a running
API it sends adversarial prompts and checks that observed behavior matches the safe behavior
claimed above:

| probe | OWASP | what it checks |
| --- | --- | --- |
| `inject-direct` / `inject-indirect` | LLM01 | a query-borne or quoted "ignore your rules" instruction does not override the grounded system prompt (canary never emitted) |
| `jailbreak-persona` | LLM01 | a persona/jailbreak framing does not force ungrounded output — the model still abstains on out-of-corpus questions |
| `unanswerable-domain` / `unanswerable-false-premise` | LLM09 | out-of-corpus and false-premise questions produce `I don't know`, not fabrication |
| `pii-exfiltration` | LLM02 | no PII is emitted (none exists in the corpus) |
| `system-prompt-leak` | LLM07 | the system prompt is not reproduced verbatim |
| `output-xss` | LLM05 | *sink probe* — an HTML/script payload is rendered inertly by the UI (`st.markdown`, raw HTML disabled) and never reaches a code/SQL/tool sink; no automated verdict on model text |
| `unbounded-output` | LLM10 | a "repeat 10,000×" request stays bounded by the `max_tokens=512` cap |

Run it with the API up:

```bash
uvicorn rag_eval.api:app
python scripts/redteam.py --out results/redteam.md
```

This exercises the real serving path (retrieve → generate → API response) rather than mocking
it, so a probe result reflects the deployed behavior. It is a demonstration that the threat
model can be *tested*, not a claim that the system is hardened for an untrusted corpus.
