"""Streamlit demo UI for the rag_eval service. Run: `streamlit run app.py`.

A thin HTTP client of the FastAPI service (see rag_eval.api). Point it at a
running API with the RAG_API_URL env var (default http://localhost:8000).
"""

from __future__ import annotations

import streamlit as st

from rag_eval import apiclient

st.set_page_config(page_title="rag-evaluation", layout="wide")

base_url = st.sidebar.text_input("API URL", value=apiclient.DEFAULT_BASE_URL)

try:
    health = apiclient.health(base_url)
    strategies = health["strategies"]
    st.sidebar.success("API reachable")
    st.sidebar.caption(f"config_fingerprint: `{health['config_fingerprint']}`")
except apiclient.APIError as exc:
    st.sidebar.error(f"API unreachable: {exc}")
    strategies = []

ask_tab, benchmark_tab = st.tabs(["Ask", "Benchmark"])

with ask_tab:
    st.header("Ask a question")
    question = st.text_area(
        "Question", placeholder="What is attention in transformers?"
    )
    strategy = st.selectbox("Strategy", options=strategies) if strategies else None
    k = st.slider("k (chunks retrieved)", min_value=1, max_value=20, value=5)

    if st.button("Submit", disabled=not (question and strategies)):
        try:
            result = apiclient.query(
                base_url, question=question, strategy=strategy, k=k
            )
        except apiclient.APIError as exc:
            st.error(f"Query failed: {exc}")
        else:
            if result["abstained"]:
                st.warning("The model abstained (no answer grounded in the corpus).")
            st.markdown(result["answer"])
            st.caption(
                f"strategy: {result['strategy']} · "
                f"retrieval latency: {result['latency_ms']:.1f} ms"
            )

            cited = set(result["cited_chunk_ids"])
            st.subheader("Retrieved chunks")
            for chunk in result["chunks"]:
                marker = "✅ cited" if chunk["chunk_id"] in cited else ""
                with st.expander(
                    f"[{chunk['chunk_id']}] score={chunk['score']:.4f} {marker}"
                ):
                    st.write(chunk["text"])

with benchmark_tab:
    st.header("Strategy ablation")
    try:
        report = apiclient.ablation(base_url)
    except apiclient.APIError as exc:
        st.error(f"Could not load ablation: {exc}")
        report = None

    if report is None:
        st.info("No ablation report yet. Run `python scripts/run_benchmark.py` first.")
    else:
        st.caption(
            f"config_fingerprint: `{report['config_fingerprint']}` · "
            f"eval_set_hash: `{report['eval_set_hash']}` · "
            f"baseline: {report['baseline_strategy']} · "
            f"n_examples: {report['rows'][0]['n_examples'] if report['rows'] else 0} · "
            f"{report['timestamp']}"
        )
        rows = []
        for row in report["rows"]:
            gate = row["gate_passed"]
            rows.append(
                {
                    "strategy": row["strategy"],
                    "faithfulness": row["faithfulness"]["point"],
                    "context_recall": row["context_recall"]["point"],
                    "recall_at_k": row["recall_at_k"]["point"],
                    "ndcg_at_k": row["ndcg_at_k"]["point"],
                    "p95_latency_ms": row["p95_latency_ms"],
                    "cost_per_query_usd": row["cost_per_query_usd"],
                    "gate": "—" if gate is None else ("✓" if gate else "✗"),
                }
            )
        st.dataframe(rows, hide_index=True)
