from streamlit.testing.v1 import AppTest

from rag_eval import apiclient

_HEALTH = {
    "status": "ok",
    "strategies": ["dense", "hybrid"],
    "config_fingerprint": "defa37d5071c7049",
}

_QUERY = {
    "answer": "Attention weights tokens by relevance. [paper::0]",
    "abstained": False,
    "cited_chunk_ids": ["paper::0"],
    "strategy": "dense",
    "latency_ms": 12.3,
    "chunks": [
        {
            "chunk_id": "paper::0",
            "score": 0.91,
            "text": "some passage",
            "parent_id": "paper",
        }
    ],
}

_ABLATION = {
    "run_id": "abcd1234",
    "timestamp": "20260101T000000Z",
    "config_fingerprint": "defa37d5071c7049",
    "eval_set_hash": "403ff26e0eff39a0",
    "prompt_template_hash": "de4b5a5833ff80ce",
    "baseline_strategy": "dense",
    "rows": [
        {
            "strategy": "dense",
            "n_examples": 59,
            "run_id": "dense-run",
            "gate_passed": True,
            "faithfulness": {"point": 0.93, "lo": 0.9, "hi": 0.95},
            "answer_relevancy": {"point": 0.84, "lo": 0.8, "hi": 0.88},
            "context_precision": {"point": 0.71, "lo": 0.68, "hi": 0.74},
            "context_recall": {"point": 0.86, "lo": 0.83, "hi": 0.89},
            "recall_at_k": {"point": 0.75, "lo": 0.72, "hi": 0.78},
            "precision_at_k": {"point": 0.15, "lo": 0.13, "hi": 0.17},
            "mrr": {"point": 0.47, "lo": 0.44, "hi": 0.5},
            "ndcg_at_k": {"point": 0.54, "lo": 0.51, "hi": 0.57},
            "p95_latency_ms": 37.0,
            "cost_per_query_usd": 0.0118,
            "abstention_accuracy": 0.97,
            "abstention_rate": 0.12,
        }
    ],
}


def _patch(monkeypatch) -> None:
    monkeypatch.setattr(apiclient, "health", lambda base_url: _HEALTH)
    monkeypatch.setattr(apiclient, "query", lambda base_url, **kwargs: _QUERY)
    monkeypatch.setattr(apiclient, "ablation", lambda base_url: _ABLATION)


def test_app_renders_health_and_strategies(monkeypatch) -> None:
    _patch(monkeypatch)

    at = AppTest.from_file("app.py").run()

    assert not at.exception
    assert "dense" in at.selectbox[0].options


def test_app_query_renders_answer(monkeypatch) -> None:
    _patch(monkeypatch)

    at = AppTest.from_file("app.py").run()
    at.text_area[0].set_value("What is attention?").run()
    at.button[0].click().run()

    assert not at.exception
    markdowns = " ".join(md.value for md in at.markdown)
    assert "Attention weights tokens" in markdowns


def test_app_benchmark_renders_without_error(monkeypatch) -> None:
    _patch(monkeypatch)

    at = AppTest.from_file("app.py").run()

    assert not at.exception
    assert len(at.dataframe) >= 1


def test_app_handles_unreachable_api(monkeypatch) -> None:
    def _raise(base_url):
        raise apiclient.APIError("refused")

    monkeypatch.setattr(apiclient, "health", _raise)
    monkeypatch.setattr(apiclient, "ablation", _raise)

    at = AppTest.from_file("app.py").run()

    assert not at.exception
