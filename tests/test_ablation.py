from pathlib import Path

import pytest

from rag_eval.evaluation.ablation import (
    AblationReport,
    build_ablation,
    format_markdown,
    save_ablation,
)
from rag_eval.evaluation.report import MetricCI, StrategyReport


def _ci(point: float) -> MetricCI:
    return MetricCI(point=point, lo=point, hi=point)


def _report(**overrides: object) -> StrategyReport:
    fields: dict[str, object] = {
        "run_id": "candidate",
        "timestamp": "20260101T000000Z",
        "strategy": "dense",
        "n_examples": 59,
        "config_fingerprint": "defa37d5071c7049",
        "eval_set_hash": "403ff26e0eff39a0",
        "prompt_template_hash": "de4b5a5833ff80ce",
        "faithfulness": _ci(0.93),
        "answer_relevancy": _ci(0.84),
        "context_precision": _ci(0.71),
        "context_recall": _ci(0.86),
        "recall_at_k": _ci(0.75),
        "precision_at_k": _ci(0.15),
        "mrr": _ci(0.47),
        "ndcg_at_k": _ci(0.54),
        "p95_latency_ms": 37.0,
        "cost_per_query_usd": 0.0118,
        "abstention_accuracy": 0.97,
        "abstention_rate": 0.12,
        "examples": [],
    }
    fields.update(overrides)
    return StrategyReport(**fields)


def test_build_ablation_rows_and_baseline() -> None:
    dense = _report(strategy="dense", run_id="dense-run")
    hybrid = _report(
        strategy="hybrid",
        run_id="hybrid-run",
        faithfulness=_ci(0.95),
        context_recall=_ci(0.88),
    )

    ablation = build_ablation(
        [dense, hybrid],
        baseline_strategy="dense",
        gate_results={"dense": True, "hybrid": False},
    )

    assert isinstance(ablation, AblationReport)
    assert ablation.baseline_strategy == "dense"
    assert [row.strategy for row in ablation.rows] == ["dense", "hybrid"]

    dense_row, hybrid_row = ablation.rows
    assert dense_row.gate_passed is True
    assert hybrid_row.gate_passed is False
    assert hybrid_row.faithfulness.point == 0.95
    assert hybrid_row.context_recall.point == 0.88


def test_build_ablation_rejects_divergent_provenance() -> None:
    dense = _report(strategy="dense")
    other = _report(strategy="hybrid", config_fingerprint="deadbeef00000000")

    with pytest.raises(ValueError, match="divergent config_fingerprint"):
        build_ablation([dense, other])


def test_format_markdown_contains_table_and_baseline_marker() -> None:
    dense = _report(strategy="dense", run_id="dense-run")
    hybrid = _report(strategy="hybrid", run_id="hybrid-run", faithfulness=_ci(0.95))

    ablation = build_ablation(
        [dense, hybrid],
        baseline_strategy="dense",
        gate_results={"dense": True, "hybrid": True},
    )

    markdown = format_markdown(ablation)

    assert "| strategy |" in markdown
    assert "dense (baseline)" in markdown
    assert "hybrid" in markdown
    assert "(+0.0200)" in markdown
    assert "✓" in markdown


def test_save_ablation_writes_json_and_markdown(tmp_path: Path) -> None:
    dense = _report(strategy="dense")
    ablation = build_ablation([dense], baseline_strategy="dense")

    json_path, md_path = save_ablation(ablation, tmp_path)

    assert json_path.exists()
    assert md_path.exists()

    loaded = AblationReport.model_validate_json(json_path.read_text())
    assert loaded == ablation
