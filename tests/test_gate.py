from pathlib import Path

from rag_eval.config import load_config
from rag_eval.evaluation.report import MetricCI, StrategyReport
from rag_eval.gate.regression import check_regression, load_baseline

CFG = load_config()
GATE = CFG.evaluation.gate


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


def test_clean_pass() -> None:
    baseline = _report()
    candidate = _report(run_id="candidate")

    result = check_regression(candidate, baseline, GATE)

    assert result.passed is True


def test_floor_breach() -> None:
    baseline = _report()
    candidate = _report(faithfulness=_ci(0.90))

    result = check_regression(candidate, baseline, GATE)

    assert result.passed is False
    failing = [c for c in result.checks if not c.passed]
    assert any(c.kind == "floor" and c.name == "faithfulness" for c in failing)


def test_tolerance_regression() -> None:
    baseline = _report()
    candidate = _report(
        context_recall=_ci(0.86 - GATE.tolerance["context_recall"] - 0.01)
    )

    result = check_regression(candidate, baseline, GATE)

    assert result.passed is False
    failing = [c for c in result.checks if not c.passed]
    assert any(c.kind == "tolerance" and c.name == "context_recall" for c in failing)


def test_operational_ceiling_breach() -> None:
    baseline = _report()
    ceiling = GATE.operational_ceilings["p95_latency_ms"]
    candidate = _report(p95_latency_ms=ceiling + 1)

    result = check_regression(candidate, baseline, GATE)

    assert result.passed is False
    failing = [c for c in result.checks if not c.passed]
    assert any(c.kind == "ceiling" and c.name == "p95_latency_ms" for c in failing)


def test_provenance_mismatch() -> None:
    baseline = _report()
    candidate = _report(config_fingerprint="deadbeef00000000")

    result = check_regression(candidate, baseline, GATE)

    assert result.passed is False
    failing = [c for c in result.checks if not c.passed]
    assert any(
        c.kind == "provenance" and c.name == "config_fingerprint" for c in failing
    )


def test_committed_baseline_self_check() -> None:
    baseline = load_baseline(Path("results/baseline.json"))

    result = check_regression(baseline, baseline, GATE)

    assert result.passed is True
