"""Regression gate logic: provenance, floor, tolerance, and ceiling checks."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from rag_eval.config import GateConfig
from rag_eval.evaluation.report import StrategyReport

CheckKind = Literal["provenance", "floor", "tolerance", "ceiling"]

_PROVENANCE_FIELDS = ("config_fingerprint", "eval_set_hash", "prompt_template_hash")


class GateCheck(BaseModel):
    name: str
    kind: CheckKind
    passed: bool
    observed: float | str | None
    threshold: float | str | None
    message: str


class GateResult(BaseModel):
    passed: bool
    checks: list[GateCheck]
    baseline_run_id: str
    candidate_run_id: str


def _metric_point(report: StrategyReport, name: str) -> float:
    value = getattr(report, name)
    if hasattr(value, "point"):
        return float(value.point)
    return float(value)


def check_regression(
    candidate: StrategyReport, baseline: StrategyReport, gate: GateConfig
) -> GateResult:
    checks: list[GateCheck] = []

    for field_name in _PROVENANCE_FIELDS:
        candidate_value = getattr(candidate, field_name)
        baseline_value = getattr(baseline, field_name)
        passed = candidate_value == baseline_value
        checks.append(
            GateCheck(
                name=field_name,
                kind="provenance",
                passed=passed,
                observed=candidate_value,
                threshold=baseline_value,
                message=(
                    f"{field_name} matches baseline"
                    if passed
                    else f"{field_name} {candidate_value!r} != baseline {baseline_value!r}"
                ),
            )
        )

    for name, floor in gate.floors.items():
        observed = _metric_point(candidate, name)
        passed = observed >= floor
        checks.append(
            GateCheck(
                name=name,
                kind="floor",
                passed=passed,
                observed=observed,
                threshold=floor,
                message=f"{name}={observed:.4f} {'>=' if passed else '<'} floor {floor}",
            )
        )

    for name, tolerance in gate.tolerance.items():
        observed = _metric_point(candidate, name)
        baseline_value = _metric_point(baseline, name)
        threshold = baseline_value - tolerance
        passed = observed >= threshold
        checks.append(
            GateCheck(
                name=name,
                kind="tolerance",
                passed=passed,
                observed=observed,
                threshold=threshold,
                message=(
                    f"{name}={observed:.4f} {'>=' if passed else '<'} "
                    f"baseline {baseline_value:.4f} - tolerance {tolerance} = {threshold:.4f}"
                ),
            )
        )

    for name, ceiling in gate.operational_ceilings.items():
        if ceiling is None:
            continue
        observed = _metric_point(candidate, name)
        passed = observed <= ceiling
        checks.append(
            GateCheck(
                name=name,
                kind="ceiling",
                passed=passed,
                observed=observed,
                threshold=ceiling,
                message=f"{name}={observed:.4f} {'<=' if passed else '>'} ceiling {ceiling}",
            )
        )

    return GateResult(
        passed=all(check.passed for check in checks),
        checks=checks,
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
    )


def load_baseline(path: Path = Path("results/baseline.json")) -> StrategyReport:
    return StrategyReport.model_validate_json(path.read_text())


def format_result(result: GateResult) -> str:
    lines = [
        f"Gate {'PASSED' if result.passed else 'FAILED'} "
        f"(candidate={result.candidate_run_id[:8]} baseline={result.baseline_run_id[:8]})"
    ]
    for check in result.checks:
        status = "PASS" if check.passed else "FAIL"
        lines.append(f"  [{status}] {check.kind:>10} {check.name}: {check.message}")
    return "\n".join(lines)
