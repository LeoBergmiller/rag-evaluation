"""Cross-strategy ablation report: aggregates StrategyReports into one comparison."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from rag_eval.evaluation.report import MetricCI, StrategyReport

_DELTA_METRICS = ("faithfulness", "context_recall", "recall_at_k", "ndcg_at_k")

_PROVENANCE_FIELDS = ("config_fingerprint", "eval_set_hash", "prompt_template_hash")


class AblationRow(BaseModel):
    strategy: str
    n_examples: int
    run_id: str
    gate_passed: bool | None

    faithfulness: MetricCI
    answer_relevancy: MetricCI
    context_precision: MetricCI
    context_recall: MetricCI

    recall_at_k: MetricCI
    precision_at_k: MetricCI
    mrr: MetricCI
    ndcg_at_k: MetricCI

    p95_latency_ms: float
    cost_per_query_usd: float
    abstention_accuracy: float
    abstention_rate: float


class AblationReport(BaseModel):
    run_id: str
    timestamp: str
    config_fingerprint: str
    eval_set_hash: str
    prompt_template_hash: str
    baseline_strategy: str
    rows: list[AblationRow]


def _row_from_report(report: StrategyReport, gate_passed: bool | None) -> AblationRow:
    return AblationRow(
        strategy=report.strategy,
        n_examples=report.n_examples,
        run_id=report.run_id,
        gate_passed=gate_passed,
        faithfulness=report.faithfulness,
        answer_relevancy=report.answer_relevancy,
        context_precision=report.context_precision,
        context_recall=report.context_recall,
        recall_at_k=report.recall_at_k,
        precision_at_k=report.precision_at_k,
        mrr=report.mrr,
        ndcg_at_k=report.ndcg_at_k,
        p95_latency_ms=report.p95_latency_ms,
        cost_per_query_usd=report.cost_per_query_usd,
        abstention_accuracy=report.abstention_accuracy,
        abstention_rate=report.abstention_rate,
    )


def build_ablation(
    reports: list[StrategyReport],
    baseline_strategy: str = "dense",
    gate_results: dict[str, bool] | None = None,
) -> AblationReport:
    """Aggregate per-strategy reports into one ablation report.

    All reports must share the same provenance triple (config_fingerprint,
    eval_set_hash, prompt_template_hash) -- comparing across configs is meaningless.
    """
    if not reports:
        raise ValueError("build_ablation requires at least one report")

    first = reports[0]
    for report in reports[1:]:
        for field_name in _PROVENANCE_FIELDS:
            if getattr(report, field_name) != getattr(first, field_name):
                raise ValueError(
                    f"Reports have divergent {field_name}: "
                    f"{getattr(first, field_name)!r} (strategy={first.strategy!r}) != "
                    f"{getattr(report, field_name)!r} (strategy={report.strategy!r})"
                )

    gate_results = gate_results or {}
    rows = [
        _row_from_report(report, gate_results.get(report.strategy))
        for report in reports
    ]

    return AblationReport(
        run_id=StrategyReport.new_run_id(),
        timestamp=StrategyReport.now_timestamp(),
        config_fingerprint=first.config_fingerprint,
        eval_set_hash=first.eval_set_hash,
        prompt_template_hash=first.prompt_template_hash,
        baseline_strategy=baseline_strategy,
        rows=rows,
    )


def format_markdown(report: AblationReport) -> str:
    baseline_row = next(
        (row for row in report.rows if row.strategy == report.baseline_strategy), None
    )

    lines = [
        f"# Ablation report ({report.timestamp})",
        "",
        f"config_fingerprint=`{report.config_fingerprint}` "
        f"eval_set_hash=`{report.eval_set_hash}` "
        f"prompt_template_hash=`{report.prompt_template_hash}` "
        f"n_examples={report.rows[0].n_examples if report.rows else 0}",
        "",
        "| strategy | faithfulness (Δ) | context_recall (Δ) | recall_at_k (Δ) | "
        "ndcg_at_k (Δ) | p95_latency_ms | cost_per_query_usd | gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in report.rows:
        cells = [
            f"{row.strategy}" + (" (baseline)" if row is baseline_row else ""),
        ]
        for metric in _DELTA_METRICS:
            point = getattr(row, metric).point
            cell = f"{point:.4f}"
            if baseline_row is not None and row is not baseline_row:
                delta = point - getattr(baseline_row, metric).point
                cell += f" ({delta:+.4f})"
            cells.append(cell)
        cells.append(f"{row.p95_latency_ms:.2f}")
        cells.append(f"{row.cost_per_query_usd:.4f}")
        if row.gate_passed is None:
            cells.append("-")
        else:
            cells.append("✓" if row.gate_passed else "✗")
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def save_ablation(report: AblationReport, results_dir: Path) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{report.timestamp}_ablation_{report.run_id[:8]}"
    json_path = results_dir / f"{stem}.json"
    md_path = results_dir / f"{stem}.md"

    json_path.write_text(report.model_dump_json(indent=2))
    md_path.write_text(format_markdown(report))

    return json_path, md_path
