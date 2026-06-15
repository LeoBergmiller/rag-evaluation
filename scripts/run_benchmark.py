"""Run the full strategy sweep and write a cross-strategy ablation report.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --strategies dense,hybrid
    python scripts/run_benchmark.py --no-gate
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rag_eval.config import load_config
from rag_eval.evaluation.ablation import build_ablation, format_markdown, save_ablation
from rag_eval.evaluation.harness import default_eval_examples, evaluate_strategy
from rag_eval.evaluation.report import StrategyReport, save
from rag_eval.gate.regression import check_regression, load_baseline
from rag_eval.retrieval.registry import load_resources, registered_strategies

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full strategy benchmark sweep."
    )
    parser.add_argument("--strategies", type=str, default=None)
    parser.add_argument("--gate", dest="gate", action="store_true", default=True)
    parser.add_argument("--no-gate", dest="gate", action="store_false")
    args = parser.parse_args()

    strategies = (
        args.strategies.split(",") if args.strategies else registered_strategies()
    )

    cfg = load_config()
    resources = load_resources(cfg)
    examples = default_eval_examples(cfg)

    baseline = load_baseline() if args.gate else None

    reports: list[StrategyReport] = []
    gate_results: dict[str, bool] = {}
    for strategy in strategies:
        report = evaluate_strategy(strategy, cfg, resources, examples)
        path = save(report, Path("results"))
        print(f"{strategy}: wrote {path}")
        reports.append(report)

        if baseline is not None:
            result = check_regression(report, baseline, cfg.evaluation.gate)
            gate_results[strategy] = result.passed

    ablation = build_ablation(
        reports, baseline_strategy="dense", gate_results=gate_results
    )
    json_path, md_path = save_ablation(ablation, Path("results"))
    print(f"\nWrote {json_path}\nWrote {md_path}\n")
    print(format_markdown(ablation))


if __name__ == "__main__":
    main()
