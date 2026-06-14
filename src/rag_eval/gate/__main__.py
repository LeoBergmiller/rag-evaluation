"""CLI entry point: `python -m rag_eval.gate [--candidate PATH] [--baseline PATH]`.

With no arguments, runs a self-check of the committed baseline against itself —
an offline check that the baseline still clears its own floors and ceilings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_eval.config import load_config
from rag_eval.gate.regression import check_regression, format_result, load_baseline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the regression gate.")
    parser.add_argument("--baseline", type=Path, default=Path("results/baseline.json"))
    parser.add_argument("--candidate", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    baseline = load_baseline(args.baseline)
    candidate = (
        load_baseline(args.candidate) if args.candidate is not None else baseline
    )

    result = check_regression(candidate, baseline, cfg.evaluation.gate)
    print(format_result(result))
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
