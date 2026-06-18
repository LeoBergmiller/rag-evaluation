"""Compute judge-vs-human agreement on the rubric scorer.

Usage:
    python scripts/judge_agreement.py [--labels data/eval/judge_labels.jsonl]

Reads your hand-labeled file, runs score_example() on the same inputs, then
computes Cohen's kappa for correctness and completeness (ordinal 0/1/2) and
both percent agreement and kappa for citation_valid (bool).

Prerequisites:
  - OPENAI_API_KEY in environment (or .env)
  - data/eval/judge_labels.jsonl exists and contains at least a few rows
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score

load_dotenv()

# resolve package root so the script works whether run from repo root or scripts/
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from rag_eval.config import load_config  # noqa: E402
from rag_eval.evaluation.rubric import RubricScore, score_example  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


def _load_labels(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


@app.command()
def main(
    labels: Path = typer.Option(
        Path("data/eval/judge_labels.jsonl"),
        help="Path to your hand-labeled JSONL file.",
    ),
) -> None:
    if not labels.exists():
        typer.echo(f"Labels file not found: {labels}", err=True)
        raise typer.Exit(1)

    rows = _load_labels(labels)
    if not rows:
        typer.echo("Labels file is empty.", err=True)
        raise typer.Exit(1)

    cfg = load_config()

    human_correctness: list[int] = []
    human_completeness: list[int] = []
    human_citation: list[int] = []
    judge_correctness: list[int] = []
    judge_completeness: list[int] = []
    judge_citation: list[int] = []
    skipped = 0

    typer.echo(f"Scoring {len(rows)} examples with the rubric judge...")
    for i, row in enumerate(rows, 1):
        score: RubricScore | None = score_example(
            question=row["question"],
            reference_answer=row["reference_answer"],
            model_answer=row["model_answer"],
            cited_chunk_ids=row["cited_chunk_ids"],
            chunks_by_id=row["cited_chunk_texts"],
            cfg=cfg,
        )
        if score is None:
            logger.warning("Judge returned None for example %s — skipping", row["example_id"])
            skipped += 1
            continue

        human_correctness.append(int(row["correctness"]))
        human_completeness.append(int(row["completeness"]))
        human_citation.append(1 if row["citation_valid"] else 0)
        judge_correctness.append(score.correctness)
        judge_completeness.append(score.completeness)
        judge_citation.append(1 if score.citation_valid else 0)

        if i % 10 == 0:
            typer.echo(f"  {i}/{len(rows)} done")

    n = len(human_correctness)
    if n < 2:
        typer.echo(f"Too few comparable examples ({n}) — cannot compute kappa.", err=True)
        raise typer.Exit(1)

    kappa_correctness = cohen_kappa_score(human_correctness, judge_correctness, weights="linear")
    kappa_completeness = cohen_kappa_score(human_completeness, judge_completeness, weights="linear")
    kappa_citation = cohen_kappa_score(human_citation, judge_citation)
    pct_citation = sum(h == j for h, j in zip(human_citation, judge_citation)) / n * 100

    typer.echo(f"\n=== Judge-vs-Human Agreement (n={n}, skipped={skipped}) ===\n")
    typer.echo(f"{'metric':<28}  {'Cohen κ':>9}  {'% agree':>9}")
    typer.echo("-" * 52)
    typer.echo(f"{'correctness (0-2, linear κ)':<28}  {kappa_correctness:>9.3f}  {'—':>9}")
    typer.echo(f"{'completeness (0-2, linear κ)':<28}  {kappa_completeness:>9.3f}  {'—':>9}")
    typer.echo(f"{'citation_valid (bool)':<28}  {kappa_citation:>9.3f}  {pct_citation:>8.1f}%")


if __name__ == "__main__":
    app()
