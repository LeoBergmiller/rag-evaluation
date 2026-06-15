"""Operator CLI: `python -m rag_eval.cli {ingest|query|evaluate}`."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from rag_eval.config import load_config
from rag_eval.evaluation.harness import default_eval_examples, evaluate_strategy
from rag_eval.evaluation.report import save
from rag_eval.gate.regression import check_regression, format_result, load_baseline
from rag_eval.generation.generator import Generator
from rag_eval.ingest.pipeline import run_ingest
from rag_eval.retrieval.registry import (
    build_retriever,
    load_resources,
    registered_strategies,
)

logging.basicConfig(level=logging.INFO)

app = typer.Typer()


@app.command()
def ingest(config: Path | None = None) -> None:
    """Download -> chunk -> embed -> index the corpus."""
    cfg = load_config(config)
    manifest = run_ingest(cfg)
    typer.echo(
        f"Ingested {manifest.n_papers} papers into {manifest.n_chunks} chunks\n"
        f"embedding_model={manifest.embedding_model} "
        f"embedding_dimension={manifest.embedding_dimension}\n"
        f"index_type={manifest.index_type} "
        f"config_fingerprint={manifest.config_fingerprint}"
    )


@app.command()
def query(
    question: str,
    strategy: str = typer.Option(None, "--strategy", "-s"),
    k: int = typer.Option(None, "--k"),
    config: Path | None = None,
) -> None:
    """Run a single query against a retrieval strategy and generate an answer."""
    cfg = load_config(config)
    strategy = strategy or cfg.retrieval.strategy
    k = k or cfg.retrieval.top_k

    resources = load_resources(cfg)
    retriever = build_retriever(strategy, cfg, resources)
    result = retriever.retrieve(question, k)

    generator = Generator(cfg.generation)
    gen_result = generator.generate(question, result.chunks)

    typer.echo(f"Strategy: {strategy} (latency {result.latency_ms:.1f} ms)")
    typer.echo("Retrieved chunks:")
    for chunk in result.chunks:
        typer.echo(f"  [{chunk.chunk_id}] score={chunk.score:.4f}")
    typer.echo(f"\nAnswer: {gen_result.answer}")
    typer.echo(f"Cited chunks: {gen_result.cited_chunk_ids}")
    typer.echo(f"Abstained: {gen_result.abstained}")


@app.command()
def evaluate(
    strategy: list[str] = typer.Option(None, "--strategy", "-s"),
    gate: bool = typer.Option(True, "--gate/--no-gate"),
    config: Path | None = None,
) -> None:
    """Run the eval harness for one or more strategies, save reports, and gate them."""
    cfg = load_config(config)
    strategies = strategy or registered_strategies()

    resources = load_resources(cfg)
    examples = default_eval_examples(cfg)

    baseline = load_baseline() if gate else None
    any_failed = False

    for strategy_name in strategies:
        report = evaluate_strategy(strategy_name, cfg, resources, examples)
        path = save(report, Path("results"))
        typer.echo(
            f"{strategy_name}: faithfulness={report.faithfulness.point:.4f} "
            f"context_recall={report.context_recall.point:.4f} "
            f"recall_at_k={report.recall_at_k.point:.4f} "
            f"p95_latency_ms={report.p95_latency_ms:.2f} "
            f"cost_per_query_usd={report.cost_per_query_usd:.4f} "
            f"-> {path}"
        )

        if baseline is not None:
            result = check_regression(report, baseline, cfg.evaluation.gate)
            typer.echo(format_result(result))
            if not result.passed:
                any_failed = True

    if any_failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
