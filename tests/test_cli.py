from pathlib import Path

from typer.testing import CliRunner

from rag_eval import cli
from rag_eval.gate.regression import GateCheck, GateResult
from rag_eval.generation.generator import GenerationResult
from rag_eval.ingest.index import IngestManifest
from rag_eval.retrieval.base import RetrievalResult, ScoredChunk

runner = CliRunner()


def _report(**overrides: object) -> object:
    from rag_eval.evaluation.report import MetricCI, StrategyReport

    def _ci(point: float) -> MetricCI:
        return MetricCI(point=point, lo=point, hi=point)

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


def test_ingest_command(monkeypatch) -> None:
    manifest = IngestManifest.create(
        config_fingerprint="defa37d5071c7049",
        embedding_model="BAAI/bge-base-en-v1.5",
        embedding_dimension=768,
        n_papers=10,
        n_chunks=200,
        index_type="flat",
    )
    monkeypatch.setattr(cli, "run_ingest", lambda cfg: manifest)

    result = runner.invoke(cli.app, ["ingest"])

    assert result.exit_code == 0
    assert "10 papers" in result.stdout
    assert "200 chunks" in result.stdout
    assert "defa37d5071c7049" in result.stdout


def test_query_command(monkeypatch) -> None:
    chunk = ScoredChunk(chunk_id="paper::0", text="some passage", score=0.9)
    retrieval_result = RetrievalResult(
        query="what?", chunks=[chunk], latency_ms=12.3, strategy="dense"
    )

    class FakeRetriever:
        name = "dense"

        def retrieve(self, query: str, k: int) -> RetrievalResult:
            return retrieval_result

    class FakeGenerator:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def generate(self, question, chunks):
            return GenerationResult(
                answer="The answer is 42.",
                cited_chunk_ids=["paper::0"],
                abstained=False,
            )

    monkeypatch.setattr(cli, "load_resources", lambda cfg: object())
    monkeypatch.setattr(
        cli, "build_retriever", lambda strategy, cfg, r: FakeRetriever()
    )
    monkeypatch.setattr(cli, "Generator", FakeGenerator)

    result = runner.invoke(cli.app, ["query", "what is the answer?"])

    assert result.exit_code == 0
    assert "The answer is 42." in result.stdout
    assert "paper::0" in result.stdout
    assert "Abstained: False" in result.stdout


def test_evaluate_command_no_gate(monkeypatch) -> None:
    report = _report()

    monkeypatch.setattr(cli, "load_resources", lambda cfg: object())
    monkeypatch.setattr(cli, "default_eval_examples", lambda cfg: [])
    monkeypatch.setattr(
        cli, "evaluate_strategy", lambda strategy, cfg, resources, examples: report
    )

    saved: list[object] = []
    monkeypatch.setattr(
        cli, "save", lambda r, d: saved.append(r) or Path("results/fake.json")
    )

    result = runner.invoke(cli.app, ["evaluate", "--strategy", "dense", "--no-gate"])

    assert result.exit_code == 0
    assert "dense:" in result.stdout
    assert "faithfulness=0.9300" in result.stdout
    assert saved == [report]


def test_evaluate_command_gate_pass(monkeypatch) -> None:
    report = _report()
    gate_result = GateResult(
        passed=True,
        checks=[
            GateCheck(
                name="faithfulness",
                kind="floor",
                passed=True,
                observed=0.93,
                threshold=0.92,
                message="ok",
            )
        ],
        baseline_run_id="baseline",
        candidate_run_id="candidate",
    )

    monkeypatch.setattr(cli, "load_resources", lambda cfg: object())
    monkeypatch.setattr(cli, "default_eval_examples", lambda cfg: [])
    monkeypatch.setattr(
        cli, "evaluate_strategy", lambda strategy, cfg, resources, examples: report
    )
    monkeypatch.setattr(cli, "save", lambda r, d: Path("results/fake.json"))
    monkeypatch.setattr(cli, "load_baseline", lambda: report)
    monkeypatch.setattr(cli, "check_regression", lambda *a, **k: gate_result)

    result = runner.invoke(cli.app, ["evaluate", "--strategy", "dense"])

    assert result.exit_code == 0
    assert "PASSED" in result.stdout


def test_evaluate_command_gate_fail(monkeypatch) -> None:
    report = _report()
    gate_result = GateResult(
        passed=False,
        checks=[
            GateCheck(
                name="faithfulness",
                kind="floor",
                passed=False,
                observed=0.80,
                threshold=0.92,
                message="too low",
            )
        ],
        baseline_run_id="baseline",
        candidate_run_id="candidate",
    )

    monkeypatch.setattr(cli, "load_resources", lambda cfg: object())
    monkeypatch.setattr(cli, "default_eval_examples", lambda cfg: [])
    monkeypatch.setattr(
        cli, "evaluate_strategy", lambda strategy, cfg, resources, examples: report
    )
    monkeypatch.setattr(cli, "save", lambda r, d: Path("results/fake.json"))
    monkeypatch.setattr(cli, "load_baseline", lambda: report)
    monkeypatch.setattr(cli, "check_regression", lambda *a, **k: gate_result)

    result = runner.invoke(cli.app, ["evaluate", "--strategy", "dense"])

    assert result.exit_code != 0
    assert "FAILED" in result.stdout
