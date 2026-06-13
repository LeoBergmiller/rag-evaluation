from rag_eval.config import load_config
from rag_eval.evaluation.dataset import EvalDataset, EvalExample


def _make_example(id: str, *, answerable: bool, split: str = "test") -> EvalExample:
    if answerable:
        return EvalExample(
            id=id,
            question=f"question {id}",
            reference_answer=f"answer {id}",
            reference_chunk_ids=[f"paper::{id}"],
            answerable=True,
            split=split,
        )
    return EvalExample(
        id=id,
        question=f"question {id}",
        reference_answer=None,
        reference_chunk_ids=None,
        answerable=False,
        split=split,
    )


def test_load_and_validate_schema() -> None:
    cfg = load_config()
    dataset = EvalDataset.load(cfg.evaluation.eval_set)

    assert len(dataset.examples) > 0
    for example in dataset.examples:
        assert isinstance(example, EvalExample)
        assert example.split in ("dev", "test")


def test_hash_stable() -> None:
    cfg = load_config()

    h1 = EvalDataset.load(cfg.evaluation.eval_set).hash()
    h2 = EvalDataset.load(cfg.evaluation.eval_set).hash()

    assert h1 == h2
    assert len(h1) == 16


def test_dev_test_disjoint() -> None:
    cfg = load_config()
    dataset = EvalDataset.load(cfg.evaluation.eval_set)

    dev_ids = {e.id for e in dataset.dev}
    test_ids = {e.id for e in dataset.test}
    all_ids = {e.id for e in dataset.examples}

    assert dev_ids.isdisjoint(test_ids)
    assert dev_ids | test_ids == all_ids


def test_unanswerable_present() -> None:
    cfg = load_config()
    dataset = EvalDataset.load(cfg.evaluation.eval_set)

    assert any(not e.answerable for e in dataset.examples)


def test_reference_consistency() -> None:
    cfg = load_config()
    dataset = EvalDataset.load(cfg.evaluation.eval_set)

    for example in dataset.examples:
        if example.answerable:
            assert example.reference_answer is not None
            assert example.reference_chunk_ids
        else:
            assert example.reference_answer is None
            assert example.reference_chunk_ids is None


def test_assign_splits_deterministic() -> None:
    examples = [_make_example(f"ex{i}", answerable=True) for i in range(10)]

    result1 = EvalDataset.assign_splits(examples, dev_frac=0.4, seed=0)
    result2 = EvalDataset.assign_splits(examples, dev_frac=0.4, seed=0)

    splits1 = {e.id: e.split for e in result1}
    splits2 = {e.id: e.split for e in result2}
    assert splits1 == splits2
    assert sum(1 for e in result1 if e.split == "dev") == 4
