"""Read-only reviewer for a gold eval set — makes hand-verification practical.

Walks each example one at a time showing the question, the reference answer, and the
FULL text of every cited chunk, so you can confirm the answer is actually grounded
without cross-referencing chunks.jsonl by hand. Unanswerable examples show the passage
they were topically grounded on, flagged for rewrite (they must be plausible in-domain
questions NOT answerable from that passage).

Defaults to read-only (you edit the .jsonl in your editor). Pass --edit for an
interactive mode that rewrites/deletes examples and safely writes the file back.

Usage:
    python scripts/review_eval_set.py                              # medical set, all
    python scripts/review_eval_set.py --filter unanswerable        # just the rewrites
    python scripts/review_eval_set.py --filter unanswerable --edit # guided rewrite
    python scripts/review_eval_set.py --split test --start 12
    python scripts/review_eval_set.py --config configs/config.yaml # the arXiv set
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from rag_eval.config import load_config  # noqa: E402
from rag_eval.evaluation.dataset import EvalDataset, EvalExample  # noqa: E402
from rag_eval.ingest.pipeline import load_chunks  # noqa: E402

_WIDTH = 88


def _hr(char: str = "─") -> None:
    print(char * _WIDTH)


def _wrap(text: str, indent: int = 2) -> str:
    prefix = " " * indent
    return textwrap.fill(
        text, width=_WIDTH - indent, initial_indent=prefix, subsequent_indent=prefix
    )


def _source_chunk_id(example_id: str) -> str | None:
    """Recover the chunk an example was synthesized from (encoded in its id).

    ids look like "answerable::3::PMC13098181::19"; the chunk id is everything after
    the first two "::"-separated fields (chunk ids themselves contain "::").
    """
    parts = example_id.split("::")
    return "::".join(parts[2:]) if len(parts) > 2 else None


def _print_chunk(chunk_id: str, chunks_by_id: dict[str, str]) -> None:
    print(f"  [{chunk_id}]:")
    print(_wrap(chunks_by_id.get(chunk_id, "(text not in index)"), indent=4))


def _review_one(
    example: EvalExample, i: int, total: int, chunks: dict[str, str]
) -> None:
    _hr("═")
    tag = "ANSWERABLE" if example.answerable else "UNANSWERABLE — REWRITE"
    print(f"  Example {i}/{total}  [{example.id}]  split={example.split}  {tag}")
    _hr()

    print("\nQUESTION:")
    print(_wrap(example.question))

    if example.answerable:
        print("\nREFERENCE ANSWER:")
        print(_wrap(example.reference_answer or "(none)"))
        print("\nCITED CHUNK(S) — verify the answer is stated/entailed here:")
        for chunk_id in example.reference_chunk_ids or []:
            _print_chunk(chunk_id, chunks)
        print("\n  CHECK: answer grounded in the chunk? question specific & clear?")
    else:
        print("\nSOURCE PASSAGE (topical grounding only — the question must NOT be")
        print("answerable from it):")
        source = _source_chunk_id(example.id)
        if source:
            _print_chunk(source, chunks)
        print(
            "\n  REWRITE: replace with a plausible IN-DOMAIN medical question that this\n"
            "  passage does NOT answer (avoid ML-framed questions). Keep answerable=false,\n"
            "  reference_answer=null, reference_chunk_ids=null."
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only gold eval set reviewer.")
    parser.add_argument("--config", type=str, default="configs/config_med.yaml")
    parser.add_argument(
        "--filter", choices=["all", "answerable", "unanswerable"], default="all"
    )
    parser.add_argument("--split", choices=["all", "dev", "test"], default="all")
    parser.add_argument("--start", type=int, default=1, help="1-indexed start example")
    parser.add_argument(
        "--edit",
        action="store_true",
        help="interactively rewrite/delete examples (safely writes back the .jsonl)",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    dataset = EvalDataset.load(cfg.evaluation.eval_set)
    chunks = {chunk.chunk_id: chunk.text for chunk in load_chunks(cfg.corpus.index_dir)}

    # Keyed by id + an explicit order list so edits/deletes rewrite the whole file
    # in place without index bookkeeping; unedited lines re-serialize identically.
    order = [e.id for e in dataset.examples]
    by_id = {e.id: e for e in dataset.examples}

    def _selected(eid: str) -> bool:
        example = by_id[eid]
        if args.filter == "answerable" and not example.answerable:
            return False
        if args.filter == "unanswerable" and example.answerable:
            return False
        return args.split == "all" or example.split == args.split

    def _save() -> None:
        EvalDataset(examples=[by_id[eid] for eid in order]).save(
            cfg.evaluation.eval_set
        )

    review_ids = [eid for eid in order if _selected(eid)]
    total = len(review_ids)
    n_unanswerable = sum(1 for eid in review_ids if not by_id[eid].answerable)
    mode = "EDIT (writes back)" if args.edit else "read-only"
    print(
        f"\nReviewing {cfg.evaluation.eval_set} — {total} examples "
        f"({total - n_unanswerable} answerable, {n_unanswerable} unanswerable) — {mode}."
    )
    if not args.edit:
        print("Edit the .jsonl in your editor. [Enter] advances, Ctrl-C stops.")
    print()

    try:
        for i, eid in enumerate(review_ids[args.start - 1 :], start=args.start):
            if eid not in by_id:  # deleted earlier this session
                continue
            _review_one(by_id[eid], i, total, chunks)
            if not args.edit:
                input("  [Enter] next · Ctrl-C to stop: ")
                continue
            cmd = input("  [Enter]=keep · e=edit · d=delete · q=quit: ").strip().lower()
            if cmd == "q":
                break
            if cmd == "d":
                order.remove(eid)
                del by_id[eid]
                _save()
                print("  ✗ deleted (saved)")
            elif cmd == "e":
                updates: dict[str, str] = {}
                new_q = input("    new question (Enter=keep): ").strip()
                if new_q:
                    updates["question"] = new_q
                if by_id[eid].answerable:
                    new_a = input("    new reference answer (Enter=keep): ").strip()
                    if new_a:
                        updates["reference_answer"] = new_a
                if updates:
                    by_id[eid] = by_id[eid].model_copy(update=updates)
                    _save()
                    print("  ✓ saved")
                else:
                    print("  (no change)")
    except KeyboardInterrupt:
        print("\n\nStopped — progress saved. Re-run with --start N to resume.")

    if args.edit:
        final = EvalDataset.load(cfg.evaluation.eval_set)
        print(
            f"\nFinal: {len(final.examples)} examples parse OK → {cfg.evaluation.eval_set}"
        )


if __name__ == "__main__":
    main()
