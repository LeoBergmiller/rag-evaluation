"""Interactive terminal labeling tool for the rubric worksheet.

Usage:
    python scripts/label_examples.py

Shows each example one at a time (question → reference answer → model answer →
cited chunks), prompts for correctness (0/1/2), completeness (0/1/2),
citation_valid (y/n), and an optional rationale, then writes immediately to
data/eval/judge_labels.jsonl. Resume safely — already-labeled examples are
skipped on restart.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

_WORKSHEET = Path("data/eval/labeling_worksheet.jsonl")
_LABELS = Path("data/eval/judge_labels.jsonl")
_WIDTH = 88


def _hr(char: str = "─") -> None:
    print(char * _WIDTH)


def _wrap(text: str, indent: int = 2) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=_WIDTH - indent, initial_indent=prefix, subsequent_indent=prefix)


def _prompt_int(prompt: str, choices: tuple[int, ...]) -> int:
    valid = "/".join(str(c) for c in choices)
    while True:
        raw = input(f"  {prompt} [{valid}]: ").strip()
        try:
            val = int(raw)
            if val in choices:
                return val
        except ValueError:
            pass
        print(f"  → enter {valid}")


def _prompt_bool(prompt: str) -> bool:
    while True:
        raw = input(f"  {prompt} [y/n]: ").strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  → enter y or n")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    worksheet = _load_jsonl(_WORKSHEET)
    if not worksheet:
        print(f"Worksheet not found or empty: {_WORKSHEET}")
        return

    existing = {row["example_id"] for row in _load_jsonl(_LABELS)}
    todo = [row for row in worksheet if row["example_id"] not in existing]
    total = len(worksheet)
    done = len(existing)

    print(f"\nLabeling worksheet: {total} examples total, {done} already labeled, {len(todo)} remaining.")
    if not todo:
        print("All examples labeled. Run scripts/judge_agreement.py to compute agreement.")
        return
    print("Press Ctrl-C at any time to stop — progress is saved after each example.\n")

    with _LABELS.open("a") as out:
        for i, row in enumerate(todo, start=done + 1):
            _hr("═")
            print(f"  Example {i}/{total}  [{row['example_id']}]")
            _hr()

            print("\nQUESTION:")
            print(_wrap(row["question"]))

            print("\nREFERENCE ANSWER:")
            print(_wrap(row["reference_answer"]))

            print("\nMODEL ANSWER:")
            for line in row["model_answer"].splitlines():
                print(_wrap(line) if line.strip() else "")

            if row["cited_chunk_texts"]:
                print("\nCITED CHUNK TEXTS:")
                for cid, text in row["cited_chunk_texts"].items():
                    print(f"  [{cid}]:")
                    print(_wrap(text, indent=4))
            else:
                print("\nCITED CHUNK TEXTS: (none)")

            print()
            _hr()
            print("  CORRECTNESS vs reference answer:")
            print("    0 = factually wrong/contradicts reference")
            print("    1 = partially correct, notable errors or omissions")
            print("    2 = correct and consistent with reference")
            correctness = _prompt_int("correctness", (0, 1, 2))

            print("  COMPLETENESS vs reference answer:")
            print("    0 = misses most key points")
            print("    1 = covers some but misses at least one important element")
            print("    2 = covers all key points")
            completeness = _prompt_int("completeness", (0, 1, 2))

            print("  CITATION VALID — do [chunk_id] citations support the claims?")
            print("    y = every citation supports its claim  |  n = at least one doesn't (or none cited)")
            citation_valid = _prompt_bool("citation_valid")

            rationale = input("  rationale (optional, press Enter to skip): ").strip()

            label = {
                **row,
                "correctness": correctness,
                "completeness": completeness,
                "citation_valid": citation_valid,
                "rationale": rationale,
            }
            out.write(json.dumps(label) + "\n")
            out.flush()
            print(f"  ✓ saved ({i}/{total} done)\n")

    print("\nAll done! Run: python scripts/judge_agreement.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped. Run again to resume from where you left off.")
