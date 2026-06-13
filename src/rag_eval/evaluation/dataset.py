"""Gold eval set: schema, JSONL IO, content hash, and dev/test split assignment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class EvalExample(BaseModel):
    id: str
    question: str
    reference_answer: str | None
    reference_chunk_ids: list[str] | None
    answerable: bool = True
    split: Literal["dev", "test"]


class EvalDataset(BaseModel):
    examples: list[EvalExample]

    @classmethod
    def load(cls, path: Path) -> EvalDataset:
        examples = []
        with path.open("r") as f:
            for line in f:
                examples.append(EvalExample.model_validate(json.loads(line)))
        return cls(examples=examples)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for example in self.examples:
                f.write(example.model_dump_json() + "\n")

    def hash(self) -> str:
        """Content hash over examples (sorted by id) — ties a baseline to its eval set."""
        canon = [
            example.model_dump()
            for example in sorted(self.examples, key=lambda e: e.id)
        ]
        digest = hashlib.sha256(json.dumps(canon, sort_keys=True).encode())
        return digest.hexdigest()[:16]

    @property
    def dev(self) -> list[EvalExample]:
        return [e for e in self.examples if e.split == "dev"]

    @property
    def test(self) -> list[EvalExample]:
        return [e for e in self.examples if e.split == "test"]

    @staticmethod
    def assign_splits(
        examples: list[EvalExample], dev_frac: float, seed: int
    ) -> list[EvalExample]:
        """Deterministically assign each example to "dev" or "test".

        Order is a seeded shuffle keyed by `(seed, example.id)` so the assignment
        is stable across runs regardless of input order.
        """
        order = sorted(
            range(len(examples)),
            key=lambda i: hashlib.sha256(
                f"{seed}:{examples[i].id}".encode()
            ).hexdigest(),
        )
        n_dev = round(len(examples) * dev_frac)
        dev_indices = set(order[:n_dev])
        return [
            example.model_copy(update={"split": "dev" if i in dev_indices else "test"})
            for i, example in enumerate(examples)
        ]
