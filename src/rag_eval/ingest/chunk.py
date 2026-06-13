"""Token-based recursive chunking with parent-document ids."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Chunk(BaseModel):
    """A single chunk of a paper, ready for embedding/indexing."""

    chunk_id: str
    text: str
    parent_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def chunk_text(
    text: str,
    paper_id: str,
    *,
    length_function: Callable[[str], int],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Split a paper's text into chunks with a shared parent (paper-level) id.

    `parent_id` is the paper id for every chunk produced here. PDF text lacks
    reliable section boundaries, so finer section-level parents are a documented
    future enhancement rather than part of the MVP.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
    )

    pieces = splitter.split_text(text)
    chunks = [
        Chunk(
            chunk_id=f"{paper_id}::{i}",
            text=piece,
            parent_id=paper_id,
            metadata={"parent_id": paper_id, "chunk_index": i},
        )
        for i, piece in enumerate(pieces)
    ]
    logger.debug("Chunked paper %s into %d chunks", paper_id, len(chunks))
    return chunks
