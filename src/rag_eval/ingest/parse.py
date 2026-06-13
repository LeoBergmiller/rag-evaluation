"""Extract and clean text from downloaded paper PDFs."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)

# Matches a standalone "References" / "Bibliography" heading line, allowing for
# numbering (e.g. "7. References") and case variation.
_REFERENCES_HEADING = re.compile(
    r"^\s*(?:\d+\.?\s*)?(references|bibliography)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def extract_text(pdf_path: Path, *, strip_references: bool = True) -> str:
    """Extract full text from a PDF, optionally dropping the references section."""
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages)

    if strip_references:
        text = _strip_references(text)

    return text


def _strip_references(text: str) -> str:
    """Drop everything from the last References/Bibliography heading onward."""
    matches = list(_REFERENCES_HEADING.finditer(text))
    if not matches:
        return text

    last_match = matches[-1]
    return text[: last_match.start()].rstrip()
