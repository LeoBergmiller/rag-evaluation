"""Extract clean text from PubMed Central JATS XML full-text articles.

Mirrors the role `parse.extract_text` plays for arXiv PDFs: given one article, return
title + abstract + body text with the reference list dropped, ready for the SAME
chunk -> embed -> index pipeline. JATS references live structurally in `<back>`/
`<ref-list>`, so dropping references is just a matter of not traversing them.

Namespace-robust: PMC JATS core elements are usually un-namespaced, but we match on
local tag names so a namespaced document parses identically.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

logger = logging.getLogger(__name__)

_WS_RE = re.compile(r"[ \t\r\f\v]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")

# JATS block-level elements: emit a line break at their boundaries so adjacent blocks
# (e.g. a <title> and the <p> that follows it) don't fuse into "BackgroundA durable...".
# Inline elements (italic, xref, bold, sup, …) are deliberately absent so their text
# stays tight within a sentence.
_BLOCK_TAGS = frozenset(
    {
        "sec",
        "title",
        "p",
        "abstract",
        "caption",
        "label",
        "list",
        "list-item",
        "table-wrap",
        "fig",
        "disp-quote",
        "boxed-text",
        "def-item",
        "tr",
        "th",
        "td",
    }
)


def _local(tag: str) -> str:
    """Local element name, stripping any ``{namespace}`` prefix."""
    return tag.rsplit("}", 1)[-1]


def _first(element: ET.Element | None, name: str) -> ET.Element | None:
    if element is None:
        return None
    for descendant in element.iter():
        if _local(descendant.tag) == name:
            return descendant
    return None


def _flatten(element: ET.Element, parts: list[str]) -> None:
    """Depth-first serialize, inserting newlines around block-level elements."""
    block = _local(element.tag) in _BLOCK_TAGS
    if block:
        parts.append("\n")
    if element.text:
        parts.append(element.text)
    for child in element:
        _flatten(child, parts)
        if child.tail:
            parts.append(child.tail)
    if block:
        parts.append("\n")


def _text(element: ET.Element | None) -> str:
    """Flatten an element's text content (block-aware) and normalize whitespace."""
    if element is None:
        return ""
    parts: list[str] = []
    _flatten(element, parts)
    lines = [_WS_RE.sub(" ", line).strip() for line in "".join(parts).splitlines()]
    collapsed = "\n".join(line for line in lines if line)
    return _BLANKLINES_RE.sub("\n\n", collapsed).strip()


def parse_jats(xml: str, *, strip_references: bool = True) -> str:
    """Parse a JATS XML string into plain article text (title + abstract + body).

    When ``strip_references`` is False the ``<back>`` matter (references, etc.) is
    appended; by default it is omitted.
    """
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        logger.warning("Failed to parse JATS XML: %s", exc)
        return ""

    article = root if _local(root.tag) == "article" else _first(root, "article")
    if article is None:
        logger.warning("No <article> element found in JATS XML")
        return ""

    front = _first(article, "front")
    parts = [
        _text(_first(front, "article-title")),
        _text(_first(front, "abstract")),
        _text(_first(article, "body")),
    ]
    if not strip_references:
        parts.append(_text(_first(article, "back")))

    return "\n\n".join(part for part in parts if part).strip()


def extract_jats_text(xml_path: Path, *, strip_references: bool = True) -> str:
    """Extract article text from a JATS XML file on disk."""
    return parse_jats(xml_path.read_text(), strip_references=strip_references)
