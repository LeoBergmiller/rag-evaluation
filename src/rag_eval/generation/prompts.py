"""Prompt loading: templates live as .txt files, never as f-strings in logic."""

from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

PROMPTS_DIR = Path(__file__).parent / "prompts"

_SYSTEM_PROMPT_FILE = "system.txt"
_USER_PROMPT_FILE = "user.txt"
_HYDE_PROMPT_FILE = "hyde.txt"


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()


def build_answer_prompt() -> ChatPromptTemplate:
    system_prompt = load_prompt(_SYSTEM_PROMPT_FILE)
    user_prompt = load_prompt(_USER_PROMPT_FILE)
    return ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", user_prompt)]
    )


def build_hyde_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("human", load_prompt(_HYDE_PROMPT_FILE))])


def prompt_template_hash() -> str:
    """Hash of the prompt template content (a controlled variable for the eval)."""
    digest = hashlib.sha256()
    for name in (_SYSTEM_PROMPT_FILE, _USER_PROMPT_FILE):
        digest.update(load_prompt(name).encode())
    return digest.hexdigest()[:16]
