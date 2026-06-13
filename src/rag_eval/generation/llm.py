"""Provider-agnostic chat model construction.

Shared by the generator (this package) and the judge (step 7) so the
generate-vs-judge model family swap is config-only.
"""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


def build_chat_model(
    provider: str, model: str, temperature: float, max_tokens: int
) -> BaseChatModel:
    return init_chat_model(
        model,
        model_provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
    )
