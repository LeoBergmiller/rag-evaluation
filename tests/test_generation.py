import os

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag_eval.generation.generator import Generator, format_context
from rag_eval.generation.prompts import build_answer_prompt, prompt_template_hash
from rag_eval.retrieval.base import ScoredChunk

CHUNKS = [
    ScoredChunk(
        chunk_id="paper::0",
        text="The model uses a transformer architecture.",
        score=0.9,
        parent_id="paper",
    ),
    ScoredChunk(
        chunk_id="paper::1",
        text="Training used 8 GPUs for 3 days.",
        score=0.8,
        parent_id="paper",
    ),
]


def test_build_answer_prompt_variables() -> None:
    prompt = build_answer_prompt()

    assert set(prompt.input_variables) == {"context", "question"}


def test_format_context_includes_chunk_ids() -> None:
    context = format_context(CHUNKS)

    assert "[paper::0]" in context
    assert "[paper::1]" in context
    assert "transformer architecture" in context


def test_generate_parses_valid_citations() -> None:
    fake_model = FakeListChatModel(
        responses=["The model uses a transformer architecture [paper::0]."]
    )
    generator = Generator(_dummy_generation_cfg(), chat_model=fake_model)

    result = generator.generate("What architecture is used?", CHUNKS)

    assert result.cited_chunk_ids == ["paper::0"]
    assert result.abstained is False


def test_generate_filters_unretrieved_citations() -> None:
    fake_model = FakeListChatModel(responses=["According to [paper::99], it's great."])
    generator = Generator(_dummy_generation_cfg(), chat_model=fake_model)

    result = generator.generate("What is great?", CHUNKS)

    assert result.cited_chunk_ids == []


def test_generate_detects_abstention() -> None:
    fake_model = FakeListChatModel(responses=["I don't know."])
    generator = Generator(_dummy_generation_cfg(), chat_model=fake_model)

    result = generator.generate("What is the meaning of life?", CHUNKS)

    assert result.abstained is True
    assert result.cited_chunk_ids == []


def test_prompt_template_hash_stable() -> None:
    h1 = prompt_template_hash()
    h2 = prompt_template_hash()

    assert h1 == h2
    assert len(h1) == 16


def _dummy_generation_cfg():
    from rag_eval.config import load_config

    return load_config().generation


@pytest.mark.skipif(
    os.environ.get("RAG_EVAL_RUN_LLM") != "1",
    reason="set RAG_EVAL_RUN_LLM=1 to run against a real LLM provider",
)
def test_generate_real_llm() -> None:
    from rag_eval.config import load_config

    cfg = load_config()
    generator = Generator(cfg.generation)

    in_corpus = generator.generate("What architecture is used?", CHUNKS)
    assert in_corpus.cited_chunk_ids
    assert not in_corpus.abstained

    out_of_corpus = generator.generate("What is the capital of the moon?", CHUNKS)
    assert out_of_corpus.abstained
