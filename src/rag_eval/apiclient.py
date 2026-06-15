"""Thin HTTP client for the rag_eval FastAPI service, used by the Streamlit UI."""

from __future__ import annotations

import os

import httpx

DEFAULT_BASE_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")

_DEFAULT_TIMEOUT = 10.0
_QUERY_TIMEOUT = 120.0


class APIError(Exception):
    """Raised when the API is unreachable or returns an error status."""


def health(base_url: str = DEFAULT_BASE_URL) -> dict:
    return _get(base_url, "/health", timeout=_DEFAULT_TIMEOUT)


def query(
    base_url: str = DEFAULT_BASE_URL,
    *,
    question: str,
    strategy: str | None = None,
    k: int | None = None,
) -> dict:
    payload: dict = {"question": question}
    if strategy is not None:
        payload["strategy"] = strategy
    if k is not None:
        payload["k"] = k

    try:
        response = httpx.post(f"{base_url}/query", json=payload, timeout=_QUERY_TIMEOUT)
    except httpx.HTTPError as exc:
        raise APIError(f"Could not reach API at {base_url}: {exc}") from exc

    if response.status_code != 200:
        raise APIError(f"API returned {response.status_code}: {response.text}")
    return response.json()


def ablation(base_url: str = DEFAULT_BASE_URL) -> dict | None:
    try:
        response = httpx.get(f"{base_url}/ablation", timeout=_DEFAULT_TIMEOUT)
    except httpx.HTTPError as exc:
        raise APIError(f"Could not reach API at {base_url}: {exc}") from exc

    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise APIError(f"API returned {response.status_code}: {response.text}")
    return response.json()


def _get(base_url: str, path: str, *, timeout: float) -> dict:
    try:
        response = httpx.get(f"{base_url}{path}", timeout=timeout)
    except httpx.HTTPError as exc:
        raise APIError(f"Could not reach API at {base_url}: {exc}") from exc

    if response.status_code != 200:
        raise APIError(f"API returned {response.status_code}: {response.text}")
    return response.json()
