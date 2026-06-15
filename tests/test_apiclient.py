import httpx
import pytest

from rag_eval import apiclient


def _patch_transport(monkeypatch, handler) -> None:
    transport = httpx.MockTransport(handler)

    def fake_get(url, **kwargs):
        kwargs.pop("timeout", None)
        with httpx.Client(transport=transport) as client:
            return client.get(url, **kwargs)

    def fake_post(url, **kwargs):
        kwargs.pop("timeout", None)
        with httpx.Client(transport=transport) as client:
            return client.post(url, **kwargs)

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(httpx, "post", fake_post)


def test_health_success(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok", "strategies": ["dense"]})

    _patch_transport(monkeypatch, handler)
    result = apiclient.health("http://test")

    assert result["status"] == "ok"
    assert result["strategies"] == ["dense"]


def test_query_success(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/query"
        return httpx.Response(200, json={"answer": "hi", "abstained": False})

    _patch_transport(monkeypatch, handler)
    result = apiclient.query("http://test", question="q", strategy="dense", k=3)

    assert result["answer"] == "hi"


def test_ablation_returns_none_on_404(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "none"})

    _patch_transport(monkeypatch, handler)

    assert apiclient.ablation("http://test") is None


def test_ablation_success(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"rows": []})

    _patch_transport(monkeypatch, handler)

    assert apiclient.ablation("http://test") == {"rows": []}


def test_non_2xx_raises(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    _patch_transport(monkeypatch, handler)

    with pytest.raises(apiclient.APIError):
        apiclient.health("http://test")


def test_connection_error_raises(monkeypatch) -> None:
    def fake_get(url, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", fake_get)

    with pytest.raises(apiclient.APIError):
        apiclient.health("http://test")
