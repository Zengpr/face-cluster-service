"""API contract tests using FastAPI TestClient — uses stubbed embedder."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


def _stub_embedder(monkeypatch) -> None:
    """Replace FaceEmbedder.get with a deterministic stub returning
    normalized random vectors so the test suite never touches a 500MB
    onnx model. This is the standard pattern for keeping API tests fast
    and model-agnostic in CI.
    """
    from app.services import face_embedder as fe

    class _Stub:
        def embed_image(self, image_rgb):
            v = np.random.default_rng(abs(hash(image_rgb.tobytes())))
            v = v.standard_normal(512).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-9
            return v, 1

    monkeypatch.setattr(fe.FaceEmbedder, "get", classmethod(lambda cls: _Stub()))


@pytest.fixture()
def client(monkeypatch) -> "TestClient":
    _stub_embedder(monkeypatch)
    # Disable redis dependency during tests by stubbing task_store.
    from app.services import tasks as t

    _mem: dict[str, dict] = {}

    async def _submit(payload):
        import secrets
        tid = secrets.token_urlsafe(12)
        _mem[tid] = {"state": "pending"}
        return tid

    async def _store_result(tid, result):
        _mem[tid] = {"state": "succeeded", "result": result}

    async def _store_failure(tid, error):
        _mem[tid] = {"state": "failed", "error": error}

    async def _fetch(tid):
        return _mem.get(tid)

    monkeypatch.setattr(t, "submit_task", _submit)
    monkeypatch.setattr(t, "store_result", _store_result)
    monkeypatch.setattr(t, "store_failure", _store_failure)
    monkeypatch.setattr(t, "fetch_status", _fetch)
    monkeypatch.setattr(t, "_running_tasks", _mem)

    from app.main import app
    return TestClient(app, raise_server_exceptions=False)


def _png_bytes(rng_seed: int, size: int = 64) -> bytes:
    from PIL import Image
    rng = np.random.default_rng(rng_seed)
    arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint(client):
    # /health should not require redis in our stubbed setup
    import app.api.meta as meta
    import redis.asyncio as redis
    class _F:
        async def ping(self): return True
        async def aclose(self): pass
    def _from_url(*a, **k): return _F()
    client._app.dependency_overrides[redis.from_url] = lambda *a, **k: _F()

    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("healthy", "degraded")
    assert "detector" in body


def test_cluster_no_files(client):
    r = client.post("/cluster", files=[])
    assert r.status_code == 400
    assert r.json()["detail"]["error"]["code"] == 4001


def test_cluster_too_many_files(client, monkeypatch):
    from app.core import config as cfg
    monkeypatch.setattr(cfg.settings, "max_images_per_request", 2)
    files = [("files", ("a.png", _png_bytes(1), "image/png")),
             ("files", ("b.png", _png_bytes(2), "image/png")),
             ("files", ("c.png", _png_bytes(3), "image/png"))]
    r = client.post("/cluster", files=files)
    assert r.status_code == 400
    assert r.json()["detail"]["error"]["code"] == 4002


def test_cluster_success(client):
    files = [
        ("files", ("a.png", _png_bytes(1), "image/png")),
        ("files", ("b.png", _png_bytes(2), "image/png")),
        ("files", ("c.png", _png_bytes(3), "image/png")),
    ]
    r = client.post("/cluster", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_images"] == 3
    assert body["n_clusters"] >= 1
    assert "clusters" in body
    assert "label_by_file" in body


def test_cluster_invalid_threshold(client):
    files = [("files", ("a.png", _png_bytes(1), "image/png"))]
    r = client.post("/cluster", files=files, data={"threshold": "9.9"})
    assert r.status_code == 400
    assert r.json()["detail"]["error"]["code"] == 4005


def test_cluster_unsupported_content_type(client):
    files = [("files", ("a.txt", b"hello", "text/plain"))]
    r = client.post("/cluster", files=files)
    assert r.status_code == 400
    assert r.json()["detail"]["error"]["code"] == 4004


def test_cluster_corrupt_payload(client):
    files = [("files", ("a.png", b"not really a png", "image/png"))]
    r = client.post("/cluster", files=files)
    assert r.status_code == 400
    assert r.json()["detail"]["error"]["code"] == 4006


def test_cluster_async_pipeline(client):
    files = [
        ("files", ("a.png", _png_bytes(1), "image/png")),
        ("files", ("b.png", _png_bytes(2), "image/png")),
    ]
    r = client.post("/cluster/async", files=files)
    assert r.status_code == 200, r.text
    task_id = r.json()["task_id"]
    s = client.get(f"/cluster/async/{task_id}")
    assert s.status_code == 200
    assert s.json()["state"] in ("pending", "running", "succeeded", "failed")


def test_cluster_async_unknown_task(client):
    r = client.get("/cluster/async/does_not_exist")
    assert r.status_code == 404
    assert r.json()["detail"]["error"]["code"] == 5005


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "fc_requests_total" in r.text
