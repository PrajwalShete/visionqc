"""Tests for the isolated inference worker.

Covers:

* the worker imports and runs in ``--fake`` mode without anomalib/torch,
* the ``/health`` and ``/infer`` HTTP contract (via FastAPI ``TestClient``,
  which runs the warm-up lifespan), and
* an end-to-end check that the REAL ``HTTPInferenceClient`` from ``visionqc``
  can talk to a live worker booted on a real localhost port.
"""

from __future__ import annotations

import base64
import importlib.util
import socket
import threading
import time
from collections.abc import Iterator

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from visionqc.inference_client.client import (
    FakeInferenceClient,
    HTTPInferenceClient,
    InferenceUnavailable,
)
from visionqc_inference import heatmap as hm
from visionqc_inference.model import FakeModel
from visionqc_inference.worker import build_model, create_app


def _sample_jpeg(seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    return hm.encode_jpeg(frame)


# --- optional-import discipline ----------------------------------------------


def test_worker_imports_without_anomalib() -> None:
    # The main test env deliberately has no ai extra; the worker must import.
    assert importlib.util.find_spec("anomalib") is None
    assert importlib.util.find_spec("torch") is None


def test_build_model_fake_needs_no_ml_deps() -> None:
    model = build_model(fake=True, model_path=None)
    assert isinstance(model, FakeModel)
    model.warmup()  # must not raise


def test_build_model_real_requires_model_path() -> None:
    with pytest.raises(ValueError, match="model-path"):
        build_model(fake=False, model_path=None)


# --- model behavior -----------------------------------------------------------


def test_fake_model_score_matches_client() -> None:
    image = _sample_jpeg(11)
    result = FakeModel().infer(image)
    assert result.score == pytest.approx(FakeInferenceClient.score_for(image))
    assert 0.0 <= result.score < 1.0


def test_fake_model_is_deterministic() -> None:
    image = _sample_jpeg(5)
    a = FakeModel().infer(image)
    b = FakeModel().infer(image)
    assert a.score == b.score
    assert a.heatmap_jpeg == b.heatmap_jpeg


def test_fake_model_heatmap_is_valid_jpeg_matching_frame_size() -> None:
    image = _sample_jpeg(2)
    result = FakeModel().infer(image)
    decoded = hm.decode_image(result.heatmap_jpeg)
    assert decoded.shape == (64, 64, 3)


def test_fake_model_tolerates_non_image_bytes() -> None:
    # Tiny synthetic payloads (as used in orchestrator tests) must not crash.
    result = FakeModel().infer(b"\xff\xd8sample-frame\xff\xd9")
    assert 0.0 <= result.score < 1.0
    assert result.heatmap_jpeg[:2] == b"\xff\xd8"


# --- HTTP contract via TestClient (runs warm-up lifespan) --------------------


@pytest.fixture
def worker_client() -> Iterator[TestClient]:
    app = create_app(FakeModel())
    with TestClient(app) as client:
        yield client


def test_health_reports_warmed_up_after_startup(worker_client: TestClient) -> None:
    resp = worker_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["warmed_up"] is True
    assert body["model_version"] == "fake-1.0"
    assert body["device"] == "cpu"


def test_infer_returns_contract_fields(worker_client: TestClient) -> None:
    image = _sample_jpeg(3)
    resp = worker_client.post(
        "/infer",
        content=image,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= {"score", "heatmap_jpeg_b64", "latency_ms", "model_version"}
    assert 0.0 <= body["score"] < 1.0
    assert body["model_version"] == "fake-1.0"
    assert body["latency_ms"] >= 0.0
    # heatmap_jpeg_b64 decodes to a real JPEG.
    raw = base64.b64decode(body["heatmap_jpeg_b64"])
    assert raw[:2] == b"\xff\xd8"


def test_infer_is_deterministic_over_http(worker_client: TestClient) -> None:
    image = _sample_jpeg(9)
    headers = {"Content-Type": "application/octet-stream"}
    first = worker_client.post("/infer", content=image, headers=headers).json()
    second = worker_client.post("/infer", content=image, headers=headers).json()
    assert first["score"] == second["score"]


def test_infer_empty_body_returns_400(worker_client: TestClient) -> None:
    resp = worker_client.post(
        "/infer",
        content=b"",
        headers={"Content-Type": "application/octet-stream"},
    )
    assert resp.status_code == 400


# --- end-to-end: the REAL HTTPInferenceClient against a live worker ----------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _LiveWorker:
    """Boots the worker with uvicorn in a background thread on a real port."""

    def __init__(self) -> None:
        import uvicorn

        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        config = uvicorn.Config(
            create_app(FakeModel()),
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self) -> _LiveWorker:
        self._thread.start()
        # Wait until the worker is up AND warm-up has completed.
        deadline = time.time() + 15.0
        while time.time() < deadline:
            try:
                resp = httpx.get(f"{self.base_url}/health", timeout=0.5)
                if resp.status_code == 200 and resp.json().get("warmed_up") is True:
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
        raise RuntimeError("worker did not become ready in time")

    def __exit__(self, *_exc: object) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10.0)


@pytest.fixture(scope="module")
def live_worker() -> Iterator[_LiveWorker]:
    with _LiveWorker() as worker:
        yield worker


async def test_real_http_client_can_talk_to_worker(live_worker: _LiveWorker) -> None:
    client = HTTPInferenceClient(f"{live_worker.base_url}/infer", timeout_s=5.0)
    try:
        image = _sample_jpeg(21)
        response = await client.infer(image)
        # HTTPInferenceClient normalized the worker's JSON into InferenceResponse.
        assert response.model_version == "fake-1.0"
        assert 0.0 <= response.score < 1.0
        assert response.latency_ms >= 0.0
        assert response.heatmap_jpeg is not None
        assert response.heatmap_jpeg[:2] == b"\xff\xd8"  # decoded JPEG bytes
        # Score parity with the in-process fake client on the same bytes.
        assert response.score == pytest.approx(FakeInferenceClient.score_for(image))
    finally:
        await client.close()


async def test_real_http_client_times_out_gracefully(live_worker: _LiveWorker) -> None:
    # A microscopic timeout must surface as InferenceUnavailable (→ FAULT), not a hang.
    client = HTTPInferenceClient(f"{live_worker.base_url}/infer", timeout_s=1e-6)
    try:
        with pytest.raises(InferenceUnavailable):
            await client.infer(_sample_jpeg(1))
    finally:
        await client.close()
