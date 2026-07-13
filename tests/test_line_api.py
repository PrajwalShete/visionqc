"""Line-control API (/line/*) exercised end-to-end via the TestClient.

The ``client`` fixture (see ``conftest.py``) wires the full app with a
:class:`FakeInferenceClient`, a synthetic image source, and a fast lifecycle
watchdog, so these tests drive the real simulator through HTTP.
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient


def _activate_recipe(client: TestClient, *, threshold: float = 0.5) -> None:
    created = client.post(
        "/recipes",
        json={
            "name": "synthetic",
            "category": "synthetic",
            "model_name": "padim",
            "anomaly_threshold": threshold,
        },
    ).json()
    client.post(f"/recipes/{created['id']}/activate")


def test_status_initial(client: TestClient) -> None:
    body = client.get("/line/status").json()
    assert body["state"] == "STOPPED"
    assert body["running"] is False
    assert body["active_faults"] == []
    assert body["source"]["type"] == "synthetic"


def test_start_and_stop(client: TestClient) -> None:
    started = client.post("/line/start")
    assert started.status_code == 200
    assert started.json()["running"] is True

    stopped = client.post("/line/stop")
    assert stopped.status_code == 200
    assert stopped.json()["running"] is False


def test_start_is_idempotent(client: TestClient) -> None:
    assert client.post("/line/start").json()["running"] is True
    assert client.post("/line/start").json()["running"] is True
    client.post("/line/stop")


def test_set_speed(client: TestClient) -> None:
    body = client.post("/line/speed", json={"interval_s": 0.5}).json()
    assert body["interval_s"] == 0.5
    assert client.post("/line/speed", json={"interval_s": 0}).status_code == 422
    assert client.post("/line/speed", json={"interval_s": -1}).status_code == 422


def test_toggle_faults(client: TestClient) -> None:
    on = client.post("/line/faults", json={"fault": "camera_loss", "enabled": True}).json()
    assert "camera_loss" in on["active_faults"]

    off = client.post("/line/faults", json={"fault": "camera_loss", "enabled": False}).json()
    assert "camera_loss" not in off["active_faults"]

    bad = client.post("/line/faults", json={"fault": "meltdown", "enabled": True})
    assert bad.status_code == 422


def test_switch_source(client: TestClient) -> None:
    synthetic = client.post("/line/source", json={"type": "synthetic", "defect_rate": 0.5}).json()
    assert synthetic["source"]["type"] == "synthetic"
    assert synthetic["source"]["defect_rate"] == 0.5

    # A directory source needs a valid path.
    assert client.post("/line/source", json={"type": "directory"}).status_code == 422
    missing = client.post("/line/source", json={"type": "directory", "path": "/no/such/dir"})
    assert missing.status_code == 422


def test_products_flow_to_terminal_states(client: TestClient) -> None:
    """Start the line and confirm products flow all the way to terminal states."""

    _activate_recipe(client, threshold=0.5)
    client.post("/line/speed", json={"interval_s": 0.02})
    client.post("/line/start")

    terminal = 0
    try:
        for _ in range(200):
            status = client.get("/line/status").json()
            terminal = status["counters"]["terminal"]
            if terminal >= 5:
                break
            time.sleep(0.02)
    finally:
        client.post("/line/stop")

    assert terminal >= 5

    status = client.get("/line/status").json()
    assert status["counters"]["lost"] == 0
    assert status["ticks"] >= 5

    products = client.get("/products").json()
    assert products["count"] >= 5
    # Products still mid-flight at query time have a null outcome; every product
    # that has finalized must carry a terminal disposition.
    outcomes = [p["outcome"] for p in products["items"] if p["outcome"] is not None]
    assert len(outcomes) >= 5
    assert all(o in {"PASS", "REJECT", "FAULT"} for o in outcomes)
