"""API routes via TestClient (lifespan + fake inference client)."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["line_state"] == "RUNNING"
    assert body["zero_silent_loss"] is True
    assert body["reconciliation"]["lost"] == 0


def test_stats(client: TestClient) -> None:
    resp = client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert "live" in body and "persisted" in body
    assert body["active_alarms"] == 0


def test_recipe_create_activate_flow(client: TestClient) -> None:
    created = client.post(
        "/recipes",
        json={
            "name": "bottle",
            "category": "bottle",
            "model_name": "padim",
            "anomaly_threshold": 0.5,
            "confidence_margin": 0.05,
        },
    )
    assert created.status_code == 201
    recipe = created.json()
    assert recipe["version"] == 1
    assert recipe["active"] == 0

    activated = client.post(f"/recipes/{recipe['id']}/activate")
    assert activated.status_code == 200
    assert activated.json()["active"] == 1

    listing = client.get("/recipes")
    assert listing.status_code == 200
    assert listing.json()["active"]["id"] == recipe["id"]


def test_activate_unknown_recipe_404(client: TestClient) -> None:
    resp = client.post("/recipes/999/activate")
    assert resp.status_code == 404


def test_new_version_does_not_mutate_old(client: TestClient) -> None:
    base = {"name": "nut", "category": "metal_nut", "model_name": "padim"}
    r1 = client.post("/recipes", json={**base, "anomaly_threshold": 0.4}).json()
    client.post("/recipes", json={**base, "anomaly_threshold": 0.7})
    listing = client.get("/recipes").json()
    v1 = next(r for r in listing["items"] if r["id"] == r1["id"])
    assert v1["anomaly_threshold"] == 0.4
    assert v1["version"] == 1


def test_products_empty(client: TestClient) -> None:
    resp = client.get("/products")
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 0}


def test_product_detail_404(client: TestClient) -> None:
    resp = client.get("/products/nope")
    assert resp.status_code == 404


def test_alarms_empty(client: TestClient) -> None:
    resp = client.get("/alarms")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0


def test_product_lifecycle_persisted_via_bus(client: TestClient) -> None:
    """Drive the tracker through the app context; DB-writer persists it."""

    ctx = client.app.state.ctx
    portal = client.portal  # anyio blocking portal from the running lifespan

    async def _run() -> str:
        pid = await ctx.tracker.trigger("api-p1")
        await ctx.tracker.mark_captured(pid)
        await ctx.tracker.mark_inferred(pid, score=0.1, model_version="m", latency_ms=1.0)
        await ctx.tracker.mark_decided(pid, outcome="PASS", reason="within_tolerance", score=0.1)
        await ctx.tracker.finalize_pass(pid, reason="within_tolerance", score=0.1)
        return pid

    pid = portal.call(_run)

    # Poll the detail endpoint until the async DB writer has caught up.
    detail = None
    for _ in range(50):
        resp = client.get(f"/products/{pid}")
        if resp.status_code == 200 and resp.json().get("outcome") == "PASS":
            detail = resp.json()
            break
        import time

        time.sleep(0.02)

    assert detail is not None
    assert detail["state"] == "PASS"
    assert detail["outcome"] == "PASS"
    event_types = [e["event_type"] for e in detail["events"]]
    assert "TriggerFired" in event_types
    assert "ProductFinalized" in event_types

    health = client.get("/health").json()
    assert health["reconciliation"]["pass"] == 1
    assert health["reconciliation"]["lost"] == 0
