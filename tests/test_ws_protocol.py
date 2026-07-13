"""WebSocket dashboard protocol tests.

Covers the hello snapshot, JSON event frames, the binary evidence frame
(``[len][meta][jpeg]`` round-trip), per-client drop-oldest backpressure, and
dead-client ejection.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from visionqc.api.ws import (
    ConnectionManager,
    _ClientChannel,
    decode_binary_frame,
    encode_binary_frame,
)
from visionqc.evidence.store import encode_jpeg


def _jpeg_bytes(color: tuple[int, int, int] = (0, 128, 255)) -> bytes:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:] = color
    return encode_jpeg(img)


def _drain_hello(ws) -> dict:
    msg = json.loads(ws.receive_text())
    assert msg["kind"] == "hello"
    return msg


# --------------------------------------------------------------------------- #
# Pure protocol framing
# --------------------------------------------------------------------------- #
def test_binary_frame_encode_decode_roundtrip() -> None:
    meta = {"product_id": "p1", "image_kind": "heatmap", "ts": "t", "outcome": "REJECT"}
    jpeg = _jpeg_bytes()
    frame = encode_binary_frame(meta, jpeg)

    # 4-byte big-endian length header precedes the meta JSON.
    meta_len = int.from_bytes(frame[:4], "big")
    assert meta_len == len(json.dumps(meta, separators=(",", ":")).encode("utf-8"))

    decoded_meta, decoded_jpeg = decode_binary_frame(frame)
    assert decoded_meta == meta
    assert decoded_jpeg == jpeg


def test_decode_rejects_truncated_frame() -> None:
    with pytest.raises(ValueError):
        decode_binary_frame(b"\x00\x00")
    with pytest.raises(ValueError):
        # Claims a 100-byte meta but has none.
        decode_binary_frame((100).to_bytes(4, "big") + b"{}")


# --------------------------------------------------------------------------- #
# Hello + event frames over a live TestClient socket
# --------------------------------------------------------------------------- #
def test_hello_delivered_on_connect(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        hello = _drain_hello(ws)
        payload = hello["payload"]
        assert payload["line_state"] == "RUNNING"
        assert "reconciliation" in payload
        assert payload["reconciliation"]["lost"] == 0
        assert isinstance(payload["active_alarms"], list)


def test_event_frame_shape(client: TestClient) -> None:
    ctx = client.app.state.ctx
    portal = client.portal
    with client.websocket_connect("/ws") as ws:
        _drain_hello(ws)
        portal.call(ctx.tracker.trigger, "wsp-evt")

        # First event on the wire should be the TriggerFired we just published.
        msg = json.loads(ws.receive_text())
        assert msg["kind"] == "event"
        assert msg["type"] == "TriggerFired"
        assert msg["ts"] is not None
        assert msg["event_id"]
        assert msg["payload"]["product_id"] == "wsp-evt"


def test_binary_evidence_frame_pushed_on_finalize(client: TestClient) -> None:
    ctx = client.app.state.ctx
    portal = client.portal
    jpeg = _jpeg_bytes()

    async def _run() -> str:
        pid = "wsp-bin"
        await ctx.tracker.trigger(pid)
        await ctx.tracker.mark_captured(pid)
        for kind in ("raw", "heatmap"):
            ref = await ctx.evidence.save_jpeg(pid, kind, jpeg)
            await ctx.repo.insert_evidence(pid, kind, ref.path, ref.sha256, ref.mime)
        await ctx.tracker.mark_inferred(pid, score=0.2, model_version="m", latency_ms=3.0)
        await ctx.tracker.mark_decided(pid, outcome="PASS", reason="within_tolerance", score=0.2)
        await ctx.tracker.finalize_pass(pid, reason="within_tolerance", score=0.2)
        return pid

    with client.websocket_connect("/ws") as ws:
        _drain_hello(ws)
        portal.call(_run)

        binary = None
        for _ in range(40):
            frame = ws.receive()
            if frame.get("bytes") is not None:
                binary = frame["bytes"]
                break
        assert binary is not None, "expected a binary evidence frame"

        meta, jpeg_out = decode_binary_frame(binary)
        assert meta["product_id"] == "wsp-bin"
        assert meta["image_kind"] in {"raw", "heatmap"}
        assert meta["outcome"] == "PASS"
        assert "ts" in meta
        # JPEG payload must be decodable back to an image.
        import cv2

        decoded = cv2.imdecode(np.frombuffer(jpeg_out, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape[0] > 0 and decoded.shape[1] > 0


# --------------------------------------------------------------------------- #
# Backpressure: bounded per-client queues, drop-oldest
# --------------------------------------------------------------------------- #
def test_frame_queue_drops_oldest() -> None:
    ch = _ClientChannel(websocket=object(), event_maxsize=10, frame_maxsize=2)
    ch.offer_frame(b"a")
    ch.offer_frame(b"b")
    ch.offer_frame(b"c")  # overflow -> oldest ("a") dropped

    assert ch.frames.qsize() == 2
    assert ch.dropped_frames == 1
    assert ch.frames.get_nowait() == b"b"
    assert ch.frames.get_nowait() == b"c"


def test_event_queue_drops_oldest() -> None:
    ch = _ClientChannel(websocket=object(), event_maxsize=2, frame_maxsize=3)
    ch.offer_event("1")
    ch.offer_event("2")
    ch.offer_event("3")  # overflow -> oldest ("1") dropped

    assert ch.events.qsize() == 2
    assert ch.dropped_events == 1
    assert ch.events.get_nowait() == "2"


def test_broadcast_wraps_event_and_never_raises() -> None:
    mgr = ConnectionManager(queue_maxsize=3)
    ch = _ClientChannel(websocket=object(), event_maxsize=10, frame_maxsize=3)
    mgr._channels[object()] = ch  # dummy key; we hold the channel directly
    # Point the manager's channel list at our channel.
    mgr._channels = {id(ch): ch}  # type: ignore[dict-item]

    wire = {
        "event_id": "e1",
        "ts_wall": "2026-07-13T00:00:00+00:00",
        "ts_mono": 1.0,
        "type": "DecisionMade",
        "payload": {"product_id": "p1", "outcome": "PASS"},
    }
    mgr.broadcast(wire)

    frame = json.loads(ch.events.get_nowait())
    assert frame["kind"] == "event"
    assert frame["type"] == "DecisionMade"
    assert frame["ts"] == wire["ts_wall"]
    assert frame["payload"]["product_id"] == "p1"


# --------------------------------------------------------------------------- #
# Dead-client ejection
# --------------------------------------------------------------------------- #
class _FailingWS:
    """A socket whose sends always fail — models a dead client."""

    async def send_text(self, message: str) -> None:
        raise RuntimeError("connection reset")

    async def send_bytes(self, data: bytes) -> None:  # pragma: no cover
        raise RuntimeError("connection reset")


async def test_dead_client_ejected_on_send_failure() -> None:
    mgr = ConnectionManager()
    ws = _FailingWS()
    ch = _ClientChannel(websocket=ws, event_maxsize=10, frame_maxsize=3)
    mgr._channels[ws] = ch  # type: ignore[index]
    ch.offer_event('{"kind":"event"}')

    # The sender attempts one send, which raises, and it ejects the client.
    await mgr._sender(ch)

    assert ws not in mgr._channels
    assert mgr.client_count == 0
