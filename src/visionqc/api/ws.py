"""WebSocket hub + dashboard message protocol.

One WebSocket per client carries two frame types (architecture §2):

* **text frames** — UTF-8 JSON. Every bus event is forwarded as
  ``{"kind": "event", "type", "ts", "event_id", "payload"}``. On connect the
  client also receives one ``{"kind": "hello", ...}`` snapshot so the UI paints
  immediately (current line state + reconciliation + active alarms).
* **binary frames** — a JPEG evidence image framed as
  ``[4-byte big-endian meta length][UTF-8 JSON meta][raw JPEG bytes]`` in a
  single binary message. ``meta`` = ``{product_id, image_kind, ts, outcome}``
  with ``image_kind`` one of ``"raw" | "heatmap"``. When a product is finalized
  its evidence images are loaded **once** and the same bytes are fanned out to
  every client.

Backpressure (architecture gotcha #2): each client has two bounded queues —
a small **frames** queue and a larger **events** queue — both **drop-oldest** on
overflow. A dedicated sender task drains them and the client is ejected on the
first failed send, so one dead/slow client can never stall the broadcast loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from ..events.schemas import EventType

logger = logging.getLogger(__name__)

#: Default size of the (larger) per-client events queue.
DEFAULT_EVENT_MAXSIZE = 512
#: Evidence kinds pushed as binary frames, in priority order.
_EVIDENCE_KINDS = ("heatmap", "raw")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def encode_binary_frame(meta: dict[str, Any], jpeg: bytes) -> bytes:
    """Frame a JPEG image + JSON meta as ``[len(4, BE)][meta][jpeg]``."""

    meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    return len(meta_bytes).to_bytes(4, "big") + meta_bytes + jpeg


def decode_binary_frame(frame: bytes) -> tuple[dict[str, Any], bytes]:
    """Inverse of :func:`encode_binary_frame` — returns ``(meta, jpeg)``.

    Provided for tests and any Python consumer; the browser does the same parse
    in ``dashboard.js``.
    """

    if len(frame) < 4:
        raise ValueError("binary frame too short")
    meta_len = int.from_bytes(frame[:4], "big")
    meta_end = 4 + meta_len
    if meta_end > len(frame):
        raise ValueError("binary frame meta length exceeds payload")
    meta = json.loads(frame[4:meta_end].decode("utf-8"))
    return meta, frame[meta_end:]


class _ClientChannel:
    """Per-client outbound state: bounded events + frames queues (drop-oldest)."""

    def __init__(self, websocket: WebSocket, event_maxsize: int, frame_maxsize: int) -> None:
        self.websocket = websocket
        self.events: asyncio.Queue[str] = asyncio.Queue(maxsize=event_maxsize)
        self.frames: asyncio.Queue[bytes] = asyncio.Queue(maxsize=frame_maxsize)
        self.dropped_events = 0
        self.dropped_frames = 0

    @staticmethod
    def _offer(queue: asyncio.Queue[Any], item: Any) -> bool:
        """Enqueue ``item``, discarding the oldest on overflow. Returns True if
        something was dropped."""

        try:
            queue.put_nowait(item)
            return False
        except asyncio.QueueFull:
            dropped = False
            try:
                queue.get_nowait()
                dropped = True
            except asyncio.QueueEmpty:  # pragma: no cover - race safety
                pass
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:  # pragma: no cover - race safety
                dropped = True
            return dropped

    def offer_event(self, message: str) -> None:
        if self._offer(self.events, message):
            self.dropped_events += 1

    def offer_frame(self, frame: bytes) -> None:
        if self._offer(self.frames, frame):
            self.dropped_frames += 1


class ConnectionManager:
    """Tracks connected clients and fans events + evidence frames out to them."""

    def __init__(self, queue_maxsize: int = 3, event_maxsize: int = DEFAULT_EVENT_MAXSIZE) -> None:
        self._frame_maxsize = queue_maxsize
        self._event_maxsize = event_maxsize
        self._channels: dict[WebSocket, _ClientChannel] = {}
        self._lock = asyncio.Lock()
        # Bound to the app context on first connect so evidence can be loaded
        # for binary frames without touching main.py's ws bridge.
        self._ctx: Any | None = None
        # Strong refs to in-flight evidence-load tasks (else GC may cancel them).
        self._evidence_tasks: set[asyncio.Task[None]] = set()

    # ---- registration -------------------------------------------------
    async def connect(self, websocket: WebSocket) -> _ClientChannel:
        """Accept a socket and register a channel for it."""

        await websocket.accept()
        if self._ctx is None:
            self._ctx = getattr(websocket.app.state, "ctx", None)
        channel = _ClientChannel(websocket, self._event_maxsize, self._frame_maxsize)
        async with self._lock:
            self._channels[websocket] = channel
        return channel

    async def disconnect(self, websocket: WebSocket) -> None:
        """Eject a client from the manager."""

        async with self._lock:
            self._channels.pop(websocket, None)

    @property
    def client_count(self) -> int:
        return len(self._channels)

    # ---- broadcast ----------------------------------------------------
    def broadcast(self, message: dict[str, Any]) -> None:
        """Fan a bus event (its ``to_wire()`` dict) out to every client.

        Non-blocking: overflow is per-client drop-oldest, so a slow client never
        stalls the broadcast. Finalized products also schedule an evidence load
        that pushes binary frames.
        """

        frame = {
            "kind": "event",
            "type": message.get("type"),
            "ts": message.get("ts_wall"),
            "event_id": message.get("event_id"),
            "payload": message.get("payload"),
        }
        encoded = json.dumps(frame)
        for channel in list(self._channels.values()):
            channel.offer_event(encoded)

        if (
            message.get("type") == EventType.PRODUCT_FINALIZED.value
            and self._channels
            and self._ctx is not None
        ):
            payload = message.get("payload") or {}
            product_id = payload.get("product_id")
            if product_id:
                self._schedule_evidence(product_id, payload.get("outcome"))

    def _schedule_evidence(self, product_id: str, outcome: str | None) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - broadcast always runs in loop
            return
        task = loop.create_task(self._push_evidence(product_id, outcome))
        self._evidence_tasks.add(task)
        task.add_done_callback(self._evidence_tasks.discard)

    async def _push_evidence(self, product_id: str, outcome: str | None) -> None:
        """Load a finalized product's evidence images once and fan them out."""

        ctx = self._ctx
        if ctx is None:  # pragma: no cover - guarded by caller
            return
        try:
            rows = await ctx.repo.evidence_for(product_id)
        except Exception:
            logger.debug("evidence lookup failed for %s", product_id, exc_info=True)
            return
        by_kind = {row["kind"]: row for row in rows}
        ts = _now_iso()
        for kind in _EVIDENCE_KINDS:
            row = by_kind.get(kind)
            if row is None or not row.get("path"):
                continue
            path = Path(row["path"])
            try:
                data = await asyncio.to_thread(path.read_bytes)
            except OSError:
                logger.debug("evidence file missing: %s", path, exc_info=True)
                continue
            frame = encode_binary_frame(
                {
                    "product_id": product_id,
                    "image_kind": kind,
                    "ts": ts,
                    "outcome": outcome,
                },
                data,
            )
            for channel in list(self._channels.values()):
                channel.offer_frame(frame)

    # ---- per-client tasks ---------------------------------------------
    async def _sender(self, channel: _ClientChannel) -> None:
        """Drain a client's events + frames queues; eject on first send failure.

        Two persistent ``get()`` futures are raced with :func:`asyncio.wait` so a
        pending get never removes an item it then loses — only completed gets are
        consumed, and the losers are cancelled at teardown while still empty.
        """

        ws = channel.websocket
        ev_task: asyncio.Task[str] | None = None
        fr_task: asyncio.Task[bytes] | None = None
        try:
            while True:
                if ev_task is None:
                    ev_task = asyncio.ensure_future(channel.events.get())
                if fr_task is None:
                    fr_task = asyncio.ensure_future(channel.frames.get())
                done, _ = await asyncio.wait(
                    {ev_task, fr_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if ev_task in done:
                    message = ev_task.result()
                    ev_task = None
                    await ws.send_text(message)
                if fr_task in done:
                    data = fr_task.result()
                    fr_task = None
                    await ws.send_bytes(data)
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            pass
        except Exception:
            logger.debug("ws sender error; ejecting client", exc_info=True)
        finally:
            for task in (ev_task, fr_task):
                if task is not None:
                    task.cancel()
            await self.disconnect(channel.websocket)

    async def _send_hello(self, websocket: WebSocket) -> None:
        """Push the initial snapshot so the dashboard paints without waiting."""

        ctx = self._ctx or getattr(websocket.app.state, "ctx", None)
        payload: dict[str, Any] = {}
        if ctx is not None:
            payload["line_state"] = getattr(ctx, "line_state", "UNKNOWN")
            try:
                payload["worker_status"] = ctx.worker_status()
            except Exception:  # pragma: no cover - defensive
                payload["worker_status"] = "unknown"
            try:
                payload["reconciliation"] = ctx.tracker.reconcile().as_dict()
            except Exception:  # pragma: no cover - defensive
                payload["reconciliation"] = {}
            payload["clients"] = self.client_count
            try:
                payload["active_alarms"] = await ctx.repo.list_alarms(active_only=True)
            except Exception:  # pragma: no cover - defensive
                payload["active_alarms"] = []
        hello = {"kind": "hello", "ts": _now_iso(), "payload": payload}
        await websocket.send_text(json.dumps(hello))

    async def serve(self, websocket: WebSocket) -> None:
        """Run one client: accept, send hello, then race send/receive loops."""

        channel = await self.connect(websocket)
        try:
            await self._send_hello(websocket)
        except Exception:
            logger.debug("hello send failed; ejecting", exc_info=True)
            await self.disconnect(websocket)
            return
        sender = asyncio.create_task(self._sender(channel))
        try:
            while True:
                # Inbound frames are drained only to detect disconnect.
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.debug("ws receive error", exc_info=True)
        finally:
            sender.cancel()
            await self.disconnect(websocket)


# --------------------------------------------------------------------------- #
# Evidence image serving (for the traceability drawer)
# --------------------------------------------------------------------------- #
evidence_router = APIRouter()


@evidence_router.get("/evidence/{product_id}/{kind}")
async def get_evidence(request: Request, product_id: str, kind: str) -> FileResponse:
    """Serve a stored evidence JPEG by product + kind (``raw`` | ``heatmap``)."""

    if kind not in {"raw", "heatmap"}:
        raise HTTPException(status_code=404, detail="unknown evidence kind")
    ctx = request.app.state.ctx
    rows = await ctx.repo.evidence_for(product_id)
    match = next((row for row in rows if row.get("kind") == kind), None)
    if match is None:
        raise HTTPException(status_code=404, detail="evidence not found")
    path = Path(match["path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="evidence file missing")
    return FileResponse(str(path), media_type=match.get("mime", "image/jpeg"))


__all__ = [
    "ConnectionManager",
    "decode_binary_frame",
    "encode_binary_frame",
    "evidence_router",
]
