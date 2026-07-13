"""WebSocket hub: bounded per-client queues with drop-oldest backpressure.

This is the skeleton the dashboard task will build the message protocol on. What
works today: a :class:`ConnectionManager` that fans bus events out to every
connected client through a bounded queue (``maxsize`` small, **drop-oldest** on
overflow), with separate send/recv tasks per socket and immediate ejection of a
client whose send fails — one dead client can never stall the broadcast loop
(architecture gotcha #2).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class _ClientChannel:
    """Per-client bounded outbound queue with drop-oldest semantics."""

    def __init__(self, websocket: WebSocket, maxsize: int) -> None:
        self.websocket = websocket
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0

    def offer(self, message: str) -> None:
        """Enqueue ``message``, discarding the oldest if the queue is full."""

        try:
            self.queue.put_nowait(message)
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()
                self.dropped += 1
            except asyncio.QueueEmpty:  # pragma: no cover
                pass
            try:
                self.queue.put_nowait(message)
            except asyncio.QueueFull:  # pragma: no cover
                self.dropped += 1


class ConnectionManager:
    """Tracks connected clients and broadcasts JSON messages to them."""

    def __init__(self, queue_maxsize: int = 3) -> None:
        self._queue_maxsize = queue_maxsize
        self._channels: dict[WebSocket, _ClientChannel] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> _ClientChannel:
        """Accept a socket and register a channel for it."""

        await websocket.accept()
        channel = _ClientChannel(websocket, self._queue_maxsize)
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

    def broadcast(self, message: dict[str, Any]) -> None:
        """Fan a JSON-serializable message out to every client's queue.

        Non-blocking: overflow is handled per client by drop-oldest, so a slow
        client never stalls the broadcast.
        """

        encoded = json.dumps(message)
        for channel in list(self._channels.values()):
            channel.offer(encoded)

    async def _sender(self, channel: _ClientChannel) -> None:
        """Drain a client's queue to its socket; eject on first send failure."""

        try:
            while True:
                message = await channel.queue.get()
                await channel.websocket.send_text(message)
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception:
            logger.debug("ws sender error; ejecting client", exc_info=True)
        finally:
            await self.disconnect(channel.websocket)

    async def serve(self, websocket: WebSocket) -> None:
        """Run one client: accept, then race the send and receive loops.

        The receive loop only drains inbound frames (and detects disconnect);
        the dashboard task will define the inbound protocol.
        """

        channel = await self.connect(websocket)
        sender = asyncio.create_task(self._sender(channel))
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.debug("ws receive error", exc_info=True)
        finally:
            sender.cancel()
            await self.disconnect(websocket)


__all__ = ["ConnectionManager"]
