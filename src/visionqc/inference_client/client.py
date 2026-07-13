"""Client to the isolated GPU inference worker.

Every call is wrapped in ``asyncio.wait_for`` with the configured timeout. On
timeout, connection error, or a worker error response, a typed
:class:`InferenceUnavailable` is raised — the orchestrator maps it to a FAULT
disposition, a fail-safe alarm, and a degraded-mode flag (never a hang, never a
silent pass).

:class:`FakeInferenceClient` produces deterministic scores from the image hash
for tests and local development without a GPU.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from typing import Protocol

import httpx


class InferenceUnavailable(Exception):
    """Raised when the inference worker is unreachable, slow, or errored."""


@dataclass(frozen=True)
class InferenceResponse:
    """Normalized inference result returned to the orchestrator."""

    score: float
    model_version: str
    latency_ms: float
    heatmap_jpeg: bytes | None = None


class InferenceClient(Protocol):
    """Interface the orchestrator depends on (real or fake)."""

    async def infer(self, image: bytes) -> InferenceResponse:
        """Run inference on JPEG ``image`` bytes; raise on failure."""
        ...

    async def close(self) -> None:
        """Release any underlying resources."""
        ...


def _decode_heatmap(value: object) -> bytes | None:
    """Accept a base64 string or raw bytes heatmap field from the worker."""

    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return base64.b64decode(value)
    return None


class HTTPInferenceClient:
    """httpx-based client for the localhost worker."""

    def __init__(self, url: str, timeout_s: float) -> None:
        self._url = url
        self._timeout_s = timeout_s
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s))

    async def infer(self, image: bytes) -> InferenceResponse:
        """POST image bytes to the worker and normalize the response."""

        try:
            response = await self._client.post(
                self._url,
                content=image,
                headers={"Content-Type": "application/octet-stream"},
            )
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            raise InferenceUnavailable(f"inference timed out after {self._timeout_s}s") from exc
        except httpx.HTTPError as exc:
            raise InferenceUnavailable(f"inference request failed: {exc}") from exc
        except ValueError as exc:  # malformed JSON body
            raise InferenceUnavailable(f"malformed inference response: {exc}") from exc

        try:
            return InferenceResponse(
                score=float(data["score"]),
                model_version=str(data.get("model_version", "unknown")),
                latency_ms=float(data.get("latency_ms", 0.0)),
                heatmap_jpeg=_decode_heatmap(
                    data.get("heatmap_jpeg_b64") or data.get("heatmap_jpeg")
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise InferenceUnavailable(f"incomplete inference response: {exc}") from exc

    async def close(self) -> None:
        await self._client.aclose()


class FakeInferenceClient:
    """Deterministic in-process client — score derived from the image hash.

    Useful for tests and GPU-less local runs. ``fail_on`` lets a test force an
    :class:`InferenceUnavailable` for specific image payloads.
    """

    def __init__(
        self,
        model_version: str = "fake-1.0",
        latency_ms: float = 5.0,
        fail: bool = False,
    ) -> None:
        self._model_version = model_version
        self._latency_ms = latency_ms
        self._fail = fail

    @staticmethod
    def score_for(image: bytes) -> float:
        """Map image bytes to a stable score in ``[0, 1)``."""

        digest = hashlib.sha256(image).digest()
        return int.from_bytes(digest[:4], "big") / 2**32

    async def infer(self, image: bytes) -> InferenceResponse:
        if self._fail:
            raise InferenceUnavailable("fake client configured to fail")
        return InferenceResponse(
            score=self.score_for(image),
            model_version=self._model_version,
            latency_ms=self._latency_ms,
            heatmap_jpeg=None,
        )

    async def close(self) -> None:
        return None


__all__ = [
    "FakeInferenceClient",
    "HTTPInferenceClient",
    "InferenceClient",
    "InferenceResponse",
    "InferenceUnavailable",
]
