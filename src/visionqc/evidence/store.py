"""Evidence store: persist JPEG bytes to disk with a content hash.

Images are laid out as ``<root>/YYYY-MM-DD/<product_id>/<kind>.jpg`` (per the
architecture doc). No image bytes go in the database — only path + SHA-256 +
MIME + timestamp, recorded separately by the repository.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class EvidenceRef:
    """A stored evidence image: absolute path + content hash + kind."""

    product_id: str
    kind: str
    path: str
    sha256: str
    mime: str = "image/jpeg"


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    """Encode a BGR/grayscale numpy image to JPEG bytes via OpenCV."""

    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:  # pragma: no cover - encode failure is not reproducible in tests
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


class EvidenceStore:
    """Filesystem-backed evidence writer."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def _path_for(self, product_id: str, kind: str, day: str) -> Path:
        return self._root / day / product_id / f"{kind}.jpg"

    async def save_jpeg(
        self,
        product_id: str,
        kind: str,
        data: bytes,
        *,
        when: datetime | None = None,
    ) -> EvidenceRef:
        """Persist ``data`` (JPEG bytes) and return an :class:`EvidenceRef`.

        The write and hashing run in a worker thread so the event loop is never
        blocked on disk I/O.
        """

        if kind not in {"raw", "heatmap"}:
            raise ValueError(f"unknown evidence kind: {kind!r}")
        moment = when or datetime.now(UTC)
        day = moment.strftime("%Y-%m-%d")
        path = self._path_for(product_id, kind, day)

        def _write() -> str:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return hashlib.sha256(data).hexdigest()

        digest = await asyncio.to_thread(_write)
        return EvidenceRef(
            product_id=product_id,
            kind=kind,
            path=str(path.resolve()),
            sha256=digest,
        )


__all__ = ["EvidenceRef", "EvidenceStore", "encode_jpeg"]
