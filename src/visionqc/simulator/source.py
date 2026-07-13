"""Image sources — where the virtual camera gets its frames.

Two implementations back the demo:

* :class:`DirectoryImageSource` loops forever over a directory tree of JPEG/PNG
  files. This is how MVTec test images are fed on EC2. Ground truth is inferred
  from the first-level sub-directory name (``good/`` vs a defect category such as
  ``broken_large/``) so the dashboard can score the model against known labels.
* :class:`SyntheticImageSource` procedurally generates plain gray "product"
  squares, a configurable fraction of which receive an obvious dark blob
  "defect". This lets the system demo end-to-end with **zero** dataset.

Both are asynchronous iterators yielding :class:`SourceImage` — a
``(image_bytes, ground_truth_label, source_name)`` triple — and never raise
``StopAsyncIteration`` in normal operation: they cycle indefinitely so the line
can run for as long as the operator likes.
"""

from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

#: File extensions recognized as product frames.
_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})

#: Ground-truth label used for nominal (non-defective) products.
GOOD_LABEL = "good"


@dataclass(frozen=True)
class SourceImage:
    """A single frame produced by an :class:`ImageSource`.

    :param data: Encoded image bytes (JPEG/PNG), ready for the inference worker.
    :param label: Ground-truth label (``"good"`` or a defect category), or
        ``None`` when the source cannot attribute one.
    :param source: Human-readable name of the source that produced the frame.
    """

    data: bytes
    label: str | None
    source: str

    @property
    def is_defect(self) -> bool:
        """Whether the ground truth marks this frame as defective."""

        return self.label is not None and self.label != GOOD_LABEL


class ImageSource(ABC):
    """Asynchronous, infinitely-cycling source of product frames."""

    #: Short, stable identifier for the source (surfaced in ``/line/status``).
    name: str = "source"

    def __aiter__(self) -> AsyncIterator[SourceImage]:
        return self

    async def __anext__(self) -> SourceImage:
        return await self.next_image()

    @abstractmethod
    async def next_image(self) -> SourceImage:
        """Return the next frame, cycling back to the start when exhausted."""

    async def close(self) -> None:  # noqa: B027 - optional cleanup hook, no-op by default
        """Release any resources held by the source (no-op by default)."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return a JSON-friendly description for the status endpoint."""


class DirectoryImageSource(ImageSource):
    """Loops forever over the image files found beneath ``root``.

    Files are discovered recursively, sorted for determinism, and cycled. The
    ground-truth label of each frame is the name of its first-level
    sub-directory relative to ``root`` (e.g. ``root/good/000.png`` → ``"good"``,
    ``root/broken/003.png`` → ``"broken"``); a file directly under ``root`` gets
    a ``None`` label.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        name: str = "directory",
        extensions: frozenset[str] = _IMAGE_EXTENSIONS,
    ) -> None:
        self._root = Path(root)
        self.name = name
        self._extensions = extensions
        self._paths: list[Path] = self._scan()
        if not self._paths:
            raise FileNotFoundError(f"no images ({sorted(extensions)}) found under {self._root}")
        self._index = 0

    def _scan(self) -> list[Path]:
        if not self._root.is_dir():
            raise FileNotFoundError(f"image directory does not exist: {self._root}")
        return sorted(
            p for p in self._root.rglob("*") if p.is_file() and p.suffix.lower() in self._extensions
        )

    def _label_for(self, path: Path) -> str | None:
        try:
            relative = path.relative_to(self._root)
        except ValueError:  # pragma: no cover - path always under root
            return None
        # relative.parts == (subdir, ..., filename); the first part is the label.
        return relative.parts[0] if len(relative.parts) > 1 else None

    async def next_image(self) -> SourceImage:
        path = self._paths[self._index % len(self._paths)]
        self._index += 1
        data = await asyncio.to_thread(path.read_bytes)
        return SourceImage(data=data, label=self._label_for(path), source=self.name)

    def describe(self) -> dict[str, Any]:
        return {
            "type": "directory",
            "name": self.name,
            "path": str(self._root),
            "image_count": len(self._paths),
            "position": self._index % len(self._paths),
        }


class SyntheticImageSource(ImageSource):
    """Procedurally generates gray product squares with occasional dark blobs.

    A configurable ``defect_rate`` fraction of frames receive an obvious dark
    rectangular blob (label ``"defect"``); the rest are plain gray (label
    ``"good"``). Frames are encoded to JPEG so downstream code sees realistic
    image bytes. A seeded RNG makes the defect sequence reproducible for tests.
    """

    def __init__(
        self,
        *,
        name: str = "synthetic",
        defect_rate: float = 0.2,
        size: int = 256,
        seed: int | None = None,
        image_format: str = ".jpg",
    ) -> None:
        if not 0.0 <= defect_rate <= 1.0:
            raise ValueError("defect_rate must be within [0, 1]")
        if size <= 16:
            raise ValueError("size must be greater than 16 pixels")
        self.name = name
        self._defect_rate = defect_rate
        self._size = size
        self._format = image_format
        self._rng = random.Random(seed)
        self._count = 0
        self._defects = 0

    def _render(self, defect: bool) -> bytes:
        # Plain gray product with mild sensor-like noise for realism.
        base = np.full((self._size, self._size, 3), 160, dtype=np.uint8)
        noise = self._rng.randint(0, 6)
        if noise:
            base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        if defect:
            span = self._size // 5
            x = self._rng.randint(span, self._size - 2 * span)
            y = self._rng.randint(span, self._size - 2 * span)
            cv2.rectangle(base, (x, y), (x + span, y + span), (20, 20, 20), thickness=-1)
        ok, buffer = cv2.imencode(self._format, base)
        if not ok:  # pragma: no cover - imencode failure is not expected
            raise RuntimeError(f"failed to encode synthetic {self._format} frame")
        return buffer.tobytes()

    async def next_image(self) -> SourceImage:
        defect = self._rng.random() < self._defect_rate
        data = await asyncio.to_thread(self._render, defect)
        self._count += 1
        if defect:
            self._defects += 1
        label = "defect" if defect else GOOD_LABEL
        return SourceImage(data=data, label=label, source=self.name)

    def describe(self) -> dict[str, Any]:
        return {
            "type": "synthetic",
            "name": self.name,
            "defect_rate": self._defect_rate,
            "size": self._size,
            "generated": self._count,
            "defects": self._defects,
        }


def build_source(
    source_type: str,
    *,
    path: Path | str | None = None,
    defect_rate: float = 0.2,
    seed: int | None = None,
) -> ImageSource:
    """Construct an :class:`ImageSource` from a ``synthetic``/``directory`` type.

    :raises ValueError: if ``source_type`` is unknown or a directory source is
        requested without a ``path``.
    """

    normalized = source_type.strip().lower()
    if normalized == "synthetic":
        return SyntheticImageSource(defect_rate=defect_rate, seed=seed)
    if normalized == "directory":
        if path is None:
            raise ValueError("a directory source requires a 'path'")
        return DirectoryImageSource(path)
    raise ValueError(f"unknown source type: {source_type!r} (expected synthetic|directory)")


__all__ = [
    "GOOD_LABEL",
    "DirectoryImageSource",
    "ImageSource",
    "SourceImage",
    "SyntheticImageSource",
    "build_source",
]
