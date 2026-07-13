"""Evidence store — filesystem layout, hashing, JPEG encode."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from visionqc.evidence.store import EvidenceStore, encode_jpeg


async def test_save_jpeg_layout_and_hash(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    data = b"\xff\xd8fake-jpeg-bytes\xff\xd9"
    when = datetime(2026, 7, 13, tzinfo=UTC)
    ref = await store.save_jpeg("prod123", "raw", data, when=when)

    expected_path = tmp_path / "2026-07-13" / "prod123" / "raw.jpg"
    assert Path(ref.path) == expected_path.resolve()
    assert expected_path.read_bytes() == data
    assert ref.sha256 == hashlib.sha256(data).hexdigest()
    assert ref.kind == "raw"
    assert ref.mime == "image/jpeg"


async def test_invalid_kind_rejected(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    with pytest.raises(ValueError):
        await store.save_jpeg("p", "bogus", b"x")


async def test_heatmap_kind_allowed(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    ref = await store.save_jpeg("p", "heatmap", b"data")
    assert ref.kind == "heatmap"
    assert Path(ref.path).name == "heatmap.jpg"


def test_encode_jpeg_roundtrips() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    jpeg = encode_jpeg(image)
    assert jpeg[:2] == b"\xff\xd8"  # JPEG SOI marker
    assert len(jpeg) > 0
