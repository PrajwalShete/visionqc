"""Unit tests for the pure heatmap functions (numpy-only, no anomalib)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from visionqc_inference import heatmap as hm

# --- normalize_map ------------------------------------------------------------


def test_normalize_map_scales_to_unit_range() -> None:
    amap = np.array([[0.0, 5.0], [10.0, 2.5]], dtype=np.float32)
    out = hm.normalize_map(amap)
    assert out.dtype == np.float32
    assert out.shape == (2, 2)
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)
    # 5.0 sits halfway between 0 and 10.
    assert out[0, 1] == pytest.approx(0.5)


def test_normalize_map_flat_input_returns_zeros() -> None:
    amap = np.full((4, 4), 7.0, dtype=np.float32)
    out = hm.normalize_map(amap)
    assert np.all(out == 0.0)
    assert out.shape == (4, 4)


def test_normalize_map_squeezes_batch_and_channel_axes() -> None:
    amap = np.random.default_rng(0).random((1, 1, 8, 8)).astype(np.float32)
    out = hm.normalize_map(amap)
    assert out.ndim == 2
    assert out.shape == (8, 8)


def test_normalize_map_respects_explicit_range() -> None:
    amap = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    out = hm.normalize_map(amap, vmin=0.0, vmax=1.0)
    # Values already in [0,1] with a pinned range pass through unchanged.
    assert out[0, 0] == pytest.approx(0.0)
    assert out[1, 0] == pytest.approx(1.0)
    assert out[0, 1] == pytest.approx(0.5)


def test_normalize_map_clips_out_of_range_values() -> None:
    amap = np.array([[-1.0, 2.0]], dtype=np.float32)
    out = hm.normalize_map(amap, vmin=0.0, vmax=1.0)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_normalize_map_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        hm.normalize_map(np.zeros((3, 4, 5), dtype=np.float32))


# --- apply_colormap -----------------------------------------------------------


def test_apply_colormap_returns_bgr_uint8() -> None:
    normalized = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    out = hm.apply_colormap(normalized)
    assert out.dtype == np.uint8
    assert out.shape == (8, 8, 3)


def test_apply_colormap_monotone_input_is_uniform() -> None:
    normalized = np.zeros((5, 5), dtype=np.float32)
    out = hm.apply_colormap(normalized)
    # Every pixel maps to the same color for a constant input.
    assert np.all(out == out[0, 0])


def test_apply_colormap_accepts_uint8() -> None:
    u8 = np.arange(256, dtype=np.uint8).reshape(16, 16)
    out = hm.apply_colormap(u8, colormap=cv2.COLORMAP_TURBO)
    assert out.shape == (16, 16, 3)
    assert out.dtype == np.uint8


# --- blend_overlay ------------------------------------------------------------


def test_blend_overlay_shape_and_dtype() -> None:
    base = np.full((32, 32, 3), 100, dtype=np.uint8)
    heat = np.full((32, 32, 3), 200, dtype=np.uint8)
    out = hm.blend_overlay(base, heat, alpha=0.5)
    assert out.shape == base.shape
    assert out.dtype == np.uint8
    # 0.5*200 + 0.5*100 = 150.
    assert np.allclose(out, 150, atol=1)


def test_blend_overlay_alpha_zero_returns_base() -> None:
    base = np.full((16, 16, 3), 80, dtype=np.uint8)
    heat = np.full((16, 16, 3), 255, dtype=np.uint8)
    out = hm.blend_overlay(base, heat, alpha=0.0)
    assert np.allclose(out, base, atol=1)


def test_blend_overlay_alpha_one_returns_heatmap() -> None:
    base = np.full((16, 16, 3), 80, dtype=np.uint8)
    heat = np.full((16, 16, 3), 255, dtype=np.uint8)
    out = hm.blend_overlay(base, heat, alpha=1.0)
    assert np.allclose(out, heat, atol=1)


def test_blend_overlay_resizes_mismatched_heatmap() -> None:
    base = np.full((64, 48, 3), 100, dtype=np.uint8)
    heat = np.full((16, 16, 3), 200, dtype=np.uint8)  # lower resolution
    out = hm.blend_overlay(base, heat, alpha=0.5)
    assert out.shape == (64, 48, 3)


def test_blend_overlay_rejects_bad_alpha() -> None:
    base = np.zeros((4, 4, 3), dtype=np.uint8)
    heat = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        hm.blend_overlay(base, heat, alpha=1.5)


# --- encode_jpeg / decode_image round-trip -----------------------------------


def test_encode_jpeg_returns_valid_jpeg_bytes() -> None:
    image = np.full((24, 24, 3), 128, dtype=np.uint8)
    data = hm.encode_jpeg(image)
    assert isinstance(data, bytes)
    assert len(data) > 0
    # JPEG SOI / EOI markers.
    assert data[:2] == b"\xff\xd8"
    assert data[-2:] == b"\xff\xd9"


def test_encode_decode_round_trip_preserves_shape() -> None:
    # A smooth gradient (not high-frequency noise) survives lossy JPEG well.
    ys = np.linspace(0, 255, 40, dtype=np.float32)[:, None]
    xs = np.linspace(0, 255, 56, dtype=np.float32)[None, :]
    channel = ((ys + xs) / 2.0).astype(np.uint8)
    image = np.repeat(channel[:, :, None], 3, axis=2)
    decoded = hm.decode_image(hm.encode_jpeg(image, quality=95))
    assert decoded.shape == image.shape
    assert decoded.dtype == np.uint8
    # JPEG is lossy; assert approximate reconstruction rather than equality.
    assert float(np.mean(np.abs(decoded.astype(int) - image.astype(int)))) < 5.0


def test_decode_image_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        hm.decode_image(b"not-an-image")


# --- synthetic_anomaly_map ---------------------------------------------------


def test_synthetic_anomaly_map_range_and_shape() -> None:
    m = hm.synthetic_anomaly_map(30, 40, seed=3)
    assert m.shape == (30, 40)
    assert m.dtype == np.float32
    assert m.min() >= 0.0
    assert m.max() <= 1.0


def test_synthetic_anomaly_map_is_deterministic() -> None:
    a = hm.synthetic_anomaly_map(20, 20, seed=42, intensity=0.7)
    b = hm.synthetic_anomaly_map(20, 20, seed=42, intensity=0.7)
    assert np.array_equal(a, b)


def test_synthetic_anomaly_map_seed_changes_output() -> None:
    a = hm.synthetic_anomaly_map(20, 20, seed=1)
    b = hm.synthetic_anomaly_map(20, 20, seed=999)
    assert not np.array_equal(a, b)


def test_synthetic_anomaly_map_rejects_bad_size() -> None:
    with pytest.raises(ValueError):
        hm.synthetic_anomaly_map(0, 10)


# --- overlay_anomaly_map (end-to-end) ----------------------------------------


def test_overlay_anomaly_map_produces_jpeg() -> None:
    base = np.full((48, 48, 3), 90, dtype=np.uint8)
    amap = hm.synthetic_anomaly_map(48, 48, seed=7, intensity=0.9)
    data = hm.overlay_anomaly_map(base, amap)
    assert data[:2] == b"\xff\xd8"
    # The overlay decodes back to the base resolution.
    decoded = hm.decode_image(data)
    assert decoded.shape == base.shape


def test_overlay_anomaly_map_low_res_map_over_full_frame() -> None:
    base = np.full((64, 64, 3), 120, dtype=np.uint8)
    amap = hm.synthetic_anomaly_map(16, 16, seed=2)  # coarse map
    data = hm.overlay_anomaly_map(base, amap, vmin=0.0, vmax=1.0)
    decoded = hm.decode_image(data)
    assert decoded.shape == (64, 64, 3)
