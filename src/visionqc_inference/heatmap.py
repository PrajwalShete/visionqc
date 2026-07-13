"""Pure heatmap functions — anomaly map → colored overlay → JPEG bytes.

These functions operate on plain :mod:`numpy` arrays only. There is **no**
``anomalib`` / torch dependency here, which keeps them trivially unit-testable
and importable in any environment that has ``numpy`` + ``opencv``.

Pipeline (composed by :func:`overlay_anomaly_map`):

    normalize_map → apply_colormap → blend_overlay → encode_jpeg

Conventions:

* Images are OpenCV **BGR** ``uint8`` arrays of shape ``(H, W, 3)``.
* An *anomaly map* is a 2-D float array of arbitrary scale; it is min-max
  normalized into ``[0, 1]`` before colorizing. A supplied ``(vmin, vmax)``
  pins normalization to a model's calibrated range (e.g. anomalib's 0-1 space
  with ``vmin=0, vmax=1``) so colors are comparable across frames.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "apply_colormap",
    "blend_overlay",
    "decode_image",
    "encode_jpeg",
    "normalize_map",
    "overlay_anomaly_map",
    "synthetic_anomaly_map",
]

# Default overlay tuning — a translucent JET heatmap over the source frame.
DEFAULT_ALPHA = 0.45
DEFAULT_COLORMAP = cv2.COLORMAP_JET
DEFAULT_JPEG_QUALITY = 90


def normalize_map(
    anomaly_map: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Min-max normalize an anomaly map into ``float32`` values in ``[0, 1]``.

    Extra singleton dimensions (e.g. a leading batch/channel axis from a torch
    tensor of shape ``(1, 1, H, W)``) are squeezed away. When the map is flat
    (``vmax - vmin`` below epsilon) a zero array is returned instead of dividing
    by ~0. Pass explicit ``vmin`` / ``vmax`` to normalize against a fixed range.
    """

    array = np.asarray(anomaly_map, dtype=np.float32)
    # Drop singleton axes (e.g. a leading batch/channel from a (1, 1, H, W)
    # tensor) one at a time, but never collapse a genuine 2-D map to 1-D.
    while array.ndim > 2:
        squeezable = [ax for ax, size in enumerate(array.shape) if size == 1]
        if not squeezable:
            raise ValueError(f"anomaly map must reduce to 2-D, got shape {array.shape!r}")
        array = np.squeeze(array, axis=squeezable[0])
    if array.ndim != 2:
        raise ValueError(f"anomaly map must reduce to 2-D, got shape {array.shape!r}")

    lo = float(array.min()) if vmin is None else float(vmin)
    hi = float(array.max()) if vmax is None else float(vmax)
    if hi - lo < 1e-12:
        return np.zeros_like(array, dtype=np.float32)

    normalized = (array - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def apply_colormap(
    normalized: np.ndarray,
    *,
    colormap: int = DEFAULT_COLORMAP,
) -> np.ndarray:
    """Colorize a ``[0, 1]`` normalized map into a BGR ``uint8`` heatmap.

    Input may be float in ``[0, 1]`` or already ``uint8``; either way it is
    scaled to the 0-255 range OpenCV colormaps expect.
    """

    array = np.asarray(normalized)
    if array.ndim != 2:
        raise ValueError(f"normalized map must be 2-D, got shape {array.shape!r}")

    if array.dtype == np.uint8:
        scaled = array
    else:
        scaled = (np.clip(array.astype(np.float32), 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return cv2.applyColorMap(scaled, colormap)


def blend_overlay(
    base_bgr: np.ndarray,
    heatmap_bgr: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> np.ndarray:
    """Alpha-blend ``heatmap_bgr`` over ``base_bgr`` (``alpha`` = heatmap weight).

    The heatmap is resized to the base frame's resolution when they differ, so
    a low-resolution anomaly map overlays a full-resolution camera frame
    cleanly. Result is a BGR ``uint8`` array the size of ``base_bgr``.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if base_bgr.ndim != 3 or base_bgr.shape[2] != 3:
        raise ValueError(f"base image must be HxWx3 BGR, got shape {base_bgr.shape!r}")

    base = base_bgr if base_bgr.dtype == np.uint8 else base_bgr.astype(np.uint8)
    heatmap = heatmap_bgr
    if heatmap.shape[:2] != base.shape[:2]:
        # cv2.resize takes (width, height).
        heatmap = cv2.resize(
            heatmap,
            (base.shape[1], base.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    if heatmap.dtype != np.uint8:
        heatmap = heatmap.astype(np.uint8)

    return cv2.addWeighted(heatmap, alpha, base, 1.0 - alpha, 0.0)


def encode_jpeg(image_bgr: np.ndarray, *, quality: int = DEFAULT_JPEG_QUALITY) -> bytes:
    """Encode a BGR ``uint8`` image to JPEG bytes."""

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 BGR image, got shape {image_bgr.shape!r}")
    image = image_bgr if image_bgr.dtype == np.uint8 else image_bgr.astype(np.uint8)

    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed to encode heatmap JPEG")
    return buffer.tobytes()


def decode_image(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG (or any cv2-readable) bytes into a BGR ``uint8`` array."""

    array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode image bytes")
    return image


def synthetic_anomaly_map(
    height: int,
    width: int,
    *,
    seed: int = 0,
    intensity: float = 1.0,
) -> np.ndarray:
    """Deterministic pseudo-anomaly field for the worker's ``--fake`` mode.

    Produces a smooth ``float32`` map in ``[0, 1]`` with a Gaussian "hot spot"
    whose center is placed from ``seed`` plus a mild diagonal gradient — a
    plausible-looking heatmap for demos with zero model. Same inputs → same map.
    """

    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got ({height}, {width})")

    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]

    # Hash the seed into a stable blob center within [0.2, 0.8].
    cy = 0.2 + 0.6 * (((seed * 2654435761) >> 8) & 0xFF) / 255.0
    cx = 0.2 + 0.6 * (((seed * 40503) >> 4) & 0xFF) / 255.0
    sigma = 0.18

    blob = np.exp(-(((ys - cy) ** 2) + ((xs - cx) ** 2)) / (2.0 * sigma * sigma))
    gradient = 0.15 * (xs + ys) / 2.0
    field = np.clip(intensity * blob + gradient, 0.0, 1.0)
    return field.astype(np.float32)


def overlay_anomaly_map(
    base_bgr: np.ndarray,
    anomaly_map: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
    colormap: int = DEFAULT_COLORMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> bytes:
    """End-to-end: normalize → colorize → blend → JPEG-encode. Returns bytes."""

    normalized = normalize_map(anomaly_map, vmin=vmin, vmax=vmax)
    heatmap = apply_colormap(normalized, colormap=colormap)
    blended = blend_overlay(base_bgr, heatmap, alpha=alpha)
    return encode_jpeg(blended, quality=quality)
