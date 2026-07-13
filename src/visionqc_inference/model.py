"""Model backends for the inference worker.

Two implementations behind one tiny interface:

* :class:`FakeModel` — no ML deps. Deterministic score from the image-bytes
  hash + a synthetic gradient heatmap. Lets the whole demo run end-to-end on
  any machine with zero GPU/model, and is what the tests use.
* :class:`AnomalibModel` — loads an exported ``.pt`` (from anomalib's
  ``ExportType.TORCH``) via ``TorchInferencer``. **anomalib/torch are imported
  lazily inside the loader**, so importing this module never requires the ``ai``
  extra. Only constructing an :class:`AnomalibModel` pulls anomalib in.

Both return a normalized score in ``[0, 1]`` (anomalib convention: ``>= 0.5``
is anomalous) plus an overlay heatmap as JPEG bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from . import heatmap as hm

__all__ = ["AnomalibModel", "FakeModel", "InferenceModel", "ModelResult"]


class ModelResult:
    """Result of a single inference: score, overlay JPEG, and model version."""

    __slots__ = ("heatmap_jpeg", "model_version", "score")

    def __init__(self, score: float, heatmap_jpeg: bytes, model_version: str) -> None:
        self.score = float(score)
        self.heatmap_jpeg = heatmap_jpeg
        self.model_version = model_version


@runtime_checkable
class InferenceModel(Protocol):
    """Interface the worker depends on."""

    version: str
    device: str

    def infer(self, image_bytes: bytes) -> ModelResult:
        """Run inference on JPEG ``image_bytes``."""
        ...

    def warmup(self) -> None:
        """Run one dummy inference to page in weights / build CUDA kernels."""
        ...


def _dummy_jpeg(size: int = 256) -> bytes:
    """A neutral gray JPEG used for warm-up inference."""

    gray = np.full((size, size, 3), 127, dtype=np.uint8)
    return hm.encode_jpeg(gray)


def _score_from_hash(image_bytes: bytes) -> float:
    """Stable score in ``[0, 1)`` from the image hash.

    Mirrors ``visionqc.inference_client.FakeInferenceClient.score_for`` so the
    fake worker and the in-process fake client agree on scores for the same
    image — handy for cross-checking the demo.
    """

    digest = hashlib.sha256(image_bytes).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


class FakeModel:
    """Deterministic, dependency-free model for demos and tests."""

    def __init__(self, model_version: str = "fake-1.0") -> None:
        self.version = model_version
        self.device = "cpu"

    def infer(self, image_bytes: bytes) -> ModelResult:
        score = _score_from_hash(image_bytes)

        # Decode the incoming frame for the overlay; fall back to a blank canvas
        # if the payload is not a real image (e.g. tiny synthetic test bytes).
        try:
            base = hm.decode_image(image_bytes)
        except ValueError:
            base = np.full((256, 256, 3), 40, dtype=np.uint8)

        height, width = base.shape[:2]
        seed = int.from_bytes(hashlib.sha256(image_bytes).digest()[4:8], "big")
        anomaly_map = hm.synthetic_anomaly_map(height, width, seed=seed, intensity=score)
        # Fake maps are already in [0, 1] — pin normalization so brightness
        # tracks the score rather than being re-stretched per frame.
        overlay = hm.overlay_anomaly_map(base, anomaly_map, vmin=0.0, vmax=1.0)
        return ModelResult(score=score, heatmap_jpeg=overlay, model_version=self.version)

    def warmup(self) -> None:
        self.infer(_dummy_jpeg())


class AnomalibModel:
    """anomalib ``TorchInferencer`` backend (lazy import of anomalib/torch).

    Loads an exported ``.pt`` produced by ``ExportType.TORCH``. Requires the env
    var ``TRUST_REMOTE_CODE=1`` (anomalib's pickle-load gate); we set it
    defensively if the caller has not. ``device="auto"`` picks CUDA when present.
    """

    def __init__(self, model_path: str | Path, *, device: str = "auto") -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"model file not found: {model_path}")

        # anomalib's TorchInferencer refuses to unpickle without this gate.
        os.environ.setdefault("TRUST_REMOTE_CODE", "1")

        # --- Lazy imports: only happen when a real model is actually loaded. ---
        from anomalib.deploy import TorchInferencer

        self._inferencer = TorchInferencer(path=str(model_path), device=device)
        self._model_path = model_path
        self.device = str(getattr(self._inferencer, "device", device))
        self.version = self._resolve_version(model_path)

    @staticmethod
    def _resolve_version(model_path: Path) -> str:
        """Derive a human model version from a sidecar ``metadata.json`` if any."""

        meta_path = model_path.with_name("metadata.json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                category = meta.get("category", "?")
                model = meta.get("model", "?")
                version = meta.get("anomalib_version", "")
                suffix = f"@anomalib-{version}" if version else ""
                return f"{category}/{model}{suffix}"
            except (json.JSONDecodeError, OSError):
                pass
        # Fall back to <category>/<model> inferred from models/<cat>/<model>/model.pt
        parts = model_path.resolve().parts
        if len(parts) >= 3:
            return f"{parts[-3]}/{parts[-2]}"
        return model_path.stem

    def infer(self, image_bytes: bytes) -> ModelResult:
        import io

        from PIL import Image

        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        prediction = self._inferencer.predict(pil)

        score = self._to_float(prediction.pred_score)
        anomaly_map = self._to_numpy_2d(prediction.anomaly_map)

        base = hm.decode_image(image_bytes)
        # anomalib scores/maps are min-max normalized into [0, 1] (0.5 = thresh),
        # so pin the colormap to that fixed range.
        overlay = hm.overlay_anomaly_map(base, anomaly_map, vmin=0.0, vmax=1.0)
        return ModelResult(score=score, heatmap_jpeg=overlay, model_version=self.version)

    def warmup(self) -> None:
        self.infer(_dummy_jpeg())

    @staticmethod
    def _to_float(value: object) -> float:
        """Coerce a torch tensor / numpy scalar / python number to a float."""

        return float(_to_ndarray(value).reshape(-1)[0])

    @staticmethod
    def _to_numpy_2d(value: object) -> np.ndarray:
        return np.squeeze(_to_ndarray(value))


def _to_ndarray(value: object) -> np.ndarray:
    """Convert a torch tensor (CPU/GPU, maybe grad) or array-like to ``float32``.

    Duck-typed so this module needs no torch import.
    """

    if hasattr(value, "detach"):  # torch.Tensor
        value = value.detach().cpu().numpy()  # type: ignore[union-attr]
    return np.asarray(value, dtype=np.float32)
