"""Standalone FastAPI inference worker (the isolated GPU process).

Serves the HTTP contract consumed by
``visionqc.inference_client.HTTPInferenceClient``:

* ``POST /infer`` — raw image bytes in the request body → JSON
  ``{score, heatmap_jpeg_b64, latency_ms, model_version}``.
* ``GET /health`` — ``{status, model_version, warmed_up, device}``.

The model backend is selected at startup:

* ``--fake`` / ``VISIONQC_WORKER_FAKE=1`` → :class:`~visionqc_inference.model.FakeModel`
  (no GPU, no anomalib, deterministic — the demo/test path).
* otherwise ``--model-path PATH`` loads a real anomalib model
  (:class:`~visionqc_inference.model.AnomalibModel`).

A warm-up inference runs during lifespan startup; ``/health`` reports
``warmed_up`` only after it succeeds — mirroring the systemd ``READY=1`` gate
described in ARCHITECTURE.md.

This module imports cleanly WITHOUT anomalib installed: the real backend is
imported lazily only when a non-fake model is requested.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .model import InferenceModel

__all__ = ["build_model", "create_app", "main"]


@dataclass
class WorkerState:
    """Mutable worker state shared across requests."""

    model: InferenceModel
    warmed_up: bool = field(default=False)


def build_model(
    *,
    fake: bool,
    model_path: str | None,
    device: str = "auto",
    model_version: str | None = None,
) -> InferenceModel:
    """Construct the model backend from resolved worker options.

    ``fake`` forces :class:`FakeModel`. Otherwise ``model_path`` is required and
    an :class:`AnomalibModel` is loaded (which lazily imports anomalib).
    """

    if fake:
        from .model import FakeModel

        return FakeModel(model_version=model_version or "fake-1.0")

    if not model_path:
        raise ValueError("a --model-path is required when not running in --fake mode")

    from .model import AnomalibModel

    return AnomalibModel(model_path, device=device)


def create_app(model: InferenceModel) -> FastAPI:
    """Build the FastAPI app around a ready :class:`InferenceModel`.

    Warm-up runs in the lifespan startup so an in-process ``TestClient`` (and a
    real uvicorn boot) both gate ``/health.warmed_up`` on a successful dummy
    inference.
    """

    state = WorkerState(model=model)
    # Model access is serialized: TorchInferencer is not guaranteed thread-safe
    # and a single CUDA context must not see concurrent forward passes. Running
    # inference in a worker thread (under this lock) keeps /health responsive
    # even while a slow or wedged inference is in flight.
    infer_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            await asyncio.to_thread(model.warmup)
            state.warmed_up = True
        except Exception as exc:
            # Leave warmed_up=False; /health will report not-ready and the
            # supervisor keeps the app up so the dashboard degrades gracefully.
            print(f"[worker] warm-up inference failed: {exc!r}")
        yield

    app = FastAPI(title="VisionQC Inference Worker", version="0.1.0", lifespan=lifespan)
    app.state.worker = state

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok" if state.warmed_up else "starting",
            "model_version": state.model.version,
            "warmed_up": state.warmed_up,
            "device": state.model.device,
        }

    @app.post("/infer")
    async def infer(request: Request) -> JSONResponse:
        image = await request.body()
        if not image:
            return JSONResponse(status_code=400, content={"error": "empty request body"})

        start = time.perf_counter()
        try:
            async with infer_lock:
                result = await asyncio.to_thread(state.model.infer, image)
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": f"inference failed: {exc}"})
        latency_ms = (time.perf_counter() - start) * 1000.0

        return JSONResponse(
            content={
                "score": result.score,
                "heatmap_jpeg_b64": base64.b64encode(result.heatmap_jpeg).decode("ascii"),
                "latency_ms": latency_ms,
                "model_version": result.model_version,
            }
        )

    return app


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m visionqc_inference.worker",
        description="VisionQC isolated GPU inference worker.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8001, help="bind port (default 8001)")
    parser.add_argument(
        "--fake",
        action="store_true",
        default=_env_flag("VISIONQC_WORKER_FAKE"),
        help="serve deterministic fake scores + synthetic heatmap (no GPU/model).",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("VISIONQC_MODEL_PATH"),
        help="path to an exported anomalib .pt (ExportType.TORCH). Ignored in --fake.",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("VISIONQC_DEVICE", "auto"),
        help="anomalib TorchInferencer device (default 'auto').",
    )
    parser.add_argument("--log-level", default="info", help="uvicorn log level")
    return parser.parse_args(argv)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint: build the model, then serve with uvicorn."""

    args = _parse_args(argv)
    model = build_model(fake=args.fake, model_path=args.model_path, device=args.device)
    app = create_app(model)

    import uvicorn

    mode = "fake" if args.fake else f"model={args.model_path}"
    print(f"[worker] starting on {args.host}:{args.port} ({mode}, version={model.version})")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
