#!/usr/bin/env python
"""Benchmark exported-model inference latency (feeds the G04 latency gate).

Loads an exported anomalib ``.pt`` via ``TorchInferencer``, runs a warm-up
burst, then N timed single-image inferences and reports p50/p95/p99.

Example:

    TRUST_REMOTE_CODE=1 uv run python scripts/benchmark_inference.py \
        --model-path models/bottle/padim/model.pt --iterations 200
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

import numpy as np


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (pct in [0, 100])."""

    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _make_test_frame(size: int) -> object:
    """A PIL RGB image the TorchInferencer can consume."""

    from PIL import Image

    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="exported anomalib .pt path")
    parser.add_argument("--iterations", type=int, default=200, help="timed inferences (def 200)")
    parser.add_argument("--warmup", type=int, default=10, help="warm-up inferences (default 10)")
    parser.add_argument("--device", default="auto", help="TorchInferencer device (default auto)")
    parser.add_argument("--image-size", type=int, default=256, help="synthetic frame size")
    args = parser.parse_args(argv)

    os.environ.setdefault("TRUST_REMOTE_CODE", "1")

    from anomalib.deploy import TorchInferencer

    model_path = Path(args.model_path)
    if not model_path.exists():
        parser.error(f"model not found: {model_path}")

    print(f"[bench] loading {model_path} on device={args.device} ...")
    inferencer = TorchInferencer(path=str(model_path), device=args.device)
    frame = _make_test_frame(args.image_size)

    print(f"[bench] warm-up: {args.warmup} inferences ...")
    for _ in range(args.warmup):
        inferencer.predict(frame)

    print(f"[bench] timing: {args.iterations} inferences ...")
    latencies: list[float] = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        inferencer.predict(frame)
        latencies.append((time.perf_counter() - start) * 1000.0)

    device = str(getattr(inferencer, "device", args.device))
    print(f"\n{'=' * 56}")
    print(f"Inference latency — {model_path.name} on {device}")
    print(f"iterations={args.iterations}  image_size={args.image_size}")
    print(f"{'=' * 56}")
    print(f"  mean : {statistics.mean(latencies):8.2f} ms")
    print(f"  p50  : {_percentile(latencies, 50):8.2f} ms")
    print(f"  p95  : {_percentile(latencies, 95):8.2f} ms")
    print(f"  p99  : {_percentile(latencies, 99):8.2f} ms")
    print(f"  min  : {min(latencies):8.2f} ms")
    print(f"  max  : {max(latencies):8.2f} ms")
    print(f"{'=' * 56}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
