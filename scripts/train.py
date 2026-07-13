#!/usr/bin/env python
"""Train an anomalib model on MVTec AD and export it for the inference worker.

Runs on the EC2 GPU box with the ``ai`` extra installed (see docs/TRAINING.md).
Per docs/AI_PIPELINE.md (anomalib v2.5.0):

* datamodule: ``MVTecAD`` (auto-downloads, no registration)
* models: ``Padim(resnet18)`` or ``Patchcore(wide_resnet50_2, coreset 0.1)``
* ``Engine.fit`` + ``Engine.test`` → capture image/pixel AUROC
* export ``ExportType.TORCH`` → ``models/<category>/<model>/model.pt``
* write ``metadata.json`` next to it (thresholds, AUROC, versions, timestamp)

Example:

    TRUST_REMOTE_CODE=1 uv run python scripts/train.py \
        --category all --model padim --data-root ./datasets/MVTecAD \
        --output-root ./models
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

CATEGORIES = ("bottle", "hazelnut", "metal_nut")
MVTEC_MANUAL_URL = (
    "https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads "
    "(or the anomalib mydrive.ch mirror) — extract into --data-root/<category>/"
)


def _build_model(model_name: str):
    """Instantiate the requested anomalib model with docs-specified backbones."""

    from anomalib.models import Padim, Patchcore

    if model_name == "padim":
        return Padim(
            backbone="resnet18",
            layers=["layer1", "layer2", "layer3"],
            pre_trained=True,
        )
    if model_name == "patchcore":
        return Patchcore(
            backbone="wide_resnet50_2",
            layers=["layer2", "layer3"],
            coreset_sampling_ratio=0.1,
            pre_trained=True,
        )
    raise ValueError(f"unknown model {model_name!r}; choose padim or patchcore")


def _model_backbone(model_name: str) -> str:
    return "resnet18" if model_name == "padim" else "wide_resnet50_2"


def _build_datamodule(category: str, data_root: str):
    """Build the MVTecAD datamodule, wrapping first-use download in try/except."""

    from anomalib.data import MVTecAD

    datamodule = MVTecAD(
        root=data_root,
        category=category,
        train_batch_size=32,
        eval_batch_size=32,
    )
    try:
        datamodule.prepare_data()  # triggers auto-download + hashsum check
    except Exception as exc:
        print(
            f"\n[train] MVTec AD auto-download failed for {category!r}: {exc}\n"
            f"[train] Download manually from:\n    {MVTEC_MANUAL_URL}\n",
            file=sys.stderr,
        )
        raise
    return datamodule


def _extract_auroc(test_results: object) -> dict[str, float | None]:
    """Pull image/pixel AUROC out of ``Engine.test`` results (list of dicts)."""

    metrics: dict[str, float] = {}
    if isinstance(test_results, list):
        for entry in test_results:
            if isinstance(entry, dict):
                metrics.update(entry)
    elif isinstance(test_results, dict):
        metrics.update(test_results)

    def _find(*needles: str) -> float | None:
        for key, value in metrics.items():
            lowered = key.lower()
            if all(n in lowered for n in needles):
                try:
                    return round(float(value), 4)
                except (TypeError, ValueError):
                    return None
        return None

    return {
        "image_AUROC": _find("image", "auroc"),
        "pixel_AUROC": _find("pixel", "auroc"),
    }


def _extract_thresholds(model: object) -> dict[str, float | None]:
    """Read thresholds off ``model.post_processor`` as plain floats."""

    post = getattr(model, "post_processor", None)
    if post is None:
        return {}

    def _as_float(name: str) -> float | None:
        value = getattr(post, name, None)
        if value is None:
            return None
        try:
            return float(value.item() if hasattr(value, "item") else value)
        except (TypeError, ValueError):
            return None

    return {
        "image_threshold": _as_float("image_threshold"),
        "pixel_threshold": _as_float("pixel_threshold"),
        "normalized_image_threshold": _as_float("normalized_image_threshold"),
    }


def _anomalib_version() -> str:
    try:
        from importlib.metadata import version

        return version("anomalib")
    except Exception:
        return "unknown"


def train_one(
    category: str,
    model_name: str,
    data_root: str,
    output_root: Path,
) -> dict[str, object]:
    """Train, test, and export one (category, model). Returns a summary row."""

    from anomalib.deploy import ExportType
    from anomalib.engine import Engine

    print(f"\n{'=' * 70}\n[train] {model_name} on {category}\n{'=' * 70}")

    datamodule = _build_datamodule(category, data_root)
    model = _build_model(model_name)

    results_dir = output_root / "_results"
    engine = Engine(max_epochs=1, default_root_dir=str(results_dir))
    engine.fit(model=model, datamodule=datamodule)
    test_results = engine.test(model=model, datamodule=datamodule)

    auroc = _extract_auroc(test_results)
    thresholds = _extract_thresholds(model)

    # Export to Torch and place at models/<category>/<model>/model.pt
    target_dir = output_root / category / model_name
    target_dir.mkdir(parents=True, exist_ok=True)
    exported = engine.export(model=model, export_type=ExportType.TORCH)
    target_pt = target_dir / "model.pt"
    if exported is not None and Path(exported).exists():
        shutil.copy2(Path(exported), target_pt)
    else:
        print(f"[train] WARNING: export returned {exported!r}; model.pt may be missing")

    metadata = {
        "category": category,
        "model": model_name,
        "backbone": _model_backbone(model_name),
        "thresholds": thresholds,
        "results": auroc,
        "trained_at": datetime.now(UTC).isoformat(),
        "anomalib_version": _anomalib_version(),
        "export_type": "torch",
    }
    (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[train] exported → {target_pt}")
    print(f"[train] metadata → {target_dir / 'metadata.json'}")

    return {
        "category": category,
        "model": model_name,
        "image_AUROC": auroc["image_AUROC"],
        "pixel_AUROC": auroc["pixel_AUROC"],
        "path": str(target_pt),
    }


def _print_summary(rows: list[dict[str, object]]) -> None:
    print(f"\n{'=' * 70}\nTRAINING SUMMARY\n{'=' * 70}")
    header = f"{'category':<12} {'model':<10} {'image AUROC':>12} {'pixel AUROC':>12}"
    print(header)
    print("-" * len(header))
    for row in rows:
        img = row["image_AUROC"]
        pix = row["pixel_AUROC"]
        img_s = f"{img:.4f}" if isinstance(img, float) else "n/a"
        pix_s = f"{pix:.4f}" if isinstance(pix, float) else "n/a"
        print(f"{row['category']:<12} {row['model']:<10} {img_s:>12} {pix_s:>12}")
    print("=" * 70)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category",
        choices=[*CATEGORIES, "all"],
        default="bottle",
        help="MVTec category, or 'all' for the three demo categories.",
    )
    parser.add_argument(
        "--model",
        choices=["padim", "patchcore"],
        default="padim",
        help="anomaly model to train.",
    )
    parser.add_argument(
        "--data-root",
        default="./datasets/MVTecAD",
        help="dataset root (auto-downloaded here if missing).",
    )
    parser.add_argument(
        "--output-root",
        default="./models",
        help="where models/<category>/<model>/model.pt is written.",
    )
    args = parser.parse_args(argv)

    categories = list(CATEGORIES) if args.category == "all" else [args.category]
    output_root = Path(args.output_root)

    rows: list[dict[str, object]] = []
    for category in categories:
        rows.append(train_one(category, args.model, args.data_root, output_root))

    _print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
