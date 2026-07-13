# AI Pipeline Reference — anomalib v2.5.0 (verified)

Verified against PyPI metadata, GitHub tag `lib/v2.5.0`, and `anomalib.readthedocs.io/en/lib-v2.5.0/`. Items marked *inferred* need a quick source check before relying on them.

## Installation

Python >= 3.10. **Use the extras — they pin torch correctly and avoid the torch 2.9 ONNX export bug (#3050):**

```bash
# CPU (macOS arm64 local dev)
uv pip install "anomalib[cpu]"            # torch >=2.6,<=2.8

# CUDA 12.x (T4 / AWS DLAMI)
uv pip install "anomalib[cu126]"          # torch >=2.6,<=2.8  — NOT [cu130] (lands on broken torch 2.9)

# add OpenVINO export/inference support
uv pip install "anomalib[cu126,openvino]"
```

If installing torch separately for any reason: pin `torch>=2.6,<2.9`.

## Dataset — auto-download, no registration

```python
from anomalib.data import MVTecAD   # class is MVTecAD, not MVTec

datamodule = MVTecAD(
    root="./datasets/MVTecAD",
    category="bottle",              # demo categories: bottle, hazelnut, metal_nut
    train_batch_size=32,
    eval_batch_size=32,
)
```

- Auto-downloads from a mydrive.ch mirror with hashsum check — **no mvtec.com registration required**. Mirror has broken before in old versions → wrap first use in try/except with a manual-download fallback message.
- **No `image_size` constructor arg in v2** — resizing goes through `augmentations`/`train_augmentations` (torchvision transforms v2) or defaults to the model's pre_processor transform.

## Training

```python
from anomalib.models import Padim, Patchcore     # exact casing: Padim, Patchcore
from anomalib.engine import Engine

model = Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"], pre_trained=True)
# showcase: Patchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3"], coreset_sampling_ratio=0.1)

engine = Engine(max_epochs=1)   # epochs irrelevant — both are single-pass feature/memory-bank methods
engine.fit(model=model, datamodule=datamodule)
engine.test(model=model, datamodule=datamodule)
```

Checkpoints: `results/<model.name>/<dataset_name>/<category>/latest/weights/lightning/model.ckpt` (override root with `Engine(default_root_dir=...)`).

## Thresholding & scores

- `F1AdaptiveThreshold` is the **default** post-processor behavior — computed during validation via PR sweep.
- Output `pred_score` / `anomaly_map` are **min-max normalized to [0,1] where >= 0.5 = anomalous** (raw threshold maps to 0.5).
- Access: `model.post_processor.image_threshold`, `.pixel_threshold`, `.normalized_image_threshold` (0.5), plus `image_min/image_max` buffers.

## Inference (service path)

For the long-running worker, export to Torch format then use `TorchInferencer` (no Lightning Trainer):

```python
from anomalib.deploy import ExportType
engine.export(model=model, export_type=ExportType.TORCH, ckpt_path="...ckpt")  # produces model.pt

from anomalib.deploy import TorchInferencer
inferencer = TorchInferencer(path="path/to/model.pt", device="auto")
pred = inferencer.predict(image)     # path | PIL.Image | torch.Tensor
pred.pred_score    # torch.Tensor, normalized 0-1
pred.anomaly_map   # torch.Tensor heatmap
```

- **Gotcha:** `TorchInferencer` requires env `TRUST_REMOTE_CODE=1` (pickle load gate). Docs call it "legacy" but it is the documented lightweight service path.
- Alternative: `OpenVINOInferencer(path="model.xml", device="CPU")` returns numpy directly (`NumpyImageBatch`) — convenient CPU fallback, needs `openvino` extra.
- `Engine.predict()` with `PredictDataset` is the full-Lightning batch path (fine for eval scripts, not the service).

## Export

```python
from anomalib.deploy import ExportType, CompressionType
engine.export(model=model, export_type=ExportType.ONNX,  input_size=(256, 256), ckpt_path="...")
engine.export(model=model, export_type=ExportType.OPENVINO, compression_type=CompressionType.FP16, ...)
```

**Do NOT assume exported outputs numerically match torch** — known issues (#2303). Validate score parity on a sample set before switching the worker to an exported model.

## Visualization

```python
from anomalib.visualization import visualize_anomaly_map   # standalone, no Engine needed
vis = visualize_anomaly_map(anomaly_map, colormap=True, normalize=True)
```

Return type (numpy vs PIL) *inferred* — check `src/anomalib/visualization/image/functional.py` before use.

## Gotchas summary

1. Install via extras (`[cpu]`/`[cu126]`) — never bare `pip install anomalib`, never `[cu130]`.
2. Class names: `MVTecAD`, `Padim`, `Patchcore` — v0/v1 names/imports are dead.
3. `TorchInferencer` needs `TRUST_REMOTE_CODE=1` and a `.pt` from `ExportType.TORCH` (not `.ckpt`).
4. Scores are normalized 0–1, threshold 0.5 — the decision engine's recipe threshold operates in this space.
5. Repo git tags are `lib/v2.5.0` scheme (monorepo) if ever cloning from source.
6. Warm-up inference at worker startup — first CUDA forward pass takes seconds.
