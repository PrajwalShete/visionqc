# VisionQC — Training & Inference Runbook

How to train the anomaly-detection models on EC2, export them for the inference
worker, benchmark latency, and start the worker (fake vs. real). Anomalib API
details are in [AI_PIPELINE.md](./AI_PIPELINE.md); the worker's place in the
system is in [ARCHITECTURE.md](./ARCHITECTURE.md).

The **main app + tests never need the ML stack** — the worker runs in `--fake`
mode without it. You only install the `ai` extra on the training/GPU box.

---

## 1. Install the `ai` extra

The `ai` optional-dependency group pins `anomalib[cpu]==2.5.0` (torch 2.6–2.8)
so `uv sync --extra ai` works for local CPU dev on macOS/Linux.

```bash
# Local dev (CPU) — matches pyproject's pinned ai extra:
uv sync --extra ai
```

**On the GPU box (g4dn.xlarge / T4, AWS Deep Learning AMI), override the torch
wheel to the CUDA build.** Do NOT use `[cu130]` — it lands on the broken torch
2.9 ONNX-export path (anomalib #3050). Use `[cu126]`:

```bash
# GPU box: install the CUDA 12.6 torch build over the synced env.
uv sync --extra ai
uv pip install "anomalib[cu126]==2.5.0"      # overrides the CPU torch with cu126
# (add ,openvino if you want the OpenVINO CPU-fallback export path)
```

Verify CUDA is visible:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 2. Train all three demo categories, both models

`scripts/train.py` auto-downloads MVTec AD (no mvtec.com registration — uses the
anomalib mydrive.ch mirror). If the mirror is down it prints a manual-download
URL and exits; extract the archive into `--data-root/<category>/` and re-run.

`TRUST_REMOTE_CODE=1` is required for the export/inference pickle path.

```bash
# Padim (fast, resnet18) for all three categories:
TRUST_REMOTE_CODE=1 uv run python scripts/train.py \
    --category all --model padim \
    --data-root ./datasets/MVTecAD --output-root ./models

# Patchcore (stronger, wide_resnet50_2, coreset 0.1) for all three:
TRUST_REMOTE_CODE=1 uv run python scripts/train.py \
    --category all --model patchcore \
    --data-root ./datasets/MVTecAD --output-root ./models
```

Single category (e.g. iterate on `bottle` only):

```bash
TRUST_REMOTE_CODE=1 uv run python scripts/train.py --category bottle --model padim
```

Each run prints a per-category **image AUROC / pixel AUROC** results table plus a
final training-summary table.

### Expected outputs

For every `(category, model)` the script writes:

```
models/
  <category>/
    <model>/
      model.pt          # ExportType.TORCH export — this is what the worker loads
      metadata.json     # category, model, backbone, thresholds (from
                        # model.post_processor), AUROC results, trained_at,
                        # anomalib_version
  _results/             # anomalib Engine working dir (checkpoints/logs)
```

`models/`, `datasets/`, and `*.pt` are git-ignored (size + MVTec CC BY-NC-SA
license) — never commit them.

`metadata.json` example:

```json
{
  "category": "bottle",
  "model": "padim",
  "backbone": "resnet18",
  "thresholds": {
    "image_threshold": 13.42,
    "pixel_threshold": 12.87,
    "normalized_image_threshold": 0.5
  },
  "results": { "image_AUROC": 0.998, "pixel_AUROC": 0.981 },
  "trained_at": "2026-07-13T12:00:00+00:00",
  "anomalib_version": "2.5.0",
  "export_type": "torch"
}
```

---

## 3. Benchmark inference latency (feeds the G04 gate)

Loads the exported model via `TorchInferencer`, warms up, then times N
inferences and reports p50/p95/p99.

```bash
TRUST_REMOTE_CODE=1 uv run python scripts/benchmark_inference.py \
    --model-path models/bottle/padim/model.pt \
    --iterations 200 --warmup 10
```

Run it on the **target GPU** (T4) — p95/p99 there is what the G04 latency gate is
measured against. `--device cpu` gives a CPU baseline for comparison.

---

## 4. Start the inference worker

The worker (`visionqc_inference.worker`) is the isolated GPU process the main app
talks to over `http://127.0.0.1:8001/infer` (see `VISIONQC_INFERENCE_WORKER_URL`
in `visionqc.config`).

### Fake mode — no GPU, no model, deterministic (demos / CI / local dev)

Serves scores derived from the image-bytes hash + a synthetic gradient heatmap.
Lets the entire pipeline run end-to-end on any machine.

```bash
uv run python -m visionqc_inference.worker --port 8001 --fake
# equivalently:
VISIONQC_WORKER_FAKE=1 uv run python -m visionqc_inference.worker --port 8001
```

### Real mode — loads an exported anomalib model

```bash
TRUST_REMOTE_CODE=1 uv run python -m visionqc_inference.worker \
    --model-path models/bottle/padim/model.pt --port 8001
# --device defaults to "auto" (CUDA when present); override with --device cpu.
```

### Smoke-check either mode

```bash
curl -s http://127.0.0.1:8001/health
# {"status":"ok","model_version":"...","warmed_up":true,"device":"cpu"|"cuda:0"}

curl -s -X POST http://127.0.0.1:8001/infer \
    -H "Content-Type: application/octet-stream" \
    --data-binary @some_frame.jpg
# {"score":0-1,"heatmap_jpeg_b64":"...","latency_ms":..,"model_version":".."}
```

`/health.warmed_up` flips to `true` only after the startup warm-up inference
succeeds — mirror this in the systemd `READY=1` readiness gate so the app only
routes traffic once the model's first (slow) forward pass is done.

---

## 5. Production (systemd, per ARCHITECTURE.md §5)

Run the worker as `qc-inference.service` with `Restart=on-failure`,
`RestartSec=2`, a `StartLimitBurst` cap, and readiness gated on `/health`
returning `warmed_up: true`. The app tolerates worker absence — inference
failures become FAULT dispositions with a "degraded" banner rather than a hang.
