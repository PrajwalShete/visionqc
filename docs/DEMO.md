# VisionQC — Demo Runbook

## Local run (no GPU, no dataset — fake inference + synthetic products)

```bash
uv sync --extra dev
VISIONQC_INFERENCE_FAKE=1 VISIONQC_SIMULATOR_ENABLED=1 \
  uv run uvicorn --factory visionqc.main:create_app --port 8090
```

Open http://localhost:8090 — then create + activate a recipe (fresh DB fail-safes to FAULT without one):

```bash
curl -X POST localhost:8090/recipes -H 'Content-Type: application/json' \
  -d '{"name":"synthetic-demo","category":"synthetic","model_name":"fake-1.0","anomaly_threshold":0.5}'
curl -X POST localhost:8090/recipes/1/activate
```

## EC2 run (real model, MVTec images)

```bash
# after: uv sync --extra ai  (swap to anomalib[cu126] per docs/TRAINING.md)
uv run python scripts/train.py --category bottle --model patchcore   # once
TRUST_REMOTE_CODE=1 uv run python -m visionqc_inference.worker \
  --model-path models/bottle/patchcore/model.pt --port 8001 &
VISIONQC_SIMULATOR_ENABLED=1 VISIONQC_SIMULATOR_SOURCE_TYPE=directory \
  VISIONQC_SIMULATOR_SOURCE_DIR=datasets/MVTecAD/bottle/test \
  uv run uvicorn --factory visionqc.main:create_app --host 0.0.0.0 --port 8090
```

Create/activate a recipe with the trained threshold (see models/<cat>/<model>/metadata.json).

## The demo script (10 minutes)

1. **Open on the dashboard, line running.** Products flow; PASS/REJECT tiles count; point at the live heatmap overlays localizing real defects. Mention throughput + mean-latency tiles (spec G04).
2. **The zero-silent-loss counter.** "Every trigger ends in PASS, REJECT, or an explicit FAULT — watch this stay at 0 lost, it's computed from the DB, not the UI."
3. **The killer moment — kill the camera.** Flip the CAMERA LOSS toggle. CRITICAL alarms fire instantly, products FAULT (never silently pass), the line keeps running, lost stays 0. Flip it back — instant recovery. "This is what fail-safe means."
4. **Proof of rejection.** Point at a REJECT product's event chain in the feed: RejectCommanded → RejectConfirmed. "We don't assume the rejector fired — we confirm it." Then flip REJECT FAIL: watchdog FAULTs the product with a CRITICAL alarm.
5. **Traceability.** Click any product → full record: state timeline, anomaly score, decision reason, recipe version used, evidence images. "Any customer complaint, we pull the exact product in seconds." (Spec D01–D03.)
6. **Recipes.** Show recipe versioning via /recipes — immutable versions, activation, every product stamped with the recipe that judged it. "New product = new recipe, not new code."
7. **Close on the architecture** (docs/ARCHITECTURE.md): edge-first, inference isolated in its own process (a model crash degrades gracefully — it never freezes the line), single-writer transactional records.

## Rehearsal checklist

- [ ] Fresh DB? Create + activate recipe first (or expect FAULT storm — which is itself demonstrably correct behavior).
- [ ] Worker warmed up (`/health` on 8001 → warmed_up: true) before starting the line.
- [ ] Clear old alarms before the audience arrives (`POST /alarms/{id}/clear`).
- [ ] Set line speed ~2s interval for narration pace; crank to 0.5s for the throughput flex.
- [ ] Have `/health` open in a second tab — zero_silent_loss: true is the closer.
