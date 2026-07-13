# VisionQC — Industrial AI Vision Quality-Control Platform

An **edge-first, fail-safe AI vision inspection system** for production lines: every product that passes the camera gets a deterministic **PASS / REJECT / FAULT** decision, full traceability, and evidence imagery — with no silent failures.

> **Status:** Active development — demo milestone targeting anomaly detection on the MVTec AD benchmark (bottle, hazelnut, metal_nut) with a simulated production line and live operator dashboard.

## Core principles

- **Deterministic product lifecycle** — every trigger creates a Product ID and *must* end in PASS, REJECT, or explicit FAULT. Zero silent loss.
- **Fail-safe by default** — camera loss, model failure, or storage fault raises a timestamped alarm and a safe action, never a silent pass.
- **Edge-first** — inspection, decision, rejection, and traceability run fully offline; no cloud dependency.
- **Configuration over code** — new products are commissioned through recipes, not source-code changes.
- **Complete traceability** — every product record stores decision, defects, evidence images, and the exact recipe/model versions that produced it.

## Architecture (demo milestone)

```
Virtual Camera ──trigger──▶ Product Lifecycle ──frame──▶ AI Inference (anomalib)
     │                            │                            │ anomaly map
     │                            ▼                            ▼
 Fault injection            Event Bus ◀──────────── Decision Engine (PASS/REJECT/FAULT)
                                 │                            │
                     ┌───────────┼────────────┐               ▼
                     ▼           ▼            ▼        Virtual Reject Station
              Traceability   Alarms      WebSocket            │
              (records +                 Dashboard ◀── reject confirmation
               evidence)
```

- **AI:** [anomalib](https://github.com/open-edge-platform/anomalib) — PaDiM (fast baseline) and PatchCore (high-accuracy showcase), trained on good samples only
- **Dataset:** [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) (research/demo use — CC BY-NC-SA 4.0, see [Licensing note](#licensing-note))
- **Backend:** Python / FastAPI, async event bus, SQLite (WAL) product records
- **Dashboard:** live inspection view with defect heatmap overlays, counters, alarms, traceability search, recipe versioning

## Repository layout

```
visionqc/
├── src/visionqc/       # application source
├── scripts/            # training, data prep, benchmarks
├── tests/              # pytest suites
├── docs/               # architecture & demo runbook
└── deploy/             # EC2 provisioning & service units
```

## Licensing note

The MVTec AD dataset is licensed **CC BY-NC-SA 4.0 (non-commercial)** and is **not** included in this repository. It is used here strictly for research and technical demonstration. Trained model weights and dataset files are excluded via `.gitignore`.

---

© 2026. All rights reserved.
