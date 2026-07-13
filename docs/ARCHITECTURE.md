# VisionQC Architecture (Demo Milestone)

Research-backed decisions for the demo system. Target: single EC2 g4dn.xlarge (T4), solo developer, live-audience reliability.

## 1. Service architecture — modular monolith + one isolated GPU worker

- **One FastAPI process** (`qc_app`, the "line controller") contains all orchestration modules connected by an in-process async event bus. No Redis/NATS/ZeroMQ broker, no microservices.
- **Exactly one separate OS process** for GPU inference (`qc_inference`) — owns the CUDA context and loaded model. The API process talks to it over localhost, wrapping every call in `asyncio.wait_for(..., timeout=N)`.
  - On timeout/worker death: in-flight product → **FAULT** (lifecycle preserved), fail-safe alarm raised, supervisor restarts only the worker. The dashboard never freezes.
  - Rationale: ThreadPoolExecutor is NOT crash isolation (GIL, unkillable threads, shared CUDA context). A hung CUDA call must not take down the app.
- **Event bus:** hand-rolled `asyncio.Queue`-per-subscriber bus (~50 lines). Publisher stamps `event_id`, monotonic + wall-clock `ts`, `type`, `payload`. Each subscriber (WebSocket hub, DB writer, alarm engine, stats) gets its own **bounded** queue — a slow DB writer can never stall the dashboard feed.
- **Deterministic lifecycle:** explicit per-product state machine — `TRIGGERED → CAPTURED → INFERRED → DECIDED → PASS/REJECT`, any error path → `FAULT`. A watchdog task forces FAULT on any product stuck in a non-terminal state. Every state transition is persisted; no event is dropped silently.

## 2. Real-time dashboard — one WebSocket, binary + JSON frames; vanilla frontend

- **Single WebSocket per client, two frame types:** binary frames for JPEG images (camera frame / heatmap evidence), text frames for JSON events. No MJPEG (separate unsynchronized connection), no base64 (~33% bloat + CPU).
- Pair each binary frame with a JSON "frame-meta" message (product_id, camera_id, ts) for correlation.
- **Backpressure (the #1 failure mode):** bounded `asyncio.Queue` per client (maxsize ~3 for frames), **drop-oldest on overflow**, separate send/recv tasks per socket, try/except `WebSocketDisconnect` on every send with immediate ejection from the connection manager — one dead client must never stall the broadcast loop.
- **Frontend: plain HTML + vanilla JS + Tailwind (CDN) + DaisyUI.** No React/Vite — the interaction is "WebSocket pushes images/events into DOM updates," naturally imperative JS. DaisyUI provides `stat`, `badge`, `alert`, status-dot components for pass/reject tiles and alarm banners. Chart.js or uPlot for live charts. Served as static files from FastAPI.

## 3. Data layer — SQLite (WAL), single-writer task, evidence on filesystem

```sql
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA busy_timeout=5000;
PRAGMA cache_size=-65536;
PRAGMA temp_store=MEMORY;
```

- **One dedicated writer connection** fed by the event-bus DB-writer subscriber (which IS the single writer), plus a small pool of read-only connections for dashboard queries. Never share one pool for reads and writes ("database is locked" root cause).
- Write transactions use `BEGIN IMMEDIATE` — `busy_timeout` does not help read→write upgrades.
- **Evidence images: filesystem** organized `evidence/YYYY-MM-DD/<product_id>/`, with path + SHA-256 + MIME + timestamp columns in the DB. No BLOBs in DB (3–5x slower queries, inflated WAL/backups). No S3 in the capture path.
- **Recipe versioning: immutable rows** — `recipes(id, name, version, params_json, created_at)`; never UPDATE a released recipe; every product record stores the `recipe_id` it was inspected under.

## 4. Tooling — uv, src-layout, ruff, pytest

- **uv** for venv/deps/lockfile/run. **src-layout** (`src/visionqc/...`). **ruff** for lint + format. **hatchling** build backend. All config in `pyproject.toml`.
- Tests in top-level `tests/` mirroring modules; `conftest.py` provides `TestClient`, tmp SQLite, fake inference via `app.dependency_overrides`.

## 5. Deployment — native systemd on AWS Deep Learning AMI (no Docker)

- Two units: `qc-app.service` and `qc-inference.service`, each `Restart=on-failure`, `RestartSec=2`, `StartLimitBurst` to prevent restart storms.
- Worker gets `WatchdogSec` + `sd_notify` heartbeat (catches alive-but-wedged CUDA); readiness (`READY=1`) only after model warm-up inference completes.
- Health-check systemd timer curling `/health`, restart on failure.
- App tolerates worker absence: FAULT dispositions + alarm banner, dashboard stays live ("inference degraded" is itself a fail-safe feature to demo).

## Module map

```
src/
├── visionqc/                # main process (FastAPI)
│   ├── main.py              # app factory; lifespan starts bus/subscribers/simulator
│   ├── config.py            # pydantic-settings
│   ├── events/{bus,schemas}.py
│   ├── simulator/           # virtual camera, trigger generator, virtual reject station, fault injection
│   ├── lifecycle/           # product state machine + stuck-product watchdog → FAULT
│   ├── inference_client/    # localhost client; timeouts; degraded mode
│   ├── decision/            # thresholds from active recipe → PASS/REJECT
│   ├── alarms/              # fail-safe alarm rules subscriber
│   ├── recipes/             # immutable versioned recipes
│   ├── db/                  # aiosqlite; single-writer + read pool; migrations
│   ├── evidence/            # image save/load, hashing
│   ├── api/                 # REST routes + ws.py (hub, bounded queues, drop-oldest)
│   └── static/              # index.html, dashboard.js (Tailwind+DaisyUI CDN)
└── visionqc_inference/      # SEPARATE process: owns CUDA context
    ├── worker.py            # localhost server loop; sd_notify heartbeat
    └── model.py             # anomalib model load + heatmap generation
```

## Gotchas (mitigations baked into design)

1. GPU inference must never run in the API process — worker + timeout → FAULT, never a hang.
2. WebSocket backpressure — bounded queues, drop-oldest, eject dead clients on first failed send.
3. SQLite locking — single writer task, `BEGIN IMMEDIATE`, read-only pool.
4. Silent product loss — state machine + watchdog + reconciliation query rendering a "0 lost" dashboard counter (demo talking point).
5. First-inference latency — warm-up inference at worker startup; readiness gated on it.
6. JPEG encoding — encode overlay once, broadcast same bytes to all clients.
7. Worker restart storm — RestartSec + StartLimitBurst; app degrades gracefully.
8. Demo-day hygiene — dashboard visibly shows "line stopped / inference degraded" instead of freezing.
