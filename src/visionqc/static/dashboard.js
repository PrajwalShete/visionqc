/* VisionQC operator dashboard — single WebSocket client (binary + JSON frames).
 *
 * Protocol (mirrors src/visionqc/api/ws.py):
 *   text  : {kind:"hello"|"event", type, ts, payload}
 *   binary: [4-byte BE meta length][UTF-8 JSON meta][raw JPEG bytes]
 *           meta = {product_id, image_kind:"raw"|"heatmap", ts, outcome}
 */
"use strict";

const $ = (id) => document.getElementById(id);
const MAX_EVENTS = 100;
const SCORE_WINDOW = 40;

const state = {
  ws: null,
  backoff: 500,
  connected: false,
  pass: 0,
  reject: 0,
  fault: 0,
  finalizedTimes: [], // wall-clock ms of recent finalizations (throughput)
  latencies: [], // recent decision latencies (ms)
  lastImageUrls: [], // object URLs to revoke
  activeAlarms: new Map(), // id/code -> alarm
  controlsDisabled: false,
  threshold: null,
};

/* ------------------------------------------------------------------ */
/* Clock                                                              */
/* ------------------------------------------------------------------ */
function tickClock() {
  const now = new Date();
  $("clock").textContent = now.toLocaleTimeString("en-GB");
  $("clock-date").textContent = now.toISOString().slice(0, 10);
}
setInterval(tickClock, 1000);
tickClock();

/* ------------------------------------------------------------------ */
/* Connection state                                                  */
/* ------------------------------------------------------------------ */
function setConnected(ok, label) {
  state.connected = ok;
  const dot = $("conn-dot");
  dot.className = "live-dot " + (ok ? "dot-on" : "dot-off");
  $("conn-label").textContent = label || (ok ? "LIVE" : "DISCONNECTED");
}

function setLineState(s) {
  const el = $("line-state");
  el.textContent = s || "UNKNOWN";
  const map = {
    RUNNING: "bg-green-600",
    STOPPED: "bg-red-600",
    DEGRADED: "bg-amber-500",
  };
  el.className =
    "badge badge-lg ml-2 border-none font-bold text-white " +
    (map[s] || "bg-slate-600");
}

/* ------------------------------------------------------------------ */
/* WebSocket with exponential-backoff reconnect                       */
/* ------------------------------------------------------------------ */
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const url = `${proto}://${location.host}/ws`;
  let ws;
  try {
    ws = new WebSocket(url);
  } catch (e) {
    scheduleReconnect();
    return;
  }
  ws.binaryType = "arraybuffer";
  state.ws = ws;
  setConnected(false, "CONNECTING");

  ws.onopen = () => {
    state.backoff = 500;
    setConnected(true, "LIVE");
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") handleText(ev.data);
    else handleBinary(ev.data);
  };
  ws.onclose = () => {
    setConnected(false, "DISCONNECTED");
    scheduleReconnect();
  };
  ws.onerror = () => {
    try {
      ws.close();
    } catch (e) {
      /* ignore */
    }
  };
}

function scheduleReconnect() {
  const delay = Math.min(state.backoff, 8000);
  state.backoff = Math.min(state.backoff * 2, 8000);
  setTimeout(connect, delay);
}

/* ------------------------------------------------------------------ */
/* Text frames                                                        */
/* ------------------------------------------------------------------ */
function handleText(raw) {
  let msg;
  try {
    msg = JSON.parse(raw);
  } catch (e) {
    return;
  }
  if (msg.kind === "hello") return applyHello(msg.payload || {});
  if (msg.kind === "event") return applyEvent(msg);
}

function applyHello(p) {
  if (p.line_state) setLineState(p.line_state);
  if (p.reconciliation) applyReconciliation(p.reconciliation);
  if (Array.isArray(p.active_alarms)) {
    p.active_alarms.forEach((a) => upsertAlarm(a));
    renderAlarms();
  }
}

function applyReconciliation(r) {
  if (typeof r.pass === "number") {
    state.pass = r.pass;
    $("stat-pass").textContent = r.pass;
  }
  if (typeof r.reject === "number") {
    state.reject = r.reject;
    $("stat-reject").textContent = r.reject;
  }
  if (typeof r.fault === "number") {
    state.fault = r.fault;
    $("stat-fault").textContent = r.fault;
  }
  updateDonut();
  if (typeof r.lost === "number") applyLost(r.lost);
}

function applyLost(lost) {
  $("zero-lost-num").textContent = lost;
  const tile = $("zero-lost");
  if (lost === 0) {
    tile.className = "vqc-panel border-green-500/40 px-3 py-3 text-center";
    $("zero-lost-num").className = "";
    $("zero-lost-num").style.color = "#4ade80";
  } else {
    tile.className = "vqc-panel border-red-500/60 px-3 py-3 text-center";
    $("zero-lost-num").style.color = "#f87171";
  }
}

function applyEvent(msg) {
  const type = msg.type;
  const p = msg.payload || {};
  addEventRow(msg.ts, type, p);

  if (type === "InferenceCompleted" && typeof p.score === "number") {
    pushScore(p.ts || msg.ts, p.score);
    if (typeof p.latency_ms === "number") pushLatency(p.latency_ms);
  }
  if (type === "DecisionMade" && typeof p.score === "number") {
    // score already pushed at inference; nothing extra
  }
  if (type === "ProductFinalized") {
    const outcome = p.outcome;
    if (outcome === "PASS") state.pass++;
    else if (outcome === "REJECT") state.reject++;
    else if (outcome === "FAULT") state.fault++;
    $("stat-pass").textContent = state.pass;
    $("stat-reject").textContent = state.reject;
    $("stat-fault").textContent = state.fault;
    updateDonut();
    markThroughput();
    setOverlay(p.product_id, outcome);
    if (p.timings && typeof p.timings.inference_ms === "number") {
      pushLatency(p.timings.inference_ms);
    }
  }
  if (type === "AlarmRaised") {
    upsertAlarm(p);
    renderAlarms();
  }
  if (type === "AlarmCleared") {
    removeAlarm(p);
    renderAlarms();
  }
  if (type === "LineStateChanged" && p.state) {
    setLineState(p.state);
  }
}

/* ------------------------------------------------------------------ */
/* Binary frames (evidence images)                                    */
/* ------------------------------------------------------------------ */
function handleBinary(buf) {
  let meta, jpeg;
  try {
    const view = new DataView(buf);
    const metaLen = view.getUint32(0, false); // big-endian
    const metaBytes = new Uint8Array(buf, 4, metaLen);
    meta = JSON.parse(new TextDecoder("utf-8").decode(metaBytes));
    jpeg = new Uint8Array(buf, 4 + metaLen);
  } catch (e) {
    return;
  }
  const blob = new Blob([jpeg], { type: "image/jpeg" });
  const objUrl = URL.createObjectURL(blob);
  const img = $("live-image");
  img.src = objUrl;
  $("image-empty").style.display = "none";
  $("image-kind").textContent = (meta.image_kind || "").toUpperCase();
  setOverlay(meta.product_id, meta.outcome);

  // Revoke previous object URLs to avoid leaks (keep the latest couple).
  state.lastImageUrls.push(objUrl);
  while (state.lastImageUrls.length > 3) {
    URL.revokeObjectURL(state.lastImageUrls.shift());
  }
}

function setOverlay(productId, outcome) {
  if (productId) $("overlay-product").textContent = productId;
  const badge = $("overlay-outcome");
  badge.textContent = outcome || "—";
  const frame = $("image-frame");
  frame.classList.remove("frame-pass", "frame-reject", "frame-fault");
  const cmap = {
    PASS: ["frame-pass", "bg-green-600"],
    REJECT: ["frame-reject", "bg-red-600"],
    FAULT: ["frame-fault", "bg-amber-500"],
  };
  const c = cmap[outcome];
  if (c) {
    frame.classList.add(c[0]);
    badge.className =
      "badge badge-sm border-none font-bold text-white " + c[1];
  } else {
    badge.className =
      "badge badge-sm border-none bg-slate-700 font-bold text-white";
  }
}

/* ------------------------------------------------------------------ */
/* Event feed                                                         */
/* ------------------------------------------------------------------ */
const OUTCOME_BADGE = {
  PASS: "bg-green-600",
  REJECT: "bg-red-600",
  FAULT: "bg-amber-500",
};

function addEventRow(ts, type, p) {
  const tbody = $("event-feed");
  const tr = document.createElement("tr");
  tr.className = "hover cursor-pointer border-[var(--vqc-border)]";
  const pid = p.product_id || "";
  const outcome = p.outcome || "";
  const t = ts ? new Date(ts).toLocaleTimeString("en-GB") : "";
  const badge = outcome
    ? `<span class="badge badge-xs border-none text-white ${
        OUTCOME_BADGE[outcome] || "bg-slate-600"
      }">${outcome}</span>`
    : "";
  tr.innerHTML = `
    <td class="whitespace-nowrap text-slate-500">${t}</td>
    <td class="whitespace-nowrap text-slate-300">${type}</td>
    <td class="max-w-[90px] truncate text-slate-400">${pid}</td>
    <td>${badge}</td>`;
  if (pid) tr.addEventListener("click", () => openProduct(pid));
  tbody.prepend(tr);
  while (tbody.rows.length > MAX_EVENTS) tbody.deleteRow(tbody.rows.length - 1);
  $("feed-count").textContent = tbody.rows.length;
}

/* ------------------------------------------------------------------ */
/* Throughput + latency                                               */
/* ------------------------------------------------------------------ */
function markThroughput() {
  const now = Date.now();
  state.finalizedTimes.push(now);
  const cutoff = now - 60000;
  state.finalizedTimes = state.finalizedTimes.filter((t) => t >= cutoff);
  $("stat-throughput").textContent = state.finalizedTimes.length;
}

function pushLatency(ms) {
  state.latencies.push(ms);
  if (state.latencies.length > 50) state.latencies.shift();
  const mean =
    state.latencies.reduce((a, b) => a + b, 0) / state.latencies.length;
  $("stat-latency").textContent = mean.toFixed(0);
}

/* ------------------------------------------------------------------ */
/* Charts                                                             */
/* ------------------------------------------------------------------ */
let scoreChart, donutChart;

function initCharts() {
  const gridColor = "rgba(148,163,184,0.12)";
  scoreChart = new Chart($("score-chart"), {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "score",
          data: [],
          borderColor: "#38bdf8",
          backgroundColor: "rgba(56,189,248,0.15)",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.25,
          fill: true,
        },
        {
          label: "threshold",
          data: [],
          borderColor: "#ef4444",
          borderWidth: 1,
          borderDash: [4, 4],
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: {
          min: 0,
          max: 1,
          ticks: { color: "#64748b", font: { size: 9 } },
          grid: { color: gridColor },
        },
      },
    },
  });

  donutChart = new Chart($("defect-donut"), {
    type: "doughnut",
    data: {
      labels: ["Pass", "Reject", "Fault"],
      datasets: [
        {
          data: [0, 0, 0],
          backgroundColor: ["#22c55e", "#ef4444", "#f59e0b"],
          borderColor: "#11161f",
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "62%",
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: "#94a3b8", font: { size: 9 }, boxWidth: 8 },
        },
      },
    },
  });
}

function pushScore(ts, score) {
  const c = scoreChart;
  if (!c) return;
  c.data.labels.push("");
  c.data.datasets[0].data.push(score);
  const th = state.threshold;
  c.data.datasets[1].data.push(typeof th === "number" ? th : null);
  while (c.data.datasets[0].data.length > SCORE_WINDOW) {
    c.data.labels.shift();
    c.data.datasets[0].data.shift();
    c.data.datasets[1].data.shift();
  }
  c.update("none");
}

function updateDonut() {
  if (!donutChart) return;
  donutChart.data.datasets[0].data = [state.pass, state.reject, state.fault];
  donutChart.update("none");
}

/* ------------------------------------------------------------------ */
/* Alarms                                                             */
/* ------------------------------------------------------------------ */
function alarmKey(a) {
  return a.alarm_id != null ? "id:" + a.alarm_id : "code:" + a.code;
}

function upsertAlarm(a) {
  if (!a || !a.code) return;
  state.activeAlarms.set(alarmKey(a), a);
}

function removeAlarm(a) {
  state.activeAlarms.delete(alarmKey(a));
}

function renderAlarms() {
  const banner = $("alarm-banner");
  banner.innerHTML = "";
  if (state.activeAlarms.size === 0) {
    banner.classList.add("hidden");
    return;
  }
  banner.classList.remove("hidden");
  for (const a of state.activeAlarms.values()) {
    const sev = (a.severity || "WARNING").toUpperCase();
    const critical = sev === "CRITICAL";
    const div = document.createElement("div");
    div.className =
      "alert flex items-center justify-between border-none py-2 text-white " +
      (critical ? "bg-red-700" : "bg-amber-600");
    div.innerHTML = `
      <div class="flex items-center gap-2">
        <span class="badge badge-sm border-none bg-black/30 font-bold">${sev}</span>
        <span class="font-bold">${a.code}</span>
        <span class="text-sm opacity-90">${a.message || ""}</span>
      </div>`;
    if (a.alarm_id != null) {
      const btn = document.createElement("button");
      btn.className = "btn btn-xs border-none bg-black/30 text-white";
      btn.textContent = "CLEAR";
      btn.addEventListener("click", () => clearAlarm(a.alarm_id));
      div.appendChild(btn);
    }
    banner.appendChild(div);
  }
}

async function clearAlarm(id) {
  try {
    await fetch(`/alarms/${id}/clear`, { method: "POST" });
    state.activeAlarms.delete("id:" + id);
    renderAlarms();
  } catch (e) {
    /* ignore */
  }
}

/* ------------------------------------------------------------------ */
/* REST polling (health reconciliation, alarms, threshold)            */
/* ------------------------------------------------------------------ */
async function pollHealth() {
  try {
    const r = await fetch("/health");
    if (!r.ok) return;
    const h = await r.json();
    if (h.line_state) setLineState(h.line_state);
    if (h.reconciliation) applyReconciliation(h.reconciliation);
    else if (typeof h.zero_silent_loss === "boolean" && !h.zero_silent_loss) {
      // fall through
    }
  } catch (e) {
    /* offline — WS reconnect handles visibility */
  }
}

async function pollAlarms() {
  try {
    const r = await fetch("/alarms?active_only=true");
    if (!r.ok) return;
    const body = await r.json();
    state.activeAlarms.clear();
    (body.items || []).forEach((a) => upsertAlarm(a));
    renderAlarms();
  } catch (e) {
    /* ignore */
  }
}

async function loadThreshold() {
  try {
    const r = await fetch("/recipes");
    if (!r.ok) return;
    const body = await r.json();
    const active = body.active;
    if (active && typeof active.anomaly_threshold === "number") {
      state.threshold = active.anomaly_threshold;
    }
  } catch (e) {
    /* ignore */
  }
}

/* ------------------------------------------------------------------ */
/* Traceability drawer                                                */
/* ------------------------------------------------------------------ */
async function openProduct(pid) {
  const modal = $("product-modal");
  $("modal-title").textContent = "Product " + pid;
  $("modal-detail").innerHTML =
    '<span class="text-slate-500">loading…</span>';
  $("modal-timeline").innerHTML = "";
  $("modal-raw").src = `/evidence/${pid}/raw`;
  $("modal-heatmap").src = `/evidence/${pid}/heatmap`;
  modal.showModal();
  try {
    const r = await fetch(`/products/${pid}`);
    if (!r.ok) {
      $("modal-detail").innerHTML =
        '<span class="text-red-400">not found</span>';
      return;
    }
    const d = await r.json();
    renderProductDetail(d);
  } catch (e) {
    $("modal-detail").innerHTML =
      '<span class="text-red-400">error loading product</span>';
  }
}

function row(k, v) {
  return `<div class="flex justify-between gap-3 border-b border-[var(--vqc-border)] py-1">
    <span class="vqc-label">${k}</span><span class="text-slate-200">${
    v == null ? "—" : v
  }</span></div>`;
}

function renderProductDetail(d) {
  $("modal-detail").innerHTML =
    row("state", d.state) +
    row("outcome", d.outcome) +
    row("score", d.anomaly_score != null ? Number(d.anomaly_score).toFixed(4) : "—") +
    row("reason", d.decision_reason) +
    row("model", d.model_version) +
    row("recipe", d.recipe_id) +
    row("triggered", d.trigger_ts);
  const events = d.events || [];
  $("modal-timeline").innerHTML =
    '<div class="vqc-label mb-1">state timeline</div>' +
    events
      .map((e) => {
        const t = e.ts_wall
          ? new Date(e.ts_wall).toLocaleTimeString("en-GB")
          : "";
        return `<div class="flex items-center gap-2 text-xs">
          <span class="text-slate-600">${t}</span>
          <span class="text-sky-400">${e.event_type}</span>
          <span class="text-slate-500">${e.from_state || ""}→${
          e.to_state || ""
        }</span></div>`;
      })
      .join("");
}

/* ------------------------------------------------------------------ */
/* Line controls (defensive — disable on 404)                         */
/* ------------------------------------------------------------------ */
async function linePost(path, body) {
  try {
    const r = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (r.status === 404) {
      disableControls();
      return false;
    }
    return r.ok;
  } catch (e) {
    return false;
  }
}

function disableControls() {
  if (state.controlsDisabled) return;
  state.controlsDisabled = true;
  ["btn-start", "btn-stop", "speed-slider", "fault-camera", "fault-reject"].forEach(
    (id) => {
      const el = $(id);
      el.disabled = true;
      el.classList.add("opacity-40", "cursor-not-allowed");
      el.title = "line control endpoints unavailable";
    }
  );
  $("controls-note").textContent = "endpoints offline";
}

async function probeControls() {
  // Feature-detect the /line/* API; disable controls if absent.
  try {
    const r = await fetch("/line/status");
    if (r.status === 404) disableControls();
  } catch (e) {
    /* leave enabled; POST will disable on 404 */
  }
}

function wireControls() {
  $("btn-start").addEventListener("click", () =>
    linePost("/line/start")
  );
  $("btn-stop").addEventListener("click", () => linePost("/line/stop"));
  const slider = $("speed-slider");
  slider.addEventListener("input", () => {
    $("speed-value").textContent = slider.value + "%";
  });
  slider.addEventListener("change", () =>
    linePost("/line/speed", { speed: Number(slider.value) })
  );
  $("fault-camera").addEventListener("change", (e) =>
    linePost("/line/fault", { fault: "camera_loss", enabled: e.target.checked })
  );
  $("fault-reject").addEventListener("change", (e) =>
    linePost("/line/fault", { fault: "reject_failure", enabled: e.target.checked })
  );
}

/* ------------------------------------------------------------------ */
/* Boot                                                               */
/* ------------------------------------------------------------------ */
function boot() {
  initCharts();
  wireControls();
  connect();
  loadThreshold();
  pollHealth();
  pollAlarms();
  probeControls();
  setInterval(pollHealth, 5000);
  setInterval(pollAlarms, 7000);
  setInterval(loadThreshold, 30000);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
