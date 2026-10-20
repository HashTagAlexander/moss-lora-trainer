/* MOSS-Image LoRA Trainer — single-page UI (no build step). */
"use strict";

// ------------------------------------------------------------------ utils
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const enc = encodeURIComponent;
let view = $("#view");

async function api(path, opts = {}) {
  const init = { ...opts };
  if (opts.body && !(opts.body instanceof FormData)) {
    init.body = JSON.stringify(opts.body);
    init.headers = { "Content-Type": "application/json" };
  }
  const res = await fetch(path, init);
  const type = res.headers.get("content-type") || "";
  const data = type.includes("json") ? await res.json() : await res.text();
  if (!res.ok) throw new Error((data && data.detail) || res.statusText);
  return data;
}

function toast(msg, kind = "") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), kind === "error" ? 6000 : 3000);
}

async function guard(fn, okMsg) {
  try {
    const r = await fn();
    if (okMsg) toast(okMsg, "ok");
    return r;
  } catch (e) {
    toast(e.message, "error");
    throw e;
  }
}

const fmtDur = (s) => {
  if (s == null || !isFinite(s)) return "—";
  s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h ? `${h}h ${m}m` : m ? `${m}m ${sec}s` : `${sec}s`;
};
const fmtBytes = (b) => {
  if (b == null) return "—";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
  return `${b.toFixed(b < 10 && i ? 1 : 0)} ${u[i]}`;
};
const fmtLoss = (v) => (v == null ? "—" : v < 0.01 ? v.toExponential(2) : v.toFixed(4));
const timeAgo = (t) => {
  const d = Date.now() / 1000 - t;
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return `${Math.floor(d / 86400)}d ago`;
};

const I = {
  play: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 5.5v13a1 1 0 0 0 1.5.86l11-6.5a1 1 0 0 0 0-1.72l-11-6.5A1 1 0 0 0 7 5.5Z"/></svg>',
  stop: '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>',
  plus: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg>',
  trash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/></svg>',
  edit: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 1 1 3 3L7 19l-4 1 1-4Z"/></svg>',
  copy: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>',
  download: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12m0 0 5-5m-5 5-5-5M4 21h16"/></svg>',
  upload: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V4m0 0-5 5m5-5 5 5M4 20h16"/></svg>',
  reset: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/></svg>',
  image: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.1-3.1a2 2 0 0 0-2.8 0L6 21"/></svg>',
  flask: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M9 3h6M10 3v6L4.5 18.5A1.7 1.7 0 0 0 6 21h12a1.7 1.7 0 0 0 1.5-2.5L14 9V3"/><path d="M7 15h10"/></svg>',
  left: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m15 18-6-6 6-6"/></svg>',
  right: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m9 18 6-6-6-6"/></svg>',
  spark: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9Z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8Z"/></svg>',
  dice: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8" cy="8" r="1.2" fill="currentColor"/><circle cx="16" cy="16" r="1.2" fill="currentColor"/><circle cx="12" cy="12" r="1.2" fill="currentColor"/></svg>',
  reuse: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 0 1 15-6.7L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-15 6.7L3 16"/><path d="M3 21v-5h5"/></svg>',
  x: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg>',
};

const ACTIVE = new Set(["starting", "loading", "caching_text", "caching_latents", "training", "sampling", "saving"]);
const STATE_LABEL = {
  new: "Not started", queued: "Queued", starting: "Starting", loading: "Loading model",
  caching_text: "Encoding captions", caching_latents: "Encoding images", training: "Training",
  sampling: "Sampling", saving: "Saving", done: "Done", stopped: "Stopped", error: "Error",
};
function badge(state) {
  const cls = ACTIVE.has(state) ? "running" : state === "queued" ? "queued" : state === "done" ? "done" : state === "error" ? "error" : "";
  return `<span class="badge ${cls}">${esc(STATE_LABEL[state] || state)}</span>`;
}
function pct(st) {
  if (!st || !st.total_steps) return 0;
  return Math.min(100, (100 * (st.step || 0)) / st.total_steps);
}

// ------------------------------------------------------------------ modal / lightbox
function modal({ title, body, confirm = "OK", danger = false }) {
  return new Promise((resolve) => {
    const bg = document.createElement("div");
    bg.className = "modal-bg";
    bg.innerHTML = `<div class="card modal"><div class="card-head"><h2>${esc(title)}</h2></div>
      <div class="card-body">${body}</div>
      <div class="card-body row" style="justify-content:flex-end;border-top:1px solid var(--border)">
        <button class="btn btn-ghost" data-act="cancel">Cancel</button>
        <button class="btn ${danger ? "btn-danger" : "btn-primary"}" data-act="ok">${esc(confirm)}</button></div></div>`;
    document.body.appendChild(bg);
    const input = $("input,textarea", bg);
    if (input) setTimeout(() => input.focus(), 30);
    const done = (ok) => {
      const values = Object.fromEntries($$("[name]", bg).map((e) => [e.name, e.value]));
      bg.remove();
      resolve(ok ? values : null);
    };
    bg.addEventListener("click", (e) => {
      if (e.target === bg || e.target.closest("[data-act=cancel]")) done(false);
      if (e.target.closest("[data-act=ok]")) done(true);
    });
    bg.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && e.target.tagName !== "TEXTAREA") done(true);
      if (e.key === "Escape") done(false);
    });
  });
}
const confirmBox = (title, text, confirm = "Delete") =>
  modal({ title, body: `<p class="muted" style="margin:0">${esc(text)}</p>`, confirm, danger: true });

function lightbox(list, index) {
  const bg = document.createElement("div");
  bg.className = "lightbox";
  const show = () => {
    const it = list[index];
    bg.innerHTML = `<button class="btn btn-ghost close">${I.x}</button>
      ${list.length > 1 ? `<button class="btn nav-btn l">${I.left}</button><button class="btn nav-btn r">${I.right}</button>` : ""}
      <img src="${esc(it.src)}"><div class="cap">${it.title ? `<b>${esc(it.title)}</b><br>` : ""}${esc(it.caption || "")}</div>`;
  };
  const move = (d) => { index = (index + d + list.length) % list.length; show(); };
  const close = () => { bg.remove(); document.removeEventListener("keydown", key); };
  const key = (e) => {
    if (e.key === "Escape") close();
    if (e.key === "ArrowLeft") move(-1);
    if (e.key === "ArrowRight") move(1);
  };
  bg.addEventListener("click", (e) => {
    if (e.target.closest(".l")) return move(-1);
    if (e.target.closest(".r")) return move(1);
    if (e.target === bg || e.target.closest(".close")) close();
  });
  document.addEventListener("keydown", key);
  show();
  document.body.appendChild(bg);
}

// ------------------------------------------------------------------ router
let timers = [];
let INFO = null;
function every(fn, ms) { fn(); timers.push(setInterval(fn, ms)); }
function clearTimers() { timers.forEach(clearInterval); timers = []; }

const routes = [
  [/^\/$/, "dashboard", pageDashboard],
  [/^\/new$/, "new", pageNewJob],
  [/^\/jobs\/([^/]+)$/, "dashboard", pageJob],
  [/^\/datasets$/, "datasets", pageDatasets],
  [/^\/datasets\/([^/]+)$/, "datasets", pageDataset],
  [/^\/generate$/, "generate", pageGenerate],
  [/^\/settings$/, "settings", pageSettings],
];

async function route() {
  clearTimers();
  const raw = location.hash.slice(1) || "/";
  const [path, query] = raw.split("?");
  const params = Object.fromEntries(new URLSearchParams(query || ""));
  for (const [re, nav, fn] of routes) {
    const m = path.match(re);
    if (m) {
      $$("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === nav));
      // fresh element per page so listeners from the previous page are dropped
      const fresh = view.cloneNode(false);
      view.replaceWith(fresh);
      view = fresh;
      try {
        await fn(...m.slice(1).map(decodeURIComponent), params);
      } catch (e) {
        view.innerHTML = `<div class="card empty">${esc(e.message)}</div>`;
      }
      window.scrollTo(0, 0);
      return;
    }
  }
  location.hash = "#/";
}
window.addEventListener("hashchange", route);

// ------------------------------------------------------------------ sidebar GPU
async function refreshGpu() {
  try {
    const g = await api("/api/gpu");
    const el = $("#sidebar-gpu");
    if (!g.gpus.length) { el.innerHTML = '<div class="faint">No NVIDIA GPU detected</div>'; return; }
    el.innerHTML = g.gpus.map((x) => `
      <div class="gpu-name">${esc(x.name)}</div>
      <div class="row" style="justify-content:space-between"><span>GPU</span><span>${x.util ?? "—"}%</span></div>
      <div class="mini-bar"><div style="width:${x.util || 0}%"></div></div>
      <div class="row" style="justify-content:space-between"><span>VRAM</span><span>${(x.mem_used / 1024).toFixed(1)} / ${(x.mem_total / 1024).toFixed(0)} GB</span></div>
      <div class="mini-bar"><div style="width:${(100 * x.mem_used) / x.mem_total}%"></div></div>
      <div class="row" style="justify-content:space-between"><span>${x.temp ?? "—"}°C</span><span>${x.power ? Math.round(x.power) : "—"} / ${x.power_limit ? Math.round(x.power_limit) : "—"} W</span></div>`).join("");
  } catch { /* ignore */ }
}

// ------------------------------------------------------------------ dashboard
async function pageDashboard() {
  view.innerHTML = `
    <div class="page-head"><div><h1>Dashboard</h1><div class="sub">Train LoRAs for MOSS-Image on this machine</div></div>
      <a class="btn btn-primary" href="#/new">${I.plus} New training job</a></div>
    <div class="grid grid-4" id="gpu-stats" style="margin-bottom:18px"></div>
    <div class="card"><div class="card-head"><h2>Training jobs</h2></div><div id="jobs"></div></div>`;
  let jobsKey = "";
  every(async () => {
    const [jobs, g] = await Promise.all([api("/api/jobs"), api("/api/gpu")]);
    const gpu = g.gpus[0];
    $("#gpu-stats").innerHTML = (gpu ? [
      ["GPU", gpu.name.replace("NVIDIA ", ""), `${gpu.util ?? 0}% utilization`],
      ["VRAM", `${(gpu.mem_used / 1024).toFixed(1)} GB`, `of ${(gpu.mem_total / 1024).toFixed(0)} GB`],
      ["Temperature", `${gpu.temp ?? "—"}°C`, `${gpu.power ? Math.round(gpu.power) : "—"} W draw`],
    ] : [["GPU", "None", "nvidia-smi not found"]]).concat([["Disk free", fmtBytes(g.disk?.free), `of ${fmtBytes(g.disk?.total)}`]])
      .map(([l, v, h]) => `<div class="card stat"><div class="label">${l}</div><div class="value" style="font-size:${v.length > 14 ? 16 : 22}px">${esc(v)}</div><div class="hint">${esc(h)}</div></div>`).join("");

    const key = JSON.stringify(jobs);
    if (key === jobsKey) return;
    jobsKey = key;
    const el = $("#jobs");
    if (!jobs.length) {
      el.innerHTML = `<div class="empty">${I.flask}<div>No jobs yet.</div><div style="margin-top:12px"><a class="btn btn-primary" href="#/new">${I.plus} Create your first job</a></div></div>`;
      return;
    }
    el.innerHTML = `<table><thead><tr><th>Name</th><th>Dataset</th><th>Status</th><th style="width:28%">Progress</th><th>ETA</th><th></th></tr></thead><tbody>
      ${jobs.map((j) => {
        const st = j.status || {};
        const running = ACTIVE.has(st.state) || st.state === "queued";
        return `<tr class="click" data-job="${esc(j.name)}">
          <td><b>${esc(j.name)}</b><div class="faint" style="font-size:12px">${esc((j.model || "").split("/").pop())}</div></td>
          <td>${esc(j.dataset)}</td>
          <td>${badge(st.state || "new")}</td>
          <td><div class="row" style="gap:10px;flex-wrap:nowrap"><div class="progress grow"><div style="width:${pct(st)}%"></div></div>
            <span class="mono muted">${st.step || 0}/${st.total_steps || j.steps}</span></div></td>
          <td class="mono muted">${st.state === "training" ? fmtDur(st.eta_seconds) : "—"}</td>
          <td style="text-align:right">${running
            ? `<button class="btn btn-sm" data-stop="${esc(j.name)}">${I.stop} Stop</button>`
            : st.state !== "done" ? `<button class="btn btn-sm" data-start="${esc(j.name)}">${I.play} ${st.step ? "Resume" : "Start"}</button>` : ""}</td></tr>`;
      }).join("")}</tbody></table>`;
  }, 3000);
  view.addEventListener("click", async (e) => {
    const s = e.target.closest("[data-start]"), t = e.target.closest("[data-stop]"), r = e.target.closest("[data-job]");
    if (s) { e.stopPropagation(); await guard(() => api(`/api/jobs/${enc(s.dataset.start)}/start`, { method: "POST" }), "Job queued"); return route(); }
    if (t) { e.stopPropagation(); await guard(() => api(`/api/jobs/${enc(t.dataset.stop)}/stop`, { method: "POST" }), "Stopping…"); return route(); }
    if (r) location.hash = `#/jobs/${enc(r.dataset.job)}`;
  });
}

// ------------------------------------------------------------------ loss chart
function lossChart(el) {
  let rows = [], smooth = 0.9, logY = false;
  el.innerHTML = `<div class="chart"><svg></svg><div class="tip"></div></div>`;
  const svg = $("svg", el), tip = $(".tip", el);
  function ema(vals) {
    let last = 0, out = [], n = 0;
    for (const v of vals) { n++; last = last * smooth + (1 - smooth) * v; out.push(last / (1 - Math.pow(smooth, n))); }
    return out;
  }
  function draw() {
    const W = svg.clientWidth || 800, H = svg.clientHeight || 280;
    const P = { l: 54, r: 12, t: 12, b: 26 };
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    if (!rows.length) {
      svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle" class="axis-label" style="font-size:12px">Loss appears once training starts</text>`;
      return;
    }
    const xs = rows.map((r) => r.step), raw = rows.map((r) => r.loss), sm = ema(raw);
    const f = (v) => (logY ? Math.log10(Math.max(v, 1e-8)) : v);
    const sorted = raw.map(f).sort((a, b) => a - b);
    let lo = sorted[Math.floor(sorted.length * 0.01)], hi = sorted[Math.floor(sorted.length * 0.99)] ?? sorted[sorted.length - 1];
    const smF = sm.map(f);
    lo = Math.min(lo, ...smF); hi = Math.max(hi, ...smF);
    if (hi - lo < 1e-9) { hi += 0.5; lo -= 0.5; }
    const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
    const x0 = xs[0], x1 = Math.max(xs[xs.length - 1], x0 + 1);
    const X = (s) => P.l + ((s - x0) / (x1 - x0)) * (W - P.l - P.r);
    const Y = (v) => P.t + (1 - (f(v) - lo) / (hi - lo)) * (H - P.t - P.b);
    const stride = Math.max(1, Math.floor(rows.length / 1500));
    const path = (vals) => vals.map((v, i) => (i % stride && i !== vals.length - 1 ? "" : `${i ? "L" : "M"}${X(xs[i]).toFixed(1)},${Math.min(H - P.b, Math.max(P.t, Y(v))).toFixed(1)}`)).join("");
    let grid = "";
    for (let i = 0; i <= 4; i++) {
      const v = lo + ((hi - lo) * i) / 4, y = P.t + (1 - i / 4) * (H - P.t - P.b);
      const label = logY ? Math.pow(10, v).toPrecision(2) : v.toPrecision(3);
      grid += `<line class="grid-line" x1="${P.l}" x2="${W - P.r}" y1="${y}" y2="${y}"/><text class="axis-label" x="${P.l - 8}" y="${y + 3}" text-anchor="end">${label}</text>`;
    }
    for (let i = 0; i <= 5; i++) {
      const s = Math.round(x0 + ((x1 - x0) * i) / 5);
      grid += `<text class="axis-label" x="${X(s)}" y="${H - 8}" text-anchor="middle">${s}</text>`;
    }
    const smPath = path(sm);
    svg.innerHTML = `<defs><linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#8b6cff" stop-opacity=".25"/><stop offset="1" stop-color="#8b6cff" stop-opacity="0"/></linearGradient></defs>
      ${grid}<path class="area" d="${smPath}L${X(xs[xs.length - 1])},${H - P.b}L${X(xs[0])},${H - P.b}Z"/>
      <path class="raw" d="${path(raw)}"/><path class="smooth" d="${smPath}"/>
      <line class="cursor" y1="${P.t}" y2="${H - P.b}" x1="-10" x2="-10"/><circle r="4" fill="#a58dff" cx="-10" cy="-10"/>`;
    svg.onmousemove = (e) => {
      const rect = svg.getBoundingClientRect();
      const sx = ((e.clientX - rect.left) / rect.width) * W;
      const step = x0 + ((sx - P.l) / (W - P.l - P.r)) * (x1 - x0);
      let i = xs.findIndex((s) => s >= step);
      if (i < 0) i = xs.length - 1;
      const cx = X(xs[i]), cy = Y(sm[i]);
      $(".cursor", svg).setAttribute("x1", cx); $(".cursor", svg).setAttribute("x2", cx);
      $("circle", svg).setAttribute("cx", cx); $("circle", svg).setAttribute("cy", cy);
      tip.style.display = "block";
      tip.style.left = `${(cx / W) * 100}%`; tip.style.top = `${(cy / H) * rect.height}px`;
      tip.innerHTML = `step ${xs[i]} · loss ${fmtLoss(raw[i])} · avg ${fmtLoss(sm[i])}`;
    };
    svg.onmouseleave = () => { tip.style.display = "none"; };
  }
  window.addEventListener("resize", draw);
  return {
    set(r) { rows = r; draw(); },
    add(r) { if (r.length) { const last = rows.length ? rows[rows.length - 1].step : 0; rows = rows.concat(r.filter((x) => x.step > last)); draw(); } },
    smoothing(v) { smooth = v; draw(); },
    log(v) { logY = v; draw(); },
    last() { return rows.length ? rows[rows.length - 1].step : 0; },
    rows: () => rows,
  };
}

// ------------------------------------------------------------------ job page
const CKPT_NOTE = {
  moss: 'LoRA files use ComfyUI-compatible keys (<span class="mono">diffusion_model.layers.N.proj.*</span>). Load with a standard LoraLoaderModelOnly node on top of the MOSS-Image loader, or use <span class="mono">generate.py</span>.',
  krea2: 'LoRA files are in diffusers format (<span class="mono">transformer.transformer_blocks.N.*</span>), loadable by diffusers\' Krea2Pipeline and ComfyUI.',
  fibo2: 'LoRA files are in diffusers format (<span class="mono">transformer.layers.N.*</span>). Load with <span class="mono">pipe.transformer.load_lora_adapter(folder, weight_name="&lt;file&gt;.safetensors")</span> on BriaFibo2Pipeline; they work on Fibo 2.0 Dev and Edit Turbo.',
};

async function pageJob(name) {
  if (!INFO) INFO = await api("/api/info");
  let job = await api(`/api/jobs/${enc(name)}`);
  const family = (INFO.model_specs[job.config.model] || {}).family;
  const base = `/api/jobs/${enc(name)}`;
  let tab = "overview";
  view.innerHTML = `
    <div class="page-head">
      <div><div class="row"><h1>${esc(name)}</h1><span id="j-badge"></span></div>
        <div class="sub">${esc(job.config.model)} · dataset <a href="#/datasets/${enc(job.config.dataset)}">${esc(job.config.dataset)}</a>${job.config.trigger_word ? ` · trigger <span class="mono">${esc(job.config.trigger_word)}</span>` : ""}</div></div>
      <div class="row" id="j-actions"></div>
    </div>
    <div class="card" style="margin-bottom:16px"><div class="card-body">
      <div class="row" style="justify-content:space-between;margin-bottom:10px"><div id="j-msg" class="muted"></div><div id="j-step" class="mono"></div></div>
      <div class="progress big"><div id="j-bar"></div></div>
      <div class="grid grid-4" style="margin-top:14px" id="j-stats"></div>
    </div></div>
    <div class="tabs" id="tabs">
      ${["overview", "samples", "checkpoints", "log", "config"].map((t) => `<div class="tab" data-tab="${t}">${t[0].toUpperCase() + t.slice(1)}</div>`).join("")}
    </div>
    <div id="tab-overview">
      <div class="card" style="margin-bottom:16px"><div class="card-head"><h2>Loss</h2>
        <div class="row"><label class="check"><input type="checkbox" id="logy"> Log scale</label>
        <span class="muted" style="font-size:12px">Smoothing</span><input type="range" id="smooth" min="0" max="0.99" step="0.01" value="0.9" style="width:110px;accent-color:var(--accent)"></div></div>
        <div class="card-body" id="chart"></div></div>
      <div class="card"><div class="card-head"><h2>Latest samples</h2><a href="javascript:void 0" data-goto="samples" class="muted" style="font-size:12px">All samples →</a></div><div class="card-body" id="latest"></div></div>
    </div>
    <div id="tab-samples" hidden><div class="card"><div class="card-body" id="samples"></div></div></div>
    <div id="tab-checkpoints" hidden><div class="card" id="ckpts"></div></div>
    <div id="tab-log" hidden><pre class="log" id="log">Loading…</pre></div>
    <div id="tab-config" hidden><pre class="log" id="cfg"></pre></div>`;

  const chart = lossChart($("#chart"));
  chart.set(await api(`${base}/metrics`));
  $("#smooth").oninput = (e) => chart.smoothing(+e.target.value);
  $("#logy").onchange = (e) => chart.log(e.target.checked);
  $("#cfg").textContent = JSON.stringify(job.config, null, 2);

  const setTab = (t) => {
    tab = t;
    $$("#tabs .tab").forEach((x) => x.classList.toggle("active", x.dataset.tab === t));
    ["overview", "samples", "checkpoints", "log", "config"].forEach((x) => ($(`#tab-${x}`).hidden = x !== t));
    if (t === "log") refreshLog();
    if (t === "overview") chart.set(chart.rows());
  };
  $("#tabs").onclick = (e) => { const t = e.target.closest("[data-tab]"); if (t) setTab(t.dataset.tab); };
  setTab("overview");

  function sampleList() {
    return job.samples.slice().sort((a, b) => b.step - a.step || a.index - b.index)
      .map((s) => ({ ...s, src: `${base}/sample/${enc(s.file)}`, title: `Step ${s.step}`, caption: s.prompt }));
  }
  function renderSamples() {
    const list = sampleList();
    const steps = [...new Set(list.map((s) => s.step))];
    if (!list.length) {
      const msg = '<div class="empty">' + I.image + "<div>No samples yet</div></div>";
      $("#latest").innerHTML = msg; $("#samples").innerHTML = msg; return;
    }
    const prompts = [...new Map(list.map((s) => [s.index, s.prompt])).entries()].sort((a, b) => a[0] - b[0]);
    const legend = `<div class="prompt-legend">${prompts.map(([, p]) => `<div title="${esc(p)}">${esc(p)}</div>`).join("")}</div>`;
    const rowHtml = (step) => `<div class="sample-row"><div class="sample-step">step<br>${step}</div><div class="sample-imgs">
      ${list.map((s, i) => (s.step === step ? `<img loading="lazy" src="${esc(s.src)}" data-i="${i}" title="${esc(s.prompt)}">` : "")).join("")}</div></div>`;
    $("#latest").innerHTML = legend + rowHtml(steps[0]);
    $("#samples").innerHTML = legend + steps.map(rowHtml).join("");
  }
  view.addEventListener("click", (e) => {
    const img = e.target.closest("img[data-i]");
    if (img) lightbox(sampleList(), +img.dataset.i);
    const g = e.target.closest("[data-goto]");
    if (g) setTab(g.dataset.goto);
  });

  function renderCkpts() {
    const el = $("#ckpts");
    if (!job.checkpoints.length) { el.innerHTML = `<div class="empty">No checkpoints saved yet</div>`; return; }
    el.innerHTML = `<table><thead><tr><th>File</th><th>Size</th><th>Saved</th><th></th></tr></thead><tbody>
      ${job.checkpoints.slice().sort((a, b) => b.mtime - a.mtime).map((c) => `<tr><td class="mono">${esc(c.file)}</td><td class="muted">${fmtBytes(c.size)}</td>
      <td class="muted">${timeAgo(c.mtime)}</td><td style="text-align:right"><a class="btn btn-sm" href="#/generate?lora=${enc(`jobs/${name}/checkpoints/${c.file}`)}">${I.spark} Generate</a> <a class="btn btn-sm" href="${base}/checkpoint/${enc(c.file)}" download>${I.download} Download</a></td></tr>`).join("")}
      </tbody></table><div class="card-body faint" style="font-size:12px;border-top:1px solid var(--border)">${CKPT_NOTE[family] || CKPT_NOTE.moss}</div>`;
  }

  function renderStatus() {
    const st = job.status || {};
    const running = ACTIVE.has(st.state);
    $("#j-badge").innerHTML = badge(st.state || "new");
    $("#j-bar").style.width = `${pct(st)}%`;
    $("#j-step").textContent = `${st.step || 0} / ${st.total_steps || job.config.train.steps} steps`;
    $("#j-msg").textContent = st.message || (st.state === "training" ? "Training…" : STATE_LABEL[st.state] || "");
    if (st.state === "error") $("#j-msg").style.color = "var(--red)"; else $("#j-msg").style.color = "";
    $("#j-stats").innerHTML = [
      ["Loss", fmtLoss(st.loss)],
      ["Speed", st.sec_per_step ? `${st.sec_per_step.toFixed(2)} s/step` : "—"],
      ["ETA", running ? fmtDur(st.eta_seconds) : "—"],
      ["Epoch", st.epoch ?? "—"],
    ].map(([l, v]) => `<div><div class="muted" style="font-size:12px">${l}</div><div style="font-size:18px;font-weight:600" class="mono">${esc(v)}</div></div>`).join("");
    const queued = st.state === "queued";
    const stopping = running && /Stopping/.test(st.message || "");
    $("#j-actions").innerHTML = (running || queued
      ? `<button class="btn" data-a="stop">${I.stop} ${queued ? "Dequeue" : "Stop"}</button>${stopping ? `<button class="btn btn-danger" data-a="kill">Force kill</button>` : ""}`
      : st.state === "done" ? `<button class="btn btn-primary" data-a="more">${I.play} Train more</button>`
      : `<button class="btn btn-primary" data-a="start">${I.play} ${job.resumable ? "Resume" : "Start"}</button>`) +
      `<a class="btn" href="#/new?edit=${enc(name)}" ${running || queued ? 'style="pointer-events:none;opacity:.45"' : ""}>${I.edit} Edit</a>
       <a class="btn" href="#/new?clone=${enc(name)}">${I.copy} Clone</a>
       <button class="btn btn-ghost" data-a="reset" ${running || queued ? "disabled" : ""} title="Delete checkpoints, samples and metrics">${I.reset}</button>
       <button class="btn btn-ghost btn-danger" data-a="delete" ${running ? "disabled" : ""}>${I.trash}</button>`;
  }
  $("#j-actions").onclick = async (e) => {
    const a = e.target.closest("[data-a]")?.dataset.a;
    if (!a) return;
    if (a === "start") await guard(() => api(`${base}/start`, { method: "POST" }), "Job queued");
    if (a === "more") {
      const v = await modal({ title: "Train more", confirm: "Continue training",
        body: `<label class="field">Additional steps<input type="number" name="extra" value="500" min="1"><span class="help">Continues from the final state (optimizer included) up to ${job.config.train.steps} + N steps.</span></label>` });
      if (!v || !(+v.extra > 0)) return;
      const body = { ...job.config, train: { ...job.config.train, steps: job.config.train.steps + Number(v.extra) }, _overwrite: true, _start: true };
      await guard(() => api("/api/jobs", { method: "POST", body }), "Job queued");
    }
    if (a === "stop") await guard(() => api(`${base}/stop`, { method: "POST" }), "Stop requested — saving a checkpoint first");
    if (a === "kill") await guard(() => api(`${base}/stop?force=1`, { method: "POST" }), "Killed");
    if (a === "reset") {
      if (!(await confirmBox("Reset job?", "Deletes all checkpoints, samples, metrics and the log for this job. The config is kept.", "Reset"))) return;
      await guard(() => api(`${base}/reset`, { method: "POST" }), "Job reset");
      chart.set([]);
    }
    if (a === "delete") {
      if (!(await confirmBox("Delete job?", `Permanently delete "${name}" including checkpoints and samples.`))) return;
      await guard(() => api(base, { method: "DELETE" }), "Deleted");
      location.hash = "#/"; return;
    }
    await refresh();
  };

  async function refreshLog() {
    const txt = await api(`${base}/log?lines=400`);
    const el = $("#log");
    const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
    el.textContent = txt || "(no output yet)";
    if (atBottom) el.scrollTop = el.scrollHeight;
  }

  let sampleKey = "", ckptKey = "";
  async function refresh() {
    job = await api(base);
    renderStatus();
    const sk = JSON.stringify(job.samples);
    if (sk !== sampleKey) { sampleKey = sk; renderSamples(); }
    const ck = JSON.stringify(job.checkpoints);
    if (ck !== ckptKey) { ckptKey = ck; renderCkpts(); }
    const m = await api(`${base}/metrics?since=${chart.last()}`);
    if (m.length) chart.add(m);
    else if ((job.status.step || 0) < chart.last()) chart.set(await api(`${base}/metrics`));
    if (tab === "log") refreshLog();
  }
  every(refresh, 2500);
}

// ------------------------------------------------------------------ new job form
const F = {
  text: (path, label, help = "", attrs = "") => ({ path, label, help, type: "text", attrs }),
  num: (path, label, help = "", attrs = "") => ({ path, label, help, type: "number", attrs }),
  sel: (path, label, options, help = "") => ({ path, label, help, type: "select", options }),
  bool: (path, label, help = "") => ({ path, label, help, type: "bool" }),
  chips: (path, label, options, help = "", numeric = false) => ({ path, label, help, type: "chips", options, numeric }),
  area: (path, label, help = "", full = true) => ({ path, label, help, type: "lines", full }),
};
const getPath = (o, p) => p.split(".").reduce((a, k) => (a == null ? a : a[k]), o);
function setPath(o, p, v) {
  const keys = p.split(".");
  let cur = o;
  keys.slice(0, -1).forEach((k) => { cur[k] = cur[k] && typeof cur[k] === "object" ? cur[k] : {}; cur = cur[k]; });
  cur[keys[keys.length - 1]] = v;
}

function renderField(f, cfg) {
  const v = getPath(cfg, f.path);
  const help = f.help ? `<span class="help">${f.help}</span>` : "";
  const cls = f.full ? "field full" : "field";
  if (f.type === "bool")
    return `<div class="field" style="justify-content:flex-end"><label class="check"><input type="checkbox" data-path="${f.path}" data-type="bool" ${v ? "checked" : ""}> ${esc(f.label)}</label>${help}</div>`;
  if (f.type === "select")
    return `<label class="${cls}">${esc(f.label)}<select data-path="${f.path}" data-type="text">${f.options.map(([val, lab]) => `<option value="${esc(val)}" ${String(val) === String(v) ? "selected" : ""}>${esc(lab)}</option>`).join("")}</select>${help}</label>`;
  if (f.type === "chips") {
    const sel = new Set((v || []).map(String));
    return `<div class="field full"><label class="field">${esc(f.label)}</label><div class="chips" data-path="${f.path}" data-type="chips" data-numeric="${f.numeric ? 1 : ""}">
      ${f.options.map(([val, lab]) => `<span class="chip ${sel.has(String(val)) ? "on" : ""}" data-value="${esc(val)}">${esc(lab)}</span>`).join("")}</div>${help}</div>`;
  }
  if (f.type === "lines")
    return `<label class="${cls}">${esc(f.label)}<textarea data-path="${f.path}" data-type="lines" rows="4">${esc((v || []).join("\n"))}</textarea>${help}</label>`;
  return `<label class="${cls}">${esc(f.label)}<input type="${f.type}" data-path="${f.path}" data-type="${f.type}" value="${esc(v)}" ${f.attrs}>${help}</label>`;
}

function readForm(root, cfg) {
  $$("[data-path]", root).forEach((el) => {
    const t = el.dataset.type;
    let v;
    if (t === "bool") v = el.checked;
    else if (t === "number") v = el.value === "" ? null : Number(el.value);
    else if (t === "lines") v = el.value.split("\n").map((s) => s.trim()).filter(Boolean);
    else if (t === "chips") v = $$(".chip.on", el).map((c) => (el.dataset.numeric ? Number(c.dataset.value) : c.dataset.value));
    else v = el.value;
    if (el.dataset.path === "sample.negative_prompt") v = el.value;
    setPath(cfg, el.dataset.path, v);
  });
  return cfg;
}

async function pageNewJob(params) {
  if (!INFO) INFO = await api("/api/info");
  const datasets = await api("/api/datasets");
  let cfg = JSON.parse(JSON.stringify(INFO.defaults));
  let mode = "new";
  if (params.edit || params.clone) {
    const src = await api(`/api/jobs/${enc(params.edit || params.clone)}`);
    cfg = src.config;
    if (params.clone) { cfg.name = `${cfg.name}_copy`; mode = "clone"; } else mode = "edit";
  } else if (params.dataset) cfg.dataset = params.dataset;
  if (!cfg.dataset && datasets.length) cfg.dataset = datasets[0].name;

  const targetsField = (model) => F.chips("network.targets", "Target modules",
    Object.entries(INFO.families[(INFO.model_specs[model] || {}).family] || {}),
    (INFO.model_specs[model] || {}).family === "fibo2" ? "Bria's recipe trains all four (every Linear layer)." : "");
  const sections = [
    ["Job", [
      F.text("name", "Job name", "Also used as the LoRA filename.", mode === "edit" ? "readonly" : ""),
      F.sel("model", "Base model", Object.entries(INFO.trainable), "Changing it loads that model's recommended settings. Gated models (Fibo 2.0) need a Hugging Face token in <a href=\"#/settings\">Settings</a>."),
      F.sel("dataset", "Dataset", datasets.length ? datasets.map((d) => [d.name, `${d.name} (${d.count} images)`]) : [["", "— create a dataset first —"]],
        `<a href="#/datasets">Manage datasets</a>`),
      F.text("trigger_word", "Trigger word", "Prepended to captions; replaces <span class='mono'>[trigger]</span> in captions and sample prompts."),
    ]],
    ["Network", [
      F.num("network.rank", "Rank", "16–64 is typical for characters / styles.", 'min="1" max="512"'),
      F.num("network.alpha", "Alpha", "Scale = alpha / rank.", 'step="any"'),
      F.num("network.dropout", "Dropout", "", 'step="0.01" min="0" max="0.5"'),
      F.text("network.layers", "Blocks", "Blank = all blocks; or e.g. <span class='mono'>0-20,30</span>."),
      targetsField(cfg.model),
    ]],
    ["Dataset", [
      F.chips("data.resolutions", "Training resolutions", [[512, "512"], [768, "768"], [1024, "1024"], [1280, "1280"], [1536, "1536"]],
        "Images are bucketed by aspect ratio at each selected area. 1024 is the model's main training resolution.", true),
      F.num("data.repeats", "Repeats", "Times each image appears per epoch.", 'min="1"'),
      F.num("data.caption_dropout", "Caption dropout", "Chance of training with an empty caption.", 'step="0.01" min="0" max="1"'),
      F.bool("data.flip", "Horizontal flip augmentation", "Doubles the dataset. Avoid for text/asymmetric subjects."),
    ]],
    ["Training", [
      F.num("train.steps", "Steps", "", 'min="1"'),
      F.num("train.batch_size", "Batch size", "", 'min="1"'),
      F.num("train.grad_accum", "Gradient accumulation", "", 'min="1"'),
      F.num("train.lr", "Learning rate", "1e-4 for AdamW; 1.0 for Prodigy.", 'step="any"'),
      F.sel("train.optimizer", "Optimizer", [["adamw", "AdamW"], ["adamw8bit", "AdamW 8-bit"], ["prodigy", "Prodigy"], ["adafactor", "Adafactor"]]),
      F.sel("train.lr_scheduler", "LR schedule", [["constant", "Constant"], ["cosine", "Cosine"], ["linear", "Linear decay"]]),
      F.num("train.warmup_steps", "Warmup steps", "", 'min="0"'),
      F.num("train.weight_decay", "Weight decay", "", 'step="any" min="0"'),
      F.num("train.max_grad_norm", "Max grad norm", "0 disables clipping.", 'step="any" min="0"'),
      F.num("train.seed", "Seed"),
      F.bool("train.gradient_checkpointing", "Gradient checkpointing", "Saves VRAM at ~25% speed cost. 96 GB cards can usually turn this off at 1024."),
    ]],
    ["Timesteps", [
      F.sel("train.timestep_sampling", "Timestep sampling", [["moss", "MOSS pretraining (truncated logit-normal)"], ["flux_shift", "Resolution-shifted logit-normal (Krea 2 / FLUX)"], ["fibo", "Fibo 2.0 (Bria: shifted logit-normal + 10% uniform)"], ["logit_normal", "Logit-normal"], ["shift", "Shifted uniform"], ["uniform", "Uniform"]],
        "MOSS = the distribution from the final pretraining stage (tech report, Table 6)."),
      F.num("train.shift", "Shift", "For “Shifted uniform”.", 'step="any"'),
      F.num("train.logit_mean", "Logit mean", "For “Logit-normal”.", 'step="any"'),
      F.num("train.logit_std", "Logit std", "For “Logit-normal”.", 'step="any"'),
    ]],
    ["Saving", [
      F.num("save.every", "Save every N steps", "0 = only at the end.", 'min="0"'),
      F.num("save.keep_last", "Keep last N", "Older intermediate checkpoints are deleted.", 'min="0"'),
      F.sel("save.dtype", "Save precision", [["bf16", "bf16"], ["fp16", "fp16"], ["fp32", "fp32"]]),
    ]],
    ["Sampling", [
      F.num("sample.every", "Sample every N steps", "0 disables sampling.", 'min="0"'),
      F.num("sample.width", "Width", "", 'step="16"'),
      F.num("sample.height", "Height", "", 'step="16"'),
      F.num("sample.steps", "Steps", "50 is best; 30 is faster.", 'min="1"'),
      F.num("sample.cfg", "CFG", "1 = off. Krea 2's “4.5” equals 5.5 here.", 'step="0.1"'),
      F.num("sample.shift", "Scheduler shift", "MOSS: 5. Krea 2: 0 = auto by resolution.", 'step="0.01"'),
      F.num("sample.seed", "Seed"),
      F.num("sample.uncond_skip_layer", "Uncond skip layer", "MOSS only: negative-branch layer skip (default 22, −1 off)."),
      F.bool("sample.sample_first", "Sample before training", "Baseline images at step 0."),
      F.area("sample.prompts", "Sample prompts", "One per line. <span class='mono'>[trigger]</span> is replaced by your trigger word. Fibo 2.0 works best with one-line VGL JSON captions."),
      { path: "sample.negative_prompt", label: "Negative prompt", help: "Leave empty to match the official default (Fibo 2.0: the empty VGL caption).", type: "text", attrs: "", full: true },
    ]],
  ];

  const title = mode === "edit" ? `Edit ${cfg.name}` : mode === "clone" ? "Clone job" : "New training job";
  view.innerHTML = `
    <div class="page-head"><div><h1>${esc(title)}</h1><div class="sub">Defaults follow the MOSS-Image tech report where it applies; tweak as needed.</div></div>
      <label class="check"><input type="checkbox" id="json-mode"> Edit as JSON</label></div>
    <form id="form">${sections.map(([t, fields]) => `<div class="card form-section"><div class="card-head"><h3>${t}</h3></div><div class="card-body">${fields.map((f) => renderField(f, cfg)).join("")}</div></div>`).join("")}</form>
    <div id="json-wrap" hidden><textarea id="json" class="mono" style="min-height:560px"></textarea></div>
    <div class="sticky-bar"><a class="btn btn-ghost" href="#/">Cancel</a>
      <button class="btn" id="save">Save</button><button class="btn btn-primary" id="save-start">${I.play} Save & start</button></div>`;

  const form = $("#form");
  let prevModel = cfg.model;
  // A model's training overrides and default negative, as {path: value}.
  const modelPaths = (model) => {
    const spec = INFO.model_specs[model] || {}, out = {};
    const walk = (o, pre) => Object.entries(o || {}).forEach(([k, v]) => (v && typeof v === "object" && !Array.isArray(v) ? walk(v, `${pre}${k}.`) : (out[pre + k] = v)));
    walk(spec.overrides, "");
    out["sample.negative_prompt"] = (spec.defaults || {}).negative || "";
    return out;
  };
  form.addEventListener("change", (e) => {
    if (e.target.dataset.path !== "model") return;
    const model = e.target.value;
    const d = (INFO.model_specs[model] || {}).defaults || {};
    const set = (path, v) => { const el = $(`[data-path="${path}"]`, form); if (el && v !== undefined) el.value = v; };
    set("sample.steps", d.steps); set("sample.cfg", d.cfg); set("sample.shift", d.shift);
    set("sample.uncond_skip_layer", d.skip_layer); set("train.timestep_sampling", d.timestep_sampling);
    // Swap in the new model's overrides wherever the field still holds the previous model's value.
    const prev = modelPaths(prevModel), next = modelPaths(model), cur = readForm(form, {});
    const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
    for (const path of new Set([...Object.keys(prev), ...Object.keys(next)])) {
      const was = path in prev ? prev[path] : getPath(INFO.defaults, path);
      const now = path in next ? next[path] : getPath(INFO.defaults, path);
      if (!same(getPath(cur, path), was)) continue;
      setPath(cur, path, now);
      const el = $(`[data-path="${path}"]`, form);
      if (!el || el.dataset.type === "chips") continue;
      if (el.dataset.type === "lines") el.value = (now || []).join("\n");
      else if (el.dataset.type === "bool") el.checked = !!now;
      else el.value = now ?? "";
    }
    // Target groups differ per model family.
    const options = Object.keys(INFO.families[(INFO.model_specs[model] || {}).family] || {});
    let targets = (getPath(cur, "network.targets") || []).filter((t) => options.includes(t));
    if (!targets.length) targets = getPath(INFO.defaults, "network.targets").filter((t) => options.includes(t));
    const chips = $('[data-path="network.targets"]', form).closest(".field.full");
    chips.outerHTML = renderField(targetsField(model), { network: { targets } });
    prevModel = model;
    toast("Loaded the recommended settings for this model", "ok");
  });
  form.addEventListener("click", (e) => { const c = e.target.closest(".chip"); if (c) c.classList.toggle("on"); });
  $("#json-mode").onchange = (e) => {
    if (e.target.checked) { $("#json").value = JSON.stringify(readForm(form, cfg), null, 2); }
    else {
      try { cfg = JSON.parse($("#json").value); } catch (err) { toast("Invalid JSON: " + err.message, "error"); e.target.checked = true; return; }
      const net = sections.find(([t]) => t === "Network")[1];
      net[net.findIndex((f) => f.path === "network.targets")] = targetsField(cfg.model);
      prevModel = cfg.model;
      form.innerHTML = sections.map(([t, fields]) => `<div class="card form-section"><div class="card-head"><h3>${t}</h3></div><div class="card-body">${fields.map((f) => renderField(f, cfg)).join("")}</div></div>`).join("");
    }
    form.hidden = e.target.checked; $("#json-wrap").hidden = !e.target.checked;
  };
  const submit = async (start) => {
    let body;
    if ($("#json-mode").checked) {
      try { body = JSON.parse($("#json").value); } catch (err) { return toast("Invalid JSON: " + err.message, "error"); }
    } else body = readForm(form, cfg);
    body._start = start;
    body._overwrite = mode === "edit";
    const r = await guard(() => api("/api/jobs", { method: "POST", body }), start ? "Job queued" : "Saved");
    location.hash = `#/jobs/${enc(r.name)}`;
  };
  $("#save").onclick = () => submit(false);
  $("#save-start").onclick = () => submit(true);
}

// ------------------------------------------------------------------ datasets
async function pageDatasets() {
  view.innerHTML = `<div class="page-head"><div><h1>Datasets</h1><div class="sub">Folders of images with matching .txt captions</div></div>
    <button class="btn btn-primary" id="new-ds">${I.plus} New dataset</button></div><div id="ds"></div>`;
  $("#new-ds").onclick = async () => {
    const v = await modal({ title: "New dataset", confirm: "Create",
      body: `<label class="field">Name<input type="text" name="name" placeholder="my_character"><span class="help">Letters, digits, . _ - only.</span></label>` });
    if (!v) return;
    await guard(() => api("/api/datasets", { method: "POST", body: { name: v.name.trim() } }));
    location.hash = `#/datasets/${enc(v.name.trim())}`;
  };
  const list = await api("/api/datasets");
  if (!list.length) { $("#ds").innerHTML = `<div class="card empty">${I.image}<div>No datasets yet. Create one and drop your images in.</div></div>`; return; }
  $("#ds").innerHTML = `<div class="grid grid-3">${list.map((d) => `
    <a class="card ds-card" href="#/datasets/${enc(d.name)}">
      <div class="cover" style="${d.cover ? `background-image:url('/api/datasets/${enc(d.name)}/image/${enc(d.cover)}?thumb=1')` : ""}"></div>
      <div class="info"><b>${esc(d.name)}</b><div class="muted" style="font-size:12px">${d.count} images · ${d.captioned} captioned</div></div></a>`).join("")}</div>`;
}

async function pageDataset(name) {
  const base = `/api/datasets/${enc(name)}`;
  view.innerHTML = `
    <div class="page-head"><div><h1>${esc(name)}</h1><div class="sub" id="ds-sub"></div></div>
      <div class="row">
        <select id="bulk" style="width:auto"><option value="">Bulk caption…</option><option value="prepend">Prepend text</option><option value="append">Append text</option><option value="replace">Find & replace</option><option value="fill_empty">Fill empty captions</option></select>
        <button class="btn" id="vlm" title="Write Fibo 2.0 VGL captions with a vision model (OpenRouter)">${I.spark} Caption with AI</button>
        <a class="btn btn-primary" href="#/new?dataset=${enc(name)}">${I.flask} Train on this</a>
        <button class="btn btn-ghost btn-danger" id="del-ds">${I.trash}</button></div></div>
    <div class="dropzone" id="drop">${I.upload}<div><b>Drop images, .txt/.json captions or a .zip here</b></div><div class="faint" style="font-size:12px">or click to browse · captions pair with images by filename</div>
      <input type="file" id="file" multiple accept="image/*,.txt,.json,.zip" hidden></div>
    <div id="cap-run"></div>
    <div style="height:18px"></div><div class="items" id="items"></div>`;

  let items = [];
  const isJson = (s) => s.trimStart().startsWith("{");
  async function load() {
    const d = await api(base);
    items = d.items;
    const cap = items.filter((i) => i.caption.trim()).length;
    $("#ds-sub").textContent = `${items.length} images · ${cap} captioned${items.length - cap ? ` · ${items.length - cap} missing captions` : ""}`;
    $("#items").innerHTML = items.length ? items.map((it, i) => `
      <div class="item ${it.caption.trim() ? "" : "nocap"}" data-file="${esc(it.file)}">
        <div class="img-wrap"><img loading="lazy" src="${base}/image/${enc(it.file)}?thumb=1&v=${it.mtime}" data-i="${i}">
          <button class="btn btn-sm vlm" data-vlm="${esc(it.file)}" title="Caption this image with AI (VGL)">${I.spark}</button>
          <button class="btn btn-sm del" data-del="${esc(it.file)}">${I.trash}</button></div>
        <textarea placeholder="Write a caption…" class="${isJson(it.caption) ? "json" : ""}">${esc(it.caption)}</textarea>
        <div class="meta"><span title="${esc(it.file)}">${esc(it.file.length > 28 ? it.file.slice(0, 25) + "…" : it.file)}</span><span class="state">${fmtBytes(it.size)}</span></div></div>`).join("")
      : `<div class="faint" style="grid-column:1/-1;text-align:center;padding:20px">Empty dataset</div>`;
  }
  await load();

  async function uploadFiles(files) {
    if (!files.length) return;
    const fd = new FormData();
    [...files].forEach((f) => fd.append("files", f));
    $("#drop").classList.add("over");
    $("#drop div").textContent = `Uploading ${files.length} file(s)…`;
    try {
      const r = await guard(() => api(`${base}/upload`, { method: "POST", body: fd }));
      toast(`Added ${r.images} images, ${r.captions} captions`, "ok");
    } finally {
      $("#drop").classList.remove("over");
      $("#drop div").innerHTML = "<b>Drop images, .txt captions or a .zip here</b>";
      await load();
    }
  }
  const drop = $("#drop");
  drop.onclick = () => $("#file").click();
  $("#file").onchange = (e) => uploadFiles(e.target.files);
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); uploadFiles(e.dataTransfer.files); };

  const timers = new Map();
  async function saveCaption(card) {
    const file = card.dataset.file, ta = $("textarea", card);
    await guard(() => api(`${base}/caption/${enc(file)}`, { method: "PUT", body: { caption: ta.value } }));
    card.classList.toggle("nocap", !ta.value.trim());
    const st = $(".state", card);
    st.innerHTML = '<span class="saved">Saved ✓</span>';
    const it = items.find((x) => x.file === file);
    if (it) it.caption = ta.value;
  }
  $("#items").addEventListener("input", (e) => {
    const card = e.target.closest(".item");
    if (!card) return;
    $(".state", card).textContent = "Editing…";
    clearTimeout(timers.get(card));
    timers.set(card, setTimeout(() => saveCaption(card), 900));
  });
  $("#items").addEventListener("focusout", (e) => {
    const card = e.target.closest(".item");
    if (card && e.target.tagName === "TEXTAREA" && timers.has(card)) { clearTimeout(timers.get(card)); timers.delete(card); saveCaption(card); }
  });
  $("#items").addEventListener("click", async (e) => {
    const one = e.target.closest("[data-vlm]");
    if (one) {
      if (!(await needOpenRouterKey())) return;
      await guard(() => api(`${base}/autocaption`, { method: "POST", body: { scope: "files", files: [one.dataset.vlm] } }));
      one.closest(".item").classList.add("captioning");
      return pollCaptions();
    }
    const del = e.target.closest("[data-del]");
    if (del) {
      if (!(await confirmBox("Delete image?", `Delete ${del.dataset.del} and its caption.`))) return;
      await guard(() => api(`${base}/item/${enc(del.dataset.del)}`, { method: "DELETE" }));
      return load();
    }
    const img = e.target.closest("img[data-i]");
    if (img) lightbox(items.map((it) => ({ src: `${base}/image/${enc(it.file)}`, title: it.file, caption: it.caption })), +img.dataset.i);
  });
  $("#bulk").onchange = async (e) => {
    const action = e.target.value;
    const title = e.target.selectedOptions?.[0]?.text || "Bulk caption";
    e.target.value = "";
    if (!action) return;
    const body = action === "replace"
      ? `<div class="grid"><label class="field">Find<input type="text" name="find"></label><label class="field">Replace with<input type="text" name="text"></label></div>`
      : `<label class="field">Text<input type="text" name="text" placeholder="${action === "prepend" ? "mytrigger, " : action === "append" ? ", extra tags" : "caption for images without one"}"><span class="help">${action === "prepend" || action === "append" ? "Include separators (e.g. a comma and space) yourself." : ""}</span></label>`;
    const v = await modal({ title, body, confirm: "Apply" });
    if (!v) return;
    const r = await guard(() => api(`${base}/bulk`, { method: "POST", body: { action, ...v } }));
    toast(`Updated ${r.changed} captions`, "ok");
    load();
  };
  $("#del-ds").onclick = async () => {
    if (!(await confirmBox("Delete dataset?", `Permanently delete "${name}" and all its images.`))) return;
    await guard(() => api(base, { method: "DELETE" }), "Dataset deleted");
    location.hash = "#/datasets";
  };

  // ---- AI captions (VGL for Fibo 2.0, via OpenRouter)
  async function needOpenRouterKey() {
    const s = await api("/api/settings");
    if (s.has_openrouter_key) return s;
    await modal({ title: "OpenRouter key needed", confirm: "Open Settings",
      body: `<p class="muted" style="margin:0">AI captioning sends each image to a vision model through <a href="https://openrouter.ai" target="_blank" rel="noopener">OpenRouter</a>. Add your OpenRouter API key in Settings first.</p>` })
      .then((v) => { if (v) location.hash = "#/settings"; });
    return null;
  }
  $("#vlm").onclick = async () => {
    const s = await needOpenRouterKey();
    if (!s) return;
    const missing = items.filter((i) => !i.caption.trim()).length;
    const v = await modal({ title: "Caption with AI", confirm: "Start captioning", body: `
      <p class="muted" style="margin:0 0 12px">Writes a structured <a href="https://docs.bria.ai/vgl" target="_blank" rel="noopener">VGL</a> caption (JSON) for each image, the format Fibo 2.0 is trained on, using <span class="mono">${esc(s.caption_model)}</span>. Your trigger word is added to the short description at training time.</p>
      <label class="field">Images<select name="scope">
        <option value="missing">Images without a caption (${missing})</option>
        <option value="all">All images (${items.length}), replacing existing captions</option></select></label>
      <label class="field" style="margin-top:12px">Extra instructions <span class="help">Optional, e.g. “The woman in every photo is called Mira.”</span>
        <textarea name="extra" rows="3"></textarea></label>` });
    if (!v) return;
    await guard(() => api(`${base}/autocaption`, { method: "POST", body: { scope: v.scope, extra: v.extra } }), "Captioning started");
    pollCaptions();
  };

  let capTimer = null, seen = 0, runStarted = null;
  async function pollCaptions() {
    const st = await api(`${base}/autocaption`);
    const el = $("#cap-run");
    if (!el) return;
    if (st.state === "idle") { el.innerHTML = ""; return; }
    if (st.started !== runStarted) { runStarted = st.started; seen = 0; }
    // Fill in captions as they land, without touching a caption being edited.
    if (st.finished && st.finished.length > seen) {
      const fresh = st.finished.slice(seen);
      seen = st.finished.length;
      const d = await api(base);
      for (const f of fresh) {
        const it = d.items.find((x) => x.file === f), card = $$("#items .item").find((c) => c.dataset.file === f);
        const local = items.find((x) => x.file === f);
        if (local && it) local.caption = it.caption;
        if (!card || !it) continue;
        card.classList.remove("captioning", "nocap");
        const ta = $("textarea", card);
        if (document.activeElement !== ta) { ta.value = it.caption; ta.classList.toggle("json", isJson(it.caption)); }
      }
      const cap = items.filter((i) => i.caption.trim()).length;
      $("#ds-sub").textContent = `${items.length} images · ${cap} captioned${items.length - cap ? ` · ${items.length - cap} missing captions` : ""}`;
    }
    const running = st.state === "running";
    const pctDone = st.total ? (100 * (st.done + st.failed)) / st.total : 0;
    el.innerHTML = `<div class="card cap-run"><div class="card-body">
      <div class="row" style="justify-content:space-between;margin-bottom:8px">
        <div>${running ? `Captioning with <span class="mono">${esc(st.model)}</span>…` : st.state === "cancelled" ? "Captioning stopped" : "Captioning finished"}
          <span class="muted">${st.done}/${st.total} done${st.failed ? `, ${st.failed} failed` : ""}</span></div>
        ${running ? `<button class="btn btn-sm" id="cap-cancel">${I.stop} Stop</button>` : `<button class="btn btn-sm btn-ghost" id="cap-close">${I.x}</button>`}</div>
      <div class="progress big"><div style="width:${pctDone.toFixed(1)}%"></div></div>
      ${st.errors && st.errors.length ? `<div class="errors">${st.errors.slice(-5).map((x) => `<div>${esc(x)}</div>`).join("")}</div>` : ""}</div></div>`;
    $("#cap-cancel")?.addEventListener("click", () => api(`${base}/autocaption/cancel`, { method: "POST" }).then(pollCaptions));
    $("#cap-close")?.addEventListener("click", () => { el.innerHTML = ""; });
    if (!running) $$("#items .item.captioning").forEach((c) => c.classList.remove("captioning"));
    clearTimeout(capTimer);
    if (running) capTimer = setTimeout(() => { if (document.body.contains(el)) pollCaptions(); }, 1500);
  }
  api(`${base}/autocaption`).then((st) => { if (st.state === "running") pollCaptions(); });
}


// ------------------------------------------------------------------ generate
const GEN_KEY = "moss.generate.v1";
const ASPECTS = [["1:1", 1, 1], ["4:3", 4, 3], ["3:4", 3, 4], ["3:2", 3, 2], ["2:3", 2, 3], ["16:9", 16, 9], ["9:16", 9, 16]];
const sizeFor = (a, b, mp) => {
  const area = mp * 1024 * 1024, w = Math.sqrt((area * a) / b);
  return [Math.round(w / 16) * 16, Math.round(area / w / 16) * 16];
};
function loadGenState() { try { return JSON.parse(localStorage.getItem(GEN_KEY)) || {}; } catch { return {}; } }
function saveGenState(st) { try { localStorage.setItem(GEN_KEY, JSON.stringify(st)); } catch { /* private mode */ } }

async function pageGenerate(params) {
  if (!INFO) INFO = await api("/api/info");
  let loras = await api("/api/loras");
  const defaults = { prompt: "", negative: "", loras: [], aspect: "1:1", mp: 1, width: 1024, height: 1024, steps: 50,
    cfg: 3, seed: -1, count: 1, shift: 5, skip_layer: 22, model: INFO.defaults.model, image: "" };
  const st = { ...defaults, ...loadGenState() };
  st.loras = (st.loras || []).filter((l) => loras.some((x) => x.id === l.id));
  if (params.lora) {
    const info = loras.find((x) => x.id === params.lora);
    if (info && !st.loras.some((l) => l.id === info.id)) st.loras.push({ id: info.id, strength: 1 });
    if (info && info.trigger && !st.prompt.includes(info.trigger)) st.prompt = `${info.trigger}, ${st.prompt}`.replace(/, $/, "");
    if (info && info.model) st.model = info.model;
    saveGenState(st);
    history.replaceState(null, "", "#/generate");
  }
  const loraInfo = (id) => loras.find((x) => x.id === id) || { label: id.split("/").pop(), trigger: "" };

  view.innerHTML = `
    <div class="page-head"><div><h1>Generate</h1><div class="sub">Try your LoRAs with each model's official sampler</div></div>
      <div class="row" id="g-worker"></div></div>
    <div class="gen-layout">
      <div class="card gen-controls"><div class="card-body gen-form">
        <label class="field">Prompt<textarea id="g-prompt" rows="5" placeholder="Describe the image… (⌘/Ctrl + Enter to generate)"></textarea></label>
        <label class="field">Negative prompt<textarea id="g-negative" rows="2" placeholder="Optional, e.g. worst quality, bad anatomy, watermark"></textarea></label>
        <div class="field" id="g-src-field" hidden><div class="row" style="justify-content:space-between"><span>Source image <span class="help">optional: edit this image</span></span>
          <label class="btn btn-sm btn-ghost" style="cursor:pointer">${I.upload} Upload<input type="file" id="g-src-file" accept="image/*" hidden></label></div>
          <div id="g-src"></div></div>
        <div class="field"><div class="row" style="justify-content:space-between"><span>LoRAs</span>
          <label class="btn btn-sm btn-ghost" style="cursor:pointer" title="Upload a .safetensors LoRA">${I.upload} Upload<input type="file" id="g-upload" accept=".safetensors" multiple hidden></label></div>
          <div id="g-loras" class="lora-list"></div>
          <select id="g-add"></select></div>
        <div class="field">Size
          <div class="chips" id="g-aspects">${ASPECTS.map(([k]) => `<span class="chip" data-aspect="${k}">${k}</span>`).join("")}</div>
          <div class="row" style="gap:8px;margin-top:4px;flex-wrap:nowrap">
            <select id="g-mp" style="width:96px"><option value="1">1 MP</option><option value="1.5">1.5 MP</option><option value="2">2 MP</option></select>
            <input type="number" id="g-w" step="16" min="256" max="2048"><span class="muted">×</span><input type="number" id="g-h" step="16" min="256" max="2048"></div></div>
        <div class="gen-nums">
          <label class="field">Steps<input type="number" id="g-steps" min="1" max="150"></label>
          <label class="field">CFG<input type="number" id="g-cfg" step="0.1" min="0" max="20"></label>
          <label class="field">Images<input type="number" id="g-count" min="1" max="8"></label></div>
        <label class="field">Seed <span class="help">−1 = random each time</span>
          <div class="row" style="gap:6px;flex-wrap:nowrap"><input type="number" id="g-seed">
          <button class="btn btn-sm" id="g-random" title="Random">${I.dice}</button></div></label>
        <details><summary>Advanced</summary><div class="gen-nums" style="margin-top:10px">
          <label class="field" style="grid-column:1/-1">Base model<select id="g-model">${Object.entries(INFO.models).map(([k, v]) => `<option value="${esc(k)}">${esc(v)}</option>`).join("")}</select></label>
          <label class="field">Shift<input type="number" id="g-shift" step="0.1"></label>
          <label class="field">Skip layer<input type="number" id="g-skip" min="-1" max="41" title="Negative-branch layer skip, -1 = off"></label></div></details>
        <div class="faint" id="g-model-badge" style="font-size:12px;margin-bottom:-6px"></div>
        <button class="btn btn-primary" id="g-go">${I.spark} Generate</button>
      </div></div>
      <div class="gen-results"><div class="card" id="g-status"></div><div id="g-gallery" class="gen-gallery"></div><div id="g-more"></div></div>
    </div>`;

  // ---- form <-> state
  const fields = { prompt: "#g-prompt", negative: "#g-negative", width: "#g-w", height: "#g-h", steps: "#g-steps", cfg: "#g-cfg",
    count: "#g-count", seed: "#g-seed", shift: "#g-shift", skip_layer: "#g-skip", model: "#g-model", mp: "#g-mp" };
  const numeric = new Set(["width", "height", "steps", "cfg", "count", "seed", "shift", "skip_layer", "mp"]);
  for (const [k, sel] of Object.entries(fields)) $(sel).value = st[k];
  const syncAspect = () => $$("#g-aspects .chip").forEach((c) => c.classList.toggle("on", c.dataset.aspect === st.aspect));
  syncAspect();
  const applyAspect = () => {
    const a = ASPECTS.find(([k]) => k === st.aspect);
    if (!a) return;
    [st.width, st.height] = sizeFor(a[1], a[2], +st.mp);
    $("#g-w").value = st.width; $("#g-h").value = st.height;
  };
  view.addEventListener("input", (e) => {
    const k = Object.keys(fields).find((key) => e.target.matches(fields[key]));
    if (!k) return;
    st[k] = numeric.has(k) ? Number(e.target.value) : e.target.value;
    if (k === "width" || k === "height") { st.aspect = ""; syncAspect(); }
    if (k === "mp") applyAspect();
    saveGenState(st);
  });
  const family = () => (INFO.model_specs[st.model] || {}).family;
  $("#g-model").onchange = (e) => {
    st.model = e.target.value;
    const d = (INFO.model_specs[st.model] || {}).defaults || {};
    Object.assign(st, { steps: d.steps ?? st.steps, cfg: d.cfg ?? st.cfg, shift: d.shift ?? st.shift, skip_layer: d.skip_layer ?? st.skip_layer });
    for (const k of ["steps", "cfg", "shift", "skip_layer"]) $(fields[k]).value = st[k];
    const prevNeg = ((INFO.model_specs[prevModel] || {}).defaults || {}).negative || "";
    if (st.negative === prevNeg) { st.negative = d.negative || ""; $("#g-negative").value = st.negative; }
    prevModel = st.model;
    saveGenState(st); renderLoras(); renderModelBadge(); renderSource();
    toast("Loaded the recommended settings for this model", "ok");
  };
  let prevModel = st.model;

  // ---- source image (Fibo 2.0 edits)
  function renderSource() {
    const edit = !!(INFO.model_specs[st.model] || {}).edit;
    $("#g-src-field").hidden = !edit;
    $("#g-prompt").placeholder = edit
      ? (st.image ? "The edit instruction, e.g. “Make the jacket red leather”… (⌘/Ctrl + Enter)" : "Describe the image, or paste a VGL JSON caption… (⌘/Ctrl + Enter)")
      : "Describe the image… (⌘/Ctrl + Enter to generate)";
    $("#g-src").innerHTML = st.image
      ? `<div class="src-img"><img src="/api/generations/input/${enc(st.image)}"><button class="btn btn-sm" id="g-src-remove" title="Remove">${I.x}</button></div>
         <div class="help" style="margin-top:4px">The result keeps the source's aspect ratio at about the size set below.${st.model.includes("turbo") ? "" : " Edit Turbo does this in 4 steps."}</div>`
      : `<div class="faint" style="font-size:12px">None: text-to-image${st.model.includes("turbo") ? ". Edit Turbo is made for editing; text-to-image works best on Fibo 2.0 Dev." : ""}</div>`;
  }
  renderSource();
  $("#g-src-file").onchange = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f);
    const r = await guard(() => api("/api/generate/input", { method: "POST", body: fd }));
    st.image = r.id; saveGenState(st); renderSource();
    e.target.value = "";
  };
  $("#g-src").onclick = (e) => { if (e.target.closest("#g-src-remove")) { st.image = ""; saveGenState(st); renderSource(); } };
  $("#g-mp").onchange = (e) => { st.mp = +e.target.value; applyAspect(); saveGenState(st); };
  $("#g-aspects").onclick = (e) => {
    const c = e.target.closest("[data-aspect]");
    if (!c) return;
    st.aspect = c.dataset.aspect; syncAspect(); applyAspect(); saveGenState(st);
  };
  $("#g-random").onclick = (e) => { e.preventDefault(); st.seed = -1; $("#g-seed").value = -1; saveGenState(st); };

  function renderModelBadge() {
    $("#g-model-badge").textContent = `Model: ${(INFO.model_specs[st.model] || {}).label || st.model} (change under Advanced)`;
  }
  renderModelBadge();

  // ---- LoRA list
  function renderLoras() {
    $("#g-loras").innerHTML = st.loras.length ? st.loras.map((l, i) => {
      const info = loraInfo(l.id);
      return `<div class="lora-row" data-i="${i}">
        <div class="lora-name" title="${esc(l.id)}">${esc(info.label)}${info.trigger ? `<span class="faint"> · ${esc(info.trigger)}</span>` : ""}${info.family && info.family !== family() ? `<span style="color:var(--red)"> · other model</span>` : ""}</div>
        <input type="range" min="-1" max="2" step="0.05" value="${l.strength}" data-strength>
        <input type="number" step="0.05" value="${l.strength}" data-strength-num>
        ${info.trigger ? `<button class="btn btn-sm btn-ghost" data-trigger title="Add trigger word to prompt">T</button>` : "<span></span>"}
        <button class="btn btn-sm btn-ghost" data-remove title="Remove">${I.x}</button></div>`;
    }).join("") : `<div class="faint" style="font-size:12px">No LoRA, base model only</div>`;
    const groups = {};
    loras.filter((x) => !st.loras.some((l) => l.id === x.id) && (!x.family || x.family === family())).forEach((x) => (groups[x.group] = groups[x.group] || []).push(x));
    $("#g-add").innerHTML = `<option value="">${loras.length ? "+ Add a LoRA…" : "No LoRAs yet: train one first"}</option>` +
      Object.entries(groups).map(([g, xs]) => `<optgroup label="${esc(g)}">${xs.map((x) => `<option value="${esc(x.id)}">${esc(x.label)}</option>`).join("")}</optgroup>`).join("");
  }
  renderLoras();
  $("#g-add").onchange = (e) => {
    if (!e.target.value) return;
    st.loras.push({ id: e.target.value, strength: 1 });
    saveGenState(st); renderLoras();
  };
  $("#g-loras").addEventListener("input", (e) => {
    const row = e.target.closest(".lora-row");
    if (!row) return;
    const l = st.loras[+row.dataset.i], v = Number(e.target.value);
    if (!isFinite(v)) return;
    l.strength = v;
    if (e.target.matches("[data-strength]")) $("[data-strength-num]", row).value = v;
    else $("[data-strength]", row).value = v;
    saveGenState(st);
  });
  $("#g-loras").addEventListener("click", (e) => {
    const row = e.target.closest(".lora-row");
    if (!row) return;
    const l = st.loras[+row.dataset.i];
    if (e.target.closest("[data-remove]")) { st.loras.splice(+row.dataset.i, 1); saveGenState(st); renderLoras(); }
    if (e.target.closest("[data-trigger]")) {
      const t = loraInfo(l.id).trigger;
      if (!st.prompt.includes(t)) { st.prompt = st.prompt ? `${t}, ${st.prompt}` : t; $("#g-prompt").value = st.prompt; saveGenState(st); }
    }
  });
  $("#g-upload").onchange = async (e) => {
    const fd = new FormData();
    [...e.target.files].forEach((f) => fd.append("files", f));
    if (!e.target.files.length) return;
    toast("Uploading LoRA…");
    const r = await guard(() => api("/api/loras/upload", { method: "POST", body: fd }));
    loras = await api("/api/loras");
    r.added.forEach((f) => st.loras.push({ id: `loras/${f}`, strength: 1 }));
    saveGenState(st); renderLoras();
    toast(`Added ${r.added.join(", ")}`, "ok");
  };

  // ---- submit
  async function submit() {
    const body = { ...st, loras: st.loras.map((l) => ({ id: l.id, strength: l.strength, name: loraInfo(l.id).label })) };
    if (!(INFO.model_specs[st.model] || {}).edit) delete body.image;
    await guard(() => api("/api/generate", { method: "POST", body }));
    refreshStatus();
  }
  $("#g-go").onclick = submit;
  $("#g-prompt").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); submit(); } });

  // ---- status
  const W_LABEL = { unloaded: "Model not loaded", starting: "Starting", loading: "Loading model", idle: "Starting", ready: "Model ready", generating: "Generating", exiting: "Unloading" };
  async function refreshStatus() {
    const g = await api("/api/generate/status");
    const busy = g.current || g.queue.length;
    $("#g-worker").innerHTML = `<span class="badge ${g.state === "ready" ? "done" : g.state === "unloaded" ? "" : "running"}">${esc(W_LABEL[g.state] || g.state)}${g.model ? ` · ${esc(g.model.split("/").pop())}` : ""}</span>
      ${g.state !== "unloaded" ? `<button class="btn btn-sm btn-ghost" id="g-unload" title="Free GPU memory">Unload</button>` : ""}`;
    let html = "";
    if (g.current) {
      const p = g.current.progress;
      const frac = p ? (p.image + p.step / p.steps) / p.count : 0;
      html += `<div class="card-body"><div class="row" style="justify-content:space-between;margin-bottom:8px">
        <div class="muted" style="min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1">${esc(g.current.prompt)}</div>
        <button class="btn btn-sm" data-cancel="${esc(g.current.id)}">${I.stop} Cancel</button></div>
        <div class="progress big"><div style="width:${(frac * 100).toFixed(1)}%"></div></div>
        <div class="faint mono" style="font-size:12px;margin-top:6px">${p ? `image ${p.image + 1}/${p.count} · step ${p.step}/${p.steps}` : esc(g.message || "Preparing…")}</div></div>`;
    }
    if (g.queue.length) html += `<div class="card-body faint" style="font-size:12px;border-top:1px solid var(--border)">${g.queue.length} more queued</div>`;
    if (g.training && busy) html += `<div class="card-body faint" style="font-size:12px;border-top:1px solid var(--border)">A training job is running, so the text encoder stays off the GPU and new prompts encode a little slower.</div>`;
    if (g.last_error) html += `<div class="card-body" style="color:var(--red);font-size:13px;border-top:1px solid var(--border)">${esc(g.last_error)}</div>`;
    if (!html) html = `<div class="card-body faint" style="font-size:13px">${g.state === "unloaded" ? "The model loads on your first generation (about a minute), then stays loaded until it's idle for 15 minutes." : "Ready. Write a prompt and hit Generate."}</div>`;
    $("#g-status").innerHTML = html;
    if (g.version !== lastVersion) { lastVersion = g.version; loadGallery(); }
  }
  $("#g-status").onclick = async (e) => {
    const c = e.target.closest("[data-cancel]");
    if (c) { await guard(() => api(`/api/generate/cancel/${enc(c.dataset.cancel)}`, { method: "POST" })); refreshStatus(); }
  };
  $("#g-worker").onclick = async (e) => {
    if (e.target.closest("#g-unload")) { await guard(() => api("/api/generate/unload", { method: "POST" }), "Model unloaded"); refreshStatus(); }
  };

  // ---- gallery
  let items = [], lastVersion = null, limit = 60;
  const fileUrl = (f) => `/api/generations/file/${enc(f)}`;
  const describe = (it) => [...(it.image_id ? ["edit"] : []), `seed ${it.seed}`, `${it.width}×${it.height}`, `${it.steps} steps`, `cfg ${it.cfg}`,
    ...(it.loras || []).map((l) => `${l.name} @ ${l.strength}`)].join(" · ");
  async function loadGallery() {
    const r = await api(`/api/generations?limit=${limit}`);
    items = r.items;
    $("#g-gallery").innerHTML = items.length ? items.map((it, i) => `
      <div class="gen-card" data-i="${i}">
        <img loading="lazy" src="${fileUrl(it.thumb)}" style="aspect-ratio:${it.width}/${it.height}" data-open="${i}">
        <div class="gen-ov"><div class="gen-ov-text">${esc(it.prompt)}</div><div class="row" style="gap:4px;flex-wrap:nowrap">
          <button class="btn btn-sm" data-reuse="${i}" title="Use these settings">${I.reuse}</button>
          <a class="btn btn-sm" href="${fileUrl(it.file)}" download title="Download PNG">${I.download}</a>
          <button class="btn btn-sm btn-danger" data-del="${i}" title="Delete">${I.trash}</button></div></div></div>`).join("")
      : `<div class="card empty" style="grid-column:1/-1">${I.image}<div>Generated images will show up here</div></div>`;
    $("#g-more").innerHTML = r.total > items.length ? `<div style="text-align:center;margin-top:14px"><button class="btn" id="g-load-more">Show more (${r.total - items.length})</button></div>` : "";
  }
  $("#g-more").onclick = (e) => { if (e.target.closest("#g-load-more")) { limit += 60; loadGallery(); } };
  $("#g-gallery").addEventListener("click", async (e) => {
    const open = e.target.closest("[data-open]"), reuse = e.target.closest("[data-reuse]"), del = e.target.closest("[data-del]");
    if (open) lightbox(items.map((it) => ({ src: fileUrl(it.file), title: describe(it), caption: it.prompt })), +open.dataset.open);
    if (reuse) {
      const it = items[+reuse.dataset.reuse];
      Object.assign(st, { prompt: it.prompt, negative: it.negative || "", width: it.width, height: it.height, steps: it.steps,
        cfg: it.cfg, shift: it.shift ?? 5, skip_layer: it.skip_layer ?? 22, seed: it.seed, model: it.model || st.model, aspect: "", image: it.image_id || "",
        loras: (it.loras || []).filter((l) => loras.some((x) => x.id === l.path)).map((l) => ({ id: l.path, strength: l.strength })) });
      saveGenState(st);
      for (const [k, sel] of Object.entries(fields)) if (k !== "mp") $(sel).value = st[k];
      syncAspect(); renderLoras(); renderSource(); renderModelBadge(); prevModel = st.model;
      window.scrollTo({ top: 0, behavior: "smooth" });
      toast("Settings loaded (same seed)", "ok");
    }
    if (del) {
      if (!(await confirmBox("Delete image?", "This permanently deletes the image."))) return;
      await guard(() => api(`/api/generations/${enc(items[+del.dataset.del].file)}`, { method: "DELETE" }));
      loadGallery();
    }
  });

  every(refreshStatus, 1500);
}

// ------------------------------------------------------------------ settings
async function pageSettings() {
  let s = await api("/api/settings");
  view.innerHTML = `
    <div class="page-head"><div><h1>Settings</h1><div class="sub">API keys and model downloads. Keys are stored on this machine (<span class="mono">workspace/settings.json</span>) and never shown in full again.</div></div></div>
    <div class="settings">
      <div class="card"><div class="card-head"><h2>Hugging Face</h2><span id="hf-state"></span></div><div class="card-body">
        <div class="field">Access token
          <div class="secret-row"><input type="password" id="hf-token" autocomplete="off" placeholder="hf_…">
            <button class="btn btn-primary" id="hf-save">Save</button><button class="btn btn-ghost" id="hf-clear">Remove</button></div>
          <span class="help">Needed for gated models. Fibo 2.0 is gated: open <a href="https://huggingface.co/briaai/Fibo-2.0-dev" target="_blank" rel="noopener">Fibo 2.0 Dev</a> and <a href="https://huggingface.co/briaai/Fibo-2.0-edit-turbo" target="_blank" rel="noopener">Fibo 2.0 Edit Turbo</a>, accept the license with your account, then create a <b>Read</b> token at <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener">huggingface.co/settings/tokens</a> and paste it here.</span></div>
        <div class="row"><button class="btn" id="hf-check">Check access</button><div id="hf-result" class="access-list"></div></div>
      </div></div>
      <div class="card"><div class="card-head"><h2>Models</h2><span class="faint" style="font-size:12px">Models also download on first use</span></div><div id="models"></div></div>
      <div class="card"><div class="card-head"><h2>AI captioning (OpenRouter)</h2><span id="or-state"></span></div><div class="card-body">
        <div class="field">OpenRouter API key
          <div class="secret-row"><input type="password" id="or-key" autocomplete="off" placeholder="sk-or-…">
            <button class="btn btn-primary" id="or-save">Save</button><button class="btn btn-ghost" id="or-clear">Remove</button></div>
          <span class="help">Used by “Caption with AI” in the dataset viewer to write Fibo 2.0 VGL captions. Get a key at <a href="https://openrouter.ai/keys" target="_blank" rel="noopener">openrouter.ai/keys</a>.</span></div>
        <label class="field">Vision model
          <div class="secret-row"><input type="text" id="or-model" list="or-models"><button class="btn" id="or-model-save">Save</button><span></span></div>
          <datalist id="or-models">${["google/gemini-3.5-flash-lite", "google/gemini-3.5-flash", "google/gemini-2.5-flash", "qwen/qwen3-vl-235b-a22b-instruct", "anthropic/claude-haiku-5.5", "openai/gpt-5-mini"].map((m) => `<option value="${m}">`).join("")}</datalist>
          <span class="help">Any OpenRouter model that takes images. The default, <span class="mono">${esc(s.caption_model_default)}</span>, is what Bria's own VGL tool uses.</span></label>
        <div class="row"><button class="btn" id="or-check">Check key</button><div id="or-result" class="muted" style="font-size:12.5px"></div></div>
      </div></div>
    </div>`;

  const keyBadge = (has, hint) => `<span class="badge ${has ? "done" : ""}">${has ? `Saved · ${esc(hint)}` : "Not set"}</span>`;
  function render() {
    $("#hf-state").innerHTML = s.has_hf_token ? keyBadge(true, s.hf_token) : s.env_hf_token ? `<span class="badge done">From environment (HF_TOKEN)</span>` : keyBadge(false);
    $("#or-state").innerHTML = keyBadge(s.has_openrouter_key, s.openrouter_key);
    $("#or-model").value = s.caption_model;
  }
  render();
  const save = async (body, msg) => { s = await guard(() => api("/api/settings", { method: "PUT", body }), msg); render(); };
  $("#hf-save").onclick = async () => {
    const v = $("#hf-token").value.trim();
    if (!v) return toast("Paste a token first", "error");
    await save({ hf_token: v }, "Hugging Face token saved");
    $("#hf-token").value = "";
    $("#hf-check").click();
  };
  $("#hf-clear").onclick = () => save({ hf_token: "" }, "Token removed");
  $("#or-save").onclick = async () => {
    const v = $("#or-key").value.trim();
    if (!v) return toast("Paste a key first", "error");
    await save({ openrouter_key: v }, "OpenRouter key saved");
    $("#or-key").value = "";
  };
  $("#or-clear").onclick = () => save({ openrouter_key: "" }, "Key removed");
  $("#or-model-save").onclick = () => save({ caption_model: $("#or-model").value.trim() }, "Caption model saved");
  $("#hf-check").onclick = async () => {
    $("#hf-result").innerHTML = '<span class="faint">Checking…</span>';
    const r = await api("/api/settings/check-hf", { method: "POST" });
    const label = { ok: ["ok", "access granted"], gated: ["bad", "license not accepted yet: open the model page and accept it"], no_access: ["bad", "no access (check the token)"] };
    $("#hf-result").innerHTML = (r.user ? `<div class="ok">Signed in as <b>${esc(r.user)}</b></div>` : `<div class="bad">${esc(r.error || "No token")}</div>`) +
      Object.entries(r.access).map(([repo, a]) => {
        const [cls, text] = !r.user && a !== "ok" ? ["bad", "needs a token"] : label[a] || ["bad", a];
        return `<div class="${cls}"><a href="https://huggingface.co/${esc(repo)}" target="_blank" rel="noopener">${esc(repo)}</a>: ${esc(text)}</div>`;
      }).join("");
  };
  $("#or-check").onclick = async () => {
    $("#or-result").textContent = "Checking…";
    try {
      const k = await api("/api/settings/check-openrouter", { method: "POST" });
      $("#or-result").innerHTML = `<span style="color:var(--green)">Key works</span>${k.label ? ` · ${esc(k.label)}` : ""}${k.usage != null ? ` · used $${Number(k.usage).toFixed(2)}` : ""}${k.limit != null ? ` of $${Number(k.limit).toFixed(2)}` : ""}`;
    } catch (e) { $("#or-result").innerHTML = `<span style="color:var(--red)">${esc(e.message)}</span>`; }
  };

  const DL = { none: "", downloading: "running", done: "done", error: "error" };
  const DL_LABEL = { none: "Not downloaded", downloading: "Downloading", done: "Downloaded", error: "Failed" };
  let modelsKey = "";
  every(async () => {
    const models = await api("/api/models");
    const key = JSON.stringify(models);
    if (key === modelsKey) return;
    modelsKey = key;
    $("#models").innerHTML = `<table><thead><tr><th>Model</th><th>Use</th><th>Status</th><th></th></tr></thead><tbody>
      ${models.map((m) => `<tr><td><b>${esc(m.label)}</b><div class="faint mono" style="font-size:11.5px">${m.repo ? `<a href="https://huggingface.co/${esc(m.repo)}" target="_blank" rel="noopener">${esc(m.repo)}</a>` : esc(m.id)}${m.gated ? " · gated" : ""}</div></td>
        <td class="muted">${m.trainable ? "Training + generation" : "Generation"}</td>
        <td><span class="badge ${DL[m.download.state]}">${DL_LABEL[m.download.state]}</span>${m.download.message ? `<div class="faint mono dl-msg" title="${esc(m.download.message)}">${esc(m.download.message)}</div>` : ""}</td>
        <td style="text-align:right">${m.download.state === "downloading" ? "" : `<button class="btn btn-sm" data-dl="${esc(m.id)}">${I.download} ${m.download.state === "done" ? "Verify" : "Download"}</button>`}</td></tr>`).join("")}
      </tbody></table>`;
  }, 2000);
  $("#models").onclick = async (e) => {
    const b = e.target.closest("[data-dl]");
    if (!b) return;
    b.disabled = true;
    await guard(() => api("/api/models/download", { method: "POST", body: { model: b.dataset.dl } }), "Download started");
    modelsKey = "";
  };
}

// ------------------------------------------------------------------ boot
refreshGpu();
setInterval(refreshGpu, 5000);
route();
