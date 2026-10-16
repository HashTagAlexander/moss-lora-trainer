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
async function pageJob(name) {
  let job = await api(`/api/jobs/${enc(name)}`);
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
      <td class="muted">${timeAgo(c.mtime)}</td><td style="text-align:right"><a class="btn btn-sm" href="${base}/checkpoint/${enc(c.file)}" download>${I.download} Download</a></td></tr>`).join("")}
      </tbody></table><div class="card-body faint" style="font-size:12px;border-top:1px solid var(--border)">
      LoRA files use ComfyUI-compatible keys (<span class="mono">diffusion_model.layers.N.proj.*</span>). Load with a standard LoraLoaderModelOnly node on top of the MOSS-Image loader, or use <span class="mono">generate.py</span>.</div>`;
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

  const sections = [
    ["Job", [
      F.text("name", "Job name", "Also used as the LoRA filename.", mode === "edit" ? "readonly" : ""),
      F.sel("model", "Base model", Object.entries(INFO.models)),
      F.sel("dataset", "Dataset", datasets.length ? datasets.map((d) => [d.name, `${d.name} (${d.count} images)`]) : [["", "— create a dataset first —"]],
        `<a href="#/datasets">Manage datasets</a>`),
      F.text("trigger_word", "Trigger word", "Prepended to captions; replaces <span class='mono'>[trigger]</span> in captions and sample prompts."),
    ]],
    ["Network", [
      F.num("network.rank", "Rank", "16–64 is typical for characters / styles.", 'min="1" max="512"'),
      F.num("network.alpha", "Alpha", "Scale = alpha / rank.", 'step="any"'),
      F.num("network.dropout", "Dropout", "", 'step="0.01" min="0" max="0.5"'),
      F.text("network.layers", "Blocks", "Blank = all 42 blocks; or e.g. <span class='mono'>0-20,30</span>."),
      F.chips("network.targets", "Target modules", [["attn", "Attention (q, k, v, o)"], ["ffn", "Feed-forward"], ["text", "Text refiner blocks"]]),
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
      F.sel("train.timestep_sampling", "Timestep sampling", [["moss", "MOSS pretraining (truncated logit-normal)"], ["logit_normal", "Logit-normal"], ["shift", "Shifted uniform"], ["uniform", "Uniform"]],
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
      F.num("sample.cfg", "CFG", "", 'step="0.1"'),
      F.num("sample.shift", "Scheduler shift", "", 'step="0.1"'),
      F.num("sample.seed", "Seed"),
      F.num("sample.uncond_skip_layer", "Uncond skip layer", "Negative-branch layer skip (official default 22, −1 off)."),
      F.bool("sample.sample_first", "Sample before training", "Baseline images at step 0."),
      F.area("sample.prompts", "Sample prompts", "One per line. <span class='mono'>[trigger]</span> is replaced by your trigger word."),
      { path: "sample.negative_prompt", label: "Negative prompt", help: "Leave empty to match the official default.", type: "text", attrs: "", full: true },
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
  form.addEventListener("click", (e) => { const c = e.target.closest(".chip"); if (c) c.classList.toggle("on"); });
  $("#json-mode").onchange = (e) => {
    if (e.target.checked) { $("#json").value = JSON.stringify(readForm(form, cfg), null, 2); }
    else {
      try { cfg = JSON.parse($("#json").value); } catch (err) { toast("Invalid JSON: " + err.message, "error"); e.target.checked = true; return; }
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
        <a class="btn btn-primary" href="#/new?dataset=${enc(name)}">${I.flask} Train on this</a>
        <button class="btn btn-ghost btn-danger" id="del-ds">${I.trash}</button></div></div>
    <div class="dropzone" id="drop">${I.upload}<div><b>Drop images, .txt captions or a .zip here</b></div><div class="faint" style="font-size:12px">or click to browse · captions pair with images by filename</div>
      <input type="file" id="file" multiple accept="image/*,.txt,.zip" hidden></div>
    <div style="height:18px"></div><div class="items" id="items"></div>`;

  let items = [];
  async function load() {
    const d = await api(base);
    items = d.items;
    const cap = items.filter((i) => i.caption.trim()).length;
    $("#ds-sub").textContent = `${items.length} images · ${cap} captioned${items.length - cap ? ` · ${items.length - cap} missing captions` : ""}`;
    $("#items").innerHTML = items.length ? items.map((it, i) => `
      <div class="item ${it.caption.trim() ? "" : "nocap"}" data-file="${esc(it.file)}">
        <div class="img-wrap"><img loading="lazy" src="${base}/image/${enc(it.file)}?thumb=1&v=${it.mtime}" data-i="${i}">
          <button class="btn btn-sm del" data-del="${esc(it.file)}">${I.trash}</button></div>
        <textarea placeholder="Write a caption…">${esc(it.caption)}</textarea>
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
}

// ------------------------------------------------------------------ boot
refreshGpu();
setInterval(refreshGpu, 5000);
route();
