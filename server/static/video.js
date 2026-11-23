/* Generate tab, Video section: Sulphur 3 (a MiniMax H3 fine-tune). Video with sound, 5 to 15 seconds at 24 fps,
 * from a prompt and optionally a first and/or last frame. Generation only; see mosstrainer/video.py.
 */
"use strict";

const GenVideo = (() => {
  const KEY = "moss.video.v1";
  // the model makes 17 * n + 5 frames at 24 fps, n = 7 … 20
  const LENGTHS = Array.from({ length: 14 }, (_, i) => (17 * (i + 7) + 5) / 24);
  // 768 px short edge, sides in multiples of 32
  const SHAPES = { "16:9": [1344, 768], "9:16": [768, 1344], "1:1": [768, 768], "4:3": [1024, 768], "3:4": [768, 1024] };
  const inputUrl = (id) => `/api/generations/input/${enc(id)}`;
  const fileUrl = (f) => `/api/videos/file/${enc(f)}`;
  const load = () => { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; } };

  function mount(root, { gpu }) {
    const models = INFO.video_models || {};
    const ids = Object.keys(models);
    if (!ids.length) return { refresh() {} };
    const first = models[ids[0]].defaults || {};
    const st = { prompt: "", model: ids[0], seconds: LENGTHS[0], shape: "16:9", steps: first.steps || 20, seed: -1, image: "", last_image: "", turbo: false, ...load() };
    if (!models[st.model]) st.model = ids[0];
    const turboSteps = () => models[st.model].turbo_steps;
    const save = () => { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch { /* private mode */ } };
    let items = [], download = null, polling = false;

    root.innerHTML = `
      <div class="page-head video-head"><div><h1>Video</h1><div class="sub">Video with sound, 5 to 15 seconds at 24 fps. Generation only.</div></div></div>
      <div class="gen-layout">
        <div class="card"><div class="card-body gen-form">
          <label class="field">Prompt<textarea id="v-prompt" rows="5" placeholder="Describe the shot: what is seen, how it moves, what is heard…"></textarea></label>
          <div class="gen-nums" style="grid-template-columns:1fr 1fr">
            <div class="field"><span>First frame <span class="help">optional</span></span><div id="v-image"></div></div>
            <div class="field"><span>Last frame <span class="help">optional</span></span><div id="v-last_image"></div></div></div>
          <div class="field">Shape<div class="chips" id="v-shapes">${Object.keys(SHAPES).map((k) => `<span class="chip" data-shape="${k}">${k}</span>`).join("")}</div>
            <div class="help" id="v-shape-note"></div></div>
          <div class="gen-nums">
            <label class="field">Length<select id="v-seconds">${LENGTHS.map((s) => `<option value="${s}">${s.toFixed(1)} s</option>`).join("")}</select></label>
            <label class="field">Steps<input type="number" id="v-steps" min="1" max="100"></label>
            <label class="field">Seed<input type="number" id="v-seed" title="-1 = random each time"></label></div>
          <label class="check" id="v-turbo-row" style="font-size:13px"><input type="checkbox" id="v-turbo"> <span id="v-turbo-text"></span></label>
          <label class="field" ${ids.length > 1 ? "" : "hidden"}>Video model<select id="v-model">${ids.map((k) => `<option value="${esc(k)}">${esc(models[k].label)}</option>`).join("")}</select></label>
          <div id="v-download"></div>
          <button class="btn btn-primary" id="v-go" style="justify-content:center;padding:10px;font-size:14px">${I.spark} Generate video</button>
          <div class="help">Uses the GPU picked above. A video model and an image model can't be loaded on one GPU at the same time, so switching between them reloads.</div>
        </div></div>
        <div class="gen-results"><div id="v-gallery" class="video-gallery"></div></div>
      </div>`;

    // ---- form
    for (const k of ["prompt", "steps", "seed", "model"]) $(`#v-${k}`, root).value = st[k];
    $("#v-seconds", root).value = LENGTHS.reduce((a, b) => (Math.abs(b - st.seconds) < Math.abs(a - st.seconds) ? b : a));
    root.addEventListener("input", (e) => {
      const k = (e.target.id || "").replace(/^v-/, "");
      if (!["prompt", "steps", "seed", "seconds", "model"].includes(k)) return;
      if (k === "model" && !models[e.target.value]) return;
      st[k] = k === "prompt" || k === "model" ? e.target.value : Number(e.target.value);
      if (k === "model") {  // another model: its own step count, turbo off
        st.turbo = false;
        st.steps = (models[st.model].defaults || {}).steps || 20;
        $("#v-steps", root).value = st.steps;
        renderTurbo(); loadDownload();
      }
      save();
    });
    // Turbo: the model's step-distillation LoRA. Switching it sets the step count it was made for.
    function renderTurbo() {
      const n = turboSteps();
      if (!n) st.turbo = false;
      $("#v-turbo-row", root).hidden = !n;
      $("#v-turbo", root).checked = !!st.turbo;
      $("#v-turbo-text", root).textContent = n ? `Turbo: ${n} steps with a speed LoRA (faster, some quality cost)` : "";
    }
    $("#v-turbo", root).onchange = (e) => {
      st.turbo = e.target.checked;
      st.steps = st.turbo ? turboSteps() : (models[st.model].defaults || {}).steps || 20;
      $("#v-steps", root).value = st.steps;
      save();
    };
    renderTurbo();
    function renderShape() {
      $$("#v-shapes .chip", root).forEach((c) => c.classList.toggle("on", !st.image && c.dataset.shape === st.shape));
      $("#v-shape-note", root).textContent = st.image ? "With a first frame, the video takes the frame's shape (768 px short side)."
        : `${SHAPES[st.shape][0]} × ${SHAPES[st.shape][1]}`;
    }
    $("#v-shapes", root).onclick = (e) => {
      const c = e.target.closest("[data-shape]");
      if (c && !st.image) { st.shape = c.dataset.shape; save(); renderShape(); }
    };
    function renderFrame(key) {
      $(`#v-${key}`, root).innerHTML = st[key]
        ? `<div class="src-img"><img src="${inputUrl(st[key])}" alt=""><button class="btn btn-sm" data-remove="${key}" title="Remove">${I.x}</button></div>`
        : `<label class="btn btn-sm btn-ghost" style="cursor:pointer">${I.upload} Upload<input type="file" data-frame="${key}" accept="image/*" hidden></label>`;
    }
    const renderFrames = () => { renderFrame("image"); renderFrame("last_image"); renderShape(); };
    renderFrames();
    root.addEventListener("change", async (e) => {
      const key = e.target.dataset.frame;
      if (!key || !e.target.files[0]) return;
      const fd = new FormData();
      fd.append("file", e.target.files[0]);
      st[key] = (await guard(() => api("/api/generate/input", { method: "POST", body: fd }))).id;
      save(); renderFrames();
    });

    // ---- the model's download (about 78 GB: say so before the first run)
    function renderDownload() {
      const d = download;
      if (!d) { $("#v-download", root).innerHTML = ""; return; }
      const state = d.download.state === "done" ? `<span class="addon-ok">${I.check} Ready</span>`
        : d.download.state === "downloading" ? `<span class="faint">Downloading…</span>`
        : `<button class="btn btn-sm" data-get>${I.download} Download</button>`;
      $("#v-download", root).innerHTML = `<div class="addon"><span>${esc(d.label)}</span><span class="grow"></span>${state}</div>
        ${d.download.state === "downloading" ? `<div class="faint mono dl-msg" style="max-width:none" title="${esc(d.download.message)}">${esc(d.download.message)}</div>` : ""}
        ${d.download.state === "error" ? `<div class="help" style="color:var(--red)">${esc(d.download.message)}</div>` : ""}
        ${d.download.state === "none" ? `<div class="help">About 80 GB the first time: the video model (40 GB), its text encoder (27 GB, int8), the decoders (11 GB) and the turbo LoRA (1.4 GB). Generating without downloading first works too, but shows no progress while it downloads.</div>` : ""}`;
    }
    async function loadDownload() {
      if (!document.body.contains(root)) { polling = false; return; }
      try { download = (await api("/api/models")).find((m) => m.id === st.model) || null; } catch { /* keep the last answer */ }
      renderDownload();
      polling = !!download && download.download.state === "downloading";
      if (polling) setTimeout(loadDownload, 2000);
    }

    // ---- generate
    async function submit() {
      const [width, height] = st.image ? [0, 0] : SHAPES[st.shape];
      await guard(() => api("/api/video/generate", { method: "POST", body: { ...st, width, height, gpu: gpu() } }), "Video queued");
    }
    root.addEventListener("click", async (e) => {
      const on = (sel) => e.target.closest(sel);
      if (on("#v-go")) return submit();
      if (on("[data-get]")) {
        await guard(() => api("/api/models/download", { method: "POST", body: { model: st.model } }), "Download started");
        if (!polling) loadDownload();
        return;
      }
      const rm = on("[data-remove]");
      if (rm) { st[rm.dataset.remove] = ""; save(); renderFrames(); return; }
      const card = on(".video-card");
      if (!card) return;
      const it = items[+card.dataset.i];
      if (on("[data-reuse]")) {
        Object.assign(st, { prompt: it.prompt, steps: it.steps, seed: it.seed, seconds: it.seconds, model: models[it.model] ? it.model : st.model,
          image: it.image_id || "", last_image: it.last_image_id || "", turbo: !!it.turbo });
        const shape = Object.keys(SHAPES).find((k) => SHAPES[k][0] === it.width && SHAPES[k][1] === it.height);
        if (shape) st.shape = shape;
        save();
        for (const k of ["prompt", "steps", "seed", "model"]) $(`#v-${k}`, root).value = st[k];
        $("#v-seconds", root).value = LENGTHS.reduce((a, b) => (Math.abs(b - st.seconds) < Math.abs(a - st.seconds) ? b : a));
        renderFrames(); renderTurbo();
        $(".video-head", root).scrollIntoView({ behavior: "smooth" });
        toast("Settings loaded (same seed)", "ok");
      }
      if (on("[data-del]")) {
        if (!(await confirmBox("Delete video?", "This permanently deletes the video."))) return;
        await guard(() => api(`/api/videos/${enc(it.file)}`, { method: "DELETE" }));
        refresh();
      }
    });
    $("#v-prompt", root).addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); submit(); } });

    // ---- gallery
    async function refresh() {
      items = (await api("/api/videos?limit=60")).items;
      const key = items.map((it) => it.file).join("|");
      const box = $("#v-gallery", root);
      if (box.dataset.key === key) return;  // don't rebuild (and stop) videos that are playing
      box.dataset.key = key;
      box.innerHTML = items.length ? items.map((it, i) => `
        <div class="card video-card" data-i="${i}">
          <video controls preload="none" playsinline poster="${fileUrl(it.thumb)}" src="${fileUrl(it.file)}" style="aspect-ratio:${it.width}/${it.height}"></video>
          <div class="card-body">
            <div class="video-prompt" title="${esc(it.prompt)}">${esc(it.prompt)}</div>
            <div class="row" style="justify-content:space-between;gap:8px;flex-wrap:nowrap">
              <span class="faint" style="font-size:12px">${[`${it.seconds} s`, `${it.width}×${it.height}`, `${it.steps} steps${it.turbo ? " (turbo)" : ""}`, `seed ${it.seed}`,
                ...(it.image_id ? ["first frame"] : []), ...(it.last_image_id ? ["last frame"] : []), ...(it.took ? [`took ${fmtDur(it.took)}`] : [])].join(" · ")}</span>
              <span class="row" style="gap:4px;flex-wrap:nowrap">
                <a class="btn btn-sm" href="${fileUrl(it.file)}?download=1" title="Download MP4">${I.download}</a>
                <button class="btn btn-sm" data-reuse title="Use these settings">${I.reuse}</button>
                <button class="btn btn-sm" data-del title="Delete">${I.trash}</button></span></div></div></div>`).join("")
        : `<div class="card"><div class="card-body faint" style="font-size:13px">No videos yet. The first one loads the model, which takes a few minutes.</div></div>`;
    }

    refresh();
    loadDownload();
    return { refresh };
  }

  return { mount };
})();
