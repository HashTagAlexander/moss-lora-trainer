/* Generate tab extras for Krea 2 / Kroma: inpainting (a painted mask), control images and regional prompts.
 *
 * The files they need are community add-ons (mosstrainer/addons.py); each shows a Download button until it is there.
 * State lives in the Generate tab's own `st`: mask, inpaint {mode, strength, keep}, control {type, image, photo,
 * strength} and regions [{x, y, w, h, prompt}] with boxes in 0-1 of the picture.
 */
"use strict";

const GenExtras = (() => {
  const TYPES = { depth: "Depth", pose: "Pose", canny: "Canny" };
  const HINTS = {
    depth: "Keeps the picture's 3D layout. Works together with a source image, a mask and regions.",
    pose: "Keeps a person's pose. Trained on Krea 2 Turbo; try strength 0.8–1.",
    canny: "Follows the picture's outlines, loosely (experimental). Try strength around 0.7.",
  };
  const COLORS = ["#8b6cff", "#4fd1c5", "#ff8a65", "#5aa9ff", "#f06292", "#9ccc65", "#ffb74d", "#e0e0e0"];
  const PAINT = [255, 61, 110];
  const inputUrl = (id) => `/api/generations/input/${enc(id)}`;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  function defaults(st) {
    st.mask = st.mask || "";
    st.inpaint = { mode: "lora", strength: 1, keep: true, ...(st.inpaint || {}) };
    st.control = { type: "", image: "", photo: "", strength: 1, ...(st.control || {}) };
    st.regions = Array.isArray(st.regions) ? st.regions : [];
  }

  /* What a generation request adds for these extras. */
  function body(st, krea) {
    const out = { mask: undefined, inpaint: undefined, control: undefined, regions: undefined };
    if (!krea) return out;
    if (st.image && st.mask) out.inpaint = { mask: st.mask, mode: st.inpaint.mode, strength: st.inpaint.strength, keep: st.inpaint.keep };
    if (st.control.type && st.control.image) out.control = { type: st.control.type, image: st.control.image, strength: st.control.strength };
    const regions = st.regions.filter((r) => r.prompt.trim());
    if (regions.length) out.regions = regions;
    return out;
  }

  async function upload(file, name) {
    const fd = new FormData();
    fd.append("file", file, name || file.name);
    return (await guard(() => api("/api/generate/input", { method: "POST", body: fd }))).id;
  }

  // ---------------------------------------------------------------- mask painter
  function maskEditor(imageId, maskId) {
    return new Promise((resolve) => {
      const bg = document.createElement("div");
      bg.className = "modal-bg";
      bg.innerHTML = `<div class="card modal mask-editor"><div class="card-head"><h2>Paint over what should be repainted</h2></div>
        <div class="card-body"><div class="mask-stage"><img alt=""><canvas></canvas></div>
          <div class="row mask-tools">
            <span class="chips"><span class="chip on" data-tool="paint">Paint</span><span class="chip" data-tool="erase">Erase</span></span>
            <label class="row" style="gap:6px;font-size:12px;color:var(--muted)">Brush <input type="range" min="4" max="220" value="48" data-brush></label>
            <span class="grow"></span>
            <button class="btn btn-sm" data-act="invert">Invert</button><button class="btn btn-sm" data-act="clear">Clear</button></div></div>
        <div class="card-body row" style="justify-content:flex-end;border-top:1px solid var(--border)">
          <button class="btn btn-ghost" data-act="cancel">Cancel</button><button class="btn btn-primary" data-act="ok">Use mask</button></div></div>`;
      document.body.appendChild(bg);
      const img = $("img", bg), cv = $("canvas", bg), g = cv.getContext("2d", { willReadFrequently: true });
      let tool = "paint", last = null;
      const close = (value) => { bg.remove(); resolve(value); };
      const setPixels = (fn) => {  // fn(isPainted) -> should it be painted
        const d = g.getImageData(0, 0, cv.width, cv.height), px = d.data;
        for (let i = 0; i < px.length; i += 4) {
          const on = fn(px[i + 3] > 127, px[i]);
          px[i] = PAINT[0]; px[i + 1] = PAINT[1]; px[i + 2] = PAINT[2]; px[i + 3] = on ? 255 : 0;
        }
        g.putImageData(d, 0, 0);
      };
      img.onload = () => {
        const scale = Math.min(1, 1536 / Math.max(img.naturalWidth, img.naturalHeight));
        cv.width = Math.round(img.naturalWidth * scale);
        cv.height = Math.round(img.naturalHeight * scale);
        if (!maskId) return;
        const old = new Image();
        old.onload = () => {  // a saved mask is white on black: show it as paint
          g.drawImage(old, 0, 0, cv.width, cv.height);
          setPixels((_, red) => red > 127);
        };
        old.src = inputUrl(maskId);
      };
      img.src = inputUrl(imageId);

      const at = (e) => {
        const r = cv.getBoundingClientRect(), k = cv.width / r.width;
        return { x: (e.clientX - r.left) * k, y: (e.clientY - r.top) * k, w: +$("[data-brush]", bg).value * k };
      };
      const stroke = (from, to) => {
        g.globalCompositeOperation = tool === "erase" ? "destination-out" : "source-over";
        g.strokeStyle = g.fillStyle = `rgb(${PAINT})`;
        g.lineWidth = to.w; g.lineCap = g.lineJoin = "round";
        g.beginPath(); g.moveTo(from.x, from.y); g.lineTo(to.x, to.y); g.stroke();
        g.beginPath(); g.arc(to.x, to.y, to.w / 2, 0, Math.PI * 2); g.fill();
        g.globalCompositeOperation = "source-over";
      };
      cv.addEventListener("pointerdown", (e) => { cv.setPointerCapture(e.pointerId); last = at(e); stroke(last, last); });
      cv.addEventListener("pointermove", (e) => { if (!last) return; const p = at(e); stroke(last, p); last = p; });
      const end = () => { last = null; };
      cv.addEventListener("pointerup", end);
      cv.addEventListener("pointercancel", end);
      bg.addEventListener("click", (e) => {
        const t = e.target.closest("[data-tool]"), act = e.target.closest("[data-act]")?.dataset.act;
        if (t) { tool = t.dataset.tool; $$("[data-tool]", bg).forEach((c) => c.classList.toggle("on", c === t)); }
        if (act === "clear") g.clearRect(0, 0, cv.width, cv.height);
        if (act === "invert") setPixels((on) => !on);
        if (act === "cancel" || e.target === bg) close(null);
        if (act === "ok") {
          const d = g.getImageData(0, 0, cv.width, cv.height), px = d.data;
          let any = false;
          for (let i = 0; i < px.length; i += 4) {
            const on = px[i + 3] > 127;
            any = any || on;
            px[i] = px[i + 1] = px[i + 2] = on ? 255 : 0; px[i + 3] = 255;
          }
          if (!any) return close("");  // nothing painted: no mask
          const out = document.createElement("canvas");
          out.width = cv.width; out.height = cv.height;
          out.getContext("2d").putImageData(d, 0, 0);
          out.toBlob((blob) => close(blob), "image/png");
        }
      });
    });
  }

  // ---------------------------------------------------------------- the three sections
  function mount({ st, save, krea, onSource }) {
    defaults(st);
    const root = $("#g-control");
    let addons = {}, polling = false, making = false, drag = null;
    const dims = {};  // picture id -> [width, height], for the region frame's shape

    function chip(id) {
      const a = addons[id];
      if (!a) return "";
      const pct = (a.message.match(/(\d+)%/) || [])[1];
      const state = a.state === "done" ? `<span class="addon-ok">${I.check} Ready</span>`
        : a.state === "downloading" ? `<span class="faint">Downloading${pct ? ` ${pct}%` : "…"}</span>`
        : `<button class="btn btn-sm" data-get="${id}">${I.download} Download</button>`;
      return `<div class="addon" title="${esc(a.about)} License: ${esc(a.license)}.">
          <a href="${esc(a.url)}" target="_blank" rel="noopener">${esc(a.label)}</a>
          <span class="faint">${esc(a.by)} · ${fmtBytes(a.size)}</span><span class="grow"></span>${state}</div>
        ${a.state === "error" ? `<div class="help" style="color:var(--red)">${esc(a.message)}</div>` : ""}`;
    }
    async function loadAddons() {
      if (!document.body.contains(root)) { polling = false; return; }  // left the page
      try { addons = Object.fromEntries((await api("/api/addons")).map((a) => [a.id, a])); } catch { /* keep the last list */ }
      renderInpaint(); renderControl();
      polling = Object.values(addons).some((a) => a.state === "downloading");
      if (polling) setTimeout(loadAddons, 2000);
    }
    async function getAddon(id) {
      await guard(() => api("/api/addons/download", { method: "POST", body: { id } }));
      if (!polling) loadAddons();
    }

    function renderInpaint() {
      const el = $("#g-inpaint"), show = krea() && !!st.image;
      el.hidden = !show;
      if (!show) return;
      const lora = st.inpaint.mode === "lora";
      el.innerHTML = `<div class="row" style="gap:6px">
          <button class="btn btn-sm" data-act="mask">${I.edit} ${st.mask ? "Edit mask" : "Paint a mask to inpaint"}</button>
          ${st.mask ? `<button class="btn btn-sm btn-ghost" data-act="mask-clear">${I.x} Remove mask</button>` : ""}</div>
        ${st.mask ? `<div class="row" style="gap:8px">
            <select data-k="mode" style="width:auto"><option value="lora" ${lora ? "selected" : ""}>With the inpaint LoRA</option><option value="blend" ${lora ? "" : "selected"}>No LoRA</option></select>
            ${lora ? `<input type="number" data-k="strength" step="0.05" min="0" max="2" value="${st.inpaint.strength}" title="Inpaint LoRA strength" style="width:72px">` : ""}
            <label class="check" style="font-size:12px"><input type="checkbox" data-k="keep" ${st.inpaint.keep ? "checked" : ""}> Keep the rest as it is</label></div>
          ${lora ? chip("inpaint") + `<div class="help">The prompt says what goes in the painted part.</div>`
            : `<div class="help">The model repaints the painted part by itself and the rest is put back at every step. Needs no download and works with your own LoRAs, but edges blend less well.</div>`}` : ""}`;
    }

    function renderControl() {
      root.hidden = !krea();
      if (root.hidden) return;
      const c = st.control, t = c.type, made = c.photo && c.photo !== c.image;
      root.innerHTML = `<summary>Control image${t ? ` · ${TYPES[t]}` : ""}</summary><div class="extras-body">
        <div class="chips"><span class="chip ${t ? "" : "on"}" data-type="">Off</span>${Object.entries(TYPES).map(([k, v]) => `<span class="chip ${t === k ? "on" : ""}" data-type="${k}">${v}</span>`).join("")}</div>
        ${t ? `<div class="help">${HINTS[t]}</div>
          ${c.image ? `<div class="src-img"><img src="${inputUrl(c.image)}" alt=""><button class="btn btn-sm" data-act="control-remove" title="Remove">${I.x}</button></div>
              <div class="row" style="gap:6px">
                <button class="btn btn-sm" data-act="control-make" ${making ? "disabled" : ""}>${I.spark} ${making ? "Making the map…" : `Make ${t === "canny" ? "an edge" : `a ${t}`} map from ${made ? "the photo again" : "this picture"}`}</button>
                ${made ? `<button class="btn btn-sm btn-ghost" data-act="control-asis">Use the photo itself</button>` : ""}
                <label class="btn btn-sm btn-ghost" style="cursor:pointer">${I.upload} Replace<input type="file" data-file="control" accept="image/*" hidden></label></div>
              <div class="help">${made ? `This ${t} map is what the model follows.` : `Used as it is: fine if it's already ${t === "canny" ? "an edge" : `a ${t}`} map. A photo needs the button above.`}</div>`
            : `<label class="btn btn-sm" style="cursor:pointer;justify-self:start">${I.upload} Upload a photo or a ready-made map<input type="file" data-file="control" accept="image/*" hidden></label>`}
          <label class="field" style="max-width:120px">Strength<input type="number" data-k="strength" step="0.05" min="0" max="2" value="${c.strength}"></label>
          ${chip(t)}${t === "canny" ? "" : chip(`${t}-estimator`)}
          ${t !== "depth" && st.image ? `<div class="help" style="color:var(--red)">${TYPES[t]} control uses the model's one reference image, so it can't run together with a source image. Remove the source image, or use Depth.</div>` : ""}` : ""}</div>`;
    }

    function basis() {  // the picture the result takes its shape from, if any
      return st.image || (st.control.type && st.control.image) || "";
    }
    function frameSize() {
      const id = basis();
      if (!id) return [st.width, st.height];
      if (!dims[id]) {
        dims[id] = [st.width, st.height];
        const im = new Image();
        im.onload = () => { dims[id] = [im.naturalWidth, im.naturalHeight]; renderRegions(); };
        im.src = inputUrl(id);
      }
      return dims[id];
    }
    function renderRegions() {
      const el = $("#g-regions");
      el.hidden = !krea();
      if (el.hidden) return;
      const [fw, fh] = frameSize(), id = basis();
      el.innerHTML = `<summary>Regions${st.regions.length ? ` · ${st.regions.length}` : ""}</summary><div class="extras-body">
        <div class="help">Drag on the frame to give an area its own prompt (up to 8). The main prompt still describes the whole picture, so say there where things are too. Where boxes overlap, the higher number wins.</div>
        <div class="region-pad" style="aspect-ratio:${fw}/${fh};${id ? `background-image:linear-gradient(rgba(10,12,16,.55),rgba(10,12,16,.55)),url('${inputUrl(id)}')` : ""}">
          ${st.regions.map((r, i) => `<div class="region-box" data-i="${i}" style="left:${r.x * 100}%;top:${r.y * 100}%;width:${r.w * 100}%;height:${r.h * 100}%;--c:${COLORS[i % 8]}"><span>${i + 1}</span></div>`).join("")}</div>
        ${st.regions.map((r, i) => `<div class="region-row" data-i="${i}"><span class="region-dot" style="background:${COLORS[i % 8]}">${i + 1}</span>
          <textarea rows="2" placeholder="What is in this area">${esc(r.prompt)}</textarea>
          <button class="btn btn-sm btn-ghost" data-act="region-del" title="Remove">${I.x}</button></div>`).join("")}</div>`;
    }

    function refresh() { renderInpaint(); renderControl(); renderRegions(); }

    // ---- inpainting
    $("#g-inpaint").addEventListener("click", async (e) => {
      const act = e.target.closest("[data-act]")?.dataset.act, get = e.target.closest("[data-get]");
      if (get) return getAddon(get.dataset.get);
      if (act === "mask-clear") st.mask = "";
      else if (act === "mask") {
        const blob = await maskEditor(st.image, st.mask);
        if (blob == null) return;
        st.mask = blob ? await upload(blob, `${st.image.replace(/\.[^.]+$/, "")}-mask.png`) : "";
      } else return;
      save(); onSource(); renderInpaint();
    });
    $("#g-inpaint").addEventListener("change", (e) => {
      const k = e.target.dataset.k;
      if (!k) return;
      st.inpaint[k] = k === "keep" ? e.target.checked : k === "strength" ? clamp(+e.target.value || 0, 0, 2) : e.target.value;
      save();
      if (k === "mode") renderInpaint();
    });

    // ---- control image
    root.addEventListener("click", async (e) => {
      const type = e.target.closest("[data-type]"), act = e.target.closest("[data-act]")?.dataset.act, get = e.target.closest("[data-get]");
      if (get) return getAddon(get.dataset.get);
      const c = st.control;
      if (type) c.type = type.dataset.type;
      else if (act === "control-remove") c.image = c.photo = "";
      else if (act === "control-asis") c.image = c.photo;
      else if (act === "control-make") {
        making = true; renderControl();
        try {
          const photo = c.photo || c.image;
          const r = await guard(() => api("/api/generate/preprocess", { method: "POST", body: { type: c.type, image: photo } }));
          c.photo = photo; c.image = r.id;
        } catch { /* the toast said why */ } finally { making = false; }
      } else return;
      save(); renderControl(); renderRegions();
    });
    root.addEventListener("change", async (e) => {
      if (e.target.dataset.file === "control" && e.target.files[0]) {
        st.control.image = st.control.photo = await upload(e.target.files[0]);
        save(); renderControl(); renderRegions();
      } else if (e.target.dataset.k === "strength") {
        st.control.strength = clamp(+e.target.value || 0, 0, 2); save();
      }
    });

    // ---- regions: drag on the frame to draw a box, drag a box to move it
    const regions = $("#g-regions");
    const place = (el, r) => Object.assign(el.style, { left: `${r.x * 100}%`, top: `${r.y * 100}%`, width: `${r.w * 100}%`, height: `${r.h * 100}%` });
    regions.addEventListener("pointerdown", (e) => {
      const pad = e.target.closest(".region-pad");
      if (!pad) return;
      const rect = pad.getBoundingClientRect(), box = e.target.closest(".region-box");
      const px = clamp((e.clientX - rect.left) / rect.width, 0, 1), py = clamp((e.clientY - rect.top) / rect.height, 0, 1);
      if (box) drag = { rect, i: +box.dataset.i, el: box, dx: px - st.regions[+box.dataset.i].x, dy: py - st.regions[+box.dataset.i].y };
      else {
        if (st.regions.length >= 8) return toast("Eight regions at most", "error");
        const el = document.createElement("div");
        el.className = "region-box";
        el.style.setProperty("--c", COLORS[st.regions.length % 8]);
        pad.appendChild(el);
        drag = { rect, el, x0: px, y0: py, r: { x: px, y: py, w: 0, h: 0, prompt: "" } };
        place(el, drag.r);
      }
      pad.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    regions.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const px = clamp((e.clientX - drag.rect.left) / drag.rect.width, 0, 1), py = clamp((e.clientY - drag.rect.top) / drag.rect.height, 0, 1);
      if (drag.r) Object.assign(drag.r, { x: Math.min(px, drag.x0), y: Math.min(py, drag.y0), w: Math.abs(px - drag.x0), h: Math.abs(py - drag.y0) });
      else {
        const r = st.regions[drag.i];
        r.x = clamp(px - drag.dx, 0, 1 - r.w); r.y = clamp(py - drag.dy, 0, 1 - r.h);
      }
      place(drag.el, drag.r || st.regions[drag.i]);
    });
    const drop = () => {
      if (!drag) return;
      const added = drag.r && drag.r.w > 0.04 && drag.r.h > 0.04;
      if (added) st.regions.push(drag.r);
      drag = null;
      save(); renderRegions();
      if (added) $(`.region-row[data-i="${st.regions.length - 1}"] textarea`, regions)?.focus();
    };
    regions.addEventListener("pointerup", drop);
    regions.addEventListener("pointercancel", drop);
    regions.addEventListener("input", (e) => {
      const row = e.target.closest(".region-row");
      if (row && e.target.matches("textarea")) { st.regions[+row.dataset.i].prompt = e.target.value; save(); }
    });
    regions.addEventListener("click", (e) => {
      const row = e.target.closest(".region-row");
      if (row && e.target.closest("[data-act=region-del]")) { st.regions.splice(+row.dataset.i, 1); save(); renderRegions(); }
    });

    refresh();
    loadAddons();
    return { refresh };
  }

  return { defaults, body, mount };
})();
