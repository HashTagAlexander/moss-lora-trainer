/* VGL prompt editor (Fibo 2.0) with a layout canvas.
 *
 * Fibo 2.0 reads where things are as words, not coordinates: "location" ("top-left", "center",
 * "bottom-right foreground") and "relative_size" ("small", "medium", "large within frame"). So the
 * boxes here snap to the 3x3 grid those words describe and are only a picture of the prompt.
 */
"use strict";

const VGL = (() => {
  const COLS = ["left", "center", "right"], ROWS = ["top", "center", "bottom"];
  const DEPTHS = ["", "foreground", "midground", "background"];
  const SIZES = [["small", 0.24], ["medium", 0.4], ["medium-to-large", 0.55], ["large within frame", 0.72]];
  const COLORS = ["#8b6cff", "#4fd1c5", "#ff8a65", "#5aa9ff", "#f06292", "#9ccc65", "#ffb74d"];
  // Bria's VGL key order (bria_vgl.model.VGL), the order Fibo 2.0 saw captions in
  const ORDER = ["short_description", "objects", "background_setting", "lighting", "aesthetics",
    "photographic_characteristics", "style_medium", "text_render", "context", "artistic_style"];
  const SUB = {
    objects: ["description", "location", "relationship", "relative_size", "shape_and_color", "texture", "appearance_details",
      "number_of_objects", "pose", "expression", "clothing", "action", "gender", "skin_tone_and_texture", "orientation"],
    lighting: ["conditions", "direction", "shadows"],
    aesthetics: ["composition", "color_scheme", "mood_atmosphere"],
    photographic_characteristics: ["depth_of_field", "focus", "camera_angle", "lens_focal_length"],
    text_render: ["text", "location", "size", "color", "font", "appearance_details"],
  };
  const PERSON = ["pose", "expression", "clothing", "action", "gender", "skin_tone_and_texture"];

  // ---- location / size words <-> boxes
  function parseLocation(loc) {
    const s = String(loc || "").toLowerCase();
    return {
      col: /left/.test(s) ? 0 : /right/.test(s) ? 2 : 1,
      row: /top|upper/.test(s) ? 0 : /bottom|lower/.test(s) ? 2 : 1,
      depth: (s.match(/foreground|midground|background/) || [""])[0],
    };
  }
  function locationText(row, col, depth) {
    const base = row === 1 && col === 1 ? "center" : row === 1 ? `center-${COLS[col]}`
      : col === 1 ? `${ROWS[row]}-center` : `${ROWS[row]}-${COLS[col]}`;
    return depth ? `${base} ${depth}` : base;
  }
  function sizeFrac(size) {
    const s = String(size || "").toLowerCase();
    if (/within frame|fills|full/.test(s)) return 0.72;
    if (/medium.to.large/.test(s)) return 0.55;
    if (/large/.test(s)) return 0.62;
    if (/medium/.test(s)) return 0.4;
    if (/small|tiny/.test(s)) return 0.24;
    return 0.34;
  }
  const sizeText = (f) => SIZES.reduce((a, b) => (Math.abs(b[1] - f) < Math.abs(a[1] - f) ? b : a))[0];
  function boxOf(entry, kind) {
    const { row, col } = parseLocation(entry.location);
    const f = sizeFrac(kind === "text" ? entry.size : entry.relative_size);
    const clamp = (c) => Math.min(Math.max(c - f / 2, 0), 1 - f);
    return { x: clamp((col + 0.5) / 3), y: clamp((row + 0.5) / 3), w: f, h: f };
  }
  function parse(prompt) {
    try {
      const v = JSON.parse(prompt);
      return v && typeof v === "object" && !Array.isArray(v) ? v : null;
    } catch { return null; }
  }
  /** Boxes for a prompt's objects and texts, in 0..1 frame units (empty for a plain-text prompt). */
  function layout(prompt) {
    const v = typeof prompt === "string" ? parse(prompt) : prompt;
    if (!v) return [];
    const objs = (Array.isArray(v.objects) ? v.objects : []).filter((o) => o && typeof o === "object")
      .map((o, i) => ({ ...boxOf(o, "object"), kind: "object", i, color: COLORS[i % COLORS.length],
        label: String(o.description || `object ${i + 1}`) }));
    const texts = (Array.isArray(v.text_render) ? v.text_render : []).filter((t) => t && typeof t === "object")
      .map((t, i) => ({ ...boxOf(t, "text"), kind: "text", i, color: "#f5c451", label: `“${t.text || ""}”` }));
    return objs.concat(texts);
  }
  function boxesHtml(boxes, { selected = null, handles = false } = {}) {
    return boxes.map((b) => `<div class="vgl-box ${b.kind}${selected && selected.kind === b.kind && selected.i === b.i ? " sel" : ""}"
      data-kind="${b.kind}" data-i="${b.i}" style="left:${b.x * 100}%;top:${b.y * 100}%;width:${b.w * 100}%;height:${b.h * 100}%;--c:${b.color}">
      <span class="vgl-tag">${esc(b.label.length > 38 ? b.label.slice(0, 36) + "…" : b.label)}</span>${handles ? '<span class="vgl-resize"></span>' : ""}</div>`).join("");
  }

  // ---- cleaning / serialising, as captioner.tidy does it
  const isEmpty = (v) => v == null || v === "" || (Array.isArray(v) && !v.length) || (typeof v === "object" && !Array.isArray(v) && !Object.keys(v).length);
  function tidy(v) {
    const ordered = (d, order) => {
      const out = {};
      for (const k of order.concat(Object.keys(d).filter((k) => !order.includes(k)))) {
        if (!(k in d)) continue;
        const x = typeof d[k] === "string" ? d[k].trim() : d[k];
        if (!isEmpty(x)) out[k] = x;
      }
      return out;
    };
    const out = {};
    for (const k of ORDER.concat(Object.keys(v).filter((k) => !ORDER.includes(k)))) {
      if (!(k in v)) continue;
      let x = v[k];
      if (SUB[k] && Array.isArray(x)) x = x.map((e) => (e && typeof e === "object" ? ordered(e, SUB[k]) : e)).filter((e) => !isEmpty(e));
      else if (SUB[k] && x && typeof x === "object") x = ordered(x, SUB[k]);
      else if (typeof x === "string") x = x.trim();
      if (k === "text_render" && Array.isArray(x) && !x.length) out[k] = x;  // [] says "no text in the image"
      else if (!isEmpty(x)) out[k] = x;
    }
    return out;
  }

  // ---- editor
  const getPath = (o, p) => p.split(".").reduce((a, k) => (a == null ? a : a[k]), o);
  function setPath(o, p, v) {
    const keys = p.split(".");
    let cur = o;
    keys.slice(0, -1).forEach((k, i) => {
      if (cur[k] == null || typeof cur[k] !== "object") cur[k] = /^\d+$/.test(keys[i + 1]) ? [] : {};
      cur = cur[k];
    });
    cur[keys[keys.length - 1]] = v;
  }
  const FIELD_HELP = {
    short_description: "The whole image in one to three sentences.",
    style_medium: "e.g. photograph, oil painting, 3D render, 2D animation still",
    artistic_style: "3 words at most, e.g. flat cel-shaded, clean lineart",
    context: "What kind of image this is, e.g. a frame from an animated TV series",
    background_setting: "The environment, and background things not listed as objects",
  };
  const LABEL = (k) => k.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());

  /**
   * Open the editor on a prompt (VGL JSON or plain text). Resolves with the new prompt (pretty JSON) or null.
   * width/height set the canvas' aspect ratio.
   */
  function open({ prompt = "", width = 1024, height = 1024, title = "VGL prompt editor" } = {}) {
    return new Promise((resolve) => {
      let v = parse(prompt) || { short_description: prompt.trim(), objects: [], text_render: [] };
      v = JSON.parse(JSON.stringify(v));
      if (!Array.isArray(v.objects)) v.objects = [];
      if (!Array.isArray(v.text_render)) v.text_render = [];
      let selected = null, tab = "form";

      const bg = document.createElement("div");
      bg.className = "modal-bg";
      bg.innerHTML = `<div class="card vgl-editor" role="dialog" aria-label="${esc(title)}">
        <div class="card-head"><h2>${esc(title)}</h2>
          <div class="row"><div class="tabs vgl-tabs"><div class="tab active" data-tab="form">Editor</div><div class="tab" data-tab="json">JSON</div></div>
          <button class="btn btn-ghost btn-sm" data-act="cancel" title="Close">${I.x}</button></div></div>
        <div class="vgl-body">
          <div class="vgl-left">
            <div class="vgl-canvas" style="aspect-ratio:${width}/${height};width:min(100%, calc(62vh * ${width / height}))"><div class="vgl-grid"></div><div class="vgl-boxes"></div></div>
            <div class="help">Drag a box to move it, drag its corner to resize. Fibo 2.0 reads position and size as words
              (<span class="mono">location</span>, <span class="mono">relative_size</span>), so boxes snap to a 3×3 grid: a rough layout, not exact coordinates.</div>
            <div class="row" style="gap:6px"><button class="btn btn-sm" data-act="add-object">${I.plus} Object</button>
              <button class="btn btn-sm" data-act="add-person">${I.plus} Person</button><button class="btn btn-sm" data-act="add-text">${I.plus} Text</button></div>
          </div>
          <div class="vgl-right"><div class="vgl-form"></div><textarea class="vgl-json mono" hidden spellcheck="false"></textarea></div>
        </div>
        <div class="card-body row" style="justify-content:space-between;border-top:1px solid var(--border)">
          <button class="btn btn-ghost" data-act="copy" title="Copy as one line, e.g. for sample prompts">${I.copy} Copy as one line</button>
          <div class="row"><button class="btn btn-ghost" data-act="cancel">Cancel</button><button class="btn btn-primary" data-act="apply">Use this prompt</button></div></div></div>`;
      document.body.appendChild(bg);
      const $e = (s) => bg.querySelector(s);
      const canvas = $e(".vgl-canvas"), boxesEl = $e(".vgl-boxes"), form = $e(".vgl-form"), json = $e(".vgl-json");

      const field = (path, label, { area = false, help = "", placeholder = "" } = {}) => {
        const val = getPath(v, path) ?? "";
        const input = area
          ? `<textarea data-k="${path}" rows="2" placeholder="${esc(placeholder)}">${esc(val)}</textarea>`
          : `<input type="text" data-k="${path}" value="${esc(val)}" placeholder="${esc(placeholder)}">`;
        return `<label class="field">${esc(label)}${input}${help ? `<span class="help">${esc(help)}</span>` : ""}</label>`;
      };
      const locField = (path, entry) => {
        const { row, col, depth } = parseLocation(entry.location);
        return `<div class="vgl-pair">${field(path + ".location", "Location", { placeholder: "e.g. top-left foreground" })}
          <label class="field">Depth<select data-depth="${path}">${DEPTHS.map((d) => `<option value="${d}" ${d === depth ? "selected" : ""}>${d || "—"}</option>`).join("")}</select></label></div>
          <span class="help vgl-loc-hint" data-hint="${path}">Grid cell: ${ROWS[row]} row, ${COLS[col]} column</span>`;
      };
      function renderForm() {
        const objs = v.objects.map((o, i) => {
          const p = `objects.${i}`, person = PERSON.some((k) => o[k] != null && o[k] !== "") || o._person;
          return `<div class="vgl-entry ${selected && selected.kind === "object" && selected.i === i ? "sel" : ""}" data-kind="object" data-i="${i}" style="--c:${COLORS[i % COLORS.length]}">
            <div class="row" style="justify-content:space-between"><b>${person ? "Person" : "Object"} ${i + 1}</b>
              <button class="btn btn-sm btn-ghost btn-danger" data-del="object" data-i="${i}" title="Remove">${I.trash}</button></div>
            ${field(p + ".description", "Description", { area: true, placeholder: "what it is and what it looks like" })}
            ${locField(p, o)}
            ${field(p + ".relative_size", "Relative size", { placeholder: "small, medium, large within frame" })}
            ${person ? `<div class="vgl-pair">${field(p + ".pose", "Pose")}${field(p + ".expression", "Expression")}</div>
              ${field(p + ".clothing", "Clothing", { area: true })}<div class="vgl-pair">${field(p + ".action", "Action")}${field(p + ".gender", "Gender")}</div>` : ""}
            <details><summary>More details</summary><div class="vgl-sub">
              ${field(p + ".relationship", "Relationship", { placeholder: "how it relates to the other objects" })}
              ${field(p + ".shape_and_color", "Shape and color")}${field(p + ".texture", "Texture")}
              ${field(p + ".appearance_details", "Appearance details")}${field(p + ".orientation", "Orientation", { placeholder: "e.g. facing left" })}
              ${person ? field(p + ".skin_tone_and_texture", "Skin tone and texture") : field(p + ".number_of_objects", "Number of objects", { placeholder: "only for a group of similar things" })}
            </div></details></div>`;
        }).join("");
        const texts = v.text_render.map((t, i) => {
          const p = `text_render.${i}`;
          return `<div class="vgl-entry text ${selected && selected.kind === "text" && selected.i === i ? "sel" : ""}" data-kind="text" data-i="${i}" style="--c:#f5c451">
            <div class="row" style="justify-content:space-between"><b>Text ${i + 1}</b>
              <button class="btn btn-sm btn-ghost btn-danger" data-del="text" data-i="${i}" title="Remove">${I.trash}</button></div>
            ${field(p + ".text", "Text, exactly as written")}${locField(p, t)}
            <div class="vgl-pair">${field(p + ".size", "Size")}${field(p + ".color", "Color")}</div>
            ${field(p + ".font", "Font", { placeholder: "e.g. bold sans-serif, handwritten script" })}${field(p + ".appearance_details", "Appearance details")}</div>`;
        }).join("");
        form.innerHTML = `
          ${field("short_description", "Short description", { area: true, help: FIELD_HELP.short_description })}
          <div class="vgl-pair">${field("style_medium", "Style / medium", { help: FIELD_HELP.style_medium })}${field("artistic_style", "Artistic style", { help: FIELD_HELP.artistic_style })}</div>
          ${field("context", "Context", { help: FIELD_HELP.context })}
          ${field("background_setting", "Background", { area: true, help: FIELD_HELP.background_setting })}
          <h3 class="vgl-h">Objects</h3>${objs || '<div class="faint" style="font-size:12px">None yet: add one on the left.</div>'}
          <h3 class="vgl-h">Text in the image</h3>${texts || '<div class="faint" style="font-size:12px">No text (Fibo renders none).</div>'}
          <details class="vgl-group"><summary>Lighting</summary><div class="vgl-sub">${SUB.lighting.map((k) => field(`lighting.${k}`, LABEL(k))).join("")}</div></details>
          <details class="vgl-group"><summary>Aesthetics</summary><div class="vgl-sub">${SUB.aesthetics.map((k) => field(`aesthetics.${k}`, LABEL(k))).join("")}</div></details>
          <details class="vgl-group"><summary>Camera</summary><div class="vgl-sub">${SUB.photographic_characteristics.map((k) => field(`photographic_characteristics.${k}`, LABEL(k))).join("")}</div></details>`;
      }
      const drawBoxes = () => { boxesEl.innerHTML = boxesHtml(layout(v), { selected, handles: true }); };
      const output = () => {
        const clean = JSON.parse(JSON.stringify(v));
        (clean.objects || []).forEach((o) => {
          delete o._person;
          if (/^\s*\d+\s*$/.test(String(o.number_of_objects ?? ""))) o.number_of_objects = Number(o.number_of_objects);
        });
        return tidy(clean);
      };
      const select = (sel, scroll) => {
        selected = sel;
        drawBoxes();
        form.querySelectorAll(".vgl-entry").forEach((e) => e.classList.toggle("sel", !!sel && e.dataset.kind === sel.kind && +e.dataset.i === sel.i));
        if (scroll && sel) form.querySelector(`.vgl-entry[data-kind="${sel.kind}"][data-i="${sel.i}"]`)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
      };
      renderForm(); drawBoxes();

      // form edits
      form.addEventListener("input", (e) => {
        const k = e.target.dataset.k;
        if (!k) return;
        setPath(v, k, e.target.value);
        if (/\.(location|relative_size|size|description|text)$/.test(k)) {
          drawBoxes();
          const p = k.replace(/\.location$/, "");
          const hint = form.querySelector(`[data-hint="${p}"]`);
          if (hint && k.endsWith(".location")) { const { row, col } = parseLocation(e.target.value); hint.textContent = `Grid cell: ${ROWS[row]} row, ${COLS[col]} column`; }
        }
      });
      form.addEventListener("change", (e) => {
        const p = e.target.dataset.depth;
        if (p == null) return;
        const entry = getPath(v, p), { row, col } = parseLocation(entry.location);
        entry.location = locationText(row, col, e.target.value);
        const inp = form.querySelector(`[data-k="${p}.location"]`);
        if (inp) inp.value = entry.location;
      });
      form.addEventListener("focusin", (e) => {
        const card = e.target.closest(".vgl-entry");
        if (card) select({ kind: card.dataset.kind, i: +card.dataset.i }, false);
      });
      form.addEventListener("click", (e) => {
        const d = e.target.closest("[data-del]");
        if (!d) return;
        (d.dataset.del === "text" ? v.text_render : v.objects).splice(+d.dataset.i, 1);
        selected = null; renderForm(); drawBoxes();
      });

      // canvas: drag to move, corner to resize; release snaps to the words
      canvas.addEventListener("pointerdown", (e) => {
        const el = e.target.closest(".vgl-box");
        if (!el) return select(null);
        e.preventDefault();
        const kind = el.dataset.kind, i = +el.dataset.i, entry = (kind === "text" ? v.text_render : v.objects)[i];
        select({ kind, i }, true);
        const box = boxesEl.querySelector(`.vgl-box[data-kind="${kind}"][data-i="${i}"]`);
        const rect = canvas.getBoundingClientRect(), resizing = !!e.target.closest(".vgl-resize");
        const start = { x: e.clientX, y: e.clientY, b: boxOf(entry, kind) };
        let cur = { ...start.b }, moved = false;
        const move = (ev) => {
          moved = moved || Math.abs(ev.clientX - start.x) + Math.abs(ev.clientY - start.y) > 3;
          const dx = (ev.clientX - start.x) / rect.width, dy = (ev.clientY - start.y) / rect.height;
          if (resizing) {
            const f = Math.min(Math.max(start.b.w + Math.max(dx, dy), 0.12), 0.9);
            cur = { x: Math.min(start.b.x, 1 - f), y: Math.min(start.b.y, 1 - f), w: f, h: f };
          } else {
            cur = { ...start.b, x: Math.min(Math.max(start.b.x + dx, 0), 1 - start.b.w), y: Math.min(Math.max(start.b.y + dy, 0), 1 - start.b.h) };
          }
          Object.assign(box.style, { left: `${cur.x * 100}%`, top: `${cur.y * 100}%`, width: `${cur.w * 100}%`, height: `${cur.h * 100}%` });
        };
        const up = () => {
          window.removeEventListener("pointermove", move);
          window.removeEventListener("pointerup", up);
          if (!moved) return;
          const sizeKey = kind === "text" ? "size" : "relative_size";
          // the inverse of boxOf: where the box sits within the room it has to move in
          const cell = (pos, size) => (1 - size < 0.02 ? 1 : Math.round((pos / (1 - size)) * 2));
          if (resizing) entry[sizeKey] = sizeText(cur.w);
          else entry.location = locationText(cell(cur.y, cur.h), cell(cur.x, cur.w), parseLocation(entry.location).depth);
          renderForm(); select({ kind, i }, true);
        };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
      });

      // tabs, buttons
      const syncJson = () => {
        try { v = JSON.parse(json.value); } catch (err) { toast("Invalid JSON: " + err.message, "error"); return false; }
        if (!v || typeof v !== "object" || Array.isArray(v)) { toast("The prompt has to be a JSON object", "error"); return false; }
        if (!Array.isArray(v.objects)) v.objects = [];
        if (!Array.isArray(v.text_render)) v.text_render = [];
        return true;
      };
      const setTab = (t) => {
        if (t === tab) return;
        if (t === "form" && !syncJson()) return;
        if (t === "json") json.value = JSON.stringify(output(), null, 2);
        tab = t;
        bg.querySelectorAll(".vgl-tabs .tab").forEach((x) => x.classList.toggle("active", x.dataset.tab === t));
        form.hidden = t !== "form"; json.hidden = t !== "json";
        if (t === "form") { selected = null; renderForm(); drawBoxes(); }
      };
      const done = (val) => { bg.remove(); document.removeEventListener("keydown", key); resolve(val); };
      const key = (e) => { if (e.key === "Escape") done(null); };
      document.addEventListener("keydown", key);
      bg.addEventListener("click", async (e) => {
        if (e.target === bg) return done(null);
        const t = e.target.closest("[data-tab]");
        if (t) return setTab(t.dataset.tab);
        const act = e.target.closest("[data-act]")?.dataset.act;
        if (!act) return;
        if (act === "cancel") return done(null);
        if (tab === "json" && (act === "apply" || act === "copy") && !syncJson()) return;
        if (act === "apply") return done(JSON.stringify(output(), null, 2));
        if (act === "copy") {
          try { await navigator.clipboard.writeText(JSON.stringify(output())); toast("Copied as one line", "ok"); }
          catch { toast("Couldn't copy: your browser blocked clipboard access", "error"); }
          return;
        }
        if (tab === "json") setTab("form");
        if (act === "add-object" || act === "add-person") {
          v.objects.push({ description: "", location: "center", relative_size: act === "add-person" ? "medium-to-large" : "medium",
            ...(act === "add-person" ? { _person: true, pose: "", expression: "" } : {}) });
          selected = { kind: "object", i: v.objects.length - 1 };
        }
        if (act === "add-text") {
          v.text_render.push({ text: "", location: "top-center", size: "medium", color: "", font: "" });
          selected = { kind: "text", i: v.text_render.length - 1 };
        }
        renderForm(); select(selected, true);
        form.querySelector(`.vgl-entry[data-kind="${selected.kind}"][data-i="${selected.i}"] textarea, .vgl-entry[data-kind="${selected.kind}"][data-i="${selected.i}"] input`)?.focus();
      });
    });
  }

  return { open, layout, boxesHtml, parse };
})();
