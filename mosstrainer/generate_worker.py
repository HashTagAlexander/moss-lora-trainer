"""Long-lived image generation worker used by the web UI's Generate tab.

Protocol: one JSON request per line on stdin; one JSON event per line on stdout.
Library output is redirected to stderr so it can't corrupt the event stream.

Requests:
  {"cmd": "generate", "id", "model", "prompt", "negative", "width", "height", "steps", "cfg",
   "shift", "seed", "count", "skip_layer", "loras": [{"path", "strength", "name"}], "low_vram", "out_dir",
   "image" (optional source image path to edit; Fibo 2.0, Krea 2), "image_id",
   Krea 2 only:
   "inpaint": {"mask": path (white = repaint), "lora": path or None (None: no LoRA, the rest of the picture is
               put back at every step), "strength", "ref": {...}, "keep": paste the result into the source},
   "control": {"type": "depth" | "pose" | "canny", "image": path of the map, "lora": path, "strength", "ref"},
   "regions": [{"x", "y", "w", "h" (0-1 of the picture), "prompt"}]}
  {"cmd": "offload"}   move the text encoder to CPU (e.g. because training started)
SIGUSR1 cancels the generation in progress. The worker exits after `--idle` seconds without requests.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import queue
import random
import signal
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

_proto = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

import torch  # noqa: E402

from .backends import get_backend  # noqa: E402
from .lora import LoRAStack  # noqa: E402


def emit(**event):
    _proto.write(json.dumps(event) + "\n")
    _proto.flush()


class Cancelled(Exception):
    pass


class _FakeTE:
    """Deterministic random embeddings, used only by tests (MOSS_TEST_FAKE_TE=<dim>)."""

    def __init__(self, dim):
        self.dim, self.device, self.model = dim, torch.device("cpu"), torch.nn.Identity()

    def encode(self, prompt):
        g = torch.Generator().manual_seed(abs(hash(prompt)) % 2**31)
        return torch.randn(4 + len(prompt) % 9, self.dim, generator=g).to(torch.bfloat16)


class Worker:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = None
        self.backend = None
        self.te = None
        self.transformer = None
        self.vae = None
        self.loras = None
        self.embeds: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.cancel = False

    # ---- model lifecycle
    def load(self, model_id: str, low_vram: bool):
        if self.model_id == model_id:
            return
        self.free()
        emit(event="status", state="loading", message=f"Loading {model_id.split('/')[-1]} (first time downloads it)…")
        backend = get_backend(model_id)
        backend.prefetch()
        te_device = torch.device("cpu") if low_vram else self.device
        fake_dim = os.environ.get("MOSS_TEST_FAKE_TE")  # tests only: skip the 9B encoder
        self.te = _FakeTE(int(fake_dim)) if fake_dim else backend.load_text_encoder(te_device)
        self.transformer = backend.load_transformer(self.device)
        self.vae = backend.load_vae(self.device)
        self.loras = LoRAStack(self.transformer, name_map=backend.lora_name_map)
        self.backend = backend
        self.model_id = model_id
        emit(event="status", state="ready", model=model_id, message="Model loaded")

    def free(self):
        self.te = self.transformer = self.vae = self.loras = self.backend = None
        self.model_id = None
        self.embeds.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def offload_te(self):
        if self.te is not None and self.te.device != torch.device("cpu"):
            self.te.model.to("cpu")
            self.te.device = torch.device("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def encode(self, prompt: str, low_vram: bool, images=None, key=None) -> torch.Tensor:
        key = key or prompt
        if key in self.embeds:
            self.embeds.move_to_end(key)
            return self.embeds[key]
        moved = False
        if self.device.type == "cuda" and self.te.device.type == "cpu":
            self.te.model.to(self.device)
            self.te.device = self.device
            moved = True
        try:
            emb = self.te.encode(prompt, images=images) if images else self.te.encode(prompt)
        finally:
            if moved and low_vram:
                self.offload_te()
        self.embeds[key] = emb
        if len(self.embeds) > 32:
            self.embeds.popitem(last=False)
        return emb

    # ---- requests
    def generate(self, req: dict):
        rid = req["id"]
        low_vram = bool(req.get("low_vram"))
        self.load(req["model"], low_vram)
        if low_vram:
            self.offload_te()
        inpaint, control, regions = req.get("inpaint"), req.get("control"), req.get("regions") or []
        if (inpaint or control or regions) and self.backend.family != "krea2":
            raise ValueError("Inpainting, control images and regions work with Krea 2 and Kroma models only.")
        specs = [(lr["path"], float(lr.get("strength", 1.0))) for lr in req.get("loras", [])]
        specs += [(x["lora"], float(x.get("strength", 1.0))) for x in (inpaint, control) if x and x.get("lora")]
        unmatched = self.loras.activate(specs, self.device)
        for path, miss in unmatched.items():
            if miss:
                emit(event="warning", id=rid, message=f"{Path(path).name}: {miss} layers didn't match this model")

        from PIL import Image, ImageFilter, ImageOps

        def load(path, mode="RGB"):
            with Image.open(path) as im:
                return ImageOps.exif_transpose(im).convert(mode)

        width, height = int(req["width"]) // 16 * 16, int(req["height"]) // 16 * 16
        extra, ctx, thumbs, key = {}, {}, None, None
        source = paste = None  # the source image, and (source, mask) at the result's size to paste the result into
        emit(event="status", state="generating", message="Encoding prompt…")
        if req.get("image"):  # edit a source image (Fibo 2.0, Krea 2): it conditions the prompt and the transformer
            if not self.backend.supports_edit:
                raise ValueError("This model can't edit images; remove the source image or pick a Fibo 2.0 or Krea 2 model.")
            src = Path(req["image"])
            source = load(src)
            key = f"{src.resolve()}\x00{src.stat().st_mtime_ns}"
        ref_image = source  # what the model gets as its reference image, and how it reads it
        if inpaint:
            if source is None:
                raise ValueError("Inpainting needs a source image.")
            mask = load(inpaint["mask"], "L").resize(source.size).point(lambda v: 255 if v >= 128 else 0)
            mkey = f"{Path(inpaint['mask']).resolve()}\x00{Path(inpaint['mask']).stat().st_mtime_ns}"
            if inpaint.get("lora"):  # the LoRA repaints what is black in its reference
                ref_image = Image.composite(Image.new("RGB", source.size), source, mask)
                ctx.update(inpaint.get("ref") or {})
                key += "\x00" + mkey
            else:
                ref_image = None
        if control and control["type"] != "depth":  # pose and canny LoRAs read their map as the reference image
            if source is not None:
                raise ValueError(f"{control['type'].capitalize()} control uses the reference image itself, so it "
                                 "can't be combined with a source image. Remove one of them (depth control can).")
            cpath = Path(control["image"])
            ref_image = load(cpath)
            ctx.update(control.get("ref") or {})
            key = f"{cpath.resolve()}\x00{cpath.stat().st_mtime_ns}"

        if ref_image is not None and ctx.get("ref_t0", True):  # ostris' edit: Qwen3-VL sees the reference too
            ctx_refs, thumbs, width, height = self.backend.prepare_edit(self.vae, ref_image, width, height)
            if self.backend.family == "krea2":
                ctx["refs"] = ctx_refs
            else:
                extra["context"] = ctx_refs
            key = f"{req['prompt']}\x00{key}\x00{width}x{height}"
            pos = self.encode(req["prompt"], low_vram, images=thumbs, key=key)
        else:
            basis = next((x for x in (ref_image, source, load(control["image"]) if control else None)
                          if x is not None), None)
            if basis is not None:  # the result takes this picture's shape, at about the size asked for
                width, height = self.backend.edit_size(basis.width, basis.height, width * height)
            if ref_image is not None:  # a plain in-context LoRA: the reference at the result's size, text-only prompt
                ctx["refs"] = [self.backend.encode_images(self.vae, self.backend.edit_target(ref_image, width, height))]
            pos = self.encode(req["prompt"], low_vram)
            early = self.backend.prompt_parts(req["prompt"])[1]
            if early:
                extra["early"] = self.encode(early, low_vram)
        if thumbs and self.backend.edit_null_with_images:  # the negative sees the reference too (Krea 2)
            neg = self.encode(req.get("negative") or "", low_vram, images=thumbs, key="neg\x00" + key)
        else:
            neg = self.encode(self.backend.negative_prompt(req.get("negative") or ""), low_vram)

        if inpaint:  # the source and its mask at the result's size (same shape, so a plain resize)
            fitted = (source.resize((width, height), Image.LANCZOS), mask.resize((width, height)))
            if not inpaint.get("lora"):
                extra["inpaint"] = self.backend.inpaint_inputs(self.vae, *fitted)
            if inpaint.get("keep", True):
                paste = fitted
        if control and control["type"] == "depth":
            from .lora import read_control_input
            weight = read_control_input(control["lora"])
            depth_map = load(control["image"], "L").convert("RGB")
            tokens = self.backend.control_tokens(self.vae, depth_map, width, height)
            if weight is None or weight.shape[1] != tokens.shape[-1]:
                raise ValueError("The depth control file has no widened input layer for this model.")
            ctx["control"] = (tokens, weight)
        if regions:
            tokens = self.backend.region_tokens(regions, width, height)
            ctx["regions"] = [(self.encode(r["prompt"], low_vram), t) for r, t in zip(regions, tokens) if t.any()]
        if ctx:
            extra["context"] = ctx

        out_dir = Path(req["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        count = max(1, min(int(req.get("count", 1)), 8))
        base_seed = int(req.get("seed", -1))
        if base_seed < 0:
            base_seed = random.randint(0, 2**31 - 1)
        skip = int(req.get("skip_layer", 22))
        steps = int(req.get("steps", 50))

        for i in range(count):
            seed = base_seed + i

            def progress(step, total, i=i):
                if self.cancel:
                    raise Cancelled()
                emit(event="progress", id=rid, image=i, count=count, step=step, steps=total)

            t0 = time.time()
            img = self.backend.sample(self.transformer, self.vae, pos, neg, width=width, height=height,
                                      steps=steps, cfg=float(req.get("cfg", 3.0)), seed=seed,
                                      shift=float(req.get("shift", 0.0)), skip_layer=skip, device=self.device,
                                      progress=progress, **extra)
            if paste:  # keep everything outside the mask exactly as it was, with a soft edge
                soft = paste[1].filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(4))
                img = Image.composite(img.convert("RGB"), paste[0], soft)
            stem = f"{time.strftime('%Y%m%d-%H%M%S')}_{rid}_{i}"
            img.save(out_dir / f"{stem}.png")
            thumb = img.copy()
            thumb.thumbnail((512, 512))
            thumb.convert("RGB").save(out_dir / f"{stem}.thumb.jpg", quality=85)
            meta = {k: req.get(k) for k in ("model", "prompt", "negative", "steps", "cfg", "shift", "skip_layer",
                                            "image_id")}
            if inpaint:
                meta["inpaint"] = {"mask": Path(inpaint["mask"]).name, "lora": bool(inpaint.get("lora")),
                                   "strength": inpaint.get("strength", 1.0)}
            if control:
                meta["control"] = {"type": control["type"], "image": Path(control["image"]).name,
                                   "strength": control.get("strength", 1.0)}
            if regions:
                meta["regions"] = regions
            meta.update(width=width, height=height, seed=seed, created=time.time(), seconds=round(time.time() - t0, 1),
                        loras=[{"name": lr.get("name") or Path(lr["path"]).name, "path": lr.get("id", lr["path"]),
                                "strength": lr.get("strength", 1.0)} for lr in req.get("loras", [])])
            (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))
            emit(event="image", id=rid, file=f"{stem}.png", index=i)
        emit(event="done", id=rid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idle", type=float, default=900, help="exit after this many idle seconds")
    args = ap.parse_args()
    worker = Worker()

    def on_cancel(signum, frame):
        worker.cancel = True
    signal.signal(signal.SIGUSR1, on_cancel)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    lines: queue.Queue = queue.Queue()

    def reader():
        for raw in sys.stdin:
            lines.put(raw)
        lines.put(None)  # server went away
    threading.Thread(target=reader, daemon=True).start()

    emit(event="status", state="idle", message="Worker started")
    last = time.time()
    while True:
        try:
            line = lines.get(timeout=5.0)
        except queue.Empty:
            if time.time() - last > args.idle:
                emit(event="status", state="exiting", message="Idle timeout, freeing GPU memory")
                return
            continue
        if line is None:
            return
        last = time.time()
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        if req.get("cmd") == "offload":
            worker.offload_te()
            continue
        if req.get("cmd") != "generate":
            continue
        worker.cancel = False
        try:
            worker.generate(req)
        except Cancelled:
            emit(event="cancelled", id=req.get("id"))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            emit(event="error", id=req.get("id"),
                 message="Out of GPU memory. A training job may be using the GPU; try a smaller size or wait.")
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            emit(event="error", id=req.get("id"), message=f"{type(e).__name__}: {e}")
        finally:
            worker.cancel = False
            last = time.time()
        emit(event="status", state="ready", model=worker.model_id, message="")


if __name__ == "__main__":
    main()
