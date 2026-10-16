"""Long-lived image generation worker used by the web UI's Generate tab.

Protocol: one JSON request per line on stdin; one JSON event per line on stdout.
Library output is redirected to stderr so it can't corrupt the event stream.

Requests:
  {"cmd": "generate", "id", "model", "prompt", "negative", "width", "height", "steps", "cfg",
   "shift", "seed", "count", "skip_layer", "loras": [{"path", "strength", "name"}], "low_vram", "out_dir"}
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

from .lora import LoRAStack  # noqa: E402
from .model import (MOSSImageScheduler, TextEncoder, load_transformer, load_vae, resolve_model,  # noqa: E402
                    sample)


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
        emit(event="status", state="loading", message=f"Loading {model_id.split('/')[-1]}…")
        root = resolve_model(model_id)
        te_device = torch.device("cpu") if low_vram else self.device
        fake_dim = os.environ.get("MOSS_TEST_FAKE_TE")  # tests only: skip the 9B encoder
        self.te = _FakeTE(int(fake_dim)) if fake_dim else TextEncoder(root, te_device)
        self.transformer = load_transformer(root).to(self.device).eval()
        self.vae = load_vae(root).to(self.device)
        self.loras = LoRAStack(self.transformer)
        self.model_id = model_id
        emit(event="status", state="ready", model=model_id, message="Model loaded")

    def free(self):
        self.te = self.transformer = self.vae = self.loras = None
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

    def encode(self, prompt: str, low_vram: bool) -> torch.Tensor:
        if prompt in self.embeds:
            self.embeds.move_to_end(prompt)
            return self.embeds[prompt]
        moved = False
        if self.device.type == "cuda" and self.te.device.type == "cpu":
            self.te.model.to(self.device)
            self.te.device = self.device
            moved = True
        try:
            emb = self.te.encode(prompt)
        finally:
            if moved and low_vram:
                self.offload_te()
        self.embeds[prompt] = emb
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
        specs = [(lr["path"], float(lr.get("strength", 1.0))) for lr in req.get("loras", [])]
        unmatched = self.loras.activate(specs, self.device)
        for path, miss in unmatched.items():
            if miss:
                emit(event="warning", id=rid, message=f"{Path(path).name}: {miss} layers didn't match this model")

        emit(event="status", state="generating", message="Encoding prompt…")
        pos = self.encode(req["prompt"], low_vram)
        neg = self.encode(req.get("negative", ""), low_vram)

        out_dir = Path(req["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        count = max(1, min(int(req.get("count", 1)), 8))
        base_seed = int(req.get("seed", -1))
        if base_seed < 0:
            base_seed = random.randint(0, 2**31 - 1)
        skip = int(req.get("skip_layer", 22))
        scheduler = MOSSImageScheduler(shift=float(req.get("shift", 5.0)))
        width, height = int(req["width"]) // 16 * 16, int(req["height"]) // 16 * 16
        steps = int(req.get("steps", 50))

        for i in range(count):
            seed = base_seed + i

            def progress(step, total, i=i):
                if self.cancel:
                    raise Cancelled()
                emit(event="progress", id=rid, image=i, count=count, step=step, steps=total)

            t0 = time.time()
            img = sample(self.transformer, scheduler, self.vae, pos, neg, width=width, height=height,
                         steps=steps, cfg=float(req.get("cfg", 3.0)), seed=seed,
                         uncond_skip_layers=(skip,) if skip >= 0 else (), device=self.device,
                         progress=progress)
            stem = f"{time.strftime('%Y%m%d-%H%M%S')}_{rid}_{i}"
            img.save(out_dir / f"{stem}.png")
            thumb = img.copy()
            thumb.thumbnail((512, 512))
            thumb.convert("RGB").save(out_dir / f"{stem}.thumb.jpg", quality=85)
            meta = {k: req.get(k) for k in ("model", "prompt", "negative", "steps", "cfg", "shift", "skip_layer")}
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
