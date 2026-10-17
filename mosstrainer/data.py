"""Dataset scanning, aspect bucketing and on-disk latent / text-embedding caches."""
from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file, save_file

from .model import bucket_sizes, closest_bucket

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class Item:
    image: Path
    caption: str
    bucket: tuple[int, int]
    flip: bool = False
    latent_path: Path | None = None
    text_key: str = ""


def read_caption(image: Path) -> str:
    txt = image.with_suffix(".txt")
    return txt.read_text(encoding="utf-8").strip() if txt.exists() else ""


def apply_trigger(caption: str, trigger: str) -> str:
    trigger = (trigger or "").strip()
    if not trigger:
        return caption.replace("[trigger]", "").strip()
    if "[trigger]" in caption:
        return caption.replace("[trigger]", trigger)
    if trigger in caption:
        return caption
    return f"{trigger}, {caption}" if caption else trigger


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."))


def build_items(folder: Path, resolutions: list[int], trigger: str, flip: bool, log=print) -> list[Item]:
    images = list_images(folder)
    if not images:
        raise ValueError(f"No images found in {folder}")
    buckets = {r: bucket_sizes(r) for r in resolutions}
    items = []
    for path in images:
        with Image.open(path) as im:
            w, h = ImageOps.exif_transpose(im).size
        caption = apply_trigger(read_caption(path), trigger)
        if not caption:
            log(f"warning: {path.name} has no caption and no trigger word")
        for res in resolutions:
            # Skip resolutions that would upscale a small image by more than ~1.4x per side.
            if len(resolutions) > 1 and w * h < 0.5 * res * res:
                continue
            bucket = closest_bucket(w, h, buckets[res])
            items.append(Item(path, caption, bucket))
            if flip:
                items.append(Item(path, caption, bucket, flip=True))
    return items


def load_pixels(path: Path, bucket: tuple[int, int], flip: bool) -> torch.Tensor:
    bw, bh = bucket
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        im = im.convert("RGB")
        scale = max(bw / im.width, bh / im.height)
        nw, nh = max(bw, round(im.width * scale)), max(bh, round(im.height * scale))
        im = im.resize((nw, nh), Image.LANCZOS)
        left, top = (nw - bw) // 2, (nh - bh) // 2
        im = im.crop((left, top, left + bw, top + bh))
        if flip:
            im = ImageOps.mirror(im)
        t = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8).view(bh, bw, 3)
    return t.permute(2, 0, 1).float().div(127.5).sub(1.0)


def _digest(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()


def text_key(model_id: str, prompt: str) -> str:
    return _digest("text-v1", model_id, prompt)


def cache_latents(items: list[Item], vae, cache_dir: Path, progress=None, encode_fn=None, tag: str = "lat-v1"):
    if encode_fn is None:
        from .model import encode_images as encode_fn
    encode_images = encode_fn
    cache_dir.mkdir(parents=True, exist_ok=True)
    for it in items:
        st = it.image.stat()
        it.latent_path = cache_dir / f"{_digest(tag, it.image.resolve(), st.st_mtime_ns, st.st_size, it.bucket, it.flip)}.safetensors"
    todo = [it for it in items if not it.latent_path.exists()]
    for i, it in enumerate(todo):
        pixels = load_pixels(it.image, it.bucket, it.flip).unsqueeze(0)
        latent = encode_images(vae, pixels)[0].to(torch.bfloat16).cpu()
        save_file({"latent": latent}, str(it.latent_path))
        if progress:
            progress(i + 1, len(todo))
    return len(todo)


class TextCache:
    def __init__(self, cache_dir: Path, model_id: str):
        self.dir = cache_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id
        self.mem: dict[str, torch.Tensor] = {}

    def path(self, prompt: str) -> Path:
        return self.dir / f"{text_key(self.model_id, prompt)}.safetensors"

    def missing(self, prompts) -> list[str]:
        return sorted({p for p in prompts if not self.path(p).exists()})

    def fill(self, encoder, prompts, progress=None):
        todo = self.missing(prompts)
        for i, p in enumerate(todo):
            save_file({"embeds": encoder.encode(p).contiguous()}, str(self.path(p)))
            if progress:
                progress(i + 1, len(todo))
        return len(todo)

    def get(self, prompt: str) -> torch.Tensor:
        if prompt not in self.mem:
            self.mem[prompt] = load_file(str(self.path(prompt)))["embeds"]
        return self.mem[prompt]


class BucketSampler:
    """Yields batches of items that share a bucket, reshuffled every epoch."""

    def __init__(self, items: list[Item], batch_size: int, repeats: int = 1, seed: int = 0):
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.groups = defaultdict(list)
        for it in items:
            for _ in range(max(1, repeats)):
                self.groups[it.bucket].append(it)
        self.epoch = 0
        self._queue: list[list[Item]] = []

    def _refill(self):
        batches = []
        for group in self.groups.values():
            group = group[:]
            self.rng.shuffle(group)
            for i in range(0, len(group), self.batch_size):
                batches.append(group[i:i + self.batch_size])
        self.rng.shuffle(batches)
        self._queue = batches
        self.epoch += 1

    def next(self) -> list[Item]:
        if not self._queue:
            self._refill()
        return self._queue.pop()

    def __len__(self):
        return sum((len(g) + self.batch_size - 1) // self.batch_size for g in self.groups.values())


def load_latent(item: Item) -> torch.Tensor:
    return load_file(str(item.latent_path))["latent"]
