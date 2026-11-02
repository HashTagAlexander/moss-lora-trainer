"""Masks for masked training, made from words with CLIPSeg (CIDAS/clipseg-rd64-refined, built into transformers).

    python -m mosstrainer.automask --dataset DIR --keep "person" [--remove "text, logo"] [--overwrite]

Writes <image>-masklabel.png beside each image: white where the words in --keep are, minus --remove. CLIPSeg sees
the picture at 352x352, so its masks are soft and coarse: fine for weighting the training loss, which is all they
do here. Paint your own for anything precise. Progress: one JSON line per image on stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps

from .data import MASK_SUFFIX, list_images

MODEL = "CIDAS/clipseg-rd64-refined"


def _words(text: str) -> list[str]:
    return [w.strip() for w in (text or "").split(",") if w.strip()]


@torch.no_grad()
def make_mask(model, proc, image: Image.Image, keep: list[str], remove: list[str], device) -> Image.Image:
    prompts = keep + remove
    inputs = proc(text=prompts, images=[image] * len(prompts), padding=True, return_tensors="pt").to(device)
    p = model(**inputs).logits.float().sigmoid()
    p = p.view(len(prompts), *p.shape[-2:])
    # each word on its own scale (CLIPSeg's raw confidences vary a lot between words); unseen words stay ~0
    p = torch.where(p.amax(dim=(1, 2), keepdim=True) > 0.1, p / p.amax(dim=(1, 2), keepdim=True), torch.zeros_like(p))
    m = p[:len(keep)].amax(0)
    if remove:
        m = m - p[len(keep):].amax(0)
    m = ((m - 0.3) / 0.4).clamp(0, 1)
    m = torch.nn.functional.max_pool2d(m[None, None], 7, 1, 3)  # grow a little past the edges
    m = torch.nn.functional.avg_pool2d(m, 5, 1, 2)[0, 0]
    out = Image.fromarray((m.cpu().numpy() * 255).round().astype("uint8"), "L")
    return out.resize(image.size, Image.BILINEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--keep", default="person")
    ap.add_argument("--remove", default="")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    keep, remove = _words(args.keep), _words(args.remove)
    if not keep:
        sys.exit("say what to keep, e.g. --keep person")
    todo = [p for p in list_images(args.dataset)
            if args.overwrite or not p.with_name(p.stem + MASK_SUFFIX + ".png").exists()]
    print(json.dumps({"total": len(todo), "done": 0, "message": "Loading CLIPSeg"}), flush=True)
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = CLIPSegProcessor.from_pretrained(MODEL)
    model = CLIPSegForImageSegmentation.from_pretrained(MODEL).to(device).eval()
    for i, path in enumerate(todo, 1):
        with Image.open(path) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
        make_mask(model, proc, image, keep, remove, device).save(path.with_name(path.stem + MASK_SUFFIX + ".png"))
        print(json.dumps({"total": len(todo), "done": i, "file": path.name}), flush=True)


if __name__ == "__main__":
    main()
