"""Control images for the Generate tab, made from a photo: a depth map, a pose skeleton or an edge map.

    python -m mosstrainer.preprocess --type depth|pose|canny --image photo.jpg --out map.png

Each matches what its control LoRA was trained on (see mosstrainer/addons.py):
  depth  Depth Anything V2 Large, inverse depth scaled to the picture's own range, near = white
  pose   OpenPose-style body skeleton on black (18 points; people from RT-DETR, joints from ViTPose)
  canny  cv2.Canny(gray, 100, 200) rebuilt in numpy (OpenCV isn't a dependency): white edges on black
Depth and pose need their estimator downloaded first (`python -m mosstrainer.addons depth-estimator`).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

TYPES = ("depth", "pose", "canny")


# --------------------------------------------------------------------------- canny

def canny(image: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    """OpenCV's Canny, step for step: 3x3 Sobel on the grey image (edges repeated), |dx| + |dy|, thinning along the
    gradient with its fixed-point direction test, then weak edges kept where they touch a strong one."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.int32)
    gray = (rgb @ np.array([9798, 19235, 3735]) + 16384) >> 15  # OpenCV's fixed-point RGB -> grey
    g = np.pad(gray, 1, mode="edge")
    dx = (g[:-2, 2:] + 2 * g[1:-1, 2:] + g[2:, 2:]) - (g[:-2, :-2] + 2 * g[1:-1, :-2] + g[2:, :-2])
    dy = (g[2:, :-2] + 2 * g[2:, 1:-1] + g[2:, 2:]) - (g[:-2, :-2] + 2 * g[:-2, 1:-1] + g[:-2, 2:])
    mag = np.abs(dx) + np.abs(dy)
    m = np.pad(mag, 1)  # zero outside the picture

    def at(oy, ox):  # the magnitude one step away, per pixel
        if isinstance(ox, int):
            return m[1 + oy:m.shape[0] - 1 + oy, 1 + ox:m.shape[1] - 1 + ox]
        yy, xx = np.indices(mag.shape)
        return m[yy + 1 + oy, xx + 1 + ox]

    x, y = np.abs(dx).astype(np.int64), np.abs(dy).astype(np.int64) << 15
    tg22 = x * 13573  # tan(22.5 deg) in 15-bit fixed point
    tg67 = tg22 + (x << 16)
    across = (y < tg22) & (mag > at(0, -1)) & (mag >= at(0, 1))
    along = (y > tg67) & (mag > at(-1, 0)) & (mag >= at(1, 0))
    s = np.where((dx ^ dy) < 0, -1, 1)
    diagonal = (y >= tg22) & (y <= tg67) & (mag > at(-1, -s)) & (mag > at(1, s))
    peak = (mag > low) & (across | along | diagonal)
    strong = peak & (mag > high)
    weak = peak & ~strong
    while True:  # grow the strong edges through the weak ones they touch
        p = np.pad(strong, 1)
        near = np.zeros_like(strong)
        for oy in (0, 1, 2):
            for ox in (0, 1, 2):
                near |= p[oy:oy + strong.shape[0], ox:ox + strong.shape[1]]
        grown = near & weak
        if not grown.any():
            break
        strong |= grown
        weak &= ~grown
    return Image.fromarray(strong.astype(np.uint8) * 255, "L").convert("RGB")


# --------------------------------------------------------------------------- depth

@torch.no_grad()
def depth(image: Image.Image, model_dir, device) -> Image.Image:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    proc = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForDepthEstimation.from_pretrained(model_dir, dtype=dtype).to(device).eval()
    pixels = proc(images=[image], return_tensors="pt")["pixel_values"].to(device, dtype)
    d = model(pixel_values=pixels).predicted_depth[None].float()
    d = torch.nn.functional.interpolate(d, size=(image.height, image.width), mode="bilinear", align_corners=False)[0, 0]
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    return Image.fromarray((d.cpu().numpy() * 255).astype(np.uint8), "L").convert("RGB")


# --------------------------------------------------------------------------- pose

# OpenPose's 18 body points from COCO's 17 (None: the neck, halfway between the shoulders), its limbs and colours.
_FROM_COCO = [0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
_LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
          (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
_COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
           (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
           (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]


def draw_pose(people, size: tuple[int, int], threshold: float = 0.3) -> Image.Image:
    """people: [(keypoints [17, 2] in pixels, scores [17])] in COCO order. Limbs are ellipses in 60% of their
    colour with the joints as dots on top, on black: the picture controlnet_aux's OpenPose / DWPose body draws."""
    width, height = size
    canvas = Image.new("RGB", size)
    draw = ImageDraw.Draw(canvas)
    stick = max(4.0, min(width, height) / 256)
    for keypoints, scores in people:
        pts = []
        for src in _FROM_COCO:
            if src is None:
                ok = scores[5] >= threshold and scores[6] >= threshold
                pts.append(((keypoints[5][0] + keypoints[6][0]) / 2, (keypoints[5][1] + keypoints[6][1]) / 2) if ok else None)
            else:
                pts.append((float(keypoints[src][0]), float(keypoints[src][1])) if scores[src] >= threshold else None)
        for (a, b), color in zip(_LIMBS, _COLORS):
            if pts[a] is None or pts[b] is None:
                continue
            (ax, ay), (bx, by) = pts[a], pts[b]
            cx, cy, half, angle = (ax + bx) / 2, (ay + by) / 2, math.hypot(bx - ax, by - ay) / 2, math.atan2(by - ay, bx - ax)
            outline = [(cx + half * math.cos(t) * math.cos(angle) - stick * math.sin(t) * math.sin(angle),
                        cy + half * math.cos(t) * math.sin(angle) + stick * math.sin(t) * math.cos(angle))
                       for t in (2 * math.pi * k / 36 for k in range(36))]
            draw.polygon(outline, fill=tuple(int(c * 0.6) for c in color))
        for pt, color in zip(pts, _COLORS):
            if pt is not None:
                draw.ellipse((pt[0] - stick, pt[1] - stick, pt[0] + stick, pt[1] + stick), fill=color)
    return canvas


@torch.no_grad()
def find_people(image: Image.Image, pose_dir, detector_dir, device):
    """[(keypoints [17, 2], scores [17])] for every person RT-DETR finds (the whole picture when it finds nobody)."""
    from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation
    try:
        import scipy  # noqa: F401  (ViTPose's image processor warps each person's box with it)
    except ImportError:
        raise FileNotFoundError("The pose estimator needs scipy: pip install scipy (or upload a ready-made pose map).")
    det_proc = AutoProcessor.from_pretrained(detector_dir)
    detector = RTDetrForObjectDetection.from_pretrained(detector_dir).to(device).eval()
    found = detector(**det_proc(images=image, return_tensors="pt").to(device))
    found = det_proc.post_process_object_detection(found, target_sizes=torch.tensor([(image.height, image.width)]),
                                                   threshold=0.3)[0]
    boxes = found["boxes"][found["labels"] == 0].float().cpu().numpy()  # COCO label 0: person
    if len(boxes) == 0:
        boxes = np.array([[0, 0, image.width, image.height]], dtype=np.float32)
    boxes[:, 2] -= boxes[:, 0]  # corners -> x, y, width, height
    boxes[:, 3] -= boxes[:, 1]
    pose_proc = AutoProcessor.from_pretrained(pose_dir)
    model = VitPoseForPoseEstimation.from_pretrained(pose_dir).to(device).eval()
    out = model(**pose_proc(image, boxes=[boxes], return_tensors="pt").to(device))
    people = pose_proc.post_process_pose_estimation(out, boxes=[boxes])[0]
    return [(p["keypoints"].float().cpu().numpy(), p["scores"].float().cpu().numpy()) for p in people]


def pose(image: Image.Image, pose_dir, detector_dir, device) -> Image.Image:
    return draw_pose(find_people(image, pose_dir, detector_dir, device), image.size)


# --------------------------------------------------------------------------- command line

def make(kind: str, image: Image.Image, device=None) -> Image.Image:
    from . import addons
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.convert("RGB")
    image.thumbnail((1024, 1024) if kind == "canny" else (2048, 2048), Image.LANCZOS)
    if kind == "canny":
        return canny(image)
    dirs = addons.model_dirs(f"{kind}-estimator")
    if not dirs:
        raise FileNotFoundError(f"The {kind} estimator isn't downloaded yet: download it first, or upload a ready-made {kind} map.")
    return depth(image, dirs[0], device) if kind == "depth" else pose(image, dirs[0], dirs[1], device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", required=True, choices=TYPES)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    with Image.open(args.image) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")
    try:
        make(args.type, image).save(args.out)
    except FileNotFoundError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
