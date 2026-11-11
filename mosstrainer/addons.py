"""Optional downloads for the Generate tab's inpainting and control images (Krea 2 / Kroma).

Community files from Hugging Face, kept in its cache like the base models. Nothing here is fetched until asked for:
the Generate tab shows a Download button, or

    python -m mosstrainer.addons depth depth-estimator
"""
from __future__ import annotations

import sys
from pathlib import Path

_MODEL_FILES = ["config.json", "preprocessor_config.json", "model.safetensors"]

# "files": [(repo, [paths])]. "ref": how a reference-image LoRA reads its image (see Krea2Backend._forward_ctx).
ADDONS = {
    "inpaint": {
        "label": "Inpaint LoRA", "by": "Cierpliwy", "size": 457_111_992, "license": "Krea 2 Community License",
        "files": [("Cierpliwy/krea2-inpaint-edit", ["krea2_inpaint_edit_v2_17500.safetensors"])],
        "about": "Fills the blacked-out part of the source image from the prompt. Trained on Krea 2 Raw.",
        "ref": {"ref_t0": True, "isolate": True}},
    "depth": {
        "label": "Depth control-LoRA", "by": "Patil", "size": 861_995_928, "license": "Krea 2 Community License",
        "files": [("Patil/Krea-2-depth-controlnet", ["depth-control-lora.safetensors"])],
        "about": "Follows a depth map (near = white). Trained on Krea 2 Raw, works on Turbo."},
    "pose": {
        "label": "Pose control LoRA", "by": "thedeoxen", "size": 228_587_504, "license": "Apache 2.0",
        "files": [("thedeoxen/Krea-2-pose-controlnet", ["krea2_turbo_openpose_controlnet.safetensors"])],
        "about": "Follows an OpenPose skeleton on black. Trained on Krea 2 Turbo; strength 0.8-1.",
        "ref": {"ref_t0": True, "isolate": False}},
    "canny": {
        "label": "Canny control LoRA", "by": "nynxz (NK2E)", "size": 192_742_712, "license": "Krea 2 Community License",
        "files": [("nynxz/NK2E", ["comfy/canny_v0.1/NK2E-canny-v0.1.safetensors"])],
        "about": "Follows an edge map, loosely (experimental). Trained on Krea 2 Raw; strength around 0.7.",
        "ref": {"ref_t0": False, "isolate": False}},
    "depth-estimator": {
        "label": "Depth estimator", "by": "Depth Anything V2 Large", "size": 1_341_322_868,
        "license": "CC BY-NC 4.0 (non-commercial)",
        "files": [("depth-anything/Depth-Anything-V2-Large-hf", _MODEL_FILES)],
        "about": "Makes the depth map from a photo. Not needed if you upload a ready-made depth map."},
    "pose-estimator": {
        "label": "Pose estimator", "by": "ViTPose + RT-DETR", "size": 343_673_076 + 172_175_856, "license": "Apache 2.0",
        "files": [("usyd-community/vitpose-base-simple", _MODEL_FILES), ("PekingU/rtdetr_r50vd_coco_o365", _MODEL_FILES)],
        "about": "Makes the body skeleton from a photo. Not needed if you upload a ready-made pose map."},
}


def url(addon: str) -> str:
    return "https://huggingface.co/" + ADDONS[addon]["files"][0][0]


def paths(addon: str) -> list[Path] | None:
    """The add-on's files in the Hugging Face cache, or None when any is missing. Never downloads."""
    from huggingface_hub import try_to_load_from_cache
    out = []
    for repo, files in ADDONS[addon]["files"]:
        for name in files:
            hit = try_to_load_from_cache(repo, name)
            if not isinstance(hit, str):
                return None
            out.append(Path(hit))
    return out


def lora_path(addon: str) -> Path | None:
    found = paths(addon)
    return found[0] if found else None


def model_dirs(addon: str) -> list[Path] | None:
    """One folder per repo of an estimator (what `from_pretrained` takes), or None when it isn't downloaded."""
    found = paths(addon)
    if not found:
        return None
    return [next(p.parent for p in found if p.name == "config.json" and repo.replace("/", "--") in str(p))
            for repo, _ in ADDONS[addon]["files"]]


def fetch(addon: str) -> list[str]:
    from huggingface_hub import hf_hub_download
    return [hf_hub_download(repo, name) for repo, files in ADDONS[addon]["files"] for name in files]


if __name__ == "__main__":
    names = sys.argv[1:]
    unknown = [n for n in names if n not in ADDONS]
    if unknown or not names:
        sys.exit(f"usage: python -m mosstrainer.addons {' | '.join(ADDONS)}")
    for n in names:
        print("Done:", *fetch(n), flush=True)
