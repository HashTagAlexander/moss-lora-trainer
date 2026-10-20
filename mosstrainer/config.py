"""Job configuration defaults, shared by the trainer and the web UI."""
from __future__ import annotations

import copy
import re

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

import json
import os

# Every model the trainer knows. "family" selects the backend in mosstrainer/backends.py;
# "defaults" are the recommended sampling settings (cfg uses the standard convention: 1 = off).
_KROMA = "lodestones/Kroma"
_KREA_TE = "Qwen/Qwen3-VL-4B-Instruct"   # byte-identical to the encoder Krea 2 ships
_KREA_VAE = "Qwen/Qwen-Image"            # byte-identical to the Krea 2 VAE
_MOSS_DEFAULTS = {"steps": 30, "cfg": 3.0, "shift": 5.0, "skip_layer": 22, "timestep_sampling": "moss"}
_KREA_DEFAULTS = {"steps": 28, "cfg": 5.5, "shift": 0.0, "skip_layer": -1, "timestep_sampling": "flux_shift"}
_KREA_TURBO_DEFAULTS = {"steps": 8, "cfg": 1.0, "shift": 1.15, "skip_layer": -1, "timestep_sampling": "flux_shift"}
# Fibo 2.0 reads structured VGL captions (JSON). The negative is the one Bria's model card uses for text-to-image.
FIBO_NEGATIVE_T2I = (
    '{"short_description":"","objects":[{"description":"the main subject","texture":"smooth, featureless, '
    'plastic-like, airbrushed, no fine surface detail","appearance_details":"soft blurred edges, simplified shapes, '
    'no fine detail","skin_tone_and_texture":"waxy airbrushed skin, poreless, plastic smooth"}],"background_setting":'
    '"a plain seamless studio backdrop, perfectly smooth gradient, blurred and featureless, no visible texture",'
    '"lighting":{},"aesthetics":{"artifacts":["oversaturated colours","neon colour cast","overexposed highlights",'
    '"lens flare","slight blur","motion blur","haze"]},"photographic_characteristics":{"depth_of_field":"shallow, '
    'heavy bokeh obscuring most of the frame","focus":"soft, slightly out of focus, lacking fine texture"},'
    '"style_medium":"","context":"","artistic_style":""}')
_FIBO_DEFAULTS = {"steps": 50, "cfg": 5.0, "shift": 0.0, "skip_layer": -1, "timestep_sampling": "fibo",
                  "negative": FIBO_NEGATIVE_T2I}
_FIBO_TURBO_DEFAULTS = {"steps": 4, "cfg": 1.0, "shift": 0.0, "skip_layer": -1, "timestep_sampling": "fibo"}
# The recipe of Bria's fine_tuning/train_lora.py: LoRA on every Linear layer, AdamW with weight decay 0.01,
# 10% caption dropout to the null caption.
_FIBO_OVERRIDES = {
    "network": {"targets": ["attn", "ffn", "text", "other"]},
    "data": {"caption_dropout": 0.1},
    "train": {"weight_decay": 0.01},
    "sample": {"prompts": [
        '{"short_description":"[trigger], a portrait photo in soft natural window light","style_medium":"photograph"}',
        '{"short_description":"[trigger] standing on a city street at night under neon signs",'
        '"text_render":[{"text":"FIBO","location":"top center","size":"large","color":"pink neon",'
        '"font":"neon sign lettering"}],"style_medium":"photograph"}',
    ]},
}

# LoRA target groups per family: key -> label (the regexes live in backends.py).
FAMILY_TARGETS = {
    "moss": {"attn": "Attention (q, k, v, o)", "ffn": "Feed-forward", "text": "Text refiner blocks"},
    "krea2": {"attn": "Attention (q, k, v, gate, o)", "ffn": "Feed-forward", "text": "Text fusion blocks"},
    "fibo2": {"attn": "Attention (self + text cross-attention)", "ffn": "Feed-forward", "text": "Text perceiver",
              "other": "Modulation & embedders"},
}

MODEL_SPECS = {
    "OpenMOSS-Team/MOSS-Image-Base": {
        "label": "MOSS-Image Base (recommended for LoRA)", "family": "moss", "trainable": True,
        "defaults": _MOSS_DEFAULTS},
    "OpenMOSS-Team/MOSS-Image-SFT": {
        "label": "MOSS-Image SFT", "family": "moss", "trainable": True, "defaults": _MOSS_DEFAULTS},
    "kroma-v0.3-teacher": {
        "label": "Kroma v0.3 teacher (Krea 2, recommended for LoRA)", "family": "krea2", "trainable": True,
        "repo": _KROMA, "file": "kroma-sensei-booru-e6-teacher-velocity-v0.3.safetensors",
        "text_encoder": _KREA_TE, "vae": _KREA_VAE, "defaults": _KREA_DEFAULTS},
    "kroma-v0.3-base": {
        "label": "Kroma v0.3 base (Krea 2)", "family": "krea2", "trainable": True,
        "repo": _KROMA, "file": "kroma-v0.3-base.safetensors",
        "text_encoder": _KREA_TE, "vae": _KREA_VAE, "defaults": _KREA_DEFAULTS},
    "kroma-v0.3.1-turbo": {
        "label": "Kroma v0.3.1 Turbo OPD (Krea 2, generation only)", "family": "krea2", "trainable": False,
        "repo": _KROMA, "file": "kroma-v0.3.1-turbo-opd.safetensors",
        "text_encoder": _KREA_TE, "vae": _KREA_VAE, "defaults": _KREA_TURBO_DEFAULTS},
    "krea2-raw": {
        "label": "Krea 2 Raw", "family": "krea2", "trainable": True,
        "repo": "Comfy-Org/Krea-2", "file": "diffusion_models/krea2_raw_bf16.safetensors",
        "text_encoder": _KREA_TE, "vae": _KREA_VAE, "defaults": _KREA_DEFAULTS},
    "krea2-turbo": {
        "label": "Krea 2 Turbo (generation only)", "family": "krea2", "trainable": False,
        "repo": "Comfy-Org/Krea-2", "file": "diffusion_models/krea2_turbo_bf16.safetensors",
        "text_encoder": _KREA_TE, "vae": _KREA_VAE, "defaults": _KREA_TURBO_DEFAULTS},
    # Gated on Hugging Face: accept the license and add a token in Settings.
    "briaai/Fibo-2.0-dev": {
        "label": "Fibo 2.0 Dev (Bria, recommended for LoRA)", "family": "fibo2", "trainable": True, "gated": True,
        "edit": True, "defaults": _FIBO_DEFAULTS, "overrides": _FIBO_OVERRIDES},
    "briaai/Fibo-2.0-edit-turbo": {
        "label": "Fibo 2.0 Edit Turbo (4 steps, generation only)", "family": "fibo2", "trainable": False,
        "gated": True, "edit": True, "defaults": _FIBO_TURBO_DEFAULTS},
}

# Extra models (local paths etc.) can be registered from a JSON file, e.g. for tests.
if os.environ.get("MOSS_EXTRA_MODELS"):
    MODEL_SPECS.update(json.loads(open(os.environ["MOSS_EXTRA_MODELS"]).read()))

MODELS = {k: v["label"] for k, v in MODEL_SPECS.items()}
TRAINABLE = {k: v["label"] for k, v in MODEL_SPECS.items() if v.get("trainable")}


def model_spec(model_id: str) -> dict:
    if model_id not in MODEL_SPECS:
        raise ValueError(f"Unknown model: {model_id}")
    return MODEL_SPECS[model_id]

DEFAULTS = {
    "name": "my_lora",
    "model": "OpenMOSS-Team/MOSS-Image-Base",
    "dataset": "",
    "trigger_word": "",
    "network": {
        "rank": 32,
        "alpha": 32,
        "dropout": 0.0,
        "targets": ["attn", "ffn"],   # attn | ffn | text
        "layers": "",                  # "" = all, else e.g. "0-20,30"
    },
    "data": {
        "resolutions": [1024],
        "flip": False,
        "repeats": 1,
        "caption_dropout": 0.05,
    },
    "train": {
        "steps": 2000,
        "batch_size": 1,
        "grad_accum": 1,
        "lr": 1e-4,
        "optimizer": "adamw",          # adamw | adamw8bit | prodigy | adafactor
        "weight_decay": 0.0,
        "lr_scheduler": "constant",    # constant | cosine | linear
        "warmup_steps": 0,
        "max_grad_norm": 1.0,
        "timestep_sampling": "moss",   # moss | flux_shift | logit_normal | shift | uniform
        "shift": 3.0,
        "logit_mean": 0.0,
        "logit_std": 1.0,
        "gradient_checkpointing": True,
        "seed": 42,
    },
    "save": {
        "every": 250,
        "keep_last": 4,
        "dtype": "bf16",               # bf16 | fp16 | fp32
    },
    "sample": {
        "every": 250,
        "sample_first": True,
        "prompts": [
            "[trigger], a portrait photo, soft natural light",
            "[trigger] standing on a city street at night, neon signs that read \"MOSS\"",
        ],
        "negative_prompt": "",
        "width": 1024,
        "height": 1024,
        "steps": 30,
        "cfg": 3.0,
        "shift": 5.0,
        "seed": 42,
        "uncond_skip_layer": 22,       # -1 disables
    },
}


def defaults_for(model_id: str) -> dict:
    """DEFAULTS with the model's recommended sampling / timestep settings (and training overrides) filled in."""
    spec = MODEL_SPECS.get(model_id, {})
    d = spec.get("defaults", {})
    return merge(merge(DEFAULTS, {
        "model": model_id,
        "train": {"timestep_sampling": d.get("timestep_sampling", "moss")},
        "sample": {"steps": d.get("steps", 30), "cfg": d.get("cfg", 3.0), "shift": d.get("shift", 5.0),
                   "uncond_skip_layer": d.get("skip_layer", 22), "negative_prompt": d.get("negative", "")},
    }), spec.get("overrides", {}))


def merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_layers(spec) -> list[int] | None:
    if spec in (None, "", "all"):
        return None
    if isinstance(spec, list):
        return [int(x) for x in spec]
    layers = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            layers.update(range(int(a), int(b) + 1))
        else:
            layers.add(int(part))
    return sorted(layers)


def validate(cfg: dict) -> dict:
    spec = MODEL_SPECS.get((cfg or {}).get("model", DEFAULTS["model"]))
    if spec is None:
        raise ValueError(f"Unknown model: {cfg.get('model')}")
    cfg = merge(defaults_for(cfg.get("model", DEFAULTS["model"])), cfg)
    if not spec.get("trainable"):
        raise ValueError(f"{spec['label']} is for generation only; train on its teacher/base model instead.")
    if not NAME_RE.match(cfg["name"]):
        raise ValueError("Name may only contain letters, digits, '.', '_' and '-'.")
    if not cfg["dataset"]:
        raise ValueError("Pick a dataset.")
    res = cfg["data"]["resolutions"]
    if isinstance(res, (int, str)):
        res = [res]
    res = sorted({int(r) // 16 * 16 for r in res})
    if not res or any(r < 256 or r > 2048 for r in res):
        raise ValueError("Resolutions must be between 256 and 2048.")
    cfg["data"]["resolutions"] = res
    n = cfg["network"]
    n["rank"], n["alpha"] = int(n["rank"]), float(n["alpha"])
    if not 1 <= n["rank"] <= 512:
        raise ValueError("Rank must be between 1 and 512.")
    if not n["targets"]:
        raise ValueError("Select at least one LoRA target.")
    allowed = FAMILY_TARGETS.get(spec["family"], {})
    n["targets"] = [x for x in n["targets"] if x in allowed]
    if not n["targets"]:
        raise ValueError(f"Select at least one LoRA target ({', '.join(allowed)}).")
    parse_layers(n["layers"])
    t = cfg["train"]
    for key in ("steps", "batch_size", "grad_accum", "warmup_steps", "seed"):
        t[key] = int(t[key])
    for key in ("lr", "weight_decay", "max_grad_norm", "shift", "logit_mean", "logit_std"):
        t[key] = float(t[key])
    if t["steps"] < 1 or t["batch_size"] < 1 or t["grad_accum"] < 1:
        raise ValueError("Steps, batch size and grad accumulation must be >= 1.")
    s = cfg["sample"]
    s["width"], s["height"] = int(s["width"]) // 16 * 16, int(s["height"]) // 16 * 16
    if isinstance(s["prompts"], str):
        s["prompts"] = [p for p in s["prompts"].splitlines() if p.strip()]
    for key in ("every", "steps", "seed", "uncond_skip_layer"):
        s[key] = int(s[key])
    for key in ("cfg", "shift"):
        s[key] = float(s[key])
    cfg["save"]["every"] = int(cfg["save"]["every"])
    cfg["save"]["keep_last"] = int(cfg["save"]["keep_last"])
    return cfg
