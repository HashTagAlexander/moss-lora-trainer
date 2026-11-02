"""Fibo 2.0 LoRAs in the layout ComfyUI loads.

ComfyUI runs Fibo 2.0 from Comfy-Org's repacked file (Comfy-Org/Fibo-2.0), whose layers differ from diffusers':
  attention.to_q / to_k / to_v        -> attention.qkv (one Linear, rows q | k | v)
  attention.to_out.0                  -> attention.out
  cross/self_attention.to_k / to_v    -> to_kv (rows k | v); to_out.0 -> to_out; to_q stays
  feed_forward / ffn .w1 / .w3        -> w13 (rows w1 | w3), each half zero-padded to a multiple of 256
  feed_forward / ffn .w2              -> w2, whose input is that padded width
and ComfyUI has no key mapping from diffusers names for this model, so a diffusers-format LoRA doesn't load there.
Separate LoRAs fuse into one exactly: the down projections stack, the up projections go block-diagonal, and the
padding rows/columns are zero. Keys follow Comfy-Org's own Fibo 2.0 LoRA:
``diffusion_model.<module>.lora_down.weight`` / ``lora_up.weight`` / ``alpha`` (alpha = rank: scale 1, since our
diffusers files already fold alpha/rank into the up weights).
"""
from __future__ import annotations

import re

import torch

PAD = 256  # Comfy-Org pads the feed-forward width to this multiple (9536 -> 9728)

# diffusers module -> (ComfyUI module, slot in the fused Linear, number of slots); slot None = not fused
_RULES = [
    (re.compile(r"^(.*\.attention)\.to_(q|k|v)$"), lambda m: (f"{m[1]}.qkv", "qkv".index(m[2]), 3)),
    (re.compile(r"^(.*\.attention)\.to_out\.0$"), lambda m: (f"{m[1]}.out", None, 1)),
    (re.compile(r"^(.*_attention)\.to_q$"), lambda m: (f"{m[1]}.to_q", None, 1)),
    (re.compile(r"^(.*_attention)\.to_(k|v)$"), lambda m: (f"{m[1]}.to_kv", "kv".index(m[2]), 2)),
    (re.compile(r"^(.*_attention)\.to_out\.0$"), lambda m: (f"{m[1]}.to_out", None, 1)),
    (re.compile(r"^(.*\.(?:feed_forward|ffn))\.w(1|3)$"), lambda m: (f"{m[1]}.w13", "13".index(m[2]), 2)),
]


def comfy_name(name: str) -> tuple[str, int | None, int]:
    for rx, fn in _RULES:
        m = rx.match(name)
        if m:
            return fn(m)
    return name, None, 1


def _padded(n: int) -> int:
    return -(-n // PAD) * PAD


def comfyui_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Our diffusers-format Fibo 2.0 LoRA (``transformer.<m>.lora_A/B.weight``, scale folded into B) in ComfyUI's
    layout. Raises ValueError for anything that isn't a plain LoRA."""
    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for key, a in sd.items():
        if key.endswith(".lora_A.weight"):
            name = key[len("transformer."):-len(".lora_A.weight")]
            pairs[name] = (a, sd[key.replace(".lora_A.", ".lora_B.")])
        elif not key.endswith(".lora_B.weight"):
            raise ValueError(f"not a plain LoRA key: {key}")
    fused: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
    out: dict[str, torch.Tensor] = {}

    def put(cname, down, up):
        out[f"diffusion_model.{cname}.lora_down.weight"] = down.contiguous()
        out[f"diffusion_model.{cname}.lora_up.weight"] = up.contiguous()
        out[f"diffusion_model.{cname}.alpha"] = torch.tensor(float(down.shape[0]), dtype=down.dtype)

    for name, (a, b) in pairs.items():
        cname, slot, n = comfy_name(name)
        if slot is not None:
            fused.setdefault(cname, {})[slot] = (a, b)
        elif re.search(r"\.(?:feed_forward|ffn)\.w2$", name):  # input is the padded hidden width
            put(cname, torch.nn.functional.pad(a, (0, _padded(a.shape[1]) - a.shape[1])), b)
        else:
            put(cname, a, b)
    for cname, slots in fused.items():
        n = 2 if cname.endswith((".to_kv", ".w13")) else 3
        rows = next(iter(slots.values()))[1].shape[0]
        if cname.endswith(".w13"):
            rows = _padded(rows)
        ref = next(iter(slots.values()))[0]
        down = torch.cat([slots[s][0] for s in sorted(slots)], 0)
        up = torch.zeros(n * rows, down.shape[0], dtype=ref.dtype)
        col = 0
        for s in sorted(slots):
            a, b = slots[s]
            up[s * rows:s * rows + b.shape[0], col:col + a.shape[0]] = b
            col += a.shape[0]
        put(cname, down, up)
    return out
