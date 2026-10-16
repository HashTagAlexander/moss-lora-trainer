"""Minimal LoRA for MOSS-Image, saved in a ComfyUI-loadable format.

Keys look like ``diffusion_model.layers.0.proj.q.lora_A.weight`` / ``lora_B.weight`` / ``alpha``,
which ComfyUI's generic LoRA loader maps onto the MOSS-Image diffusion model.
"""
from __future__ import annotations

import json
import math
import re

import torch
from safetensors.torch import load_file, save_file
from torch import nn

PREFIX = "diffusion_model."

# Target groups -> regexes over module names in MOSSImageTransformer.
TARGETS = {
    "attn": r"^layers\.\d+\.proj\.(q|k|v|o)$",
    "ffn": r"^layers\.\d+\.proj\.(ffn_up_gate|ffn_down)$",
    "text": r"^text_embedder\.blocks\.\d+\.(qkv|proj|ffn_gate|ffn_up|ffn_down)$",
}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.multiplier = 1.0
        self.lora_A = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device, dtype=torch.float32)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def weight(self):  # the transformer reads `.weight.dtype` for casting
        return self.base.weight

    def forward(self, x):
        out = self.base(x)
        if self.multiplier == 0:
            return out
        delta = self.lora_B(self.lora_A(self.dropout(x).to(self.lora_A.weight.dtype)))
        return out + (delta * (self.scale * self.multiplier)).to(out.dtype)


def _layer_index(name):
    m = re.match(r"^layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def inject(model: nn.Module, rank: int, alpha: float, targets=("attn", "ffn"), dropout=0.0,
           layers: list[int] | None = None) -> dict[str, LoRALinear]:
    patterns = [re.compile(TARGETS[t]) for t in targets]
    found = {}
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or not any(p.match(name) for p in patterns):
            continue
        idx = _layer_index(name)
        if layers is not None and idx is not None and idx not in layers:
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        wrapped = LoRALinear(module, rank, alpha, dropout)
        if isinstance(parent, nn.ModuleDict):
            parent[child] = wrapped
        else:
            setattr(parent, child, wrapped)
        found[name] = wrapped
    if not found:
        raise ValueError(f"No LoRA targets matched {targets}")
    return found


def set_multiplier(modules: dict[str, LoRALinear], value: float):
    for m in modules.values():
        m.multiplier = value


def parameters(modules: dict[str, LoRALinear]):
    for m in modules.values():
        yield m.lora_A.weight
        yield m.lora_B.weight


def state_dict(modules: dict[str, LoRALinear], dtype=torch.bfloat16) -> dict[str, torch.Tensor]:
    sd = {}
    for name, m in modules.items():
        sd[f"{PREFIX}{name}.lora_A.weight"] = m.lora_A.weight.detach().to("cpu", dtype).contiguous()
        sd[f"{PREFIX}{name}.lora_B.weight"] = m.lora_B.weight.detach().to("cpu", dtype).contiguous()
        sd[f"{PREFIX}{name}.alpha"] = torch.tensor(float(m.alpha))
    return sd


def save(modules, path, metadata: dict | None = None, dtype=torch.bfloat16):
    meta = {"format": "pt", "modelspec.architecture": "moss-image/lora"}
    for k, v in (metadata or {}).items():
        meta[k] = v if isinstance(v, str) else json.dumps(v)
    save_file(state_dict(modules, dtype), str(path), metadata=meta)


def load_into(modules: dict[str, LoRALinear], path):
    sd = load_file(str(path))
    missing = []
    for name, m in modules.items():
        a, b = sd.get(f"{PREFIX}{name}.lora_A.weight"), sd.get(f"{PREFIX}{name}.lora_B.weight")
        if a is None or b is None:
            missing.append(name)
            continue
        m.lora_A.weight.data.copy_(a.float())
        m.lora_B.weight.data.copy_(b.float())
    return missing


@torch.no_grad()
def merge_file_into(model: nn.Module, path, multiplier: float = 1.0) -> int:
    """Merge a saved LoRA file directly into a (plain, un-injected) transformer's weights."""
    sd = load_file(str(path))
    merged = 0
    for key in sd:
        if not key.endswith(".lora_A.weight"):
            continue
        name = key[len(PREFIX):-len(".lora_A.weight")] if key.startswith(PREFIX) else key[:-len(".lora_A.weight")]
        a = sd[key].float()
        b = sd[key.replace("lora_A", "lora_B")].float()
        alpha = float(sd.get(key.replace("lora_A.weight", "alpha"), torch.tensor(a.shape[0])))
        linear = model.get_submodule(name)
        delta = (b @ a) * (alpha / a.shape[0]) * multiplier
        linear.weight.add_(delta.to(linear.weight.device, linear.weight.dtype))
        merged += 1
    return merged
