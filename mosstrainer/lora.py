"""Minimal LoRA and LoKr for MOSS-Image, Krea 2 and Fibo 2.0, saved in ComfyUI/diffusers-loadable formats.

MOSS-Image ("moss" format): ``diffusion_model.layers.0.proj.q.lora_A.weight`` / ``lora_B.weight`` / ``alpha``,
which ComfyUI's generic LoRA loader maps onto the MOSS-Image diffusion model.
Krea 2 / Fibo 2.0 ("diffusers" format): ``transformer.<module>.lora_A.weight`` / ``lora_B.weight`` with alpha/rank
folded into lora_B, loadable by the diffusers pipelines (``pipe.transformer.load_lora_adapter``) and ComfyUI.
LoKr (LyCORIS' Kronecker adapter, delta W = kron(w1, w2), both full matrices so the scale is 1): ``<prefix><module>
.lokr_w1`` / ``.lokr_w2`` / ``.alpha`` under the same prefixes, the format ComfyUI and LyCORIS load.
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


    def adapter_params(self) -> dict[str, torch.Tensor]:
        return {"A": self.lora_A.weight, "B": self.lora_B.weight}


def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """LyCORIS' split of a dimension into (m, n), m <= n, m * n = dimension: m = factor when it divides the
    dimension, otherwise the most square split with m <= factor (-1 = as square as possible)."""
    if factor > 0 and dimension % factor == 0:
        m, n = factor, dimension // factor
        return (n, m) if m > n else (m, n)
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > m + n or new_m > factor:
            break
        m, n = new_m, new_n
    return (n, m) if m > n else (m, n)


def kron_apply(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """x @ kron(w1, w2).T without building the full matrix: (X @ w2.T) then w1 @ it, per row of x."""
    (out_m, in_m), (out_n, in_n) = w1.shape, w2.shape
    xs = x.reshape(*x.shape[:-1], in_m, in_n)
    y = torch.matmul(w1, torch.matmul(xs, w2.t()))  # (..., out_m, out_n)
    return y.reshape(*x.shape[:-1], out_m * out_n)


class LoKrLinear(nn.Module):
    """LoKr adapter: delta W = kron(w1, w2) with w1 (out_m x in_m) small and w2 (out_n x in_n) full, as LyCORIS
    makes it with a full second matrix. w1 starts random and w2 at zero, so training starts from the base model."""

    def __init__(self, base: nn.Linear, factor: int = 8):
        super().__init__()
        self.base = base
        self.multiplier = 1.0
        out_m, out_n = factorization(base.out_features, factor)
        in_m, in_n = factorization(base.in_features, factor)
        dev = base.weight.device
        # in a child module registered after `base`, so the model's first parameter stays the base weight (some
        # forwards read their compute dtype off it), as with LoRALinear's lora_A / lora_B
        self.lokr = nn.Module()
        self.lokr.w1 = nn.Parameter(torch.empty(out_m, in_m, device=dev, dtype=torch.float32))
        self.lokr.w2 = nn.Parameter(torch.zeros(out_n, in_n, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lokr.w1, a=math.sqrt(5))

    @property
    def lokr_w1(self):
        return self.lokr.w1

    @property
    def lokr_w2(self):
        return self.lokr.w2

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        out = self.base(x)
        if self.multiplier == 0:
            return out
        delta = kron_apply(x.to(torch.float32), self.lokr_w1, self.lokr_w2)
        return out + (delta * self.multiplier).to(out.dtype)

    def adapter_params(self) -> dict[str, torch.Tensor]:
        return {"w1": self.lokr_w1, "w2": self.lokr_w2}


def _layer_index(name, layer_re=r"^layers\.(\d+)\."):
    m = re.match(layer_re, name)
    return int(m.group(1)) if m else None


def _replace(model: nn.Module, name: str, new: nn.Module):
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    if isinstance(parent, nn.ModuleDict):
        parent[child] = new
    elif isinstance(parent, (nn.ModuleList, nn.Sequential)) and child.isdigit():
        parent[int(child)] = new
    else:
        setattr(parent, child, new)


def inject(model: nn.Module, rank: int, alpha: float, targets=("attn", "ffn"), dropout=0.0,
           layers: list[int] | None = None, target_map: dict[str, str] | None = None,
           layer_re: str = r"^layers\.(\d+)\.", kind: str = "lora", factor: int = 8) -> dict[str, nn.Module]:
    """Wrap the matching Linears with a LoRA (rank, alpha, dropout) or a LoKr (factor) adapter."""
    target_map = target_map or TARGETS
    patterns = [re.compile(target_map[t]) for t in targets]
    found = {}
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or not any(p.match(name) for p in patterns):
            continue
        idx = _layer_index(name, layer_re)
        if layers is not None and idx is not None and idx not in layers:
            continue
        wrapped = LoKrLinear(module, factor) if kind == "lokr" else LoRALinear(module, rank, alpha, dropout)
        _replace(model, name, wrapped)
        found[name] = wrapped
    if not found:
        raise ValueError(f"No LoRA targets matched {targets}")
    return found


def set_multiplier(modules: dict[str, LoRALinear], value: float):
    for m in modules.values():
        m.multiplier = value


def parameters(modules: dict[str, nn.Module]):
    for m in modules.values():
        yield from m.adapter_params().values()


def state_dict(modules: dict[str, nn.Module], dtype=torch.bfloat16, fmt: str = "moss") -> dict[str, torch.Tensor]:
    sd = {}
    for name, m in modules.items():
        if isinstance(m, LoKrLinear):
            prefix = "transformer." if fmt == "diffusers" else PREFIX
            sd[f"{prefix}{name}.lokr_w1"] = m.lokr_w1.detach().to("cpu", dtype).contiguous()
            sd[f"{prefix}{name}.lokr_w2"] = m.lokr_w2.detach().to("cpu", dtype).contiguous()
            sd[f"{prefix}{name}.alpha"] = torch.tensor(1.0)  # full matrices: loaders use scale 1 regardless
            continue
        a = m.lora_A.weight.detach().float()
        b = m.lora_B.weight.detach().float()
        if fmt == "diffusers":  # no alpha keys: fold alpha/rank into B so every loader gets the same delta
            sd[f"transformer.{name}.lora_A.weight"] = a.to("cpu", dtype).contiguous()
            sd[f"transformer.{name}.lora_B.weight"] = (b * m.scale).to("cpu", dtype).contiguous()
        else:
            sd[f"{PREFIX}{name}.lora_A.weight"] = a.to("cpu", dtype).contiguous()
            sd[f"{PREFIX}{name}.lora_B.weight"] = b.to("cpu", dtype).contiguous()
            sd[f"{PREFIX}{name}.alpha"] = torch.tensor(float(m.alpha))
    return sd


def save(modules, path, metadata: dict | None = None, dtype=torch.bfloat16, fmt: str = "moss", arch: str = ""):
    """Write the adapter file; returns its state dict (for other layouts of the same weights)."""
    arch = arch or ("krea2/lora" if fmt == "diffusers" else "moss-image/lora")
    if any(isinstance(m, LoKrLinear) for m in modules.values()):
        arch = arch.replace("/lora", "/lokr")
    meta = {"format": "pt", "modelspec.architecture": arch}
    for k, v in (metadata or {}).items():
        meta[k] = v if isinstance(v, str) else json.dumps(v)
    sd = state_dict(modules, dtype, fmt)
    save_file(sd, str(path), metadata=meta)
    return sd


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
    merged = 0
    for name, adapter in read_lora_file(path).items():
        linear = model.get_submodule(name)
        kind, x1, x2, scale = adapter
        delta = (torch.kron(x1.float(), x2.float()) if kind == "lokr" else x2.float() @ x1.float()) * scale * multiplier
        linear.weight.add_(delta.to(linear.weight.device, linear.weight.dtype))
        merged += 1
    return merged


# --------------------------------------------------------------------------- inference-time stacking

def _peft_alpha(path) -> float | None:
    """lora_alpha from diffusers/PEFT adapter metadata (e.g. Bria's Fibo 2.0 trainer), which has no alpha keys."""
    from safetensors import safe_open
    with safe_open(str(path), "pt") as f:
        raw = (f.metadata() or {}).get("lora_adapter_metadata")
    try:
        meta = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return None
    alpha = meta.get("transformer.lora_alpha", meta.get("lora_alpha"))
    return float(alpha) if alpha else None


def _strip_prefix(name: str) -> str:
    for prefix in ("model.diffusion_model.", PREFIX, "transformer.", "base_model.model."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def read_lora_file(path) -> dict[str, tuple[str, torch.Tensor, torch.Tensor, float]]:
    """Return {module_name: ("lora", A, B, scale) | ("lokr", w1, w2, scale)} for lora_A/lora_B, lora_down/lora_up,
    bare A/B (Krea 2 control-LoRAs) and LyCORIS LoKr files (full or low-rank factors)."""
    sd = load_file(str(path))
    peft_alpha = _peft_alpha(path)
    out = {}
    for key in sd:
        if key.endswith((".lokr_w1", ".lokr_w1_a")):
            name = key.rsplit(".lokr_w1", 1)[0]
            if name in out or _strip_prefix(name) in out:
                continue
            w1 = sd.get(name + ".lokr_w1")
            if w1 is None:
                w1 = sd[name + ".lokr_w1_a"] @ sd[name + ".lokr_w1_b"]
            w2, dim = sd.get(name + ".lokr_w2"), None
            if w2 is None:
                if name + ".lokr_w2_a" not in sd or name + ".lokr_t2" in sd:
                    continue  # Tucker-decomposed LoKr isn't supported here
                w2, dim = sd[name + ".lokr_w2_a"] @ sd[name + ".lokr_w2_b"], sd[name + ".lokr_w2_b"].shape[0]
            if name + ".lokr_w1_b" in sd:
                dim = sd[name + ".lokr_w1_b"].shape[0]
            alpha = sd.get(name + ".alpha")
            scale = float(alpha) / dim if alpha is not None and dim else 1.0  # ComfyUI's rule
            out[_strip_prefix(name)] = ("lokr", w1, w2, scale)
            continue
        for down, up in ((".lora_A.weight", ".lora_B.weight"), (".lora_down.weight", ".lora_up.weight"), (".A", ".B")):
            if not key.endswith(down):
                continue
            name = key[:-len(down)]
            b = sd.get(name + up)
            if b is None:
                continue
            a = sd[key]
            alpha = sd.get(name + ".alpha")
            rank = a.shape[0]
            if alpha is None and peft_alpha:
                alpha = peft_alpha
            scale = (float(alpha) / rank) if alpha is not None else 1.0
            out[_strip_prefix(name)] = ("lora", a, b, scale)
    return out


def read_control_input(path) -> torch.Tensor | None:
    """The control half of a control-LoRA's widened input layer: `first.weight` is [hidden, 2 * C] (the picture's
    channels, then the control image's); returns [hidden, C], or None for an ordinary LoRA file."""
    from safetensors import safe_open
    with safe_open(str(path), "pt", device="cpu") as f:
        for key in f.keys():
            if key.endswith(("first.weight", "img_in.weight")):
                weight = f.get_tensor(key)
                if weight.ndim == 2 and weight.shape[1] % 2 == 0:
                    return weight[:, weight.shape[1] // 2:].clone()
    return None


class DynamicLoRALinear(nn.Module):
    """Linear layer with any number of LoRA adapters applied on the fly (base weights untouched)."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.adapters: list[tuple[str, torch.Tensor, torch.Tensor, float]] = []

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        out = self.base(x)
        for kind, a, b, scale in self.adapters:
            if kind == "lokr":
                delta = kron_apply(x.to(a.dtype), a, b)
            else:
                delta = nn.functional.linear(nn.functional.linear(x.to(a.dtype), a), b)
            out = out + (delta * scale).to(out.dtype)
        return out


class LoRAStack:
    """Activates a set of LoRA files with strengths on a transformer; switching is instant and exact."""

    def __init__(self, model: nn.Module, max_cached: int = 8, name_map=None):
        self.model = model
        self.name_map = name_map or (lambda n: n)
        self.wrapped: dict[str, DynamicLoRALinear] = {}
        self.cache: dict[tuple, dict] = {}
        self.max_cached = max_cached

    def _wrap(self, name):
        if name in self.wrapped:
            return self.wrapped[name]
        try:
            module = self.model.get_submodule(name)
        except AttributeError:
            return None
        if isinstance(module, DynamicLoRALinear):
            return module
        if not isinstance(module, nn.Linear):
            return None
        wrapped = DynamicLoRALinear(module)
        _replace(self.model, name, wrapped)
        self.wrapped[name] = wrapped
        return wrapped

    def activate(self, specs, device, dtype=torch.bfloat16) -> dict[str, int]:
        """specs: iterable of (path, strength). Returns {path: number_of_unmatched_modules}."""
        for m in self.wrapped.values():
            m.adapters.clear()
        unmatched = {}
        for path, strength in specs:
            if not strength:
                continue
            from pathlib import Path
            p = Path(path)
            key = (str(p.resolve()), p.stat().st_mtime_ns)
            if key not in self.cache:
                if len(self.cache) >= self.max_cached:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[key] = {self.name_map(n): (k, a.to(device, dtype), b.to(device, dtype), s)
                                   for n, (k, a, b, s) in read_lora_file(p).items()}
            miss = 0
            for name, (kind, a, b, scale) in self.cache[key].items():
                mod = self._wrap(name)
                if mod is None:
                    miss += 1
                    continue
                mod.adapters.append((kind, a, b, scale * float(strength)))
            unmatched[str(path)] = miss
        return unmatched
