"""ComfyUI's int8 "ConvRot" weights, run as they are stored (8 bits per weight in memory, not expanded to bf16).

The format (comfy-kitchen's `int8_tensorwise` with `convrot`, as in Comfy-Org's `*_int8_convrot.safetensors`):

    <layer>.weight         int8 [out, in]   the weight, rotated along its input side and rounded
    <layer>.weight_scale   f32  [out, 1]    one scale per output row
    <layer>.comfy_quant    uint8 JSON       {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}

The rotation is a fixed orthogonal, symmetric Hadamard matrix H applied to every `groupsize`-wide block of the input
side, so  x W^T = (x H)(W H)^T : the layer rotates its *input* the same way and multiplies by the stored integers.
No rotation matrix is in the file; H is H4 (x) H4 (x) H4 (x) H4 / 16 for a group of 256.

`load_qwen3vl_int8` loads Comfy-Org's int8 Qwen3-VL (MiniMax H3's text encoder) into transformers' own model class.
"""
from __future__ import annotations

import copy
import json

import torch
import torch.nn.functional as F
from torch import nn

_H4 = ((1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1))


def regular_hadamard(n: int) -> torch.Tensor:
    """The n x n regular Hadamard matrix comfy-kitchen rotates with (n a power of 4), scaled to be orthogonal."""
    base = torch.tensor(_H4, dtype=torch.float32)
    h = base
    while h.shape[0] < n:
        h = torch.kron(h, base)
    if h.shape[0] != n:
        raise ValueError(f"ConvRot group size must be a power of 4, got {n}")
    return h / n ** 0.5


class Int8Linear(nn.Module):
    """A linear layer over stored int8 weights. `groupsize` 0 means the weights were not rotated."""

    def __init__(self, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor | None = None,
                 groupsize: int = 0):
        super().__init__()
        if weight.dtype != torch.int8:
            raise TypeError(f"expected int8 weights, got {weight.dtype}")
        self.out_features, self.in_features = weight.shape
        self.groupsize = int(groupsize)
        if self.groupsize and self.in_features % self.groupsize:
            raise ValueError(f"{self.in_features} inputs can't be split into groups of {self.groupsize}")
        self.register_buffer("weight", weight)
        self.register_buffer("weight_scale", weight_scale.reshape(-1).float())
        self.register_buffer("bias", None if bias is None else bias.float())
        self.register_buffer("hadamard", regular_hadamard(self.groupsize) if self.groupsize else None, persistent=False)

    def _apply(self, fn, recurse=True):
        # moves follow the model (`.to(device)`), casts don't: the integers must stay int8 and a `.to(bfloat16)` meant
        # for the rest of the model would round the scales to 8 bits
        for name, tensor in list(self._buffers.items()):
            if tensor is not None:
                self._buffers[name] = fn(tensor).to(tensor.dtype)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        if self.groupsize:
            h = (h.unflatten(-1, (-1, self.groupsize)) @ self.hadamard).flatten(-2)
        # the integers are exact in bfloat16, so in a bfloat16 model the big product runs at the model's own
        # precision; only the rotation and the per-row scale need float32
        work = torch.bfloat16 if x.dtype == torch.bfloat16 else torch.float32
        y = F.linear(h.to(work), self.weight.to(work)).float() * self.weight_scale
        if self.bias is not None:
            y = y + self.bias
        return y.to(x.dtype)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, int8, groupsize={self.groupsize}"


def _set_module(root: nn.Module, name: str, module: nn.Module) -> None:
    parent, _, leaf = name.rpartition(".")
    setattr(root.get_submodule(parent) if parent else root, leaf, module)


def load_qwen3vl_int8(path, config, device="cpu", dtype=torch.bfloat16):
    """Comfy-Org's int8 ConvRot Qwen3-VL as a `Qwen3VLForConditionalGeneration` MiniMax H3's pipeline can call.

    `config`: the full model's `Qwen3VLConfig` (MiniMaxAI/MiniMax-H3 `text_encoder/config.json`). The Comfy file holds
    only what H3 uses: the vision tower, the token embeddings and the first 50 decoder layers (H3 conditions on the
    hidden state after layer 50, before the final norm), with the 350 layer projections as int8. So the model is built
    with those 50 layers and no final norm, which makes `hidden_states[50]` that state; the config keeps saying 51
    layers because diffusers refuses a stack that ends at the layer it reads (its last hidden state is normally
    post-norm). The language-model head is dropped: H3 never calls it.
    """
    import accelerate
    from safetensors import safe_open
    from transformers import Qwen3VLForConditionalGeneration

    with safe_open(str(path), "pt", device="cpu") as f:
        keys = list(f.keys())
        layers = 1 + max(int(k.split(".")[2]) for k in keys if k.startswith("model.layers."))
        cfg = copy.deepcopy(config)
        cfg.text_config.num_hidden_layers = layers
        with accelerate.init_empty_weights():
            model = Qwen3VLForConditionalGeneration(cfg)
        model.lm_head = nn.Identity()
        model.model.language_model.norm = nn.Identity()
        model.config.text_config.num_hidden_layers = layers + 1

        def target(key: str) -> str:
            if key.startswith("visual."):
                return "model." + key
            return "model.language_model." + key[len("model."):]

        quantized = {k[:-len(".comfy_quant")] for k in keys if k.endswith(".comfy_quant")}
        state, used = {}, set()
        for base in sorted(quantized):
            spec = json.loads(bytes(f.get_tensor(base + ".comfy_quant").tolist()).decode())
            if spec.get("format") != "int8_tensorwise":
                raise ValueError(f"{base}: unsupported quantization {spec.get('format')!r} (expected int8_tensorwise)")
            bias = f.get_tensor(base + ".bias") if base + ".bias" in keys else None
            layer = Int8Linear(f.get_tensor(base + ".weight"), f.get_tensor(base + ".weight_scale"), bias,
                               int(spec.get("convrot_groupsize", 0)) if spec.get("convrot") else 0)
            _set_module(model, target(base), layer)
            used.update(base + s for s in (".comfy_quant", ".weight", ".weight_scale", ".bias"))
        for key in keys:
            if key not in used:
                state[target(key)] = f.get_tensor(key).to(dtype)
    expected = {k for k in model.state_dict() if not isinstance(model.get_submodule(k.rpartition(".")[0]), Int8Linear)}
    missing, unexpected = expected - set(state), set(state) - expected
    if missing or unexpected:
        raise ValueError(f"{path}: doesn't match the text encoder's layout (missing {sorted(missing)[:4]}, "
                         f"unexpected {sorted(unexpected)[:4]})")
    model.load_state_dict(state, strict=False, assign=True)
    return model.requires_grad_(False).eval().to(device)
