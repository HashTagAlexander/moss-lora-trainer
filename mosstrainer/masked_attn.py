"""Attention under a full token-to-token mask, for Krea 2 regional prompts and isolated reference images.

torch's fused attention is used when it can be trusted: only the kernels that take a mask are allowed (some builds'
cuDNN kernel accepts one and ignores it), and a tiny probe checks the result against the exact tiled version below.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_FUSED_OK: dict[tuple, bool] = {}


def tiled_attention(q, k, v, allow, tile: int = 1024):
    """Exact masked attention, a slice of queries at a time. q, k, v: [B, H, L, D]; allow: bool, [L, L] or
    anything that broadcasts to [B, H, L, L] (True = may attend)."""
    length, scale = q.shape[2], q.shape[-1] ** -0.5
    kt = k.transpose(-1, -2)
    out = torch.empty_like(q)
    low = torch.finfo(q.dtype).min
    for s in range(0, length, tile):
        scores = (q[:, :, s:s + tile] @ kt) * scale
        rows = allow[..., s:s + tile, :] if allow.shape[-2] == length else allow
        probs = torch.softmax(scores.masked_fill(~rows, low), dim=-1, dtype=torch.float32)
        out[:, :, s:s + tile] = probs.to(v.dtype) @ v
    return out


def _fused(q, k, v, allow):
    if q.device.type != "cuda":
        return F.scaled_dot_product_attention(q, k, v, attn_mask=allow)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=allow)


def _fused_works(device, dtype) -> bool:
    """True when fused attention on this device applies masks: an only-yourself mask must match the tiled result."""
    try:
        g = torch.Generator().manual_seed(0)
        q, k, v = (torch.randn(1, 2, 16, 8, generator=g).to(device, dtype) for _ in range(3))
        eye = torch.eye(16, dtype=torch.bool, device=device)
        diff = (_fused(q, k, v, eye).float() - tiled_attention(q, k, v, eye).float()).abs().max().item()
        return diff < (1e-3 if dtype == torch.float32 else 0.05)
    except Exception:  # noqa: BLE001
        return False


def masked_attention(q, k, v, allow):
    """Attention with a full mask. q, k, v: [B, H, L, D]; allow: bool [..., L, L], True = may attend."""
    key = (str(q.device), q.dtype)
    if key not in _FUSED_OK:
        _FUSED_OK[key] = _fused_works(q.device, q.dtype)
    if _FUSED_OK[key]:
        try:
            return _fused(q, k, v, allow)
        except RuntimeError:  # no kernel for this size, or out of memory: the tiled version needs less
            _FUSED_OK[key] = False
            if q.device.type == "cuda":
                torch.cuda.empty_cache()
    return tiled_attention(q, k, v, allow)


def install(model) -> None:
    """Give a Krea2Transformer2DModel's blocks a processor that sends full masks through `masked_attention`.
    Key-padding masks ([B, 1, 1, L]) and no mask keep diffusers' own path, so training is untouched."""
    from diffusers.models.transformers import transformer_krea2 as k2

    class Krea2MaskedAttnProcessor(k2.Krea2AttnProcessor):
        def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
            if attention_mask is None or attention_mask.shape[-2] == 1:
                return super().__call__(attn, hidden_states, attention_mask, image_rotary_emb)
            query = attn.norm_q(attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim)))
            key = attn.norm_k(attn.to_k(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim)))
            value = attn.to_v(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
            gate = attn.to_gate(hidden_states)
            if image_rotary_emb is not None:
                query = k2.apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = k2.apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
            groups = attn.num_heads // attn.num_kv_heads
            if groups > 1:
                key, value = key.repeat_interleave(groups, dim=2), value.repeat_interleave(groups, dim=2)
            out = masked_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attention_mask)
            out = out.transpose(1, 2).flatten(2, 3)
            return attn.to_out[0](out * torch.sigmoid(gate))

    for block in model.transformer_blocks:
        if type(block.attn.processor).__name__ != "Krea2MaskedAttnProcessor":
            block.attn.set_processor(Krea2MaskedAttnProcessor())
