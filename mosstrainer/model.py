"""MOSS-Image components for training.

The transformer and scheduler are adapted from OpenMOSS-Team/MOSS-Image-Base's
``pipeline_moss_image.py`` (Apache-2.0). Changes vs. the reference:
  * optional per-block gradient checkpointing,
  * helpers for latent (de)normalisation and text encoding without the pipeline.
State-dict keys are identical, so the official weights load unchanged.
"""
from __future__ import annotations

import gc
import math
from pathlib import Path

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

DEFAULT_MODEL = "OpenMOSS-Team/MOSS-Image-Base"


# --------------------------------------------------------------------------- transformer

class _RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def _situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _attention(q, k, v, rope, heads):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rotate(q.reshape(shape), rope).transpose(1, 2)
    k = _rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    return F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v).transpose(1, 2).flatten(2)


class _TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))


class _TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([_TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope):
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope)
        return x


class _Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.proj = nn.ModuleDict({
            name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")
        })
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope, modulation):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        dtype = self.proj["q"].weight.dtype
        y = y.to(dtype)
        q = self.q_norm(self.proj["q"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj["k"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj["o"](_attention(q, k, self.proj["v"](y), rope, self.heads)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
        return x + self.proj["ffn_down"](_situ(hidden, gate)) * gate_f[:, None]


class _RoPE(nn.Module):
    def __init__(self, head_dim, base):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        self.register_buffer("bands", self._default_bands(torch.device("cpu")))

    def _default_bands(self, device):
        exponent = torch.arange(self.num_bands, device="cpu", dtype=torch.float32) / self.num_bands
        return (1 / (self.base ** exponent)).to(device)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.bands = self._default_bands(self.bands.device)
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.bands = self._default_bands(self.bands.device)

    def forward(self, height, width, text_length, device):
        if self.bands.dtype != torch.float32:
            self.bands = self._default_bands(device)
        h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).float()
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        phase = torch.cat((image, text), 0).unsqueeze(-1) * self.bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


class MOSSImageTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self, dim=3072, ffn_dim=8192, num_heads=24, num_layers=42,
                 text_dim=4096, freq_dim=256, latent_channels=128, rope_base=10000.0):
        super().__init__()
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = _TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = _RoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([_Block(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = _RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)
        self.grad_ckpt = False

    def set_gradient_checkpointing(self, enabled: bool):
        self.grad_ckpt = enabled

    def forward(self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask=None, skip_layers=()):
        skip_layers = frozenset(skip_layers or ())
        batch, _, height, width = hidden_states.shape
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        times = torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32).reshape(-1)
        times = times.expand(batch)
        half = self.config.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
        mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(batch, 6, -1)
        head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(batch, 2, -1)
        use_ckpt = self.grad_ckpt and torch.is_grad_enabled()
        outputs = []
        # Each sample runs at its true text length, exactly like the reference pipeline.
        for index in range(batch):
            context = encoder_hidden_states[index:index + 1, mask[index].bool()]
            video = hidden_states[index:index + 1].flatten(2).transpose(1, 2)
            video = self.video_embedder(video.to(self.video_embedder.weight.dtype).contiguous())
            rope = self.rope(height, width, context.shape[1], hidden_states.device).to(video.dtype)
            text = self.text_embedder(context, rope[:, height * width:])
            x = torch.cat((video, text), 1)
            m = mod[index:index + 1]
            for layer_index, layer in enumerate(self.layers):
                if layer_index in skip_layers:
                    continue
                if use_ckpt:
                    x = checkpoint(layer, x, rope, m, use_reentrant=False)
                else:
                    x = layer(x, rope, m)
            x = self.final_norm_video(x[:, :height * width].to(self.final_norm_video.weight.dtype))
            x = x * (1 + head[index:index + 1, 1, None]) + head[index:index + 1, 0, None]
            x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
            outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
        return torch.cat(outputs)


# --------------------------------------------------------------------------- scheduler

class MOSSImageScheduler(FlowMatchEulerDiscreteScheduler):
    """Euler integration on the training project's nearest-grid schedule."""

    @register_to_config
    def __init__(self, num_train_timesteps=1000, shift=5.0, training_shift=1.0):
        super().__init__(num_train_timesteps=num_train_timesteps, shift=1.0)
        self.register_to_config(shift=shift, training_shift=training_shift)

    def set_timesteps(self, num_inference_steps=50, device=None):
        total = self.config.num_train_timesteps

        def grid(count, shift):
            lower = shift / (total + shift - 1)
            values = torch.linspace(1, lower, count)
            return shift * values / (1 + (shift - 1) * values)

        sigmas = grid(num_inference_steps, self.config.shift)
        training = grid(total, self.config.training_shift).flip(0)
        upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
        lower = (upper - 1).clamp(min=0)
        nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
        sigmas = torch.unique_consecutive(training[nearest])
        self._shift = 1.0
        super().set_timesteps(device=device, sigmas=sigmas.tolist())


# --------------------------------------------------------------------------- loading helpers

def resolve_model(model: str) -> str:
    """Return a local directory for a Hub id or local path (downloads if needed)."""
    path = Path(model).expanduser()
    if path.is_dir():
        return str(path.resolve())
    from huggingface_hub import snapshot_download
    return snapshot_download(model, allow_patterns=[
        "model_index.json", "*.py", "scheduler/*", "text_encoder/*", "tokenizer/*",
        "transformer/*", "vae/*",
    ])


def load_transformer(root: str, dtype=torch.bfloat16) -> MOSSImageTransformer:
    model = MOSSImageTransformer.from_pretrained(root, subfolder="transformer", torch_dtype=dtype)
    model.requires_grad_(False)
    return model


def load_vae(root: str, dtype=torch.float32):
    from diffusers import AutoencoderKLFlux2
    vae = AutoencoderKLFlux2.from_pretrained(root, subfolder="vae", torch_dtype=dtype)
    vae.requires_grad_(False).eval()
    return vae


def _bn_stats(vae, ref):
    bn = vae.bn
    std = (bn.running_var + vae.config.batch_norm_eps).sqrt().to(ref).view(1, -1, 1, 1)
    mean = bn.running_mean.to(ref).view(1, -1, 1, 1)
    return mean, std


@torch.no_grad()
def encode_images(vae, pixels: torch.Tensor) -> torch.Tensor:
    """pixels in [-1, 1], [B,3,H,W] -> normalised, patchified latents [B,128,H/16,W/16]."""
    out = vae.encode(pixels.to(device=vae.device, dtype=vae.dtype))
    z = out.latent_dist.mode() if hasattr(out, "latent_dist") else out[0].mode()
    b, c, h, w = z.shape
    z = z.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
    mean, std = _bn_stats(vae, z)
    return ((z - mean) / std).float()


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """Inverse of encode_images; returns pixels in [-1, 1]."""
    z = latents.to(device=vae.device, dtype=vae.dtype)
    mean, std = _bn_stats(vae, z)
    z = z * std + mean
    b, c, h, w = z.shape
    z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
    return vae.decode(z, return_dict=False)[0].clamp(-1, 1)


class TextEncoder:
    """Qwen3.5-9B prompt encoder, formatted exactly like MOSSImagePipeline.encode_prompt."""

    def __init__(self, root: str, device, dtype=torch.bfloat16):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(root, subfolder="tokenizer")
        self.model = AutoModel.from_pretrained(root, subfolder="text_encoder", dtype=dtype).to(device).eval()
        self.model.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def encode(self, prompt: str, max_length: int = 4096) -> torch.Tensor:
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=True)
        inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
        out = self.model(**inputs, use_cache=False, return_dict=True)
        return out.last_hidden_state[0].to(torch.bfloat16).cpu()

    def unload(self):
        del self.model
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def pad_embeds(rows: list[torch.Tensor], device=None):
    length = max(r.shape[0] for r in rows)
    embeds = rows[0].new_zeros(len(rows), length, *rows[0].shape[1:])
    mask = torch.zeros(len(rows), length, dtype=torch.bool)
    for i, r in enumerate(rows):
        embeds[i, :r.shape[0]] = r
        mask[i, :r.shape[0]] = True
    if device is not None:
        embeds, mask = embeds.to(device), mask.to(device)
    return embeds, mask


# --------------------------------------------------------------------------- sampling

@torch.no_grad()
def sample(transformer, scheduler, vae, embeds, neg_embeds, *, width, height, steps=30, cfg=3.0,
           seed=42, uncond_skip_layers=(22,), device="cuda", progress=None):
    """Generate one image with cached prompt embeddings. Returns a PIL image."""
    from PIL import Image
    pos, pos_mask = pad_embeds([embeds], device)
    neg, neg_mask = pad_embeds([neg_embeds], device)
    gen = torch.Generator("cpu").manual_seed(seed)
    latents = torch.randn((1, transformer.config.latent_channels, height // 16, width // 16),
                          generator=gen, dtype=torch.float32).to(device)
    scheduler.set_timesteps(steps, device=device)
    timesteps = scheduler.timesteps
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
        for i, t in enumerate(timesteps):
            pred = transformer(latents, t, pos, pos_mask).float()
            if cfg != 1.0:
                uncond = transformer(latents, t, neg, neg_mask, skip_layers=uncond_skip_layers).float()
                pred = uncond + cfg * (pred - uncond)
            latents = scheduler.step(pred, t, latents, return_dict=False)[0]
            if progress:
                progress(i + 1, len(timesteps))
    pixels = decode_latents(vae, latents)[0]
    arr = ((pixels.float().permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


def snap_size(width: int, height: int, multiple: int = 16):
    return max(multiple, int(width) // multiple * multiple), max(multiple, int(height) // multiple * multiple)


def bucket_sizes(resolution: int, multiple: int = 16, min_ratio=1 / 3, max_ratio=3.0, step=64):
    """Aspect buckets with area ~= resolution**2 and dims divisible by `multiple`."""
    area = resolution * resolution
    sizes = set()
    w = step
    while w <= resolution * 3:
        h = int(round(area / w / multiple)) * multiple
        if h >= multiple and min_ratio <= w / h <= max_ratio:
            sizes.add((w, h))
            sizes.add((h, w))  # keep portrait and landscape buckets symmetric
        w += step
    sizes.add((resolution // multiple * multiple,) * 2)
    return sorted(sizes)


def closest_bucket(width: int, height: int, buckets):
    ratio = math.log(width / height)
    return min(buckets, key=lambda b: abs(math.log(b[0] / b[1]) - ratio))
