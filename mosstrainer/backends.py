"""Model-family backends: everything that differs between MOSS-Image and Krea 2 (Kroma).

Each backend provides text encoding, VAE (de)coding, transformer loading, the training forward pass,
sampling, and the LoRA naming/format. The trainer and the generation worker only talk to this API.
Latents are kept unpacked as [B, C, h, w] everywhere; packing happens inside `predict`.
"""
from __future__ import annotations

import gc
import math
import re

import torch
from torch import nn

from .config import model_spec


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _autocast(device):
    device = torch.device(device)
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


class Backend:
    family = ""
    multiple = 16                      # image sides must be divisible by this
    lora_targets: dict[str, str] = {}  # group -> regex over module names
    lora_layer_re = r""                # captures the block index used by the "Blocks" filter
    lora_format = ""                   # see lora.save
    latent_tag = ""                    # part of the latent-cache key
    text_cache_id = ""                 # part of the text-cache key

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.spec = model_spec(model_id)

    def tokens(self, latents: torch.Tensor) -> int:
        """Image token count for a latent batch [B, C, h, w] (used by resolution-aware shifts)."""
        raise NotImplementedError

    def lora_name_map(self, name: str) -> str:
        """Map module names found in third-party LoRA files onto this model's module names."""
        return name


# --------------------------------------------------------------------------- MOSS-Image

class MossBackend(Backend):
    family = "moss"
    lora_targets = {
        "attn": r"^layers\.\d+\.proj\.(q|k|v|o)$",
        "ffn": r"^layers\.\d+\.proj\.(ffn_up_gate|ffn_down)$",
        "text": r"^text_embedder\.blocks\.\d+\.(qkv|proj|ffn_gate|ffn_up|ffn_down)$",
    }
    lora_layer_re = r"^layers\.(\d+)\."
    lora_format = "moss"
    latent_tag = "lat-v1"

    def __init__(self, model_id):
        super().__init__(model_id)
        self.source = self.spec.get("path", model_id)
        self.text_cache_id = model_id
        self._root = None

    @property
    def root(self):
        if self._root is None:
            from .model import resolve_model
            self._root = resolve_model(self.source)
        return self._root

    def prefetch(self):
        return self.root

    def load_text_encoder(self, device):
        from .model import TextEncoder
        return TextEncoder(self.root, device)

    def load_vae(self, device):
        from .model import load_vae
        return load_vae(self.root).to(device)

    def encode_images(self, vae, pixels):
        from .model import encode_images
        return encode_images(vae, pixels)

    def load_transformer(self, device):
        from .model import load_transformer
        return load_transformer(self.root).to(device)

    def set_gradient_checkpointing(self, model, enabled):
        model.set_gradient_checkpointing(enabled)

    def tokens(self, latents):
        return latents.shape[-2] * latents.shape[-1]

    def predict(self, model, x, sigma, embeds, mask):
        return model(x, sigma * 1000, embeds, mask)

    def sample(self, model, vae, pos, neg, *, width, height, steps, cfg, seed, shift, skip_layer, device,
               progress=None):
        from .model import MOSSImageScheduler, sample
        return sample(model, MOSSImageScheduler(shift=shift if shift > 0 else 5.0), vae, pos, neg,
                      width=width, height=height, steps=steps, cfg=cfg, seed=seed,
                      uncond_skip_layers=(skip_layer,) if skip_layer >= 0 else (), device=device,
                      progress=progress)


# --------------------------------------------------------------------------- Krea 2 / Kroma

KREA2_PREFIX = ("<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, "
                "text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n")
KREA2_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
KREA2_TAPS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)


class Krea2TextEncoder:
    """Qwen3-VL-4B prompt encoder, laid out exactly like diffusers' Krea2Pipeline.get_text_hidden_states.

    The pipeline pads every prompt to 512 tokens and masks the padding everywhere downstream (text
    positions are all zero in RoPE), so the padding has no effect on the image. Only the valid tokens
    are returned (and cached): [L, 12, 2560].
    """
    start_idx, num_suffix, max_len = 34, 5, 512

    def __init__(self, source, device, dtype=torch.bfloat16):
        from transformers import AutoTokenizer, Qwen3VLModel
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.tokenizer.padding_side = "right"  # [prefix | prompt | PAD | suffix], as in the reference
        self.model = Qwen3VLModel.from_pretrained(source, dtype=dtype).to(device).eval()
        self.model.requires_grad_(False)
        self.device = torch.device(device)
        self.taps = KREA2_TAPS

    @torch.no_grad()
    def encode(self, prompt: str) -> torch.Tensor:
        tok = self.tokenizer
        text = tok([KREA2_PREFIX + prompt], truncation=True, padding="max_length",
                   max_length=self.max_len + self.start_idx - self.num_suffix, return_tensors="pt")
        suffix = tok([KREA2_SUFFIX], return_tensors="pt")
        ids = torch.cat([text.input_ids, suffix.input_ids], 1).to(self.device)
        mask = torch.cat([text.attention_mask, suffix.attention_mask], 1).bool().to(self.device)
        position_ids = (mask.long().cumsum(-1) - 1).clamp(min=0).unsqueeze(0).expand(3, -1, -1)
        out = self.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, output_hidden_states=True)
        hs = torch.stack([out.hidden_states[i] for i in self.taps], dim=2)[0, self.start_idx:]
        return hs[mask[0, self.start_idx:]].to(torch.bfloat16).cpu()

    def unload(self):
        self.model = None
        _free()


# Original (Krea / ComfyUI / AI-Toolkit) module names -> diffusers names, for loading foreign LoRAs.
_K2_ATTN = {"wq": "to_q", "wk": "to_k", "wv": "to_v", "wo": "to_out.0", "gate": "to_gate"}
_K2_FF = {"gate": "ff.gate", "up": "ff.up", "down": "ff.down"}
_K2_STANDALONE = {"first": "img_in", "last.linear": "final_layer.linear", "tmlp.0": "time_embed.linear_1",
                  "tmlp.2": "time_embed.linear_2", "tproj.1": "time_mod_proj", "txtmlp.1": "txt_in.linear_1",
                  "txtmlp.3": "txt_in.linear_2", "txtfusion.projector": "text_fusion.projector"}


class Krea2Backend(Backend):
    family = "krea2"
    lora_targets = {
        "attn": r"^transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v|to_gate|to_out\.0)$",
        "ffn": r"^transformer_blocks\.\d+\.ff\.(gate|up|down)$",
        "text": r"^text_fusion\.(layerwise_blocks|refiner_blocks)\.\d+\.(attn\.(to_q|to_k|to_v|to_gate|to_out\.0)|ff\.(gate|up|down))$",
    }
    lora_layer_re = r"^transformer_blocks\.(\d+)\."
    lora_format = "diffusers"
    latent_tag = "lat-krea2-v1"
    text_cache_id = "krea2-qwen3vl4b-v1"
    patch = 2
    vae_scale = 8

    # ---- downloads
    def _transformer_path(self):
        if self.spec.get("path"):
            return self.spec["path"]
        from huggingface_hub import hf_hub_download
        return hf_hub_download(self.spec["repo"], self.spec["file"])

    def _te_source(self):
        src = self.spec["text_encoder"]
        if "/" in src and not src.startswith(("/", ".")):
            from huggingface_hub import snapshot_download
            return snapshot_download(src, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"])
        return src

    def _vae_source(self):
        src = self.spec["vae"]
        if "/" in src and not src.startswith(("/", ".")):
            from huggingface_hub import snapshot_download
            return snapshot_download(src, allow_patterns=["vae/*"]) + "/vae"
        return src

    def prefetch(self):
        return self._transformer_path(), self._te_source(), self._vae_source()

    # ---- components
    def load_text_encoder(self, device):
        return Krea2TextEncoder(self._te_source(), device)

    def load_vae(self, device):
        from diffusers import AutoencoderKLQwenImage
        vae = AutoencoderKLQwenImage.from_pretrained(self._vae_source(), torch_dtype=torch.float32)
        return vae.requires_grad_(False).eval().to(device)

    def _vae_stats(self, vae, ref):
        mean = torch.tensor(vae.config.latents_mean, device=ref.device, dtype=ref.dtype).view(1, -1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device=ref.device, dtype=ref.dtype).view(1, -1, 1, 1)
        return mean, std

    @torch.no_grad()
    def encode_images(self, vae, pixels):
        x = pixels.to(device=vae.device, dtype=vae.dtype).unsqueeze(2)
        out = vae.encode(x)
        z = (out.latent_dist.mode() if hasattr(out, "latent_dist") else out[0].mode())[:, :, 0]
        mean, std = self._vae_stats(vae, z)
        return ((z - mean) / std).float()

    @torch.no_grad()
    def decode(self, vae, latents):
        z = latents.to(device=vae.device, dtype=vae.dtype)
        mean, std = self._vae_stats(vae, z)
        img = vae.decode((z * std + mean).unsqueeze(2), return_dict=False)[0][:, :, 0]
        return img.clamp(-1, 1)

    def load_transformer(self, device):
        """Load an original-format single file (Kroma / Krea 2) into diffusers' Krea2Transformer2DModel."""
        import accelerate
        from diffusers import Krea2Transformer2DModel
        from diffusers.loaders.single_file_utils import convert_krea2_transformer_checkpoint_to_diffusers
        from safetensors import safe_open
        path = self._transformer_path()
        if path.endswith("/") or not path.endswith(".safetensors"):  # diffusers-format directory (tests)
            model = Krea2Transformer2DModel.from_pretrained(path, torch_dtype=torch.bfloat16)
            return self._finish(model, device)
        keep = set(Krea2Transformer2DModel._keep_in_fp32_modules)
        sd = {}
        with safe_open(path, "pt", device="cpu") as f:  # tensor by tensor: no full fp32 copy in RAM
            for key in f.keys():
                name = key.removeprefix("model.diffusion_model.").removeprefix("diffusion_model.")
                sd[name] = f.get_tensor(key)
        sd = convert_krea2_transformer_checkpoint_to_diffusers(sd)
        with accelerate.init_empty_weights():
            model = Krea2Transformer2DModel(**self._infer_config(sd))
        for name in list(sd):
            module = name.rsplit(".", 1)[0]
            fp32 = any(part in keep for part in module.split("."))
            sd[name] = sd[name].to(device=device, dtype=torch.float32 if fp32 else torch.bfloat16)
        model.load_state_dict(sd, strict=True, assign=True)
        del sd
        _free()
        return self._finish(model, device)

    @staticmethod
    def _infer_config(sd):
        """Read the architecture off tensor shapes (same idea as ComfyUI's model detection)."""
        def count(prefix):
            return len({k[len(prefix):].split(".")[0] for k in sd if k.startswith(prefix)})
        hd = sd["transformer_blocks.0.attn.norm_q.weight"].shape[0]
        thd = sd["text_fusion.refiner_blocks.0.attn.norm_q.weight"].shape[0]
        return dict(
            in_channels=sd["img_in.weight"].shape[1], num_layers=count("transformer_blocks."), attention_head_dim=hd,
            num_attention_heads=sd["transformer_blocks.0.attn.to_q.weight"].shape[0] // hd,
            num_key_value_heads=sd["transformer_blocks.0.attn.to_k.weight"].shape[0] // hd,
            intermediate_size=sd["transformer_blocks.0.ff.gate.weight"].shape[0],
            timestep_embed_dim=sd["time_embed.linear_1.weight"].shape[1],
            text_hidden_dim=sd["txt_in.linear_1.weight"].shape[1],
            num_text_layers=sd["text_fusion.projector.weight"].shape[1],
            text_num_attention_heads=sd["text_fusion.refiner_blocks.0.attn.to_q.weight"].shape[0] // thd,
            text_num_key_value_heads=sd["text_fusion.refiner_blocks.0.attn.to_k.weight"].shape[0] // thd,
            text_intermediate_size=sd["text_fusion.refiner_blocks.0.ff.gate.weight"].shape[0],
            num_layerwise_text_blocks=count("text_fusion.layerwise_blocks."),
            num_refiner_text_blocks=count("text_fusion.refiner_blocks."),
            axes_dims_rope=(hd - 12 * (hd // 16), 6 * (hd // 16), 6 * (hd // 16)),
        )

    def _finish(self, model, device):
        model.requires_grad_(False)
        return model.to(device).eval()

    def set_gradient_checkpointing(self, model, enabled):
        if enabled:
            model.enable_gradient_checkpointing()
        else:
            model.disable_gradient_checkpointing()

    # ---- forward
    def tokens(self, latents):
        return (latents.shape[-2] // self.patch) * (latents.shape[-1] // self.patch)

    def _pack(self, x):
        b, c, h, w = x.shape
        p = self.patch
        return x.view(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5).reshape(b, (h // p) * (w // p), c * p * p)

    def _unpack(self, x, h, w):
        b, _, cpp = x.shape
        p = self.patch
        c = cpp // (p * p)
        return x.view(b, h // p, w // p, c, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)

    def predict(self, model, x, sigma, embeds, mask):
        _, _, h, w = x.shape
        gh, gw = h // self.patch, w // self.patch
        text_len = embeds.shape[1]
        ids = torch.zeros(text_len + gh * gw, 3, device=x.device)
        ids[text_len:, 1] = torch.arange(gh, device=x.device).repeat_interleave(gw).float()
        ids[text_len:, 2] = torch.arange(gw, device=x.device).repeat(gh).float()
        dtype = torch.bfloat16 if x.device.type == "cuda" else next(model.parameters()).dtype
        out = model(hidden_states=self._pack(x).to(dtype), encoder_hidden_states=embeds.to(dtype),
                    timestep=sigma.to(dtype), position_ids=ids,
                    encoder_attention_mask=None if bool(mask.all()) else mask, return_dict=False)[0]
        return self._unpack(out.float(), h, w)

    @staticmethod
    def calculate_mu(seq_len, base_seq=256, max_seq=6400, base_shift=0.5, max_shift=1.15):
        m = (max_shift - base_shift) / (max_seq - base_seq)
        return seq_len * m + (base_shift - m * base_seq)

    @torch.no_grad()
    def sample(self, model, vae, pos, neg, *, width, height, steps, cfg, seed, shift, skip_layer, device,
               progress=None, latents=None, output="pil"):
        import numpy as np
        from diffusers import FlowMatchEulerDiscreteScheduler
        from PIL import Image

        from .model import pad_embeds
        sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, use_dynamic_shifting=True, base_shift=0.5,
                                                max_shift=1.15, base_image_seq_len=256, max_image_seq_len=6400,
                                                time_shift_type="exponential")
        h, w = height // self.vae_scale, width // self.vae_scale
        gen = torch.Generator("cpu").manual_seed(seed)
        if latents is None:
            latents = torch.randn((1, 16, h, w), generator=gen, dtype=torch.float32)
        latents = latents.to(device, torch.float32)
        mu = shift if shift > 0 else self.calculate_mu((h // self.patch) * (w // self.patch))
        sched.set_timesteps(sigmas=np.linspace(1.0, 1 / steps, steps), mu=mu, device=device)
        p_emb, p_mask = pad_embeds([pos], device)
        n_emb, n_mask = pad_embeds([neg], device)
        timesteps = sched.timesteps
        with _autocast(device):
            for i, t in enumerate(timesteps):
                sig = (t / 1000).reshape(1).to(device)
                pred = self.predict(model, latents, sig, p_emb, p_mask)
                if cfg != 1.0:
                    uncond = self.predict(model, latents, sig, n_emb, n_mask)
                    pred = uncond + cfg * (pred - uncond)
                latents = sched.step(pred, t, latents, return_dict=False)[0]
                if progress:
                    progress(i + 1, len(timesteps))
        if output == "latent":
            return latents
        img = self.decode(vae, latents)[0]
        arr = ((img.float().permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype("uint8")
        return Image.fromarray(arr)

    # ---- LoRA naming
    def lora_name_map(self, name):
        m = re.match(r"^blocks\.(\d+)\.(attn|mlp)\.(\w+)$", name)
        if m:
            idx, kind, sub = m.groups()
            return f"transformer_blocks.{idx}." + (f"attn.{_K2_ATTN[sub]}" if kind == "attn" else _K2_FF.get(sub, sub))
        m = re.match(r"^txtfusion\.(layerwise_blocks|refiner_blocks)\.(\d+)\.(attn|mlp)\.(\w+)$", name)
        if m:
            block, idx, kind, sub = m.groups()
            return f"text_fusion.{block}.{idx}." + (f"attn.{_K2_ATTN[sub]}" if kind == "attn" else _K2_FF.get(sub, sub))
        return _K2_STANDALONE.get(name, name)


def get_backend(model_id: str) -> Backend:
    family = model_spec(model_id)["family"]
    return {"moss": MossBackend, "krea2": Krea2Backend}[family](model_id)


def prefetch(model_id: str):
    return get_backend(model_id).prefetch()


def sigma_shift_flux(u: torch.Tensor, mu: float) -> torch.Tensor:
    """Exponential (FLUX/Krea) time shift applied to uniform-ish samples u in (0, 1)."""
    return math.exp(mu) / (math.exp(mu) + (1 / u - 1))
