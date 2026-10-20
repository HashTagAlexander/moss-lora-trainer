"""Model-family backends: everything that differs between MOSS-Image, Krea 2 (Kroma) and Fibo 2.0.

Each backend provides text encoding, VAE (de)coding, transformer loading, the training forward pass,
sampling, and the LoRA naming/format. The trainer and the generation worker only talk to this API.
Latents are kept unpacked as [B, C, h, w] everywhere; packing happens inside `predict`.
"""
from __future__ import annotations

import gc
import json
import math
import re
from pathlib import Path

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
    lora_arch = ""                     # modelspec.architecture metadata of saved LoRAs
    latent_tag = ""                    # part of the latent-cache key
    text_cache_id = ""                 # part of the text-cache key
    empty_prompt = ""                  # what caption dropout and an empty negative prompt encode (cache key)
    use_autocast = True                # run the transformer under bf16 autocast
    supports_edit = False              # can condition on source images (Generate tab)

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.spec = model_spec(model_id)

    def tokens(self, latents: torch.Tensor) -> int:
        """Image token count for a latent batch [B, C, h, w] (used by resolution-aware shifts)."""
        raise NotImplementedError

    def negative_prompt(self, prompt: str) -> str:
        """Cache key of a negative prompt (empty = the model's default)."""
        return prompt or self.empty_prompt

    def prompt_parts(self, prompt: str) -> tuple[str, str | None]:
        """(prompt, early): `early` is an extra prompt to cache, used instead at high noise levels (or None)."""
        return prompt, None

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
    lora_arch = "moss-image/lora"
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
    lora_arch = "krea2/lora"
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


# --------------------------------------------------------------------------- Fibo 2.0 (Bria)

# The null caption Fibo 2.0 was trained with (BriaFibo2Pipeline.DEFAULT_NEGATIVE_PROMPT): what caption dropout trains
# on and what an empty negative prompt means.
FIBO_NULL_CAPTION = ('{"short_description":"","objects":[],"background_setting":"","lighting":{},'
                     '"aesthetics":{},"photographic_characteristics":{},"style_medium":"",'
                     '"context":"","artistic_style":""}')
# Cache-key prefixes: a caption encoded without the images of its text (Fibo 2.0 uses it above this noise level), and
# a prompt encoded exactly as written, as the pipeline encodes negative prompts and Bria's trainer the null caption.
FIBO_NO_TEXT_IMAGES = "\x00fibo2:no-text-images\x00"
FIBO_RAW = "\x00fibo2:raw\x00"
FIBO_TEXT_IMAGES_MAX_SIGMA = 0.9
FIBO_FONTS = ("Aileron-Regular.otf", "DejaVuSans.ttf")  # what training drew the text images in, in order
FIBO_FILES = ["model_index.json", "transformer/*", "text_encoder/*", "processor/*", "vae/*", "scheduler/*", "fonts/*"]


def hf_download(repo: str, allow_patterns=None) -> str:
    """snapshot_download with a readable error for gated repos (the token comes from Settings as HF_TOKEN)."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError
    try:
        return snapshot_download(repo, allow_patterns=allow_patterns)
    except RepositoryNotFoundError as e:  # also raised (as GatedRepoError) for gated repos without access
        raise RuntimeError(
            f"No access to {repo}. It is gated: accept its license on https://huggingface.co/{repo} with your "
            f"Hugging Face account, then add a token with read access under Settings. ({type(e).__name__})") from None


def _compact(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _json_object(text: str) -> dict | None:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def fibo_texts(caption) -> list[str]:
    """Texts of a caption's text_render entries, the ones BriaFibo2Pipeline draws an image of."""
    entries = (caption.get("text_render") or []) if isinstance(caption, dict) else []
    return [e["text"] for e in entries if isinstance(e, dict) and isinstance(e.get("text"), str) and e["text"]]


def fibo_prompt(prompt: str) -> str:
    """A text-to-image prompt as BriaFibo2Pipeline passes it on: a JSON caption cleaned and serialised compactly,
    the form Fibo 2.0 was trained on; anything else (a plain sentence) as it is."""
    from diffusers.pipelines.bria_fibo2.pipeline_bria_fibo2 import BriaFibo2Pipeline
    caption, _ = BriaFibo2Pipeline._parse_prompt(prompt)
    return prompt if isinstance(caption, str) else _compact(caption or {})


class FiboTextEncoder:
    """Qwen3-VL-4B prompt encoder, through BriaFibo2Pipeline.encode_prompt itself (text images included).

    Returns [L, 6, 7680] (six bundles of three normalized Qwen3-VL layers per token), token-major so that
    model.pad_embeds pads the tokens. A single prompt is never padded, so every token is valid.
    """
    out_dtype = torch.bfloat16  # what's cached (tests use float32 to compare with the pipeline exactly)

    def __init__(self, root, device, dtype=torch.bfloat16):
        from diffusers import BriaFibo2Pipeline
        from diffusers.pipelines.bria_fibo2.text_images import load_font
        self.pipe = BriaFibo2Pipeline.from_pretrained(root, transformer=None, vae=None, dtype=dtype)
        self.model = self.pipe.text_encoder.requires_grad_(False).eval().to(device)
        self.device = torch.device(device)
        fonts = [Path(root) / "fonts" / name for name in FIBO_FONTS]
        self.fonts = [load_font(str(p)) for p in fonts if p.exists()]

    @torch.no_grad()
    def encode(self, prompt: str, images=None) -> torch.Tensor:
        from diffusers.pipelines.bria_fibo2.text_images import draw_text_image
        pipe = self.pipe
        if images:  # an edit: the pipeline builds the edit prompt from the instruction (and caption of the result)
            obj = _json_object(prompt)
            if obj is not None and "edit_instruction" not in obj:
                raise ValueError('To edit an image, write the edit instruction, or a JSON caption of the result '
                                 'with an "edit_instruction" key.')
            text = prompt
            caption, _ = pipe._parse_prompt(prompt)
        elif prompt.startswith(FIBO_RAW):
            text, caption = prompt[len(FIBO_RAW):], None
        elif prompt.startswith(FIBO_NO_TEXT_IMAGES):  # the caption with each text entry's image marker left empty
            caption = json.loads(fibo_prompt(prompt[len(FIBO_NO_TEXT_IMAGES):]))
            text, caption = _compact(pipe._mark_text_images(caption, "")), None
        else:
            text = fibo_prompt(prompt)
            caption, _ = pipe._parse_prompt(text)
        text_images = [draw_text_image(t, self.fonts) for t in fibo_texts(caption)] or None
        emb, mask = pipe.encode_prompt(text, images, self.device, text_images=text_images)
        emb = emb[0] if mask is None else emb[0][:, mask[0]]
        return emb.transpose(0, 1).to(self.out_dtype).cpu().contiguous()

    def unload(self):
        self.pipe = self.model = None
        _free()


class Fibo2Backend(Backend):
    family = "fibo2"
    lora_targets = {
        "attn": r"^(layers|noise_refiner)\.\d+\.(attention|cross_attention)\.(to_q|to_k|to_v|to_out\.0)$",
        "ffn": r"^(layers|noise_refiner)\.\d+\.feed_forward\.(w1|w2|w3)$",
        "text": r"^perceiver\.layers\.\d+\.((cross|self)_attention\.(to_q|to_k|to_v|to_out\.0)|ffn\.(w1|w2|w3))$",
        "other": r"^(x_embedder|t_embedder\.mlp\.\d+|final_layer\.(linear|adaLN_modulation\.\d+)"
                 r"|(layers|noise_refiner)\.\d+\.adaLN_modulation\.\d+)$",
    }
    lora_layer_re = r"^layers\.(\d+)\."
    lora_format = "diffusers"
    lora_arch = "fibo2/lora"
    latent_tag = "lat-fibo2-v1"
    empty_prompt = FIBO_RAW + FIBO_NULL_CAPTION
    use_autocast = False  # Bria's trainer and pipeline run the transformer in plain bf16
    supports_edit = True
    patch = 2
    vae_scale = 8

    def __init__(self, model_id):
        super().__init__(model_id)
        self.text_cache_id = f"fibo2-v1:{model_id}"
        self._root = None

    @property
    def root(self):
        if self._root is None:
            self._root = self.spec.get("path") or hf_download(self.spec.get("repo", self.model_id), FIBO_FILES)
        return self._root

    def prefetch(self):
        return self.root

    def negative_prompt(self, prompt):
        return FIBO_RAW + (prompt or FIBO_NULL_CAPTION)

    def prompt_parts(self, prompt):
        from diffusers.pipelines.bria_fibo2.pipeline_bria_fibo2 import BriaFibo2Pipeline
        caption, _ = BriaFibo2Pipeline._parse_prompt(fibo_prompt(prompt))
        return prompt, (FIBO_NO_TEXT_IMAGES + prompt if fibo_texts(caption) else None)

    # ---- components
    def load_text_encoder(self, device):
        return FiboTextEncoder(self.root, device)

    def load_vae(self, device):
        from diffusers import AutoencoderKLFlux2
        dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
        vae = AutoencoderKLFlux2.from_pretrained(self.root, subfolder="vae", torch_dtype=dtype)
        return vae.requires_grad_(False).eval().to(device)

    @staticmethod
    def _bn(vae, ref):
        mean = vae.bn.running_mean.view(1, -1, 1, 1).to(ref.device, ref.dtype)
        scale = (1 / torch.sqrt(vae.bn.running_var + vae.config.batch_norm_eps)).view(1, -1, 1, 1)
        return mean, scale.to(ref.device, ref.dtype)

    @torch.no_grad()
    def encode_images(self, vae, pixels):
        """As BriaFibo2Pipeline.prepare_image_latents: VAE mode, normalized by the VAE's BatchNorm statistics."""
        from diffusers import BriaFibo2Pipeline as P
        z = P._patchify_latents(vae.encode(pixels.contiguous().to(vae.device, vae.dtype)).latent_dist.mode().float())
        mean, scale = self._bn(vae, z)
        return P._unpatchify_latents((z - mean) * scale)

    @torch.no_grad()
    def decode(self, vae, latents):
        from diffusers import BriaFibo2Pipeline as P
        z = P._patchify_latents(latents.to(vae.device, torch.float32))
        mean, scale = self._bn(vae, z)
        z = P._unpatchify_latents(z / scale + mean)
        return vae.decode(z.to(vae.dtype), return_dict=False)[0].float().clamp(-1, 1)

    def load_transformer(self, device):
        from diffusers import BriaFibo2Transformer2DModel
        model = BriaFibo2Transformer2DModel.from_pretrained(self.root, subfolder="transformer",
                                                            torch_dtype=torch.bfloat16)
        return model.requires_grad_(False).to(device).eval()

    def set_gradient_checkpointing(self, model, enabled):
        if enabled:
            model.enable_gradient_checkpointing()
        else:
            model.disable_gradient_checkpointing()

    # ---- forward
    def tokens(self, latents):
        return (latents.shape[-2] // self.patch) * (latents.shape[-1] // self.patch)

    @staticmethod
    def calculate_mu(seq_len, base_seq=256, max_seq=4096, base_shift=0.5, max_shift=1.15):
        m = (max_shift - base_shift) / (max_seq - base_seq)
        return seq_len * m + (base_shift - m * base_seq)

    def predict(self, model, x, sigma, embeds, mask, context=None):
        dtype = torch.bfloat16 if x.device.type == "cuda" else next(model.parameters()).dtype
        out = model(hidden_states=x.to(dtype), timestep=sigma.float(),
                    encoder_hidden_states=embeds.transpose(1, 2).to(dtype),  # [B, 6, L, D]
                    encoder_attention_mask=None if bool(mask.all()) else mask,
                    context_latents=None if context is None else [c.to(dtype) for c in context],
                    return_dict=False)[0]
        return out.float()

    @torch.no_grad()
    def sample(self, model, vae, pos, neg, *, width, height, steps, cfg, seed, shift, skip_layer, device,
               progress=None, latents=None, output="pil", early=None, context=None):
        """BriaFibo2Pipeline's denoising loop: guidance as a second pass when cfg > 1; `early` (the caption without
        its text images) above noise level 0.9; `context` the latents of the images to edit."""
        import numpy as np
        from diffusers import FlowMatchEulerDiscreteScheduler
        from PIL import Image

        from .model import pad_embeds
        sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True,
                                                base_shift=0.5, max_shift=1.15, base_image_seq_len=256,
                                                max_image_seq_len=4096, time_shift_type="exponential")
        h, w = height // self.vae_scale, width // self.vae_scale
        gen = torch.Generator("cpu").manual_seed(seed)
        if latents is None:
            latents = torch.randn((1, 32, h, w), generator=gen, dtype=torch.float32)
        latents = latents.to(device, torch.float32)
        mu = shift if shift > 0 else self.calculate_mu((h // self.patch) * (w // self.patch))
        sched.set_timesteps(sigmas=np.linspace(1.0, 1 / steps, steps), mu=mu, device=device)
        p_emb, p_mask = pad_embeds([pos], device)
        e_emb, e_mask = pad_embeds([early], device) if early is not None else (p_emb, p_mask)
        n_emb, n_mask = pad_embeds([neg], device)
        timesteps = sched.timesteps
        for i, t in enumerate(timesteps):
            sig = (t / 1000).reshape(1).to(device)
            high = float(t) / 1000 > FIBO_TEXT_IMAGES_MAX_SIGMA
            pred = self.predict(model, latents, sig, e_emb if high else p_emb, e_mask if high else p_mask, context)
            if cfg > 1.0:
                uncond = self.predict(model, latents, sig, n_emb, n_mask, context)
                pred = uncond + cfg * (pred - uncond)
            latents = sched.step(pred, t, latents, return_dict=False)[0]
            if progress:
                progress(i + 1, len(timesteps))
        if output == "latent":
            return latents
        img = self.decode(vae, latents)[0]
        arr = ((img.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype("uint8")
        return Image.fromarray(arr)

    # ---- editing (Generate tab)
    def prepare_edit(self, vae, image, width, height):
        """The source image as BriaFibo2Pipeline prepares it: cropped to the size Fibo 2.0 was trained at for its
        aspect ratio (about width x height pixels). Returns (context latents, Qwen3-VL's copy, width, height)."""
        from diffusers.pipelines.bria_fibo2.image_processor import BriaFibo2ImageProcessor
        proc = BriaFibo2ImageProcessor(vae_scale_factor=self.vae_scale * self.patch)
        width, height = proc.get_image_size(image.width / image.height, area=width * height)
        crops, thumbs = proc.preprocess([image], height=height, width=width, area=width * height,
                                        text_encoder_image_size=768)
        return [self.encode_images(vae, crops[0])], thumbs, width, height


def get_backend(model_id: str) -> Backend:
    family = model_spec(model_id)["family"]
    return {"moss": MossBackend, "krea2": Krea2Backend, "fibo2": Fibo2Backend}[family](model_id)


def prefetch(model_id: str):
    return get_backend(model_id).prefetch()


def sigma_shift_flux(u: torch.Tensor, mu: float) -> torch.Tensor:
    """Exponential (FLUX/Krea) time shift applied to uniform-ish samples u in (0, 1)."""
    return math.exp(mu) / (math.exp(mu) + (1 / u - 1))
