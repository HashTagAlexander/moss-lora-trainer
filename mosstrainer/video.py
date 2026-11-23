"""Video generation with Sulphur 3 (a MiniMax H3 fine-tune): Generate tab only, nothing here trains.

MiniMax H3 makes video and its stereo soundtrack in one pass: 24 fps, 5 to 15 seconds, 768 px short edge by default,
guidance-distilled (no negative prompt, no CFG). diffusers runs it as `MiniMaxH3ModularPipeline`; this module builds
that pipeline from three places, to keep downloads and GPU memory down:

  video model    SulphurAI/Sulphur-3 `s3.safetensors` (40 GB, bf16). A "pruned" single file: the timestep MLP and the
                 full-width AdaLN weights (13B of H3's 33B parameters) are replaced by a small table of the
                 time-embedding curve, `adaln_t_table` [1025, 8]. diffusers refuses such files, so `load_transformer`
                 fits the table into diffusers' own model class (see `_curve_transformer`).
  text encoder   Comfy-Org/MiniMax-H3's int8 ConvRot Qwen3-VL 32B (27 GB instead of 67), kept as int8 on the GPU by
                 `mosstrainer.int8_convrot`.
  everything else  MiniMaxAI/MiniMax-H3 as it is: the video and audio VAEs, tokenizer, processor and both schedulers.
  turbo (optional) lightx2v's 8-step distillation LoRA for H3 (1.4 GB), applied at run time with a lower video shift
                 (see `read_lora` and `VideoBackend.set_turbo`).

About 73 GB of weights in all, which fits a 96 GB card with every part loaded (see `VideoBackend.load`).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch import nn

from .config import model_spec

FPS = 24
# num_frames must be 17 * n + 5 and last 5 to 15 seconds: n = 7 … 20
FRAME_CHOICES = [17 * n + 5 for n in range(7, 21)]
_BASE_PARTS = ["vae", "audio_vae", "tokenizer", "processor", "scheduler", "audio_scheduler"]


def snap_frames(seconds: float) -> int:
    """The frame count H3 can make that is closest to `seconds` (124 … 345 frames at 24 fps)."""
    return min(FRAME_CHOICES, key=lambda n: abs(n - seconds * FPS))


# --------------------------------------------------------------------------- the pruned ("curve") transformer

class _CurveTimeEmbed(nn.Module):
    """Stands in for `time_embedder`: the time-embedding curve sampled on a grid over t in [0, 1], read with linear
    interpolation (ComfyUI's `MiniMaxH3Model._time_embedding`)."""

    def __init__(self, table: torch.Tensor):
        super().__init__()
        self.register_buffer("table", table.float())

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        table = self.table
        pos = timestep.to(table.device, torch.float32).clamp(0.0, 1.0) * (table.shape[0] - 1)
        low = pos.floor().long().clamp(max=table.shape[0] - 2)
        return torch.lerp(table[low], table[low + 1], (pos - low).unsqueeze(1))


class _CurveModulation(nn.Module):
    """A block's AdaLN projection for a curve checkpoint: the same six chunks as diffusers'
    `MiniMaxH3AdaLayerNormModulation`, but straight from the curve coordinates (no SiLU: the curve already holds the
    activated embedding) and computed in float32, then handed over at the block's precision."""

    def __init__(self, linear: nn.Linear, hidden_size: int, like: nn.Parameter):
        super().__init__()
        self.linear, self.hidden_size = linear, hidden_size
        self._like = [like]  # a parameter of the block, for its dtype; in a list so it isn't registered twice

    def forward(self, temb: torch.Tensor):
        out = self.linear(temb.to(self.linear.weight.dtype)).to(self._like[0].dtype)
        return out.view(-1, 6 * self.hidden_size).chunk(6, dim=-1)


class _CurveNormOut(nn.Module):
    """diffusers' `MiniMaxH3AdaLayerNormOut` for a curve checkpoint (again no SiLU)."""

    def __init__(self, norm: nn.Module, linear: nn.Linear):
        super().__init__()
        self.norm, self.linear = norm, linear

    def forward(self, hidden_states, temb, timestep_indices):
        shift, scale = self.linear(temb.to(self.linear.weight.dtype)).to(hidden_states.dtype).chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(0, timestep_indices)


def _curve_transformer(model, table: torch.Tensor):
    """Turn a `MiniMaxH3Transformer3DModel` built with `time_embed_dim = table.shape[1]` into the curve form: the
    transformer's `timestep` (t = 1 - sigma, in [0, 1]) goes to the table instead of the sinusoidal MLP."""
    model.time_proj = nn.Identity()
    model.time_embedder = _CurveTimeEmbed(table)
    for block in model.transformer_blocks:
        block.adaln_proj = _CurveModulation(block.adaln_proj.linear, block.adaln_proj.hidden_size, block.norm1.weight)
    model.norm_out = _CurveNormOut(model.norm_out.norm, model.norm_out.linear)
    return model


def load_transformer(path, config: dict, device="cpu", dtype=torch.bfloat16):
    """An original-format MiniMax H3 single file (ComfyUI's layout, stacked q/k/v) as a diffusers
    `MiniMaxH3Transformer3DModel`. Pruned files (with `adaln_t_table`) get the curve modules above; their small
    AdaLN projections run in float32 as ComfyUI runs them. Tensors are read one at a time and assigned, so nothing
    is held twice."""
    import accelerate
    from diffusers import MiniMaxH3Transformer3DModel
    from diffusers.loaders.single_file_utils import convert_minimax_h3_transformer_checkpoint_to_diffusers
    from safetensors import safe_open

    config = {k: v for k, v in config.items() if not k.startswith("_")}
    state = {}
    with safe_open(str(path), "pt", device="cpu") as f:
        for key in f.keys():
            name = key.removeprefix("model.diffusion_model.").removeprefix("diffusion_model.")
            state[name] = f.get_tensor(key)
    table = state.pop("adaln_t_table", None)
    if table is not None:
        config["time_embed_dim"] = table.shape[1]
    state = convert_minimax_h3_transformer_checkpoint_to_diffusers(state, config)
    with accelerate.init_empty_weights():
        model = MiniMaxH3Transformer3DModel(**config)
    if table is not None:
        _curve_transformer(model, table)
    keep32 = tuple(MiniMaxH3Transformer3DModel._keep_in_fp32_modules)
    for name in list(state):
        adaln = table is not None and (".adaln_proj." in name or name.startswith("norm_out.linear."))
        fp32 = adaln or any(part in name for part in keep32)
        state[name] = state[name].to(device=device, dtype=torch.float32 if fp32 else dtype)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    missing = [k for k in missing if not k.startswith(("rope.", "time_embedder.table"))]
    if missing or unexpected:
        raise ValueError(f"{path}: not a MiniMax H3 transformer this loader knows "
                         f"(missing {missing[:4]}, unexpected {list(unexpected)[:4]})")
    return model.requires_grad_(False).eval().to(device)


def read_lora(path) -> dict[str, tuple[torch.Tensor, torch.Tensor, float]]:
    """A MiniMax H3 LoRA in diffusers' key layout (lightx2v's turbo files: `<module>.lora_A.default.weight`, PEFT's
    adapter name in the key) as {module name: (A, B, alpha / rank)}. Alpha comes from a `<module>.alpha` tensor, or
    the file's `alpha` metadata, or else equals the rank."""
    from safetensors import safe_open
    out = {}
    with safe_open(str(path), "pt", device="cpu") as f:
        keys = set(f.keys())
        file_alpha = (f.metadata() or {}).get("alpha")
        for key in keys:
            name, sep, tail = key.partition(".lora_A.")
            if not sep or not tail.endswith("weight"):
                continue
            up = f"{name}.lora_B.{tail}"
            if up not in keys:
                continue
            a, b = f.get_tensor(key), f.get_tensor(up)
            alpha = f.get_tensor(f"{name}.alpha").item() if f"{name}.alpha" in keys else float(file_alpha or a.shape[0])
            out[name.removeprefix("transformer.")] = (a, b, alpha / a.shape[0])
    return out


# --------------------------------------------------------------------------- the model as the trainer sees it

class VideoBackend:
    """Sulphur 3 / MiniMax H3 for the Generate tab. Spec keys: `repo` + `file` (or `path`) for the video model,
    `text_encoder` = [repo, file] (or a path) for the int8 text encoder, `base` for the rest of the pipeline, and
    optionally `turbo` = {"repo", "file" (or "path"), "steps", "shift"} for a step-distillation LoRA."""

    family = "h3"

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.spec = model_spec(model_id)
        self.pipe = None
        self.device = torch.device("cpu")
        self._turbo = None       # {module name: DynamicLoRALinear} once the turbo LoRA has been loaded
        self._turbo_on = False
        self._shift = None       # the pipeline's own video shift

    # ---- downloads
    def _transformer_path(self) -> str:
        if self.spec.get("path"):
            return self.spec["path"]
        from huggingface_hub import hf_hub_download
        return hf_hub_download(self.spec["repo"], self.spec["file"])

    def _text_encoder_path(self) -> str:
        src = self.spec["text_encoder"]
        if isinstance(src, str):
            return src
        from huggingface_hub import hf_hub_download
        return hf_hub_download(src[0], src[1])

    def _base(self) -> str:
        """The base pipeline's small files and VAEs (not its 66 GB transformer and text encoder weights)."""
        from huggingface_hub import snapshot_download
        snapshot_download(self.spec["base"], allow_patterns=[
            "modular_model_index.json", "transformer/config.json", "text_encoder/config.json",
            *(f"{part}/*" for part in _BASE_PARTS)])
        return self.spec["base"]

    def _turbo_path(self) -> str | None:
        turbo = self.spec.get("turbo")
        if not turbo:
            return None
        if turbo.get("path"):
            return turbo["path"]
        from huggingface_hub import hf_hub_download
        return hf_hub_download(turbo["repo"], turbo["file"])

    def prefetch(self):
        self._turbo_path()
        return self._transformer_path(), self._text_encoder_path(), self._base()

    def _config(self, part: str) -> dict:
        from huggingface_hub import hf_hub_download
        return json.loads(Path(hf_hub_download(self.spec["base"], f"{part}/config.json")).read_text())

    # ---- loading
    def load(self, device, status=lambda message: None):
        from diffusers import ComponentsManager, ModularPipeline
        from transformers import Qwen3VLConfig

        from .int8_convrot import load_qwen3vl_int8
        self.device = device = torch.device(device)
        cuda = device.type == "cuda"
        dtype = torch.bfloat16 if cuda else torch.float32
        transformer_path, te_path, base = self.prefetch()
        # With room for all the weights plus working memory (MOSS_VIDEO_RESERVE_GB, default 14), every part goes
        # straight onto the card and stays there: the case on a 96 GB card. Otherwise the parts load to RAM and
        # diffusers' ComponentsManager moves each one on and off the card as the pipeline reaches it.
        reserve = float(os.environ.get("MOSS_VIDEO_RESERVE_GB", "14"))
        need = os.path.getsize(te_path) + os.path.getsize(transformer_path) + (6 + reserve) * 2**30  # 6: the decoders
        direct = not cuda or torch.cuda.mem_get_info(device)[0] >= need
        manager = None if direct else ComponentsManager()
        home = device if direct else "cpu"
        # every workflow on one pipeline: a request with no frame runs text-to-video, one with a first or last frame
        # runs the keyframe workflow. (The reference-image workflow needs its own transformer, which isn't loaded.)
        pipe = ModularPipeline.from_pretrained(base, components_manager=manager)
        status("Loading the text encoder (int8)…")
        text_encoder = load_qwen3vl_int8(te_path, Qwen3VLConfig.from_dict(self._config("text_encoder")), home, dtype)
        status("Loading the video model…")
        transformer = load_transformer(transformer_path, self._config("transformer"), home, dtype)
        pipe.update_components(transformer=transformer, text_encoder=text_encoder)
        status("Loading the video and audio decoders…")
        pipe.load_components(names=_BASE_PARTS, dtype=dtype)
        if direct:
            pipe.to(device)
        else:
            manager.enable_auto_cpu_offload(device=device, memory_reserve_margin=f"{reserve:g}GB")
        self.pipe, self._shift = pipe, pipe.scheduler.shift
        return self

    def set_turbo(self, on: bool) -> None:
        """Switch the step-distillation LoRA on or off. It is read on first use and added to each layer's output at
        run time (the base weights stay as they are, so off is exactly the base model); on, the video schedule also
        gets the shift the LoRA was distilled with. The audio schedule keeps its own."""
        turbo = self.spec.get("turbo")
        if on and not turbo:
            raise ValueError("This video model has no turbo LoRA.")
        if on and self._turbo is None:
            from .lora import DynamicLoRALinear, _replace
            model = self.pipe.transformer
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
            wrapped, missing = {}, []
            for name, (a, b, scale) in read_lora(self._turbo_path()).items():
                try:
                    layer = model.get_submodule(name)
                except AttributeError:
                    layer = None
                if not isinstance(layer, nn.Linear) or layer.weight.shape != (b.shape[0], a.shape[1]):
                    missing.append(name)
                    continue
                wrapped[name] = DynamicLoRALinear(layer)
                wrapped[name].turbo = ("lora", a.to(self.device, dtype), b.to(self.device, dtype), scale)
                _replace(model, name, wrapped[name])
            if missing or not wrapped:
                raise ValueError(f"The turbo LoRA doesn't fit this model ({len(missing)} layers, e.g. {missing[:2]}).")
            self._turbo = wrapped
        for layer in (self._turbo or {}).values():
            layer.adapters[:] = [layer.turbo] if on else []
        self.pipe.scheduler.set_shift(float(turbo.get("shift", self._shift)) if on else self._shift)
        self._turbo_on = bool(on)

    def unload(self):
        self.pipe = None

    # ---- generating
    @torch.no_grad()
    def generate(self, prompt: str, *, seconds: float = 5.0, width: int | None = None, height: int | None = None,
                 steps: int = 20, seed: int = 0, image=None, last_image=None, progress=None):
        """Returns (frames, audio, sampling rate, width, height): `frames` is a list of PIL images at 24 fps and
        `audio` the stereo soundtrack. `image` / `last_image` are optional first / last frames; without a size the
        video takes the first frame's shape (16:9 without one) at a 768 px short edge. `progress(step, total)` is
        called after every denoising step and may raise to cancel."""
        kwargs = {"prompt": prompt, "num_frames": snap_frames(seconds), "num_inference_steps": int(steps) + 1,
                  "generator": torch.Generator().manual_seed(int(seed))}
        if image is not None:
            kwargs["image"] = image
        if last_image is not None:
            kwargs["last_image"] = last_image
        if width and height:
            kwargs.update(width=int(width) // 32 * 32, height=int(height) // 32 * 32)
        done = [0]

        def after_step(module, args, output):  # one transformer call per step: H3 is guidance-distilled
            done[0] += 1
            if progress:
                progress(done[0], int(steps))

        hook = self.pipe.transformer.register_forward_hook(after_step)
        try:
            out = self.pipe(**kwargs, output=["videos", "audio", "sampling_rate", "width", "height"])
        finally:
            hook.remove()
        return out["videos"][0], out["audio"][0], int(out["sampling_rate"]), int(out["width"]), int(out["height"])


_AAC_RATES = (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000)


def save_video(frames, audio, sampling_rate: int, path) -> None:
    """One MP4 with the soundtrack (PyAV, through diffusers). H3's audio is 32 kHz stereo; a soundtrack at a rate
    AAC can't store is resampled to the nearest one it can."""
    from diffusers.utils.export_utils import encode_video
    if audio is not None and sampling_rate not in _AAC_RATES:
        rate = min(_AAC_RATES, key=lambda r: abs(r - sampling_rate))
        audio = torch.nn.functional.interpolate(audio.float()[None], scale_factor=rate / sampling_rate, mode="linear")[0]
        sampling_rate = rate
    encode_video(frames, fps=FPS, output_path=str(path), audio=audio, audio_sample_rate=sampling_rate)
