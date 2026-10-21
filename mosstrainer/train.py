"""MOSS-Image LoRA training process.

Run by the web UI as ``python -m mosstrainer.train --job-dir workspace/jobs/<name>``.
Communicates through files in the job directory:
  config.json     job configuration (input)
  status.json     current state / progress (output, rewritten atomically)
  metrics.jsonl   one line per optimizer step
  samples/        sample images + index.json
  checkpoints/    LoRA safetensors + training_state.pt (for resume)
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import signal
import time
import traceback
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from . import lora as lora_lib
from .config import DEFAULTS, FIBO_NEGATIVE_T2I, defaults_for, parse_layers, validate
from .data import (BucketSampler, EditItem, TextCache, apply_trigger, build_edit_items, build_items, cache_edit_embeds,
                   cache_edit_latents, cache_latents, dataset_path, edit_cache_paths, edit_prompt, edit_row_problem,
                   is_edit_dataset, load_edit, load_latent, load_rgb, read_edit_rows)
from .backends import Backend, get_backend, sigma_shift_flux
from .model import pad_embeds


class StopRequested(Exception):
    pass


class Job:
    def __init__(self, job_dir: Path):
        self.dir = job_dir
        self.cfg = validate(json.loads((job_dir / "config.json").read_text()))
        self.workspace = job_dir.parent.parent
        self.samples_dir = job_dir / "samples"
        self.ckpt_dir = job_dir / "checkpoints"
        self.samples_dir.mkdir(exist_ok=True)
        self.ckpt_dir.mkdir(exist_ok=True)
        self.stop = False
        self.status = {"state": "starting", "step": 0, "total_steps": self.cfg["train"]["steps"],
                       "message": "", "pid": os.getpid(), "started_at": time.time()}

    def set(self, **kw):
        self.status.update(kw, updated_at=time.time())
        tmp = self.dir / "status.json.tmp"
        tmp.write_text(json.dumps(self.status))
        os.replace(tmp, self.dir / "status.json")

    def log(self, msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def check_stop(self):
        if self.stop:
            raise StopRequested()

    def progress(self, state, label):
        def cb(i, n):
            self.check_stop()
            self.set(state=state, message=f"{label} {i}/{n}", sub_progress=i / max(n, 1))
        return cb


# --------------------------------------------------------------------------- timesteps

def sample_sigmas(n: int, t: dict, device, tokens: int = 4096) -> torch.Tensor:
    """Noise levels for a batch: drawn from the timestep type, then skewed by the timestep bias the way ai-toolkit
    does it (draw 1000, sort from high to low noise, pick the index u^3 for high noise or 1 - u^3 for low noise)."""
    bias = t.get("timestep_bias", "balanced")
    if bias == "balanced":
        return _draw_sigmas(n, t, tokens).clamp(0.001, 1.0).to(device)
    schedule = _draw_sigmas(1000, t, tokens).sort(descending=True).values
    u3 = torch.rand(n) ** 3
    idx = (u3 if bias == "high_noise" else 1 - u3).mul(1000).long().clamp(0, 999)
    return schedule[idx].clamp(0.001, 1.0).to(device)


def _draw_sigmas(n: int, t: dict, tokens: int) -> torch.Tensor:
    mode = t["timestep_sampling"]
    if mode == "sigmoid":
        # ai-toolkit's "sigmoid": logit-normal around the middle noise levels (sigmoid of a standard normal).
        sig = torch.sigmoid(torch.randn(n))
    elif mode == "fibo":
        # Bria's Fibo 2.0 trainer: sigmoid(N(mu, 1)) with the resolution shift mu, mixed with 10% uniform.
        from .backends import Fibo2Backend
        mu = Fibo2Backend.calculate_mu(tokens)
        sig = torch.where(torch.rand(n) < 0.1, torch.rand(n), torch.sigmoid(torch.randn(n) + mu))
    elif mode == "flux_shift":
        # FLUX / Krea 2: logit-normal u, then the resolution-aware exponential shift used at inference.
        from .backends import Krea2Backend
        u = torch.sigmoid(torch.randn(n) * t["logit_std"] + t["logit_mean"]).clamp(1e-4, 1 - 1e-4)
        sig = sigma_shift_flux(u, Krea2Backend.calculate_mu(tokens))
    elif mode == "moss":
        # MOSS-Image final pretraining stage (tech report, Table 6): S = 1 - sigmoid(Z),
        # Z ~ N(-1.09, 0.95), truncated at 0.95 then rescaled onto [0, 1], mixed with 1% uniform.
        mu, s, cut, rho = -1.09, 0.95, 0.95, 0.01
        kept = torch.empty(0)
        while kept.numel() < n:  # rejection sampling of the truncation, vectorised
            v = 1 - torch.sigmoid(torch.randn(2 * n + 8) * s + mu)
            kept = torch.cat([kept, v[v <= cut] / cut])
        sig = kept[:n]
        uniform = torch.rand(n) < rho
        sig[uniform] = torch.rand(int(uniform.sum()))
    elif mode == "logit_normal":
        sig = torch.sigmoid(torch.randn(n) * t["logit_std"] + t["logit_mean"])
    elif mode == "shift":
        u, sh = torch.rand(n), t["shift"]
        sig = sh * u / (1 + (sh - 1) * u)
    else:
        sig = torch.rand(n)
    return sig


# --------------------------------------------------------------------------- optimizer

def make_optimizer(params, t):
    name, lr, wd = t["optimizer"], t["lr"], t["weight_decay"]
    if name == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr=lr, betas=(0.9, 0.999), weight_decay=wd)
    if name == "prodigy":
        from prodigyopt import Prodigy
        return Prodigy(params, lr=lr, weight_decay=wd, safeguard_warmup=True, use_bias_correction=True)
    if name == "automagic3":
        # Adapts one learning rate for the whole LoRA from how consistently the updates point the same way; the
        # configured lr is only its starting point. Not fused, so gradient accumulation and clipping still work.
        from .automagic3 import Automagic3
        return Automagic3(params, lr=lr, weight_decay=wd, fused=False)
    if name == "adafactor":
        from transformers.optimization import Adafactor
        return Adafactor(params, lr=lr, weight_decay=wd, relative_step=False, scale_parameter=False,
                         warmup_init=False)
    return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=wd,
                             fused=torch.cuda.is_available())


def make_scheduler(opt, t):
    total, warm, kind = t["steps"], t["warmup_steps"], t["lr_scheduler"]
    if t["optimizer"] == "automagic3":  # sets its own learning rate
        warm, kind = 0, "constant"

    def factor(step):
        if warm and step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        if kind == "cosine":
            return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
        if kind == "linear":
            return max(0.0, 1 - p)
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


# --------------------------------------------------------------------------- edit pairs

def edit_samples(cfg: dict, dataset_dir: Path, log) -> list[EditItem]:
    """Sample edits: each sample prompt applied to source images (sample.sources: one line per sample, several images
    separated by commas; cycled). With the default sample prompts, the first pairs' own instructions and sources."""
    sc, trigger = cfg["sample"], cfg["trigger_word"]
    rows = [r for r in read_edit_rows(dataset_dir) if not edit_row_problem(dataset_dir, r)]
    prompts = sc["prompts"]
    if not prompts or prompts in (DEFAULTS["sample"]["prompts"], defaults_for(cfg["model"])["sample"]["prompts"]):
        prompts = [edit_prompt(r["instruction"]) for r in rows[:2]]
        log("samples: the first pairs' instructions on their own sources (set sample prompts and sources to choose)")
    lines = [[rel.strip() for rel in line.split(",") if rel.strip()] for line in sc["sources"]]
    missing = [rel for line in lines for rel in line
               if (path := dataset_path(dataset_dir, rel)) is None or not path.is_file()]
    if missing:
        raise ValueError(f"Sample source images not found in the dataset: {', '.join(missing)}")
    lines = [line[:4] for line in lines if line] or [r["sources"] for r in rows]
    area = sc["width"] * sc["height"]
    return [EditItem([dataset_path(dataset_dir, rel) for rel in lines[i % len(lines)]], None, apply_trigger(p, trigger),
                     area) for i, p in enumerate(prompts)]


def side_by_side(sources: list[Path], result):
    """The sources next to the edited result, all at the result's height."""
    from PIL import Image
    h = result.height
    tiles = [(im := load_rgb(p)).resize((max(1, round(im.width * h / im.height)), h), Image.LANCZOS) for p in sources]
    tiles.append(result.convert("RGB"))
    out = Image.new("RGB", (sum(t.width for t in tiles) + 8 * (len(tiles) - 1), h), (12, 14, 18))
    x = 0
    for tile in tiles:
        out.paste(tile, (x, 0))
        x += tile.width + 8
    return out


# --------------------------------------------------------------------------- main

def run(job: Job):
    cfg, t, sc = job.cfg, job.cfg["train"], job.cfg["sample"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(t["seed"])
    torch.manual_seed(t["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dataset_dir = job.workspace / "datasets" / cfg["dataset"]
    cache_root = job.workspace / "cache"

    job.set(state="loading", message="Downloading / locating model weights (first run can take a while)")
    backend: Backend = get_backend(cfg["model"])
    job.log(f"model: {cfg['model']} ({backend.family}) -> {backend.prefetch()}")

    # Edit pairs (edits.jsonl): Bria's edit recipe. The prompt is encoded together with the source images, the
    # sources' latents condition the transformer, and context dropout sometimes leaves those latents out.
    edit = is_edit_dataset(dataset_dir)
    samples: list[EditItem] = []
    if edit:
        if not backend.supports_edit:
            raise ValueError(f"Dataset '{cfg['dataset']}' holds edit pairs, which need a Fibo 2.0 model.")
        items = build_edit_items(dataset_dir, cfg["data"]["resolutions"], cfg["trigger_word"], job.log)
        job.log(f"dataset: {len(items)} edit pairs from {dataset_dir} (edit LoRA)")
        if cfg["data"]["flip"]:
            job.log("note: flip augmentation is off for edit pairs")
        if t["batch_size"] > 1:
            job.log(f"note: edit pairs train one at a time; batch size {t['batch_size']} becomes gradient accumulation")
        samples = edit_samples(cfg, dataset_dir, job.log) if sc["every"] > 0 else []
        edit_cache_paths(items + samples, cache_root / "edit", backend.text_cache_id)
    else:
        items = build_items(dataset_dir, cfg["data"]["resolutions"], cfg["trigger_word"], cfg["data"]["flip"], job.log)
        job.log(f"dataset: {len(items)} training entries from {dataset_dir}")

    sample_prompts = [] if edit else [apply_trigger(p, cfg["trigger_word"]) for p in sc["prompts"]]
    # Edits use the empty caption as their negative (Bria's model card), not the text-to-image default.
    neg_prompt = backend.negative_prompt("" if edit and sc["negative_prompt"] == FIBO_NEGATIVE_T2I
                                         else sc["negative_prompt"])
    early = {} if edit else {p: backend.prompt_parts(p)[1] for p in {it.caption for it in items} | set(sample_prompts)}
    prompts = set(early) | {backend.empty_prompt, neg_prompt} | {e for e in early.values() if e}

    # ---- text embeddings (the encoder is loaded only if something is missing, then freed)
    text_cache = TextCache(cache_root / "text", backend.text_cache_id)
    missing = text_cache.missing(prompts)
    edit_todo = [it for it in items + samples if not it.emb_path.exists()] if edit else []
    if missing or edit_todo:
        job.set(state="caching_text", message=f"Loading text encoder to embed {len(missing) + len(edit_todo)} prompts")
        job.log(f"encoding {len(missing) + len(edit_todo)} prompts")
        enc = backend.load_text_encoder(device)
        text_cache.fill(enc, missing, job.progress("caching_text", "Encoding captions"))
        if edit_todo:
            cache_edit_embeds(items + samples, backend, enc, job.progress("caching_text", "Encoding edit prompts"))
        enc.unload()
        del enc
    job.check_stop()

    # ---- latents
    job.set(state="caching_latents", message="Loading VAE")
    vae = backend.load_vae(device)
    progress = job.progress("caching_latents", "Encoding images")
    if edit:
        n = cache_edit_latents(items + samples, backend, vae, progress)
        job.log(f"latents: {n} new, {len(items) + len(samples) - n} cached")
    else:
        n = cache_latents(items, vae, cache_root / "latents", progress, encode_fn=backend.encode_images,
                          tag=backend.latent_tag)
        job.log(f"latents: {n} new, {len(items) - n} cached")
    buckets = {}
    for it in items:
        buckets[it.bucket] = buckets.get(it.bucket, 0) + 1
    job.log("buckets: " + ", ".join(f"{w}x{h}:{n}" for (w, h), n in sorted(buckets.items())))
    job.check_stop()

    # ---- transformer + LoRA
    job.set(state="loading", message="Loading transformer")
    transformer = backend.load_transformer(device)
    net = cfg["network"]
    modules = lora_lib.inject(transformer, net["rank"], net["alpha"], net["targets"], net["dropout"],
                              parse_layers(net["layers"]), backend.lora_targets, backend.lora_layer_re)
    backend.set_gradient_checkpointing(transformer, t["gradient_checkpointing"])
    params = list(lora_lib.parameters(modules))
    job.log(f"LoRA: {len(modules)} modules, {sum(p.numel() for p in params) / 1e6:.2f}M trainable params")
    for p in params:
        p.requires_grad_(True)
    opt = make_optimizer(params, t)
    lr_sched = make_scheduler(opt, t)
    if t["optimizer"] == "automagic3" and (t["lr_scheduler"] != "constant" or t["warmup_steps"]):
        job.log("note: Automagic sets its own learning rate, so the LR schedule and warmup are ignored")

    step = 0
    state_path = job.ckpt_dir / "training_state.pt"
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        mismatch = [n for n, m in modules.items()
                    if st["lora"].get(n + ".A") is None or st["lora"][n + ".A"].shape != m.lora_A.weight.shape]
        if mismatch:
            raise RuntimeError("Saved training state doesn't match the current network settings "
                               "(rank/targets/blocks changed). Reset the job to start fresh.")
        for name, m in modules.items():
            m.lora_A.weight.data.copy_(st["lora"][name + ".A"])
            m.lora_B.weight.data.copy_(st["lora"][name + ".B"])
        opt.load_state_dict(st["optimizer"])
        lr_sched.load_state_dict(st["scheduler"])
        step = st["step"]
        job.log(f"resumed from step {step}")

    save_dtype = {"fp16": torch.float16, "fp32": torch.float32}.get(cfg["save"]["dtype"], torch.bfloat16)

    def do_samples(at_step):
        if not sample_prompts and not samples:
            return
        index_path = job.samples_dir / "index.json"
        index = json.loads(index_path.read_text()) if index_path.exists() else []
        index = [e for e in index if e["step"] != at_step]
        for i, s in enumerate(samples):  # edits: the source next to its edited result
            job.set(state="sampling", message=f"Sample {i + 1}/{len(samples)}")
            d = load_edit(s)
            img = backend.sample(transformer, vae, d["embeds"], text_cache.get(neg_prompt), width=s.bucket[0],
                                 height=s.bucket[1], steps=sc["steps"], cfg=sc["cfg"], seed=sc["seed"] + i,
                                 shift=sc["shift"], skip_layer=sc["uncond_skip_layer"], device=device,
                                 context=[c[None].to(device) for c in d["context"]])
            fname = f"{at_step:06d}_{i:02d}.jpg"
            side_by_side(s.sources, img).save(job.samples_dir / fname, quality=92)
            index.append({"step": at_step, "index": i, "prompt": s.caption, "file": fname,
                          "sources": [p.relative_to(dataset_dir.resolve()).as_posix() for p in s.sources]})
            index_path.write_text(json.dumps(index))
        for i, prompt in enumerate(sample_prompts):
            job.set(state="sampling", message=f"Sample {i + 1}/{len(sample_prompts)}")
            extra = {"early": text_cache.get(early[prompt])} if early[prompt] else {}
            img = backend.sample(transformer, vae, text_cache.get(prompt), text_cache.get(neg_prompt),
                                 width=sc["width"], height=sc["height"], steps=sc["steps"], cfg=sc["cfg"],
                                 seed=sc["seed"] + i, shift=sc["shift"], skip_layer=sc["uncond_skip_layer"],
                                 device=device, **extra)
            fname = f"{at_step:06d}_{i:02d}.jpg"
            img.save(job.samples_dir / fname, quality=92)
            index.append({"step": at_step, "index": i, "prompt": prompt, "file": fname})
            index_path.write_text(json.dumps(index))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def do_save(at_step, final=False):
        job.set(state="saving", message=f"Saving step {at_step}")
        name = cfg["name"] if final else f"{cfg['name']}_{at_step:06d}"
        meta = {"ss_base_model": cfg["model"], "ss_steps": str(at_step), "ss_trigger": cfg["trigger_word"],
                "ss_network_rank": str(net["rank"]), "ss_network_alpha": str(net["alpha"]),
                "moss_trainer_config": cfg}
        lora_lib.save(modules, job.ckpt_dir / f"{name}.safetensors", meta, save_dtype, backend.lora_format,
                      backend.lora_arch)
        torch.save({
            "lora": {**{k + ".A": m.lora_A.weight.detach().cpu() for k, m in modules.items()},
                     **{k + ".B": m.lora_B.weight.detach().cpu() for k, m in modules.items()}},
            "optimizer": opt.state_dict(), "scheduler": lr_sched.state_dict(), "step": at_step,
        }, state_path.with_suffix(".tmp"))
        os.replace(state_path.with_suffix(".tmp"), state_path)
        keep = cfg["save"]["keep_last"]
        if keep > 0:
            ckpts = sorted(job.ckpt_dir.glob(f"{cfg['name']}_*.safetensors"))
            for old in ckpts[:-keep]:
                old.unlink(missing_ok=True)
        job.log(f"saved {name}.safetensors")

    if step == 0 and sc["sample_first"] and sc["every"] > 0:
        do_samples(0)

    sampler = BucketSampler(items, 1 if edit else t["batch_size"], cfg["data"]["repeats"], seed=t["seed"] + step)
    accum = t["grad_accum"] * (t["batch_size"] if edit else 1)
    edit_mem: dict[str, dict] = {}
    latent_mem: dict[str, torch.Tensor] = {}
    metrics = open(job.dir / "metrics.jsonl", "a")
    times = deque(maxlen=20)
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and backend.use_autocast)
    drop, context_drop = cfg["data"]["caption_dropout"], cfg["data"]["context_dropout"]
    job.log(f"training {t['steps']} steps, {len(sampler)} batches/epoch")
    job.set(state="training", message="", step=step)

    def edit_step(it: EditItem) -> float:
        key = str(it.lat_path)
        if key not in edit_mem:  # target and source latents are small; the prompt embeddings are read each time
            d = load_edit(it)
            edit_mem[key] = {"x0": d["x0"], "context": d["context"]}
        d = edit_mem[key]
        x0 = d["x0"][None].to(device, torch.float32)
        noise = torch.randn_like(x0)
        sig = sample_sigmas(1, t, device, backend.tokens(x0))
        emb = text_cache.get(backend.empty_prompt) if random.random() < drop else load_file(str(it.emb_path))["embeds"]
        context = None if random.random() < context_drop else [c[None].to(device) for c in d["context"]]
        embeds, mask = pad_embeds([emb], device)
        noisy = (1 - sig.view(-1, 1, 1, 1)) * x0 + sig.view(-1, 1, 1, 1) * noise
        with autocast:
            pred = backend.predict(transformer, noisy, sig, embeds, mask, context=context)
        loss = F.mse_loss(pred.float(), noise - x0) / accum
        loss.backward()
        return loss.item()

    transformer.train()
    while step < t["steps"]:
        job.check_stop()
        t0 = time.time()
        loss_total = 0.0
        for _ in range(accum):
            batch = sampler.next()
            if edit:
                loss_total += edit_step(batch[0])
                continue
            lats = []
            for it in batch:
                key = str(it.latent_path)
                if key not in latent_mem:
                    latent_mem[key] = load_latent(it)
                lats.append(latent_mem[key])
            x0 = torch.stack(lats).to(device, torch.float32)
            noise = torch.randn_like(x0)
            sig = sample_sigmas(len(batch), t, device, backend.tokens(x0))
            # Fibo 2.0: above noise level 0.9 a caption is seen without the images of its text, as at inference.
            caps = [backend.empty_prompt if random.random() < drop
                    else early[it.caption] if early[it.caption] and s > 0.9 else it.caption
                    for it, s in zip(batch, sig.tolist())]
            embeds, mask = pad_embeds([text_cache.get(c) for c in caps], device)
            s = sig.view(-1, 1, 1, 1)
            noisy = (1 - s) * x0 + s * noise
            with autocast:
                pred = backend.predict(transformer, noisy, sig, embeds, mask)
            loss = F.mse_loss(pred.float(), noise - x0) / accum
            loss.backward()
            loss_total += loss.item()
        if not math.isfinite(loss_total):
            raise RuntimeError(f"Loss became {loss_total} at step {step}; try a lower learning rate.")
        grad_norm = torch.nn.utils.clip_grad_norm_(params, t["max_grad_norm"]) if t["max_grad_norm"] > 0 else None
        opt.step()
        lr_sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        times.append(time.time() - t0)
        sec = sum(times) / len(times)
        lr = opt.param_groups[0]["lr"]
        if hasattr(opt, "get_avg_learning_rate"):  # Automagic: the adapted rate
            lr = opt.get_avg_learning_rate()
        if t["optimizer"] == "prodigy":
            lr = lr * opt.param_groups[0].get("d", 1.0)
        metrics.write(json.dumps({"step": step, "loss": loss_total, "lr": lr, "sec": round(times[-1], 3),
                                  "grad_norm": float(grad_norm) if grad_norm is not None else None,
                                  "epoch": sampler.epoch}) + "\n")
        metrics.flush()
        job.set(state="training", step=step, loss=loss_total, sec_per_step=sec, epoch=sampler.epoch,
                eta_seconds=sec * (t["steps"] - step), message="")

        last = step >= t["steps"]
        if cfg["save"]["every"] > 0 and step % cfg["save"]["every"] == 0 and not last:
            do_save(step)
        if sc["every"] > 0 and (step % sc["every"] == 0 or last):
            transformer.eval()
            do_samples(step)
            transformer.train()
        if job.stop and not last:
            do_save(step)
            raise StopRequested()

    do_save(step, final=True)
    metrics.close()
    job.set(state="done", step=step, message="Finished", eta_seconds=0)
    del transformer, vae
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True, type=Path)
    args = ap.parse_args()
    job = Job(args.job_dir.resolve())

    def handle(signum, frame):
        job.log("stop requested, finishing current step...")
        job.stop = True
    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)

    try:
        run(job)
    except StopRequested:
        job.set(state="stopped", message="Stopped by user")
        job.log("stopped")
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        job.set(state="error", message=f"{type(e).__name__}: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
