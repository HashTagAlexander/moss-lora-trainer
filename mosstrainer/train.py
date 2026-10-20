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

from . import lora as lora_lib
from .config import parse_layers, validate
from .data import BucketSampler, TextCache, apply_trigger, build_items, cache_latents, load_latent
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
    mode = t["timestep_sampling"]
    if mode == "fibo":
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
        out = torch.empty(n)
        for i in range(n):
            if random.random() < rho:
                out[i] = random.random()
                continue
            while True:
                v = 1 - torch.sigmoid(torch.randn(()) * s + mu).item()
                if v <= cut:
                    out[i] = v / cut
                    break
        sig = out
    elif mode == "logit_normal":
        sig = torch.sigmoid(torch.randn(n) * t["logit_std"] + t["logit_mean"])
    elif mode == "shift":
        u, sh = torch.rand(n), t["shift"]
        sig = sh * u / (1 + (sh - 1) * u)
    else:
        sig = torch.rand(n)
    return sig.clamp(0.001, 1.0).to(device)


# --------------------------------------------------------------------------- optimizer

def make_optimizer(params, t):
    name, lr, wd = t["optimizer"], t["lr"], t["weight_decay"]
    if name == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr=lr, betas=(0.9, 0.999), weight_decay=wd)
    if name == "prodigy":
        from prodigyopt import Prodigy
        return Prodigy(params, lr=lr, weight_decay=wd, safeguard_warmup=True, use_bias_correction=True)
    if name == "adafactor":
        from transformers.optimization import Adafactor
        return Adafactor(params, lr=lr, weight_decay=wd, relative_step=False, scale_parameter=False,
                         warmup_init=False)
    return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=wd,
                             fused=torch.cuda.is_available())


def make_scheduler(opt, t):
    total, warm, kind = t["steps"], t["warmup_steps"], t["lr_scheduler"]

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

    items = build_items(dataset_dir, cfg["data"]["resolutions"], cfg["trigger_word"], cfg["data"]["flip"], job.log)
    job.log(f"dataset: {len(items)} training entries from {dataset_dir}")
    buckets = {}
    for it in items:
        buckets[it.bucket] = buckets.get(it.bucket, 0) + 1
    job.log("buckets: " + ", ".join(f"{w}x{h}:{n}" for (w, h), n in sorted(buckets.items())))

    sample_prompts = [apply_trigger(p, cfg["trigger_word"]) for p in sc["prompts"]]
    neg_prompt = backend.negative_prompt(sc["negative_prompt"])
    early = {p: backend.prompt_parts(p)[1] for p in {it.caption for it in items} | set(sample_prompts)}
    prompts = set(early) | {backend.empty_prompt, neg_prompt} | {e for e in early.values() if e}

    # ---- text embeddings (Qwen3.5-9B is loaded only if something is missing, then freed)
    text_cache = TextCache(cache_root / "text", backend.text_cache_id)
    missing = text_cache.missing(prompts)
    if missing:
        job.set(state="caching_text", message=f"Loading text encoder to embed {len(missing)} prompts")
        job.log(f"encoding {len(missing)} prompts")
        enc = backend.load_text_encoder(device)
        text_cache.fill(enc, missing, job.progress("caching_text", "Encoding captions"))
        enc.unload()
        del enc
    job.check_stop()

    # ---- latents
    job.set(state="caching_latents", message="Loading VAE")
    vae = backend.load_vae(device)
    n = cache_latents(items, vae, cache_root / "latents", job.progress("caching_latents", "Encoding images"),
                      encode_fn=backend.encode_images, tag=backend.latent_tag)
    job.log(f"latents: {n} new, {len(items) - n} cached")
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
        if not sample_prompts:
            return
        index_path = job.samples_dir / "index.json"
        index = json.loads(index_path.read_text()) if index_path.exists() else []
        index = [e for e in index if e["step"] != at_step]
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

    sampler = BucketSampler(items, t["batch_size"], cfg["data"]["repeats"], seed=t["seed"] + step)
    latent_mem: dict[str, torch.Tensor] = {}
    metrics = open(job.dir / "metrics.jsonl", "a")
    times = deque(maxlen=20)
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and backend.use_autocast)
    drop = cfg["data"]["caption_dropout"]
    job.log(f"training {t['steps']} steps, {len(sampler)} batches/epoch")
    job.set(state="training", message="", step=step)

    transformer.train()
    while step < t["steps"]:
        job.check_stop()
        t0 = time.time()
        loss_total = 0.0
        for _ in range(t["grad_accum"]):
            batch = sampler.next()
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
            loss = F.mse_loss(pred.float(), noise - x0) / t["grad_accum"]
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
