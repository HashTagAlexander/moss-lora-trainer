"""Krea 2 / Kroma smoke test on CPU with tiny random models (needs network once for the Qwen3-VL tokenizer).

    python tests/smoke_test_krea2.py

Checks, against diffusers' own Krea2Pipeline:
  * our Qwen3-VL prompt encoding == Krea2Pipeline.get_text_hidden_states (valid tokens),
  * original-format (Kroma/ComfyUI) checkpoint conversion + loading,
  * our sampler == Krea2Pipeline denoising for the same start latents (CFG convention included),
  * saved LoRAs load with diffusers' Krea2 LoRA loader and give the same output as our runtime stack,
then runs the real trainer process (caching, training, sampling, saving, resume).
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TD, F, HD, KV, I, TI = 64, 64, 32, 1, 128, 96  # text dim, features, head dim, kv heads, mlp, text mlp
TINY = dict(in_channels=64, num_layers=2, attention_head_dim=HD, num_attention_heads=F // HD, num_key_value_heads=KV,
            intermediate_size=I, timestep_embed_dim=256, text_hidden_dim=TD, num_text_layers=12,
            text_num_attention_heads=2, text_num_key_value_heads=2, text_intermediate_size=TI,
            axes_dims_rope=(8, 12, 12))


def tiny_original_checkpoint(path):
    """Kroma's real key names with shapes scaled down, converted by the same code that loads Kroma."""
    keys = {
        "first.weight": [F, 64], "first.bias": [F], "last.linear.weight": [64, F], "last.linear.bias": [64],
        "last.modulation.lin": [2, F], "last.norm.scale": [F], "tmlp.0.weight": [F, 256], "tmlp.0.bias": [F],
        "tmlp.2.weight": [F, F], "tmlp.2.bias": [F], "tproj.1.weight": [6 * F, F], "tproj.1.bias": [6 * F],
        "txtfusion.projector.weight": [1, 12], "txtmlp.0.scale": [TD], "txtmlp.1.weight": [F, TD],
        "txtmlp.1.bias": [F], "txtmlp.3.weight": [F, F], "txtmlp.3.bias": [F],
    }
    for i in range(2):
        b = f"blocks.{i}."
        keys.update({b + "attn.wq.weight": [F, F], b + "attn.wk.weight": [KV * HD, F], b + "attn.wv.weight": [KV * HD, F],
                     b + "attn.gate.weight": [F, F], b + "attn.wo.weight": [F, F], b + "attn.qknorm.qnorm.scale": [HD],
                     b + "attn.qknorm.knorm.scale": [HD], b + "mlp.gate.weight": [I, F], b + "mlp.up.weight": [I, F],
                     b + "mlp.down.weight": [F, I], b + "mod.lin": [6 * F], b + "prenorm.scale": [F],
                     b + "postnorm.scale": [F]})
        for blk in ("layerwise_blocks", "refiner_blocks"):
            t = f"txtfusion.{blk}.{i}."
            keys.update({t + "attn.wq.weight": [TD, TD], t + "attn.wk.weight": [TD, TD], t + "attn.wv.weight": [TD, TD],
                         t + "attn.gate.weight": [TD, TD], t + "attn.wo.weight": [TD, TD],
                         t + "attn.qknorm.qnorm.scale": [TD // 2], t + "attn.qknorm.knorm.scale": [TD // 2],
                         t + "mlp.gate.weight": [TI, TD], t + "mlp.up.weight": [TI, TD], t + "mlp.down.weight": [TD, TI],
                         t + "prenorm.scale": [TD], t + "postnorm.scale": [TD]})
    g = torch.Generator().manual_seed(0)
    save_file({k: (torch.randn(s, generator=g) * 0.05).contiguous() for k, s in keys.items()}, str(path))


def build(tmp: Path):
    from diffusers import AutoencoderKLQwenImage
    from huggingface_hub import snapshot_download
    from transformers import Qwen3VLConfig, Qwen3VLModel
    te_dir = tmp / "te"
    tok = snapshot_download("Qwen/Qwen3-VL-4B-Instruct", allow_patterns=["tokenizer*", "vocab.json", "merges.txt", "*.jinja"])
    cfg = Qwen3VLConfig(
        text_config=dict(hidden_size=TD, num_hidden_layers=36, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         intermediate_size=128, vocab_size=151936,
                         rope_parameters={"rope_type": "default", "rope_theta": 5000000, "mrope_section": [2, 3, 3],
                                          "mrope_interleaved": True}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=TD,
                           deepstack_visual_indexes=[0]))
    torch.manual_seed(0)
    Qwen3VLModel(cfg).save_pretrained(te_dir)
    for f in Path(tok).iterdir():
        (te_dir / f.name).write_bytes(f.read_bytes())
    AutoencoderKLQwenImage(base_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1).save_pretrained(tmp / "vae")
    tiny_original_checkpoint(tmp / "tiny-kroma.safetensors")
    spec = {"tiny-krea2": {"label": "tiny Krea 2", "family": "krea2", "trainable": True,
                           "path": str(tmp / "tiny-kroma.safetensors"), "text_encoder": str(te_dir),
                           "vae": str(tmp / "vae"),
                           "defaults": {"steps": 3, "cfg": 2.0, "shift": 0.0, "skip_layer": -1,
                                        "timestep_sampling": "flux_shift"}}}
    (tmp / "models.json").write_text(json.dumps(spec))
    os.environ["MOSS_EXTRA_MODELS"] = str(tmp / "models.json")


def check_against_diffusers(tmp: Path):
    import diffusers
    from diffusers import FlowMatchEulerDiscreteScheduler, Krea2Pipeline
    from mosstrainer import lora as L
    from mosstrainer.backends import get_backend
    from mosstrainer.model import pad_embeds

    import mosstrainer.config as C
    C.MODEL_SPECS.update(json.loads((tmp / "models.json").read_text()))
    be = get_backend("tiny-krea2")
    te = be.load_text_encoder("cpu")
    te.model.float()
    model = be.load_transformer("cpu").float()
    vae = be.load_vae("cpu")
    sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, use_dynamic_shifting=True, base_shift=0.5,
                                            max_shift=1.15, base_image_seq_len=256, max_image_seq_len=6400)
    pipe = Krea2Pipeline(scheduler=sched, vae=vae, text_encoder=te.model, tokenizer=te.tokenizer, transformer=model)

    prompt = "a red fox sitting in fresh snow, morning light"
    with torch.no_grad():
        ref, mask = pipe.get_text_hidden_states([prompt], device=torch.device("cpu"))
    ours = te.encode(prompt).float()
    ref_valid = ref[0][mask[0]].to(torch.bfloat16).float()
    print("text: tokens", ours.shape, "max diff vs pipeline", (ours - ref_valid).abs().max().item())
    assert ours.shape == ref_valid.shape and torch.allclose(ours, ref_valid, atol=1e-6)

    # sampler parity: same start latents, CFG 1+g <-> pipeline g
    pos, neg = te.encode(prompt), te.encode("")
    g = torch.Generator().manual_seed(3)
    start = torch.randn((1, 16, 32, 32), generator=g)
    ours = be.sample(model, vae, pos, neg, width=256, height=256, steps=4, cfg=2.5, seed=0, shift=0.0, skip_layer=-1,
                     device="cpu", latents=start, output="latent")
    with torch.no_grad():
        ref_lat = pipe(prompt_embeds=ref.float(), prompt_embeds_mask=mask, negative_prompt=None,
                       negative_prompt_embeds=pipe.get_text_hidden_states([""], device=torch.device("cpu"))[0].float(),
                       negative_prompt_embeds_mask=pipe.get_text_hidden_states([""], device=torch.device("cpu"))[1],
                       height=256, width=256, num_inference_steps=4, guidance_scale=1.5,
                       latents=be._pack(start), output_type="latent").images
    ref_lat = be._unpack(ref_lat.float(), 32, 32)
    print("sampler: max diff vs Krea2Pipeline", (ours - ref_lat).abs().max().item())
    assert torch.allclose(ours, ref_lat, atol=2e-2), "sampler diverges from diffusers"

    # LoRA format: train-format file -> diffusers loader == our runtime stack
    mods = L.inject(model, 4, 8, ("attn", "ffn", "text"), target_map=be.lora_targets, layer_re=be.lora_layer_re)
    for m in mods.values():
        torch.nn.init.normal_(m.lora_B.weight, std=0.05)
    L.save(mods, tmp / "l.safetensors", {}, torch.float32, be.lora_format)
    x = torch.randn(1, 16, 16, 16)
    sig = torch.tensor([0.6])
    e, msk = pad_embeds([pos.float()])
    with torch.no_grad():
        injected = be.predict(model, x, sig, e, msk)
    for n, m in mods.items():  # unwrap back to plain linears
        L._replace(model, n, m.base)
    stack = L.LoRAStack(model, name_map=be.lora_name_map)
    stack.activate([(tmp / "l.safetensors", 1.0)], "cpu", torch.float32)
    with torch.no_grad():
        stacked = be.predict(model, x, sig, e, msk)
    stack.activate([], "cpu")
    for n, m in stack.wrapped.items():
        L._replace(model, n, m.base)
    sd = Krea2Pipeline.lora_state_dict(str(tmp / "l.safetensors"))
    sd = sd[0] if isinstance(sd, tuple) else sd
    model.load_lora_adapter(sd, adapter_name="t", prefix="transformer")
    with torch.no_grad():
        diff = be.predict(model, x, sig, e, msk)
    print("lora: stack vs injected", (stacked - injected).abs().max().item(), "| diffusers loader vs injected",
          (diff - injected).abs().max().item(), "| diffusers", diffusers.__version__)
    assert torch.allclose(stacked, injected, atol=1e-4) and torch.allclose(diff, injected, atol=1e-4)


def run_trainer(tmp: Path):
    ws = tmp / "workspace"
    ds = ws / "datasets" / "toy"
    ds.mkdir(parents=True)
    for i, size in enumerate([(320, 256), (256, 320), (256, 256)]):
        Image.new("RGB", size, (70 * i, 120, 200 - 50 * i)).save(ds / f"img{i}.png")
        (ds / f"img{i}.txt").write_text(f"a flat color swatch number {i}")
    job = ws / "jobs" / "k2"
    job.mkdir(parents=True)
    (job / "config.json").write_text(json.dumps({
        "name": "k2", "model": "tiny-krea2", "dataset": "toy", "trigger_word": "zxq",
        "network": {"rank": 4, "alpha": 4, "targets": ["attn", "ffn"]}, "data": {"resolutions": [256]},
        "train": {"steps": 4, "batch_size": 2, "lr": 1e-3}, "save": {"every": 2},
        "sample": {"every": 2, "prompts": ["[trigger] swatch"], "width": 256, "height": 256}}))

    def go():
        r = subprocess.run([sys.executable, "-m", "mosstrainer.train", "--job-dir", str(job)], cwd=ROOT,
                           capture_output=True, text=True, env=os.environ.copy())
        print(r.stdout[-2500:], r.stderr[-2500:])
        return json.loads((job / "status.json").read_text()), r.stdout
    st, out = go()
    assert st["state"] == "done", st
    cfg = json.loads((job / "config.json").read_text())
    cfg["train"]["steps"] = 6
    (job / "config.json").write_text(json.dumps(cfg))
    st, out = go()
    assert st["state"] == "done" and st["step"] == 6 and "resumed from step 4" in out, st
    from safetensors import safe_open
    with safe_open(str(job / "checkpoints" / "k2.safetensors"), "pt") as f:
        keys = list(f.keys())
    assert keys and all(k.startswith("transformer.transformer_blocks.") for k in keys) and not any("alpha" in k for k in keys)
    print("trainer OK, LoRA keys e.g.", keys[0])


def run_worker(tmp: Path):
    """Generate through the Generate tab's worker process, with the LoRA the trainer just saved."""
    lora = tmp / "workspace" / "jobs" / "k2" / "checkpoints" / "k2.safetensors"
    proc = subprocess.Popen([sys.executable, "-m", "mosstrainer.generate_worker"], cwd=ROOT, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=os.environ.copy())
    req = {"cmd": "generate", "id": "t1", "model": "tiny-krea2", "prompt": "zxq swatch", "negative": "", "width": 256,
           "height": 256, "steps": 3, "cfg": 2.0, "shift": 0.0, "seed": 5, "count": 2, "skip_layer": -1,
           "loras": [{"path": str(lora), "strength": 0.8, "name": "k2"}], "out_dir": str(tmp / "gen")}
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.flush()
    events = []
    for line in proc.stdout:
        ev = json.loads(line)
        events.append(ev["event"])
        if ev["event"] in ("done", "error"):
            assert ev["event"] == "done", ev
            break
    proc.terminate()
    pngs = sorted((tmp / "gen").glob("*.png"))
    assert len(pngs) == 2 and "warning" not in events, (events, pngs)
    print("worker OK:", [p.name for p in pngs])


def main():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        build(tmp)
        check_against_diffusers(tmp)
        run_trainer(tmp)
        run_worker(tmp)
        print("KREA2 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
