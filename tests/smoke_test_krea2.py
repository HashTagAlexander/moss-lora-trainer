"""Krea 2 / Kroma smoke test on CPU with tiny random models (needs network once for the Qwen3-VL tokenizer).

    python tests/smoke_test_krea2.py

Checks, against diffusers' own Krea2Pipeline:
  * our Qwen3-VL prompt encoding == Krea2Pipeline.get_text_hidden_states (valid tokens),
  * original-format (Kroma/ComfyUI) checkpoint conversion + loading,
  * our sampler == Krea2Pipeline denoising for the same start latents (CFG convention included),
  * saved LoRAs load with diffusers' Krea2 LoRA loader and give the same output as our runtime stack,
  * the Generate tab's extras against independent formulations: in-context references, isolated references
    (== a K/V cache pass), a control-LoRA's input (== a widened first layer), regional prompts and inpainting,
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
    keep = ["tokenizer*", "vocab.json", "merges.txt", "*.jinja", "*preprocessor_config.json"]
    tok = snapshot_download("Qwen/Qwen3-VL-4B-Instruct", allow_patterns=keep)
    cfg = Qwen3VLConfig(
        text_config=dict(hidden_size=TD, num_hidden_layers=36, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         intermediate_size=128, vocab_size=151936,
                         rope_parameters={"rope_type": "default", "rope_theta": 5000000, "mrope_section": [2, 3, 3],
                                          "mrope_interleaved": True}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=TD,
                           deepstack_visual_indexes=[0]))
    torch.manual_seed(0)
    Qwen3VLModel(cfg).save_pretrained(te_dir)
    for f in Path(tok).iterdir():  # the snapshot folder can also hold other files fetched earlier (the real config)
        if any(f.match(p) for p in keep):
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


def check_extras(tmp: Path):
    """Inpainting, control images and regional prompts (Krea2Backend._forward_ctx), each against another way of
    computing the same thing."""
    import copy

    import torch.nn.functional as Fn
    from diffusers.models.transformers.transformer_krea2 import apply_rotary_emb
    from mosstrainer.backends import get_backend
    be = get_backend("tiny-krea2")
    model = be.load_transformer("cpu").float()
    g = torch.Generator().manual_seed(1)

    def rn(*shape):
        return torch.randn(*shape, generator=g)

    gh, gw, n_txt = 8, 12, 7
    x, ref, sig = rn(1, 16, 16, 24), rn(1, 16, 16, 24), torch.tensor([0.6])
    txt, tmask = rn(1, n_txt, 12, TD), torch.ones(1, n_txt, dtype=torch.bool)
    ids = torch.cat([torch.zeros(n_txt, 3), be._grid_ids(gh, gw, 0, "cpu")])

    def kv_cached():
        """ai-toolkit's kv_cache: the reference alone at t=0, its keys and values reused by the denoising pass."""
        def embed(t):
            return model.time_embed(t, dtype=torch.float32)

        def block(blk, h, mod, rope, cached=None, keep=None):
            ps, psh, pg, qs, qsh, qg = (mod.unflatten(-1, (6, -1)) + blk.scale_shift_table).unbind(-2)
            a, hin = blk.attn, (1 + ps) * blk.norm1(h) + psh
            q = apply_rotary_emb(a.norm_q(a.to_q(hin).unflatten(-1, (a.num_heads, a.head_dim))), rope, sequence_dim=1)
            k = apply_rotary_emb(a.norm_k(a.to_k(hin).unflatten(-1, (a.num_kv_heads, a.head_dim))), rope, sequence_dim=1)
            v = a.to_v(hin).unflatten(-1, (a.num_kv_heads, a.head_dim))
            rep = a.num_heads // a.num_kv_heads
            q, k, v = q.transpose(1, 2), k.repeat_interleave(rep, 2).transpose(1, 2), v.repeat_interleave(rep, 2).transpose(1, 2)
            if keep is not None:
                keep.append((k, v))
            if cached is not None:
                k, v = torch.cat([k, cached[0]], 2), torch.cat([v, cached[1]], 2)
            out = Fn.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2, 3)
            h = h + pg * a.to_out[0](out * torch.sigmoid(a.to_gate(hin)))
            return h + qg * blk.ff((1 + qs) * blk.norm2(h) + qsh)

        mod = model.time_mod_proj(Fn.gelu(embed(sig), approximate="tanh"))
        mod0 = model.time_mod_proj(Fn.gelu(embed(torch.zeros_like(sig)), approximate="tanh"))
        h, cache = model.img_in(be._pack(ref)), []
        for blk in model.transformer_blocks:
            h = block(blk, h, mod0, model.rotary_emb(be._grid_ids(gh, gw, 1, "cpu")), keep=cache)
        h = torch.cat([model.txt_in(model.text_fusion(txt)), model.img_in(be._pack(x))], 1)
        for blk, kv in zip(model.transformer_blocks, cache):
            h = block(blk, h, mod, model.rotary_emb(ids), cached=kv)
        return be._unpack(model.final_layer(h[:, n_txt:], embed(sig)), 16, 24)

    with torch.no_grad():
        stock = be.predict(model, x, sig, txt, tmask)
        # a plain in-context reference == the stock model over [target | reference], the reference on frame 1
        both = torch.cat([ids, be._grid_ids(gh, gw, 1, "cpu")])
        want = model(hidden_states=torch.cat([be._pack(x), be._pack(ref)], 1), encoder_hidden_states=txt, timestep=sig,
                     position_ids=both, return_dict=False)[0][:, :gh * gw]
        plain = be.predict(model, x, sig, txt, tmask, context={"refs": [ref], "ref_t0": False})
        assert torch.allclose(plain, be._unpack(want, 16, 24), atol=1e-5), "in-context reference"
        # isolated references == the K/V cache pass
        iso = be.predict(model, x, sig, txt, tmask, context={"refs": [ref], "isolate": True})
        d_iso = (iso - kv_cached()).abs().max().item()
        assert d_iso < 1e-5 and (iso - be.predict(model, x, sig, txt, tmask, context=[ref])).abs().max() > 1e-6
        # a control-LoRA's input == its widened first layer [base | control] on concatenated tokens
        ctrl, wc = be._pack(rn(1, 16, 16, 24)), rn(F, 64) * 0.05
        wide = copy.deepcopy(model)
        wide.img_in = torch.nn.Linear(128, F)
        wide.img_in.weight.data = torch.cat([model.img_in.weight.data, wc], 1)
        wide.img_in.bias.data = model.img_in.bias.data.clone()
        want = wide(hidden_states=torch.cat([be._pack(x), ctrl], -1), encoder_hidden_states=txt, timestep=sig,
                    position_ids=ids, return_dict=False)[0]
        out = be.predict(model, x, sig, txt, tmask, context={"control": (ctrl, wc)})
        d_ctrl = (out - be._unpack(want, 16, 24)).abs().max().item()
        assert d_ctrl < 1e-5 and (out - stock).abs().max() > 1e-6
        # regions: after one block, a region's prompt has reached its own image tokens and nothing else
        boxes = [{"x": 0, "y": 0, "w": 0.5, "h": 1}, {"x": 0.5, "y": 0, "w": 0.5, "h": 1}]
        left, right = be.region_tokens(boxes, 24 * 8, 16 * 8)
        assert left.view(gh, gw)[:, :6].all() and not left.view(gh, gw)[:, 6:].any() and (left ^ right).all()
        ra, seen = rn(5, 12, TD), []
        hook = model.transformer_blocks[0].register_forward_hook(lambda m, i, o: seen.append(o[0, n_txt + 5 + 6:]))
        for rb in (rn(6, 12, TD), rn(6, 12, TD)):
            be.predict(model, x, sig, txt, tmask, context={"regions": [(ra, left), (rb, right)]})
        hook.remove()
        moved = (seen[0] - seen[1]).abs()
        assert moved[left].max() < 1e-7 and moved[right].max() > 1e-6, "regional prompt leaked"
    # inpainting without a LoRA: what the mask leaves alone comes back as the source's latents exactly
    vae = be.load_vae("cpu")
    known, mask = rn(1, 16, 16, 24), torch.zeros(1, 1, 16, 24)
    mask[..., 12:] = 1
    lat = be.sample(model, vae, rn(7, 12, TD).to(torch.bfloat16), rn(4, 12, TD).to(torch.bfloat16), width=192, height=128,
                    steps=3, cfg=2.0, seed=5, shift=0.0, skip_layer=-1, device="cpu", output="latent",
                    inpaint=(known, mask), context={"regions": [(ra, left)]})
    assert torch.allclose(lat[..., :12], known[..., :12], atol=1e-5) and not torch.allclose(lat[..., 12:], known[..., 12:], atol=1e-2)
    print(f"extras: isolated refs vs K/V cache {d_iso:.1e} | control input vs widened layer {d_ctrl:.1e} | "
          "in-context, regions and inpainting OK")


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
        check_extras(tmp)
        run_trainer(tmp)
        run_worker(tmp)
        print("KREA2 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
