"""Video (Sulphur 3 / MiniMax H3) smoke test on CPU, with diffusers' official tiny H3 pipeline (44 MB, fetched once).

    python tests/smoke_test_video.py

Sulphur 3 is loaded from two files diffusers can't read by itself, so each loader is checked against another way of
computing the same thing, on tiny stand-ins made here in the real files' formats:
  * the int8 ConvRot text encoder (Comfy-Org's layout, made with the reference quantizer) == the same model with
    exactly dequantized float weights, through the pipeline's own text-encoder call,
  * the pruned "curve" transformer (ComfyUI's key layout, timestep MLP replaced by its table) == the full
    transformer with the file's rounding, on what the pipeline really feeds it,
  * the turbo LoRA (lightx2v's key layout) added at run time == the same deltas merged into the weights, with the
    video schedule on the LoRA's shift while it is on and back on the model's own afterwards,
then the generation worker makes an MP4 with sound from them, with and without turbo.
"""
import copy
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPO = "hf-internal-testing/tiny-minimax-h3-modular-pipe"
H4 = ((1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1))

def ref_hadamard(n):
    h = torch.tensor(H4, dtype=torch.float32)
    while h.shape[0] < n: h = torch.kron(h, torch.tensor(H4, dtype=torch.float32))
    return h / n ** 0.5
def ref_rotate(w, gs):
    o, i = w.shape
    return (w.reshape(o, i // gs, gs) @ ref_hadamard(gs).T).reshape(o, i)
def ref_quantize(w, gs):
    wf = ref_rotate(w.float(), gs)
    scale = (wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-30) / 127.0).clamp(min=1e-30)
    return (wf / scale).round_().clamp_(-127, 127).to(torch.int8), scale

def randomize(model, std):
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim >= 2: p.normal_(0, std)
    return model

def comfy_text_encoder(full, path, gs=16):
    """`full`: the tiny Qwen3VLForConditionalGeneration (51 layers). Writes its first 50 layers as Comfy-Org does."""
    kinds, out = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"), {}
    for k, v in full.state_dict().items():
        if k.startswith("model.visual."): out[k[len("model."):]] = v.to(torch.bfloat16)
        elif k == "model.language_model.embed_tokens.weight": out["model.embed_tokens.weight"] = v.to(torch.bfloat16)
        elif (m := re.match(r"model\.language_model\.layers\.(\d+)\.(.+)$", k)) and int(m.group(1)) < 50:
            name = f"model.layers.{m.group(1)}.{m.group(2)}"
            if name.endswith(".weight") and name.split(".")[-2] in kinds:
                q, s = ref_quantize(v, gs); base = name[:-len(".weight")]
                out[name], out[base + ".weight_scale"] = q, s
                out[base + ".comfy_quant"] = torch.tensor(list(json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": gs}).encode()), dtype=torch.uint8)
            else:
                out[name] = v.to(torch.bfloat16)
    save_file(out, str(path), metadata={"minimax_h3_te": json.dumps({"num_hidden_layers": 50})})

def pruned_transformer(model, path, grid=1025):
    """`model`: the tiny diffusers MiniMaxH3Transformer3DModel. Writes ComfyUI's original key layout with the timestep
    MLP replaced by its curve on `grid` points: table[k] = silu(time_embedder(time_proj(k / (grid - 1)))), so the AdaLN
    weights stay as they are and the curve form is exact on the grid."""
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        t = torch.linspace(0, 1, grid)
        table = torch.nn.functional.silu(model.time_embedder(model.time_proj(t).float()))
    out = {"adaln_t_table": table.float().contiguous(), "rope.inv_freq": torch.zeros(model.config.rope_freq_dim)}
    renames = [("transformer_blocks.", "blocks."), ("token_refiner.refiner_blocks.", "token_refiner.blocks."),
               (".attn.norm_q.", ".attn.q_norm."), (".attn.norm_k.", ".attn.k_norm."), (".attn.to_out.0.", ".attn.out_proj."),
               (".ff.net.0.proj.", ".mlp.fc1."), (".ff.net.2.", ".mlp.fc2.")]
    exact = {"audio_proj_in.": "audio_patch_proj.", "proj_in.": "video_patch_proj.", "context_embedder.": "condition_proj.",
             "norm_out.norm.": "final_layer.norm.", "norm_out.linear.": "final_layer.adaln_proj.linear.",
             "audio_proj_out.": "final_layer.audio_out.", "proj_out.": "final_layer.video_out."}
    for k in sorted(sd):
        if k.startswith(("time_embedder.", "rope.")): continue
        v = sd[k]
        if k.endswith((".attn.to_k.weight", ".attn.to_v.weight")): continue
        new = k
        for a, b in exact.items():
            if new.startswith(a): new = b + new[len(a):]; break
        for a, b in renames: new = new.replace(a, b)
        if k.endswith(".attn.to_q.weight"):
            base = k[:-len("to_q.weight")]
            v = torch.cat([sd[base + "to_q.weight"], sd[base + "to_k.weight"], sd[base + "to_v.weight"]], 0)   # stacked [q; k; v]
            new = new.replace(".attn.to_q.weight", ".attn.qkv_proj.weight")
        if k.endswith(".ff.net.0.proj.weight"):
            value, gate = torch.chunk(v, 2, dim=0); v = torch.cat([gate, value], 0)   # the file fuses [gate; value]
        dt = torch.float16 if ".adaln_proj." in new else torch.float32 if new.split(".")[0] in ("video_patch_proj", "audio_patch_proj") or ".video_out." in new or ".audio_out." in new else torch.bfloat16
        out[new] = v.to(dt).contiguous()
    save_file(out, str(path))
    return table


def turbo_lora(model, path, rank=4, alpha=2.0):
    """A LoRA over the attention and feed-forward projections in the layout of lightx2v's turbo files:
    `<module>.lora_A.default.weight` (PEFT's adapter name in the key) and the alpha in the file's metadata."""
    g, out = torch.Generator().manual_seed(3), {}
    wanted = re.compile(r"(transformer_blocks|refiner_blocks)\.\d+\.(attn\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2))$")
    for name, module in model.named_modules():
        if wanted.search(name):
            out[f"{name}.lora_A.default.weight"] = torch.randn(rank, module.in_features, generator=g) * 0.3
            out[f"{name}.lora_B.default.weight"] = torch.randn(module.out_features, rank, generator=g) * 0.3
    save_file(out, str(path), metadata={"alpha": str(alpha), "key_format": "minimax-h3-diffusers"})
    return len(out) // 2


def tiny_spec(tmp: Path) -> dict:
    return {"tiny-h3": {"label": "tiny H3", "family": "h3", "kind": "video", "trainable": False,
                        "path": str(tmp / "s3_tiny.safetensors"), "text_encoder": str(tmp / "te_int8.safetensors"),
                        "base": REPO, "turbo": {"path": str(tmp / "turbo.safetensors"), "steps": 3, "shift": 6.0},
                        "defaults": {"steps": 3, "seconds": 5}}}


def check_turbo(tmp: Path):
    import mosstrainer.config as config
    from mosstrainer import video

    config.MODEL_SPECS.update(tiny_spec(tmp))
    backend = video.VideoBackend("tiny-h3").load("cpu")
    model = backend.pipe.transformer
    layers = turbo_lora(model, tmp / "turbo.safetensors")
    merged = copy.deepcopy(model)  # the reference: every delta added to its weight
    with torch.no_grad():
        for name, (a, b, scale) in video.read_lora(tmp / "turbo.safetensors").items():
            merged.get_submodule(name).weight += scale * (b @ a)
    assert layers == len(video.read_lora(tmp / "turbo.safetensors")) and scale == 0.5
    own_shift = backend.pipe.scheduler.shift

    def run(turbo):
        backend.set_turbo(turbo)
        calls = []
        hook = backend.pipe.transformer.register_forward_hook(lambda m, a, kw, out: calls.append((a, dict(kw), out)), with_kwargs=True)
        backend.generate("a red fox in the snow", seconds=5, width=64, height=64, steps=3, seed=0)
        hook.remove()
        return calls, backend.pipe.scheduler.shift

    def flat(out):
        return (out[0] if isinstance(out, (tuple, list)) else out).flatten().float()

    calls, shift = run(True)
    times = [round(float(kw["timestep"].flatten()[0]), 4) for _, kw, _ in calls]
    a, kw, _ = calls[-1]
    with torch.no_grad():
        on, want, off_now = flat(model(*a, **kw)), flat(merged(*a, **kw)), None
        backend.set_turbo(False)
        off_now = flat(model(*a, **kw))
        base = copy.deepcopy(merged)
        for name, (la, lb, sc) in video.read_lora(tmp / "turbo.safetensors").items():
            base.get_submodule(name).weight -= sc * (lb @ la)
        d_off = (off_now - flat(base(*a, **kw))).abs().max().item()
    d_on = ((on - want).abs().max() / want.abs().max()).item()
    calls_off, shift_off = run(False)
    times_off = [round(float(kw["timestep"].flatten()[0]), 4) for _, kw, _ in calls_off]
    print(f"turbo LoRA: {layers} layers, on == merged weights {d_on:.1e} (relative), off == base {d_off:.1e}; "
          f"video shift {shift:g} -> timesteps {times}, off {shift_off:g} -> {times_off}")
    # 3 steps on the grid q = 1, 2/3, 1/3: sigma = s q / (1 + (s - 1) q), t = 1 - sigma
    assert d_on < 1e-5 and d_off < 1e-5 and (on - off_now).abs().max() > 1e-4
    assert shift == 6.0 and shift_off == own_shift == 12.0
    assert times == [0.0, 0.0769, 0.25] and times_off == [0.0, 0.04, 0.1429]


def check_text_encoder(tmp: Path):
    from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds
    from mosstrainer.int8_convrot import Int8Linear, load_qwen3vl_int8, regular_hadamard
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    for n in (4, 16, 256):  # the rotation: comfy-kitchen's matrix, symmetric and orthogonal
        h = regular_hadamard(n)
        assert torch.equal(h, ref_hadamard(n)) and torch.allclose(h, h.T) and torch.allclose(h @ h, torch.eye(n), atol=1e-5)
    g = torch.Generator().manual_seed(0)
    w, x = torch.randn(48, 512, generator=g), torch.randn(3, 7, 512, generator=g)
    q, s = ref_quantize(w, 256)
    layer = Int8Linear(q, s, groupsize=256)
    d_layer = (layer(x) - x @ ref_rotate(q.float() * s, 256).T).abs().max().item()
    layer.to(torch.bfloat16)  # a cast meant for the rest of the model leaves the integers and scales alone
    assert d_layer < 1e-4 and layer.weight.dtype == torch.int8 and layer.weight_scale.dtype == torch.float32

    full = randomize(Qwen3VLForConditionalGeneration.from_pretrained(REPO, subfolder="text_encoder", dtype=torch.float32).eval(), 0.2)
    processor = AutoProcessor.from_pretrained(REPO, subfolder="processor")
    comfy_text_encoder(full, tmp / "te_int8.safetensors")
    exact = copy.deepcopy(full)  # the reference: every weight as the file holds it, dequantized
    kinds = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    with torch.no_grad():
        for k, v in exact.state_dict().items():
            m = re.match(r"model\.language_model\.layers\.(\d+)\.", k)
            if m and k.split(".")[-2] in kinds:
                qw, sc = ref_quantize(v, 16)
                v.copy_(ref_rotate(qw.float() * sc, 16))
            elif m or k.startswith(("model.visual.", "model.language_model.embed_tokens")):
                v.copy_(v.to(torch.bfloat16).float())
    te = load_qwen3vl_int8(tmp / "te_int8.safetensors", full.config, "cpu", torch.float32)
    assert sum(isinstance(m, Int8Linear) for m in te.modules()) == 350 and len(te.model.language_model.layers) == 50
    ids = processor.tokenizer("A red fox trotting through a snowy pine forest, snow crunching underfoot")["input_ids"]
    with torch.no_grad():
        got, ref, orig = (get_qwen3vl_prompt_embeds(m, processor, ids, text_encoder_layer=50, dtype=torch.float32)
                          for m in (te, exact, full))
    d_te = (got - ref).abs().max().item() / ref.abs().max().item()
    cos = torch.nn.functional.cosine_similarity(got.flatten(), orig.flatten(), dim=0).item()
    print(f"int8 text encoder: layer {d_layer:.1e}, hidden state vs dequantized weights {d_te:.1e} (relative), "
          f"vs the unquantized model cos {cos:.4f}")
    assert d_te < 2e-3 and cos > 0.98


def check_transformer(tmp: Path):
    from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline
    from huggingface_hub import hf_hub_download
    from mosstrainer import video

    full = randomize(MiniMaxH3Transformer3DModel.from_pretrained(REPO, subfolder="transformer", torch_dtype=torch.float32).eval(), 0.3)
    pruned_transformer(full, tmp / "s3_tiny.safetensors")
    config = json.loads(Path(hf_hub_download(REPO, "transformer/config.json")).read_text())
    curve = video.load_transformer(tmp / "s3_tiny.safetensors", config, "cpu", torch.float32)
    rounded = copy.deepcopy(full)  # the file stores bf16 (f16 for the AdaLN projections)
    with torch.no_grad():
        for name, p in rounded.named_parameters():
            if not (name.startswith("time_embedder.") or "proj_in" in name or "proj_out" in name):
                adaln = ".adaln_proj." in name or name.startswith("norm_out.linear")
                p.copy_(p.to(torch.float16 if adaln else torch.bfloat16).float())
    pipe = ModularPipeline.from_pretrained(REPO)
    pipe.load_components(workflow="t2va", dtype=torch.float32)
    pipe.update_components(transformer=full)
    calls = []
    hook = full.register_forward_hook(lambda m, a, kw, out: calls.append((a, kw)), with_kwargs=True)
    pipe(prompt="a red fox in the snow", num_frames=124, num_inference_steps=4, height=64, width=64,
         generator=torch.Generator().manual_seed(0), output=["videos"])
    hook.remove()

    def flat(out):
        return (out[0] if isinstance(out, (tuple, list)) else out).flatten().float()

    worst = 0.0
    with torch.no_grad():
        for a, kw in calls:
            kw = dict(kw, timestep=(kw["timestep"] * 1024).round() / 1024)  # on the table's grid the curve is exact
            worst = max(worst, ((flat(curve(*a, **kw)) - flat(rounded(*a, **kw))).abs().max()
                                / flat(rounded(*a, **kw)).abs().max()).item())
    print(f"pruned transformer: {len(calls)} calls replayed, vs the full model {worst:.1e} (relative)")
    assert len(calls) == 3 and worst < 1e-4


def run_worker(tmp: Path):
    """An MP4 through the Generate tab's worker process, from the two stand-in files."""
    import av
    (tmp / "models.json").write_text(json.dumps(tiny_spec(tmp)))
    env = {**os.environ, "MOSS_EXTRA_MODELS": str(tmp / "models.json")}
    proc = subprocess.Popen([sys.executable, "-m", "mosstrainer.generate_worker"], cwd=ROOT, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
    names = []
    for rid, turbo in (("v1", False), ("v2", True)):
        req = {"cmd": "generate", "kind": "video", "id": rid, "model": "tiny-h3", "prompt": "a red fox in the snow",
               "seconds": 5, "width": 64, "height": 64, "steps": 3, "seed": 5, "turbo": turbo, "out_dir": str(tmp / "gen")}
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        events = []
        for line in proc.stdout:
            ev = json.loads(line)
            events.append(ev)
            if ev.get("event") in ("done", "error"):
                break
        assert events[-1]["event"] == "done", events[-1]
        steps = [(e["step"], e["steps"]) for e in events if e.get("event") == "progress"]
        name = next(e["file"] for e in events if e.get("event") == "image")
        meta = json.loads((tmp / "gen" / "videos" / name).with_suffix(".json").read_text())
        with av.open(str(tmp / "gen" / "videos" / name)) as c:
            frames, audio = sum(1 for _ in c.decode(video=0)), len(c.streams.audio)
        assert steps == [(1, 3), (2, 3), (3, 3)] and frames == meta["frames"] == 124 and audio == 1 and meta["seed"] == 5
        assert meta["turbo"] is turbo and (tmp / "gen" / "videos" / name).with_suffix(".thumb.jpg").exists()
        names.append(name)
    proc.terminate()
    sizes = [(tmp / "gen" / "videos" / n).stat().st_size for n in names]
    assert sizes[0] != sizes[1], "turbo made no difference"
    print(f"worker OK: {names[0]} and {names[1]} (turbo), {frames} frames at {meta['fps']} fps with sound")


def main():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        check_text_encoder(tmp)
        check_transformer(tmp)
        check_turbo(tmp)
        run_worker(tmp)
        print("VIDEO SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
