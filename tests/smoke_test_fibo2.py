"""Fibo 2.0 smoke test on CPU with tiny random models (needs network once for the Qwen3-VL processor files).

    python tests/smoke_test_fibo2.py

Checks, against diffusers' own BriaFibo2Pipeline:
  * our sampler == the pipeline's denoising for a VGL caption with text_render (text images, the switch to the
    caption without them above noise level 0.9, the default null-caption negative and CFG),
  * editing a source image (prepare_edit + encode with images + context latents) == the pipeline's edit path,
  * our VAE decode == the pipeline's,
  * LoRA on every Linear layer, saved in diffusers format, loads with `transformer.load_lora_adapter` and gives the
    same output as the injected LoRA and our runtime stack,
plus the VGL helpers (trigger word in JSON captions, caption tidying), then runs the real trainer process (caching,
training with padded batches, sampling, saving, resume) and the Generate tab's worker (text-to-image and an edit).
Edit LoRAs: the cached inputs of an edit pair (prompt read with two source images, source and target latents) are
checked against Bria's fine_tuning/train_lora.py pre-encoding, then an edit-pair dataset is trained and resumed.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TD = 64  # text encoder hidden size; the transformer reads bundles of three layers: 3 * TD
TINY = dict(in_channels=32, patch_size=2, dim=64, n_layers=5, n_refiner_layers=1, n_heads=2, cap_feat_dim=3 * TD,
            injection_layer_ids=[0, 1, 2, 3, 4], perceiver_num_layers=1, min_num_gist_tokens=4,
            max_num_gist_tokens=8, gist_step=4, gist_min_text_len=16, gist_max_text_len=128, axes_dims=[8, 12, 12],
            axes_lens=[64, 64, 64])
CAPTION = {
    "short_description": "A red bicycle leaning against a white brick wall.",
    "objects": [{"description": "a red city bicycle", "location": "center", "texture": None}],
    "background_setting": "a white brick wall",
    "lighting": {},
    "style_medium": "photograph",
    "text_render": [{"text": "OPEN", "location": "top-left", "size": "small", "color": "green", "font": "serif"}],
    "aesthetic_score": 7.5,
}


def build(tmp: Path):
    from diffusers import AutoencoderKLFlux2, BriaFibo2Transformer2DModel, FlowMatchEulerDiscreteScheduler
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration

    import mosstrainer.config as C
    root = tmp / "tiny-fibo2"
    torch.manual_seed(0)
    BriaFibo2Transformer2DModel(**TINY).save_pretrained(root / "transformer")
    vae = AutoencoderKLFlux2(block_out_channels=[16, 16, 16, 16], layers_per_block=1, norm_num_groups=8,
                             latent_channels=32)
    vae.bn.running_mean.normal_(0, 0.3)
    vae.bn.running_var.uniform_(0.5, 2.0)
    vae.save_pretrained(root / "vae")
    cfg = Qwen3VLConfig(
        text_config=dict(hidden_size=TD, num_hidden_layers=36, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         intermediate_size=128, vocab_size=151936, tie_word_embeddings=True,
                         rope_parameters={"rope_type": "default", "rope_theta": 5000000, "mrope_section": [2, 3, 3],
                                          "mrope_interleaved": True}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=TD, patch_size=16,
                           spatial_merge_size=2, temporal_patch_size=2, deepstack_visual_indexes=[0]))
    Qwen3VLForConditionalGeneration(cfg).save_pretrained(root / "text_encoder")
    src = snapshot_download("Qwen/Qwen3-VL-4B-Instruct", allow_patterns=["*.json", "*.txt", "*.jinja"])
    AutoProcessor.from_pretrained(src).save_pretrained(root / "processor")
    FlowMatchEulerDiscreteScheduler(shift=3.0, use_dynamic_shifting=True, base_shift=0.5, max_shift=1.15,
                                    base_image_seq_len=256, max_image_seq_len=4096,
                                    time_shift_type="exponential").save_pretrained(root / "scheduler")
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "BriaFibo2Pipeline", "processor": ["transformers", "Qwen3VLProcessor"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
        "transformer": ["diffusers", "BriaFibo2Transformer2DModel"], "vae": ["diffusers", "AutoencoderKLFlux2"]}))
    spec = {"tiny-fibo2": {"label": "tiny Fibo 2.0", "family": "fibo2", "trainable": True, "edit": True,
                           "path": str(root), "overrides": C._FIBO_OVERRIDES,
                           "defaults": {"steps": 3, "cfg": 3.0, "shift": 0.0, "skip_layer": -1,
                                        "timestep_sampling": "fibo", "negative": ""}}}
    (tmp / "models.json").write_text(json.dumps(spec))
    os.environ["MOSS_EXTRA_MODELS"] = str(tmp / "models.json")
    C.MODEL_SPECS.update(spec)


def check_helpers():
    from mosstrainer import captioner
    from mosstrainer.data import apply_trigger

    cap = json.dumps({"short_description": "a woman reading", "style_medium": "photograph"})
    assert json.loads(apply_trigger(cap, "zxq"))["short_description"] == "zxq, a woman reading"
    assert json.loads(apply_trigger(cap.replace("a woman", "[trigger] woman"), "zxq"))["short_description"] == "zxq woman reading"
    assert json.loads(apply_trigger(cap.replace("a woman", "[trigger], a woman"), ""))["short_description"] == "a woman reading"
    assert apply_trigger("a woman reading", "zxq") == "zxq, a woman reading"
    reply = "```json\n" + json.dumps({"context": "c", "short_description": "s", "lighting": {"shadows": "soft",
                                      "conditions": "day", "direction": ""}, "objects": [{"location": "center",
                                      "description": "d", "pose": None}], "text_render": [], "artistic_style": None}) + "\n```"
    out = captioner.parse_reply(reply)
    assert list(out) == ["short_description", "objects", "lighting", "text_render", "context"], list(out)
    assert out["objects"] == [{"description": "d", "location": "center"}]
    assert list(out["lighting"]) == ["conditions", "shadows"]
    print("helpers OK")


def check_against_diffusers(tmp: Path):
    import diffusers
    from diffusers import BriaFibo2Pipeline, FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.bria_fibo2.pipeline_bria_fibo2 import DEFAULT_NEGATIVE_PROMPT
    from mosstrainer import lora as L
    from mosstrainer.backends import FIBO_NULL_CAPTION, get_backend
    from mosstrainer.model import pad_embeds

    be = get_backend("tiny-fibo2")
    assert FIBO_NULL_CAPTION == DEFAULT_NEGATIVE_PROMPT and be.empty_prompt.endswith(DEFAULT_NEGATIVE_PROMPT)
    te = be.load_text_encoder("cpu")
    te.model.float()
    te.out_dtype = torch.float32
    model = be.load_transformer("cpu").float()
    vae = be.load_vae("cpu")
    pipe = BriaFibo2Pipeline(transformer=model, vae=vae, text_encoder=te.model, processor=te.pipe.processor,
                             scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(be.root, subfolder="scheduler"))

    # text-to-image with text images: prompt in any JSON form, early caption above noise 0.9, null-caption negative
    prompt = json.dumps(CAPTION, indent=1)
    main, early = be.prompt_parts(prompt)
    assert early is not None
    pos, ear, neg = te.encode(main), te.encode(early), te.encode(be.negative_prompt(""))
    assert pos.shape[1:] == (6, 3 * TD) and pos.shape[0] > ear.shape[0], (pos.shape, ear.shape)
    start = torch.randn((1, 32, 32, 32), generator=torch.Generator().manual_seed(3))
    ours = be.sample(model, vae, pos, neg, width=256, height=256, steps=4, cfg=3.0, seed=0, shift=0.0, skip_layer=-1,
                     device="cpu", latents=start, output="latent", early=ear)
    ref = pipe(prompt=prompt, height=256, width=256, num_inference_steps=4, guidance_scale=3.0, latents=start,
               output_type="latent").images
    print("t2i sampler: max diff vs BriaFibo2Pipeline", (ours - ref).abs().max().item())
    assert torch.allclose(ours, ref, atol=1e-4), "sampler diverges from diffusers"
    no_early = be.sample(model, vae, pos, neg, width=256, height=256, steps=4, cfg=3.0, seed=0, shift=0.0,
                         skip_layer=-1, device="cpu", latents=start, output="latent")
    assert not torch.allclose(no_early, ref, atol=1e-4), "the caption without text images should matter"

    # decode
    ours_img = (be.decode(vae, ref) + 1) / 2
    ref_img = pipe(prompt=prompt, height=256, width=256, num_inference_steps=4, guidance_scale=3.0, latents=start,
                   output_type="pt").images
    print("decode: max diff", (ours_img - ref_img).abs().max().item())
    assert torch.allclose(ours_img, ref_img.float(), atol=1e-4)

    # editing a source image
    g = torch.Generator().manual_seed(1)
    src = Image.fromarray((torch.rand(200, 300, 3, generator=g) * 255).byte().numpy())
    ctx, thumbs, w, h = be.prepare_edit(vae, src, 256, 256)
    instruction = json.dumps({"short_description": "a blue square", "edit_instruction": "make it blue"})
    pos = te.encode(instruction, images=thumbs)
    start = torch.randn((1, 32, h // 8, w // 8), generator=g)
    ours = be.sample(model, vae, pos, neg, width=w, height=h, steps=3, cfg=2.0, seed=0, shift=0.0, skip_layer=-1,
                     device="cpu", latents=start, output="latent", context=ctx)
    ref = pipe(prompt=instruction, image=src, height=h, width=w, num_inference_steps=3, guidance_scale=2.0,
               latents=start, output_type="latent").images
    print(f"edit sampler ({w}x{h}): max diff vs BriaFibo2Pipeline", (ours - ref).abs().max().item())
    assert torch.allclose(ours, ref, atol=1e-4), "edit path diverges from diffusers"
    try:
        te.encode('{"short_description": "no instruction"}', images=thumbs)
        raise AssertionError("an edit caption without edit_instruction should be refused")
    except ValueError:
        pass

    # LoRA on every Linear layer: injected == runtime stack == diffusers' loader
    linears = sum(isinstance(m, torch.nn.Linear) for m in model.modules())
    mods = L.inject(model, 4, 8, ("attn", "ffn", "text", "other"), target_map=be.lora_targets, layer_re=be.lora_layer_re)
    assert len(mods) == linears, (len(mods), linears)
    for m in mods.values():
        torch.nn.init.normal_(m.lora_B.weight, std=0.05)
    L.save(mods, tmp / "l.safetensors", {}, torch.float32, be.lora_format, be.lora_arch)
    x = torch.randn(2, 32, 16, 16)
    sig = torch.tensor([0.6, 0.3])
    e, msk = pad_embeds([te.encode("a cat"), te.encode(main)])
    assert not msk.all()
    with torch.no_grad():
        injected = be.predict(model, x, sig, e, msk)
    for n, m in mods.items():
        L._replace(model, n, m.base)
    stack = L.LoRAStack(model, name_map=be.lora_name_map)
    stack.activate([(tmp / "l.safetensors", 1.0)], "cpu", torch.float32)
    with torch.no_grad():
        stacked = be.predict(model, x, sig, e, msk)
    stack.activate([], "cpu")
    for n, m in stack.wrapped.items():
        L._replace(model, n, m.base)
    model.load_lora_adapter(str(tmp), weight_name="l.safetensors", adapter_name="t", prefix="transformer")
    with torch.no_grad():
        diff = be.predict(model, x, sig, e, msk)
    print("lora: modules", len(mods), "| stack vs injected", (stacked - injected).abs().max().item(),
          "| diffusers loader vs injected", (diff - injected).abs().max().item(), "| diffusers", diffusers.__version__)
    assert torch.allclose(stacked, injected, atol=1e-4) and torch.allclose(diff, injected, atol=1e-4)


def run_trainer(tmp: Path):
    ws = tmp / "workspace"
    ds = ws / "datasets" / "toy"
    ds.mkdir(parents=True)
    caps = [json.dumps(CAPTION, indent=2), json.dumps({"short_description": "a flat blue swatch",
                                                       "style_medium": "digital illustration"}), "a plain green swatch"]
    for i, size in enumerate([(320, 256), (256, 320), (256, 256)]):
        Image.new("RGB", size, (70 * i, 120, 200 - 50 * i)).save(ds / f"img{i}.png")
        (ds / f"img{i}.txt").write_text(caps[i])
    (ds / "img3.png").write_bytes((ds / "img2.png").read_bytes())
    (ds / "img3.txt").write_text(caps[1])
    job = ws / "jobs" / "f2"
    job.mkdir(parents=True)
    (job / "config.json").write_text(json.dumps({
        "name": "f2", "model": "tiny-fibo2", "dataset": "toy", "trigger_word": "zxq",
        "network": {"rank": 4, "alpha": 4}, "data": {"resolutions": [256]},
        "train": {"steps": 4, "batch_size": 2, "lr": 1e-3}, "save": {"every": 2},
        "sample": {"every": 2, "width": 256, "height": 256}}))

    def go():
        r = subprocess.run([sys.executable, "-m", "mosstrainer.train", "--job-dir", str(job)], cwd=ROOT,
                           capture_output=True, text=True, env=os.environ.copy())
        print(r.stdout[-2500:], r.stderr[-2500:])
        return json.loads((job / "status.json").read_text()), r.stdout
    st, out = go()
    assert st["state"] == "done", st
    assert "LoRA: " in out and "modules" in out
    cfg = json.loads((job / "config.json").read_text())
    cfg["train"]["steps"] = 6
    (job / "config.json").write_text(json.dumps(cfg))
    st, out = go()
    assert st["state"] == "done" and st["step"] == 6 and "resumed from step 4" in out, st
    from safetensors import safe_open
    with safe_open(str(job / "checkpoints" / "f2.safetensors"), "pt") as f:
        keys, meta = list(f.keys()), f.metadata()
    assert keys and all(k.startswith("transformer.") for k in keys) and not any("alpha" in k for k in keys)
    assert any(".perceiver." in k for k in keys) and any("adaLN_modulation" in k for k in keys)
    assert meta["modelspec.architecture"] == "fibo2/lora"
    samples = json.loads((job / "samples" / "index.json").read_text())
    assert samples and all(s["prompt"].startswith('{"short_description":"zxq') for s in samples), samples[0]
    print("trainer OK, LoRA keys e.g.", keys[0])


EDIT_ROWS = [
    {"sources": ["src/a.png", "src/b.png"], "target": "tgt/ab.png",
     "instruction": {"short_description": "the two squares side by side", "edit_instruction": "put <image_1> next to "
                     "<image_2>", "text_render": [{"text": "PAIR", "location": "top", "size": "small", "color": "white",
                                                   "font": "sans"}]}},
    {"sources": ["src/a.png"], "target": "tgt/a_blue.png", "instruction": "make the square blue"},
    {"sources": ["src/missing.png"], "target": "tgt/a_blue.png", "instruction": "skipped: the source is missing"},
]


def make_edit_dataset(ds: Path):
    (ds / "src").mkdir(parents=True)
    (ds / "tgt").mkdir()
    g = torch.Generator().manual_seed(4)
    noise = lambda w, h: Image.fromarray((torch.rand(h, w, 3, generator=g) * 255).byte().numpy())  # noqa: E731
    noise(300, 200).save(ds / "src/a.png")
    noise(200, 260).save(ds / "src/b.png")
    noise(280, 210).save(ds / "tgt/ab.png")
    noise(300, 200).save(ds / "tgt/a_blue.png")
    (ds / "edits.jsonl").write_text("".join(json.dumps(r) + "\n" for r in EDIT_ROWS))


def check_edit_against_bria(tmp: Path):
    """Our cached edit inputs == Bria's trainer pre-encoding (pre_encode in fine_tuning/train_lora.py)."""
    from diffusers import BriaFibo2Pipeline, FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.bria_fibo2.text_images import draw_text_image
    from mosstrainer.backends import get_backend
    from mosstrainer.data import build_edit_items, cache_edit_embeds, cache_edit_latents, edit_cache_paths, load_edit

    ds = tmp / "edit_check"
    make_edit_dataset(ds)
    be = get_backend("tiny-fibo2")
    te = be.load_text_encoder("cpu")
    te.model.float()
    te.out_dtype = torch.float32
    vae = be.load_vae("cpu")
    logs = []
    items = build_edit_items(ds, [256], "", logs.append)
    assert len(items) == 2 and "pair 3 skipped" in logs[0], (items, logs)
    edit_cache_paths(items, tmp / "edit_cache", be.text_cache_id)
    cache_edit_embeds(items, be, te)
    cache_edit_latents(items, be, vae)

    pipe = BriaFibo2Pipeline(transformer=None, vae=vae, text_encoder=te.model, processor=te.pipe.processor,
                             scheduler=FlowMatchEulerDiscreteScheduler())
    proc, area = pipe.image_processor, 256 * 256
    for it, row in zip(items, EDIT_ROWS):
        ours = load_edit(it)
        srcs = [Image.open(ds / p).convert("RGB") for p in row["sources"]]
        target = Image.open(ds / row["target"]).convert("RGB")
        prompt = row["instruction"] if isinstance(row["instruction"], str) else json.dumps(
            row["instruction"], separators=(",", ":"), ensure_ascii=False)
        width, height = proc.get_image_size(srcs[0].width / srcs[0].height, area=area)
        vae_srcs, thumbs = proc.preprocess(srcs, height=height, width=width, area=area,
                                           text_encoder_image_size=pipe.text_encoder_image_size)
        caption, _ = pipe._parse_prompt(prompt)
        texts = [draw_text_image(e["text"], []) for e in (caption.get("text_render") or [])] \
            if isinstance(caption, dict) else []
        with torch.no_grad():
            emb, mask = pipe.encode_prompt(prompt, thumbs, torch.device("cpu"), text_images=texts or None)
            refs = pipe.prepare_image_latents(vae_srcs, 1, torch.device("cpu"), torch.float32)
            tgt, _ = proc.preprocess(target, height=height, width=width)
            x0 = pipe.prepare_image_latents(tgt, 1, torch.device("cpu"), torch.float32)[0]
        ref_emb = emb[0].transpose(0, 1) if mask is None else emb[0][:, mask[0]].transpose(0, 1)
        assert it.bucket == (width, height), (it.bucket, width, height)
        assert ours["embeds"].shape == ref_emb.shape and torch.equal(ours["embeds"], ref_emb.float())
        assert len(ours["context"]) == len(refs) and all(torch.equal(a, b[0]) for a, b in zip(ours["context"], refs))
        assert torch.equal(ours["x0"], x0[0])
        print(f"edit pair ({len(srcs)} source(s), {width}x{height}): prompt tokens {ref_emb.shape[0]}, "
              f"context {[tuple(r.shape[2:]) for r in refs]} == Bria's pre-encoding")


def run_edit_trainer(tmp: Path):
    ws = tmp / "workspace"
    make_edit_dataset(ws / "datasets" / "pairs")
    job = ws / "jobs" / "ed"
    job.mkdir(parents=True)
    (job / "config.json").write_text(json.dumps({
        "name": "ed", "model": "tiny-fibo2", "dataset": "pairs", "network": {"rank": 4, "alpha": 4},
        "data": {"resolutions": [256], "context_dropout": 0.5}, "train": {"steps": 3, "batch_size": 2, "lr": 1e-3},
        "save": {"every": 2}, "sample": {"every": 3, "width": 256, "height": 256, "steps": 2}}))

    def go():
        r = subprocess.run([sys.executable, "-m", "mosstrainer.train", "--job-dir", str(job)], cwd=ROOT,
                           capture_output=True, text=True, env=os.environ.copy())
        print(r.stdout[-2000:], r.stderr[-2000:])
        return json.loads((job / "status.json").read_text()), r.stdout
    st, out = go()
    assert st["state"] == "done", st
    assert "2 edit pairs" in out and "pair 3 skipped" in out and "becomes gradient accumulation" in out, out
    samples = json.loads((job / "samples" / "index.json").read_text())
    assert [s["sources"] for s in samples[:2]] == [["src/a.png", "src/b.png"], ["src/a.png"]], samples
    assert len({s["step"] for s in samples}) == 2, samples
    with Image.open(job / "samples" / samples[-1]["file"]) as im:
        assert im.width > im.height, im.size  # source and result side by side
    cfg = json.loads((job / "config.json").read_text())
    cfg["train"]["steps"] = 4
    cfg["sample"].update(prompts=["make it green"], sources=["src/b.png, src/a.png"])
    (job / "config.json").write_text(json.dumps(cfg))
    st, out = go()
    assert st["state"] == "done" and st["step"] == 4 and "resumed from step 3" in out, st
    samples = json.loads((job / "samples" / "index.json").read_text())
    assert samples[-1]["sources"] == ["src/b.png", "src/a.png"] and samples[-1]["prompt"] == "make it green", samples[-1]
    print("edit trainer OK:", sorted(p.name for p in (job / "checkpoints").glob("*.safetensors")))


def run_worker(tmp: Path):
    """Generate through the Generate tab's worker: text-to-image with the LoRA, then an edit of a source image."""
    lora = tmp / "workspace" / "jobs" / "f2" / "checkpoints" / "f2.safetensors"
    src = tmp / "src.png"
    Image.new("RGB", (300, 200), (200, 40, 40)).save(src)
    proc = subprocess.Popen([sys.executable, "-m", "mosstrainer.generate_worker"], cwd=ROOT, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=os.environ.copy())
    base = {"cmd": "generate", "model": "tiny-fibo2", "negative": "", "width": 256, "height": 256, "steps": 2,
            "cfg": 3.0, "shift": 0.0, "seed": 5, "count": 1, "skip_layer": -1,
            "loras": [{"path": str(lora), "strength": 0.8, "name": "f2"}], "out_dir": str(tmp / "gen")}
    reqs = [{**base, "id": "t1", "prompt": json.dumps(CAPTION), "count": 2},
            {**base, "id": "t2", "prompt": "make the square blue", "image": str(src), "image_id": "src.png"}]
    for req in reqs:
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        events = []
        for line in proc.stdout:
            ev = json.loads(line)
            events.append(ev["event"])
            if ev["event"] in ("done", "error"):
                assert ev["event"] == "done", ev
                break
        assert "warning" not in events, events
    proc.terminate()
    pngs = sorted((tmp / "gen").glob("*.png"))
    metas = [json.loads(p.with_suffix(".json").read_text()) for p in pngs]
    edit = [m for m in metas if m.get("image_id")]
    assert len(pngs) == 3 and len(edit) == 1, (pngs, metas)
    assert (edit[0]["width"], edit[0]["height"]) != (256, 256), "an edit keeps the source's aspect ratio"
    print("worker OK:", [p.name for p in pngs], "edit size", edit[0]["width"], edit[0]["height"])


def main():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        build(tmp)
        check_helpers()
        check_against_diffusers(tmp)
        run_trainer(tmp)
        run_worker(tmp)
        check_edit_against_bria(tmp)
        run_edit_trainer(tmp)
        print("FIBO2 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
