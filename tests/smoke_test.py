"""End-to-end smoke test on CPU with a tiny random MOSS-Image model (no downloads).

    python tests/smoke_test.py

Builds a tiny transformer + FLUX.2-style VAE, a 4-image dataset and a pre-filled text cache,
then runs the real trainer process: caching, training, sampling, saving, stop and resume.
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
from mosstrainer.data import TextCache, apply_trigger  # noqa: E402
from mosstrainer.model import MOSSImageTransformer  # noqa: E402

TEXT_DIM = 32


def build(tmp: Path):
    from diffusers import AutoencoderKLFlux2
    model = tmp / "tiny-moss"
    torch.manual_seed(0)
    MOSSImageTransformer(dim=64, ffn_dim=128, num_heads=2, num_layers=4, text_dim=TEXT_DIM).save_pretrained(model / "transformer")
    AutoencoderKLFlux2(block_out_channels=(16, 16, 16, 16), layers_per_block=1, norm_num_groups=8,
                       latent_channels=32).save_pretrained(model / "vae")
    ws = tmp / "workspace"
    ds = ws / "datasets" / "toy"
    ds.mkdir(parents=True)
    for i, size in enumerate([(320, 256), (256, 320), (256, 256), (300, 300)]):
        Image.new("RGB", size, (60 * i, 120, 200 - 40 * i)).save(ds / f"img{i}.png")
        (ds / f"img{i}.txt").write_text(f"a flat color swatch number {i}")
    (tmp / "models.json").write_text(json.dumps({"tiny-moss": {
        "label": "tiny MOSS", "family": "moss", "trainable": True, "path": str(model),
        "defaults": {"steps": 3, "cfg": 3.0, "shift": 5.0, "skip_layer": 22, "timestep_sampling": "moss"}}}))
    os.environ["MOSS_EXTRA_MODELS"] = str(tmp / "models.json")
    cfg = {
        "name": "toy", "model": "tiny-moss", "dataset": "toy", "trigger_word": "zxq",
        "network": {"rank": 4, "alpha": 4, "targets": ["attn", "ffn"]},
        "data": {"resolutions": [256], "caption_dropout": 0.2},
        "train": {"steps": 6, "batch_size": 2, "lr": 1e-3, "gradient_checkpointing": True, "timestep_sampling": "moss"},
        "save": {"every": 3, "keep_last": 2},
        "sample": {"every": 3, "prompts": ["[trigger] swatch"], "width": 256, "height": 256, "steps": 3},
    }
    job = ws / "jobs" / "toy"
    job.mkdir(parents=True)
    (job / "config.json").write_text(json.dumps(cfg))
    # Pre-fill the text cache so the 9B Qwen encoder is never needed.
    cache = TextCache(ws / "cache" / "text", "tiny-moss")
    from safetensors.torch import save_file
    prompts = {apply_trigger(f"a flat color swatch number {i}", "zxq") for i in range(4)} | {"", "zxq swatch"}
    for p in prompts:
        save_file({"embeds": torch.randn(5 + len(p) % 7, TEXT_DIM, dtype=torch.bfloat16)}, str(cache.path(p)))
    return job


def run(job, steps=None):
    if steps:
        cfg = json.loads((job / "config.json").read_text())
        cfg["train"]["steps"] = steps
        (job / "config.json").write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, "-m", "mosstrainer.train", "--job-dir", str(job)], cwd=ROOT,
                       capture_output=True, text=True, env=os.environ.copy())
    print(r.stdout[-3000:], r.stderr[-3000:])
    return json.loads((job / "status.json").read_text()), r.stdout


def main():
    with tempfile.TemporaryDirectory() as d:
        job = build(Path(d))
        st, _ = run(job)
        assert st["state"] == "done", st
        files = sorted(p.name for p in (job / "checkpoints").iterdir())
        samples = json.loads((job / "samples" / "index.json").read_text())
        metrics = [json.loads(x) for x in (job / "metrics.jsonl").read_text().splitlines()]
        print("checkpoints:", files)
        print("samples:", [(s["step"], s["prompt"]) for s in samples])
        print("losses:", [round(m["loss"], 4) for m in metrics])
        assert "toy.safetensors" in files and "training_state.pt" in files
        assert {s["step"] for s in samples} == {0, 3, 6}
        assert len(metrics) == 6
        st, out = run(job, steps=9)  # "train more" -> resume from step 6
        assert st["state"] == "done" and st["step"] == 9, st
        assert "resumed from step 6" in out
        print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
