"""Generate images with MOSS-Image + a trained LoRA (official diffusers pipeline).

    python generate.py --lora workspace/jobs/my_lora/checkpoints/my_lora.safetensors \
        --prompt "mytrigger, a portrait photo" --out out.png
"""
import argparse
import sys
from pathlib import Path

import torch
from diffusers import DiffusionPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mosstrainer.lora import merge_file_into  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="OpenMOSS-Team/MOSS-Image-Base")
    ap.add_argument("--lora", action="append", default=[], help="LoRA file; repeat for several. Use path:scale to set strength.")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--out", default="out.png")
    args = ap.parse_args()

    pipe = DiffusionPipeline.from_pretrained(
        args.model, custom_pipeline="pipeline_moss_image", trust_remote_code=True,
        torch_dtype={"default": torch.bfloat16, "vae": torch.float32})
    pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config, shift=args.shift)
    for spec in args.lora:
        path, _, scale = spec.partition(":") if not Path(spec).exists() else (spec, "", "")
        n = merge_file_into(pipe.transformer, path, float(scale or 1.0))
        print(f"merged {n} LoRA layers from {path} (scale {scale or 1.0})")
    pipe.to("cuda")
    out = Path(args.out)
    for i in range(args.count):
        image = pipe(args.prompt, negative_prompt=args.negative, height=args.height, width=args.width,
                     num_inference_steps=args.steps, guidance_scale=args.cfg, uncond_skip_layers=[22],
                     generator=torch.Generator("cpu").manual_seed(args.seed + i)).images[0]
        target = out if args.count == 1 else out.with_name(f"{out.stem}_{i}{out.suffix}")
        image.save(target)
        print(f"saved {target}")


if __name__ == "__main__":
    main()
