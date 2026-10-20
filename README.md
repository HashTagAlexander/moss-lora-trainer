# MOSS Trainer — LoRA training for MOSS-Image

A self-hosted web UI (in the spirit of ai-toolkit) for training LoRAs on
[OpenMOSS MOSS-Image](https://github.com/OpenMOSS/MOSS-Image), a 5B single-stream DiT
with a frozen Qwen3.5-9B text encoder and the FLUX.2 VAE.

- **Datasets**: drag-and-drop images / `.txt` or `.json` captions / `.zip`, inline caption editor, bulk prepend/append/replace,
  and **Caption with AI**: structured VGL captions for Fibo 2.0 written by a vision model through OpenRouter.
- **Jobs**: form or raw-JSON config, GPU queue, live loss chart, sample grid per step, log viewer,
  stop (saves a checkpoint first), resume, "train more", clone, reset.
- **Output**: `.safetensors` LoRAs with ComfyUI-style keys (`diffusion_model.layers.N.proj.q.lora_A.weight`, …).
- **Settings**: Hugging Face token (for gated models), OpenRouter key and caption model, one-click model downloads.

## Supported models

| Model id | Family | Train | Notes |
|---|---|---|---|
| `OpenMOSS-Team/MOSS-Image-Base` | MOSS-Image | ✓ | recommended MOSS base for LoRAs |
| `OpenMOSS-Team/MOSS-Image-SFT` | MOSS-Image | ✓ | |
| `kroma-v0.3-teacher` | Krea 2 | ✓ | [lodestones/Kroma](https://huggingface.co/lodestones/Kroma) teacher (velocity), recommended Kroma base for LoRAs |
| `kroma-v0.3-base` | Krea 2 | ✓ | stored in fp32 (~51 GB download), loaded as bf16 |
| `kroma-v0.3.1-turbo` | Krea 2 | generate only | 8 steps, CFG 1, shift 1.15. LoRAs trained on the teacher work on it |
| `krea2-raw`, `krea2-turbo` | Krea 2 | raw ✓ / turbo generate only | from Comfy-Org/Krea-2 |
| `briaai/Fibo-2.0-dev` | Fibo 2.0 | ✓ | gated. Bria's full base model: 50 steps, CFG 5. Text-to-image and editing |
| `briaai/Fibo-2.0-edit-turbo` | Fibo 2.0 | generate only | gated. 4 steps, CFG 1, made for editing a source image. LoRAs trained on Dev work on it |

Krea 2 models use diffusers' `Krea2Transformer2DModel` with the public Qwen3-VL-4B-Instruct text encoder and the
Qwen-Image VAE (both byte-identical to what Krea 2 ships). Kroma LoRAs are saved in diffusers format
(`transformer.transformer_blocks.N.attn.to_q.lora_A.weight`), which loads in diffusers' `Krea2Pipeline` and in
ComfyUI. Sampling CFG uses the standard convention (1 = off), so Krea's "guidance 4.5" is CFG 5.5 here. Shift 0
means the resolution-aware default.

### Fibo 2.0 (Bria)

[Fibo 2.0](https://huggingface.co/briaai/Fibo-2.0-dev) is gated: accept its license on Hugging Face (Dev and Edit
Turbo), then paste a read token under **Settings → Hugging Face** and press **Check access**. Download the models
from the same page, or let them download on first use. Everything runs on diffusers' `BriaFibo2Pipeline` code
(installed from diffusers main, see `requirements.txt`), following Bria's own
[`fine_tuning/train_lora.py`](https://huggingface.co/briaai/Fibo-2.0-dev/blob/main/fine_tuning/README.md):

- **Captions** are structured [VGL](https://docs.bria.ai/vgl) JSON, the format Fibo 2.0 was trained on, stored in the
  image's `.txt` (a `.json` caption dropped into a dataset is stored there too). Plain sentences also work, but JSON
  captions match what you prompt with. **Caption with AI** in the dataset viewer writes them for you: it sends each
  image to a vision model on OpenRouter with a prompt based on the one Bria's `bria-vgl` captioner uses (default model
  `google/gemini-3.5-flash-lite`, the one `bria-vgl` uses). The trigger word goes into `short_description`.
- **Training**: captions are cleaned and compacted the way the pipeline does, captions with `text_render` entries get
  the pipeline's text images (and are seen without them above noise level 0.9), caption dropout swaps in the empty VGL
  caption, timesteps are Bria's resolution-shifted logit-normal with 10% uniform, and the transformer runs in plain
  bf16. By default the LoRA covers every Linear layer (attention, feed-forward, text perceiver, modulation), as Bria's
  recipe does. LoRAs are saved in diffusers format and load with
  `pipe.transformer.load_lora_adapter(folder, weight_name="name.safetensors")`.
- **Generate tab**: text-to-image, or upload a **source image** to edit it (the prompt is then the edit instruction,
  or a VGL caption of the result with an `edit_instruction` key). Edit Turbo does edits in 4 steps without CFG.
  The default negative for Dev is the one from Bria's model card.

## How training works

| | |
|---|---|
| Trainable | LoRA on the 42 main DiT blocks: attention `q/k/v/o`, gated FFN `ffn_up_gate/ffn_down` (optionally the 2 text-refiner blocks) |
| Frozen | Everything else. The Qwen3.5-9B text encoder is never trained, as the MOSS authors recommend |
| Objective | Rectified flow: `x_t = (1-σ)·x0 + σ·ε`, predict `ε − x0`, plain MSE, timestep `1000·σ` (same as pretraining) |
| Timesteps | Default `moss` = the final pretraining stage's truncated logit-normal (tech report, Table 6). Logit-normal / shifted / uniform are also available |
| Caching | Captions are encoded once with Qwen3.5-9B and cached on disk. The encoder is then unloaded. Images are VAE-encoded once per aspect bucket |
| Sampling | The official recipe: Euler, shift 5, CFG 3, negative-branch layer skip at block 22 |

The transformer code is vendored from the model's `pipeline_moss_image.py` (Apache-2.0) with
gradient checkpointing added. Its outputs match the original bit-for-bit.

## vast.ai template (recommended)

Put this code in a GitHub repo, then create a vast.ai template that installs and starts everything on
first boot. Full field-by-field settings are in **[vast/TEMPLATE.md](vast/TEMPLATE.md)**. After that,
renting an instance with the template is all it takes: open **Open → MOSS Trainer** once
`moss-setup.log` says it's finished.

## vast.ai manual setup (RTX PRO 6000, port 9090)

1. **Rent the instance**
   - GPU: RTX PRO 6000 (96 GB). Disk: **≥ 120 GB** (the weights are about 30 GB, plus cache and checkpoints).
   - Template: any PyTorch / CUDA **12.8+** Ubuntu image (for example the official *PyTorch* template).
   - In the template's *Docker options*, add: `-p 9090:9090`
2. **Copy this folder to the instance**, for example:
   ```bash
   rsync -avz -e "ssh -p <SSH_PORT>" --exclude workspace --exclude .venv ./ root@<SSH_HOST>:/workspace/loratrainer/
   ```
3. **Run setup** on the instance:
   ```bash
   cd /workspace/loratrainer && bash setup.sh
   ```
   This installs uv, a Python 3.12 venv, PyTorch (cu128 wheels for Blackwell) and the dependencies. It
   then downloads MOSS-Image-Base and starts the UI in a tmux session on port **9090**. At the end it
   prints the URL and a generated password (user `admin`).
4. **Open the UI.** On vast.ai, port 9090 is mapped to a random external port: see *IP Port Info* on
   the instance card. Alternatively, use an SSH tunnel:
   ```bash
   ssh -p <SSH_PORT> root@<SSH_HOST> -L 9090:localhost:9090
   ```
   Then browse to http://localhost:9090.
5. *(Optional)* To restart the UI automatically after a reboot, set the instance's **on-start script** to
   `bash /workspace/loratrainer/start.sh`.

Useful setup options: `HF_TOKEN=hf_... bash setup.sh`, `MODELS="OpenMOSS-Team/MOSS-Image-Base OpenMOSS-Team/MOSS-Image-SFT" bash setup.sh`,
`NO_AUTH=1` (only behind an SSH tunnel), `TORCH_INDEX=https://download.pytorch.org/whl/cu130`.

Everything persistent lives under `/workspace/moss_workspace/` (datasets, jobs, caches) and
`/workspace/hf_cache/` (model weights).

## Workflow

1. **Datasets → New dataset**, then drop in images. Captions pair with images by filename (`foo.jpg` + `foo.txt`).
   MOSS-Image was trained on rich natural-language captions (short, medium and long), so describe each image properly
   rather than only listing tags. Put `[trigger]` in a caption to control where the trigger word goes; otherwise it is prepended
   (to `short_description` for JSON captions). For Fibo 2.0, use **Caption with AI** (needs an OpenRouter key in Settings).
2. **New Job**: pick the dataset, set a trigger word and adjust the settings. **Save & start**.
3. Watch the loss and samples on the job page. **Stop** saves a checkpoint, and **Start** resumes from it.
4. Download LoRAs from the **Checkpoints** tab.

Starting points: rank 32 / alpha 32, AdamW at 1e-4, 1500–3000 steps for 20–50 images, 1024 resolution.
With 96 GB you can usually turn off gradient checkpointing at 1024 for roughly 25% more speed.

## Using the LoRA

- **Generate tab** (built in): stack any of your checkpoints (or uploaded `.safetensors` LoRAs) with
  per-LoRA strength, pick an aspect ratio and size, and generate with the official sampler. The model stays
  loaded between generations and is unloaded after 15 idle minutes or with **Unload**. Each checkpoint
  row has a **Generate** button that pre-fills its LoRA and trigger word. Images and their settings are
  kept in `moss_workspace/generations/`, and **Use these settings** restores any past image's exact settings.
- **ComfyUI**: install the MOSS-Image ComfyUI plugin from the MOSS-Image repo, load the model with its loader, then
  apply the LoRA with a standard `LoraLoaderModelOnly` node.
- **Diffusers / CLI**:
  ```bash
  python generate.py --lora workspace/jobs/my_lora/checkpoints/my_lora.safetensors \
      --prompt "mytrigger, a portrait photo by a window" --steps 50 --out test.png
  ```
  Use `--lora path.safetensors:0.8` to set the strength. Repeat `--lora` to stack several LoRAs.

## Development

```bash
python tests/smoke_test.py        # tiny random MOSS model on CPU: caching, training, sampling, save, resume
python tests/smoke_test_krea2.py  # tiny random Krea 2 stack, checked against diffusers' Krea2Pipeline
python tests/smoke_test_fibo2.py  # tiny random Fibo 2.0 stack, checked against diffusers' BriaFibo2Pipeline
python -m server.app --port 9090
```

Layout: `mosstrainer/` (model, LoRA, data, trainer process), `server/` (FastAPI and a static UI with no build step),
`setup.sh` / `start.sh` (deployment).
