# RunPod template: MOSS-Image LoRA Trainer

A template that installs and starts the trainer on its own, on every pod start. Deploy a pod, wait for setup to
finish, then click **Connect → HTTP Service [9090]**.

## How it works on RunPod

- **Only `/workspace` survives a stop.** RunPod wipes the container disk when a pod stops, so everything the trainer
  needs lives on the volume at `/workspace`: the code (`loratrainer/`), the venv, uv and its Python (`.uv/`), the
  model weights (`hf_cache/`) and your data (`moss_workspace/`).
- **[`boot.sh`](boot.sh) runs at every start**, in the background next to RunPod's own `/start.sh` (SSH,
  Jupyter). It updates the code, then runs `setup.sh`, which reinstalls whatever the wipe removed (tmux, git) and
  starts the UI. The first start takes 10–20 minutes, mostly downloading models; later starts take a minute or two.
- **The UI is public.** It's served at `https://<pod id>-9090.proxy.runpod.net` with RunPod's HTTPS, and anyone with
  the link can reach it, so it keeps its own `admin` login. Set the password with `MOSS_TRAINER_PASSWORD`
  (below), or read the generated one from the setup log.

## Create the template

RunPod console → **My Templates** → **New Template**, then fill in:

| Field | Value |
|---|---|
| Template name | `MOSS LoRA Trainer` |
| Template type | Pod |
| Container image | an official `runpod/pytorch` image (any recent tag). Setup installs its own newer torch into the venv, so the tag doesn't matter much |
| Container start command | see below |
| Container disk | **30 GB** (temporary files only) |
| Volume disk | **150 GB** (weights ≈ 30 GB per model family, plus caches, checkpoints and the venv) |
| Volume mount path | `/workspace` |
| Expose HTTP ports | `9090` (add `8888` too if you want the image's Jupyter) |
| Expose TCP ports | `22` (SSH) |

**Container start command**: paste as one line:

```
bash -c '(curl -fsSL https://raw.githubusercontent.com/HashTagAlexander/moss-lora-trainer/main/runpod/boot.sh | bash) & /start.sh'
```

This keeps calling the image's `/start.sh`, which RunPod's images need for SSH and Jupyter. Because `boot.sh` runs
in the background, SSH comes up straight away while setup runs.

**Environment variables**: all optional:

| Variable | Purpose |
|---|---|
| `MOSS_TRAINER_PASSWORD` | the UI's `admin` password. Store it as a RunPod secret and enter `{{ RUNPOD_SECRET_moss_password }}` as the value |
| `HF_TOKEN` | Hugging Face token, needed to pre-download gated models (Fibo 2.0). Also a good secret. You can enter it later under **Settings** in the UI instead |
| `MODELS` | models to download during setup, e.g. `briaai/Fibo-2.0-dev` or `OpenMOSS-Team/MOSS-Image-Base kroma-v0.3-teacher`. Set it to `none` to skip; models also download on first use |
| `MOSS_TRAINER_UPDATE` | `0` keeps the current code instead of pulling the latest on every start |
| `MOSS_TRAINER_REPO`, `MOSS_TRAINER_BRANCH` | use a fork or another branch |

Save the template. From now on, pick it when deploying.

## Deploying a pod

1. **GPU**: 48 GB or more for Fibo 2.0 and Krea 2 (RTX PRO 6000, A100 80GB, H100, L40S). Under the GPU filters,
   allow only **CUDA 12.8 or newer** so the host driver runs the cu128 torch wheels.
2. Pick the template, check the disk sizes, and deploy. Stick to **on-demand** pods for long runs: an interruptible (spot) pod can be
   stopped mid-training (the job resumes from its last checkpoint, but you lose the steps since).
3. **Pod volume or network volume.** The pod's own volume persists while the pod is stopped but ties it to that
   machine. A network volume (attach it when deploying) outlives the pod and lets you move to another GPU in the
   same data center; mount it at `/workspace` too.

## Opening it

- **Connect → HTTP Service [9090]**, or `https://<pod id>-9090.proxy.runpod.net`. Until setup finishes the link
  shows a 502 error: give it time.
- **Setup progress and password**: the pod's **Logs → Container** tab shows setup as it runs; when it's done, the end
  shows the URL and the `admin` login. (The **System** tab only shows the image download.) The same output is in
  `/workspace/moss-setup.log`, and the password alone is in `/workspace/loratrainer/.password`.
- **Uploads**: RunPod's proxy runs through Cloudflare, which closes any request that takes longer than 100 seconds.
  Upload big datasets as several smaller zips, or copy them over SSH into
  `/workspace/moss_workspace/datasets/<name>/`.

## Day-to-day

- **Restart the UI**: `bash /workspace/loratrainer/start.sh` (it runs in tmux: `tmux attach -t moss`; UI log:
  `/workspace/moss_workspace/server.log`).
- **Update the code**: restart the pod (the boot script pulls), or run
  `cd /workspace/loratrainer && git pull && bash start.sh`.
- **Stop the pod** when you're not training: the volume keeps the weights, datasets and LoRAs. Its storage is
  still billed, at a much lower rate than the GPU.
- **Before terminating the pod**, download your LoRAs from each job's **Checkpoints** tab: terminating deletes the
  pod's volume (a network volume stays).
