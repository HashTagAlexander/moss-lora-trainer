# vast.ai template: MOSS-Image LoRA Trainer

A template that installs and starts the trainer on its own and serves it over **HTTPS**. Rent an
instance, wait for setup to finish, then click **Open → MOSS Trainer**.

## Create the template

vast.ai console → **Templates** → **New Template** (or **+ Create**), then fill in:

| Field | Value |
|---|---|
| Template name | `MOSS LoRA Trainer` |
| Image path:tag | `vastai/pytorch:2.7.0-cuda-12.8.0-py310-22.04` (setup installs its own newer torch in a venv) |
| Launch mode | Jupyter + SSH |
| Jupyter direct HTTPS | **Ticked** (turns on HTTPS for everything served through vast's portal proxy) |
| On-start script | **`entrypoint.sh`**: this starts vast's portal and runs the provisioning script. Don't put anything else here |
| Extra filters | `cpu_arch in ['arm64', 'amd64'] cuda_max_good>=12.8` |
| Disk space | **150 GB** (weights ≈ 30 GB, plus caches, checkpoints and the venv) |
| Visibility | Private |

**Docker options**: paste this as one line:

```
-p 1111:1111 -p 6006:6006 -p 8080:8080 -p 8384:8384 -p 9090:9090 -e OPEN_BUTTON_PORT=1111 -e OPEN_BUTTON_TOKEN=1 -e JUPYTER_DIR=/ -e DATA_DIRECTORY=/workspace/ -e PORTAL_CONFIG="localhost:1111:11111:/:Instance Portal|localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter Terminal|localhost:8384:18384:/:Syncthing|localhost:6006:16006:/:Tensorboard|localhost:9090:19090:/:MOSS Trainer" -e PROVISIONING_SCRIPT=https://raw.githubusercontent.com/HashTagAlexander/moss-lora-trainer/main/vast/provision.sh -e MOSS_TRAINER_REPO=https://github.com/HashTagAlexander/moss-lora-trainer.git -e NVIDIA_DRIVER_CAPABILITIES=all
```

If you'd rather use the **Environment Variables** section, enter the same variables there. Wrap
`PORTAL_CONFIG` in double quotes, because the form requires quotes when a value contains spaces.

### Why `9090:19090`

In `PORTAL_CONFIG`, an entry with **different** external and internal ports is served through vast's
Caddy proxy. Caddy listens on 9090, adds TLS and vast's login, and forwards to the trainer, which
listens privately on `127.0.0.1:19090`. `setup.sh` detects this from `PORTAL_CONFIG` on its own.
With equal ports (`9090:9090`) the trainer is reached directly over plain HTTP and uses its own
`admin` password instead.

Optional variables:

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face token, if downloads get rate-limited |
| `MODELS` | Models to pre-download, e.g. `OpenMOSS-Team/MOSS-Image-Base OpenMOSS-Team/MOSS-Image-SFT` |
| `KEEP_AUTH=1` | Keep the trainer's own `admin` password as well as vast's login |

Click **Create** (or **Save**). From now on, pick this template when renting.

## What happens when an instance starts

1. `entrypoint.sh` starts vast's services, including the Instance Portal and Caddy.
2. It then runs `PROVISIONING_SCRIPT`: `provision.sh` clones the repo to `/workspace/loratrainer` and
   runs `setup.sh`. This takes 10–20 minutes, mostly the 30 GB model download.
3. `setup.sh` registers the UI with supervisor (`moss-trainer`), so it starts again after reboots and crashes.

You can follow progress in **Open → Instance Portal → Logs → `moss-setup.log`**. The **MOSS Trainer**
link shows a "not ready" page until setup finishes.

## Opening it over HTTPS

- **Portal link:** Open → Instance Portal → **MOSS Trainer**. The Open button's token logs you in
  through vast's cookie, so there's no extra password.
- **Cloudflare link:** the portal also offers a `https://….trycloudflare.com` link for MOSS Trainer.
  It has a fully valid certificate, so there's nothing to install. Cloudflare limits uploads to
  100 MB per request, so send big datasets in several zips.
- **Direct IP link** (`https://<ip>:<port mapped to 9090>`): this uses vast's own certificate. Load vast's
  TLS certificate into your browser once (vast docs, "Jupyter direct HTTPS"), or click through the
  browser warning.

To open it directly without the Open button, you need vast's token: run `echo $OPEN_BUTTON_TOKEN` over
SSH, then visit `https://<ip>:<port>/?token=<token>` once.

## Day-to-day

- Restart the UI: `supervisorctl restart moss-trainer`
- UI log: `/var/log/portal/moss-trainer.log`
- Update the trainer code: push to the repo, then on the instance run `cd /workspace/loratrainer && git pull && supervisorctl restart moss-trainer`
- Datasets, jobs and LoRAs live in `/workspace/moss_workspace/`. Download LoRAs from the job's Checkpoints tab before destroying the instance.
