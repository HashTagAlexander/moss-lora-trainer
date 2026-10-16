# vast.ai template: MOSS-Image LoRA Trainer

A template that installs and starts the trainer on its own. Rent an instance, wait for setup
to finish, then click **Open → MOSS Trainer**.

Replace `<USER>/<REPO>` everywhere below with the GitHub repo that holds this code.

## Create the template

vast.ai console → **Templates** → **New Template** (or **+ Create**), then fill in:

| Field | Value |
|---|---|
| Template name | `MOSS LoRA Trainer` |
| Image path:tag | `vastai/pytorch:2.7.0-cuda-12.8.0-py310-22.04` (setup installs its own newer torch in a venv) |
| Launch mode | Jupyter + SSH (as in your current template) |
| On-start script | leave the default (the vastai image's entrypoint runs the provisioning script) |
| Disk space | **150 GB** (weights ≈ 30 GB, plus caches, checkpoints and the venv) |
| Visibility | Private |

**Docker options**: paste this as one line:

```
-p 1111:1111 -p 6006:6006 -p 8080:8080 -p 8384:8384 -p 9090:9090 -e OPEN_BUTTON_PORT=1111 -e OPEN_BUTTON_TOKEN=1 -e JUPYTER_DIR=/ -e DATA_DIRECTORY=/workspace/ -e PORTAL_CONFIG="localhost:1111:11111:/:Instance Portal|localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter Terminal|localhost:8384:18384:/:Syncthing|localhost:6006:16006:/:Tensorboard|localhost:9090:9090:/:MOSS Trainer" -e PROVISIONING_SCRIPT=https://raw.githubusercontent.com/<USER>/<REPO>/main/vast/provision.sh -e MOSS_TRAINER_REPO=https://github.com/<USER>/<REPO>.git -e NVIDIA_DRIVER_CAPABILITIES=all
```

If you'd rather use the **Environment Variables** section, enter the same variables there. Wrap
`PORTAL_CONFIG` in double quotes, because the form requires quotes when a value contains spaces.

Optional variables:

| Variable | Purpose |
|---|---|
| `MOSS_TRAINER_PASSWORD` | Fixed UI password. Otherwise one is generated and saved to `/workspace/loratrainer/.password` |
| `HF_TOKEN` | Hugging Face token, if downloads get rate-limited |
| `MODELS` | Models to pre-download, e.g. `OpenMOSS-Team/MOSS-Image-Base OpenMOSS-Team/MOSS-Image-SFT` |

Click **Create** (or **Save**). From now on, pick this template when renting.

## What happens when an instance starts

1. The vastai image runs `PROVISIONING_SCRIPT` on first boot.
2. `provision.sh` clones the repo to `/workspace/loratrainer` and runs `setup.sh`. This takes
   10–20 minutes, mostly the 30 GB model download.
3. `setup.sh` registers the UI with supervisor (`moss-trainer`), so it starts again after reboots and crashes.

You can follow progress in **Instance Portal → Logs → `moss-setup.log`**. The **MOSS Trainer**
link doesn't respond until setup finishes. The login is `admin` plus the password from the end of
`moss-setup.log` (or `cat /workspace/loratrainer/.password`).

## Day-to-day

- Restart the UI: `supervisorctl restart moss-trainer`
- UI log: `/var/log/portal/moss-trainer.log`
- Update the trainer code: push to the repo, then on the instance run `cd /workspace/loratrainer && git pull && supervisorctl restart moss-trainer`
- Datasets, jobs and LoRAs live in `/workspace/moss_workspace/`. Download LoRAs from the job's Checkpoints tab before destroying the instance.
