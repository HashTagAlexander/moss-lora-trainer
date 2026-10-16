#!/bin/bash
# vast.ai provisioning script for the MOSS-Image LoRA Trainer.
# Point the template's PROVISIONING_SCRIPT variable at the raw URL of this file.
#
# Template variables it reads:
#   MOSS_TRAINER_REPO      git URL of the trainer (default: github.com/HashTagAlexander/moss-lora-trainer)
#   MOSS_TRAINER_BRANCH    branch to check out (default: main)
#   MOSS_TRAINER_PASSWORD  UI password (otherwise generated; see /workspace/loratrainer/.password)
#   HF_TOKEN, MODELS, PORT  passed through to setup.sh
#
# Progress appears on the Instance Portal's Logs page (moss-setup.log).
set -o pipefail

REPO="${MOSS_TRAINER_REPO:-https://github.com/HashTagAlexander/moss-lora-trainer.git}"
BRANCH="${MOSS_TRAINER_BRANCH:-main}"
DIR=/workspace/loratrainer
LOG=/var/log/portal/moss-setup.log
mkdir -p /var/log/portal
exec > >(tee -a "$LOG") 2>&1

echo "[moss] provisioning started $(date)"
if [ ! -d "$DIR/.git" ]; then
  if [ -z "$REPO" ]; then
    echo "[moss] ERROR: set MOSS_TRAINER_REPO in the template to the trainer's git URL"; exit 1
  fi
  git clone --depth 1 --branch "$BRANCH" "$REPO" "$DIR" || { echo "[moss] ERROR: git clone failed"; exit 1; }
else
  git -C "$DIR" pull --ff-only || echo "[moss] warning: git pull failed, using existing checkout"
fi

cd "$DIR" && bash setup.sh
echo "[moss] provisioning finished $(date)"
