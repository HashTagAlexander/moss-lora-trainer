#!/bin/bash
# RunPod start script for the MOSS LoRA Trainer. On every pod start it clones the trainer into
# /workspace/loratrainer (first start) or updates it, then runs setup.sh, which reinstalls whatever the restart
# wiped from the container disk and starts the UI on port 9090. See runpod/TEMPLATE.md.
#
# Template "Container Start Command": runs this in the background, then RunPod's own /start.sh (SSH, Jupyter):
#   bash -c '(curl -fsSL https://raw.githubusercontent.com/HashTagAlexander/moss-lora-trainer/main/runpod/boot.sh | bash) & /start.sh'
#
# Template variables it reads:
#   MOSS_TRAINER_REPO      git URL of the trainer (default: github.com/HashTagAlexander/moss-lora-trainer)
#   MOSS_TRAINER_BRANCH    branch to check out (default: main)
#   MOSS_TRAINER_UPDATE=0  keep the current checkout instead of pulling on every start
#   MOSS_TRAINER_PASSWORD, HF_TOKEN, MODELS, PORT   passed through to setup.sh
#
# Progress: /workspace/moss-setup.log

# Everything runs inside main, which bash reads in full before running it: with `curl … | bash` the script itself
# arrives on stdin, so nothing below may read from there.
main() {
  local repo="${MOSS_TRAINER_REPO:-https://github.com/HashTagAlexander/moss-lora-trainer.git}"
  local branch="${MOSS_TRAINER_BRANCH:-main}"
  local dir="${MOSS_TRAINER_DIR:-/workspace/loratrainer}"
  local log="${MOSS_SETUP_LOG:-/workspace/moss-setup.log}"
  set -o pipefail
  mkdir -p "$(dirname "$log")"
  exec >>"$log" 2>&1
  echo "[moss] start $(date)"
  if [ ! -d "$(dirname "$dir")" ] || [ ! -w "$(dirname "$dir")" ]; then
    echo "[moss] ERROR: $(dirname "$dir") is missing or read-only: give the pod a volume mounted at /workspace"
    return 1
  fi
  if ! command -v git >/dev/null 2>&1; then
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git >/dev/null
  fi
  if [ ! -d "$dir/.git" ]; then
    git clone --depth 1 --branch "$branch" "$repo" "$dir" || { echo "[moss] ERROR: git clone failed"; return 1; }
  elif [ "${MOSS_TRAINER_UPDATE:-1}" != 0 ]; then
    git -C "$dir" pull --ff-only || echo "[moss] warning: git pull failed, using the existing checkout"
  fi
  cd "$dir" && bash setup.sh || { echo "[moss] ERROR: setup.sh failed (see above)"; return 1; }
  echo "[moss] ready $(date)"
}

main "$@" </dev/null
