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
#   MOSS_TRAINER_PASSWORD, HF_TOKEN, MODELS, PORT, APT_MIRROR   passed through to setup.sh
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
  # to the log file and to the container's output, so RunPod's Logs tab shows progress, the URL and the login
  exec > >(tee -a "$log") 2>&1
  echo "[moss] start $(date)"
  if [ ! -d "$(dirname "$dir")" ] || [ ! -w "$(dirname "$dir")" ]; then
    echo "[moss] ERROR: $(dirname "$dir") is missing or read-only: give the pod a volume mounted at /workspace"
    return 1
  fi
  if ! command -v git >/dev/null 2>&1; then
    # through a mirror and with a time limit: Ubuntu's own servers have slow spells that hang apt. If the mirror
    # fails, they get one try. (setup.sh does the same for what it needs; APT_MIRROR=none leaves apt alone.)
    local opts="-o Acquire::http::Timeout=15 -o Acquire::https::Timeout=15 -o Acquire::Retries=1" f
    local mirror="${APT_MIRROR-http://mirrors.edge.kernel.org/ubuntu}"
    [ "$mirror" = none ] && mirror=""
    get_git() {
      timeout -k 10 "$1" apt-get $opts -o APT::Update::Error-Mode=any update -qq \
        && DEBIAN_FRONTEND=noninteractive timeout -k 10 300 apt-get $opts install -y -qq git >/dev/null
    }
    if [ -n "$mirror" ]; then
      for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do
        [ -f "$f" ] && grep -Eq 'https?://[a-z0-9.-]*(archive|security)\.ubuntu\.com/ubuntu' "$f" || continue
        [ -e "$f.moss.orig" ] || cp -p "$f" "$f.moss.orig"
        sed -E "s#https?://[a-z0-9.-]*(archive|security)\.ubuntu\.com/ubuntu#$mirror#g" "$f.moss.orig" > "$f"
      done
      echo "[moss] apt: using the mirror $mirror"
    fi
    if ! get_git 75; then
      echo "[moss] apt didn't finish in time; trying Ubuntu's own servers"
      for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do
        [ -f "$f.moss.orig" ] && cp -p "$f.moss.orig" "$f"
      done
      get_git 120 || { echo "[moss] ERROR: couldn't install git (apt is unreachable)"; return 1; }
    fi
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
