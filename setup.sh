#!/usr/bin/env bash
# MOSS-Image LoRA Trainer — one-shot setup for a vast.ai instance (RTX PRO 6000 / Blackwell).
#
#   bash setup.sh
#
# Then open http://<public-ip>:<mapped-port-for-9090>  (user: admin, password printed at the end).
#
# Optional environment variables:
#   PORT=9090                     public web UI port (default 9090)
#                                 If PORTAL_CONFIG proxies this port through vast's Caddy (e.g.
#                                 "localhost:9090:19090:/:MOSS Trainer"), the UI listens privately on
#                                 127.0.0.1:19090 and Caddy serves it on 9090 with HTTPS + vast's login.
#   MOSS_TRAINER_PASSWORD=...     UI password (generated and saved to .password if unset)
#   NO_AUTH=1                     disable the password (only if you use an SSH tunnel!)
#   HF_TOKEN=hf_...               Hugging Face token (only needed if downloads are rate-limited)
#   MODELS="OpenMOSS-Team/MOSS-Image-Base"   models to pre-download ("" to skip)
#   TORCH_INDEX=https://download.pytorch.org/whl/cu128   torch wheel index
#   WORKSPACE_DIR=/workspace      where datasets/jobs/cache and the HF cache live
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PORT="${PORT:-9090}"
LISTEN_HOST=0.0.0.0
LISTEN_PORT="$PORT"
BEHIND_PORTAL=""
if [ -n "${PORTAL_CONFIG:-}" ]; then
  cfg="${PORTAL_CONFIG#\"}"; cfg="${cfg%\"}"
  IFS='|' read -ra _apps <<< "$cfg"
  for _app in "${_apps[@]}"; do
    IFS=':' read -r _host _ext _int _rest <<< "$_app"
    if [ "$_ext" = "$PORT" ] && [ -n "$_int" ] && [ "$_int" != "$_ext" ]; then
      BEHIND_PORTAL=1; LISTEN_HOST=127.0.0.1; LISTEN_PORT="$_int"
    fi
  done
fi
# vast's Caddy already requires its own login in front of proxied apps; don't ask twice.
if [ -n "$BEHIND_PORTAL" ] && [ -z "${KEEP_AUTH:-}" ]; then NO_AUTH=1; fi
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
MODELS="${MODELS-OpenMOSS-Team/MOSS-Image-Base}"
if [ -z "${WORKSPACE_DIR:-}" ]; then
  if [ -d /workspace ] && [ -w /workspace ]; then WORKSPACE_DIR=/workspace; else WORKSPACE_DIR="$ROOT"; fi
fi
export HF_HOME="${HF_HOME:-$WORKSPACE_DIR/hf_cache}"
export MOSS_WORKSPACE="${MOSS_WORKSPACE:-$WORKSPACE_DIR/moss_workspace}"

c_ok()   { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
c_step() { printf '\n\033[1;35m==>\033[0m \033[1m%s\033[0m\n' "$*"; }
c_warn() { printf '\033[1;33m!\033[0m %s\n' "$*"; }

c_step "System packages"
if command -v apt-get >/dev/null 2>&1; then
  SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null && SUDO="sudo"
  NEED=""
  for p in tmux curl git; do command -v "$p" >/dev/null 2>&1 || NEED="$NEED $p"; done
  if [ -n "$NEED" ]; then
    $SUDO apt-get update -qq && DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq $NEED >/dev/null || c_warn "apt install failed for:$NEED"
  fi
fi
c_ok "system packages ready"

c_step "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  c_warn "nvidia-smi not found — training needs an NVIDIA GPU"
fi

c_step "Python environment (uv)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
export UV_LINK_MODE=copy
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  uv venv --python 3.12 "$ROOT/.venv"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
c_ok "venv: $ROOT/.venv ($(python --version))"

c_step "PyTorch (CUDA wheels from $TORCH_INDEX)"
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() and tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) >= (2, 9) else 1)" 2>/dev/null; then
  uv pip install --index-url "$TORCH_INDEX" "torch>=2.9" torchvision
fi
python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print("device:", torch.cuda.get_device_name(), "sm_%d%d" % cap)
    archs = torch.cuda.get_arch_list()
    if f"sm_{cap[0]}{cap[1]}" not in archs and cap[0] >= 12:
        print("WARNING: this torch build lacks sm_%d%d kernels; set TORCH_INDEX to a newer CUDA wheel index" % cap)
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    (x @ x).sum().item()
    print("CUDA matmul OK")
else:
    print("WARNING: CUDA not available")
EOF

c_step "Python dependencies"
uv pip install -r requirements.txt
# Fast kernels for Qwen3.5's linear-attention layers (text encoding only; optional).
uv pip install flash-linear-attention >/dev/null 2>&1 && c_ok "flash-linear-attention installed" \
  || c_warn "flash-linear-attention not installed (caption encoding will just be slower)"
c_ok "dependencies installed"

c_step "Model weights → $HF_HOME"
mkdir -p "$HF_HOME" "$MOSS_WORKSPACE"
for m in $MODELS; do
  echo "downloading $m (≈30 GB on first run)…"
  python -c "import sys; from mosstrainer.model import resolve_model; print(resolve_model(sys.argv[1]))" "$m"
done
c_ok "models ready"

c_step "Web UI"
if [ -n "$BEHIND_PORTAL" ]; then
  c_ok "behind vast's portal: UI on $LISTEN_HOST:$LISTEN_PORT, served on port $PORT by Caddy (HTTPS + vast login)"
fi
if [ -n "${NO_AUTH:-}" ]; then
  PASSWORD=""
  [ -z "$BEHIND_PORTAL" ] && c_warn "authentication disabled (NO_AUTH=1)"
else
  PASSWORD="${MOSS_TRAINER_PASSWORD:-}"
  if [ -z "$PASSWORD" ]; then
    if [ -s "$ROOT/.password" ]; then PASSWORD="$(cat "$ROOT/.password")"
    else PASSWORD="$(python -c 'import secrets; print(secrets.token_urlsafe(12))')"; fi
  fi
  printf '%s' "$PASSWORD" > "$ROOT/.password"
  chmod 600 "$ROOT/.password"
fi

cat > "$ROOT/start.sh" <<EOF
#!/usr/bin/env bash
# Starts the MOSS LoRA Trainer UI on $LISTEN_HOST:$LISTEN_PORT (generated by setup.sh). Safe to use as vast.ai on-start script.
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
export HF_HOME="$HF_HOME"
export MOSS_WORKSPACE="$MOSS_WORKSPACE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MOSS_TRAINER_PASSWORD="\$( [ -z "${NO_AUTH:-}" ] && cat "$ROOT/.password" 2>/dev/null )"
if command -v tmux >/dev/null 2>&1 && [ -z "\${FOREGROUND:-}" ]; then
  tmux kill-session -t moss 2>/dev/null || true
  tmux new-session -d -s moss "python -m server.app --host $LISTEN_HOST --port $LISTEN_PORT 2>&1 | tee -a \$MOSS_WORKSPACE/server.log"
  echo "running in tmux session 'moss' (attach: tmux attach -t moss)"
else
  exec python -m server.app --host $LISTEN_HOST --port $LISTEN_PORT
fi
EOF
chmod +x "$ROOT/start.sh"

# vast.ai base images run supervisord: register the UI there so it survives reboots and crashes.
# Elsewhere fall back to a tmux session.
USE_SUPERVISOR=""
if [ -z "${NO_SUPERVISOR:-}" ] && [ -d /etc/supervisor/conf.d ] && command -v supervisorctl >/dev/null 2>&1; then
  rc=0; supervisorctl status >/dev/null 2>&1 || rc=$?
  [ "$rc" -ne 4 ] && USE_SUPERVISOR=1   # 4 = supervisord not reachable
fi
if [ -n "$USE_SUPERVISOR" ]; then
  tmux kill-session -t moss 2>/dev/null || true
  mkdir -p /var/log/portal
  cat > /etc/supervisor/conf.d/moss-trainer.conf <<EOF
[program:moss-trainer]
command=/usr/bin/env FOREGROUND=1 $ROOT/start.sh
autostart=true
autorestart=true
startsecs=5
stopsignal=TERM
stdout_logfile=/var/log/portal/moss-trainer.log
stdout_logfile_maxbytes=10MB
redirect_stderr=true
EOF
  supervisorctl reread >/dev/null && supervisorctl update >/dev/null
  supervisorctl restart moss-trainer >/dev/null || true
  RESTART_HINT="supervisorctl restart moss-trainer   (log: /var/log/portal/moss-trainer.log)"
else
  "$ROOT/start.sh"
  RESTART_HINT="$ROOT/start.sh   (log: tmux attach -t moss)"
fi

ok=""
for _ in $(seq 1 20); do
  if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$LISTEN_PORT/api/info" ${PASSWORD:+-u "admin:$PASSWORD"} | grep -q 200; then ok=1; break; fi
  sleep 1
done
if [ -n "$ok" ]; then c_ok "UI is up on port $PORT"; else c_warn "UI did not answer yet — restart with: $RESTART_HINT"; fi

PUBLIC_IP="${PUBLIC_IPADDR:-$(curl -s --max-time 3 https://ifconfig.me || echo '<instance-ip>')}"
EXT_PORT_VAR="VAST_TCP_PORT_${PORT}"
EXT_PORT="${!EXT_PORT_VAR:-}"
echo
echo "────────────────────────────────────────────────────────────"
echo " MOSS-Image LoRA Trainer"
SCHEME=http
[ -n "$BEHIND_PORTAL" ] && [ "${ENABLE_HTTPS:-false}" = "true" ] && SCHEME=https
if [ -n "$BEHIND_PORTAL" ]; then
  echo "  Open:      vast.ai Open button → Instance Portal → MOSS Trainer"
  echo "             (the portal also offers a trycloudflare.com HTTPS link with a valid certificate)"
fi
if [ -n "$EXT_PORT" ]; then
  echo "  URL:       $SCHEME://$PUBLIC_IP:$EXT_PORT"
else
  echo "  URL:       http://$PUBLIC_IP:<external port mapped to $PORT>"
  echo "             (vast.ai → instance → IP & port info; the template needs '-p $PORT:$PORT')"
fi
echo "  Tunnel:    ssh -p <ssh-port> root@<ssh-host> -L $LISTEN_PORT:localhost:$LISTEN_PORT  →  http://localhost:$LISTEN_PORT"
if [ -n "$PASSWORD" ]; then
  echo "  Login:     admin / $PASSWORD"
fi
echo "  Data:      $MOSS_WORKSPACE"
echo "  Restart:   $RESTART_HINT"
echo "────────────────────────────────────────────────────────────"
