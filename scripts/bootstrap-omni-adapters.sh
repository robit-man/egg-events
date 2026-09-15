#!/usr/bin/env bash
# Install, build, and supervise the Qwen Omni adapter that Egg's
# `omni_adapter` config block routes to.
#
# The adapter (https://github.com/robit-man/qwen-omni-adapters) fronts one
# logical Ollama tag -- robit/ornith-1.5-omni:q4km -- with Qwen3-Omni
# comprehension and Qwen3-TTS speech. It runs as its own supervised process
# because it owns two llama.cpp workers with their own GPU residency and
# lifetime; Egg talks to it over loopback and falls back to Omnius whenever it
# is not answering.
#
# This script is idempotent: re-running it updates the checkout, rebuilds only
# what changed, pulls only missing tags, and restarts the service.
set -Eeuo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vendor_dir="${EGG_OMNI_ADAPTERS_DIR:-$workspace_dir/vendor/qwen-omni-adapters}"
repository="${EGG_OMNI_ADAPTERS_REPO:-https://github.com/robit-man/qwen-omni-adapters.git}"
branch="${EGG_OMNI_ADAPTERS_REF:-main}"
omni_model="${EGG_OMNI_MODEL:-robit/ornith-1.5-omni:q4km}"
# The logical Omni tag's standard layers are the language model, so the
# language stage points back at the same tag: one Ollama slot, one resident
# copy of the weights. A distinct tag here would load the same parameters
# twice and, with OLLAMA_MAX_LOADED_MODELS=1, thrash between them.
language_model="${EGG_OMNI_LANGUAGE_MODEL:-$omni_model}"
adapter_port="${EGG_OMNI_ADAPTER_PORT:-8910}"
install_service=1
start_service=1

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap-omni-adapters.sh [options]

  --no-service   Build and pull only; do not install the systemd user unit
  --no-start     Install the unit but do not start it now
  --help         Show this help

Environment: EGG_OMNI_ADAPTERS_DIR, EGG_OMNI_ADAPTERS_REPO,
EGG_OMNI_ADAPTERS_REF, EGG_OMNI_MODEL, EGG_OMNI_LANGUAGE_MODEL,
EGG_OMNI_ADAPTER_PORT.
EOF
}

while (($#)); do
  case $1 in
    --no-service) install_service=0 ;;
    --no-start) start_service=0 ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

log() { printf '[egg-omni] %s\n' "$*"; }

for command in git cmake ollama ffmpeg; do
  command -v "$command" >/dev/null 2>&1 || {
    printf 'Missing dependency: %s (run scripts/bootstrap-jetson.sh first)\n' "$command" >&2
    exit 1
  }
done

if [[ -d "$vendor_dir/.git" ]]; then
  log "updating $vendor_dir"
  git -C "$vendor_dir" fetch --tags origin "$branch"
  # Preserve local work rather than discarding it: the checkout is a normal
  # clone an operator may be editing, not a disposable cache.
  if [[ -n "$(git -C "$vendor_dir" status --porcelain)" ]]; then
    log "local changes present; leaving the checkout at its current revision"
  else
    git -C "$vendor_dir" checkout "$branch"
    git -C "$vendor_dir" merge --ff-only "origin/$branch"
  fi
else
  log "cloning $repository into $vendor_dir"
  mkdir -p "$(dirname "$vendor_dir")"
  git clone --branch "$branch" "$repository" "$vendor_dir"
fi

# Pull the logical Omni tag (and a distinct language backend only if one was
# explicitly configured). The adapter's own supervisor also pulls missing tags
# on start; doing it here keeps a multi-gigabyte download out of the first
# spoken turn.
models=("$omni_model")
[[ "$language_model" == "$omni_model" ]] || models+=("$language_model")
for model in "${models[@]}"; do
  if ollama show "$model" >/dev/null 2>&1; then
    log "model present: $model"
  else
    log "pulling $model"
    ollama pull "$model"
  fi
done

log "building the adapter runtime (llama.cpp CUDA kernels are pinned to this SoC)"
OMNI_MODEL="$omni_model" \
OMNI_LANGUAGE_MODEL="$language_model" \
  "$vendor_dir/scripts/bootstrap.sh"

log "validating the sidecar attached to $omni_model"
"$vendor_dir/.venv/bin/python" -m qwen_omni_adapters resolve "$omni_model" >/dev/null

"$vendor_dir/.venv/bin/qwen-omni" doctor \
  --model "$omni_model" \
  --language-model "$language_model" \
  --no-tunnel || log "doctor reported findings; see the JSON above"

if ((install_service)); then
  unit_dir="$HOME/.config/systemd/user"
  mkdir -p "$unit_dir"
  cat > "$unit_dir/egg-omni-adapters.service" <<EOF
[Unit]
Description=Qwen Omni adapter for the Egg companion
After=network-online.target ollama.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$vendor_dir
Environment=OMNI_REPO_ROOT=$vendor_dir
Environment=OMNI_MODEL=$omni_model
Environment=OMNI_LANGUAGE_MODEL=$language_model
Environment=OMNI_ADAPTER_PORT=$adapter_port
Environment=PYTHONUNBUFFERED=1
# --no-tunnel keeps every component on loopback: Egg is the only client, and
# the portal's public Cloudflare URL is not wanted on this device.
ExecStart=$vendor_dir/.venv/bin/qwen-omni-daemon serve --allow-direct-gpu --no-tunnel
Restart=on-failure
RestartSec=10
TimeoutStartSec=0
TimeoutStopSec=120
UMask=0077

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable egg-omni-adapters.service
  if ((start_service)); then
    log "starting egg-omni-adapters.service"
    systemctl --user restart egg-omni-adapters.service
  fi
fi

cat <<EOF

Adapter installed. To route Egg through it, set in config/egg.yaml:

  omni_adapter:
    enabled: true

Then restart the companion. Check the adapter with:

  systemctl --user status egg-omni-adapters.service
  $vendor_dir/.venv/bin/qwen-omni-daemon status
  curl -fsS http://127.0.0.1:$adapter_port/healthz
EOF
