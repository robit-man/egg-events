#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
update_cache_dir="${XDG_CACHE_HOME:-$HOME/.cache}/egg-runtime-updates"
lock_dir="${XDG_RUNTIME_DIR:-/tmp}"
mkdir -p "$update_cache_dir"
exec 9>"$lock_dir/egg-runtime-update.lock"
if ! flock -n 9; then
  echo "Another Egg dependency update is already running."
  exit 0
fi

log() {
  printf '[egg-update] %s\n' "$*"
}

resolve_nvm_bin() {
  local service_exec
  local service_omnius
  local candidates=()
  local candidate
  service_exec="$(systemctl --user show omnius-daemon.service -p ExecStart --value 2>/dev/null || true)"
  service_omnius="$(
    sed -n 's|.*argv\[\]=[^ ]* \([^ ]*/bin/omnius\) .*|\1|p' <<<"$service_exec"
  )"
  if [[ -x "$service_omnius" && -x "$(dirname "$service_omnius")/npm" ]]; then
    dirname "$service_omnius"
    return 0
  fi
  if command -v omnius >/dev/null 2>&1; then
    candidate="$(dirname "$(command -v omnius)")"
    if [[ -x "$candidate/npm" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi
  shopt -s nullglob
  for candidate in "$HOME"/.nvm/versions/node/*/bin; do
    [[ -x "$candidate/npm" && -x "$candidate/omnius" ]] && candidates+=("$candidate")
  done
  shopt -u nullglob
  if (( ${#candidates[@]} == 0 )); then
    return 1
  fi
  printf '%s\n' "${candidates[@]}" | sort -V | tail -1
}

node_bin="$(resolve_nvm_bin)" || {
  log "No NVM installation containing npm and Omnius was found."
  exit 1
}
npm_bin="$node_bin/npm"
omnius_bin="$node_bin/omnius"
export PATH="$node_bin:$workspace_dir/.venv/bin:/usr/local/bin:/usr/bin:/bin"

installed_version() {
  "$omnius_bin" --version 2>&1 \
    | sed -n 's/^omnius v\([^[:space:]]*\).*/\1/p' \
    | head -1
}

current_version="$(installed_version)"
latest_version="$($npm_bin view omnius version --silent | tail -1)"
if [[ -z "$current_version" || -z "$latest_version" ]]; then
  log "Could not resolve installed/latest Omnius versions."
  exit 1
fi
log "Omnius installed=$current_version latest=$latest_version"

if [[ "${1:-}" == "--check" ]]; then
  if [[ "$current_version" == "$latest_version" ]]; then
    log "Omnius is current."
  else
    log "Omnius update available: $current_version -> $latest_version"
  fi
  exit 0
fi

egg_was_active=0
omnius_was_active=0
systemctl --user is-active --quiet egg-companion.service && egg_was_active=1
systemctl --user is-active --quiet omnius-daemon.service && omnius_was_active=1

conversation_is_idle() {
  local state
  state="$(curl -fsS --max-time 5 http://127.0.0.1:8788/api/state)" || return 1
  python3 -c '
import json, sys
voice = json.load(sys.stdin).get("telemetry", {}).get("voice", {})
idle = (
    voice.get("floor") == "listening"
    and voice.get("active_playback_id") is None
    and voice.get("active_barge_id") is None
    and int(voice.get("pending_ingress") or 0) == 0
)
raise SystemExit(0 if idle else 1)
' <<<"$state"
}

if (( egg_was_active )) && ! conversation_is_idle; then
  log "Conversation is active or readiness is unknown; deferring the update."
  exit 75
fi

ollama_model="${EGG_AUTO_PULL_OLLAMA_MODEL:-robit/ornith-1.5:9b}"
if command -v ollama >/dev/null 2>&1; then
  log "Refreshing Ollama model tag $ollama_model"
  ollama pull "$ollama_model"
  # Omnius releases may temporarily interpret the first slash-delimited
  # segment as a provider name before Egg's compatibility repair is loaded.
  # Keep a manifest-only basename alias so cognition remains available during
  # that transition; the canonical configured/default model stays namespaced.
  ollama_compat_model="${EGG_OLLAMA_COMPAT_MODEL:-${ollama_model#*/}}"
  if [[ "$ollama_compat_model" != "$ollama_model" ]]; then
    log "Refreshing Omnius compatibility alias $ollama_compat_model"
    ollama cp "$ollama_model" "$ollama_compat_model"
  fi
else
  log "Ollama is unavailable; model refresh skipped."
fi

omnius_dist="$($npm_bin root --global)/omnius/dist/index.js"
repair_omnius() {
  python3 "$workspace_dir/scripts/repair_omnius_audio_runtime.py"
  python3 "$workspace_dir/scripts/repair_omnius_asr_runtime.py" "$omnius_dist"
}

# Omnius can update its bundle independently when updateMode=auto. Validate
# and repair the active bundle even when npm already reports the latest tag.
repair_output="$(repair_omnius)"
printf '%s\n' "$repair_output"
repair_changed=0
grep -q 'repaired-' <<<"$repair_output" && repair_changed=1

wait_for_url() {
  local url="$1"
  local attempts="${2:-30}"
  local attempt
  for attempt in $(seq 1 "$attempts"); do
    if curl -fsS --max-time 5 "$url" >/dev/null; then
      return 0
    fi
    sleep 2
  done
  return 1
}

rejected_file="$update_cache_dir/omnius-rejected-version"
if [[ -r "$rejected_file" && "$(<"$rejected_file")" == "$latest_version" ]]; then
  log "Skipping previously rejected Omnius $latest_version until a newer release appears."
  exit 0
fi
if [[ "$current_version" == "$latest_version" ]]; then
  if (( repair_changed )); then
    log "Restarting services to load repaired Omnius runtime."
    if (( omnius_was_active )); then
      systemctl --user restart omnius-daemon.service
      wait_for_url http://127.0.0.1:11435/health/ready 45
    fi
    if (( egg_was_active )); then
      systemctl --user restart egg-companion.service
      wait_for_url http://127.0.0.1:8788/api/state 45
    fi
  fi
  log "Runtime dependencies are current."
  exit 0
fi

restore_services() {
  if (( omnius_was_active )); then
    systemctl --user start omnius-daemon.service
    wait_for_url http://127.0.0.1:11435/health/ready 45
  fi
  if (( egg_was_active )); then
    systemctl --user start egg-companion.service
    wait_for_url http://127.0.0.1:8788/api/state 45
  fi
}

rollback() {
  log "Rolling Omnius back to $current_version"
  systemctl --user stop egg-companion.service omnius-daemon.service || true
  "$npm_bin" install --global --no-audit --no-fund "omnius@$current_version"
  omnius_dist="$($npm_bin root --global)/omnius/dist/index.js"
  repair_omnius
  printf '%s\n' "$latest_version" >"$rejected_file"
  restore_services
}

log "Updating Omnius $current_version -> $latest_version"
systemctl --user stop egg-companion.service omnius-daemon.service || true
if ! "$npm_bin" install --global --no-audit --no-fund "omnius@$latest_version"; then
  rollback
  exit 1
fi
omnius_dist="$($npm_bin root --global)/omnius/dist/index.js"
if ! repair_omnius; then
  rollback
  exit 1
fi

"$omnius_bin" config set updateMode auto || true
"$omnius_bin" config set updateMode auto --local || true

if ! restore_services; then
  rollback
  exit 1
fi
rm -f "$rejected_file"
log "Omnius $latest_version passed daemon and Egg readiness checks."
