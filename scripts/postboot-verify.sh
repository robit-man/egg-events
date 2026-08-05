#!/usr/bin/env bash
set -u

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
verification_dir="$workspace_dir/data/verification"
state_file="$verification_dir/postboot-state.json"
log_file="$verification_dir/postboot-latest.log"
mkdir -p "$verification_dir"

ready=0
for attempt in $(seq 1 240); do
  if curl -fsS --max-time 3 http://127.0.0.1:8788/api/state > "$state_file" \
    && grep -q '"runtime": "live' "$state_file"; then
    ready=1
    break
  fi
  sleep 2
done

{
  printf 'Egg post-boot verification: %s\n' "$(date --iso-8601=seconds)"
  printf 'dashboard_live=%s\n' "$ready"
  if [[ "$ready" == 0 ]]; then
    cat "$state_file" 2>/dev/null || true
    exit 1
  fi

  printf '\n=== timed live trace ===\n'
  "$workspace_dir/.venv/bin/python" -m egg_companion \
    --config "$workspace_dir/config/egg.yaml" trace \
    --url http://127.0.0.1:8788 --seconds 30
  trace_status=$?

  printf '\n=== cognitive memory audit ===\n'
  "$workspace_dir/.venv/bin/python" -m egg_companion \
    --config "$workspace_dir/config/egg.yaml" memory-audit
  memory_status=$?

  printf '\n=== live hardware audit ===\n'
  "$workspace_dir/.venv/bin/python" -m egg_companion \
    --config "$workspace_dir/config/egg.yaml" audit
  audit_status=$?

  printf '\ntrace_status=%s memory_status=%s audit_status=%s\n' \
    "$trace_status" "$memory_status" "$audit_status"
  if (( trace_status || memory_status || audit_status )); then
    exit 1
  fi
} > "$log_file" 2>&1
