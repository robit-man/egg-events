#!/usr/bin/env bash
set -euo pipefail
workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${EGG_OMNIUS_TOKEN:-}" && -r "$HOME/.omnius/api.key" ]]; then
  export EGG_OMNIUS_TOKEN="$(tr -d '\r\n' < "$HOME/.omnius/api.key")"
fi
ensure_system_packages() {
  local packages=(
    python3-venv v4l-utils alsa-utils ffmpeg pulseaudio-utils libportaudio2
    build-essential cmake ninja-build git curl ca-certificates
  )
  local missing=()
  local package
  for package in "${packages[@]}"; do
    dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q 'install ok installed' || missing+=("$package")
  done
  command -v docker >/dev/null 2>&1 || missing+=("docker.io")
  if (( ${#missing[@]} )); then
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${missing[@]}"
  fi
}

ensure_local_services() {
  if ! command -v ollama >/dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh -o /tmp/egg-install-ollama.sh
    sh /tmp/egg-install-ollama.sh
  fi
  if ! command -v omnius >/dev/null 2>&1; then
    if ! command -v npm >/dev/null 2>&1; then
      echo "Omnius is missing and npm is unavailable." >&2
      return 1
    fi
    npm install --global omnius@latest
  fi
}

ensure_companion_service() {
  local unit_dir="$HOME/.config/systemd/user"
  mkdir -p "$unit_dir"
  ln -sfn "$workspace_dir/deploy/egg-whisper.service" "$unit_dir/egg-whisper.service"
  cat > "$unit_dir/egg-companion.service" <<EOF
[Unit]
Description=Egg embodied companion runtime and dashboard
After=network-online.target omnius-daemon.service egg-whisper.service pulseaudio.service sound.target
Wants=network-online.target omnius-daemon.service egg-whisper.service pulseaudio.service

[Service]
Type=simple
WorkingDirectory=$workspace_dir
ExecStart=$workspace_dir/egg
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
EOF
  cat > "$unit_dir/egg-postboot-verify.service" <<EOF
[Unit]
Description=Egg post-boot real-hardware verification
After=egg-companion.service
Wants=egg-companion.service

[Service]
Type=oneshot
WorkingDirectory=$workspace_dir
ExecStart=$workspace_dir/scripts/postboot-verify.sh

[Install]
WantedBy=default.target
EOF
  chmod +x "$workspace_dir/scripts/postboot-verify.sh"
  systemctl --user daemon-reload
  systemctl --user enable egg-companion.service
  systemctl --user enable egg-whisper.service
  systemctl --user enable egg-postboot-verify.service
  sudo loginctl enable-linger "$USER"
}

ensure_system_packages
ensure_local_services
if ! docker image inspect dustynv/whisper:r36.2.0 >/dev/null 2>&1; then
  docker pull dustynv/whisper:r36.2.0
fi
python3 -m venv --system-site-packages "$workspace_dir/.venv"
venv_python="$workspace_dir/.venv/bin/python"
venv_pip="$workspace_dir/.venv/bin/pip"

ensure_gpu_pm_guard() {
  local unit_source
  local ollama_dropin_source
  local runtime_status="unknown"
  unit_source="$(mktemp /tmp/egg-gpu-pm-guard.XXXXXX)"
  ollama_dropin_source="$(mktemp /tmp/egg-ollama-resources.XXXXXX)"
  cat > "$unit_source" <<'EOF'
[Unit]
Description=Validate Jetson GPU runtime power policy before CUDA consumers
After=systemd-udev-settle.service
Wants=systemd-udev-settle.service
Before=display-manager.service ollama.service

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'for control in /sys/devices/platform/17000000.gpu/power/control /sys/devices/platform/gpu.0/power/control; do if [ -r "$control" ]; then value=$(cat "$control"); case "$value" in auto|on) exit 0;; *) echo "Unsupported Jetson GPU runtime power control: $value" >&2; exit 1;; esac; fi; done; echo "Jetson GPU runtime power control is unavailable" >&2; exit 1'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
  cat > "$ollama_dropin_source" <<'EOF'
[Unit]
Requires=egg-gpu-pm-guard.service
After=egg-gpu-pm-guard.service

[Service]
Environment="OLLAMA_CONTEXT_LENGTH=4096"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_NUM_PARALLEL=1"
EOF
  sudo install -m 0644 "$unit_source" /etc/systemd/system/egg-gpu-pm-guard.service
  sudo install -d -m 0755 /etc/systemd/system/ollama.service.d
  sudo install -m 0644 "$ollama_dropin_source" /etc/systemd/system/ollama.service.d/egg-resources.conf
  rm -f "$unit_source" "$ollama_dropin_source"
  sudo systemctl daemon-reload
  sudo systemctl enable egg-gpu-pm-guard.service
  for path in /sys/devices/platform/17000000.gpu/power/runtime_status /sys/devices/platform/gpu.0/power/runtime_status; do
    if [[ -r "$path" ]]; then
      runtime_status="$(<"$path")"
      break
    fi
  done
  if [[ "$runtime_status" == "error" || "$runtime_status" == "unsupported" ]]; then
    echo "Jetson GPU runtime PM reports $runtime_status; the guard is installed for the required reboot." >&2
    return 0
  fi
  sudo systemctl restart egg-gpu-pm-guard.service
}

ensure_ollama_models() {
  if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama is required for Ornith cognition and vision." >&2
    return 1
  fi
  local model
  for model in robit/ornith-1.5:9b; do
    if ! ollama show "$model" >/dev/null 2>&1; then
      ollama pull "$model"
    fi
  done
}

ensure_gpu_pm_guard

if ! "$venv_python" - <<'PY'
import torch
raise SystemExit(not torch.cuda.is_available())
PY
then
  image="dustynv/l4t-pytorch:r36.2.0"
  if ! docker image inspect "$image" >/dev/null 2>&1; then
    docker pull "$image"
  fi
  "$venv_pip" uninstall -y torch torchvision >/dev/null 2>&1 || true
  container_id="$(docker create --network=none "$image")"
  cleanup() { docker rm -f "$container_id" >/dev/null 2>&1 || true; }
  trap cleanup EXIT
  site_packages="$workspace_dir/.venv/lib/python3.10/site-packages"
  for package in torch torch-2.2.0.dist-info torchvision torchvision-0.17.2+c1d70fe.dist-info torchgen; do
    docker cp "$container_id:/usr/local/lib/python3.10/dist-packages/$package" "$site_packages/$package"
  done
fi

"$venv_python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("JetPack CUDA PyTorch installation failed; generic wheels are intentionally refused.")
print(f"Using CUDA PyTorch {torch.__version__} on {torch.cuda.get_device_name(0)}")
PY
"$venv_pip" install --no-deps -e "$workspace_dir"
"$venv_pip" install --no-deps open-clip-torch ultralytics onnxruntime sounddevice pyusb webrtcvad-wheels ftfy timm pytest
yunet_model="$workspace_dir/models/face_detection_yunet_2023mar.onnx"
if [[ ! -s "$yunet_model" ]]; then
  mkdir -p "$(dirname "$yunet_model")"
  curl -fL --retry 3 --connect-timeout 15 -o "$yunet_model" \
    https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
fi
sam_model="$workspace_dir/models/sam2.1_t.pt"
if [[ ! -s "$sam_model" ]]; then
  curl -fL --retry 3 --connect-timeout 15 -o "$sam_model" \
    https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_t.pt
fi
sface_model="$workspace_dir/models/face_recognition_sface_2021dec.onnx"
if [[ ! -s "$sface_model" ]]; then
  curl -fL --retry 3 --connect-timeout 15 -o "$sface_model" \
    https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
fi
yoloe_model="$workspace_dir/models/yoloe-11s-seg-pf.pt"
if [[ ! -s "$yoloe_model" ]]; then
  curl -fL --retry 3 --connect-timeout 15 -o "$yoloe_model" \
    https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11s-seg-pf.pt
fi
pose_model="$workspace_dir/models/yolo11n-pose.pt"
if [[ ! -s "$pose_model" ]]; then
  curl -fL --retry 3 --connect-timeout 15 -o "$pose_model" \
    https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt
fi
"$venv_python" "$workspace_dir/scripts/install_dream_identity_model.py"
ensure_ollama_models
EGG_RESPEAKER_PYTHON="$venv_python" "$workspace_dir/scripts/configure-respeaker-aec.sh"
"$venv_python" -m egg_companion --config "$workspace_dir/config/egg.yaml" audit
ensure_companion_service
