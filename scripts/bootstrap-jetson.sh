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
  cat > "$unit_dir/egg-companion.service" <<EOF
[Unit]
Description=Egg embodied companion runtime and dashboard
After=network-online.target omnius-daemon.service
Wants=network-online.target omnius-daemon.service

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
  systemctl --user enable egg-postboot-verify.service
  sudo loginctl enable-linger "$USER"
}

ensure_system_packages
ensure_local_services
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

ensure_omnius_cuda_asr() {
  local asr_venv="$HOME/.omnius/runtimes/asr/transcribe-cli-node/node_modules/transcribe-cli/.venv"
  local asr_python="$asr_venv/bin/python"
  local runtime_prefix="$HOME/.local/opt/egg-ctranslate2-cuda"
  local cuda_root
  local cuda_library_path
  local build_dir

  if [[ ! -x "$asr_python" ]]; then
    echo "Omnius transcribe-cli runtime is missing: $asr_python" >&2
    return 1
  fi
  if ! command -v nvcc >/dev/null 2>&1 || ! command -v cmake >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1; then
    echo "CUDA ASR requires nvcc, cmake, and ninja." >&2
    return 1
  fi
  cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  cuda_library_path="$runtime_prefix/lib:$cuda_root/targets/aarch64-linux/lib"

  if LD_LIBRARY_PATH="$cuda_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$asr_python" - <<'PY'
import ctranslate2
raise SystemExit(0 if ctranslate2.get_cuda_device_count() > 0 else 1)
PY
  then
    return 0
  fi

  build_dir="$(mktemp -d /tmp/egg-ctranslate2-cuda.XXXXXX)"
  git clone --depth 1 --branch v4.8.1 --recurse-submodules https://github.com/OpenNMT/CTranslate2.git "$build_dir/source"
  "$asr_python" -m pip install --upgrade 'pip<25' wheel 'cmake>=3.22' 'ninja>=1.11' 'pybind11>=2.10'
  cmake -S "$build_dir/source" -B "$build_dir/source/build-cuda" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$build_dir/prefix" \
    -DWITH_CUDA=ON \
    -DWITH_CUDNN=OFF \
    -DOPENMP_RUNTIME=NONE \
    -DWITH_MKL=OFF \
    -DWITH_OPENBLAS=OFF \
    -DWITH_RUY=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=87
  cmake --build "$build_dir/source/build-cuda" --parallel 6
  cmake --install "$build_dir/source/build-cuda"
  mkdir -p "$runtime_prefix"
  cp -a "$build_dir/prefix/." "$runtime_prefix/"
  CTRANSLATE2_ROOT="$runtime_prefix" LD_LIBRARY_PATH="$cuda_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$asr_python" -m pip install --no-build-isolation --no-deps --force-reinstall "$build_dir/source/python"
  LD_LIBRARY_PATH="$cuda_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$asr_python" - <<'PY'
import ctranslate2
if ctranslate2.get_cuda_device_count() < 1:
    raise SystemExit("CTranslate2 CUDA build completed but cannot access the Jetson GPU.")
print(f"Using CUDA CTranslate2 {ctranslate2.__version__} with {ctranslate2.get_cuda_device_count()} GPU(s)")
PY
  install -d "$HOME/.config/systemd/user/omnius-daemon.service.d"
  cat > "$HOME/.config/systemd/user/omnius-daemon.service.d/egg-cuda-asr.conf" <<EOF
[Service]
Environment="LD_LIBRARY_PATH=$cuda_library_path"
Environment="OMNIUS_TRANSCRIBE_PYTHON=$asr_python"
Environment="TRANSCRIBE_PYTHON=$asr_python"
EOF
  systemctl --user daemon-reload
  systemctl --user try-restart omnius-daemon.service || true
}

ensure_ornith_vision() {
  if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama is required for Ornith Vision object classification." >&2
    return 1
  fi
  if ! ollama show robit/ornith-vision:9b >/dev/null 2>&1; then
    ollama pull robit/ornith-vision:9b
  fi
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
"$venv_pip" install --no-deps open-clip-torch ultralytics sounddevice pyusb webrtcvad-wheels ftfy timm pytest
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
ensure_omnius_cuda_asr
ensure_ornith_vision
EGG_RESPEAKER_PYTHON="$venv_python" "$workspace_dir/scripts/configure-respeaker-aec.sh"
"$venv_python" -m egg_companion --config "$workspace_dir/config/egg.yaml" audit
ensure_companion_service
