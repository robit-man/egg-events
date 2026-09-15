#!/usr/bin/env bash
# Exercise every transducer path in omni mode against the live adapter.
#
# This is a real-hardware check, not a unit test: it captures from the actual
# microphone, runs comprehension on that capture, synthesizes speech through
# Qwen3-TTS, plays it on the actual speaker, and sends a real camera clip to
# the video comprehension layer. Each stage reports PASS/FAIL independently so
# a partial stack is legible rather than a single opaque failure.
set -uo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$workspace_dir/.venv/bin/python"
adapter_url="${EGG_OMNI_ADAPTER_URL:-http://127.0.0.1:8910}"
record_seconds="${EGG_OMNI_RECORD_SECONDS:-4}"
play_audio="${EGG_OMNI_PLAY:-1}"

pass=0
fail=0
report() {
  if [[ $1 == 0 ]]; then
    printf '  PASS  %s\n' "$2"; pass=$((pass + 1))
  else
    printf '  FAIL  %s\n' "$2"; fail=$((fail + 1))
  fi
}

printf '== adapter ==\n'
curl -fsS --max-time 5 "$adapter_url/healthz" >/dev/null 2>&1
report $? "adapter /healthz"
contract=$(curl -fsS --max-time 5 "$adapter_url/api/omni/adapter/contract" 2>/dev/null)
[[ "$contract" == *"robit.ollama.omni-adapter.v1"* ]]
report $? "adapter advertises robit.ollama.omni-adapter.v1"

EGG_OMNI_ADAPTER_URL="$adapter_url" \
EGG_OMNI_RECORD_SECONDS="$record_seconds" \
EGG_OMNI_PLAY="$play_audio" \
"$python_bin" - <<'PY'
import asyncio
import io
import os
import subprocess
import sys
import wave

import numpy as np

sys.path.insert(0, os.getcwd())
from egg_companion.adapters.omni import OmniAdapterClient  # noqa: E402
from egg_companion.config import load_config  # noqa: E402

config = load_config("config/egg.yaml")
adapter = OmniAdapterClient(
    config.omni_adapter.model_copy(
        update={"base_url": os.environ["EGG_OMNI_ADAPTER_URL"]}
    )
)
seconds = float(os.environ["EGG_OMNI_RECORD_SECONDS"])
results: list[tuple[bool, str]] = []


def record_microphone() -> bytes | None:
    """Capture real microphone audio as 16 kHz mono PCM16."""

    completed = subprocess.run(
        [
            "arecord", "-q", "-D", config.audio.input_device,
            "-f", "S16_LE", "-r", "16000", "-c", "1",
            "-d", str(int(seconds)), "-t", "wav",
        ],
        capture_output=True,
    )
    if completed.returncode != 0 or not completed.stdout.startswith(b"RIFF"):
        return None
    return completed.stdout


async def main() -> None:
    print("== microphone -> comprehension ==")
    print(f"  recording {seconds:g}s from {config.audio.input_device}; speak now")
    captured = await asyncio.to_thread(record_microphone)
    if captured is None:
        results.append((False, "microphone capture"))
    else:
        with wave.open(io.BytesIO(captured), "rb") as source:
            frames = source.getnframes()
            peak = float(
                np.abs(
                    np.frombuffer(source.readframes(frames), dtype="<i2")
                ).max()
            ) / 32768
        results.append((frames > 0, f"microphone capture ({frames} frames, peak {peak:.3f})"))
        try:
            perceived = await adapter.perceive_audio(captured)
            transcript = perceived.get("transcript")
            observation = perceived.get("audio_observation")
            print(f"  transcript        : {transcript!r}")
            print(f"  audio observation : {observation!r}")
            # Either channel answering proves the ASR/audio path; a silent room
            # legitimately yields no transcript.
            results.append(
                (bool(transcript or observation), "audio comprehension (speech/sound split)")
            )
        except Exception as error:
            print(f"  error: {type(error).__name__}: {error}")
            results.append((False, "audio comprehension"))

    print("== Qwen3-TTS -> speaker ==")
    try:
        spoken = await adapter.synthesize(
            "Omni mode is online. Every transducer is wired to one weights package."
        )
        with wave.open(io.BytesIO(spoken), "rb") as source:
            rate, channels, width = (
                source.getframerate(), source.getnchannels(), source.getsampwidth()
            )
            duration = source.getnframes() / max(1, rate)
        ok = rate == 24000 and channels == 1 and width == 2
        print(f"  {rate} Hz, {channels}ch, {width * 8}-bit, {duration:.2f}s")
        results.append((ok, "Qwen3-TTS synthesis (24 kHz mono PCM16)"))
        if os.environ.get("EGG_OMNI_PLAY") == "1":
            played = subprocess.run(
                ["aplay", "-q", "-D", config.audio.output_device, "-"],
                input=spoken, capture_output=True,
            )
            results.append((played.returncode == 0, "speaker playback"))
    except Exception as error:
        print(f"  error: {type(error).__name__}: {error}")
        results.append((False, "Qwen3-TTS synthesis"))

    print("== camera -> video comprehension ==")
    try:
        import cv2

        source_device = next(
            (camera.source for camera in config.cameras if camera.enabled), None
        )
        if source_device is None:
            raise RuntimeError("no enabled camera is configured")
        capture = cv2.VideoCapture(source_device, cv2.CAP_V4L2)
        frames = []
        try:
            for _ in range(12):
                ok, frame = capture.read()
                if not ok:
                    break
                height, width = frame.shape[:2]
                scale = 640 / float(width)
                frames.append(
                    cv2.resize(frame, (640, (int(height * scale) // 2) * 2))
                )
        finally:
            capture.release()
        if len(frames) < 2:
            raise RuntimeError(f"camera {source_device} produced {len(frames)} frames")
        path = "/tmp/egg-omni-verify-clip.mp4"
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), 2.0,
            (frames[0].shape[1], frames[0].shape[0]),
        )
        for frame in frames:
            writer.write(frame)
        writer.release()
        with open(path, "rb") as handle:
            clip = handle.read()
        described = await adapter.describe_video(
            clip, prompt="Describe what happens in this clip."
        )
        print(f"  clip: {len(clip)} bytes, {len(frames)} frames from {source_device}")
        print(f"  observation: {described.get('visual_observation')!r}")
        results.append((bool(described.get("visual_observation")), "video comprehension"))
    except Exception as error:
        print(f"  error: {type(error).__name__}: {error}")
        results.append((False, "video comprehension"))

    print()
    for ok, label in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    raise SystemExit(0 if all(ok for ok, _ in results) else 1)


asyncio.run(main())
PY
python_status=$?
report $python_status "transducer round-trips"

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[[ $fail -eq 0 ]]
