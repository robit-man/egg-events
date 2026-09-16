#!/usr/bin/env python3
"""Drive a real ASR -> comprehension -> language -> TTS conversation loop.

Each turn records from the actual microphone, sends it to Qwen3-Omni as audio
(never as text), lets the adapter route comprehension -> language -> Qwen3-TTS,
and plays the spoken reply on the actual speaker. Nothing here is mocked: the
only inputs are the microphone and the model.

Usage: omni-conversation-loop.py [--turns N] [--seconds S]
"""

from __future__ import annotations

import argparse
import asyncio
import io
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egg_companion.adapters.omni import OmniAdapterClient  # noqa: E402
from egg_companion.config import load_config  # noqa: E402


def record(
    device: str, seconds: float, channels: int, asr_channel: int,
    target_rms: float, max_gain: float,
) -> bytes | None:
    """Capture from the array exactly the way the companion does.

    The ReSpeaker exposes six channels: the DSP/AEC processed stream the
    companion transcribes, four raw mics, and a playback reference. Asking
    ALSA for one channel downmixes all six -- including the reference -- which
    buries speech under whatever is playing and under the chassis hum. Take
    the configured channel instead, then apply the same RMS normalization the
    companion uses, because the DSP stream is quiet by design.
    """

    completed = subprocess.run(
        [
            "arecord", "-q", "-D", device,
            "-f", "S16_LE", "-r", "16000", "-c", str(channels),
            "-d", str(int(seconds)), "-t", "wav",
        ],
        capture_output=True,
    )
    if completed.returncode != 0 or not completed.stdout.startswith(b"RIFF"):
        return None
    with wave.open(io.BytesIO(completed.stdout), "rb") as source:
        frames = source.readframes(source.getnframes())
    samples = np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        usable = (samples.size // channels) * channels
        samples = samples[:usable].reshape(-1, channels)[:, asr_channel]
    audio = samples.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    if rms > 0:
        audio = np.clip(audio * min(max_gain, target_rms / rms), -1.0, 1.0)
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes((audio * 32767).astype("<i2").tobytes())
    return output.getvalue()


def peak_level(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes), "rb") as source:
        frames = source.readframes(source.getnframes())
    if not frames:
        return 0.0
    return float(np.abs(np.frombuffer(frames, dtype="<i2")).max()) / 32768


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8910")
    args = parser.parse_args()

    config = load_config("config/egg.yaml")
    adapter = OmniAdapterClient(
        config.omni_adapter.model_copy(
            update={
                "base_url": args.endpoint,
                "mode": "omni",
                "speech_enabled": True,
                "voice_profile_path":
                    "vendor/qwen-omni-adapters/portal/voice-profile.json",
            }
        )
    )
    input_device = config.audio.input_device
    output_device = config.audio.output_device
    history: list[dict[str, str]] = []
    spoken_turns = 0

    print(
        f"microphone: {input_device} (ch{config.audio.asr_channel} of "
        f"{config.audio.channels})   speaker: {output_device}"
    )
    print(f"{args.turns} turns, {args.seconds:g}s of listening each\n")

    for turn in range(1, args.turns + 1):
        print(f"--- turn {turn}/{args.turns} ---")
        print(f"  listening {args.seconds:g}s ...", flush=True)
        captured = await asyncio.to_thread(
            record, input_device, args.seconds,
            config.audio.channels, config.audio.asr_channel,
            config.audio.asr_target_rms, config.audio.asr_max_gain,
        )
        if captured is None:
            print("  microphone capture failed")
            continue

        started = time.monotonic()
        perceived = await adapter.perceive_audio(captured)
        asr_ms = (time.monotonic() - started) * 1000
        transcript = perceived.get("transcript")
        observation = perceived.get("audio_observation")
        # The array's own DSP VAD is independent evidence: it separates "the
        # model missed speech" from "there was no speech to miss".
        try:
            from egg_companion.adapters.audio import read_respeaker_dsp_status

            dsp = read_respeaker_dsp_status(config.audio, diagnostics=False)
            vad = f" dsp_speech={dsp.get('speech_detected')}"
        except Exception:
            vad = ""
        print(f"  peak {peak_level(captured):.3f}  asr {asr_ms:.0f}ms{vad}")
        print(f"  heard    : {transcript!r}")
        if observation:
            print(f"  sound    : {observation!r}")

        if not transcript:
            # Sound with no speech is context for the next spoken turn, never a
            # request to answer.
            print("  (no speech -- retained as context, not answered)\n")
            continue

        history.append({"role": "user", "content": str(transcript)})
        started = time.monotonic()
        reply = await adapter.chat_with_speech(history[-8:])
        reply_ms = (time.monotonic() - started) * 1000
        text = reply.get("text") or ""
        audio = reply.get("audio")
        print(f"  reply    : {text!r}  ({reply_ms:.0f}ms)")
        history.append({"role": "assistant", "content": text})

        if audio:
            with wave.open(io.BytesIO(audio), "rb") as source:
                duration = source.getnframes() / max(1, source.getframerate())
            print(f"  speaking : {duration:.2f}s of 24 kHz audio")
            await asyncio.to_thread(
                subprocess.run,
                ["aplay", "-q", "-D", output_device, "-"],
                input=audio,
            )
            spoken_turns += 1
        else:
            print("  (no audio returned)")
        print()

    print(f"completed: {spoken_turns} spoken replies over {args.turns} turns")
    return 0 if spoken_turns else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
