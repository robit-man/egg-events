#!/usr/bin/env python3
"""Round-trip a phrase: Qwen3-TTS speaks it, Qwen3-Omni transcribes it back.

This proves the ASR path with a known input, independently of whether anyone
is at the microphone. Anything the model returns has to have come from audio
it actually decoded -- the text is never given to the comprehension stage.
"""
from __future__ import annotations

import asyncio
import io
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egg_companion.adapters.omni import OmniAdapterClient  # noqa: E402
from egg_companion.config import load_config  # noqa: E402

PHRASES = ["The kettle is boiling in the kitchen."]


def to_16k_mono(wav_bytes: bytes) -> bytes:
    """Qwen3-TTS emits 24 kHz; the comprehension contract wants 16 kHz mono."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as source:
        rate = source.getframerate()
        samples = np.frombuffer(
            source.readframes(source.getnframes()), dtype="<i2"
        ).astype(np.float32) / 32768
    count = max(1, round(samples.size * 16000 / rate))
    samples = np.interp(
        np.linspace(0, 1, count, endpoint=False),
        np.linspace(0, 1, samples.size, endpoint=False),
        samples,
    )
    out = io.BytesIO()
    with wave.open(out, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes((samples * 32767).astype("<i2").tobytes())
    return out.getvalue()


async def main() -> int:
    root = Path(__file__).resolve().parents[1]
    config = load_config(str(root / "config/egg.yaml"))
    client = OmniAdapterClient(
        config.omni_adapter.model_copy(update={
            "base_url": "http://127.0.0.1:8910",
            "mode": "omni",
            "speech_enabled": True,
            "voice_profile_path": str(
                root / "vendor/qwen-omni-adapters/portal/voice-profile.json"
            ),
        })
    )
    passed = 0
    for index, phrase in enumerate(PHRASES, 1):
        print(f"--- round trip {index}/{len(PHRASES)} ---", flush=True)
        print(f"  say      : {phrase!r}", flush=True)
        started = time.monotonic()
        spoken = await client.synthesize(phrase)
        tts_ms = (time.monotonic() - started) * 1000
        audio = to_16k_mono(spoken)
        with wave.open(io.BytesIO(audio), "rb") as source:
            seconds = source.getnframes() / source.getframerate()
        print(f"  tts      : {tts_ms:.0f}ms -> {seconds:.2f}s of audio", flush=True)
        started = time.monotonic()
        heard = await client.perceive_audio(audio)
        asr_ms = (time.monotonic() - started) * 1000
        transcript = str(heard.get("transcript") or "").strip()
        print(f"  heard    : {transcript!r}  ({asr_ms:.0f}ms)", flush=True)
        expected = set(phrase.lower().strip(".").split())
        got = set(transcript.lower().strip(".").split())
        overlap = len(expected & got) / max(1, len(expected))
        print(f"  overlap  : {overlap:.0%}\n", flush=True)
        passed += overlap >= 0.6
    print(f"round trips at >=60% word overlap: {passed}/{len(PHRASES)}")
    return 0 if passed == len(PHRASES) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
