#!/usr/bin/env python3
"""End-to-end check of the managed omni pipeline, from nothing loaded.

Every heavy component is admitted by the residency manager rather than started
by hand, so this exercises the guarantee as well as the capability: speech is
synthesized, fed back through comprehension, and the transcript compared --
with the manager loading and evicting underneath to keep the module inside its
budget the whole way.
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
from egg_companion.services.residency import available_memory_gib  # noqa: E402
from egg_companion.services.residency_wiring import (  # noqa: E402
    build_residency_manager,
)

PHRASE = "The kettle is boiling in the kitchen."


def to_16k_mono(wav_bytes: bytes) -> bytes:
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


def mem(label: str) -> None:
    print(f"  {label:<34} available {available_memory_gib():5.1f} GiB", flush=True)


async def main() -> int:
    root = Path(__file__).resolve().parents[1]
    config = load_config(str(root / "config/egg.yaml"))
    manager = build_residency_manager(config)
    if manager is None:
        print("residency is disabled; nothing to verify")
        return 1

    client = OmniAdapterClient(
        config.omni_adapter.model_copy(update={
            "mode": "omni",
            "speech_enabled": True,
            "voice_profile_path": str(
                root / "vendor/qwen-omni-adapters/portal/voice-profile.json"
            ),
        }),
        manager,
    )

    print("== managed omni pipeline ==", flush=True)
    mem("start (nothing loaded)")

    started = time.monotonic()
    spoken = await client.synthesize(PHRASE)
    speak_ms = (time.monotonic() - started) * 1000
    with wave.open(io.BytesIO(spoken), "rb") as source:
        seconds = source.getnframes() / source.getframerate()
    print(f"  speech   : {speak_ms:.0f}ms -> {seconds:.2f}s of 24 kHz audio", flush=True)
    mem("after speech")

    audio = to_16k_mono(spoken)
    started = time.monotonic()
    heard = await client.perceive_audio(audio)
    asr_ms = (time.monotonic() - started) * 1000
    transcript = str(heard.get("transcript") or "").strip()
    observation = heard.get("audio_observation")
    print(f"  heard    : {transcript!r}  ({asr_ms:.0f}ms)", flush=True)
    if observation:
        print(f"  sound    : {observation!r}", flush=True)
    mem("after comprehension")

    expected = set(PHRASE.lower().strip(".").split())
    got = set(transcript.lower().strip(".").split())
    overlap = len(expected & got) / max(1, len(expected))
    print(f"  overlap  : {overlap:.0%}", flush=True)

    status = manager.status()
    print("\n== residency ==", flush=True)
    print(
        f"  budget   : {status['total_gib']} GiB total, "
        f"{status['reserve_gib']} GiB reserved, "
        f"{status['available_gib']} GiB available",
        flush=True,
    )
    for name, item in sorted(status["components"].items()):
        print(f"  {name:<20} {item['cost_gib']:>5} GiB  priority {item['priority']}",
              flush=True)

    ok = overlap >= 0.6
    print(f"\n{'PASS' if ok else 'FAIL'}: managed round trip", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
