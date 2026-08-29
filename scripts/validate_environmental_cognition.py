#!/usr/bin/env python3
"""Validate Egg's event-driven environmental cognition contract.

The deterministic half exercises structural person/scene change, habituation,
and salience falloff. ``--live`` additionally freezes the running dashboard's
fresh camera frames, asks Ornith to ground them, supplies bounded live
memory/world telemetry, and verifies that Ornith independently returns one of
the strict silence/reflect/speak/ask outcomes. The harness never plays TTS.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp

from egg_companion.adapters.omnius import OmniusClient
from egg_companion.config import EnvironmentalCognitionConfig, load_config
from egg_companion.core.environmental_cognition import EnvironmentalNoveltyTracker
from egg_companion.models import BoundingBox, Detection, Observation


def _import_service_token(environment_name: str | None, service: str) -> None:
    if not environment_name or os.getenv(environment_name):
        return
    pid = subprocess.check_output(
        ["systemctl", "--user", "show", service, "-p", "MainPID", "--value"],
        text=True,
    ).strip()
    if not pid or pid == "0":
        raise RuntimeError(f"{service} has no live process")
    prefix = f"{environment_name}=".encode()
    for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
        if item.startswith(prefix):
            os.environ[environment_name] = item[len(prefix) :].decode()
            return
    raise RuntimeError(f"{environment_name} is unavailable in {service}")


def _observation(
    at: datetime,
    *,
    people: tuple[str, ...] = (),
    objects: tuple[str, ...] = (),
) -> Observation:
    detections = [
        Detection(
            "person",
            0.9,
            BoundingBox(10 + index * 20, 10, 90 + index * 20, 190),
            {"identity_id": identity_id, "frame_shape": [240, 320]},
        )
        for index, identity_id in enumerate(people)
    ]
    detections.extend(
        Detection(
            "object",
            0.8,
            BoundingBox(20 + index * 20, 40, 70 + index * 20, 100),
            {"object_id": object_id, "frame_shape": [240, 320]},
        )
        for index, object_id in enumerate(objects)
    )
    return Observation("harness-camera", at, tuple(detections))


def deterministic_contract() -> dict[str, object]:
    settings = EnvironmentalCognitionConfig(
        minimum_salience=0.01,
        salience_half_life_seconds=10,
        habituation_half_life_seconds=100,
    )
    tracker = EnvironmentalNoveltyTracker(settings)
    wall = datetime.now(timezone.utc)
    assert tracker.observe(_observation(wall), [], 0.0, 0.0) is None
    arrival = tracker.observe(
        _observation(wall + timedelta(seconds=1), people=("person-1",)),
        [],
        0.0,
        1.0,
    )
    assert arrival is not None and "person_presence_changed" in arrival.causes
    assert tracker.observe(
        _observation(wall + timedelta(seconds=2), people=("person-1",)),
        [],
        0.0,
        2.0,
    ) is None
    departure = tracker.observe(
        _observation(wall + timedelta(seconds=3)), [], 0.0, 3.0
    )
    repeat = tracker.observe(
        _observation(wall + timedelta(seconds=4), people=("person-1",)),
        [],
        0.0,
        4.0,
    )
    tracker.observe(_observation(wall + timedelta(seconds=5)), [], 0.0, 5.0)
    changed_scene = tracker.observe(
        _observation(wall + timedelta(seconds=6), objects=("object-new",)),
        [],
        0.0,
        6.0,
    )
    assert departure is not None
    assert repeat is not None and repeat.salience < arrival.salience
    assert changed_scene is not None
    half_life_value = arrival.decayed_salience(11.0, 10.0)
    assert abs(half_life_value - arrival.salience / 2) < 1e-6
    return {
        "person_arrival": arrival.salience,
        "stable_continuation_triggered": False,
        "repeated_arrival": repeat.salience,
        "departure": departure.salience,
        "scene_change": changed_scene.salience,
        "salience_after_one_half_life": half_life_value,
    }


async def _get_json(
    session: aiohttp.ClientSession, url: str
) -> dict[str, object]:
    async with session.get(url, headers={"Cache-Control": "no-cache"}) as response:
        response.raise_for_status()
        payload = await response.json()
    if not isinstance(payload, dict):
        raise TypeError(f"{url} did not return an object")
    return payload


async def live_contract(args: argparse.Namespace) -> dict[str, object]:
    config = load_config(args.config)
    _import_service_token(config.omnius.bearer_token_env, args.service)
    client = OmniusClient(config.omnius)
    dashboard = args.dashboard.rstrip("/")
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        state = await _get_json(session, f"{dashboard}/api/state")
        telemetry = state.get("telemetry")
        if not isinstance(telemetry, dict):
            raise TypeError("live dashboard has no runtime telemetry")
        camera_records = [
            item
            for item in telemetry.get("cameras", [])
            if isinstance(item, dict) and item.get("id")
        ][: config.omnius.visual_snapshot_max_cameras]
        if not camera_records:
            raise RuntimeError("live runtime has no cameras")

        async def frame(record: dict[str, object]) -> tuple[str, bytes, str]:
            camera_id = str(record["id"])
            async with session.get(
                f"{dashboard}/api/cameras/{camera_id}/raw.jpg",
                headers={"Cache-Control": "no-cache"},
            ) as response:
                response.raise_for_status()
                payload = await response.read()
            return camera_id, payload, datetime.now(timezone.utc).isoformat()

        frames = list(await asyncio.gather(*(frame(item) for item in camera_records)))

    signal = {
        "stimulus_id": "environmental-harness-live",
        "sequence": 1,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "salience_now": 0.8,
        "structural_causes": ["live_harness_fresh_scene_sample"],
        "note": "Harness admission only; not a semantic conclusion or speech command.",
    }
    started = time.monotonic()
    assessment = await client.assess_environmental_change(frames, signal, None)
    if assessment is None:
        raise RuntimeError("live Ornith VLM assessment violated its JSON contract")
    visual_seconds = time.monotonic() - started
    memory_context = json.dumps(
        {
            "brain": telemetry.get("brain", {}),
            "environmental_cognition": telemetry.get(
                "environmental_cognition", {}
            ),
            "default_mode": telemetry.get("default_mode", {}),
            "memory_summary": state.get("memory", {}),
        },
        ensure_ascii=False,
        default=str,
    )[:6500]
    started = time.monotonic()
    deliberation = await client.deliberate_environmental_response(
        assessment,
        signal,
        memory_context,
        [],
    )
    if deliberation is None:
        raise RuntimeError("live Ornith deliberation violated its JSON contract")
    decision_seconds = time.monotonic() - started
    if deliberation["action"] in {"speak", "ask"}:
        if assessment.get("people_visible") is not True:
            raise RuntimeError("model selected speech without a visible person")
        if not deliberation.get("utterance"):
            raise RuntimeError("model selected speech without an utterance")
    return {
        "camera_ids": [item[0] for item in frames],
        "visual_seconds": round(visual_seconds, 3),
        "decision_seconds": round(decision_seconds, 3),
        "assessment": assessment,
        "deliberation": deliberation,
        "tts_played": False,
    }


async def run(args: argparse.Namespace) -> int:
    try:
        deterministic = deterministic_contract()
        print("PASS deterministic", json.dumps(deterministic, sort_keys=True))
        if args.live:
            live = await live_contract(args)
            print("PASS live-model", json.dumps(live, ensure_ascii=False, sort_keys=True))
        print(f"SUMMARY deterministic=pass live={'pass' if args.live else 'skipped'}")
        return 0
    except (
        aiohttp.ClientError,
        asyncio.TimeoutError,
        AssertionError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"FAIL {type(error).__name__}: {error}")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/egg.yaml")
    parser.add_argument("--service", default="egg-companion.service")
    parser.add_argument("--dashboard", default="http://127.0.0.1:8788")
    parser.add_argument("--live", action="store_true")
    raise SystemExit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
