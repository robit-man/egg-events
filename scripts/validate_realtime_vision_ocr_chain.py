#!/usr/bin/env python3
"""Live harness for model-selected camera/OCR chaining on one frozen frame.

The harness does not route from transcript words or prescribe a runtime tool
sequence. It gives the live conversation model its normal native tools, lets
it select each next action from accumulated evidence, and verifies that any
camera/OCR calls operate on the same ASR-boundary snapshot.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import cv2
import numpy as np

from egg_companion.config import EggConfig, load_config
from egg_companion.runtime import CompanionRuntime


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


def _isolated_config(path: str, scratch: str) -> EggConfig:
    payload = load_config(path).model_dump()
    payload["memory"]["enabled"] = False
    payload["identity"]["enabled"] = False
    payload["object_learning"]["enabled"] = False
    payload["camera_discovery"]["enabled"] = False
    payload["ocr"]["ledger_db_path"] = str(Path(scratch) / "ocr-harness.sqlite3")
    return EggConfig.model_validate(payload)


def _synthetic_display(code: str) -> np.ndarray:
    frame = np.full((1080, 1920, 3), 24, dtype=np.uint8)
    cv2.rectangle(frame, (450, 260), (1470, 820), (224, 224, 224), -1)
    cv2.rectangle(frame, (450, 260), (1470, 820), (70, 70, 70), 10)
    cv2.putText(
        frame,
        "SYSTEM VERIFICATION",
        (610, 455),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.55,
        (25, 25, 25),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        code,
        (690, 650),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.35,
        (8, 8, 8),
        6,
        cv2.LINE_AA,
    )
    return frame


async def run(args: argparse.Namespace) -> int:
    live_config = load_config(args.config)
    _import_service_token(live_config.omnius.bearer_token_env, args.service)
    with tempfile.TemporaryDirectory(prefix="egg-vision-ocr-harness-") as scratch:
        runtime = CompanionRuntime(_isolated_config(args.config, scratch))
        boundary = time.monotonic()
        runtime._latest_frames = {"harness-front": (_synthetic_display(args.code), boundary)}
        snapshot = runtime._capture_turn_visual_snapshot("vision-ocr-harness", boundary)
        turn = runtime._conversation_turns.finalize_audio_turn(
            (
                "Inspect the display in the current camera and tell me its exact verification "
                "code. If ordinary visual understanding cannot read it reliably, use whatever "
                "available perception tool can resolve the text."
            ),
            utterance_id="vision-ocr-harness",
            started_at=boundary - 0.5,
            ended_at=boundary,
        )
        context = (
            "This is a live capability harness with one camera frame frozen at the accepted "
            "utterance boundary. Select tools from evidence exactly as in normal rapid voice use."
        )
        reply = await runtime._run_realtime_tool_loop(
            turn,
            snapshot,
            turn.text,
            runtime._visual_snapshot_context(snapshot),
            context,
        )
        calls = runtime._turn_tool_calls.get(turn.utterance_id, [])
        names = [str(item.get("name")) for item in calls]
        payload = {
            "reply": reply,
            "tool_names": names,
            "tool_calls": calls,
            "frozen_camera_ids": [item.camera_id for item in snapshot.frames],
            "expected_code": args.code,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if runtime._omnius.parse_realtime_tool_call(reply) is not None:
            raise RuntimeError("model returned an unresolved tool handoff")
        normalized_reply = "".join(character for character in reply.upper() if character.isalnum())
        normalized_code = "".join(
            character for character in args.code.upper() if character.isalnum()
        )
        if normalized_code not in normalized_reply:
            raise RuntimeError(f"final reply did not contain {args.code!r}")
        if not any(name in {"asr_boundary_vision", "camera_advanced_ocr"} for name in names):
            raise RuntimeError("model answered without current pixel evidence")
        if args.require_ocr and "camera_advanced_ocr" not in names:
            raise RuntimeError("model did not select advanced OCR")
        if args.require_ocr and not any(
            item.get("name") == "camera_advanced_ocr" and item.get("success") is True
            for item in calls
        ):
            raise RuntimeError("advanced OCR was selected but did not resolve text")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/egg.yaml")
    parser.add_argument("--service", default="egg-companion.service")
    parser.add_argument("--code", default="ORBIT-729")
    parser.add_argument("--require-ocr", action="store_true")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
