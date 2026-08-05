from __future__ import annotations

import asyncio
import json
import time
from urllib.parse import urljoin

import aiohttp


async def trace_live_runtime(base_url: str, seconds: float) -> tuple[bool, dict[str, object]]:
    root = base_url.rstrip("/") + "/"
    timeout = aiohttp.ClientTimeout(total=max(10.0, seconds + 5))
    snapshots: list[dict[str, object]] = []
    started = time.monotonic()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while time.monotonic() - started < seconds:
            async with session.get(urljoin(root, "api/state"), headers={"Cache-Control": "no-store"}) as response:
                response.raise_for_status()
                snapshots.append(await response.json())
            await asyncio.sleep(0.25)
        final = snapshots[-1]
        streams: dict[str, dict[str, object]] = {}
        for camera in final.get("telemetry", {}).get("cameras", []):
            url = urljoin(root, str(camera["raw_stream_url"]).lstrip("/"))
            try:
                async with session.get(url) as response:
                    payload = await response.content.read(128)
                    streams[str(camera["id"])] = {
                        "status": response.status,
                        "content_type": response.headers.get("Content-Type"),
                        "boundary_received": b"--eggframe" in payload,
                    }
            except Exception as error:
                streams[str(camera["id"])] = {"error": str(error)}

    first, final = snapshots[0], snapshots[-1]
    first_telemetry = first.get("telemetry", {})
    final_telemetry = final.get("telemetry", {})
    first_cameras = {item["id"]: item for item in first_telemetry.get("cameras", [])}
    camera_checks: dict[str, dict[str, object]] = {}
    for camera in final_telemetry.get("cameras", []):
        camera_id = str(camera["id"])
        initial = first_cameras.get(camera_id, {})
        sequence_delta = int(camera.get("frame_sequence") or 0) - int(initial.get("frame_sequence") or 0)
        detection_delta = int(camera.get("detection_sequence") or 0) - int(
            initial.get("detection_sequence") or 0
        )
        stream = streams.get(camera_id, {})
        mask_count = sum(
            1 for detection in camera.get("detections", []) if detection.get("mask_polygon")
        )
        camera_checks[camera_id] = {
            "sequence_delta": sequence_delta,
            "detection_sequence_delta": detection_delta,
            "fps": camera.get("fps"),
            "inference_fps": camera.get("inference_fps"),
            "rotation": camera.get("resolved_rotation"),
            "raw_stream": stream,
            "detections": len(camera.get("detections", [])),
            "instance_masks": mask_count,
            "inference_updated_at": camera.get("detections_updated_at"),
            "raw_overlay_independent": sequence_delta > detection_delta,
            "pass": bool(
                sequence_delta >= max(2, int(seconds))
                and detection_delta >= 1
                and sequence_delta > detection_delta
                and float(camera.get("fps") or 0) >= 1.0
                and camera.get("resolved_rotation") is not None
                and stream.get("status") == 200
                and stream.get("boundary_received") is True
            ),
        }
    waveform_delta = int(final_telemetry.get("waveform_sequence") or 0) - int(
        first_telemetry.get("waveform_sequence") or 0
    )
    checks = final.get("checks", [])
    first_asr = first_telemetry.get("asr", {})
    final_asr = final_telemetry.get("asr", {})
    first_learning = first_telemetry.get("object_learning", {})
    final_learning = final_telemetry.get("object_learning", {})
    first_memory = first.get("memory", {}).get("stats", {})
    final_memory = final.get("memory", {}).get("stats", {})
    first_context_counts = (
        first_telemetry.get("memory", {}).get("lifecycle", {}).get("accepted_by_context", {})
    )
    final_context_counts = (
        final_telemetry.get("memory", {}).get("lifecycle", {}).get("accepted_by_context", {})
    )
    visual_episode_starts = sum(
        max(0, int(count) - int(first_context_counts.get(context, 0)))
        for context, count in final_context_counts.items()
        if context != "conversation"
    )
    visual_episode_rate = round(
        visual_episode_starts / max(seconds / 60, 1 / 60), 4
    )
    memory_deltas = {
        key: int(final_memory.get(key) or 0) - int(first_memory.get(key) or 0)
        for key in set(first_memory) | set(final_memory)
    }
    learning_counters = (
        "stable_candidates",
        "duplicate_candidates",
        "clip_queries",
        "clip_recalls",
        "vlm_requests",
        "vlm_successes",
        "vlm_rejections",
        "vlm_errors",
        "speech_deferrals",
    )
    report = {
        "duration_seconds": seconds,
        "runtime": final.get("runtime"),
        "camera_checks": camera_checks,
        "waveform": {
            "sequence_delta": waveform_delta,
            "updated_at": final_telemetry.get("waveform_updated_at"),
            "rms": final_telemetry.get("audio_rms"),
            "pass": waveform_delta >= max(5, int(seconds * 8)),
        },
        "vad": final_telemetry.get("vad"),
        "latest_transcript": final_telemetry.get("latest_transcript"),
        "latest_transcript_at": final_telemetry.get("latest_transcript_at"),
        "transcript_count_delta": int(final_telemetry.get("transcript_count") or 0)
        - int(first_telemetry.get("transcript_count") or 0),
        "asr": {
            **final_asr,
            "accepted_delta": int(final_asr.get("accepted") or 0)
            - int(first_asr.get("accepted") or 0),
            "rejected_delta": int(final_asr.get("rejected") or 0)
            - int(first_asr.get("rejected") or 0),
            "errors_delta": int(final_asr.get("errors") or 0)
            - int(first_asr.get("errors") or 0),
        },
        "object_learning": {
            **final_learning,
            "deltas": {
                key: int(final_learning.get(key) or 0) - int(first_learning.get(key) or 0)
                for key in learning_counters
            },
        },
        "microphone_direction": final_telemetry.get("microphone_direction"),
        "memory": {
            **final_memory,
            "deltas": memory_deltas,
            "episodes_per_minute": round(
                int(memory_deltas.get("episodes", 0)) / max(seconds / 60, 1 / 60), 4
            ),
            "visual_episode_starts": visual_episode_starts,
            "visual_episodes_per_minute": visual_episode_rate,
            "static_scene_ceiling_per_minute": len(camera_checks),
        },
        "runtime_errors": final_telemetry.get("runtime_errors", []),
        "hardware_failures": [check for check in checks if check.get("status") == "fail"],
        "hardware_warnings": [check for check in checks if check.get("status") == "warn"],
    }
    passed = bool(
        str(final.get("runtime", "")).startswith("live")
        and camera_checks
        and any(item["pass"] for item in camera_checks.values())
        and report["waveform"]["pass"]
        and visual_episode_rate <= len(camera_checks)
        and not report["hardware_failures"]
    )
    report["status"] = "pass" if passed else "fail"
    return passed, report


def format_live_trace(report: dict[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)
