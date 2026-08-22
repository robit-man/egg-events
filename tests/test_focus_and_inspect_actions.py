"""Regression tests for the real focus_camera and inspect_entity executors.

Both action types existed only as ontology vocabulary (parameters
registered, zero dispatch code anywhere) before this. There's no pan/tilt/
zoom hardware on this robot -- all cameras are fixed -- so focus_camera can
only honestly mean a software attention bias (see core/attention.py's
camera_focus_bonus), and inspect_entity can only act on whatever is
currently visible in some camera's latest frame (no way to go find
something that isn't).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import numpy as np

from egg_companion.cognition.architecture import CognitiveArchitecture
from egg_companion.config import EggConfig
from egg_companion.memory.pipeline import MemoryPipeline
from egg_companion.memory.store import MemoryStore
from egg_companion.models import BoundingBox, Detection, Observation
from egg_companion.runtime import CompanionRuntime
from egg_companion.world.policy import PolicyRule


def _config(tmp_path) -> EggConfig:
    return EggConfig.model_validate(
        {
            "audio": {"input_device": "default", "doa_mode": "disabled"},
            "omnius": {"model": "test", "voice_model": "test"},
            "identity": {"enabled": False},
            "object_learning": {"enabled": False},
            "camera_discovery": {"enabled": False},
            "memory": {"storage_dir": str(tmp_path / "memory")},
        }
    )


def _runtime(tmp_path):
    config = _config(tmp_path)
    store = MemoryStore(config.memory)
    pipeline = MemoryPipeline(config, store)
    runtime = object.__new__(CompanionRuntime)
    runtime._memory = pipeline
    runtime._brain = CognitiveArchitecture(None, None, None)
    runtime._latest_frames = {}
    runtime._latest_observations = {}
    return runtime, pipeline


class TestFocusCamera:
    def test_unknown_camera_fails_without_biasing_attention(self, tmp_path) -> None:
        runtime, _ = _runtime(tmp_path)
        result = asyncio.run(runtime.focus_camera("camera-nope"))
        assert result["ok"] is False
        assert "unknown camera_id" in result["reason"]
        assert runtime._brain._active_camera_focus_ids() == []

    def test_known_camera_succeeds_and_biases_attention(self, tmp_path) -> None:
        runtime, _ = _runtime(tmp_path)
        runtime._latest_frames["camera-video1"] = (np.zeros((10, 10, 3), dtype=np.uint8), 0.0)

        result = asyncio.run(runtime.focus_camera("camera-video1", duration_seconds=30.0))

        assert result["ok"] is True
        assert result["camera_id"] == "camera-video1"
        assert runtime._brain._active_camera_focus_ids() == ["camera-video1"]

    def test_nonpositive_duration_fails(self, tmp_path) -> None:
        runtime, _ = _runtime(tmp_path)
        runtime._latest_frames["camera-video1"] = (np.zeros((10, 10, 3), dtype=np.uint8), 0.0)

        result = asyncio.run(runtime.focus_camera("camera-video1", duration_seconds=0.0))

        assert result["ok"] is False
        assert runtime._brain._active_camera_focus_ids() == []

    def test_blocked_by_policy_never_applies_focus(self, tmp_path) -> None:
        runtime, pipeline = _runtime(tmp_path)
        runtime._latest_frames["camera-video1"] = (np.zeros((10, 10, 3), dtype=np.uint8), 0.0)
        pipeline._policy_validator.register(PolicyRule(
            rule_id="test-block-focus", name="block focus", description="",
            action_type="focus_camera", conditions_json='{"max_per_minute": 0}',
            block=True,
        ))

        result = asyncio.run(runtime.focus_camera("camera-video1"))

        assert result["ok"] is False
        assert runtime._brain._active_camera_focus_ids() == []
        assert pipeline._action_store.recent_executions() == []


class TestInspectEntity:
    def test_entity_not_visible_fails(self, tmp_path) -> None:
        runtime, pipeline = _runtime(tmp_path)

        result = asyncio.run(runtime.inspect_entity("object:not-there"))

        assert result["ok"] is False
        assert "not currently visible" in result["reason"]
        executions = pipeline._action_store.recent_executions()
        assert len(executions) == 1
        assert executions[0]["success"] is False

    def test_visible_entity_runs_explicit_analysis_and_records_execution(self, tmp_path) -> None:
        runtime, pipeline = _runtime(tmp_path)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detection = Detection(
            "cup", 0.7, BoundingBox(10, 10, 50, 50), {"object_id": "object:cup-1"},
        )
        observation = Observation("camera-video1", datetime.now(timezone.utc), (detection,))
        runtime._latest_observations["camera-video1"] = observation
        runtime._latest_frames["camera-video1"] = (frame, 0.0)

        seen_explicit_flag = []

        async def fake_classify(image_png, label, confidence, *, explicit_read_request=False):
            seen_explicit_flag.append(explicit_read_request)
            vlm_analysis = {
                "label": "coffee mug",
                "confidence": 0.92,
                "appearance_description": "a white ceramic mug",
            }
            ocr_result = {
                "text": "WORLD'S OKAYEST",
                "confidence": 0.8,
                "engine": "test-engine",
                "regions": [],
            }
            return vlm_analysis, ocr_result

        runtime._classify_with_ocr = fake_classify
        memory_events = []
        runtime._queue_memory_event = lambda event: memory_events.append(event)
        ocr_calls = []
        runtime._queue_ocr_memory = lambda candidate, text, result: ocr_calls.append(
            (candidate, text, result)
        )

        result = asyncio.run(runtime.inspect_entity("object:cup-1"))

        assert seen_explicit_flag == [True]
        assert result["ok"] is True
        assert result["camera_id"] == "camera-video1"
        assert result["text_found"] is True
        assert result["text"] == "WORLD'S OKAYEST"
        assert result["appearance"] == "a white ceramic mug"

        assert len(ocr_calls) == 1
        candidate, text, ocr_result = ocr_calls[0]
        assert candidate.trigger == "inspect_entity"
        assert candidate.parent_id == "object:cup-1"
        assert text == "WORLD'S OKAYEST"

        assert len(memory_events) == 1
        assert memory_events[0].event_type == "object"
        assert memory_events[0].entity_ids == ("object:cup-1",)
        assert memory_events[0].payload["detections"][0]["entity_id"] == "object:cup-1"
        assert memory_events[0].payload["detections"][0]["label"] == "coffee mug"

        executions = pipeline._action_store.recent_executions()
        assert len(executions) == 1
        assert executions[0]["success"] is True

    def test_no_text_found_skips_ocr_queue_but_still_refreshes_entity(self, tmp_path) -> None:
        runtime, _ = _runtime(tmp_path)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detection = Detection(
            "cup", 0.7, BoundingBox(10, 10, 50, 50), {"object_id": "object:cup-2"},
        )
        observation = Observation("camera-video1", datetime.now(timezone.utc), (detection,))
        runtime._latest_observations["camera-video1"] = observation
        runtime._latest_frames["camera-video1"] = (frame, 0.0)

        async def fake_classify(image_png, label, confidence, *, explicit_read_request=False):
            return {"label": "cup", "confidence": 0.5, "appearance_description": "a plain cup"}, None

        runtime._classify_with_ocr = fake_classify
        memory_events = []
        runtime._queue_memory_event = lambda event: memory_events.append(event)
        ocr_calls = []
        runtime._queue_ocr_memory = lambda candidate, text, result: ocr_calls.append(candidate)

        result = asyncio.run(runtime.inspect_entity("object:cup-2"))

        assert result["ok"] is True
        assert result["text_found"] is False
        assert result["text"] is None
        assert ocr_calls == []
        assert len(memory_events) == 1

    def test_blocked_by_policy_never_runs_analysis(self, tmp_path) -> None:
        runtime, pipeline = _runtime(tmp_path)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detection = Detection(
            "cup", 0.7, BoundingBox(10, 10, 50, 50), {"object_id": "object:cup-3"},
        )
        observation = Observation("camera-video1", datetime.now(timezone.utc), (detection,))
        runtime._latest_observations["camera-video1"] = observation
        runtime._latest_frames["camera-video1"] = (frame, 0.0)
        pipeline._policy_validator.register(PolicyRule(
            rule_id="test-block-inspect", name="block inspect", description="",
            action_type="inspect_entity", conditions_json='{"max_per_minute": 0}',
            block=True,
        ))

        calls = []

        async def fake_classify(*args, **kwargs):
            calls.append(1)
            return None, None

        runtime._classify_with_ocr = fake_classify

        result = asyncio.run(runtime.inspect_entity("object:cup-3"))

        assert result["ok"] is False
        assert calls == []
