"""Tests for the occupancy-mapping runtime loop and its world-model write.

Uses a stubbed DepthEstimator (no real subprocess/model) to isolate the
runtime orchestration logic: which camera gets picked, how the voxel grid
gets updated, and that the result reaches current_property_state.
"""

from __future__ import annotations

import asyncio

import numpy as np

from egg_companion.adapters.depth import DepthResult
from egg_companion.config import EggConfig
from egg_companion.core.occupancy import VoxelGrid
from egg_companion.memory.pipeline import MemoryPipeline
from egg_companion.memory.store import MemoryStore
from egg_companion.runtime import CompanionRuntime


def _config(tmp_path) -> EggConfig:
    return EggConfig.model_validate(
        {
            "audio": {"input_device": "default", "doa_mode": "disabled"},
            "omnius": {"model": "test", "voice_model": "test"},
            "identity": {"enabled": False},
            "object_learning": {"enabled": False},
            "camera_discovery": {"enabled": False},
            "memory": {"storage_dir": str(tmp_path / "memory")},
            # Defaults to disabled on the real robot (memory pressure); the
            # runtime-orchestration tests below need it on to exercise
            # _run_occupancy_cycle's real behavior.
            "occupancy": {"enabled": True},
        }
    )


class _FakeDepthEstimator:
    def __init__(self, result: DepthResult | None) -> None:
        self.result = result
        self.calls = 0

    async def estimate(self, image_png: bytes) -> DepthResult | None:
        self.calls += 1
        return self.result


def _runtime(tmp_path):
    config = _config(tmp_path)
    store = MemoryStore(config.memory)
    pipeline = MemoryPipeline(config, store)
    runtime = object.__new__(CompanionRuntime)
    runtime.config = config
    runtime._memory = pipeline
    runtime._latest_frames = {}
    runtime._occupancy_grids = {}
    runtime._occupancy_last_update = {}
    return runtime, pipeline


def _fake_depth_result() -> DepthResult:
    return DepthResult(
        depth=np.full((32, 32), 2.0, dtype=np.float32),
        confidence=None,
        model="fake",
    )


class TestRecordDerivedProperty:
    def test_writes_a_string_property_to_world_state(self, tmp_path) -> None:
        config = _config(tmp_path)
        store = MemoryStore(config.memory)
        pipeline = MemoryPipeline(config, store)

        pipeline.record_derived_property(
            "camera_view:camera-video1", "occupancy_summary", "nearest ~1.2m",
        )

        value = pipeline._world_query.property_value(
            "camera_view:camera-video1", "occupancy_summary",
        )
        assert value == "nearest ~1.2m"


class TestOccupancyCycle:
    def test_no_due_camera_does_nothing(self, tmp_path) -> None:
        runtime, _ = _runtime(tmp_path)
        runtime._depth_estimator = _FakeDepthEstimator(_fake_depth_result())

        result = asyncio.run(runtime._run_occupancy_cycle())

        assert result is None
        assert runtime._depth_estimator.calls == 0

    def test_disabled_config_does_nothing(self, tmp_path) -> None:
        runtime, _ = _runtime(tmp_path)
        runtime.config.occupancy.enabled = False
        runtime._depth_estimator = _FakeDepthEstimator(_fake_depth_result())
        runtime._latest_frames["camera-video1"] = (
            np.zeros((10, 10, 3), dtype=np.uint8), 0.0,
        )

        result = asyncio.run(runtime._run_occupancy_cycle())

        assert result is None
        assert runtime._depth_estimator.calls == 0

    def test_due_camera_updates_grid_and_world_state(self, tmp_path) -> None:
        runtime, pipeline = _runtime(tmp_path)
        runtime._depth_estimator = _FakeDepthEstimator(_fake_depth_result())
        runtime._latest_frames["camera-video1"] = (
            np.zeros((10, 10, 3), dtype=np.uint8), 0.0,
        )

        result = asyncio.run(runtime._run_occupancy_cycle())

        assert result == "camera-video1"
        assert runtime._depth_estimator.calls == 1
        assert "camera-video1" in runtime._occupancy_grids
        assert isinstance(runtime._occupancy_grids["camera-video1"], VoxelGrid)
        assert len(runtime._occupancy_grids["camera-video1"]) > 0
        assert "camera-video1" in runtime._occupancy_last_update

        summary = pipeline._world_query.property_value(
            "camera_view:camera-video1", "occupancy_summary",
        )
        assert summary is not None
        assert "nearest occupied surface" in summary

    def test_not_due_yet_is_skipped(self, tmp_path) -> None:
        runtime, _ = _runtime(tmp_path)
        runtime._depth_estimator = _FakeDepthEstimator(_fake_depth_result())
        runtime._latest_frames["camera-video1"] = (
            np.zeros((10, 10, 3), dtype=np.uint8), 0.0,
        )

        first = asyncio.run(runtime._run_occupancy_cycle())
        assert first == "camera-video1"

        second = asyncio.run(runtime._run_occupancy_cycle())
        assert second is None
        assert runtime._depth_estimator.calls == 1

    def test_depth_estimation_failure_does_not_crash_or_write(self, tmp_path) -> None:
        runtime, pipeline = _runtime(tmp_path)
        runtime._depth_estimator = _FakeDepthEstimator(None)
        runtime._latest_frames["camera-video1"] = (
            np.zeros((10, 10, 3), dtype=np.uint8), 0.0,
        )

        result = asyncio.run(runtime._run_occupancy_cycle())

        assert result is None
        summary = pipeline._world_query.property_value(
            "camera_view:camera-video1", "occupancy_summary",
        )
        assert summary is None

    def test_estimator_exception_is_caught_and_logged(self, tmp_path) -> None:
        class RaisingEstimator:
            async def estimate(self, image_png):
                raise RuntimeError("boom")

        runtime, _ = _runtime(tmp_path)
        runtime._depth_estimator = RaisingEstimator()
        runtime._latest_frames["camera-video1"] = (
            np.zeros((10, 10, 3), dtype=np.uint8), 0.0,
        )

        result = asyncio.run(runtime._run_occupancy_cycle())
        assert result is None

    def test_picks_only_one_camera_per_cycle(self, tmp_path) -> None:
        runtime, _ = _runtime(tmp_path)
        runtime._depth_estimator = _FakeDepthEstimator(_fake_depth_result())
        runtime._latest_frames["camera-video1"] = (
            np.zeros((10, 10, 3), dtype=np.uint8), 0.0,
        )
        runtime._latest_frames["camera-video2"] = (
            np.zeros((10, 10, 3), dtype=np.uint8), 0.0,
        )

        result = asyncio.run(runtime._run_occupancy_cycle())

        assert result in ("camera-video1", "camera-video2")
        assert runtime._depth_estimator.calls == 1
        assert len(runtime._occupancy_grids) == 1
