"""Tests for the occupancy-mapping runtime loop and its world-model write.

Uses a stubbed DepthEstimator (no real subprocess/model) to isolate the
runtime orchestration logic: which camera gets picked, how the voxel grid
gets updated, and that the result reaches current_property_state.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

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
    runtime._occupancy_grid = VoxelGrid(
        voxel_size_meters=config.occupancy.voxel_size_meters,
        max_range_meters=config.occupancy.max_range_meters,
        max_voxels=config.occupancy.max_voxels,
    )
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
        assert isinstance(runtime._occupancy_grid, VoxelGrid)
        assert len(runtime._occupancy_grid) > 0
        assert "camera-video1" in runtime._occupancy_last_update

        summary = pipeline._world_query.property_value(
            "environment:egg", "occupancy_summary",
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
            "environment:egg", "occupancy_summary",
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
        # Only the camera actually integrated this cycle gets an
        # updated timestamp -- the other stays due for the next cycle.
        assert len(runtime._occupancy_last_update) == 1

    def test_sample_stride_is_read_live_from_config_each_cycle(self, tmp_path) -> None:
        """A denser sample_stride must actually integrate more points --
        this is the whole point of the Resolution +/- dashboard control,
        wired through config.occupancy.sample_stride rather than the
        integrate_depth() default that used to be silently hardcoded."""
        runtime, _ = _runtime(tmp_path / "a")
        runtime.config.occupancy.sample_stride = 1
        runtime._depth_estimator = _FakeDepthEstimator(
            DepthResult(depth=np.full((16, 16), 2.0, dtype=np.float32), confidence=None, model="fake")
        )
        runtime._latest_frames["camera-video1"] = (
            np.zeros((16, 16, 3), dtype=np.uint8), 0.0,
        )

        asyncio.run(runtime._run_occupancy_cycle())
        dense_voxel_count = len(runtime._occupancy_grid)

        runtime2, _ = _runtime(tmp_path / "b")
        runtime2.config.occupancy.sample_stride = 8
        runtime2._depth_estimator = _FakeDepthEstimator(
            DepthResult(depth=np.full((16, 16), 2.0, dtype=np.float32), confidence=None, model="fake")
        )
        runtime2._latest_frames["camera-video1"] = (
            np.zeros((16, 16, 3), dtype=np.uint8), 0.0,
        )
        asyncio.run(runtime2._run_occupancy_cycle())
        coarse_voxel_count = len(runtime2._occupancy_grid)

        assert dense_voxel_count > coarse_voxel_count

    def test_yaw_auto_computed_from_live_camera_set_not_hardcoded(self, tmp_path) -> None:
        """A camera beyond the physically-mounted 4 (e.g. camera-video4)
        must still integrate with a correctly auto-computed yaw, not
        silently default to yaw=0 for lack of a hardcoded mapping entry."""
        runtime, _ = _runtime(tmp_path)
        runtime._depth_estimator = _FakeDepthEstimator(_fake_depth_result())
        for camera_id in ("camera-video0", "camera-video1", "camera-video2", "camera-video3", "camera-video4"):
            runtime._latest_frames[camera_id] = (np.zeros((10, 10, 3), dtype=np.uint8), 0.0)

        # Drain the queue until camera-video4 gets its turn.
        seen = set()
        for _ in range(10):
            result = asyncio.run(runtime._run_occupancy_cycle())
            if result:
                seen.add(result)
            if "camera-video4" in seen:
                break

        assert "camera-video4" in seen
        snapshot = runtime.occupancy_snapshot()
        assert snapshot["camera_yaw_degrees"]["camera-video4"] == pytest.approx(120.0)


class TestOccupancySnapshot:
    def test_includes_sample_stride_and_dynamic_camera_yaw(self, tmp_path) -> None:
        runtime, _ = _runtime(tmp_path)
        runtime.config.occupancy.sample_stride = 4
        runtime._latest_frames["camera-video0"] = (np.zeros((4, 4, 3), dtype=np.uint8), 0.0)
        runtime._latest_frames["camera-video1"] = (np.zeros((4, 4, 3), dtype=np.uint8), 0.0)

        snapshot = runtime.occupancy_snapshot()

        assert snapshot["sample_stride"] == 4
        assert snapshot["camera_yaw_degrees"]["camera-video0"] == pytest.approx(-30.0)
        assert snapshot["camera_yaw_degrees"]["camera-video1"] == pytest.approx(30.0)


class TestUpdateOccupancyResolution:
    def test_applies_and_clamps_sample_stride(self, tmp_path) -> None:
        runtime, _ = _runtime(tmp_path)

        assert runtime.update_occupancy_resolution(3) == 3
        assert runtime.config.occupancy.sample_stride == 3

        assert runtime.update_occupancy_resolution(0) == 1  # clamped to minimum
        assert runtime.update_occupancy_resolution(999) == 32  # clamped to maximum
