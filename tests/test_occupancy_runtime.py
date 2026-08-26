"""Tests for the occupancy-mapping runtime loop and its world-model write.

Uses a stubbed DepthEstimator (no real subprocess/model) to isolate the
runtime orchestration logic: which camera gets picked, how the voxel grid
gets updated, and that the result reaches current_property_state.
"""

from __future__ import annotations

import asyncio
import time

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
    runtime._occupancy_base_stride = config.occupancy.sample_stride
    runtime._occupancy_base_voxel_size_meters = config.occupancy.voxel_size_meters
    runtime._occupancy_last_update = {}
    runtime._occupancy_previous_grid = None
    runtime._occupancy_previous_grid_since = 0.0
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


class TestEncodeFrameForDepth:
    """_encode_frame_for_depth is the occupancy pipeline's own frame
    encode, deliberately separate from _encode_frame (which downscales
    to vision.dashboard_max_width for the lossy dashboard preview
    stream) -- this is what lets a full-4K-captured frame actually reach
    the depth model instead of being silently cropped to preview size
    first."""

    def _decode(self, encoded: bytes):
        import cv2

        buffer = np.frombuffer(encoded, dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    def test_passes_a_4k_frame_through_at_full_resolution(self, tmp_path) -> None:
        runtime, _ = _runtime(tmp_path)
        runtime.config.occupancy.max_input_width = 3840
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

        encoded = runtime._encode_frame_for_depth(frame)
        decoded = self._decode(encoded)

        assert decoded.shape[:2] == (2160, 3840)

    def test_downscales_only_when_exceeding_max_input_width(self, tmp_path) -> None:
        runtime, _ = _runtime(tmp_path)
        runtime.config.occupancy.max_input_width = 640
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        encoded = runtime._encode_frame_for_depth(frame)
        decoded = self._decode(encoded)

        assert decoded.shape[1] == 640
        assert decoded.shape[0] == pytest.approx(360, abs=1)

    def test_encodes_lossless_png_not_lossy_jpeg(self, tmp_path) -> None:
        runtime, _ = _runtime(tmp_path)
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        frame[8, 8] = (17, 31, 53)  # a single distinctive pixel

        encoded = runtime._encode_frame_for_depth(frame)
        decoded = self._decode(encoded)

        assert tuple(int(c) for c in decoded[8, 8]) == (17, 31, 53)


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

        assert runtime.update_occupancy_resolution(3)["sample_stride"] == 3
        assert runtime.config.occupancy.sample_stride == 3

        assert runtime.update_occupancy_resolution(0)["sample_stride"] == 1  # clamped to minimum
        assert runtime.update_occupancy_resolution(999)["sample_stride"] == 32  # clamped to maximum

    def test_derives_voxel_size_from_the_new_stride(self, tmp_path) -> None:
        """The whole point of this control auto-adjusting voxel size:
        denser sampling (lower stride) should produce visibly finer
        voxels, not just re-hit the same coarse cells harder."""
        runtime, _ = _runtime(tmp_path)
        base_stride = runtime._occupancy_base_stride
        base_voxel_size = runtime._occupancy_base_voxel_size_meters

        denser = runtime.update_occupancy_resolution(max(1, base_stride // 2))
        assert denser["voxel_size_meters"] < base_voxel_size
        assert runtime.config.occupancy.voxel_size_meters == pytest.approx(denser["voxel_size_meters"])

        coarser = runtime.update_occupancy_resolution(base_stride * 2)
        assert coarser["voxel_size_meters"] > base_voxel_size

    def test_resets_grid_and_due_schedule_so_the_change_is_visible_promptly(self, tmp_path) -> None:
        """Existing voxel indices are keyed to the old voxel size, so
        they'd decode to the wrong world position under a new size --
        and the old per-camera due schedule (up to
        update_interval_seconds, several minutes) shouldn't leave the
        grid empty/stale in the meantime after a deliberate resolution
        change."""
        runtime, _ = _runtime(tmp_path)
        runtime._occupancy_grid.integrate_depth(
            np.full((8, 8), 2.0, dtype=np.float32), None, 60.0, sample_stride=1,
        )
        assert len(runtime._occupancy_grid) > 0
        runtime._occupancy_last_update["camera-video1"] = time.monotonic()

        old_grid = runtime._occupancy_grid
        runtime.update_occupancy_resolution(4)

        assert runtime._occupancy_grid is not old_grid
        assert len(runtime._occupancy_grid) == 0
        assert runtime._occupancy_last_update == {}

    def test_keeps_the_old_grid_as_a_fallback_for_the_dashboard(self, tmp_path) -> None:
        """The fresh grid a resolution change starts is empty until
        cameras re-integrate over the next several cycles -- the old
        (occupied) grid must be kept as a fallback rather than discarded,
        so occupancy_snapshot() has something real to serve in the
        meantime instead of blanking the dashboard's 3D view."""
        runtime, _ = _runtime(tmp_path)
        runtime._occupancy_grid.integrate_depth(
            np.full((8, 8), 2.0, dtype=np.float32), None, 60.0, sample_stride=1,
        )
        old_grid = runtime._occupancy_grid

        runtime.update_occupancy_resolution(4)

        assert runtime._occupancy_previous_grid is old_grid

    def test_does_not_keep_an_already_empty_grid_as_a_fallback(self, tmp_path) -> None:
        runtime, _ = _runtime(tmp_path)

        runtime.update_occupancy_resolution(4)

        assert runtime._occupancy_previous_grid is None


class TestOccupancySnapshotPreviousGridFallback:
    """occupancy_snapshot() must bridge a resolution change's empty new
    grid with the previous (occupied) one, rather than the dashboard
    seeing zero voxels until the new grid repopulates."""

    def test_serves_the_previous_grid_while_the_new_one_is_still_empty(self, tmp_path) -> None:
        runtime, _ = _runtime(tmp_path)
        runtime._occupancy_grid.integrate_depth(
            np.full((8, 8), 2.0, dtype=np.float32), None, 60.0, sample_stride=1,
        )
        expected_voxels = runtime._occupancy_grid.occupied_voxels()
        expected_voxel_size = runtime._occupancy_grid.voxel_size

        runtime.update_occupancy_resolution(4)
        snapshot = runtime.occupancy_snapshot()

        assert snapshot["voxels"] == expected_voxels
        assert snapshot["voxel_size_meters"] == pytest.approx(expected_voxel_size)
        assert snapshot["occupied_count"] == len(expected_voxels)

    def test_switches_over_once_the_new_grid_has_real_content(self, tmp_path) -> None:
        runtime, _ = _runtime(tmp_path)
        runtime._occupancy_grid.integrate_depth(
            np.full((8, 8), 2.0, dtype=np.float32), None, 60.0, sample_stride=1,
        )
        runtime.update_occupancy_resolution(4)
        assert runtime.occupancy_snapshot()["voxels"]  # still serving the old grid

        runtime._occupancy_grid.integrate_depth(
            np.full((8, 8), 3.0, dtype=np.float32), None, 60.0, sample_stride=1,
        )
        snapshot = runtime.occupancy_snapshot()

        assert snapshot["voxel_size_meters"] == pytest.approx(runtime._occupancy_grid.voxel_size)
        assert snapshot["voxels"] == runtime._occupancy_grid.occupied_voxels()
        assert runtime._occupancy_previous_grid is None  # dropped once real data exists

    def test_stops_falling_back_once_the_previous_grid_goes_stale(self, tmp_path) -> None:
        runtime, _ = _runtime(tmp_path)
        runtime.config.occupancy.stale_after_seconds = 60.0
        runtime._occupancy_grid.integrate_depth(
            np.full((8, 8), 2.0, dtype=np.float32), None, 60.0, sample_stride=1,
        )
        runtime.update_occupancy_resolution(4)
        assert runtime.occupancy_snapshot()["voxels"]  # fresh fallback still served

        runtime._occupancy_previous_grid_since = time.monotonic() - 61.0
        snapshot = runtime.occupancy_snapshot()

        assert snapshot["voxels"] == []
        assert runtime._occupancy_previous_grid is None

    def test_no_fallback_when_there_was_never_a_previous_grid(self, tmp_path) -> None:
        runtime, _ = _runtime(tmp_path)

        snapshot = runtime.occupancy_snapshot()

        assert snapshot["voxels"] == []
        assert snapshot["voxel_size_meters"] == pytest.approx(runtime.config.occupancy.voxel_size_meters)
