"""Tests for the fused sparse voxel occupancy grid.

Pure math/data-structure tests -- no depth model involved. The depth
model itself (DA3METRIC-LARGE via a subprocess) is exercised separately
and manually against real hardware, not unit-tested here, since it needs
a ~4GB model load and a sibling project's venv.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from egg_companion.core.occupancy import (
    FREE_LOG_ODDS_THRESHOLD,
    LOG_ODDS_HIT,
    LOG_ODDS_MAX,
    LOG_ODDS_MIN,
    LOG_ODDS_MISS,
    OCCUPIED_LOG_ODDS_THRESHOLD,
    VoxelGrid,
    log_odds_to_probability,
    resolve_camera_yaw_degrees,
    resolve_voxel_size_meters,
)


def _flat_depth(height: int, width: int, value: float) -> np.ndarray:
    return np.full((height, width), value, dtype=np.float32)


def _center_pixel_depth(height: int, width: int, value: float) -> np.ndarray:
    """A depth map valid at exactly the center pixel, everything else
    zero/invalid -- a literal single ray straight down the camera's own
    boresight, unambiguous regardless of the camera-local left/right
    sign convention (the center column always projects to x=0)."""
    depth = np.zeros((height, width), dtype=np.float32)
    depth[height // 2, width // 2] = value
    return depth


class TestIntegrateDepth:
    def test_flat_surface_marks_occupied_voxel_at_expected_depth(self) -> None:
        grid = VoxelGrid(voxel_size_meters=0.5, max_range_meters=10.0, max_voxels=10_000)
        depth = _flat_depth(32, 32, value=2.0)

        integrated = grid.integrate_depth(
            depth, confidence=None, horizontal_fov_degrees=90.0,
            sample_stride=4, ray_steps=8,
        )

        assert integrated > 0
        assert grid.occupied_count() > 0
        # The center pixel projects to (x=0, y=0, z=2.0) -- voxel index
        # along Z should correspond to floor(2.0 / 0.5) = 4.
        center_index = (0, 0, 4)
        assert grid.is_occupied(center_index) is True

    def test_ray_to_surface_marks_intermediate_voxels_not_occupied(self) -> None:
        grid = VoxelGrid(voxel_size_meters=0.5, max_range_meters=10.0, max_voxels=10_000)
        depth = _flat_depth(16, 16, value=4.0)

        grid.integrate_depth(
            depth, confidence=None, horizontal_fov_degrees=90.0,
            sample_stride=4, ray_steps=16,
        )

        # A voxel roughly halfway to the surface along the center ray
        # should have been carved as not-occupied (a ray miss), not left
        # unobserved / accidentally marked occupied.
        near_index = (0, 0, 2)
        assert near_index in grid._voxels
        assert grid.is_occupied(near_index) is False

    def test_single_contrary_ray_does_not_immediately_clear_strong_occupied_evidence(
        self,
    ) -> None:
        """Bayesian log-odds fusion (not a one-way ratchet): a single
        contrary ray miss barely moves a voxel's estimate, so a
        well-observed surface survives one noisy contrary observation --
        while still being able to flip given enough accumulated evidence
        (see the eventually-clears test below)."""
        grid = VoxelGrid(voxel_size_meters=1.0, max_range_meters=10.0, max_voxels=10_000)
        # First pass: a single ray at z=1 (close), marking that voxel occupied.
        grid.integrate_depth(
            _center_pixel_depth(8, 8, value=1.0), confidence=None,
            horizontal_fov_degrees=90.0, sample_stride=1, ray_steps=4,
        )
        occupied_index = (0, 0, 1)
        assert grid.is_occupied(occupied_index) is True

        # Second pass: the same single ray now reads farther away (z=5)
        # and passes through the same near voxel on its way there -- one
        # contrary observation must not erase strong occupied evidence.
        grid.integrate_depth(
            _center_pixel_depth(8, 8, value=5.0), confidence=None,
            horizontal_fov_degrees=90.0, sample_stride=1, ray_steps=20,
        )
        assert grid.is_occupied(occupied_index) is True

    def test_sustained_contrary_evidence_eventually_clears_occupied_voxel(self) -> None:
        """Unlike a one-way ratchet, enough accumulated free-space
        evidence through a previously-occupied voxel does eventually
        flip its classification -- e.g. an object that moves away."""
        grid = VoxelGrid(voxel_size_meters=1.0, max_range_meters=10.0, max_voxels=10_000)
        grid.integrate_depth(
            _center_pixel_depth(8, 8, value=1.0), confidence=None,
            horizontal_fov_degrees=90.0, sample_stride=1, ray_steps=4,
        )
        occupied_index = (0, 0, 1)
        assert grid.is_occupied(occupied_index) is True

        for _ in range(10):
            grid.integrate_depth(
                _center_pixel_depth(8, 8, value=5.0), confidence=None,
                horizontal_fov_degrees=90.0, sample_stride=1, ray_steps=20,
            )
        assert grid.is_occupied(occupied_index) is False

    def test_log_odds_clamped_to_bounds(self) -> None:
        grid = VoxelGrid(voxel_size_meters=1.0, max_range_meters=10.0, max_voxels=10_000)
        for _ in range(50):
            grid.integrate_depth(
                _flat_depth(4, 4, value=1.0), confidence=None,
                horizontal_fov_degrees=90.0, sample_stride=2, ray_steps=2,
            )
        record = grid._voxels[(0, 0, 1)]
        assert LOG_ODDS_MIN <= record.log_odds <= LOG_ODDS_MAX

    def test_depth_beyond_max_range_is_ignored(self) -> None:
        grid = VoxelGrid(voxel_size_meters=0.5, max_range_meters=3.0, max_voxels=10_000)
        integrated = grid.integrate_depth(
            _flat_depth(8, 8, value=100.0), confidence=None,
            horizontal_fov_degrees=90.0, sample_stride=2, ray_steps=4,
        )
        assert integrated == 0
        assert len(grid) == 0

    def test_zero_or_negative_depth_is_ignored(self) -> None:
        grid = VoxelGrid(voxel_size_meters=0.5, max_range_meters=10.0, max_voxels=10_000)
        depth = _flat_depth(8, 8, value=0.0)
        depth[0, 0] = -1.0
        integrated = grid.integrate_depth(
            depth, confidence=None, horizontal_fov_degrees=90.0,
            sample_stride=1, ray_steps=4,
        )
        assert integrated == 0

    def test_confidence_below_threshold_is_skipped(self) -> None:
        grid = VoxelGrid(voxel_size_meters=0.5, max_range_meters=10.0, max_voxels=10_000)
        depth = _flat_depth(8, 8, value=2.0)
        low_conf = np.full((8, 8), 0.1, dtype=np.float32)

        integrated = grid.integrate_depth(
            depth, confidence=low_conf, horizontal_fov_degrees=90.0,
            min_confidence=0.5, sample_stride=2, ray_steps=4,
        )
        assert integrated == 0

    def test_confidence_above_threshold_is_kept(self) -> None:
        grid = VoxelGrid(voxel_size_meters=0.5, max_range_meters=10.0, max_voxels=10_000)
        depth = _flat_depth(8, 8, value=2.0)
        high_conf = np.full((8, 8), 0.9, dtype=np.float32)

        integrated = grid.integrate_depth(
            depth, confidence=high_conf, horizontal_fov_degrees=90.0,
            min_confidence=0.5, sample_stride=2, ray_steps=4,
        )
        assert integrated > 0

    def test_low_confidence_hit_contributes_weaker_evidence_than_high_confidence(
        self,
    ) -> None:
        low = VoxelGrid(voxel_size_meters=0.5, max_range_meters=10.0, max_voxels=10_000)
        high = VoxelGrid(voxel_size_meters=0.5, max_range_meters=10.0, max_voxels=10_000)
        depth = _flat_depth(4, 4, value=2.0)

        low.integrate_depth(
            depth, confidence=np.full((4, 4), 0.35, dtype=np.float32),
            horizontal_fov_degrees=90.0, min_confidence=0.3, sample_stride=1, ray_steps=2,
        )
        high.integrate_depth(
            depth, confidence=np.full((4, 4), 1.0, dtype=np.float32),
            horizontal_fov_degrees=90.0, min_confidence=0.3, sample_stride=1, ray_steps=2,
        )

        low_log_odds = low._voxels[(0, 0, 4)].log_odds
        high_log_odds = high._voxels[(0, 0, 4)].log_odds
        assert low_log_odds < high_log_odds

    def test_non_2d_depth_array_is_rejected(self) -> None:
        grid = VoxelGrid(voxel_size_meters=0.5, max_range_meters=10.0, max_voxels=10_000)
        depth_3d = np.zeros((4, 4, 3), dtype=np.float32)
        assert grid.integrate_depth(depth_3d, None, 90.0) == 0


class TestIntegrateDepthLeftRightOrientation:
    """Regression test for a real mirroring bug found on hardware: each
    camera's own reconstruction was flipped left-right internally (a
    surface visible on the left side of the actual captured frame was
    back-projected to the right of that camera's own boresight, and vice
    versa) even though the cross-camera yaw arrangement -- and therefore
    the array's overall left-to-right order -- was already correct.
    x_cam must be derived as (cx - col), not (col - cx)."""

    def test_a_surface_only_in_the_right_half_of_the_frame_projects_to_negative_x(
        self,
    ) -> None:
        # ray_steps=1 (no ray-marched free-space voxels) and a non-round
        # depth value keep this to only the actual surface (hit) points,
        # comfortably clear of voxel-boundary floating-point edge cases --
        # ray-marched points approaching the shared origin legitimately
        # pass near x=0 for any column, which isn't what's being tested
        # here (the surface reconstruction's own left/right sense).
        grid = VoxelGrid(voxel_size_meters=0.5, max_range_meters=10.0, max_voxels=10_000)
        depth = np.zeros((8, 8), dtype=np.float32)
        depth[:, 5:] = 1.7  # surface only in the frame's right-hand columns
        grid.integrate_depth(
            depth, confidence=None, horizontal_fov_degrees=90.0,
            sample_stride=1, ray_steps=1,
        )
        assert grid._voxels
        assert all(index[0] < 0 for index in grid._voxels)

    def test_a_surface_only_in_the_left_half_of_the_frame_projects_to_positive_x(
        self,
    ) -> None:
        grid = VoxelGrid(voxel_size_meters=0.5, max_range_meters=10.0, max_voxels=10_000)
        depth = np.zeros((8, 8), dtype=np.float32)
        depth[:, :3] = 1.7  # surface only in the frame's left-hand columns
        grid.integrate_depth(
            depth, confidence=None, horizontal_fov_degrees=90.0,
            sample_stride=1, ray_steps=1,
        )
        assert grid._voxels
        assert all(index[0] > 0 for index in grid._voxels)


class TestLogOddsHelpers:
    def test_log_odds_to_probability_matches_sigmoid(self) -> None:
        assert log_odds_to_probability(0.0) == pytest.approx(0.5)
        assert log_odds_to_probability(LOG_ODDS_HIT) == pytest.approx(0.9, abs=1e-6)
        assert log_odds_to_probability(LOG_ODDS_MISS) == pytest.approx(0.35, abs=1e-6)

    def test_thresholds_bracket_uncertain_region(self) -> None:
        assert FREE_LOG_ODDS_THRESHOLD < 0.0 < OCCUPIED_LOG_ODDS_THRESHOLD


class TestCapacityAndStaleness:
    def test_eviction_bounds_total_voxel_count(self) -> None:
        grid = VoxelGrid(voxel_size_meters=0.05, max_range_meters=10.0, max_voxels=5)
        depth = _flat_depth(64, 64, value=2.0)

        grid.integrate_depth(
            depth, confidence=None, horizontal_fov_degrees=90.0,
            sample_stride=2, ray_steps=2,
        )
        assert len(grid) <= 5

    def test_prune_stale_removes_old_voxels_only(self) -> None:
        grid = VoxelGrid(voxel_size_meters=0.5, max_range_meters=10.0, max_voxels=10_000)
        grid.integrate_depth(
            _flat_depth(8, 8, value=2.0), confidence=None,
            horizontal_fov_degrees=90.0, sample_stride=2, ray_steps=4, now=0.0,
        )
        assert len(grid) > 0

        removed = grid.prune_stale(stale_after_seconds=100.0, now=50.0)
        assert removed == 0
        assert len(grid) > 0

        removed = grid.prune_stale(stale_after_seconds=100.0, now=500.0)
        assert removed > 0
        assert len(grid) == 0


class TestOccupiedVoxels:
    """occupied_voxels() -- the raw per-voxel export used by the
    dashboard's 3D scene, as opposed to summarize()'s coarse text summary."""

    def test_empty_grid_returns_empty_list(self) -> None:
        grid = VoxelGrid(voxel_size_meters=0.5, max_range_meters=10.0, max_voxels=10_000)
        assert grid.occupied_voxels() == []

    def test_free_voxels_are_excluded(self) -> None:
        grid = VoxelGrid(voxel_size_meters=0.5, max_range_meters=10.0, max_voxels=10_000)
        grid.integrate_depth(
            _flat_depth(16, 16, value=4.0), confidence=None,
            horizontal_fov_degrees=90.0, sample_stride=4, ray_steps=16,
        )
        voxels = grid.occupied_voxels()
        assert voxels
        assert all(v["confidence"] > 0.5 for v in voxels)
        # occupied_count() is the authoritative count of occupied cells;
        # occupied_voxels() must return exactly that many, not more (i.e.
        # not accidentally including the free/uncertain cells too).
        assert len(voxels) == grid.occupied_count()

    def test_voxel_center_matches_grid_index_times_voxel_size(self) -> None:
        grid = VoxelGrid(voxel_size_meters=0.5, max_range_meters=10.0, max_voxels=10_000)
        grid.integrate_depth(
            _flat_depth(4, 4, value=2.0), confidence=None,
            horizontal_fov_degrees=90.0, sample_stride=1, ray_steps=2,
        )
        voxels = grid.occupied_voxels()
        assert voxels
        for v in voxels:
            # Every returned coordinate should land on a half-voxel-offset
            # center: (index + 0.5) * voxel_size.
            for axis in ("x", "y", "z"):
                remainder = (v[axis] / 0.5) % 1
                assert remainder == pytest.approx(0.5, abs=1e-6) or remainder == pytest.approx(-0.5, abs=1e-6)

    def test_confidence_is_posterior_probability_in_zero_to_one_range(self) -> None:
        grid = VoxelGrid(voxel_size_meters=1.0, max_range_meters=10.0, max_voxels=10_000)
        depth = _flat_depth(4, 4, value=2.0)
        conf = np.full((4, 4), 0.9, dtype=np.float32)
        grid.integrate_depth(
            depth, confidence=conf, horizontal_fov_degrees=90.0,
            sample_stride=1, ray_steps=2,
        )
        voxels = grid.occupied_voxels()
        assert voxels
        assert all(0.0 <= v["confidence"] <= 1.0 for v in voxels)
        assert all(v["confidence"] > 0.7 for v in voxels)

    def test_voxel_color_is_sampled_from_source_frame_at_hit_pixel(self) -> None:
        grid = VoxelGrid(voxel_size_meters=1.0, max_range_meters=10.0, max_voxels=10_000)
        depth = _flat_depth(4, 4, value=2.0)
        # BGR frame, solid red in OpenCV's native channel order.
        color_frame = np.zeros((4, 4, 3), dtype=np.uint8)
        color_frame[:, :] = (0, 0, 255)  # B, G, R -- pure red
        grid.integrate_depth(
            depth, confidence=None, horizontal_fov_degrees=90.0,
            sample_stride=1, ray_steps=2, color_frame=color_frame,
        )
        voxels = grid.occupied_voxels()
        assert voxels
        assert all(v["color"] == [255, 0, 0] for v in voxels)

    def test_no_color_frame_falls_back_to_default_color(self) -> None:
        grid = VoxelGrid(voxel_size_meters=0.5, max_range_meters=10.0, max_voxels=10_000)
        grid.integrate_depth(
            _flat_depth(4, 4, value=2.0), confidence=None,
            horizontal_fov_degrees=90.0, sample_stride=1, ray_steps=2,
        )
        voxels = grid.occupied_voxels()
        assert voxels
        assert all(v["color"] == [0x66, 0x7e, 0xa8] for v in voxels)

    def test_ray_miss_does_not_overwrite_an_existing_hit_colored_voxel(self) -> None:
        grid = VoxelGrid(voxel_size_meters=1.0, max_range_meters=10.0, max_voxels=10_000)
        green_frame = np.zeros((8, 8, 3), dtype=np.uint8)
        green_frame[:, :] = (0, 255, 0)  # B, G, R -- pure green
        grid.integrate_depth(
            _flat_depth(8, 8, value=1.0), confidence=None,
            horizontal_fov_degrees=90.0, sample_stride=2, ray_steps=4,
            color_frame=green_frame,
        )
        occupied_index = (0, 0, 1)
        assert grid._voxels[occupied_index].color == (0, 255, 0)

        # A farther hit whose ray passes through the same near voxel
        # (a "miss" observation there, color_frame=None) must not erase
        # its previously observed color.
        grid.integrate_depth(
            _flat_depth(8, 8, value=5.0), confidence=None,
            horizontal_fov_degrees=90.0, sample_stride=2, ray_steps=20,
        )
        assert grid._voxels[occupied_index].color == (0, 255, 0)


class TestSummarize:
    def test_empty_grid_summary(self) -> None:
        grid = VoxelGrid(voxel_size_meters=0.5, max_range_meters=10.0, max_voxels=10_000)
        summary = grid.summarize()
        assert summary["occupied_voxels"] == 0
        assert summary["nearest_occupied_meters"] is None
        assert grid.summary_text() == "no occupied space detected"

    def test_summary_reports_nearest_distance_and_sectors(self) -> None:
        grid = VoxelGrid(voxel_size_meters=0.5, max_range_meters=10.0, max_voxels=10_000)
        grid.integrate_depth(
            _flat_depth(32, 32, value=2.0), confidence=None,
            horizontal_fov_degrees=90.0, sample_stride=4, ray_steps=4,
        )
        summary = grid.summarize()
        assert summary["occupied_voxels"] > 0
        assert summary["nearest_occupied_meters"] is not None
        assert summary["nearest_occupied_meters"] <= 2.5
        assert "center" in summary["sectors"]
        assert "nearest occupied surface" in grid.summary_text()


class TestConcurrentReadDuringIntegration:
    """The dashboard's /api/occupancy handler runs on the asyncio event
    loop thread and calls occupied_count()/free_count()/occupied_voxels()/
    summarize() while integrate_depth() runs concurrently in a background
    thread (runtime.py wraps it in asyncio.to_thread) -- observed in
    production: 'RuntimeError: dictionary changed size during iteration'
    when a snapshot read raced a live mutation of self._voxels."""

    def test_reads_do_not_raise_while_writer_thread_mutates_concurrently(
        self,
    ) -> None:
        grid = VoxelGrid(voxel_size_meters=0.05, max_range_meters=10.0, max_voxels=2000)
        stop = threading.Event()
        errors: list[BaseException] = []

        def writer() -> None:
            depth = _flat_depth(64, 64, value=2.0)
            while not stop.is_set():
                try:
                    grid.integrate_depth(
                        depth, confidence=None, horizontal_fov_degrees=90.0,
                        sample_stride=1, ray_steps=6,
                    )
                except BaseException as error:  # noqa: BLE001
                    errors.append(error)
                    return

        writer_thread = threading.Thread(target=writer, daemon=True)
        writer_thread.start()
        try:
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                try:
                    grid.occupied_count()
                    grid.free_count()
                    grid.occupied_voxels()
                    grid.summarize()
                except BaseException as error:  # noqa: BLE001
                    errors.append(error)
                    break
        finally:
            stop.set()
            writer_thread.join(timeout=5)

        assert errors == []


class TestResolveCameraYawDegrees:
    """The physically-mounted 4-camera rig must never need a hardcoded
    per-camera-count mapping -- yaw is auto-computed from each camera's
    parsed trailing index among whatever cameras are currently known,
    evenly spaced and centered on the array midpoint, so a camera added
    later (video4, video5, ...) is automatically placed correctly."""

    def test_four_camera_default_reproduces_the_physically_mounted_spacing(self) -> None:
        known = ["camera-video0", "camera-video1", "camera-video2", "camera-video3"]
        assert resolve_camera_yaw_degrees("camera-video0", known, 60.0) == pytest.approx(-90.0)
        assert resolve_camera_yaw_degrees("camera-video1", known, 60.0) == pytest.approx(-30.0)
        assert resolve_camera_yaw_degrees("camera-video2", known, 60.0) == pytest.approx(30.0)
        assert resolve_camera_yaw_degrees("camera-video3", known, 60.0) == pytest.approx(90.0)

    def test_a_fifth_camera_is_automatically_placed_and_recenters_the_array(self) -> None:
        known = ["camera-video0", "camera-video1", "camera-video2", "camera-video3", "camera-video4"]
        # N=5, spacing=60: centered on index 2 -> -120,-60,0,60,120.
        assert resolve_camera_yaw_degrees("camera-video0", known, 60.0) == pytest.approx(-120.0)
        assert resolve_camera_yaw_degrees("camera-video2", known, 60.0) == pytest.approx(0.0)
        assert resolve_camera_yaw_degrees("camera-video4", known, 60.0) == pytest.approx(120.0)

    def test_unknown_camera_id_defaults_to_zero(self) -> None:
        assert resolve_camera_yaw_degrees("camera-video9", ["camera-video0"], 60.0) == 0.0

    def test_explicit_override_wins_over_auto_computation(self) -> None:
        known = ["camera-video0", "camera-video1"]
        overrides = {"camera-video0": -45.0}
        assert resolve_camera_yaw_degrees("camera-video0", known, 60.0, overrides) == -45.0
        # The un-overridden camera still gets the auto-computed value.
        assert resolve_camera_yaw_degrees("camera-video1", known, 60.0, overrides) == pytest.approx(30.0)

    def test_single_camera_is_centered_at_zero(self) -> None:
        assert resolve_camera_yaw_degrees("camera-video0", ["camera-video0"], 60.0) == 0.0


class TestResolveVoxelSizeMeters:
    """Denser depth sampling (a lower sample_stride) should resolve into
    proportionally finer voxels, and coarser sampling into proportionally
    coarser ones, rather than a fixed voxel size mismatched to whatever
    the dashboard's Resolution +/- control currently has sample_stride
    set to."""

    def test_matches_base_at_the_base_stride(self) -> None:
        assert resolve_voxel_size_meters(8, 8, 0.1) == pytest.approx(0.1)

    def test_denser_stride_yields_a_smaller_voxel(self) -> None:
        assert resolve_voxel_size_meters(4, 8, 0.1) == pytest.approx(0.05)

    def test_coarser_stride_yields_a_larger_voxel(self) -> None:
        assert resolve_voxel_size_meters(16, 8, 0.1) == pytest.approx(0.2)

    def test_extreme_density_is_clamped_to_the_floor(self) -> None:
        assert resolve_voxel_size_meters(1, 8, 0.1, min_voxel_size_meters=0.02) == pytest.approx(0.02)

    def test_extreme_coarseness_is_clamped_to_the_ceiling(self) -> None:
        assert resolve_voxel_size_meters(64, 8, 0.1, max_voxel_size_meters=0.6) == pytest.approx(0.6)

    def test_zero_base_stride_falls_back_to_base_size(self) -> None:
        assert resolve_voxel_size_meters(8, 0, 0.1) == 0.1
