"""Tests for the per-camera sparse voxel occupancy grid.

Pure math/data-structure tests -- no depth model involved. The depth
model itself (DA3METRIC-LARGE via a subprocess) is exercised separately
and manually against real hardware, not unit-tested here, since it needs
a ~4GB model load and a sibling project's venv.
"""

from __future__ import annotations

import numpy as np
import pytest

from egg_companion.core.occupancy import VoxelGrid


def _flat_depth(height: int, width: int, value: float) -> np.ndarray:
    return np.full((height, width), value, dtype=np.float32)


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
        assert grid._voxels[center_index].occupied is True

    def test_ray_to_surface_marks_intermediate_voxels_free(self) -> None:
        grid = VoxelGrid(voxel_size_meters=0.5, max_range_meters=10.0, max_voxels=10_000)
        depth = _flat_depth(16, 16, value=4.0)

        grid.integrate_depth(
            depth, confidence=None, horizontal_fov_degrees=90.0,
            sample_stride=4, ray_steps=16,
        )

        # A voxel roughly halfway to the surface along the center ray
        # should have been carved as free space, not left unknown.
        near_index = (0, 0, 2)
        assert near_index in grid._voxels
        assert grid._voxels[near_index].occupied is False

    def test_occupied_voxel_is_never_downgraded_to_free(self) -> None:
        grid = VoxelGrid(voxel_size_meters=1.0, max_range_meters=10.0, max_voxels=10_000)
        # First pass: something at z=1 (close), marking that voxel occupied.
        grid.integrate_depth(
            _flat_depth(8, 8, value=1.0), confidence=None,
            horizontal_fov_degrees=90.0, sample_stride=2, ray_steps=4,
        )
        occupied_index = (0, 0, 1)
        assert grid._voxels[occupied_index].occupied is True

        # Second pass: something farther away (z=5) whose ray passes
        # through the same near voxel -- must not erase the occupied mark.
        grid.integrate_depth(
            _flat_depth(8, 8, value=5.0), confidence=None,
            horizontal_fov_degrees=90.0, sample_stride=2, ray_steps=20,
        )
        assert grid._voxels[occupied_index].occupied is True

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

    def test_non_2d_depth_array_is_rejected(self) -> None:
        grid = VoxelGrid(voxel_size_meters=0.5, max_range_meters=10.0, max_voxels=10_000)
        depth_3d = np.zeros((4, 4, 3), dtype=np.float32)
        assert grid.integrate_depth(depth_3d, None, 90.0) == 0


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
        assert all(v["confidence"] > 0 for v in voxels)
        # occupied_count() is the authoritative count of occupied cells;
        # occupied_voxels() must return exactly that many, not more (i.e.
        # not accidentally including the free/ray-carved cells too).
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

    def test_confidence_is_carried_through(self) -> None:
        grid = VoxelGrid(voxel_size_meters=1.0, max_range_meters=10.0, max_voxels=10_000)
        depth = _flat_depth(4, 4, value=2.0)
        conf = np.full((4, 4), 0.73, dtype=np.float32)
        grid.integrate_depth(
            depth, confidence=conf, horizontal_fov_degrees=90.0,
            sample_stride=1, ray_steps=2,
        )
        voxels = grid.occupied_voxels()
        assert voxels
        assert all(v["confidence"] == pytest.approx(0.73) for v in voxels)


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
