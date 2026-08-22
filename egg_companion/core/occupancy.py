"""Per-camera sparse voxel occupancy grid built from monocular depth.

None of this hardware's cameras have calibrated intrinsics, and there is no
known extrinsic transform between them (no shared coordinate frame), so
this deliberately builds one independent, camera-local grid per camera
rather than attempting to fuse them into a single building-wide 3D map.
Each grid answers "what does this camera's own view look like as occupied
space" -- coarse enough for spatial reasoning ("is there room near X"), not
precise enough for anything requiring true metric accuracy or navigation.

Coordinate convention per grid: standard pinhole camera-local axes,
+X right, +Y down, +Z forward (away from the camera), in meters.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class VoxelRecord:
    occupied: bool
    confidence: float
    last_seen: float  # time.monotonic()


class VoxelGrid:
    """Sparse, bounded, camera-local occupancy grid."""

    def __init__(
        self,
        voxel_size_meters: float,
        max_range_meters: float,
        max_voxels: int,
    ) -> None:
        self.voxel_size = voxel_size_meters
        self.max_range = max_range_meters
        self.max_voxels = max_voxels
        self._voxels: dict[tuple[int, int, int], VoxelRecord] = {}

    def __len__(self) -> int:
        return len(self._voxels)

    def _to_index(self, point: tuple[float, float, float]) -> tuple[int, int, int]:
        return (
            int(math.floor(point[0] / self.voxel_size)),
            int(math.floor(point[1] / self.voxel_size)),
            int(math.floor(point[2] / self.voxel_size)),
        )

    def _mark(
        self, index: tuple[int, int, int], occupied: bool, confidence: float, now: float
    ) -> None:
        existing = self._voxels.get(index)
        if existing is not None and existing.occupied and not occupied:
            # A directly-observed surface must never be downgraded to
            # "free" just because some other sample's ray happened to
            # pass through the same cell -- occupied evidence outranks
            # free-space inference through the same voxel.
            return
        self._voxels[index] = VoxelRecord(occupied=occupied, confidence=confidence, last_seen=now)

    def integrate_depth(
        self,
        depth: np.ndarray,
        confidence: np.ndarray | None,
        horizontal_fov_degrees: float,
        min_confidence: float = 0.0,
        sample_stride: int = 8,
        ray_steps: int = 12,
        now: float | None = None,
    ) -> int:
        """Back-project a depth map into this grid using an assumed pinhole
        model (no calibrated intrinsics exist for these cameras) and carve
        free space along the ray from the camera to each observed surface.

        Returns the number of pixels integrated.
        """
        now = now if now is not None else time.monotonic()
        if depth.ndim != 2:
            return 0
        height, width = depth.shape
        if height == 0 or width == 0:
            return 0
        fx = (width / 2.0) / math.tan(math.radians(horizontal_fov_degrees) / 2.0)
        fy = fx  # assumed square pixels -- no calibration exists to say otherwise
        cx, cy = width / 2.0, height / 2.0

        integrated = 0
        for row in range(0, height, sample_stride):
            for col in range(0, width, sample_stride):
                z = float(depth[row, col])
                if z <= 0.0 or z > self.max_range or not math.isfinite(z):
                    continue
                if confidence is not None:
                    pixel_conf = float(confidence[row, col])
                    if pixel_conf < min_confidence:
                        continue
                else:
                    pixel_conf = 1.0

                x = (col - cx) * z / fx
                y = (row - cy) * z / fy
                surface_index = self._to_index((x, y, z))

                for step in range(1, ray_steps):
                    t = step / ray_steps
                    ray_index = self._to_index((x * t, y * t, z * t))
                    if ray_index == surface_index:
                        continue
                    self._mark(ray_index, occupied=False, confidence=0.5, now=now)

                self._mark(surface_index, occupied=True, confidence=pixel_conf, now=now)
                integrated += 1

        self._evict_if_over_capacity()
        return integrated

    def _evict_if_over_capacity(self) -> None:
        if len(self._voxels) <= self.max_voxels:
            return
        ordered = sorted(self._voxels.items(), key=lambda item: item[1].last_seen)
        excess = len(self._voxels) - self.max_voxels
        for key, _ in ordered[:excess]:
            del self._voxels[key]

    def prune_stale(self, stale_after_seconds: float, now: float | None = None) -> int:
        now = now if now is not None else time.monotonic()
        stale_keys = [
            key for key, record in self._voxels.items()
            if now - record.last_seen > stale_after_seconds
        ]
        for key in stale_keys:
            del self._voxels[key]
        return len(stale_keys)

    def occupied_count(self) -> int:
        return sum(1 for record in self._voxels.values() if record.occupied)

    def free_count(self) -> int:
        return sum(1 for record in self._voxels.values() if not record.occupied)

    def summarize(self) -> dict[str, object]:
        """Compact, LLM-context-friendly summary of this camera's local
        occupancy -- coarse enough to describe in a sentence, not a raw
        voxel dump. Lateral sectors and distances use the horizontal
        (X, Z) plane only; height (Y) is not summarized separately."""
        occupied = [
            (key, math.hypot(key[0] * self.voxel_size, key[2] * self.voxel_size))
            for key, record in self._voxels.items() if record.occupied
        ]
        if not occupied:
            return {
                "occupied_voxels": 0,
                "free_voxels": self.free_count(),
                "nearest_occupied_meters": None,
                "sectors": {},
            }
        sectors: dict[str, list[float]] = {}
        for (ix, _iy, _iz), distance in occupied:
            x = ix * self.voxel_size
            if x < -self.voxel_size:
                lateral = "left"
            elif x > self.voxel_size:
                lateral = "right"
            else:
                lateral = "center"
            sectors.setdefault(lateral, []).append(distance)
        sector_summary = {
            lateral: {"nearest_meters": round(min(distances), 2), "count": len(distances)}
            for lateral, distances in sectors.items()
        }
        return {
            "occupied_voxels": len(occupied),
            "free_voxels": self.free_count(),
            "nearest_occupied_meters": round(min(d for _, d in occupied), 2),
            "sectors": sector_summary,
        }

    def summary_text(self) -> str:
        """One-sentence natural-language rendering of summarize(), meant
        to be dropped directly into LLM-facing world-model context."""
        summary = self.summarize()
        if not summary["occupied_voxels"]:
            return "no occupied space detected"
        parts = [
            f"nearest occupied surface ~{summary['nearest_occupied_meters']}m away"
        ]
        for lateral in ("left", "center", "right"):
            sector = summary["sectors"].get(lateral)
            if sector:
                parts.append(f"{lateral} ~{sector['nearest_meters']}m")
        return "; ".join(parts)
