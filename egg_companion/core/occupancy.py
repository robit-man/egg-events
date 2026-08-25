"""Sparse voxel occupancy grid fused from a known multi-camera array.

The four cameras are a co-located panoramic rig (not independent, unrelated
viewpoints): video0 rightmost, sweeping counter-clockwise through video1,
video2, video3, each adjacent pair separated by a known yaw (see
config.OccupancyConfig.camera_yaw_degrees). There's still no calibrated
per-camera intrinsics (focal length/principal point are an assumed-FOV
estimate -- see DepthEstimator/assumed_hfov_degrees), but the *extrinsic*
geometry between cameras is known, so this fuses every camera's depth into
one shared "egg frame" grid via a per-camera yaw rotation (no translation:
the rig is modeled as sharing one optical center) rather than keeping each
camera's reconstruction in its own disconnected local frame.

Coordinate convention for the shared (fused) frame: +X right, +Y up,
+Z forward (the direction camera-video's local yaw=0 boresight faces),
in meters, right-handed, matching standard 3D-graphics/robotics "Y up"
convention. Camera-local back-projection itself still uses the pinhole
image convention (+Y down) before being rotated into the shared frame --
see integrate_depth's yaw_degrees parameter.
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
    """Sparse, bounded occupancy grid in the shared (fused) egg frame."""

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
        yaw_degrees: float = 0.0,
    ) -> int:
        """Back-project a depth map into this grid using an assumed pinhole
        model (no calibrated intrinsics exist for these cameras), rotate
        each point into the shared fused frame by this camera's known
        mounting yaw, and carve free space along the ray from the shared
        origin to each observed surface.

        yaw_degrees is this camera's angle in the panoramic array (see
        module docstring / config.OccupancyConfig.camera_yaw_degrees), not
        a per-call override of a calibrated pose -- pass 0.0 to integrate
        directly in camera-local coordinates without fusing.

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
        yaw = math.radians(yaw_degrees)
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)

        integrated = 0
        for row in range(0, height, sample_stride):
            for col in range(0, width, sample_stride):
                z_cam = float(depth[row, col])
                if z_cam <= 0.0 or z_cam > self.max_range or not math.isfinite(z_cam):
                    continue
                if confidence is not None:
                    pixel_conf = float(confidence[row, col])
                    if pixel_conf < min_confidence:
                        continue
                else:
                    pixel_conf = 1.0

                # Camera-local pinhole back-projection (+X right, +Y down,
                # +Z forward), then flip to shared-frame +Y up, then rotate
                # about the (now vertical) Y axis by this camera's mount
                # yaw. Rotation is linear about the shared origin, so the
                # ray-marched intermediate points below can just scale this
                # already-rotated point by t rather than re-rotating each step.
                x_cam = (col - cx) * z_cam / fx
                y_cam = -(row - cy) * z_cam / fy
                x = x_cam * cos_yaw + z_cam * sin_yaw
                y = y_cam
                z = -x_cam * sin_yaw + z_cam * cos_yaw
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

    def occupied_voxels(self) -> list[dict[str, object]]:
        """Occupied voxel centers in shared-frame meters, for the
        dashboard's 3D scene -- not the fastest representation for a large
        grid, but max_voxels already bounds how big this can get."""
        return [
            {
                "x": round((ix + 0.5) * self.voxel_size, 3),
                "y": round((iy + 0.5) * self.voxel_size, 3),
                "z": round((iz + 0.5) * self.voxel_size, 3),
                "confidence": round(record.confidence, 3),
            }
            for (ix, iy, iz), record in self._voxels.items()
            if record.occupied
        ]

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
