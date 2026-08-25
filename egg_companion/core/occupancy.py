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

Per-voxel state is a log-odds occupancy estimate, not a binary flag --
the standard Bayesian occupancy-grid-mapping formulation (Moravec &
Elfes 1985; textbook treatment in Thrun/Burgard/Fox, "Probabilistic
Robotics" ch.9) used throughout robotics mapping (OctoMap, ROS
costmap_2d) and by camera/lidar occupancy stacks generally, including
pre-transformer-era automotive occupancy grids. Each observation nudges
a voxel's log-odds up (ray endpoint: a surface hit) or down (ray
interior: free-space traversal); log-odds accumulate additively across
observations and are clamped to bounded confidence, so noisy single
observations barely move the estimate while corroborated evidence
dominates -- and, critically, sustained contrary evidence CAN flip a
voxel's classification (a chair that gets moved eventually reads as
free again), unlike a one-way "occupied, once marked, forever occupied"
ratchet. Converting log-odds back to probability uses the standard
sigmoid p = 1 / (1 + exp(-log_odds)).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

# Bounds keep any single voxel's log-odds from saturating to certainty
# after many observations, so a real change in the environment can still
# flip its classification within a bounded number of future updates.
LOG_ODDS_MIN = -6.0
LOG_ODDS_MAX = 6.0

# Per-observation log-odds increments (an inverse sensor model): a direct
# depth hit is strong evidence of occupancy (p=0.9 equivalent), a ray
# passing through a cell on its way to a farther surface is weaker
# evidence of free space (p=0.35 equivalent) -- ray misses are inherently
# less reliable than direct hits (a thin or reflective surface can be
# missed entirely), so the asymmetry favors not over-clearing real
# obstacles. Hit strength is further scaled by the depth model's
# per-pixel confidence in integrate_depth().
LOG_ODDS_HIT = math.log(0.9 / 0.1)
LOG_ODDS_MISS = math.log(0.35 / 0.65)

# Classification thresholds (p=0.7 / p=0.3 equivalent) -- voxels between
# these two bounds are neither confidently occupied nor confidently free,
# and are excluded from both occupied_count() and free_count().
OCCUPIED_LOG_ODDS_THRESHOLD = math.log(0.7 / 0.3)
FREE_LOG_ODDS_THRESHOLD = math.log(0.3 / 0.7)


def log_odds_to_probability(log_odds: float) -> float:
    return 1.0 / (1.0 + math.exp(-log_odds))


DEFAULT_VOXEL_COLOR = (0x66, 0x7e, 0xa8)  # neutral blue-gray, used when no source frame is available


@dataclass
class VoxelRecord:
    log_odds: float
    last_seen: float  # time.monotonic()
    color: tuple[int, int, int] = DEFAULT_VOXEL_COLOR  # RGB, 0-255 each


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

    def is_occupied(self, index: tuple[int, int, int]) -> bool:
        record = self._voxels.get(index)
        return record is not None and record.log_odds > OCCUPIED_LOG_ODDS_THRESHOLD

    def is_free(self, index: tuple[int, int, int]) -> bool:
        record = self._voxels.get(index)
        return record is not None and record.log_odds < FREE_LOG_ODDS_THRESHOLD

    def _update(
        self,
        index: tuple[int, int, int],
        log_odds_delta: float,
        now: float,
        color: tuple[int, int, int] | None = None,
    ) -> None:
        existing = self._voxels.get(index)
        prior = existing.log_odds if existing is not None else 0.0
        updated = max(LOG_ODDS_MIN, min(LOG_ODDS_MAX, prior + log_odds_delta))
        # A ray-miss update (color=None) passing through an already-colored
        # voxel shouldn't erase its last observed surface color -- only a
        # fresh hit (color provided) ever overwrites it.
        resolved_color = color if color is not None else (existing.color if existing is not None else DEFAULT_VOXEL_COLOR)
        self._voxels[index] = VoxelRecord(log_odds=updated, last_seen=now, color=resolved_color)

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
        color_frame: np.ndarray | None = None,
    ) -> int:
        """Back-project a depth map into this grid using an assumed pinhole
        model (no calibrated intrinsics exist for these cameras), rotate
        each point into the shared fused frame by this camera's known
        mounting yaw, and update log-odds along the ray from the shared
        origin to each observed surface (see module docstring).

        yaw_degrees is this camera's angle in the panoramic array (see
        module docstring / config.OccupancyConfig.camera_yaw_degrees), not
        a per-call override of a calibrated pose -- pass 0.0 to integrate
        directly in camera-local coordinates without fusing.

        color_frame, if given, is the source camera's raw HxWx3 uint8
        frame in OpenCV's native BGR order (this codebase's convention
        throughout) -- not necessarily the same resolution as depth, so
        pixels are mapped proportionally, not 1:1. Each surface hit is
        colored by sampling this frame at the corresponding pixel, so the
        voxel scene reflects the actual observed scene rather than an
        abstract confidence gradient.

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
        color_height, color_width = (
            color_frame.shape[:2] if color_frame is not None and color_frame.ndim == 3 else (0, 0)
        )

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

                # One free-space update per traversed cell per ray -- a
                # ray that clips a voxel across several march steps is
                # still a single observation of that cell, not several.
                visited: set[tuple[int, int, int]] = set()
                for step in range(1, ray_steps):
                    t = step / ray_steps
                    ray_index = self._to_index((x * t, y * t, z * t))
                    if ray_index == surface_index or ray_index in visited:
                        continue
                    visited.add(ray_index)
                    self._update(ray_index, LOG_ODDS_MISS, now)

                hit_color = None
                if color_height and color_width:
                    color_row = min(color_height - 1, int(row / height * color_height))
                    color_col = min(color_width - 1, int(col / width * color_width))
                    b, g, r = color_frame[color_row, color_col][:3]
                    hit_color = (int(r), int(g), int(b))
                self._update(surface_index, LOG_ODDS_HIT * pixel_conf, now, color=hit_color)
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
            key for key, record in list(self._voxels.items())
            if now - record.last_seen > stale_after_seconds
        ]
        for key in stale_keys:
            del self._voxels[key]
        return len(stale_keys)

    def occupied_count(self) -> int:
        # integrate_depth() runs in a background thread (asyncio.to_thread)
        # while the dashboard's /api/occupancy handler calls these read
        # methods on the event loop thread -- iterating self._voxels
        # directly here race with concurrent inserts/evictions there and
        # raise "RuntimeError: dictionary changed size during iteration"
        # (observed in production). list(...) snapshots before filtering.
        return sum(
            1 for record in list(self._voxels.values())
            if record.log_odds > OCCUPIED_LOG_ODDS_THRESHOLD
        )

    def free_count(self) -> int:
        return sum(
            1 for record in list(self._voxels.values())
            if record.log_odds < FREE_LOG_ODDS_THRESHOLD
        )

    def occupied_voxels(self) -> list[dict[str, object]]:
        """Occupied voxel centers in shared-frame meters, for the
        dashboard's 3D scene -- not the fastest representation for a large
        grid, but max_voxels already bounds how big this can get.
        confidence is the posterior occupancy probability (sigmoid of
        log-odds), not a raw depth-model confidence score. color is the
        actual RGB sampled from the source camera frame at the pixel that
        produced this voxel (see integrate_depth's color_frame param),
        not a synthetic confidence gradient."""
        return [
            {
                "x": round((ix + 0.5) * self.voxel_size, 3),
                "y": round((iy + 0.5) * self.voxel_size, 3),
                "z": round((iz + 0.5) * self.voxel_size, 3),
                "confidence": round(log_odds_to_probability(record.log_odds), 3),
                "color": list(record.color),
            }
            for (ix, iy, iz), record in list(self._voxels.items())
            if record.log_odds > OCCUPIED_LOG_ODDS_THRESHOLD
        ]

    def summarize(self) -> dict[str, object]:
        """Compact, LLM-context-friendly summary of the fused occupancy --
        coarse enough to describe in a sentence, not a raw voxel dump.
        Lateral sectors and distances use the horizontal (X, Z) plane
        only; height (Y) is not summarized separately."""
        occupied = [
            (key, math.hypot(key[0] * self.voxel_size, key[2] * self.voxel_size))
            for key, record in list(self._voxels.items())
            if record.log_odds > OCCUPIED_LOG_ODDS_THRESHOLD
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
