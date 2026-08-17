"""Spatial primitives and place ontology for the world model."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class SpatialRelation(Enum):
    CONTAINS = "contains"
    OVERLAPS = "overlaps"
    NEAR = "near"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    INSIDE = "inside"
    ADJACENT = "adjacent"
    DISJOINT = "disjoint"


@dataclass(frozen=True)
class BBox2D:
    x1: float
    y1: float
    x2: float
    y2: float
    frame: str = "pixel"

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def overlaps(self, other: BBox2D) -> bool:
        if self.frame != other.frame:
            return False
        return not (self.x2 <= other.x1 or other.x2 <= self.x1 or self.y2 <= other.y1 or other.y2 <= self.y1)

    def contains(self, point: tuple[float, float]) -> bool:
        return self.x1 <= point[0] <= self.x2 and self.y1 <= point[1] <= self.y2

    def iou(self, other: BBox2D) -> float:
        if self.frame != other.frame:
            return 0.0
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        union_area = self.area + other.area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0


@dataclass
class SpatialState:
    frame: str = "pixel"
    bbox: BBox2D | None = None
    center: tuple[float, float] | None = None
    rotation_deg: float | None = None
    velocity: tuple[float, float] | None = None
    timestamp: str = ""
    camera_id: str = ""

    def update(self, bbox: BBox2D | None = None, center: tuple[float, float] | None = None, timestamp: str = "") -> SpatialState:
        return SpatialState(
            frame=self.frame,
            bbox=bbox or self.bbox,
            center=center or self.center,
            rotation_deg=self.rotation_deg,
            velocity=self.velocity,
            timestamp=timestamp or self.timestamp,
            camera_id=self.camera_id,
        )


@dataclass
class PlaceConcept:
    place_id: str
    name: str
    place_type: str = "room"
    spatial_bounds: dict[str, float] = field(default_factory=dict)
    parent_place_id: str | None = None
    properties: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    source_ids: list[str] = field(default_factory=list)

    def contains_point(self, point: tuple[float, float]) -> bool:
        bounds = self.spatial_bounds
        if not bounds:
            return False
        x_min = bounds.get("x_min", float("-inf"))
        x_max = bounds.get("x_max", float("inf"))
        y_min = bounds.get("y_min", float("-inf"))
        y_max = bounds.get("y_max", float("inf"))
        return x_min <= point[0] <= x_max and y_min <= point[1] <= y_max


class SpatialReasoner:
    """Spatial reasoning primitives."""

    def spatial_relation(self, s1: SpatialState, s2: SpatialState) -> list[SpatialRelation]:
        relations = []
        if s1.bbox is None or s2.bbox is None:
            return relations
        if s1.bbox.overlaps(s2.bbox):
            relations.append(SpatialRelation.OVERLAPS)
        c1 = s1.center or s1.bbox.center
        c2 = s2.center or s2.bbox.center
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        threshold = 0.15 * max(s1.bbox.width, s1.bbox.height, 1.0)
        if abs(dx) < threshold and abs(dy) < threshold:
            relations.append(SpatialRelation.NEAR)
        if dx < -threshold:
            relations.append(SpatialRelation.LEFT_OF)
        elif dx > threshold:
            relations.append(SpatialRelation.RIGHT_OF)
        if dy < -threshold:
            relations.append(SpatialRelation.ABOVE)
        elif dy > threshold:
            relations.append(SpatialRelation.BELOW)
        if s1.bbox.contains(c2):
            relations.append(SpatialRelation.CONTAINS)
        if s2.bbox.contains(c1):
            relations.append(SpatialRelation.INSIDE)
        return relations

    def interpolate_position(self, s1: SpatialState, s2: SpatialState, alpha: float) -> tuple[float, float]:
        c1 = s1.center or ((s1.bbox.x1 + s1.bbox.x2) / 2, (s1.bbox.y1 + s1.bbox.y2) / 2) if s1.bbox else (0.0, 0.0)
        c2 = s2.center or ((s2.bbox.x1 + s2.bbox.x2) / 2, (s2.bbox.y1 + s2.bbox.y2) / 2) if s2.bbox else (0.0, 0.0)
        return (c1[0] + alpha * (c2[0] - c1[0]), c1[1] + alpha * (c2[1] - c1[1]))

    def assign_place(self, center: tuple[float, float], places: list[PlaceConcept]) -> PlaceConcept | None:
        best = None
        best_conf = 0.0
        for place in places:
            if place.contains_point(center) and place.confidence > best_conf:
                best = place
                best_conf = place.confidence
        return best


# ======================================================================
# Calibration & Transform Graph
# ======================================================================


@dataclass(frozen=True)
class Calibration:
    """A single 3×4 projection (intrinsic + extrinsic) for one camera at one time.

    ``projection_matrix`` is a flat 12‑element list representing
    ``[r11,r12,r13,tx, r21,r22,r23,ty, r31,r32,r33,tz]``.

    ``distortion`` is an optional 5‑element list for radial‑tangential
    coefficients ``[k1,k2,p1,p2,k3]``.
    """

    camera_id: str
    projection_matrix: list[float]
    distortion: list[float] | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    source: str = "unknown"
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def project(self, point_3d: tuple[float, float, float]) -> tuple[float, float]:
        """Project a 3D world point to 2D pixel coords (ignoring distortion)."""
        px, py, pz = point_3d
        P = self.projection_matrix
        denom = P[8] * px + P[9] * py + P[10] * pz + P[11]
        if abs(denom) < 1e-9:
            return (0.0, 0.0)
        return (
            (P[0] * px + P[1] * py + P[2] * pz + P[3]) / denom,
            (P[4] * px + P[5] * py + P[6] * pz + P[7]) / denom,
        )

    def is_valid_at(self, timestamp: datetime | None) -> bool:
        if timestamp is None:
            return True
        if self.valid_from and timestamp < self.valid_from:
            return False
        if self.valid_to and timestamp > self.valid_to:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "projection_matrix": list(self.projection_matrix),
            "distortion": list(self.distortion) if self.distortion else None,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "source": self.source,
            "recorded_at": self.recorded_at.isoformat(),
        }


@dataclass(frozen=True)
class Transform:
    """A single homogeneous 4×4 rigid‑body transform ``source → target``.

    ``matrix`` is a flat 16‑element list in row‑major order.
    """

    source_frame: str
    target_frame: str
    matrix: list[float]
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    source: str = "unknown"
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def apply(self, point_3d: tuple[float, float, float]) -> tuple[float, float, float]:
        """Transform a 3D point from *source_frame* to *target_frame*."""
        px, py, pz = point_3d
        m = self.matrix
        denom = m[12] * px + m[13] * py + m[14] * pz + m[15]
        if abs(denom) < 1e-9:
            return (0.0, 0.0, 0.0)
        return (
            (m[0] * px + m[1] * py + m[2] * pz + m[3]) / denom,
            (m[4] * px + m[5] * py + m[6] * pz + m[7]) / denom,
            (m[8] * px + m[9] * py + m[10] * pz + m[11]) / denom,
        )

    @property
    def inverse(self) -> Transform:
        """Compute the inverse transform (target → source)."""
        m = self.matrix
        m0, m1, m2 = m[0], m[1], m[2]
        m4, m5, m6 = m[4], m[5], m[6]
        m8, m9, m10 = m[8], m[9], m[10]
        m3, m7, m11 = m[3], m[7], m[11]
        det = m0 * (m5 * m10 - m6 * m9) - m1 * (m4 * m10 - m6 * m8) + m2 * (m4 * m9 - m5 * m8)
        if abs(det) < 1e-9:
            return Transform(self.target_frame, self.source_frame, [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1])
        inv_det = 1.0 / det
        r = [0.0] * 16
        r[0] = (m5 * m10 - m6 * m9) * inv_det
        r[1] = (m2 * m9 - m1 * m10) * inv_det
        r[2] = (m1 * m6 - m2 * m5) * inv_det
        r[4] = (m6 * m8 - m4 * m10) * inv_det
        r[5] = (m0 * m10 - m2 * m8) * inv_det
        r[6] = (m2 * m4 - m0 * m6) * inv_det
        r[8] = (m4 * m9 - m5 * m8) * inv_det
        r[9] = (m1 * m8 - m0 * m9) * inv_det
        r[10] = (m0 * m5 - m1 * m4) * inv_det
        r[3] = -(r[0] * m3 + r[1] * m7 + r[2] * m11)
        r[7] = -(r[4] * m3 + r[5] * m7 + r[6] * m11)
        r[11] = -(r[8] * m3 + r[9] * m7 + r[10] * m11)
        r[12] = r[13] = r[14] = 0.0
        r[15] = 1.0
        return Transform(self.target_frame, self.source_frame, r)

    def is_valid_at(self, timestamp: datetime | None) -> bool:
        if timestamp is None:
            return True
        if self.valid_from and timestamp < self.valid_from:
            return False
        if self.valid_to and timestamp > self.valid_to:
            return False
        return True

    def compose(self, other: Transform) -> Transform:
        """Chain two transforms: ``self`` applied first, then ``other``."""
        a = self.matrix
        b = other.matrix
        out = [0.0] * 16
        for i in range(4):
            for j in range(4):
                out[i * 4 + j] = sum(a[i * 4 + k] * b[k * 4 + j] for k in range(4))
        return Transform(self.source_frame, other.target_frame, out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
            "matrix": list(self.matrix),
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "source": self.source,
            "recorded_at": self.recorded_at.isoformat(),
        }


IDENTITY_4X4: list[float] = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]


class TransformTree:
    """Graph of named coordinate frames connected by ``Transform`` edges.

    Provides ``resolve(source, target, timestamp)`` to find the composed
    transform between any two frames (BFS + matrix composition).  Frames
    with no path between them raise ``KeyError``.

    Thread-safe; transforms are stored in-memory and optionally persisted
    to SQLite via ``load_from_sqlite`` / ``save_to_sqlite``.
    """

    def __init__(self, connection: sqlite3.Connection | None = None) -> None:
        self._lock = threading.RLock()
        self._conn = connection
        self._transforms: dict[tuple[str, str], list[Transform]] = {}
        self._calibrations: dict[str, list[Calibration]] = {}
        if self._conn is not None:
            self._ensure_tables()

    def _ensure_tables(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS transforms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_frame TEXT NOT NULL,
                    target_frame TEXT NOT NULL,
                    matrix_json TEXT NOT NULL,
                    valid_from TEXT,
                    valid_to TEXT,
                    source TEXT NOT NULL DEFAULT 'unknown',
                    recorded_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS calibrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    projection_matrix_json TEXT NOT NULL,
                    distortion_json TEXT,
                    valid_from TEXT,
                    valid_to TEXT,
                    source TEXT NOT NULL DEFAULT 'unknown',
                    recorded_at TEXT NOT NULL
                );
                """
            )

    def add_transform(self, transform: Transform) -> None:
        with self._lock:
            key = (transform.source_frame, transform.target_frame)
            self._transforms.setdefault(key, []).append(transform)

    def get_transforms(self, source: str, target: str) -> list[Transform]:
        with self._lock:
            return list(self._transforms.get((source, target), []))

    def resolve(
        self,
        source_frame: str,
        target_frame: str,
        timestamp: datetime | None = None,
    ) -> Transform | None:
        """Find the shortest-path composed transform from *source_frame* to *target_frame*.

        Returns ``None`` if no path exists.  When *timestamp* is given, only
        valid transforms are traversed; otherwise all transforms are used.
        """
        if source_frame == target_frame:
            return Transform(source_frame, target_frame, list(IDENTITY_4X4))

        # BFS over frame names
        from collections import deque

        visited: set[str] = {source_frame}
        queue: deque[tuple[str, Transform]] = deque()

        # Outgoing edges from source_frame
        for edge_target, transforms in list(self._transforms.items()):
            if edge_target[0] != source_frame:
                continue
            for t in transforms:
                if timestamp is not None and not t.is_valid_at(timestamp):
                    continue
                if edge_target[1] == target_frame:
                    return t
                visited.add(edge_target[1])
                queue.append((edge_target[1], t))

        while queue:
            current, composed = queue.popleft()
            for edge_source, edge_target in list(self._transforms.keys()):
                if edge_source != current:
                    continue
                for t in self._transforms[(edge_source, edge_target)]:
                    if timestamp is not None and not t.is_valid_at(timestamp):
                        continue
                    new_composed = composed.compose(t)
                    if edge_target == target_frame:
                        return new_composed
                    if edge_target not in visited:
                        visited.add(edge_target)
                        queue.append((edge_target, new_composed))

        # Also check reverse edges (inverse transforms)
        for edge_source, edge_target in list(self._transforms.keys()):
            if edge_target != source_frame:
                continue
            for t in self._transforms[(edge_source, edge_target)]:
                if timestamp is not None and not t.is_valid_at(timestamp):
                    continue
                inv = t.inverse
                if edge_source == target_frame:
                    return inv
                if edge_source not in visited:
                    visited.add(edge_source)
                    queue.append((edge_source, inv))

        return None

    def add_calibration(self, calibration: Calibration) -> None:
        with self._lock:
            self._calibrations.setdefault(calibration.camera_id, []).append(calibration)

    def get_calibration(self, camera_id: str, timestamp: datetime | None = None) -> Calibration | None:
        with self._lock:
            cals = self._calibrations.get(camera_id, [])
            if timestamp is None:
                return cals[-1] if cals else None
            for c in reversed(cals):
                if c.is_valid_at(timestamp):
                    return c
            return None

    def list_frames(self) -> list[str]:
        with self._lock:
            frames: set[str] = set()
            for (s, t) in self._transforms:
                frames.add(s)
                frames.add(t)
            return sorted(frames)

    def save_to_sqlite(self) -> None:
        """Persist all transforms and calibrations to SQLite."""
        if self._conn is None:
            return
        with self._lock:
            self._conn.execute("DELETE FROM transforms")
            self._conn.execute("DELETE FROM calibrations")
            for transforms in self._transforms.values():
                for t in transforms:
                    self._conn.execute(
                        """INSERT INTO transforms
                        (source_frame, target_frame, matrix_json, valid_from, valid_to, source, recorded_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            t.source_frame, t.target_frame, json.dumps(t.matrix),
                            t.valid_from.isoformat() if t.valid_from else None,
                            t.valid_to.isoformat() if t.valid_to else None,
                            t.source, t.recorded_at.isoformat(),
                        ),
                    )
            for calibrations in self._calibrations.values():
                for c in calibrations:
                    self._conn.execute(
                        """INSERT INTO calibrations
                        (camera_id, projection_matrix_json, distortion_json, valid_from, valid_to, source, recorded_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            c.camera_id, json.dumps(c.projection_matrix),
                            json.dumps(c.distortion) if c.distortion else None,
                            c.valid_from.isoformat() if c.valid_from else None,
                            c.valid_to.isoformat() if c.valid_to else None,
                            c.source, c.recorded_at.isoformat(),
                        ),
                    )
            self._conn.commit()

    def load_from_sqlite(self) -> None:
        """Load all transforms and calibrations from SQLite."""
        if self._conn is None:
            return
        with self._lock:
            for row in self._conn.execute("SELECT source_frame, target_frame, matrix_json, valid_from, valid_to, source, recorded_at FROM transforms"):
                t = Transform(
                    source_frame=row[0], target_frame=row[1],
                    matrix=json.loads(row[2]),
                    valid_from=datetime.fromisoformat(row[3]) if row[3] else None,
                    valid_to=datetime.fromisoformat(row[4]) if row[4] else None,
                    source=row[5],
                    recorded_at=datetime.fromisoformat(row[6]),
                )
                self._transforms.setdefault((t.source_frame, t.target_frame), []).append(t)
            for row in self._conn.execute("SELECT camera_id, projection_matrix_json, distortion_json, valid_from, valid_to, source, recorded_at FROM calibrations"):
                c = Calibration(
                    camera_id=row[0],
                    projection_matrix=json.loads(row[1]),
                    distortion=json.loads(row[2]) if row[2] else None,
                    valid_from=datetime.fromisoformat(row[3]) if row[3] else None,
                    valid_to=datetime.fromisoformat(row[4]) if row[4] else None,
                    source=row[5],
                    recorded_at=datetime.fromisoformat(row[6]),
                )
                self._calibrations.setdefault(c.camera_id, []).append(c)
