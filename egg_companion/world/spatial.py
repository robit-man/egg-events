"""Spatial primitives and place ontology for the world model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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
