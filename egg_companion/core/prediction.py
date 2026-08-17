from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from egg_companion.world.query import WorldQuery


@dataclass(frozen=True)
class PredictionResidual:
    residual: float
    new_entity: float
    movement: float
    action_change: float
    seen_count: int


class WorldStatePredictor:
    """Short-horizon presence, location, and action predictor.

    Optionally backed by WorldQuery for reconciled world state rather
    than relying solely on an in-memory tuple cache.
    """

    def __init__(self, query: WorldQuery | None = None) -> None:
        self._query = query
        self._states: dict[str, tuple[str | None, tuple[float, float], int]] = {}

    def set_query(self, query: WorldQuery) -> None:
        """Wire in a WorldQuery after construction (for lazy init)."""
        self._query = query

    def observe(
        self,
        entity_id: str,
        behavior: str | None,
        center: tuple[float, float],
        diagonal: float,
        conflict: float = 0.0,
    ) -> PredictionResidual:
        previous = self._states.get(entity_id)
        if previous is None and self._query is not None:
            previous = self._load_from_query(entity_id)

        seen_count = previous[2] + 1 if previous else 1
        new_entity = 1.0 if previous is None else 0.0
        action_change = (
            1.0 if previous and previous[0] != behavior and behavior is not None else 0.0
        )
        movement = 0.0
        if previous:
            distance = (
                (center[0] - previous[1][0]) ** 2 + (center[1] - previous[1][1]) ** 2
            ) ** 0.5
            movement = min(1.0, distance / max(diagonal, 1.0) * 4)
        residual = max(new_entity, movement, action_change, max(0.0, min(1.0, conflict)) * 0.5)
        self._states[entity_id] = (behavior, center, seen_count)
        return PredictionResidual(residual, new_entity, movement, action_change, seen_count)

    def _load_from_query(self, entity_id: str) -> tuple[str | None, tuple[float, float], int] | None:
        """Load entity state from WorldQuery if available."""
        try:
            ev = self._query.entity(entity_id)  # type: ignore[union-attr]
            if ev is None:
                return None
            behavior_val = None
            center = (0.5, 0.5)
            if ev.properties:
                behavior_prop = ev.properties.get("behavior")
                if behavior_prop:
                    behavior_val = str(behavior_prop.get("value", ""))
                loc_prop = ev.properties.get("current_location")
                if loc_prop:
                    loc_val = loc_prop.get("value")
                    if isinstance(loc_val, dict):
                        pos = loc_val.get("position", [0.5, 0.5])
                        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                            center = (float(pos[0]), float(pos[1]))
            return (behavior_val, center, 0)
        except Exception:
            return None
