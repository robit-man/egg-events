from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionResidual:
    residual: float
    new_entity: float
    movement: float
    action_change: float
    seen_count: int


class WorldStatePredictor:
    """Simple inspectable short-horizon presence, location, and action predictor."""

    def __init__(self) -> None:
        self._states: dict[str, tuple[str | None, tuple[float, float], int]] = {}

    def observe(
        self,
        entity_id: str,
        behavior: str | None,
        center: tuple[float, float],
        diagonal: float,
        conflict: float = 0.0,
    ) -> PredictionResidual:
        previous = self._states.get(entity_id)
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
