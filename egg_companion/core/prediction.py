from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from egg_companion.world.query import WorldQuery


@dataclass(frozen=True)
class PredictionResidual:
    residual: float
    new_entity: float
    movement: float
    action_change: float
    seen_count: int


@dataclass(frozen=True)
class TypedPrediction:
    """A single typed prediction about a world entity."""
    subject: str
    property: str
    expected_value: Any
    horizon_seconds: float
    uncertainty: float
    based_on_revision: int = 0
    reasoning: str = ""


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

    def predict(self, entity_id: str, horizon_seconds: float = 30.0) -> list[TypedPrediction]:
        """Produce typed predictions for an entity's near-future state."""
        if self._query is None:
            return []
        try:
            ev = self._query.entity(entity_id)
        except Exception:
            return []
        if ev is None or not ev.properties:
            return []

        predictions: list[TypedPrediction] = []
        revision = 0

        # Predict location continuity
        loc_prop = ev.properties.get("current_location")
        if loc_prop:
            loc_val = loc_prop.get("value")
            confidence = loc_prop.get("confidence", 0.5)
            age = self._property_age_seconds(loc_prop)
            uncertainty = min(1.0, 0.2 + (age / horizon_seconds) * 0.5) if age is not None else 0.5
            predictions.append(TypedPrediction(
                subject=entity_id,
                property="current_location",
                expected_value=loc_val,
                horizon_seconds=horizon_seconds,
                uncertainty=uncertainty,
                based_on_revision=loc_prop.get("revision", 0),
                reasoning=f"Last seen at {loc_val} (conf={confidence:.2f})",
            ))

        # Predict behavior continuity
        beh_prop = ev.properties.get("behavior")
        if beh_prop:
            beh_val = beh_prop.get("value")
            predictions.append(TypedPrediction(
                subject=entity_id,
                property="behavior",
                expected_value=beh_val,
                horizon_seconds=min(15.0, horizon_seconds),
                uncertainty=0.4,
                based_on_revision=beh_prop.get("revision", 0),
                reasoning=f"Current behavior: {beh_val}",
            ))

        # Predict observability decay
        obs_prop = ev.properties.get("observability")
        if obs_prop:
            current_obs = obs_prop.get("value", "unknown")
            if current_obs == "observed_present":
                predictions.append(TypedPrediction(
                    subject=entity_id,
                    property="observability",
                    expected_value="observed_absent",
                    horizon_seconds=horizon_seconds * 2,
                    uncertainty=0.6,
                    reasoning="Currently present; likely absent without new observation",
                ))

        return predictions

    def _property_age_seconds(self, prop_data: dict[str, Any]) -> float | None:
        try:
            valid_from = prop_data.get("valid_from", "")
            if not valid_from:
                return None
            ts = datetime.fromisoformat(valid_from)
            return (datetime.now(timezone.utc) - ts).total_seconds()
        except Exception:
            return None

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
