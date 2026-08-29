from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np

from egg_companion.models import Observation


class EnvironmentalSettings(Protocol):
    minimum_salience: float
    salience_half_life_seconds: float
    habituation_half_life_seconds: float
    raw_frame_width: int
    raw_novelty_minimum: float
    raw_surprise_sigma: float
    raw_reference_blend: float
    raw_probe_min_interval_seconds: float


@dataclass(frozen=True)
class EnvironmentalStimulus:
    """One structural change in the embodied perceptual stream.

    The signal says only that the current evidence differs from the preceding
    evidence.  It does not assign social meaning or prescribe an action; that
    remains the multimodal model's job.
    """

    stimulus_id: str
    sequence: int
    camera_id: str
    observed_at: datetime
    observed_monotonic: float
    salience: float
    raw_salience: float
    habituation: float
    causes: tuple[str, ...]
    previous_person_count: int
    current_person_count: int
    previous_person_ids: tuple[str, ...]
    current_person_ids: tuple[str, ...]
    entity_ids: tuple[str, ...]
    semantic_labels: tuple[str, ...]
    attention_components: dict[str, float]

    def decayed_salience(self, now: float, half_life_seconds: float) -> float:
        age = max(0.0, now - self.observed_monotonic)
        decay = math.exp(-math.log(2.0) * age / max(half_life_seconds, 0.001))
        return max(0.0, min(1.0, self.salience * decay))


@dataclass(frozen=True)
class _SceneState:
    person_count: int
    person_ids: frozenset[str]
    entity_ids: frozenset[str]
    semantic_labels: frozenset[str]


class EnvironmentalNoveltyTracker:
    """Convert perception changes into decaying, habituating evidence signals.

    No object name, phrase, gesture, or inferred intent maps to speech here.
    Person constellation, entity constellation, scene cues, and the existing
    prediction-error attention values are compared structurally. Repeated
    transitions lose force and recover gradually when they have not recurred.
    """

    def __init__(self, settings: EnvironmentalSettings) -> None:
        self._settings = settings
        self._states: dict[str, _SceneState] = {}
        self._habituation: dict[str, tuple[float, float]] = {}
        self._sequence = 0

    @property
    def sequence(self) -> int:
        return self._sequence

    def observe(
        self,
        observation: Observation,
        decisions: list[tuple[object, object]],
        novelty: float,
        now: float,
    ) -> EnvironmentalStimulus | None:
        state = self._state(observation)
        prior = self._states.get(observation.camera_id)
        self._states[observation.camera_id] = state

        attention_components = self._attention_components(decisions, novelty)
        components: dict[str, float] = {}
        if prior is None:
            if state.person_count:
                components["person_presence_changed"] = 1.0
            if state.entity_ids or state.semantic_labels:
                components["scene_constellation_changed"] = max(
                    attention_components.values(), default=1.0
                )
        else:
            if bool(prior.person_count) != bool(state.person_count):
                components["person_presence_changed"] = 1.0
            if prior.person_count != state.person_count:
                components["person_count_changed"] = min(
                    1.0,
                    abs(state.person_count - prior.person_count)
                    / max(state.person_count, prior.person_count, 1),
                )
            person_change = self._set_change(prior.person_ids, state.person_ids)
            if person_change:
                components["person_constellation_changed"] = person_change
            entity_change = self._set_change(prior.entity_ids, state.entity_ids)
            semantic_change = self._set_change(
                prior.semantic_labels, state.semantic_labels
            )
            scene_change = max(entity_change, semantic_change)
            if scene_change:
                components["scene_constellation_changed"] = scene_change

        prediction_signal = max(attention_components.values(), default=0.0)
        if prediction_signal:
            components["prediction_error_changed"] = prediction_signal
        raw_salience = max(components.values(), default=0.0)
        if raw_salience <= 0:
            return None

        fingerprint = self._fingerprint(prior, state, tuple(sorted(components)))
        prior_strength, prior_at = self._habituation.get(fingerprint, (0.0, now))
        recovered_strength = prior_strength * math.exp(
            -math.log(2.0)
            * max(0.0, now - prior_at)
            / max(self._settings.habituation_half_life_seconds, 0.001)
        )
        salience = raw_salience / (1.0 + recovered_strength)
        self._habituation[fingerprint] = (
            recovered_strength + raw_salience,
            now,
        )
        if salience < self._settings.minimum_salience:
            return None

        self._sequence += 1
        stimulus_id = f"environment:{self._sequence}:{fingerprint[:12]}"
        return EnvironmentalStimulus(
            stimulus_id=stimulus_id,
            sequence=self._sequence,
            camera_id=observation.camera_id,
            observed_at=observation.timestamp,
            observed_monotonic=now,
            salience=round(salience, 6),
            raw_salience=round(raw_salience, 6),
            habituation=round(recovered_strength, 6),
            causes=tuple(sorted(components)),
            previous_person_count=prior.person_count if prior else 0,
            current_person_count=state.person_count,
            previous_person_ids=tuple(sorted(prior.person_ids)) if prior else (),
            current_person_ids=tuple(sorted(state.person_ids)),
            entity_ids=tuple(sorted(state.entity_ids)),
            semantic_labels=tuple(sorted(state.semantic_labels)),
            attention_components=attention_components,
        )

    @staticmethod
    def _state(observation: Observation) -> _SceneState:
        people = [item for item in observation.detections if item.label == "person"]
        person_ids = frozenset(
            str(item.attributes.get("identity_id"))
            for item in people
            if item.attributes.get("identity_id")
        )
        entity_ids = frozenset(
            str(value)
            for item in observation.detections
            for value in (
                item.attributes.get("identity_id"),
                item.attributes.get("object_id"),
            )
            if value
        )
        return _SceneState(
            person_count=len(people),
            person_ids=person_ids,
            entity_ids=entity_ids,
            semantic_labels=frozenset(str(item) for item in observation.semantic_labels),
        )

    @staticmethod
    def _attention_components(
        decisions: list[tuple[object, object]], novelty: float
    ) -> dict[str, float]:
        values: dict[str, float] = {"effective_novelty": max(0.0, float(novelty))}
        for _target, decision in decisions:
            components = getattr(decision, "components", {})
            if not isinstance(components, dict):
                continue
            for key in (
                "prediction_error",
                "action_change",
                "movement",
                "epistemic_value",
                "observation_policy_relevance",
            ):
                value = components.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    values[key] = max(values.get(key, 0.0), float(value))
        return {
            key: round(max(0.0, min(1.0, value)), 6)
            for key, value in values.items()
            if value > 0
        }

    @staticmethod
    def _set_change(left: frozenset[str], right: frozenset[str]) -> float:
        if left == right:
            return 0.0
        union = left | right
        return 1.0 - len(left & right) / len(union) if union else 0.0

    @staticmethod
    def _fingerprint(
        prior: _SceneState | None,
        current: _SceneState,
        causes: tuple[str, ...],
    ) -> str:
        payload = repr(
            (
                prior.person_count if prior else 0,
                tuple(sorted(prior.person_ids)) if prior else (),
                tuple(sorted(prior.entity_ids)) if prior else (),
                current.person_count,
                tuple(sorted(current.person_ids)),
                tuple(sorted(current.entity_ids)),
                tuple(sorted(current.semantic_labels)),
                causes,
            )
        )
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class _RawFrameState:
    reference: np.ndarray
    mean_change: float = 0.0
    variance: float = 0.0
    observations: int = 1
    last_probe_at: float | None = None


class AdaptiveVisualNovelty:
    """Cheap, content-agnostic frame surprise used to wake sparse perception.

    It runs on a tiny grayscale thumbnail and learns each camera's ordinary
    motion/noise distribution. It never labels pixels or chooses an action.
    """

    def __init__(self, settings: EnvironmentalSettings) -> None:
        self._settings = settings
        self._states: dict[str, _RawFrameState] = {}

    def observe(
        self, camera_id: str, frame: np.ndarray, now: float
    ) -> tuple[float, bool, float]:
        thumbnail = self._thumbnail(frame)
        state = self._states.get(camera_id)
        if state is None:
            self._states[camera_id] = _RawFrameState(
                reference=thumbnail,
                last_probe_at=now,
            )
            return 1.0, True, self._settings.raw_novelty_minimum

        change = float(np.mean(np.abs(thumbnail - state.reference))) / 255.0
        deviation = change - state.mean_change
        baseline_std = math.sqrt(max(state.variance, 0.0))
        threshold = max(
            self._settings.raw_novelty_minimum,
            state.mean_change + self._settings.raw_surprise_sigma * baseline_std,
        )
        signal = max(0.0, min(1.0, change / max(threshold, 0.000001)))
        interval_clear = (
            state.last_probe_at is None
            or now - state.last_probe_at
            >= self._settings.raw_probe_min_interval_seconds
        )
        wake = change >= threshold and interval_clear
        if wake:
            state.last_probe_at = now

        blend = self._settings.raw_reference_blend
        state.reference = (1.0 - blend) * state.reference + blend * thumbnail
        stats_blend = min(0.2, max(0.01, blend))
        state.mean_change += stats_blend * deviation
        state.variance = max(
            0.0,
            (1.0 - stats_blend)
            * (state.variance + stats_blend * deviation * deviation),
        )
        state.observations += 1
        return round(signal, 6), wake, round(threshold, 6)

    def _thumbnail(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        source = frame
        if source.ndim == 3:
            source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        height, width = source.shape[:2]
        target_width = min(self._settings.raw_frame_width, width)
        target_height = max(1, round(height * target_width / max(width, 1)))
        return cv2.resize(
            source,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)


__all__ = [
    "AdaptiveVisualNovelty",
    "EnvironmentalNoveltyTracker",
    "EnvironmentalStimulus",
]
