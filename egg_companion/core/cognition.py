from __future__ import annotations

from collections import deque
from datetime import datetime

from egg_companion.cognition.interaction_policy import InteractionPolicy
from egg_companion.config import CognitiveAttentionConfig
from egg_companion.core.prediction import WorldStatePredictor
from egg_companion.models import AttentionDecision, AttentionTarget, Observation


class CognitiveAttentionController:
    """Prediction-error attention with habituation and separate speech permission."""

    def __init__(self, config: CognitiveAttentionConfig, proactive_enabled: bool) -> None:
        self.config = config
        self.proactive_enabled = proactive_enabled
        self._predictor = WorldStatePredictor()
        self._last_proactive: datetime | None = None
        self._uncertainty_questions: deque[datetime] = deque()

    def allow_uncertainty_question(self, now: datetime) -> bool:
        while self._uncertainty_questions and (now - self._uncertainty_questions[0]).total_seconds() >= 3600:
            self._uncertainty_questions.popleft()
        if len(self._uncertainty_questions) >= self.config.uncertainty_question_budget_per_hour:
            return False
        self._uncertainty_questions.append(now)
        return True

    def evaluate(self, target: AttentionTarget, observation: Observation) -> AttentionDecision:
        detection = target.detection
        entity_id = str(
            detection.attributes.get("identity_id")
            or detection.attributes.get("object_id")
            or target.track_id
        )
        center = (
            (detection.bbox.x1 + detection.bbox.x2) / 2,
            (detection.bbox.y1 + detection.bbox.y2) / 2,
        )
        behavior = detection.attributes.get("behavior")
        communicative_action = 1.0 if behavior in {"waving", "approaching"} else 0.0
        shape = detection.attributes.get("frame_shape")
        diagonal = (
            max(float(sum(value * value for value in shape) ** 0.5), 1.0)
            if isinstance(shape, list)
            else 1.0
        )
        base_conflict = 1.0 if detection.attributes.get("base_label") and detection.attributes.get("base_label") != detection.label else 0.0
        prediction = self._predictor.observe(
            entity_id, str(behavior) if behavior else None, center, diagonal, base_conflict
        )
        prediction_error = prediction.residual
        new_entity = prediction.new_entity
        action_change = prediction.action_change
        movement = prediction.movement
        habituation = 1.0 / (prediction.seen_count ** 0.5)
        speech_alignment = 0.35 if observation.microphone_direction is not None and detection.label == "person" else 0.0
        weighted = (
            self.config.new_entity_weight * new_entity
            + self.config.action_change_weight * max(action_change, communicative_action)
            + self.config.speech_weight * speech_alignment
            + self.config.prediction_error_weight * prediction_error
        )
        capture_priority = min(1.0, weighted * (0.55 + 0.45 * habituation) + (1 - detection.confidence) * 0.12)
        action = behavior in {"waving", "approaching"}
        cooldown = (
            (observation.timestamp - self._last_proactive).total_seconds()
            if self._last_proactive else float("inf")
        )
        allow_speech = bool(
            self.proactive_enabled
            and action
            and capture_priority >= self.config.interruption_threshold
            and cooldown >= self.config.proactive_rate_limit_seconds
        )
        if allow_speech:
            self._last_proactive = observation.timestamp
            reason = f"outward speech permitted: {behavior} with prediction residual"
        elif not self.proactive_enabled:
            reason = "captured internally; proactive speech disabled"
        elif not action:
            reason = "captured internally; no communicative action"
        elif capture_priority < self.config.interruption_threshold:
            reason = "captured internally; below interruption threshold"
        else:
            reason = "captured internally; proactive cooldown active"
        return AttentionDecision(
            round(capture_priority, 4), allow_speech,
            {
                "new_entity": new_entity,
                "action_change": action_change,
                "movement": round(movement, 4),
                "communicative_action": communicative_action,
                "speech_alignment": speech_alignment,
                "prediction_error": round(prediction_error, 4),
                "habituation": round(habituation, 4),
                "uncertainty": round(1 - detection.confidence, 4),
            },
            reason,
            max(0.0, self.config.proactive_rate_limit_seconds - cooldown) if cooldown != float("inf") else 0.0,
        )


__all__ = ["CognitiveAttentionController", "InteractionPolicy"]
