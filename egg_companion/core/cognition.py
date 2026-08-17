from __future__ import annotations

from collections import deque
from datetime import datetime

from egg_companion.cognition.interaction_policy import InteractionPolicy
from egg_companion.config import CognitiveAttentionConfig
from egg_companion.core.prediction import WorldStatePredictor
from egg_companion.models import (
    AttentionDecision,
    AttentionTarget,
    GraphCognitiveSignal,
    Observation,
)


class CognitiveAttentionController:
    """Prediction-error attention with habituation and separate speech permission."""

    def __init__(
        self,
        config: CognitiveAttentionConfig,
        proactive_enabled: bool,
        world_query: object | None = None,
    ) -> None:
        self.config = config
        self.proactive_enabled = proactive_enabled
        self._predictor = WorldStatePredictor(query=world_query)
        self._last_proactive: datetime | None = None
        self._uncertainty_questions: deque[datetime] = deque()

    def set_world_query(self, world_query: object) -> None:
        """Wire in a WorldQuery after construction (for lazy init)."""
        self._predictor.set_query(world_query)

    def allow_uncertainty_question(self, now: datetime) -> bool:
        while self._uncertainty_questions and (now - self._uncertainty_questions[0]).total_seconds() >= 3600:
            self._uncertainty_questions.popleft()
        if len(self._uncertainty_questions) >= self.config.uncertainty_question_budget_per_hour:
            return False
        self._uncertainty_questions.append(now)
        return True

    def evaluate(
        self,
        target: AttentionTarget,
        observation: Observation,
        graph_signal: GraphCognitiveSignal | None = None,
        observation_policy: dict[str, object] | None = None,
    ) -> AttentionDecision:
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
        graph_familiarity = graph_signal.familiarity if graph_signal else 0.0
        graph_relevance = graph_signal.structural_relevance if graph_signal else 0.0
        knowledge_gap = graph_signal.knowledge_gap if graph_signal else 1.0
        policy = observation_policy or {}
        focus_entities = {
            str(value) for value in policy.get("focus_entity_ids", []) if value
        }
        proactive_entities = {
            str(value) for value in policy.get("proactive_entity_ids", []) if value
        }
        policy_relevance = 1.0 if entity_id in focus_entities else 0.0
        raw_novelty = max(float(target.novelty), new_entity)
        effective_novelty = raw_novelty * (
            1.0 - self.config.graph_familiarity_discount * graph_familiarity
        )
        stable_entity = bool(
            detection.attributes.get("identity_id")
            or detection.attributes.get("object_id")
        )
        sensory_uncertainty = max(0.0, 1.0 - float(detection.confidence))
        reducibility = (
            1.0
            if stable_entity
            else 1.0 - self.config.irreducible_uncertainty_discount
        )
        epistemic_value = sensory_uncertainty * knowledge_gap * reducibility
        repetition_penalty = 1.0 - habituation
        speech_alignment = 0.35 if observation.microphone_direction is not None and detection.label == "person" else 0.0
        weighted = (
            self.config.new_entity_weight * effective_novelty
            + self.config.action_change_weight * max(action_change, communicative_action)
            + self.config.speech_weight * speech_alignment
            + self.config.prediction_error_weight
            * prediction_error
            * (1.0 - 0.65 * graph_familiarity)
            + self.config.epistemic_value_weight * epistemic_value
            + self.config.observation_policy_weight * policy_relevance
        )
        # Habituation suppresses recurrent novelty/prediction channels, while a
        # genuinely communicative action remains capable of attracting focus.
        capture_priority = min(
            1.0,
            weighted * (0.40 + 0.60 * habituation)
            + 0.08 * graph_relevance * epistemic_value,
        )
        model_directed_action = bool(
            stable_entity and entity_id in proactive_entities
        )
        action = behavior in {"waving", "approaching"} or model_directed_action
        cooldown = (
            (observation.timestamp - self._last_proactive).total_seconds()
            if self._last_proactive else float("inf")
        )
        allow_speech = bool(
            self.proactive_enabled
            and action
            and capture_priority >= self.config.communicative_action_threshold
            and cooldown >= self.config.proactive_rate_limit_seconds
        )
        if allow_speech:
            self._last_proactive = observation.timestamp
            reason = (
                f"outward speech permitted: {behavior} with prediction residual"
                if behavior in {"waving", "approaching"}
                else "outward speech permitted by the active model-authored observation policy"
            )
        elif not self.proactive_enabled:
            reason = "captured internally; proactive speech disabled"
        elif not action:
            reason = "captured internally; no communicative or model-directed action"
        elif capture_priority < self.config.communicative_action_threshold:
            reason = "captured internally; below communicative-action threshold"
        else:
            reason = "captured internally; proactive cooldown active"
        return AttentionDecision(
            round(capture_priority, 4), allow_speech,
            {
                "new_entity": new_entity,
                "raw_novelty": round(raw_novelty, 4),
                "effective_novelty": round(effective_novelty, 4),
                "action_change": action_change,
                "movement": round(movement, 4),
                "communicative_action": communicative_action,
                "speech_alignment": speech_alignment,
                "prediction_error": round(prediction_error, 4),
                "habituation": round(habituation, 4),
                "repetition_penalty": round(repetition_penalty, 4),
                "uncertainty": round(sensory_uncertainty, 4),
                "epistemic_value": round(epistemic_value, 4),
                "graph_familiarity": round(graph_familiarity, 4),
                "graph_relevance": round(graph_relevance, 4),
                "graph_knowledge_gap": round(knowledge_gap, 4),
                "observation_policy_relevance": policy_relevance,
                "model_directed_action": float(model_directed_action),
            },
            reason,
            max(0.0, self.config.proactive_rate_limit_seconds - cooldown) if cooldown != float("inf") else 0.0,
        )


__all__ = ["CognitiveAttentionController", "InteractionPolicy"]
