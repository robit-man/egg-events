from __future__ import annotations

import time
from dataclasses import dataclass

from egg_companion.core.attention import AttentionManager
from egg_companion.core.cognition import CognitiveAttentionController
from egg_companion.memory.fusion import EvidenceFusion, FusionResult
from egg_companion.memory.pipeline import MemoryPipeline
from egg_companion.models import (
    AttentionDecision,
    AttentionTarget,
    GraphCognitiveSignal,
    Observation,
)


@dataclass(frozen=True)
class CognitiveTick:
    """One perceive pass: which targets were noticed, each target's prediction-error
    and interruption decision, and the tick's overall novelty signal."""

    targets: list[AttentionTarget]
    decisions: list[tuple[AttentionTarget, AttentionDecision]]
    novelty: float
    graph_feedback: dict[str, GraphCognitiveSignal]
    observation_policy: dict[str, object]


class CognitiveArchitecture:
    """Composes target selection, prediction-error/interruption evaluation, and
    evidence association into one explicit perceive/associate loop, following the
    sensing + perception-cognition-action + memory framing from "Neural Brain: A
    Neuroscience-inspired Framework for Embodied Agents" (arXiv 2505.07634).

    This composes the existing attention, prediction, and memory modules; it does
    not reimplement their logic.
    """

    def __init__(
        self,
        attention: AttentionManager,
        cognitive_attention: CognitiveAttentionController,
        memory: MemoryPipeline | None,
    ) -> None:
        self.attention = attention
        self.cognitive_attention = cognitive_attention
        self.memory = memory
        # Transient, TTL'd bias from the focus_camera action -- kept here
        # (not in MemoryPipeline.observation_policy(), which is a cached,
        # persistent-store-derived read) because "pay closer attention to
        # this camera for a while" is ephemeral runtime state, not a
        # durable policy.
        self._camera_focus: dict[str, float] = {}

    def add_camera_focus(self, camera_id: str, ttl_seconds: float) -> None:
        self._camera_focus[camera_id] = time.monotonic() + max(0.0, ttl_seconds)

    def _active_camera_focus_ids(self) -> list[str]:
        now = time.monotonic()
        self._camera_focus = {
            camera_id: expires_at
            for camera_id, expires_at in self._camera_focus.items()
            if expires_at > now
        }
        return list(self._camera_focus.keys())

    def perceive(self, observation: Observation) -> CognitiveTick:
        entity_ids = [
            str(
                detection.attributes.get("identity_id")
                or detection.attributes.get("object_id")
            )
            for detection in observation.detections
            if (
                detection.attributes.get("identity_id")
                or detection.attributes.get("object_id")
            )
        ]
        graph_feedback = (
            self.memory.graph_signals(entity_ids) if self.memory and entity_ids else {}
        )
        observation_policy = dict(self.memory.observation_policy() if self.memory else {})
        active_camera_focus = self._active_camera_focus_ids()
        if active_camera_focus:
            observation_policy["focus_camera_ids"] = active_camera_focus
        targets = (
            self.attention.select(observation, graph_feedback, observation_policy)
            if self.memory
            else self.attention.select(observation)
        )
        decisions = []
        for target in targets:
            entity_id = str(
                target.detection.attributes.get("identity_id")
                or target.detection.attributes.get("object_id")
                or ""
            )
            decisions.append(
                (
                    target,
                    (
                        self.cognitive_attention.evaluate(
                            target,
                            observation,
                            graph_feedback.get(entity_id),
                            observation_policy,
                        )
                        if self.memory
                        else self.cognitive_attention.evaluate(
                            target, observation, graph_feedback.get(entity_id)
                        )
                    ),
                )
            )
        novelty = max(
            (
                float(decision.components.get("effective_novelty", 0.0))
                for _, decision in decisions
            ),
            default=0.0,
        )
        return CognitiveTick(
            targets=targets,
            decisions=decisions,
            novelty=novelty,
            graph_feedback=graph_feedback,
            observation_policy=observation_policy,
        )

    @staticmethod
    def associate_object(similarity: float, continuity: float = 0.0) -> FusionResult:
        return EvidenceFusion.object(similarity, continuity)
