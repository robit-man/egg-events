from __future__ import annotations

from dataclasses import dataclass

from egg_companion.core.attention import AttentionManager
from egg_companion.core.cognition import CognitiveAttentionController
from egg_companion.memory.fusion import EvidenceFusion, FusionResult
from egg_companion.memory.pipeline import MemoryPipeline
from egg_companion.models import AttentionDecision, AttentionTarget, Observation


@dataclass(frozen=True)
class CognitiveTick:
    """One perceive pass: which targets were noticed, each target's prediction-error
    and interruption decision, and the tick's overall novelty signal."""

    targets: list[AttentionTarget]
    decisions: list[tuple[AttentionTarget, AttentionDecision]]
    novelty: float


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

    def perceive(self, observation: Observation) -> CognitiveTick:
        targets = self.attention.select(observation)
        decisions = [(target, self.cognitive_attention.evaluate(target, observation)) for target in targets]
        novelty = max((target.priority for target in targets), default=0.0)
        return CognitiveTick(targets=targets, decisions=decisions, novelty=novelty)

    @staticmethod
    def associate_object(similarity: float, continuity: float = 0.0) -> FusionResult:
        return EvidenceFusion.object(similarity, continuity)
