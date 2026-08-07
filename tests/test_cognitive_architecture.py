from datetime import datetime, timezone
from types import SimpleNamespace

from egg_companion.cognition.architecture import CognitiveArchitecture
from egg_companion.config import CognitiveAttentionConfig
from egg_companion.core.attention import AttentionManager
from egg_companion.core.cognition import CognitiveAttentionController
from egg_companion.memory.fusion import EvidenceFusion
from egg_companion.models import AttentionTarget, BoundingBox, Detection, Observation


def _observation() -> Observation:
    person = Detection(
        "person", 0.9, BoundingBox(100, 100, 500, 900),
        {"behavior": "waving", "frame_shape": [1080, 1920]},
    )
    return Observation("camera-0", datetime.now(timezone.utc), (person,))


def _target(track_id: str, priority: float) -> AttentionTarget:
    detection = Detection("person", 0.9, BoundingBox(100, 100, 500, 900))
    return AttentionTarget(track_id, detection, 1.0, priority, "new person", "camera-0", datetime.now(timezone.utc))


class _FakeAttention:
    """Records how CognitiveArchitecture calls AttentionManager.select without
    depending on its real (random-track-id, stateful) implementation."""

    def __init__(self, targets: list[AttentionTarget]) -> None:
        self._targets = targets
        self.calls: list[Observation] = []

    def select(self, observation: Observation) -> list[AttentionTarget]:
        self.calls.append(observation)
        return self._targets


class _FakeCognitiveAttention:
    """Records how CognitiveArchitecture calls CognitiveAttentionController.evaluate."""

    def __init__(self, decision_by_track_id: dict[str, object]) -> None:
        self._decision_by_track_id = decision_by_track_id
        self.calls: list[tuple[AttentionTarget, Observation]] = []

    def evaluate(self, target: AttentionTarget, observation: Observation):
        self.calls.append((target, observation))
        return self._decision_by_track_id[target.track_id]


def test_perceive_composes_select_then_evaluate_per_target() -> None:
    observation = _observation()
    target_a, target_b = _target("track-a", 0.4), _target("track-b", 0.9)
    decision_a, decision_b = SimpleNamespace(capture_priority=0.1), SimpleNamespace(capture_priority=0.2)
    attention = _FakeAttention([target_a, target_b])
    cognitive_attention = _FakeCognitiveAttention({"track-a": decision_a, "track-b": decision_b})
    brain = CognitiveArchitecture(attention, cognitive_attention, memory=None)

    tick = brain.perceive(observation)

    assert attention.calls == [observation]
    assert tick.targets == [target_a, target_b]
    assert tick.decisions == [(target_a, decision_a), (target_b, decision_b)]
    assert [call[0] for call in cognitive_attention.calls] == [target_a, target_b]
    assert all(call[1] is observation for call in cognitive_attention.calls)
    assert tick.novelty == max(target_a.priority, target_b.priority)


def test_perceive_returns_empty_tick_when_nothing_is_selected() -> None:
    observation = _observation()
    attention = _FakeAttention([])
    cognitive_attention = _FakeCognitiveAttention({})
    brain = CognitiveArchitecture(attention, cognitive_attention, memory=None)

    tick = brain.perceive(observation)

    assert tick.targets == []
    assert tick.decisions == []
    assert tick.novelty == 0.0
    assert cognitive_attention.calls == []


def test_perceive_integrates_with_real_attention_and_prediction_modules() -> None:
    observation = _observation()
    brain = CognitiveArchitecture(
        AttentionManager(track_ttl_seconds=10, min_priority=0.1),
        CognitiveAttentionController(CognitiveAttentionConfig(), proactive_enabled=True),
        memory=None,
    )

    tick = brain.perceive(observation)

    assert len(tick.targets) == 1
    assert [target for target, _ in tick.decisions] == tick.targets
    assert tick.novelty == max((target.priority for target in tick.targets), default=0.0)


def test_associate_object_matches_evidence_fusion() -> None:
    brain = CognitiveArchitecture(
        AttentionManager(track_ttl_seconds=10, min_priority=0.1),
        CognitiveAttentionController(CognitiveAttentionConfig(), proactive_enabled=False),
        memory=None,
    )

    assert brain.associate_object(0.9, continuity=0.5) == EvidenceFusion.object(0.9, 0.5)
