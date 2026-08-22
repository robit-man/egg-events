from datetime import datetime, timezone
from types import SimpleNamespace

from egg_companion.cognition.architecture import CognitiveArchitecture
from egg_companion.config import CognitiveAttentionConfig
from egg_companion.core.attention import AttentionManager
from egg_companion.core.cognition import CognitiveAttentionController
from egg_companion.memory.fusion import EvidenceFusion
from egg_companion.models import (
    AttentionTarget, BoundingBox, Detection, GraphCognitiveSignal, Observation,
)


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
        self.calls: list[tuple[AttentionTarget, Observation, GraphCognitiveSignal | None]] = []

    def evaluate(
        self,
        target: AttentionTarget,
        observation: Observation,
        signal: GraphCognitiveSignal | None = None,
    ):
        self.calls.append((target, observation, signal))
        return self._decision_by_track_id[target.track_id]


def test_perceive_composes_select_then_evaluate_per_target() -> None:
    observation = _observation()
    target_a, target_b = _target("track-a", 0.4), _target("track-b", 0.9)
    decision_a = SimpleNamespace(
        capture_priority=0.1, components={"effective_novelty": 0.4}
    )
    decision_b = SimpleNamespace(
        capture_priority=0.2, components={"effective_novelty": 0.9}
    )
    attention = _FakeAttention([target_a, target_b])
    cognitive_attention = _FakeCognitiveAttention({"track-a": decision_a, "track-b": decision_b})
    brain = CognitiveArchitecture(attention, cognitive_attention, memory=None)

    tick = brain.perceive(observation)

    assert attention.calls == [observation]
    assert tick.targets == [target_a, target_b]
    assert tick.decisions == [(target_a, decision_a), (target_b, decision_b)]
    assert [call[0] for call in cognitive_attention.calls] == [target_a, target_b]
    assert all(call[1] is observation for call in cognitive_attention.calls)
    assert all(call[2] is None for call in cognitive_attention.calls)
    assert tick.novelty == 0.9
    assert tick.graph_feedback == {}


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
    assert tick.novelty == max(
        decision.components["effective_novelty"] for _, decision in tick.decisions
    )


class _PolicyRecordingAttention:
    """Records the observation_policy dict CognitiveArchitecture passes to
    select(), so tests can see exactly what it merged in."""

    def __init__(self) -> None:
        self.policies: list[dict[str, object]] = []

    def select(self, observation, graph_feedback=None, observation_policy=None):
        self.policies.append(observation_policy or {})
        return []


def test_add_camera_focus_merges_into_observation_policy_for_real_memory() -> None:
    class _FakeMemory:
        def observation_policy(self):
            return {"focus_terms": ["x"]}

        def graph_signals(self, entity_ids):
            return {}

    attention = _PolicyRecordingAttention()
    cognitive_attention = _FakeCognitiveAttention({})
    brain = CognitiveArchitecture(attention, cognitive_attention, memory=_FakeMemory())

    brain.add_camera_focus("camera-video1", ttl_seconds=30.0)
    observation = _observation()
    brain.perceive(observation)

    assert attention.policies[0]["focus_camera_ids"] == ["camera-video1"]
    assert attention.policies[0]["focus_terms"] == ["x"]


def test_no_active_camera_focus_leaves_observation_policy_untouched() -> None:
    class _FakeMemory:
        def observation_policy(self):
            return {"focus_terms": ["x"]}

        def graph_signals(self, entity_ids):
            return {}

    attention = _PolicyRecordingAttention()
    cognitive_attention = _FakeCognitiveAttention({})
    brain = CognitiveArchitecture(attention, cognitive_attention, memory=_FakeMemory())

    brain.perceive(_observation())

    assert "focus_camera_ids" not in attention.policies[0]


def test_camera_focus_expires_after_ttl(monkeypatch) -> None:
    import time as time_module

    attention = _FakeAttention([])
    cognitive_attention = _FakeCognitiveAttention({})
    brain = CognitiveArchitecture(attention, cognitive_attention, memory=None)

    fake_now = [1000.0]
    monkeypatch.setattr(time_module, "monotonic", lambda: fake_now[0])

    brain.add_camera_focus("camera-video1", ttl_seconds=10.0)
    assert brain._active_camera_focus_ids() == ["camera-video1"]

    fake_now[0] = 1011.0
    assert brain._active_camera_focus_ids() == []


def test_associate_object_matches_evidence_fusion() -> None:
    brain = CognitiveArchitecture(
        AttentionManager(track_ttl_seconds=10, min_priority=0.1),
        CognitiveAttentionController(CognitiveAttentionConfig(), proactive_enabled=False),
        memory=None,
    )

    assert brain.associate_object(0.9, continuity=0.5) == EvidenceFusion.object(0.9, 0.5)
