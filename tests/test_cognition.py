from datetime import datetime, timedelta, timezone

from egg_companion.config import CognitiveAttentionConfig
from egg_companion.core.cognition import CognitiveAttentionController, InteractionPolicy
from egg_companion.models import (
    AttentionTarget, BoundingBox, Detection, GraphCognitiveSignal, Observation,
)


def target(at, behavior="waving"):
    detection = Detection(
        "person", 0.9, BoundingBox(100, 100, 500, 900),
        {"behavior": behavior, "identity_id": "person-001", "frame_shape": [1080, 1920]},
    )
    observation = Observation("camera-0", at, (detection,), microphone_direction=20)
    attention = AttentionTarget("track-1", detection, 1.0, 0.9, "new person", "camera-0", at)
    return attention, observation


def test_prediction_attention_habituates_without_removing_capture() -> None:
    controller = CognitiveAttentionController(CognitiveAttentionConfig(), proactive_enabled=False)
    now = datetime.now(timezone.utc)
    attention, observation = target(now, "standing")
    first = controller.evaluate(attention, observation)
    attention, observation = target(now + timedelta(seconds=1), "standing")
    repeated = controller.evaluate(attention, observation)

    assert first.capture_priority > repeated.capture_priority
    assert first.components["prediction_error"] == 1.0
    assert repeated.components["habituation"] < first.components["habituation"]
    assert not repeated.allow_outward_speech
    assert "proactive speech disabled" in repeated.reason


def test_communicative_prediction_error_can_cross_interruption_threshold() -> None:
    config = CognitiveAttentionConfig(interruption_threshold=0.7)
    controller = CognitiveAttentionController(config, proactive_enabled=True)
    attention, observation = target(datetime.now(timezone.utc), "waving")
    decision = controller.evaluate(attention, observation)

    assert decision.capture_priority >= config.interruption_threshold
    assert decision.allow_outward_speech
    assert "waving" in decision.reason


def test_interaction_policy_has_no_wake_word_and_suppresses_duplicates() -> None:
    policy = InteractionPolicy()
    first = policy.evaluate("Could you describe that mug?", "The segmented object resembles a ceramic mug.")
    repeated = policy.evaluate("Tell me again.", "The segmented object resembles a ceramic mug.")
    silent = policy.evaluate("Room conversation", "[[SILENT]]")

    assert first.allow_speech
    assert not repeated.allow_speech
    assert repeated.reason == "duplicate response suppressed"
    assert not silent.allow_speech


def test_uncertainty_questions_obey_hourly_budget() -> None:
    config = CognitiveAttentionConfig(uncertainty_question_budget_per_hour=1)
    controller = CognitiveAttentionController(config, proactive_enabled=False)
    now = datetime.now(timezone.utc)
    assert controller.allow_uncertainty_question(now)
    assert not controller.allow_uncertainty_question(now + timedelta(minutes=10))
    assert controller.allow_uncertainty_question(now + timedelta(hours=1, seconds=1))


def test_graph_familiarity_downweights_raw_novelty_without_erasing_gap() -> None:
    now = datetime.now(timezone.utc)
    attention, observation = target(now, "standing")
    unfamiliar = CognitiveAttentionController(
        CognitiveAttentionConfig(), proactive_enabled=False
    ).evaluate(attention, observation)
    familiar = CognitiveAttentionController(
        CognitiveAttentionConfig(), proactive_enabled=False
    ).evaluate(
        attention,
        observation,
        GraphCognitiveSignal(
            "person-001", familiarity=0.95, structural_relevance=0.8,
            knowledge_gap=0.3, evidence_count=20, edge_count=8,
        ),
    )

    assert familiar.components["effective_novelty"] < unfamiliar.components["effective_novelty"]
    assert familiar.capture_priority < unfamiliar.capture_priority
    assert familiar.components["graph_relevance"] == 0.8
