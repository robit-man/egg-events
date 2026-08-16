from datetime import datetime, timezone

from egg_companion.config import EggConfig
from egg_companion.memory.pipeline import MemoryPipeline
from egg_companion.memory.store import MemoryStore
from egg_companion.models import EvidenceRef, PerceptualEvent


def test_revisable_interaction_strategy_round_trips_through_graph(tmp_path) -> None:
    config = EggConfig.model_validate(
        {
            "audio": {"input_device": "default", "doa_mode": "disabled"},
            "omnius": {"model": "test", "voice_model": "test"},
            "identity": {"enabled": False},
            "object_learning": {"enabled": False},
            "camera_discovery": {"enabled": False},
            "memory": {"storage_dir": str(tmp_path / "memory")},
        }
    )
    store = MemoryStore(config.memory)
    pipeline = MemoryPipeline(config, store)
    now = datetime.now(timezone.utc)
    evidence = EvidenceRef(
        "social-evidence-1",
        "speech",
        now,
        "model-social-reflection",
        "conversation",
        quality=0.87,
        metadata={"context_id": "turn-1", "time_local_interpretation": True},
    )
    event = PerceptualEvent(
        "social-event-1",
        "social_reflection",
        now,
        "model-social-reflection",
        (evidence,),
        (
            "interaction-state:1",
            "interaction-strategy:current",
            "person-001",
            "social-profile:person-001",
        ),
        payload={
            "entities": [
                {
                    "id": "person-001",
                    "type": "person",
                    "label": "Troy",
                    "confidence": 1.0,
                },
                {
                    "id": "interaction-state:1",
                    "type": "interaction_state",
                    "label": "engaged",
                    "confidence": 0.82,
                    "time_local": True,
                    "revisable": True,
                },
                {
                    "id": "social-profile:person-001",
                    "type": "social_profile",
                    "label": "Revisable interaction profile",
                    "confidence": 0.81,
                    "subject_id": "person-001",
                    "summary": "Gives direct, concrete corrective feedback.",
                    "sentiment_trajectory": "Frustrated and still engaged in this turn.",
                    "communication_patterns": ["States desired behavior explicitly."],
                    "interaction_preferences": ["Requested rapid grounded answers."],
                    "uncertainties": ["Whether this applies outside visual questions."],
                    "profile_scope": "observed_interactions_only",
                    "revisable": True,
                },
                {
                    "id": "interaction-strategy:current",
                    "type": "interaction_strategy",
                    "label": "Evolving interaction strategy",
                    "confidence": 0.87,
                    "directive": "Answer from current sensor evidence before elaborating.",
                    "rationale": "The prior turn explicitly requested faster grounding.",
                    "revisable": True,
                },
            ],
            "relations": [
                {
                    "source_id": "interaction-state:1",
                    "relation": "informs_interaction_strategy",
                    "target_id": "interaction-strategy:current",
                    "confidence": 0.87,
                },
                {
                    "source_id": "social-profile:person-001",
                    "relation": "models_interaction_with",
                    "target_id": "person-001",
                    "confidence": 0.81,
                    "metadata": {"not_a_personality_claim": True},
                },
            ],
            "skip_pairwise_co_observation": True,
        },
    )

    accepted, _ = pipeline.ingest(event)

    assert accepted
    assert pipeline.interaction_strategy()["directive"] == (
        "Answer from current sensor evidence before elaborating."
    )
    profiles = pipeline.social_profiles(["person-001"])
    assert profiles[0]["interaction_preferences"] == [
        "Requested rapid grounded answers."
    ]
    assert profiles[0]["profile_scope"] == "observed_interactions_only"
    detail = store.entity_detail("interaction-strategy:current")
    assert detail is not None
    assert detail["evidence"][0]["evidence_id"] == "social-evidence-1"
