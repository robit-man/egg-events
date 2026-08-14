from datetime import datetime, timedelta, timezone

from egg_companion.cognition.default_mode import DefaultModeNetwork
from egg_companion.config import DefaultModeConfig, MemoryConfig
from egg_companion.memory.context import ContextAssembler
from egg_companion.memory.store import MemoryStore
from egg_companion.models import EvidenceRef


def test_default_mode_projects_recurrent_evidence_into_revisable_meta_graph(tmp_path) -> None:
    store = MemoryStore(MemoryConfig(storage_dir=str(tmp_path / "memory")))
    now = datetime.now(timezone.utc)
    store.upsert_entity("person", "Troy", entity_id="person-1", now=now)
    store.upsert_entity("object", "amber mug", entity_id="object-1", now=now)
    for index in range(3):
        at = now + timedelta(hours=index)
        episode_id = f"episode-{index}"
        evidence_id = f"vision-{index}"
        store.append_evidence(
            EvidenceRef(
                evidence_id,
                "vision",
                at,
                "camera",
                "camera-0",
                quality=0.9,
                metadata={"label": "amber mug"},
            )
        )
        store.open_episode(at, episode_id=episode_id)
        store.append_episode_evidence(episode_id, evidence_id)
        for entity_id in ("person-1", "object-1"):
            store.link_entity_evidence(entity_id, evidence_id)
            store.link_episode_entity(episode_id, entity_id, confidence=0.9)
        store.close_episode(episode_id, at, "Troy and the amber mug were observed")
    store.append_evidence(
        EvidenceRef(
            "response-1",
            "action",
            now,
            "interaction-policy",
            "speech-output",
            quality=1.0,
            metadata={
                "input_transcript": "Is that my amber mug?",
                "candidate_response": "Yes, that is your amber mug.",
                "spoken": True,
                "reason": "fresh question-conditioned camera evidence",
            },
        )
    )
    network = DefaultModeNetwork(
        store,
        DefaultModeConfig(
            reflection_min_evidence=2,
            meta_graph_min_confirmations=2,
            meta_graph_limit=10,
        ),
    )

    result = network.run_once()

    assert result["meta_graph"]["abstractions_projected"] == 1
    assert len(result["meta_graph"]["documents"]) == 4
    graph = store.knowledge_graph_snapshot()
    relations = {link["relation"] for link in graph["links"]}
    assert {
        "supports_pattern",
        "recurrently_associated_with",
        "informs_world_model",
        "maintains",
        "guides_communication",
    } <= relations
    abstractions = [
        node for node in graph["nodes"] if node.get("subtype") == "abstraction"
    ]
    assert len(abstractions) == 1
    detail = store.entity_detail(str(abstractions[0]["source_id"]))
    assert detail is not None
    assert detail["entity"]["metadata"]["epistemic_status"] == "inferred_noncausal"
    assert len(detail["entity"]["metadata"]["source_episode_ids"]) == 3

    context = ContextAssembler(store).build(
        "Is that my mug?", "Troy and one object are visible", ("person-1",)
    )
    assert "REFLECTIVE WORKING MODEL" in context
    assert "communication-strategy" in context
    assert "association, not causation" in context
    store.close()
