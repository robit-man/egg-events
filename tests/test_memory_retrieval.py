from datetime import datetime, timezone

import numpy as np

from egg_companion.config import MemoryConfig
from egg_companion.memory.context import ContextAssembler
from egg_companion.memory.retrieval import AssociativeRetriever
from egg_companion.memory.store import MemoryStore
from egg_companion.models import EvidenceRef


def populated_store(tmp_path) -> MemoryStore:
    config = MemoryConfig(
        storage_dir=str(tmp_path / "memory"), retrieval_limit=6, graph_max_nodes=20,
        context_max_characters=1800,
    )
    store = MemoryStore(config)
    now = datetime.now(timezone.utc)
    store.upsert_entity(
        "object", "ceramic mug", {"confidence": 0.94, "label_source": "ornith-vlm"},
        "object-001", now=now,
    )
    store.upsert_entity("person", "Ada", {"confidence": 0.97}, "person-001", now=now)
    evidence = EvidenceRef(
        "evidence-mug", "vision", now, "camera", "camera-0", quality=0.91,
        metadata={"label": "ceramic mug", "action": "held by person-001"},
    )
    store.append_evidence(evidence)
    store.open_episode(now, 0.7, "episode-mug")
    store.append_episode_evidence("episode-mug", evidence.evidence_id)
    for entity_id in ("object-001", "person-001"):
        store.link_entity_evidence(entity_id, evidence.evidence_id)
        store.link_episode_entity("episode-mug", entity_id, confidence=0.9)
    store.link_entities_once(
        "person-001", "held", "object-001", 0.88, now, evidence_id=evidence.evidence_id
    )
    store.assert_claim_once(
        "object-001", "has_label", "ceramic mug", 0.94, now,
        source="ornith-vlm", evidence_id=evidence.evidence_id,
    )
    obsolete = store.assert_claim(
        "object-001", "has_label", "wrong label", 0.4, now, source="detector"
    )
    store.revise_claim(obsolete, "correct", "ornith-vlm", "ceramic mug", evidence.evidence_id, now)
    store.add_embedding(
        "entity", "object-001", "masked-object", "open-clip", np.array([1.0, 0.0]), 0.91, now
    )
    store.close_episode("episode-mug", now, "Ada held the ceramic mug")
    return store


def test_retrieval_combines_lexical_clip_live_and_graph_channels(tmp_path) -> None:
    store = populated_store(tmp_path)
    hits = AssociativeRetriever(store).retrieve(
        "Is this the ceramic mug Ada held?", ("person-001",), np.array([1.0, 0.0])
    )
    object_hit = next(hit for hit in hits if hit.owner_id == "object-001")
    assert any("CLIP" in reason for reason in object_hit.why)
    assert any("graph path" in reason for reason in object_hit.why)
    assert all(hit.owner_id != "wrong label" for hit in hits)
    assert hits == sorted(hits, key=lambda hit: (hit.score, hit.confidence), reverse=True)


def test_context_is_bounded_grounded_and_excludes_revised_claims(tmp_path) -> None:
    store = populated_store(tmp_path)
    context = ContextAssembler(store).build(
        "Is this the ceramic mug?", "one masked object is visible", ("object-001",),
        np.array([1.0, 0.0]),
    )
    assert len(context) <= store.config.context_max_characters
    assert "CURRENT SENSORY CONTEXT" in context
    assert "ceramic mug" in context
    assert "wrong label" not in context
    assert "vector_blob" not in context
    assert "relevance is not truth" in context
