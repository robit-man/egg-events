from datetime import datetime, timezone

import numpy as np
import pytest

from egg_companion.config import MemoryConfig, PrivacyConfig
from egg_companion.memory.governance import MemoryGovernance
from egg_companion.memory.store import MemoryStore


def test_governance_alias_correction_export_and_delete(tmp_path) -> None:
    store = MemoryStore(MemoryConfig(storage_dir=str(tmp_path / "memory"), graph_max_nodes=20))
    now = datetime.now(timezone.utc)
    store.upsert_entity("object", "cup", entity_id="object-001", now=now)
    claim_id = store.assert_claim("object-001", "has_label", "cup", 0.6, now, source="detector")
    store.add_embedding(
        "entity", "object-001", "masked-object", "open-clip", np.array([1.0, 0.0]), 0.8, now
    )
    governance = MemoryGovernance(store, PrivacyConfig())

    alias = governance.add_alias("object-001", "desk mug")
    correction = governance.correct_claim(claim_id, "ceramic mug")
    exported = governance.export()

    assert alias["alias"] == "desk mug"
    assert correction["replacement"] == "ceramic mug"
    assert store.claim_detail(claim_id)["state"] == "superseded"
    assert "vector_blob" not in str(exported)
    assert exported["embedding_metadata"][0]["dimensions"] == 2
    governance.delete_entity("object-001")
    assert governance.inspect_entity("object-001") is None


def test_governance_respects_export_and_deletion_controls(tmp_path) -> None:
    store = MemoryStore(MemoryConfig(storage_dir=str(tmp_path / "memory")))
    store.upsert_entity("person", entity_id="person-001")
    governance = MemoryGovernance(
        store, PrivacyConfig(export_enabled=False, deletion_enabled=False)
    )
    with pytest.raises(PermissionError):
        governance.export()
    with pytest.raises(PermissionError):
        governance.delete_entity("person-001")
