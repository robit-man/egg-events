from __future__ import annotations

from datetime import datetime, timezone

from egg_companion.config import PrivacyConfig
from egg_companion.memory.store import MemoryStore


class MemoryGovernance:
    """Local inspect, correction, export, and deletion controls without embedding serialization."""

    def __init__(self, store: MemoryStore, privacy: PrivacyConfig) -> None:
        self.store = store
        self.privacy = privacy

    def snapshot(self) -> dict[str, object]:
        return {
            "stats": self.store.memory_stats(),
            "entities": self.store.list_entities(limit=24),
            "episodes": self.store.recent_episodes(limit=12),
            "jobs": self.store.list_jobs(limit=12),
            "claim_conflicts": self.store.conflicting_claims(limit=12),
            "controls": {
                "export_enabled": self.privacy.export_enabled,
                "deletion_enabled": self.privacy.deletion_enabled,
            },
        }

    def inspect_entity(self, entity_id: str) -> dict[str, object] | None:
        return self.store.entity_detail(entity_id)

    def episodes(self) -> list[dict[str, object]]:
        return self.store.recent_episodes(limit=self.store.config.graph_max_nodes)

    def claims(self) -> list[dict[str, object]]:
        return self.store.list_claims(state=None, limit=self.store.config.graph_max_nodes)

    def add_alias(self, entity_id: str, alias: str) -> dict[str, object]:
        normalized = " ".join(alias.strip().split())
        detail = self.store.entity_detail(entity_id)
        if detail is None:
            raise KeyError(entity_id)
        if not normalized or len(normalized) > 64:
            raise ValueError("alias must contain 1-64 characters")
        now = datetime.now(timezone.utc)
        self.store.upsert_entity(
            detail["entity"]["entity_type"], normalized, detail["entity"].get("metadata"),
            entity_id, now=now,
        )
        claim_id = self.store.assert_claim_once(
            entity_id, "has_alias", normalized, 1.0, now, source="user"
        )
        return {"entity_id": entity_id, "alias": normalized, "claim_id": claim_id}

    def correct_claim(self, claim_id: str, replacement: str) -> dict[str, object]:
        normalized = " ".join(replacement.strip().split())
        claim = self.store.claim_detail(claim_id)
        if claim is None:
            raise KeyError(claim_id)
        if claim["state"] != "active":
            raise ValueError("only active claims can be corrected")
        if not normalized or len(normalized) > 128:
            raise ValueError("replacement must contain 1-128 characters")
        now = datetime.now(timezone.utc)
        revision_id = self.store.revise_claim(
            claim_id, "correct", "user", normalized, at=now
        )
        replacement_claim_id = self.store.assert_claim(
            str(claim["subject_id"]), str(claim["predicate"]), normalized, 1.0, now,
            source="user", metadata={"replaces_claim_id": claim_id},
        )
        return {
            "subject_id": claim["subject_id"],
            "predicate": claim["predicate"],
            "previous": claim["object_id_or_text"],
            "replacement": normalized,
            "revision_id": revision_id,
            "claim_id": replacement_claim_id,
        }

    def export(self) -> dict[str, object]:
        if not self.privacy.export_enabled:
            raise PermissionError("memory export is disabled")
        return {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "stats": self.store.memory_stats(),
            "entities": self.store.list_entities(limit=self.store.config.graph_max_nodes),
            "claims": self.store.list_claims(state=None, limit=self.store.config.graph_max_nodes),
            "episodes": self.store.recent_episodes(limit=self.store.config.graph_max_nodes),
            "embedding_metadata": self.store.embedding_metadata(limit=self.store.config.graph_max_nodes),
            "jobs": self.store.list_jobs(limit=self.store.config.graph_max_nodes),
        }

    def export_entity(self, entity_id: str) -> dict[str, object]:
        if not self.privacy.export_enabled:
            raise PermissionError("memory export is disabled")
        detail = self.store.entity_detail(entity_id)
        if detail is None:
            raise KeyError(entity_id)
        return {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "entity": detail,
        }

    def revise(
        self, target_type: str, target_id: str, decision: str, replacement: str | None = None
    ) -> dict[str, object]:
        if target_type == "claim":
            if decision == "correct":
                if replacement is None:
                    raise ValueError("a replacement is required for claim correction")
                return self.correct_claim(target_id, replacement)
            if decision not in {"reject", "retract"}:
                raise ValueError("claim decision must be correct, reject, or retract")
            if self.store.claim_detail(target_id) is None:
                raise KeyError(target_id)
            revision_id = self.store.revise_claim(target_id, decision, "user")
            return {"target_type": target_type, "target_id": target_id, "revision_id": revision_id}
        if target_type == "edge" and decision == "reject":
            revision_id = self.store.reject_edge(target_id)
            return {"target_type": target_type, "target_id": target_id, "revision_id": revision_id}
        raise ValueError("only claim correction/retraction and edge rejection are supported")

    def delete_entity(self, entity_id: str) -> None:
        if not self.privacy.deletion_enabled:
            raise PermissionError("memory deletion is disabled")
        if self.store.entity_detail(entity_id) is None:
            raise KeyError(entity_id)
        self.store.delete_entity_cascade(entity_id)
