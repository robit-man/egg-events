from __future__ import annotations

import hashlib

from egg_companion.config import DefaultModeConfig, EggConfig
from egg_companion.memory.entities import EntityResolver
from egg_companion.memory.context import ContextAssembler
from egg_companion.memory.consolidation import MemoryConsolidator
from egg_companion.memory.governance import MemoryGovernance
from egg_companion.memory.segmentation import EventSegmenter
from egg_companion.memory.store import MemoryStore
from egg_companion.cognition.default_mode import DefaultModeNetwork
from egg_companion.models import EpisodeDraft, EvidenceRef, PerceptualEvent


class MemoryPipeline:
    """Single-writer boundary between real-time sensing and durable local memory."""

    def __init__(self, config: EggConfig, store: MemoryStore) -> None:
        self.store = store
        self.entities = EntityResolver(store)
        self.context = ContextAssembler(store)
        self.consolidator = MemoryConsolidator(store, config.privacy)
        self.governance = MemoryGovernance(store, config.privacy)
        self.segmenter = EventSegmenter(config.memory, config.event_segmentation)
        self.default_mode = DefaultModeNetwork(
            store, getattr(config, "default_mode", DefaultModeConfig())
        )
        self.accepted_events = 0
        self.closed_episodes = 0

    def ingest(self, event: PerceptualEvent) -> tuple[bool, int]:
        accepted, drafts = self.segmenter.ingest(event)
        if accepted:
            self.accepted_events += 1
            self._persist_event(event)
        for draft in drafts:
            self._persist_episode(draft)
            self.closed_episodes += 1
        return accepted, len(drafts)

    def sync_object_profile(self, profile: dict[str, object]) -> str:
        thumbnail = profile.get("thumbnail")
        evidence_id = None
        if isinstance(thumbnail, bytes) and thumbnail:
            profile_id = str(profile["profile_id"])
            label = str(profile["label"])
            checksum = hashlib.sha256(thumbnail).hexdigest()
            label_key = hashlib.sha256(label.casefold().encode()).hexdigest()[:12]
            evidence_id = f"object-label:{profile_id}:{checksum[:16]}:{label_key}"
            media_key, persisted_checksum = self.store.persist_media(
                f"object-labels/{profile_id}/{checksum}.png", thumbnail
            )
            captured_at = self.entities._datetime(profile["last_seen"])
            evidence = EvidenceRef(
                evidence_id,
                "vision",
                captured_at,
                str(profile.get("label_source") or "object-library"),
                profile_id,
                media_key,
                self.entities._bounded_confidence(profile.get("label_confidence")),
                {
                    "label": label,
                    "label_source": profile.get("label_source"),
                    "review_state": profile.get("review_state"),
                    "label_provenance": profile.get("label_provenance")
                    if isinstance(profile.get("label_provenance"), dict)
                    else {},
                    "transparent_mask": True,
                    "automatic_classification": profile.get("label_source") == "ornith-vlm",
                },
            )
            self.store.append_evidence(evidence, checksum=persisted_checksum)
        entity_id = self.entities.sync_object_profile(profile, evidence_id)
        if evidence_id:
            self.store.link_entity_evidence(
                entity_id, evidence_id, "object-label-evidence"
            )
        return entity_id

    def sync_identity_profile(self, profile: dict[str, object]) -> str:
        return self.entities.sync_identity_profile(profile)

    def context_for(
        self,
        query: str,
        live_scene: str,
        entity_ids=(),
        query_embedding=None,
        cognitive_state: dict[str, object] | None = None,
    ) -> str:
        return self.context.build(
            query,
            live_scene,
            tuple(entity_ids),
            query_embedding,
            cognitive_state,
        )

    def graph_signals(self, entity_ids: list[str]):
        return self.store.cognitive_signals(entity_ids)

    def default_mode_pass(self) -> dict[str, object]:
        return self.default_mode.run_once()

    def conversation_history(self, limit: int = 5000) -> list[dict[str, object]]:
        return self.store.conversation_history(limit)

    def retrieval_snapshot(self) -> list[dict[str, object]]:
        return self.context.last_hits()

    def lifecycle_snapshot(self) -> dict[str, object]:
        return self.segmenter.snapshot()

    def consolidate(self) -> dict[str, object]:
        return self.consolidator.run_once()

    def stats(self) -> dict[str, int]:
        return self.store.memory_stats()

    def jobs(self) -> list[dict[str, object]]:
        return self.store.list_jobs(limit=20)

    def governance_snapshot(self) -> dict[str, object]:
        return self.governance.snapshot()

    def inspect_entity(self, entity_id: str) -> dict[str, object] | None:
        return self.governance.inspect_entity(entity_id)

    def episodes(self) -> list[dict[str, object]]:
        return self.governance.episodes()

    def claims(self) -> list[dict[str, object]]:
        return self.governance.claims()

    def add_alias(self, entity_id: str, alias: str) -> dict[str, object]:
        return self.governance.add_alias(entity_id, alias)

    def correct_claim(self, claim_id: str, replacement: str) -> dict[str, object]:
        return self.governance.correct_claim(claim_id, replacement)

    def export(self) -> dict[str, object]:
        return self.governance.export()

    def export_entity(self, entity_id: str) -> dict[str, object]:
        return self.governance.export_entity(entity_id)

    def revise(self, target_type: str, target_id: str, decision: str, replacement=None):
        return self.governance.revise(target_type, target_id, decision, replacement)

    def delete_entity(self, entity_id: str) -> None:
        self.governance.delete_entity(entity_id)

    def _persist_event(self, event: PerceptualEvent) -> None:
        confidences = self.entities.ensure_event_entities(event)
        self.store.open_episode(event.occurred_at, episode_id=event.event_id)
        for evidence in event.evidence:
            checksum = evidence.metadata.get("_media_checksum")
            self.store.append_evidence(
                evidence, checksum=str(checksum) if isinstance(checksum, str) else None
            )
            self.store.append_episode_evidence(event.event_id, evidence.evidence_id)
            for entity_id in event.entity_ids:
                if self.store.entity_detail(entity_id) is None:
                    continue
                self.store.link_entity_evidence(entity_id, evidence.evidence_id)
                self.store.link_episode_entity(
                    event.event_id, entity_id, "participant", confidences.get(entity_id, 0.0)
                )
        entity_ids = sorted(set(event.entity_ids))
        evidence_id = event.evidence[0].evidence_id if event.evidence else None
        if not event.payload.get("skip_pairwise_co_observation"):
            for index, source_id in enumerate(entity_ids):
                for target_id in entity_ids[index + 1 :]:
                    confidence = min(
                        confidences.get(source_id, 0.0), confidences.get(target_id, 0.0)
                    )
                    self.store.link_entities_once(
                        source_id, "co_observed_with", target_id, confidence,
                        event.occurred_at, {"source": event.source_id}, evidence_id,
                    )
        relations = event.payload.get("relations", ())
        if isinstance(relations, (list, tuple)):
            for relation in relations:
                if not isinstance(relation, dict):
                    continue
                source_id = relation.get("source_id")
                target_id = relation.get("target_id")
                predicate = relation.get("relation")
                if not all(
                    isinstance(value, str) and value
                    for value in (source_id, target_id, predicate)
                ):
                    continue
                if source_id not in entity_ids or target_id not in entity_ids:
                    continue
                try:
                    confidence = max(
                        0.0, min(1.0, float(relation.get("confidence") or 0.0))
                    )
                except (TypeError, ValueError):
                    confidence = 0.0
                metadata = relation.get("metadata")
                self.store.link_entities_once(
                    source_id, predicate, target_id, confidence, event.occurred_at,
                    metadata if isinstance(metadata, dict) else {"source": event.source_id},
                    evidence_id,
                )
        claims = event.payload.get("claims", ())
        if isinstance(claims, (list, tuple)):
            for claim in claims:
                if not isinstance(claim, dict):
                    continue
                subject_id = claim.get("subject_id")
                predicate = claim.get("predicate")
                value = claim.get("value")
                if not all(
                    isinstance(item, str) and item.strip()
                    for item in (subject_id, predicate, value)
                ):
                    continue
                if subject_id not in entity_ids:
                    continue
                try:
                    confidence = max(
                        0.0, min(1.0, float(claim.get("confidence") or 0.0))
                    )
                except (TypeError, ValueError):
                    confidence = 0.0
                self.store.assert_claim_once(
                    subject_id,
                    predicate,
                    value,
                    confidence,
                    event.occurred_at,
                    source=str(claim.get("source") or event.source_id),
                    evidence_id=evidence_id,
                    metadata=(
                        claim.get("metadata")
                        if isinstance(claim.get("metadata"), dict)
                        else {}
                    ),
                )
        identity_alias = event.payload.get("identity_alias")
        if isinstance(identity_alias, dict):
            alias_id = identity_alias.get("alias_id")
            canonical_id = identity_alias.get("canonical_id")
            if (
                isinstance(alias_id, str)
                and isinstance(canonical_id, str)
                and alias_id in entity_ids
                and canonical_id in entity_ids
                and alias_id != canonical_id
            ):
                self.store.coalesce_identity_evidence([identity_alias])

    def knowledge_graph_snapshot(self, node_limit: int = 1500) -> dict[str, object]:
        return self.store.knowledge_graph_snapshot(node_limit)

    def graph_node_detail(self, kind: str, source_id: str) -> dict[str, object] | None:
        return self.store.graph_node_detail(kind, source_id)

    def persist_media(self, relative_key: str, data: bytes) -> tuple[str, str]:
        return self.store.persist_media(relative_key, data)

    def evidence_media(self, evidence_id: str) -> tuple[bytes, str] | None:
        return self.store.evidence_media(evidence_id)

    def close(self, at) -> int:
        drafts = self.segmenter.flush(at)
        for draft in drafts:
            self._persist_episode(draft)
        self.closed_episodes += len(drafts)
        self.store.close()
        return len(drafts)

    def _persist_episode(self, draft: EpisodeDraft) -> None:
        self.store.open_episode(draft.started_at, max(draft.surprise.values(), default=0.0), draft.episode_id)
        for evidence in draft.evidence:
            self.store.append_evidence(evidence)
            self.store.append_episode_evidence(draft.episode_id, evidence.evidence_id)
            for entity_id in draft.entity_ids:
                if self.store.entity_detail(entity_id) is not None:
                    self.store.link_entity_evidence(entity_id, evidence.evidence_id)
                    self.store.link_episode_entity(draft.episode_id, entity_id)
        self.store.close_episode(draft.episode_id, draft.ended_at, draft.summary)
