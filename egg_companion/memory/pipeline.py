from __future__ import annotations

from egg_companion.config import EggConfig
from egg_companion.memory.entities import EntityResolver
from egg_companion.memory.context import ContextAssembler
from egg_companion.memory.consolidation import MemoryConsolidator
from egg_companion.memory.governance import MemoryGovernance
from egg_companion.memory.segmentation import EventSegmenter
from egg_companion.memory.store import MemoryStore
from egg_companion.models import EpisodeDraft, PerceptualEvent


class MemoryPipeline:
    """Single-writer boundary between real-time sensing and durable local memory."""

    def __init__(self, config: EggConfig, store: MemoryStore) -> None:
        self.store = store
        self.entities = EntityResolver(store)
        self.context = ContextAssembler(store)
        self.consolidator = MemoryConsolidator(store, config.privacy)
        self.governance = MemoryGovernance(store, config.privacy)
        self.segmenter = EventSegmenter(config.memory, config.event_segmentation)
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
        return self.entities.sync_object_profile(profile)

    def sync_identity_profile(self, profile: dict[str, object]) -> str:
        return self.entities.sync_identity_profile(profile)

    def context_for(self, query: str, live_scene: str, entity_ids=(), query_embedding=None) -> str:
        return self.context.build(query, live_scene, tuple(entity_ids), query_embedding)

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
            self.store.append_evidence(evidence)
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
        for index, source_id in enumerate(entity_ids):
            for target_id in entity_ids[index + 1 :]:
                confidence = min(confidences.get(source_id, 0.0), confidences.get(target_id, 0.0))
                self.store.link_entities_once(
                    source_id, "co_observed_with", target_id, confidence, event.occurred_at,
                    {"source": event.source_id}, evidence_id,
                )

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
