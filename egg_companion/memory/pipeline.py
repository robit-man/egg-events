from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from uuid import uuid4

from egg_companion.config import DefaultModeConfig, EggConfig
from egg_companion.memory.entities import EntityResolver
from egg_companion.memory.context import ContextAssembler
from egg_companion.memory.consolidation import MemoryConsolidator
from egg_companion.memory.governance import MemoryGovernance
from egg_companion.memory.segmentation import EventSegmenter
from egg_companion.memory.store import MemoryStore
from egg_companion.cognition.default_mode import DefaultModeNetwork
from egg_companion.models import EpisodeDraft, EvidenceRef, PerceptualEvent
from egg_companion.world.policy import PolicyViolation
from egg_companion.world.types import ActionExecution, ActionOutcome, ActionProposal


class MemoryPipeline:
    """Single-writer boundary between real-time sensing and durable local memory."""

    def __init__(self, config: EggConfig, store: MemoryStore) -> None:
        self.store = store
        self.entities = EntityResolver(store)
        default_mode_config = getattr(config, "default_mode", DefaultModeConfig())
        self.context = ContextAssembler(
            store, default_mode_config.document_context_characters
        )
        self.consolidator = MemoryConsolidator(store, config.privacy)
        self.governance = MemoryGovernance(store, config.privacy)
        self.segmenter = EventSegmenter(config.memory, config.event_segmentation)
        self.default_mode = DefaultModeNetwork(
            store, default_mode_config
        )
        self._environmental_reflection_characters = getattr(
            getattr(config, "environmental_cognition", None),
            "reflection_characters",
            900,
        )
        self.store.refresh_model_narrative_documents()
        self.accepted_events = 0
        self.closed_episodes = 0
        self._observation_policy_cache: dict[str, object] | None = None
        self._observation_policy_cached_at = 0.0
        
        # World model integration (eager initialization)
        self._world_bridge = None
        self._world_reconciler = None
        self._world_normalizer = None
        self._world_query = None
        self._world_context = None
        self._action_store = None
        self._policy_validator = None
        self._ensure_world_model()

    @property
    def world_query(self) -> object | None:
        """Expose WorldQuery for external wiring (e.g. attention controller)."""
        return self._world_query

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
                    "appearance_description": profile.get(
                        "appearance_description"
                    ),
                    "adjudication_history": profile.get("adjudication_history")
                    if isinstance(profile.get("adjudication_history"), list)
                    else [],
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

    def observation_policy(self, maximum_age_seconds: float = 15.0) -> dict[str, object]:
        now = time.monotonic()
        if (
            self._observation_policy_cache is None
            or now - self._observation_policy_cached_at >= maximum_age_seconds
        ):
            self._observation_policy_cache = self.store.observational_policy()
            self._observation_policy_cached_at = now
        return dict(self._observation_policy_cache)

    def interaction_strategy(self) -> dict[str, object]:
        entity = self.store.entity_metadata("interaction-strategy:current")
        if not isinstance(entity, dict):
            return {}
        metadata = entity.get("metadata")
        return dict(metadata) if isinstance(metadata, dict) else {}

    def social_profiles(self, entity_ids: list[str]) -> list[dict[str, object]]:
        """Return revisable interaction profiles grounded to presently relevant people."""
        profiles: list[dict[str, object]] = []
        for entity_id in dict.fromkeys(str(value) for value in entity_ids):
            entity = self.store.entity_metadata(f"social-profile:{entity_id}")
            if not isinstance(entity, dict):
                continue
            metadata = entity.get("metadata")
            if isinstance(metadata, dict):
                profiles.append(
                    {
                        "profile_id": f"social-profile:{entity_id}",
                        "subject_id": entity_id,
                        **dict(metadata),
                    }
                )
        return profiles

    def pending_narrative_semantics(self) -> dict[str, object] | None:
        records = self.store.pending_narrative_semantics(1)
        return records[0] if records else None

    def narrative_constitution(self) -> dict[str, object]:
        return self.store.narrative_constitution()

    def apply_narrative_semantics(
        self,
        local_date: str,
        input_fingerprint: str,
        semantics: dict[str, object],
        policy: dict[str, object],
        constitution_text: str | None,
        model_id: str,
        tool_audit: list[dict[str, object]],
    ) -> bool:
        from datetime import datetime, timezone

        applied = self.store.apply_narrative_semantics(
            local_date,
            input_fingerprint,
            semantics,
            policy,
            constitution_text,
            model_id,
            tool_audit,
            datetime.now(timezone.utc),
        )
        if applied:
            self.default_mode.world_model.apply_model_semantics(
                local_date, semantics, datetime.now(timezone.utc)
            )
            self._observation_policy_cache = None
            self._observation_policy_cached_at = 0.0
        return applied

    def narrative_memory_search(self, query: str) -> str:
        return self.context_for(query, "dream semantic replay")

    def narrative_graph_inspect(self, entity_ids: list[str]) -> list[dict[str, object]]:
        return [
            detail
            for entity_id in entity_ids[:12]
            for detail in [self.store.entity_detail(str(entity_id))]
            if detail is not None
        ]

    def narrative_evidence_inspect(
        self, evidence_ids: list[str]
    ) -> list[dict[str, object]]:
        return [
            detail
            for evidence_id in evidence_ids[:12]
            for detail in [self.store.evidence_detail(str(evidence_id))]
            if detail is not None
        ]

    def default_mode_pass(self) -> dict[str, object]:
        return self.default_mode.run_once()

    def dream_narrative_pass(
        self, dream_result: dict[str, object]
    ) -> dict[str, object]:
        return self.default_mode.world_model.replay_dream(dream_result)

    def narrative_backfill_pass(
        self, requested_by: str = "startup"
    ) -> dict[str, object]:
        """Replay never-narrated history without waiting for a face dream."""
        return self.default_mode.world_model.replay_dream(
            {
                "run_id": f"narrative-catchup-{uuid4()}",
                "requested_by": requested_by,
                "profiles_examined": 0,
                "merges": 0,
                "aliases": [],
            }
        )

    def daily_narratives(self, limit: int = 90) -> list[dict[str, object]]:
        return self.store.daily_narrative_index(limit)

    def daily_narrative(self, local_date: str) -> dict[str, object] | None:
        return self.store.daily_narrative_detail(local_date)

    def conversation_history(self, limit: int = 5000) -> list[dict[str, object]]:
        return self.store.conversation_history(limit)

    def retrieval_snapshot(self) -> list[dict[str, object]]:
        return self.context.last_hits()

    def reflective_context(self, maximum: int | None = None) -> str:
        return self.context.reflective_context(maximum)

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
        
        # World model integration: create WorldDelta and apply to world state
        # This happens AFTER memory persistence so raw evidence survives if world model fails
        self._apply_world_delta(event, confidences)
        if event.event_type == "environmental_reflection":
            self._update_environmental_working_set(event)

    def _update_environmental_working_set(self, event: PerceptualEvent) -> None:
        """Fold inspectable event-grounded thoughts into later recall/dream context."""

        reflection = event.payload.get("environmental_reflection")
        if not isinstance(reflection, dict):
            return
        summary = " ".join(str(reflection.get("reflection") or "").split())
        if not summary:
            return
        connections = [
            " ".join(str(item).split())
            for item in reflection.get("connections", [])
            if isinstance(item, str) and item.strip()
        ][:6]
        questions = [
            " ".join(str(item).split())
            for item in reflection.get("open_questions", [])
            if isinstance(item, str) and item.strip()
        ][:6]
        entry = f"[{event.occurred_at.isoformat()}] {summary}"
        if connections:
            entry += " Connections: " + "; ".join(connections)
        if questions:
            entry += " Open questions: " + "; ".join(questions)

        document_id = "cognitive-document:environmental-working-set"
        existing = self.store.entity_metadata(document_id)
        metadata = existing.get("metadata") if isinstance(existing, dict) else {}
        prior_content = (
            str(metadata.get("content") or "")
            if isinstance(metadata, dict)
            else ""
        )
        lines = [entry, *prior_content.splitlines()]
        content = "\n".join(dict.fromkeys(line for line in lines if line.strip()))
        content = content[: int(self._environmental_reflection_characters)]
        prior_sources = (
            list(metadata.get("source_entity_ids", []))
            if isinstance(metadata, dict)
            and isinstance(metadata.get("source_entity_ids"), list)
            else []
        )
        source_ids = list(
            dict.fromkeys(
                [
                    *event.entity_ids,
                    *prior_sources,
                ]
            )
        )[:100]
        self.store.upsert_cognitive_document(
            "environmental-working-set",
            "Environmental working set",
            content,
            float(reflection.get("confidence") or 0.0),
            source_ids,
            event.occurred_at,
        )

    def knowledge_graph_snapshot(self, node_limit: int = 1500) -> dict[str, object]:
        return self.store.knowledge_graph_snapshot(node_limit)

    def _apply_world_delta(self, event: PerceptualEvent, confidences: dict[str, float]) -> None:
        """Create WorldDelta from PerceptualEvent and apply to world state.
        
        This runs AFTER memory persistence so raw evidence survives if world model fails.
        """
        try:
            from egg_companion.world.normalize import ObservationNormalizer
            
            if self._world_normalizer is None:
                self._world_normalizer = ObservationNormalizer()
            
            evidence_ids = tuple(e.evidence_id for e in event.evidence)
            
            frame_shape = event.payload.get("frame_shape")
            
            delta = self._world_normalizer.normalize_event(
                event,
                evidence_ids=evidence_ids,
                confidences=confidences,
                frame_shape=frame_shape,
            )
            
            if delta.assertions or delta.relation_assertions or delta.events:
                self._ensure_world_model()
                if self._world_reconciler is not None:
                    conflicts = self._world_reconciler.ingest(delta)
                    if conflicts:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.debug(f"World model conflicts: {len(conflicts)}")
        
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"World model delta application failed: {e}")

    def _ensure_world_model(self) -> None:
        """Lazy initialization of world model components."""
        if self._world_reconciler is not None:
            return
        
        try:
            import sqlite3
            from egg_companion.world.reconcile import Reconciler
            from egg_companion.world.state import WorldStateStore
            from egg_companion.world.query import WorldQuery
            from egg_companion.world.context import CognitiveContext
            from egg_companion.world.relations import WorldGraphStore
            from egg_companion.world.identity import IdentityGraph
            from egg_companion.world.ontology import OntologyRegistry
            from egg_companion.world.actions import ActionStore
            from egg_companion.world.policy import PolicyValidator
            
            # Create world model connection (separate from memory store)
            world_db_path = self.store._read_connection.execute(
                "PRAGMA database_list"
            ).fetchone()
            
            # Use the same database file but separate connection for world model
            db_path = str(self.store.root / "memory.sqlite3") if hasattr(self.store, 'root') else ":memory:"
            
            world_conn = sqlite3.connect(db_path, check_same_thread=False)
            world_conn.execute("PRAGMA journal_mode=WAL")
            world_conn.execute("PRAGMA busy_timeout=5000")
            world_conn.execute("PRAGMA foreign_keys=ON")
            
            world_state = WorldStateStore(world_conn)
            ontology = OntologyRegistry(world_conn)
            self._world_reconciler = Reconciler(world_conn, world_state, ontology)
            
            world_graph = WorldGraphStore(world_conn)
            identity_graph = IdentityGraph(world_conn)
            self._world_query = WorldQuery(world_state, world_graph, identity_graph, reconciler=self._world_reconciler)
            self._world_context = CognitiveContext(self._world_query)
            self.context.set_world_context(self._world_context)

            self._action_store = ActionStore(world_conn)
            self._policy_validator = PolicyValidator(world_conn)
            
            import logging
            logger = logging.getLogger(__name__)
            logger.info("World model initialized for pipeline integration")
        
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to initialize world model: {e}")
            self._world_reconciler = None

    def propose_action(
        self,
        action_type: str,
        *,
        inputs: dict[str, object] | None = None,
        target_entity_ids: tuple[str, ...] = (),
        source_evidence_ids: tuple[str, ...] = (),
    ) -> tuple[ActionProposal, list[PolicyViolation]]:
        """Create, persist, and policy-check an ActionProposal.

        Callers must not act on the proposed effect if any returned
        violation has ``blocked=True`` — the proposal is recorded either
        way (accepted proposals feed policy_action_log for frequency
        limiting; rejected ones stay auditable).  If the world model isn't
        available, this fails open with no violations, since without a
        durable ledger there's nothing to check policy against — see
        record_action_execution for the same failure mode.
        """
        self._ensure_world_model()
        proposal = ActionProposal(
            proposal_id=f"proposal:{uuid4().hex[:16]}",
            action_type=action_type,
            target_entity_ids=tuple(target_entity_ids),
            inputs=dict(inputs or {}),
            source_evidence_ids=tuple(source_evidence_ids),
        )
        if self._action_store is None or self._policy_validator is None:
            return proposal, []

        self._action_store.propose(proposal)
        violations = self._policy_validator.validate(proposal)
        for violation in violations:
            self._policy_validator.log_violation(violation)
        blocked = any(v.blocked for v in violations)
        if not blocked:
            self._policy_validator.record_action(action_type, proposal.proposal_id)
        return proposal, violations

    def record_action_execution(
        self,
        proposal_id: str,
        *,
        success: bool,
        result: object = None,
        evidence_ids: tuple[str, ...] = (),
    ) -> None:
        """Record that a proposed action actually ran, and its outcome."""
        if self._action_store is None:
            return
        now = datetime.now(timezone.utc)
        execution_id = f"execution:{uuid4().hex[:16]}"
        self._action_store.record_execution(ActionExecution(
            execution_id=execution_id,
            proposal_id=proposal_id,
            started_at=now,
            completed_at=now,
            success=success,
            result=result,
        ))
        self._action_store.record_outcome(ActionOutcome(
            outcome_id=f"outcome:{uuid4().hex[:16]}",
            execution_id=execution_id,
            success=success,
            result=result,
            evidence_ids=tuple(evidence_ids),
            observed_at=now,
        ))

    def record_derived_property(
        self,
        entity_id: str,
        property_id: str,
        value: str,
        *,
        confidence: float = 0.6,
        authority: float = 0.5,
        source_id: str = "derived",
    ) -> None:
        """Record a single computed/derived fact directly into the world
        model, bypassing the raw-detection normalizer.

        _normalize_visual_event's "detections" contract is for perception
        output (label/bbox/behavior/gaze/...); a value computed from other
        world state (like an occupancy-grid summary) isn't a detection and
        doesn't belong wedged into that contract. This is a small, generic
        primitive for "the world model should know this fact now."
        """
        self._ensure_world_model()
        if self._world_reconciler is None:
            return
        from egg_companion.world.types import EpistemicKind, TypedValue, ValueType, WorldDelta

        delta = WorldDelta()
        delta.assertions.append({
            "subject_id": entity_id,
            "property_id": property_id,
            "value": TypedValue(raw=value, value_type=ValueType.STRING),
            "epistemic_kind": EpistemicKind.INFERENCE.value,
            "source_id": source_id,
            "evidence_ids": (),
            "confidence": confidence,
            "authority": authority,
            "valid_from": datetime.now(timezone.utc).isoformat(),
        })
        self._world_reconciler.ingest(delta)

    def backfill_world_from_evidence(self, batch_size: int = 500, max_items: int = 0) -> dict[str, object]:
        """Retroactively populate world model from existing evidence store.

        Reads vision evidence, deduplicates by entity, and bulk-ingests
        into the world model in large batches to avoid per-item transaction overhead.
        Returns progress stats.
        """
        import json
        import logging
        import time
        from collections import defaultdict
        from dataclasses import dataclass
        from datetime import datetime, timezone

        logger = logging.getLogger(__name__)

        self._ensure_world_model()
        if self._world_reconciler is None:
            return {"error": "world model not initialized", "processed": 0}
        
        if self._world_normalizer is None:
            from egg_companion.world.normalize import ObservationNormalizer
            self._world_normalizer = ObservationNormalizer()

        db_path = self.store.root / "memory.sqlite3" if hasattr(self.store, 'root') else None
        if not db_path:
            return {"error": "no database path", "processed": 0}

        import sqlite3
        read_conn = sqlite3.connect(str(db_path), check_same_thread=False)
        read_conn.row_factory = sqlite3.Row

        # Get resume marker
        marker_row = self.store._read_connection.execute(
            "SELECT metadata_json FROM entities WHERE entity_id='world-backfill-marker'"
        ).fetchone()
        already_done = False
        if marker_row:
            try:
                meta = json.loads(marker_row[0])
                if meta.get("last_evidence_id") == "completed":
                    already_done = True
            except Exception:
                pass

        if already_done:
            logger.info("World backfill already completed, skipping")
            return {"processed": 0, "entities": 0, "relations": 0, "status": "already completed"}

        # Count total
        total = read_conn.execute(
            "SELECT COUNT(*) FROM evidence WHERE modality='vision' "
            "AND json_extract(payload_json,'$.detections') IS NOT NULL"
        ).fetchone()[0]

        logger.info("World backfill: %d vision evidence items", total)

        # Step 1: Aggregate unique entity observations across all evidence
        # Group by entity_id → pick the observation with highest confidence
        entity_best: dict[str, dict] = {}
        entity_seen_count: dict[str, int] = defaultdict(int)
        relation_set: set[tuple[str, str, str]] = set()

        query = (
            "SELECT evidence_id, captured_at, source_id, payload_json FROM evidence "
            "WHERE modality='vision' "
            "AND json_extract(payload_json,'$.detections') IS NOT NULL "
            "ORDER BY evidence_id"
        )
        cursor = read_conn.execute(query)

        processed = 0
        skipped = 0
        start_time = time.monotonic()

        for row in cursor:
            evidence_id = row[0]
            captured_at = row[1]
            source_id = row[2] or "unknown"
            payload = json.loads(row[3]) if row[3] else {}

            detections = payload.get("detections", [])
            if not isinstance(detections, list) or not detections:
                skipped += 1
                continue

            frame_shape = None
            if "frame_shape" in payload:
                fs = payload["frame_shape"]
                if isinstance(fs, (list, tuple)) and len(fs) >= 2:
                    frame_shape = (int(fs[0]), int(fs[1]))

            occurred = None
            if captured_at:
                try:
                    occurred = datetime.fromisoformat(str(captured_at).replace('Z', '+00:00')) if isinstance(captured_at, str) else captured_at
                except Exception:
                    occurred = datetime.now(timezone.utc)
            else:
                occurred = datetime.now(timezone.utc)

            for det in detections:
                if not isinstance(det, dict):
                    continue
                entity_id = det.get("entity_id") or det.get("object_id") or det.get("identity_id")
                if not entity_id:
                    # Use label-based entity for tracked detections
                    label = det.get("label", "")
                    if label:
                        entity_id = f"det:{label}"
                    else:
                        continue

                entity_seen_count[entity_id] += 1
                confidence = float(det.get("confidence", 0.0))

                # Keep highest-confidence observation per entity
                if entity_id not in entity_best or confidence > entity_best[entity_id]["confidence"]:
                    camera_id = source_id.split(":")[-1] if ":" in source_id else source_id
                    entity_best[entity_id] = {
                        "label": det.get("label", "unknown"),
                        "confidence": confidence,
                        "bbox": det.get("bbox"),
                        "behavior": det.get("behavior"),
                        "source_id": source_id,
                        "camera_id": camera_id,
                        "valid_from": occurred.isoformat() if isinstance(occurred, datetime) else str(occurred),
                        "evidence_id": evidence_id,
                        "frame_shape": frame_shape,
                        "first_seen": occurred,
                    }

                # Collect relation: entity → camera
                camera_id = source_id.split(":")[-1] if ":" in source_id else source_id
                relation_set.add((entity_id, "visible_from", f"camera_view:{camera_id}"))

            processed += 1
            if processed % 5000 == 0:
                logger.info("World backfill scan: %d/%d evidence, %d unique entities so far", processed, total, len(entity_best))

        read_conn.close()

        elapsed_scan = time.monotonic() - start_time
        logger.info(
            "World backfill scan complete: %d evidence, %d unique entities, %d relations (%.1fs)",
            processed, len(entity_best), len(relation_set), elapsed_scan,
        )

        # Step 2: Bulk ingest all entities in one big transaction
        if not entity_best:
            return {"processed": processed, "entities": 0, "relations": 0, "skipped": skipped}

        @dataclass
        class _Event:
            source_id: str = "unknown"
            event_type: str = "vision"
            payload: dict = None
            entity_ids: tuple = ()
            occurred_at: datetime = None
            evidence: tuple = ()

        from egg_companion.world.types import WorldDelta

        bulk_delta = WorldDelta()
        now_iso = datetime.now(timezone.utc).isoformat()

        for entity_id, info in entity_best.items():
            label = info["label"]
            confidence = info["confidence"]
            bbox = info["bbox"]
            behavior = info["behavior"]
            source_id = info["source_id"]
            camera_id = info["camera_id"]
            valid_from = info["valid_from"]
            evidence_id = info["evidence_id"]
            frame_shape = info["frame_shape"]
            seen_count = entity_seen_count[entity_id]

            from egg_companion.world.types import TypedValue, ValueType, EpistemicKind, ObservabilityState

            bulk_delta.assertions.append({
                "subject_id": entity_id,
                "property_id": "label",
                "value": TypedValue(raw=label, value_type=ValueType.STRING),
                "epistemic_kind": EpistemicKind.OBSERVATION.value,
                "source_id": source_id,
                "evidence_ids": (evidence_id,),
                "confidence": confidence,
                "authority": 0.7,
                "valid_from": valid_from,
            })

            if bbox:
                if isinstance(bbox, dict):
                    bbox_list = [bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)]
                else:
                    bbox_list = list(bbox)
                bulk_delta.assertions.append({
                    "subject_id": entity_id,
                    "property_id": "bbox",
                    "value": TypedValue(raw=bbox_list, value_type=ValueType.GEOMETRY),
                    "epistemic_kind": EpistemicKind.OBSERVATION.value,
                    "source_id": source_id,
                    "evidence_ids": (evidence_id,),
                    "confidence": confidence,
                    "authority": 0.7,
                    "valid_from": valid_from,
                })

            if behavior:
                bulk_delta.assertions.append({
                    "subject_id": entity_id,
                    "property_id": "behavior",
                    "value": TypedValue(raw=behavior, value_type=ValueType.STRING),
                    "epistemic_kind": EpistemicKind.OBSERVATION.value,
                    "source_id": source_id,
                    "evidence_ids": (evidence_id,),
                    "confidence": confidence,
                    "authority": 0.7,
                    "valid_from": valid_from,
                })

            bulk_delta.assertions.append({
                "subject_id": entity_id,
                "property_id": "last_seen",
                "value": TypedValue(raw=valid_from, value_type=ValueType.DATETIME),
                "epistemic_kind": EpistemicKind.OBSERVATION.value,
                "source_id": source_id,
                "evidence_ids": (evidence_id,),
                "confidence": confidence,
                "authority": 0.9,
                "valid_from": valid_from,
            })

            bulk_delta.assertions.append({
                "subject_id": entity_id,
                "property_id": "observability",
                "value": TypedValue(raw=ObservabilityState.OBSERVED_PRESENT.value, value_type=ValueType.ENUM),
                "epistemic_kind": EpistemicKind.OBSERVATION.value,
                "source_id": source_id,
                "evidence_ids": (evidence_id,),
                "confidence": confidence,
                "authority": 0.9,
                "valid_from": valid_from,
            })

            bulk_delta.assertions.append({
                "subject_id": entity_id,
                "property_id": "observation_count",
                "value": TypedValue(raw=seen_count, value_type=ValueType.INTEGER),
                "epistemic_kind": EpistemicKind.OBSERVATION.value,
                "source_id": source_id,
                "evidence_ids": (evidence_id,),
                "confidence": min(1.0, 0.5 + seen_count * 0.05),
                "authority": 0.9,
                "valid_from": valid_from,
            })

            if frame_shape and bbox:
                h, w = frame_shape
                try:
                    bx = bbox_list if isinstance(bbox_list, list) else [bbox.get("x1",0), bbox.get("y1",0), bbox.get("x2",0), bbox.get("y2",0)]
                    center_x = ((bx[0] + bx[2]) / 2) / w
                    center_y = ((bx[1] + bx[3]) / 2) / h
                    bulk_delta.assertions.append({
                        "subject_id": entity_id,
                        "property_id": "current_location",
                        "value": TypedValue(
                            raw={"frame": f"{camera_id}_normalized", "position": [round(center_x, 4), round(center_y, 4)]},
                            value_type=ValueType.GEOMETRY,
                        ),
                        "epistemic_kind": EpistemicKind.OBSERVATION.value,
                        "source_id": source_id,
                        "evidence_ids": (evidence_id,),
                        "confidence": confidence * 0.8,
                        "authority": 0.7,
                        "valid_from": valid_from,
                    })
                except Exception:
                    pass

            bulk_delta.relation_assertions.append({
                "source_entity_id": entity_id,
                "relation_type_id": "visible_from",
                "target_entity_id": f"camera_view:{camera_id}",
                "confidence": confidence,
                "authority": 0.7,
                "source_id": source_id,
                "evidence_ids": (evidence_id,),
                "valid_from": valid_from,
            })

        logger.info(
            "World backfill ingesting %d assertions + %d relations for %d entities",
            len(bulk_delta.assertions), len(bulk_delta.relation_assertions), len(entity_best),
        )

        ingest_start = time.monotonic()
        try:
            conflicts = self._world_reconciler.ingest(bulk_delta)
        except Exception as e:
            logger.error("World backfill ingest failed: %s", e)
            return {"processed": processed, "entities": 0, "error": str(e)}

        ingest_elapsed = time.monotonic() - ingest_start
        total_elapsed = time.monotonic() - start_time

        self._save_backfill_marker("completed")

        stats = {
            "processed": processed,
            "entities": len(entity_best),
            "relations": len(relation_set),
            "assertions": len(bulk_delta.assertions),
            "conflicts": len(conflicts),
            "skipped": skipped,
            "scan_seconds": round(elapsed_scan, 2),
            "ingest_seconds": round(ingest_elapsed, 2),
            "total_seconds": round(total_elapsed, 2),
        }
        logger.info("World backfill complete: %s", stats)
        return stats

    def _save_backfill_marker(self, evidence_id: str) -> None:
        """Save backfill progress marker to allow resume."""
        import json
        from datetime import datetime, timezone
        try:
            self.store.upsert_entity(
                "system",
                "world-backfill-marker",
                {"last_evidence_id": evidence_id, "updated_at": datetime.now(timezone.utc).isoformat()},
                entity_id="world-backfill-marker",
            )
        except Exception:
            pass

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
