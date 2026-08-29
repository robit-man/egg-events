from __future__ import annotations

import json

import numpy as np

from egg_companion.memory.retrieval import AssociativeRetriever
from egg_companion.memory.store import MemoryStore


class ContextAssembler:
    """Serializes only source-supported, bounded memory for cognition."""

    def __init__(
        self, store: MemoryStore, reflective_context_characters: int = 1800
    ) -> None:
        self.store = store
        self.retriever = AssociativeRetriever(store)
        self.config = store.config
        self.reflective_context_characters = max(
            400, min(int(reflective_context_characters), 8000)
        )
        self._last_hits = ()
        self._world_context = None

    def set_world_context(self, world_context: object) -> None:
        """Inject the CognitiveContext from the world model layer."""
        self._world_context = world_context

    def build(
        self, query: str, live_scene: str, entity_ids: tuple[str, ...] = (),
        query_embedding: np.ndarray | None = None,
        cognitive_state: dict[str, object] | None = None,
    ) -> str:
        hits = self.retriever.retrieve(query, entity_ids, query_embedding)
        self._last_hits = tuple(hits)
        records: list[dict[str, object]] = []
        for hit in hits:
            record: dict[str, object] = {
                "memory_type": hit.owner_type,
                "memory_id": hit.owner_id,
                "relevance": hit.score,
                "confidence": hit.confidence,
                "why_retrieved": list(hit.why),
                "evidence_ids": [item.evidence_id for item in hit.provenance],
            }
            if hit.owner_type == "entity":
                detail = self.store.entity_detail(hit.owner_id)
                if not detail:
                    continue
                entity = detail["entity"]
                record.update(
                    {
                        "entity_type": entity["entity_type"],
                        "name": entity.get("display_name"),
                        "state": entity["state"],
                        "claims": [
                            {
                                "predicate": claim["predicate"],
                                "value": claim["object_id_or_text"],
                                "confidence": claim["confidence"],
                                "source": claim.get("source"),
                                "evidence_id": claim.get("evidence_id"),
                            }
                            for claim in detail["claims"] if claim["state"] == "active"
                        ][:6],
                        "last_evidence_at": detail["evidence"][0]["captured_at"] if detail["evidence"] else None,
                    }
                )
            elif hit.owner_type == "episode":
                detail = self.store.episode_detail(hit.owner_id)
                if not detail:
                    continue
                episode = detail["episode"]
                record.update(
                    {
                        "started_at": episode["started_at"],
                        "ended_at": episode["ended_at"],
                        "summary": episode["summary"],
                        "entity_ids": [entity["entity_id"] for entity in detail["entities"]],
                        "evidence": [
                            {
                                "id": evidence["evidence_id"],
                                "modality": evidence["modality"],
                                "captured_at": evidence["captured_at"],
                                "payload": evidence["payload"],
                                "quality": evidence["quality"],
                            }
                            for evidence in detail["evidence"][:4]
                        ],
                    }
                )
            else:
                claim = next(
                    (item for item in self.store.list_claims(state="active", limit=self.config.graph_max_nodes)
                     if item["claim_id"] == hit.owner_id),
                    None,
                )
                if not claim:
                    continue
                record.update(
                    {
                        "subject_id": claim["subject_id"],
                        "predicate": claim["predicate"],
                        "value": claim["object_id_or_text"],
                        "source": claim.get("source"),
                    }
                )
            records.append(record)

        maximum = self.config.context_max_characters
        reflective_context = self.reflective_context(
            min(self.reflective_context_characters, max(200, maximum // 7))
        )
        world_state = self._build_world_state_section(
            max_characters=max(250, min(1000, maximum // 5))
        )

        def bounded(value: str, limit: int) -> str:
            if len(value) <= limit:
                return value
            return value[: max(0, limit - 14)] + "...[TRUNCATED]"

        sections = [
            (
                "CURRENT SENSORY CONTEXT (live, may be uncertain):\n"
                + bounded(str(live_scene), max(180, min(600, maximum // 8)))
            ),
            (
                "COGNITIVE CONTROL STATE (bounded attention/default-mode metadata; scores guide "
                "focus but are not facts):\n"
                + bounded(
                    json.dumps(
                        cognitive_state or {}, ensure_ascii=True, separators=(",", ":")
                    ),
                    max(180, min(600, maximum // 8)),
                )
            ),
        ]
        if world_state:
            sections.append(
                "CURRENT RECONCILED WORLD STATE (derived from evidence; use as grounded "
                "context for reasoning):\n" + world_state
            )
        sections.append(
            "REFLECTIVE WORKING MODEL (derived, revisable, not raw chain-of-thought; use it "
            "as strategy and context, never as stronger evidence than its sources):\n"
            + bounded(reflective_context, max(200, min(700, maximum // 7)))
        )

        # A context window is useful only if retrieval results actually reach
        # the model. Reserve a fixed share before serializing verbose control,
        # world-state, and reflective headers; previously those headers could
        # consume the entire deployment's 3k budget and silently erase recall.
        retrieval_label = (
            "RETRIEVED LOCAL MEMORY (use only explicit claims/evidence; relevance is not truth; "
            "omit unsupported details and state uncertainty):\n"
        )
        retrieval_reserve = max(450, min(1400, maximum // 3))
        header_budget = max(0, maximum - retrieval_reserve - len(retrieval_label) - 2)
        header = "\n\n".join(sections)
        header = bounded(header, header_budget) if header_budget else ""
        prefix = (header + "\n\n" if header else "") + retrieval_label
        body = json.dumps(records, ensure_ascii=True, separators=(",", ":"))
        if len(prefix) + len(body) > maximum:
            body = bounded(body, max(0, maximum - len(prefix)))
        return (prefix + body)[:maximum]

    def _build_world_state_section(self, max_characters: int = 1000) -> str:
        """Build a compact representation of the current world state."""
        if self._world_context is None:
            return ""
        try:
            from egg_companion.world.context import CognitiveContext
            if not isinstance(self._world_context, CognitiveContext):
                return ""
            window = self._world_context.build_window(
                max_characters=max_characters, max_entities=8
            )
            if not window.entities:
                return ""
            return self._world_context.serialize_for_llm(window)
        except Exception:
            return ""

    def reflective_context(self, maximum: int | None = None) -> str:
        maximum = max(
            200,
            min(
                int(maximum or self.reflective_context_characters),
                self.reflective_context_characters,
            ),
        )
        priorities = {
            "reflective-working-set": 0,
            "communication-strategy": 1,
            "world-model": 2,
            "my-story": 3,
        }
        documents = sorted(
            self.store.cognitive_documents(),
            key=lambda item: priorities.get(
                str(item.get("metadata", {}).get("document_kind")), 99
            ),
        )
        sections: list[str] = []
        for document in documents:
            metadata = document.get("metadata")
            if not isinstance(metadata, dict) or not metadata.get("content"):
                continue
            sections.append(
                f"[{metadata.get('document_kind')} r{metadata.get('revision', 0)}] "
                f"{str(metadata['content'])}"
            )
        content = "\n\n".join(sections) or "No consolidated cognitive document exists yet."
        return content[:maximum]

    def last_hits(self) -> list[dict[str, object]]:
        return [
            {
                "owner_type": hit.owner_type,
                "owner_id": hit.owner_id,
                "score": hit.score,
                "confidence": hit.confidence,
                "why": list(hit.why),
                "evidence_ids": [evidence.evidence_id for evidence in hit.provenance],
                "evidence_count": len(hit.provenance),
            }
            for hit in self._last_hits
        ]
