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

        reflective_context = self.reflective_context(
            min(self.reflective_context_characters, max(400, self.config.context_max_characters // 3))
        )
        header = (
            "CURRENT SENSORY CONTEXT (live, may be uncertain):\n"
            f"{live_scene}\n\n"
            "COGNITIVE CONTROL STATE (bounded attention/default-mode metadata; scores guide "
            "focus but are not facts):\n"
            f"{json.dumps(cognitive_state or {}, ensure_ascii=True, separators=(',', ':'))}\n\n"
            "REFLECTIVE WORKING MODEL (derived, revisable, not raw chain-of-thought; "
            "use it as strategy and context, never as stronger evidence than its sources):\n"
            f"{reflective_context}\n\n"
            "RETRIEVED LOCAL MEMORY (use only explicit claims/evidence; relevance is not truth; "
            "omit unsupported details and state uncertainty):\n"
        )
        body = json.dumps(records, ensure_ascii=True, separators=(",", ":"))
        maximum = self.config.context_max_characters
        if len(header) >= maximum:
            return header[:maximum]
        if len(header) + len(body) > maximum:
            body = body[: max(0, maximum - len(header) - 18)] + "...[TRUNCATED]"
        return header + body

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
