from __future__ import annotations

import re
from datetime import datetime, timezone

import numpy as np

from egg_companion.memory.store import MemoryStore
from egg_companion.models import EvidenceRef, MemoryHit


_STOPWORDS = {
    "about", "after", "again", "also", "been", "could", "from", "have", "into", "just",
    "like", "some", "that", "their", "there", "these", "they", "this", "what", "when",
    "where", "which", "with", "would", "your",
}


class AssociativeRetriever:
    """Bounded multimodal candidate generation and explainable reranking."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self.config = store.config

    def retrieve(
        self, query: str, entity_ids: tuple[str, ...] = (), query_embedding: np.ndarray | None = None
    ) -> list[MemoryHit]:
        now = datetime.now(timezone.utc)
        terms = self._terms(query)
        candidates: dict[tuple[str, str], dict[str, object]] = {}
        for entity_id in entity_ids:
            if self.store.entity_detail(entity_id):
                self._add(candidates, "entity", entity_id, 0.92, 1.0, "present in live scene")

        for entity in self.store.list_entities(limit=self.config.graph_max_nodes):
            searchable = " ".join(
                str(value) for value in (entity.get("display_name"), entity.get("metadata")) if value
            ).casefold()
            overlap = sum(term in searchable for term in terms)
            if overlap:
                self._add(
                    candidates, "entity", str(entity["entity_id"]),
                    min(0.88, 0.48 + 0.12 * overlap),
                    float(entity.get("metadata", {}).get("confidence", 0.5)),
                    f"entity matched {overlap} transcript term(s)",
                )

        for evidence in self.store.search_evidence(terms, self.config.retrieval_limit * 2):
            recency = self._recency(evidence["captured_at"], now)
            quality = float(evidence.get("quality") or 0.0)
            score = 0.38 + 0.24 * quality + 0.20 * recency
            provenance = self._evidence_ref(evidence)
            associations = self.store.associations_for_evidence(str(evidence["evidence_id"]))
            for entity in associations["entities"]:
                self._add(
                    candidates, "entity", str(entity["entity_id"]), score, quality,
                    "linked transcript-matching evidence", provenance,
                )
            for episode in associations["episodes"]:
                self._add(
                    candidates, "episode", str(episode["episode_id"]), score * 0.94, quality,
                    "episode contains transcript-matching evidence", provenance,
                )

        if query_embedding is not None:
            query_vector = self._normalized(query_embedding)
            for record in self.store.embedding_records(limit=self.config.graph_max_nodes):
                vector = record["vector"]
                if vector.shape != query_vector.shape or not str(record["model_id"]).startswith("open-"):
                    continue
                similarity = float(np.dot(query_vector, self._normalized(vector)))
                if similarity < 0.18:
                    continue
                score = min(0.90, 0.30 + max(0.0, similarity) * 0.55 + float(record["quality"]) * 0.10)
                self._add(
                    candidates, str(record["owner_type"]), str(record["owner_id"]), score,
                    float(record["quality"]),
                    f"CLIP text-to-{record['modality']} similarity {similarity:.3f}",
                )

        seed_entities = [owner_id for owner_type, owner_id in candidates if owner_type == "entity"]
        for edge in self.store.graph_neighbors(seed_entities[: self.config.retrieval_limit]):
            source_id, target_id = str(edge["source_id"]), str(edge["target_id"])
            neighbor = target_id if source_id in seed_entities else source_id
            score = 0.42 * float(edge["confidence"]) / int(edge["hop"])
            self._add(
                candidates, "entity", neighbor, score, float(edge["confidence"]),
                f"{edge['relation']} graph path at hop {edge['hop']}",
            )

        for claim in self.store.list_claims(state="active", limit=self.config.graph_max_nodes):
            text = f"{claim['predicate']} {claim['object_id_or_text']}".casefold()
            overlap = sum(term in text for term in terms)
            if not overlap:
                continue
            source_bonus = 0.12 if claim.get("source") in {"user", "ornith-vlm"} else 0.0
            self._add(
                candidates, "claim", str(claim["claim_id"]),
                min(0.96, 0.50 + overlap * 0.12 + source_bonus), float(claim["confidence"]),
                f"active {claim.get('source', 'system')} claim matched transcript",
            )

        if not candidates:
            for episode in self.store.recent_episodes(limit=min(3, self.config.retrieval_limit)):
                self._add(
                    candidates, "episode", str(episode["episode_id"]),
                    0.18 * self._recency(episode["started_at"], now), float(episode.get("novelty") or 0.0),
                    "recent episodic context",
                )

        hits = [
            MemoryHit(
                owner_type=owner_type,
                owner_id=owner_id,
                score=round(float(data["score"]), 4),
                confidence=round(float(data["confidence"]), 4),
                provenance=tuple(data["provenance"]),
                why=tuple(data["why"]),
            )
            for (owner_type, owner_id), data in candidates.items()
            if owner_type in {"entity", "episode", "claim"}
        ]
        return sorted(hits, key=lambda hit: (hit.score, hit.confidence), reverse=True)[: self.config.retrieval_limit]

    @staticmethod
    def _add(candidates, owner_type, owner_id, score, confidence, why, provenance=None) -> None:
        key = (owner_type, owner_id)
        candidate = candidates.setdefault(
            key, {"score": 0.0, "confidence": 0.0, "why": [], "provenance": []}
        )
        candidate["score"] = max(float(candidate["score"]), max(0.0, min(1.0, float(score))))
        candidate["confidence"] = max(float(candidate["confidence"]), max(0.0, min(1.0, float(confidence))))
        if why not in candidate["why"]:
            candidate["why"].append(why)
        if provenance and all(item.evidence_id != provenance.evidence_id for item in candidate["provenance"]):
            candidate["provenance"].append(provenance)

    @staticmethod
    def _terms(query: str) -> list[str]:
        return [
            term for term in dict.fromkeys(re.findall(r"[a-z0-9][a-z0-9_-]+", query.casefold()))
            if len(term) >= 3 and term not in _STOPWORDS
        ][:12]

    @staticmethod
    def _recency(timestamp: str, now: datetime) -> float:
        age_hours = max(0.0, (now - datetime.fromisoformat(timestamp)).total_seconds() / 3600)
        return 1.0 / (1.0 + age_hours / 24.0)

    @staticmethod
    def _normalized(vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(value))
        return value / norm if norm else value

    @staticmethod
    def _evidence_ref(row: dict[str, object]) -> EvidenceRef:
        return EvidenceRef(
            str(row["evidence_id"]), str(row["modality"]), datetime.fromisoformat(str(row["captured_at"])),
            str(row["source_type"]), str(row["source_id"]),
            str(row["media_key"]) if row.get("media_key") else None,
            float(row.get("quality") or 0.0), dict(row.get("payload") or {}),
        )
