from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import threading
import uuid
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from egg_companion.config import MemoryConfig
from egg_companion.memory.schema import migrate
from egg_companion.models import EvidenceRef, GraphCognitiveSignal


def _identifier() -> str:
    return str(uuid.uuid4())


def _timestamp(value: datetime) -> str:
    return value.isoformat()


def _row(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    for key in tuple(result):
        if key.endswith("_json") and isinstance(result[key], str):
            try:
                result[key.removesuffix("_json")] = json.loads(result.pop(key))
            except json.JSONDecodeError:
                result[key.removesuffix("_json")] = {}
    return result


class MemoryStore:
    """Process-safe, append-first SQLite property graph for local Egg memory."""

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self.root = Path(config.storage_dir)
        self.media_root = self.root / "media"
        self.root.mkdir(parents=True, exist_ok=True)
        self.media_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.root / "memory.sqlite3", check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        migrate(self._connection)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                yield self._connection
            except Exception:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def append_evidence(
        self, evidence: EvidenceRef, embedding_key: str | None = None, checksum: str | None = None
    ) -> str:
        if evidence.media_key and Path(evidence.media_key).is_absolute():
            raise ValueError("memory media keys must be relative to the local memory directory")
        with self._transaction() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO evidence
                (evidence_id, modality, captured_at, source_type, source_id, media_key, checksum, quality, payload_json, embedding_key)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    evidence.evidence_id, evidence.modality, _timestamp(evidence.captured_at), evidence.source_type,
                    evidence.source_id, evidence.media_key, checksum, evidence.quality,
                    json.dumps(evidence.metadata, sort_keys=True), embedding_key,
                ),
            )
        return evidence.evidence_id

    def persist_media(self, relative_key: str, data: bytes) -> tuple[str, str]:
        root = self.root.resolve()
        media_root = self.media_root.resolve()
        path = (media_root / relative_key).resolve()
        if media_root not in path.parents:
            raise ValueError("media key escapes the memory media directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path.relative_to(root)), hashlib.sha256(data).hexdigest()

    def evidence_media(self, evidence_id: str) -> tuple[bytes, str] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT media_key FROM evidence WHERE evidence_id=?", (evidence_id,)
            ).fetchone()
        if row is None or not row["media_key"]:
            return None
        root = self.root.resolve()
        media_root = self.media_root.resolve()
        path = (root / str(row["media_key"])).resolve()
        if media_root not in path.parents or not path.is_file():
            return None
        return path.read_bytes(), path.suffix.casefold()

    def upsert_entity(
        self, entity_type: str, display_name: str | None = None, metadata: dict[str, Any] | None = None,
        entity_id: str | None = None, state: str = "active", now: datetime | None = None,
    ) -> str:
        entity_id = entity_id or _identifier()
        timestamp = _timestamp(now or datetime.now().astimezone())
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT metadata_json FROM entities WHERE entity_id=?", (entity_id,)
            ).fetchone()
            merged_metadata = json.loads(existing[0]) if existing else {}
            merged_metadata.update(metadata or {})
            metadata_json = json.dumps(merged_metadata, sort_keys=True)
            connection.execute(
                """INSERT INTO entities (entity_id, entity_type, display_name, state, created_at, updated_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_id) DO UPDATE SET display_name=COALESCE(excluded.display_name, entities.display_name),
                state=excluded.state, updated_at=excluded.updated_at, metadata_json=excluded.metadata_json""",
                (entity_id, entity_type, display_name, state, timestamp, timestamp, metadata_json),
            )
        return entity_id

    def open_episode(self, started_at: datetime, novelty: float = 0.0, episode_id: str | None = None) -> str:
        episode_id = episode_id or _identifier()
        timestamp = _timestamp(started_at)
        with self._transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO episodes (episode_id, started_at, state, novelty, created_at) VALUES (?, ?, 'open', ?, ?)",
                (episode_id, timestamp, novelty, timestamp),
            )
        return episode_id

    def append_episode_evidence(self, episode_id: str, evidence_id: str, role: str = "observation") -> None:
        with self._transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO episode_evidence (episode_id, evidence_id, role) VALUES (?, ?, ?)",
                (episode_id, evidence_id, role),
            )

    def link_episode_entity(
        self, episode_id: str, entity_id: str, role: str = "participant", confidence: float = 0.0
    ) -> None:
        with self._transaction() as connection:
            connection.execute(
                """INSERT INTO episode_entities (episode_id, entity_id, role, confidence) VALUES (?, ?, ?, ?)
                ON CONFLICT(episode_id, entity_id, role) DO UPDATE SET
                confidence=MAX(episode_entities.confidence, excluded.confidence)""",
                (episode_id, entity_id, role, confidence),
            )

    def link_entity_evidence(self, entity_id: str, evidence_id: str, role: str = "sighting") -> None:
        with self._transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO entity_evidence (entity_id, evidence_id, role) VALUES (?, ?, ?)",
                (entity_id, evidence_id, role),
            )

    def identity_coobservation_conflicts(
        self, profile_ids: list[str]
    ) -> set[tuple[str, str]]:
        """Return face-profile pairs backed by evidence that they were distinct together."""

        ids = sorted({str(profile_id) for profile_id in profile_ids if profile_id})
        if len(ids) < 2:
            return set()
        placeholders = ",".join("?" for _ in ids)
        values: tuple[str, ...] = tuple(ids)
        with self._lock:
            shared_evidence = self._connection.execute(
                f"""SELECT DISTINCT a.entity_id AS left_id, b.entity_id AS right_id
                FROM entity_evidence a JOIN entity_evidence b
                ON a.evidence_id=b.evidence_id AND a.entity_id < b.entity_id
                WHERE a.entity_id IN ({placeholders}) AND b.entity_id IN ({placeholders})""",
                values + values,
            ).fetchall()
            shared_episodes = self._connection.execute(
                f"""SELECT DISTINCT a.entity_id AS left_id, b.entity_id AS right_id
                FROM episode_entities a JOIN episode_entities b
                ON a.episode_id=b.episode_id AND a.entity_id < b.entity_id
                WHERE a.entity_id IN ({placeholders}) AND b.entity_id IN ({placeholders})""",
                values + values,
            ).fetchall()
            explicit_edges = self._connection.execute(
                f"""SELECT source_id AS left_id, target_id AS right_id FROM edges
                WHERE relation='co_observed_with' AND state='active'
                AND source_id IN ({placeholders}) AND target_id IN ({placeholders})""",
                values + values,
            ).fetchall()
        return {
            tuple(sorted((str(row["left_id"]), str(row["right_id"]))))
            for row in (*shared_evidence, *shared_episodes, *explicit_edges)
        }

    def identity_strong_coobservation_conflicts(
        self, profile_ids: list[str], minimum_confirmations: int = 3
    ) -> set[tuple[str, str]]:
        """Return only repeatable or spatially explicit distinct-person constraints.

        Legacy events attached every detection in a frame/episode to every other
        detection. A single duplicate detector box therefore cannot safely veto a
        strong face-template match. Repeated events remain a hard constraint, as
        does one event containing clearly separated boxes for both identities.
        """

        ids = sorted({str(profile_id) for profile_id in profile_ids if profile_id})
        if len(ids) < 2:
            return set()
        minimum = max(1, int(minimum_confirmations))
        placeholders = ",".join("?" for _ in ids)
        values: tuple[str, ...] = tuple(ids)
        with self._lock:
            shared_evidence = self._connection.execute(
                f"""SELECT a.entity_id AS left_id, b.entity_id AS right_id,
                a.evidence_id, evidence.payload_json
                FROM entity_evidence a JOIN entity_evidence b
                ON a.evidence_id=b.evidence_id AND a.entity_id < b.entity_id
                JOIN evidence ON evidence.evidence_id=a.evidence_id
                WHERE a.entity_id IN ({placeholders}) AND b.entity_id IN ({placeholders})""",
                values + values,
            ).fetchall()
            shared_episodes = self._connection.execute(
                f"""SELECT a.entity_id AS left_id, b.entity_id AS right_id,
                COUNT(DISTINCT a.episode_id) AS confirmations
                FROM episode_entities a JOIN episode_entities b
                ON a.episode_id=b.episode_id AND a.entity_id < b.entity_id
                WHERE a.entity_id IN ({placeholders}) AND b.entity_id IN ({placeholders})
                GROUP BY a.entity_id, b.entity_id""",
                values + values,
            ).fetchall()
            explicit_edges = self._connection.execute(
                f"""SELECT source_id AS left_id, target_id AS right_id,
                MAX(confirmation_count) AS confirmations FROM edges
                WHERE relation='co_observed_with' AND state='active'
                AND source_id IN ({placeholders}) AND target_id IN ({placeholders})
                GROUP BY source_id, target_id""",
                values + values,
            ).fetchall()
        confirmations: dict[tuple[str, str], set[str]] = defaultdict(set)
        hard: set[tuple[str, str]] = set()
        for row in shared_evidence:
            pair = tuple(sorted((str(row["left_id"]), str(row["right_id"]))))
            confirmations[pair].add(str(row["evidence_id"]))
            try:
                payload = json.loads(str(row["payload_json"] or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                payload = {}
            if self._payload_has_spatially_distinct_identities(payload, pair):
                hard.add(pair)
        for row in (*shared_episodes, *explicit_edges):
            pair = tuple(sorted((str(row["left_id"]), str(row["right_id"]))))
            if int(row["confirmations"] or 0) >= minimum:
                hard.add(pair)
        hard.update(pair for pair, rows in confirmations.items() if len(rows) >= minimum)
        return hard

    @staticmethod
    def _payload_has_spatially_distinct_identities(
        payload: dict[str, Any], pair: tuple[str, str]
    ) -> bool:
        detections = payload.get("detections")
        if not isinstance(detections, list):
            return False
        boxes: dict[str, list[dict[str, float]]] = defaultdict(list)
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            identity_id = str(detection.get("identity_id") or "")
            bbox = detection.get("bbox")
            if identity_id not in pair or not isinstance(bbox, dict):
                continue
            try:
                parsed = {key: float(bbox[key]) for key in ("x1", "y1", "x2", "y2")}
            except (KeyError, TypeError, ValueError):
                continue
            if parsed["x2"] > parsed["x1"] and parsed["y2"] > parsed["y1"]:
                boxes[identity_id].append(parsed)
        for left in boxes.get(pair[0], []):
            for right in boxes.get(pair[1], []):
                intersection_width = max(
                    0.0, min(left["x2"], right["x2"]) - max(left["x1"], right["x1"])
                )
                intersection_height = max(
                    0.0, min(left["y2"], right["y2"]) - max(left["y1"], right["y1"])
                )
                intersection = intersection_width * intersection_height
                left_area = (left["x2"] - left["x1"]) * (left["y2"] - left["y1"])
                right_area = (right["x2"] - right["x1"]) * (right["y2"] - right["y1"])
                iou = intersection / max(1.0, left_area + right_area - intersection)
                left_center = ((left["x1"] + left["x2"]) / 2, (left["y1"] + left["y2"]) / 2)
                right_center = ((right["x1"] + right["x2"]) / 2, (right["y1"] + right["y2"]) / 2)
                separation = ((left_center[0] - right_center[0]) ** 2 + (left_center[1] - right_center[1]) ** 2) ** 0.5
                scale = max(
                    1.0,
                    ((left["x2"] - left["x1"]) ** 2 + (left["y2"] - left["y1"]) ** 2) ** 0.5,
                    ((right["x2"] - right["x1"]) ** 2 + (right["y2"] - right["y1"]) ** 2) ** 0.5,
                )
                if iou < 0.10 and separation / scale > 0.55:
                    return True
        return False

    def coalesce_identity_evidence(
        self, aliases: list[dict[str, object]]
    ) -> dict[str, int]:
        """Project reversible identity aliases into the graph and canonical evidence view."""

        linked = copied_evidence = copied_episodes = 0
        now = _timestamp(datetime.now(timezone.utc))
        with self._transaction() as connection:
            for mapping in aliases:
                alias_id = str(mapping.get("alias_id") or "")
                canonical_id = str(mapping.get("canonical_id") or "")
                if not alias_id or not canonical_id or alias_id == canonical_id:
                    continue
                entities = connection.execute(
                    "SELECT entity_id FROM entities WHERE entity_id IN (?, ?)",
                    (alias_id, canonical_id),
                ).fetchall()
                if len(entities) != 2:
                    continue
                try:
                    similarity = max(0.0, min(1.0, float(mapping.get("similarity") or 0.0)))
                except (TypeError, ValueError):
                    similarity = 0.0
                reason = str(mapping.get("reason") or "face_identity_coalescing")
                alias_row = connection.execute(
                    "SELECT metadata_json FROM entities WHERE entity_id=?", (alias_id,)
                ).fetchone()
                metadata = json.loads(alias_row["metadata_json"]) if alias_row else {}
                metadata.update(
                    {
                        "canonical_identity_id": canonical_id,
                        "coalesced_similarity": round(similarity, 6),
                        "coalescing_reason": reason,
                    }
                )
                connection.execute(
                    "UPDATE entities SET merged_into=?, updated_at=?, metadata_json=? WHERE entity_id=?",
                    (canonical_id, now, json.dumps(metadata, sort_keys=True), alias_id),
                )
                edge_id = f"identity-alias:{alias_id}:{canonical_id}"
                connection.execute(
                    """INSERT INTO edges
                    (edge_id, source_id, relation, target_id, confidence, valid_from, state, metadata_json)
                    VALUES (?, ?, 'same_person_as', ?, ?, ?, 'active', ?)
                    ON CONFLICT(edge_id) DO UPDATE SET confidence=excluded.confidence,
                    state='active', metadata_json=excluded.metadata_json""",
                    (
                        edge_id, alias_id, canonical_id, similarity, now,
                        json.dumps({"reason": reason, "reversible": True}, sort_keys=True),
                    ),
                )
                before = connection.total_changes
                connection.execute(
                    """INSERT OR IGNORE INTO entity_evidence (entity_id, evidence_id, role)
                    SELECT ?, evidence_id, role FROM entity_evidence WHERE entity_id=?""",
                    (canonical_id, alias_id),
                )
                copied_evidence += connection.total_changes - before
                before = connection.total_changes
                connection.execute(
                    """INSERT INTO episode_entities (episode_id, entity_id, role, confidence)
                    SELECT episode_id, ?, role, confidence FROM episode_entities WHERE entity_id=?
                    ON CONFLICT(episode_id, entity_id, role) DO UPDATE SET
                    confidence=MAX(episode_entities.confidence, excluded.confidence)""",
                    (canonical_id, alias_id),
                )
                copied_episodes += connection.total_changes - before
                linked += 1
        return {
            "aliases": linked,
            "evidence_links_copied": copied_evidence,
            "episode_links_copied": copied_episodes,
        }

    def link_entities(
        self, source_id: str, relation: str, target_id: str, confidence: float, valid_from: datetime,
        metadata: dict[str, Any] | None = None, edge_id: str | None = None, evidence_id: str | None = None,
    ) -> str:
        edge_id = edge_id or _identifier()
        with self._transaction() as connection:
            connection.execute(
                """INSERT INTO edges
                (edge_id, source_id, relation, target_id, confidence, valid_from, state, metadata_json, evidence_id)
                VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)""",
                (
                    edge_id, source_id, relation, target_id, confidence, _timestamp(valid_from),
                    json.dumps(metadata or {}, sort_keys=True), evidence_id,
                ),
            )
        return edge_id

    def link_entities_once(
        self, source_id: str, relation: str, target_id: str, confidence: float, valid_from: datetime,
        metadata: dict[str, Any] | None = None, evidence_id: str | None = None,
    ) -> str:
        with self._transaction() as connection:
            existing = connection.execute(
                """SELECT edge_id, metadata_json FROM edges
                WHERE source_id=? AND relation=? AND target_id=? AND state='active'
                ORDER BY valid_from DESC LIMIT 1""",
                (source_id, relation, target_id),
            ).fetchone()
            if existing:
                merged = json.loads(existing["metadata_json"])
                merged.update(metadata or {})
                connection.execute(
                    """UPDATE edges SET confidence=MAX(confidence, ?), confirmation_count=confirmation_count+1,
                    metadata_json=?, evidence_id=COALESCE(?, evidence_id) WHERE edge_id=?""",
                    (confidence, json.dumps(merged, sort_keys=True), evidence_id, existing["edge_id"]),
                )
                return str(existing["edge_id"])
            edge_id = _identifier()
            connection.execute(
                """INSERT INTO edges
                (edge_id, source_id, relation, target_id, confidence, valid_from, state, metadata_json, evidence_id)
                VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)""",
                (
                    edge_id, source_id, relation, target_id, confidence, _timestamp(valid_from),
                    json.dumps(metadata or {}, sort_keys=True), evidence_id,
                ),
            )
            return edge_id

    def assert_claim(
        self, subject_id: str, predicate: str, object_id_or_text: str, confidence: float, valid_from: datetime,
        claim_id: str | None = None, source: str = "system", evidence_id: str | None = None,
        metadata: dict[str, Any] | None = None, state: str = "active",
    ) -> str:
        claim_id = claim_id or _identifier()
        timestamp = _timestamp(valid_from)
        with self._transaction() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO claims
                (claim_id, subject_id, predicate, object_id_or_text, confidence, state, valid_from, created_at,
                source, evidence_id, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    claim_id, subject_id, predicate, object_id_or_text, confidence, state, timestamp, timestamp,
                    source, evidence_id, json.dumps(metadata or {}, sort_keys=True),
                ),
            )
        return claim_id

    def assert_claim_once(
        self, subject_id: str, predicate: str, object_id_or_text: str, confidence: float, valid_from: datetime,
        source: str = "system", evidence_id: str | None = None, metadata: dict[str, Any] | None = None,
    ) -> str:
        with self._lock:
            existing = self._connection.execute(
                """SELECT claim_id FROM claims WHERE subject_id=? AND predicate=? AND object_id_or_text=?
                AND state='active' ORDER BY created_at DESC LIMIT 1""",
                (subject_id, predicate, object_id_or_text),
            ).fetchone()
        if existing:
            return str(existing["claim_id"])
        return self.assert_claim(
            subject_id, predicate, object_id_or_text, confidence, valid_from,
            source=source, evidence_id=evidence_id, metadata=metadata,
        )

    def revise_claim(
        self, claim_id: str, decision: str, actor: str, replacement_value: str | None = None,
        evidence_id: str | None = None, at: datetime | None = None,
    ) -> str:
        revision_id = _identifier()
        timestamp = _timestamp(at or datetime.now().astimezone())
        state = "retracted" if decision in {"retract", "reject"} else "superseded"
        with self._transaction() as connection:
            connection.execute("UPDATE claims SET state=?, valid_to=?, revised_at=? WHERE claim_id=?", (state, timestamp, timestamp, claim_id))
            connection.execute(
                """INSERT INTO revisions (revision_id, target_type, target_id, decision, replacement_value, actor, created_at, evidence_id)
                VALUES (?, 'claim', ?, ?, ?, ?, ?, ?)""",
                (revision_id, claim_id, decision, replacement_value, actor, timestamp, evidence_id),
            )
        return revision_id

    def reject_edge(
        self, edge_id: str, actor: str = "user", at: datetime | None = None
    ) -> str:
        revision_id = _identifier()
        timestamp = _timestamp(at or datetime.now().astimezone())
        with self._transaction() as connection:
            edge = connection.execute(
                "SELECT edge_id FROM edges WHERE edge_id=?", (edge_id,)
            ).fetchone()
            if edge is None:
                raise KeyError(edge_id)
            connection.execute(
                "UPDATE edges SET state='retracted', valid_to=? WHERE edge_id=?",
                (timestamp, edge_id),
            )
            connection.execute(
                """INSERT INTO revisions
                (revision_id, target_type, target_id, decision, actor, created_at)
                VALUES (?, 'edge', ?, 'reject', ?, ?)""",
                (revision_id, edge_id, actor, timestamp),
            )
        return revision_id

    def close_episode(self, episode_id: str, ended_at: datetime, summary: str | None = None) -> None:
        with self._transaction() as connection:
            connection.execute(
                "UPDATE episodes SET ended_at=?, summary=?, state='closed' WHERE episode_id=?",
                (_timestamp(ended_at), summary, episode_id),
            )

    def add_embedding(
        self, owner_type: str, owner_id: str, modality: str, model_id: str, vector: np.ndarray,
        quality: float, created_at: datetime, embedding_id: str | None = None,
    ) -> str:
        embedding_id = embedding_id or _identifier()
        normalized = np.asarray(vector, dtype=np.float32).reshape(-1)
        with self._transaction() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO embeddings (embedding_id, owner_type, owner_id, modality, model_id, dimensions, vector_blob, quality, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (embedding_id, owner_type, owner_id, modality, model_id, normalized.size, normalized.tobytes(), quality, _timestamp(created_at)),
            )
        return embedding_id

    def recent_episodes(self, limit: int | None = None) -> list[dict[str, Any]]:
        limit = limit or self.config.retrieval_limit
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM episodes ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row(row) for row in rows]

    def narrative_candidate_timestamps(
        self, entity_ids: list[str], limit: int = 10000
    ) -> list[str]:
        """Return evidence times that may need a dated narrative rebuild.

        Identity aliases identify dates whose already-built chapters need a
        revision. Never-narrated history is discovered separately from compact
        store-wide boundaries, so backlog admission does not depend on a merge.
        """
        normalized = list(dict.fromkeys(str(value) for value in entity_ids if value))
        bounded = max(1, min(int(limit), 50000))
        with self._lock:
            if normalized:
                rows: list[sqlite3.Row] = []
                for offset in range(0, len(normalized), 800):
                    batch = normalized[offset : offset + 800]
                    placeholders = ",".join("?" for _ in batch)
                    rows.extend(
                        self._connection.execute(
                            f"""SELECT DISTINCT evidence.captured_at
                            FROM evidence JOIN entity_evidence
                              ON evidence.evidence_id=entity_evidence.evidence_id
                            WHERE entity_evidence.entity_id IN ({placeholders})
                            ORDER BY evidence.captured_at DESC LIMIT ?""",
                            (*batch, bounded),
                        ).fetchall()
                    )
                values = sorted(
                    {str(row["captured_at"]) for row in rows}, reverse=True
                )[:bounded]
                if values:
                    return values
            row = self._connection.execute(
                "SELECT captured_at FROM evidence ORDER BY captured_at DESC LIMIT 1"
            ).fetchone()
        return [str(row["captured_at"])] if row is not None else []

    def narrative_history_boundaries(self) -> list[str]:
        """Return compact time boundaries spanning every retained evidence day.

        Grouping in UTC keeps this query small even for a large evidence store.
        The narrative synthesizer converts both edges of each UTC bucket into
        its configured local timezone, which also discovers days crossing a
        timezone boundary without loading every evidence timestamp into Python.
        """
        with self._lock:
            rows = self._connection.execute(
                """SELECT MIN(captured_at) AS first_at,
                MAX(captured_at) AS last_at
                FROM evidence
                GROUP BY substr(captured_at, 1, 10)
                ORDER BY first_at ASC"""
            ).fetchall()
        return list(
            dict.fromkeys(
                str(value)
                for row in rows
                for value in (row["first_at"], row["last_at"])
                if value
            )
        )

    def chronological_evidence(
        self, start_at: datetime, end_at: datetime, limit: int = 50000
    ) -> list[dict[str, Any]]:
        """Return one local-day evidence ledger with canonical associations."""
        bounded = max(1, min(int(limit), 50000))
        with self._lock:
            rows = self._connection.execute(
                """SELECT evidence.*,
                GROUP_CONCAT(DISTINCT entity_evidence.entity_id) AS entity_ids_csv,
                GROUP_CONCAT(DISTINCT episode_evidence.episode_id) AS episode_ids_csv
                FROM evidence
                LEFT JOIN entity_evidence
                  ON entity_evidence.evidence_id=evidence.evidence_id
                LEFT JOIN entities
                  ON entities.entity_id=entity_evidence.entity_id
                LEFT JOIN episode_evidence
                  ON episode_evidence.evidence_id=evidence.evidence_id
                WHERE julianday(evidence.captured_at) >= julianday(?)
                  AND julianday(evidence.captured_at) < julianday(?)
                  AND (
                    entity_evidence.entity_id IS NULL
                    OR (entities.state='active' AND entities.merged_into IS NULL)
                  )
                GROUP BY evidence.evidence_id
                ORDER BY evidence.captured_at ASC LIMIT ?""",
                (_timestamp(start_at), _timestamp(end_at), bounded),
            ).fetchall()
            records = [_row(row) for row in rows]
            entity_ids = sorted(
                {
                    entity_id
                    for record in records
                    for entity_id in str(record.get("entity_ids_csv", "") or "").split(",")
                    if entity_id
                }
            )
            episode_ids = sorted(
                {
                    episode_id
                    for record in records
                    for episode_id in str(record.get("episode_ids_csv", "") or "").split(",")
                    if episode_id
                }
            )
            entity_map: dict[str, dict[str, Any]] = {}
            for offset in range(0, len(entity_ids), 800):
                batch = entity_ids[offset : offset + 800]
                placeholders = ",".join("?" for _ in batch)
                for row in self._connection.execute(
                    f"""SELECT entity_id, entity_type, display_name, metadata_json
                    FROM entities WHERE entity_id IN ({placeholders})""",
                    batch,
                ).fetchall():
                    item = _row(row)
                    entity_map[str(item["entity_id"])] = item
            episode_map: dict[str, dict[str, Any]] = {}
            for offset in range(0, len(episode_ids), 800):
                batch = episode_ids[offset : offset + 800]
                placeholders = ",".join("?" for _ in batch)
                for row in self._connection.execute(
                    f"""SELECT episode_id, started_at, ended_at, state, novelty, summary
                    FROM episodes WHERE episode_id IN ({placeholders})""",
                    batch,
                ).fetchall():
                    episode_map[str(row["episode_id"])] = dict(row)
        for record in records:
            linked_ids = [
                value
                for value in str(record.pop("entity_ids_csv", "") or "").split(",")
                if value
            ]
            linked_episode_ids = [
                value
                for value in str(record.pop("episode_ids_csv", "") or "").split(",")
                if value
            ]
            record["entities"] = [
                entity_map[value] for value in linked_ids if value in entity_map
            ]
            record["episodes"] = [
                episode_map[value]
                for value in linked_episode_ids
                if value in episode_map
            ]
        return records

    def list_entities(
        self, entity_type: str | None = None, state: str = "active", limit: int | None = None
    ) -> list[dict[str, Any]]:
        limit = min(limit or self.config.retrieval_limit, self.config.graph_max_nodes)
        where = ["state=?"]
        values: list[Any] = [state]
        if entity_type:
            where.append("entity_type=?")
            values.append(entity_type)
        values.append(limit)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM entities WHERE {' AND '.join(where)} ORDER BY updated_at DESC LIMIT ?", values
            ).fetchall()
        return [_row(row) for row in rows]

    def cognitive_signals(
        self, entity_ids: list[str]
    ) -> dict[str, GraphCognitiveSignal]:
        """Return small, explainable graph-derived control signals.

        Counts are converted to saturating values so a long-lived hub cannot
        overwhelm current sensory evidence merely because it has a large past.
        """
        signals: dict[str, GraphCognitiveSignal] = {}
        for entity_id in dict.fromkeys(str(value) for value in entity_ids if value):
            with self._lock:
                entity = self._connection.execute(
                    "SELECT entity_id FROM entities WHERE entity_id=? AND state='active'",
                    (entity_id,),
                ).fetchone()
                if entity is None:
                    continue
                evidence_count = int(
                    self._connection.execute(
                        "SELECT COUNT(*) FROM entity_evidence WHERE entity_id=?",
                        (entity_id,),
                    ).fetchone()[0]
                )
                edge_count = int(
                    self._connection.execute(
                        """SELECT COUNT(*) FROM edges WHERE state='active'
                        AND (source_id=? OR target_id=?)""",
                        (entity_id, entity_id),
                    ).fetchone()[0]
                )
                claim_count = int(
                    self._connection.execute(
                        "SELECT COUNT(*) FROM claims WHERE subject_id=? AND state='active'",
                        (entity_id,),
                    ).fetchone()[0]
                )
                conflict_count = int(
                    self._connection.execute(
                        """SELECT COUNT(*) FROM (
                        SELECT predicate FROM claims WHERE subject_id=? AND state='active'
                        GROUP BY predicate HAVING COUNT(DISTINCT object_id_or_text) > 1
                        )""",
                        (entity_id,),
                    ).fetchone()[0]
                )
            familiarity = 1.0 - math.exp(-evidence_count / 4.0)
            structural_relevance = 1.0 - math.exp(-edge_count / 4.0)
            knowledge_gap = min(
                1.0,
                0.55 / (1.0 + claim_count) + (0.45 if conflict_count else 0.0),
            )
            signals[entity_id] = GraphCognitiveSignal(
                entity_id,
                round(familiarity, 4),
                round(structural_relevance, 4),
                round(knowledge_gap, 4),
                evidence_count,
                edge_count,
                claim_count,
                conflict_count,
            )
        return signals

    def cognitive_inventory(self, limit: int = 40) -> list[dict[str, Any]]:
        """Bounded graph inventory used by quiet-period replay and curiosity."""
        bounded = max(1, min(int(limit), self.config.graph_max_nodes))
        with self._lock:
            rows = self._connection.execute(
                """SELECT e.entity_id, e.entity_type, e.display_name, e.updated_at,
                e.metadata_json,
                (SELECT COUNT(*) FROM entity_evidence ee
                 WHERE ee.entity_id=e.entity_id) AS evidence_count,
                (SELECT COUNT(*) FROM edges ed WHERE ed.state='active'
                 AND (ed.source_id=e.entity_id OR ed.target_id=e.entity_id)) AS edge_count,
                (SELECT COUNT(*) FROM claims c WHERE c.state='active'
                 AND c.subject_id=e.entity_id) AS claim_count
                FROM entities e WHERE e.state='active' AND e.merged_into IS NULL
                AND e.entity_type IN ('person','object','content')
                ORDER BY evidence_count DESC, e.updated_at DESC LIMIT ?""",
                (bounded,),
            ).fetchall()
            records = [_row(row) for row in rows]
            for record in records:
                claims = self._connection.execute(
                    """SELECT predicate, object_id_or_text, confidence, source
                    FROM claims WHERE subject_id=? AND state='active'
                    ORDER BY confidence DESC, created_at DESC LIMIT 20""",
                    (record["entity_id"],),
                ).fetchall()
                record["claims"] = [dict(claim) for claim in claims]
        return records

    def record_default_mode_reflection(
        self,
        source_entity_id: str,
        reflection_kind: str,
        summary: str,
        confidence: float,
        metadata: dict[str, Any],
        at: datetime,
    ) -> tuple[str, bool]:
        """Project one source-supported reflection back into the graph."""
        digest = hashlib.sha256(
            f"{source_entity_id}:{reflection_kind}".encode()
        ).hexdigest()[:24]
        reflection_id = f"reflection:{digest}"
        created = self.entity_detail(reflection_id) is None
        self.upsert_entity(
            "reflection",
            summary[:300],
            {
                **metadata,
                "reflection_kind": reflection_kind,
                "source_entity_id": source_entity_id,
                "confidence": max(0.0, min(1.0, float(confidence))),
                "derived_at": at.isoformat(),
            },
            reflection_id,
            now=at,
        )
        if created:
            self.link_entities_once(
                source_entity_id,
                "evokes_reflection",
                reflection_id,
                confidence,
                at,
                {"source": "default-mode-replay"},
            )
        return reflection_id, created

    def recurrent_entity_pairs(
        self,
        minimum_confirmations: int = 2,
        limit: int = 24,
        period_seconds: int = 600,
    ) -> list[dict[str, Any]]:
        """Return motifs repeated across time windows, not adjacent frames."""
        minimum = max(2, int(minimum_confirmations))
        bounded = max(1, min(int(limit), self.config.graph_max_nodes))
        period = max(60, min(int(period_seconds), 86400))
        with self._lock:
            rows = self._connection.execute(
                """SELECT left_entity.entity_id AS left_id,
                left_entity.entity_type AS left_type,
                left_entity.display_name AS left_label,
                right_entity.entity_id AS right_id,
                right_entity.entity_type AS right_type,
                right_entity.display_name AS right_label,
                COUNT(DISTINCT CAST(strftime('%s', episode.started_at) / ? AS INTEGER))
                  AS confirmations,
                COUNT(DISTINCT left_episode.episode_id) AS observation_count,
                MAX(episode.started_at) AS last_observed_at
                FROM episode_entities left_episode
                JOIN episode_entities right_episode
                  ON left_episode.episode_id=right_episode.episode_id
                 AND left_episode.entity_id < right_episode.entity_id
                JOIN entities left_entity
                  ON left_entity.entity_id=left_episode.entity_id
                JOIN entities right_entity
                  ON right_entity.entity_id=right_episode.entity_id
                JOIN episodes episode ON episode.episode_id=left_episode.episode_id
                WHERE left_entity.state='active' AND right_entity.state='active'
                  AND left_entity.merged_into IS NULL
                  AND right_entity.merged_into IS NULL
                  AND left_entity.entity_type IN
                    ('person','object','object_category','content','sound_event')
                  AND right_entity.entity_type IN
                    ('person','object','object_category','content','sound_event')
                GROUP BY left_entity.entity_id, right_entity.entity_id
                HAVING confirmations >= ?
                ORDER BY confirmations DESC, last_observed_at DESC LIMIT ?""",
                (period, minimum, bounded),
            ).fetchall()
            records = [dict(row) for row in rows]
            for record in records:
                episode_rows = self._connection.execute(
                    """SELECT MIN(left_episode.episode_id) AS episode_id,
                    CAST(strftime('%s', episode.started_at) / ? AS INTEGER) AS period_id
                    FROM episode_entities left_episode
                    JOIN episode_entities right_episode
                      ON left_episode.episode_id=right_episode.episode_id
                    JOIN episodes episode ON episode.episode_id=left_episode.episode_id
                    WHERE left_episode.entity_id=? AND right_episode.entity_id=?
                    GROUP BY period_id ORDER BY period_id DESC LIMIT 100""",
                    (period, record["left_id"], record["right_id"]),
                ).fetchall()
                record["episode_ids"] = ",".join(
                    str(item["episode_id"])
                    for item in episode_rows
                    if item["episode_id"]
                )
                record["support_period_ids"] = [
                    str(item["period_id"])
                    for item in episode_rows
                    if item["period_id"] is not None
                ]
        return records

    def retire_inactive_meta_graph(
        self, active_abstraction_ids: list[str], at: datetime
    ) -> int:
        """Supersede derived motifs no longer admitted by current support rules."""
        active = set(active_abstraction_ids)
        timestamp = _timestamp(at)
        with self._transaction() as connection:
            rows = connection.execute(
                """SELECT entity_id FROM entities WHERE entity_type='abstraction'
                AND state='active'
                AND json_extract(metadata_json, '$.abstraction_kind')=
                  'recurring_episode_association'"""
            ).fetchall()
            stale = [str(row["entity_id"]) for row in rows if row["entity_id"] not in active]
            for abstraction_id in stale:
                connection.execute(
                    "UPDATE entities SET state='superseded', updated_at=? WHERE entity_id=?",
                    (timestamp, abstraction_id),
                )
                connection.execute(
                    """UPDATE edges SET state='retracted', valid_to=?
                    WHERE state='active' AND (
                      source_id=? OR target_id=? OR
                      json_extract(metadata_json, '$.abstraction_id')=?
                    )""",
                    (timestamp, abstraction_id, abstraction_id, abstraction_id),
                )
        return len(stale)

    def update_entity_summary(
        self, entity_id: str, summary: str, at: datetime
    ) -> None:
        """Store a bounded derived summary while retaining its source entity."""
        normalized = " ".join(summary.split())[:600]
        with self._lock:
            existing = self._connection.execute(
                "SELECT metadata_json FROM entities WHERE entity_id=?", (entity_id,)
            ).fetchone()
        if existing is None:
            return
        metadata = json.loads(existing["metadata_json"])
        if metadata.get("derived_summary") == normalized:
            return
        self.upsert_entity(
            "unknown",
            metadata={
                "derived_summary": normalized,
                "summary_updated_at": at.isoformat(),
            },
            entity_id=entity_id,
            now=at,
        )

    def upsert_cognitive_document(
        self,
        kind: str,
        title: str,
        content: str,
        confidence: float,
        source_entity_ids: list[str],
        at: datetime,
    ) -> dict[str, Any]:
        """Create or revise one stable, inspectable meta-graph document node."""
        document_id = f"cognitive-document:{kind}"
        normalized = "\n".join(
            line.rstrip() for line in content.splitlines() if line.strip()
        )[:5000]
        with self._lock:
            existing = self._connection.execute(
                "SELECT metadata_json FROM entities WHERE entity_id=?",
                (document_id,),
            ).fetchone()
        metadata = json.loads(existing["metadata_json"]) if existing else {}
        bounded_confidence = max(0.0, min(1.0, float(confidence)))
        normalized_sources = list(dict.fromkeys(source_entity_ids))[:100]
        changed = (
            metadata.get("content") != normalized
            or metadata.get("source_entity_ids") != normalized_sources
            or metadata.get("confidence") != bounded_confidence
        )
        revision = int(metadata.get("revision") or 0) + int(changed)
        if not changed and existing is not None:
            return {
                "document_id": document_id,
                "kind": kind,
                "revision": revision,
                "changed": False,
                "content": normalized,
            }
        self.upsert_entity(
            "cognitive_document",
            title[:120],
            {
                "document_kind": kind,
                "content": normalized,
                "revision": revision,
                "confidence": bounded_confidence,
                "source_entity_ids": normalized_sources,
                "epistemic_status": "derived_from_local_evidence",
                "updated_at": at.isoformat(),
            },
            document_id,
            now=at,
        )
        return {
            "document_id": document_id,
            "kind": kind,
            "revision": revision,
            "changed": changed,
            "content": normalized,
        }

    def cognitive_documents(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT entity_id, display_name, updated_at, metadata_json
                FROM entities WHERE entity_type='cognitive_document'
                AND state='active' ORDER BY updated_at DESC"""
            ).fetchall()
        return [_row(row) for row in rows]

    def upsert_daily_narrative(
        self,
        local_date: str,
        content: str,
        abstract_summary: str,
        timeline: list[dict[str, Any]],
        confidence: float,
        source_entity_ids: list[str],
        source_evidence_ids: list[str],
        source_episode_ids: list[str],
        dream_run_id: str,
        timezone_name: str,
        at: datetime,
        reviewed_evidence_count: int | None = None,
        reviewed_episode_count: int | None = None,
        narrative_schema_revision: int = 1,
    ) -> dict[str, Any]:
        """Create or revise an inspectable, evidence-backed chapter for one day."""
        narrative_id = f"daily-narrative:{local_date}"
        normalized = "\n".join(
            line.rstrip() for line in content.splitlines() if line.strip()
        )[:12000]
        source_entities = list(dict.fromkeys(source_entity_ids))[:1000]
        source_evidence = list(dict.fromkeys(source_evidence_ids))[:2000]
        source_episodes = list(dict.fromkeys(source_episode_ids))[:2000]
        evidence_count = max(
            len(source_evidence), int(reviewed_evidence_count or 0)
        )
        episode_count = max(len(source_episodes), int(reviewed_episode_count or 0))
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "content": normalized,
                    "timeline": timeline,
                    "entities": source_entities,
                    "evidence": source_evidence,
                    "episodes": source_episodes,
                    "reviewed_evidence_count": evidence_count,
                    "reviewed_episode_count": episode_count,
                    "narrative_schema_revision": narrative_schema_revision,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        with self._lock:
            existing = self._connection.execute(
                "SELECT metadata_json FROM entities WHERE entity_id=?", (narrative_id,)
            ).fetchone()
        metadata = json.loads(existing["metadata_json"]) if existing else {}
        changed = metadata.get("narrative_fingerprint") != fingerprint
        revision = int(metadata.get("revision") or 0) + int(changed)
        dream_runs = list(
            dict.fromkeys(
                [
                    *(
                        metadata.get("dream_run_ids", [])
                        if isinstance(metadata.get("dream_run_ids"), list)
                        else []
                    ),
                    dream_run_id,
                ]
            )
        )[-20:]
        self.upsert_entity(
            "daily_narrative",
            f"Daily story · {local_date}",
            {
                "document_kind": "daily-narrative",
                "local_date": local_date,
                "timezone": timezone_name,
                "content": normalized,
                "abstract_summary": abstract_summary[:1200],
                "timeline": timeline,
                "revision": revision,
                "confidence": max(0.0, min(1.0, float(confidence))),
                "source_entity_ids": source_entities,
                "source_evidence_ids": source_evidence,
                "source_episode_ids": source_episodes,
                "reviewed_evidence_count": evidence_count,
                "reviewed_episode_count": episode_count,
                "narrative_schema_revision": max(
                    1, int(narrative_schema_revision)
                ),
                "dream_run_ids": dream_runs,
                "last_dream_run_id": dream_run_id,
                "last_replayed_at": at.isoformat(),
                "narrative_fingerprint": fingerprint,
                "epistemic_status": "derived_from_chronological_local_evidence",
            },
            narrative_id,
            now=at,
        )
        return {
            "narrative_id": narrative_id,
            "local_date": local_date,
            "revision": revision,
            "changed": changed,
            "timeline_entries": len(timeline),
            "evidence_count": evidence_count,
            "episode_count": episode_count,
            "entity_count": len(source_entities),
            "abstract_summary": abstract_summary[:1200],
        }

    def recent_daily_narratives(self, limit: int = 30) -> list[dict[str, Any]]:
        bounded = max(1, min(int(limit), 3650))
        with self._lock:
            rows = self._connection.execute(
                """SELECT entity_id, display_name, updated_at, metadata_json
                FROM entities WHERE entity_type='daily_narrative' AND state='active'
                ORDER BY json_extract(metadata_json, '$.local_date') DESC LIMIT ?""",
                (bounded,),
            ).fetchall()
        return [_row(row) for row in rows]

    def daily_narrative_index(self, limit: int = 90) -> list[dict[str, Any]]:
        """Return compact newest-first chapter cards for the Narrative page."""
        records: list[dict[str, Any]] = []
        for item in self.recent_daily_narratives(limit):
            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                continue
            records.append(
                {
                    "narrative_id": str(item["entity_id"]),
                    "local_date": metadata.get("local_date"),
                    "timezone": metadata.get("timezone"),
                    "revision": int(metadata.get("revision") or 0),
                    "confidence": float(metadata.get("confidence") or 0.0),
                    "abstract_summary": str(metadata.get("abstract_summary") or ""),
                    "timeline_entries": len(metadata.get("timeline", []))
                    if isinstance(metadata.get("timeline"), list)
                    else 0,
                    "evidence_count": int(
                        metadata.get("reviewed_evidence_count")
                        or (
                            len(metadata.get("source_evidence_ids", []))
                            if isinstance(metadata.get("source_evidence_ids"), list)
                            else 0
                        )
                    ),
                    "episode_count": int(
                        metadata.get("reviewed_episode_count")
                        or (
                            len(metadata.get("source_episode_ids", []))
                            if isinstance(metadata.get("source_episode_ids"), list)
                            else 0
                        )
                    ),
                    "entity_count": len(metadata.get("source_entity_ids", []))
                    if isinstance(metadata.get("source_entity_ids"), list)
                    else 0,
                    "last_dream_run_id": metadata.get("last_dream_run_id"),
                    "last_replayed_at": metadata.get("last_replayed_at"),
                    "updated_at": item.get("updated_at"),
                }
            )
        return records

    def daily_narrative_detail(self, local_date: str) -> dict[str, Any] | None:
        """Return one chapter with nested periods, episodes, and artifact cards."""
        narrative_id = f"daily-narrative:{local_date}"
        with self._lock:
            row = self._connection.execute(
                """SELECT entity_id, display_name, updated_at, metadata_json
                FROM entities WHERE entity_id=? AND entity_type='daily_narrative'
                AND state='active'""",
                (narrative_id,),
            ).fetchone()
            if row is None:
                return None
            narrative = _row(row)
            metadata = narrative.get("metadata")
            if not isinstance(metadata, dict):
                return None
            evidence_ids = [
                str(value)
                for value in metadata.get("source_evidence_ids", [])
                if value
            ]
            episode_ids = [
                str(value)
                for value in metadata.get("source_episode_ids", [])
                if value
            ]
            evidence_map: dict[str, dict[str, Any]] = {}
            for offset in range(0, len(evidence_ids), 800):
                batch = evidence_ids[offset : offset + 800]
                placeholders = ",".join("?" for _ in batch)
                for evidence_row in self._connection.execute(
                    f"""SELECT evidence_id, modality, captured_at, source_type,
                    source_id, media_key, quality, payload_json FROM evidence
                    WHERE evidence_id IN ({placeholders})""",
                    batch,
                ).fetchall():
                    item = _row(evidence_row)
                    payload = item.pop("payload", {})
                    item["text"] = self._narrative_evidence_text(
                        payload if isinstance(payload, dict) else {}
                    )
                    if item.get("media_key"):
                        item["artifact_url"] = (
                            f"/api/memory/evidence/{item['evidence_id']}/media"
                        )
                    evidence_map[str(item["evidence_id"])] = item
            episode_map: dict[str, dict[str, Any]] = {}
            for offset in range(0, len(episode_ids), 800):
                batch = episode_ids[offset : offset + 800]
                placeholders = ",".join("?" for _ in batch)
                for episode_row in self._connection.execute(
                    f"""SELECT episode_id, started_at, ended_at, state, novelty,
                    summary FROM episodes WHERE episode_id IN ({placeholders})""",
                    batch,
                ).fetchall():
                    episode_map[str(episode_row["episode_id"])] = dict(episode_row)
        timeline: list[dict[str, Any]] = []
        for raw_entry in metadata.get("timeline", []):
            if not isinstance(raw_entry, dict):
                continue
            entry = dict(raw_entry)
            entry["artifacts"] = [
                evidence_map[evidence_id]
                for evidence_id in entry.get("evidence_ids", [])
                if evidence_id in evidence_map
            ]
            entry["episodes"] = [
                episode_map[episode_id]
                for episode_id in entry.get("episode_ids", [])
                if episode_id in episode_map
            ]
            timeline.append(entry)
        return {
            "narrative_id": narrative_id,
            "display_name": narrative.get("display_name"),
            "updated_at": narrative.get("updated_at"),
            "local_date": metadata.get("local_date"),
            "timezone": metadata.get("timezone"),
            "revision": int(metadata.get("revision") or 0),
            "confidence": float(metadata.get("confidence") or 0.0),
            "abstract_summary": metadata.get("abstract_summary"),
            "content": metadata.get("content"),
            "timeline": timeline,
            "source_entity_ids": metadata.get("source_entity_ids", []),
            "source_evidence_count": len(evidence_map),
            "source_episode_count": len(episode_map),
            "reviewed_evidence_count": int(
                metadata.get("reviewed_evidence_count") or len(evidence_map)
            ),
            "reviewed_episode_count": int(
                metadata.get("reviewed_episode_count") or len(episode_map)
            ),
            "dream_run_ids": metadata.get("dream_run_ids", []),
            "last_replayed_at": metadata.get("last_replayed_at"),
        }

    @staticmethod
    def _narrative_evidence_text(payload: dict[str, Any]) -> str:
        for key in (
            "transcript",
            "text",
            "summary",
            "analysis",
            "candidate_response",
            "corrected_label",
        ):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return " ".join(value.split())[:1200]
        classifications = payload.get("classifications")
        if isinstance(classifications, list):
            labels = [
                str(item.get("label"))
                for item in classifications
                if isinstance(item, dict) and item.get("label")
            ]
            if labels:
                return "Audio context: " + ", ".join(labels[:12])
        labels = payload.get("labels")
        if isinstance(labels, list) and labels:
            return "Observed labels: " + ", ".join(map(str, labels[:16]))
        return ""

    def link_daily_narrative_sources(
        self,
        narrative_id: str,
        evidence_ids: list[str],
        episode_ids: list[str],
    ) -> dict[str, int]:
        """Attach a chapter to its retained artifacts and source episodes in bulk."""
        evidence = list(dict.fromkeys(str(value) for value in evidence_ids if value))[:2000]
        episodes = list(dict.fromkeys(str(value) for value in episode_ids if value))[:2000]
        with self._transaction() as connection:
            before = connection.total_changes
            connection.executemany(
                "INSERT OR IGNORE INTO entity_evidence (entity_id, evidence_id, role) VALUES (?, ?, 'daily-story-source')",
                [(narrative_id, evidence_id) for evidence_id in evidence],
            )
            evidence_links = connection.total_changes - before
            before = connection.total_changes
            connection.executemany(
                """INSERT INTO episode_entities
                (episode_id, entity_id, role, confidence)
                VALUES (?, ?, 'chronological-entry', 0.9)
                ON CONFLICT(episode_id, entity_id, role) DO UPDATE SET
                confidence=MAX(episode_entities.confidence, excluded.confidence)""",
                [(episode_id, narrative_id) for episode_id in episodes],
            )
            episode_links = connection.total_changes - before
        return {
            "evidence_links": evidence_links,
            "episode_links": episode_links,
        }

    def upsert_derived_edge(
        self,
        edge_id: str,
        source_id: str,
        relation: str,
        target_id: str,
        confidence: float,
        confirmations: int,
        at: datetime,
        metadata: dict[str, Any],
    ) -> str:
        """Project a stable derived synapse without inflating repeat counts."""
        with self._transaction() as connection:
            connection.execute(
                """INSERT INTO edges
                (edge_id, source_id, relation, target_id, confidence, valid_from,
                 confirmation_count, state, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?)
                ON CONFLICT(edge_id) DO UPDATE SET confidence=excluded.confidence,
                confirmation_count=excluded.confirmation_count, state='active',
                metadata_json=excluded.metadata_json""",
                (
                    edge_id,
                    source_id,
                    relation,
                    target_id,
                    max(0.0, min(1.0, float(confidence))),
                    _timestamp(at),
                    max(1, int(confirmations)),
                    json.dumps(metadata, sort_keys=True),
                ),
            )
        return edge_id

    def recent_interaction_outcomes(self, limit: int = 200) -> list[dict[str, Any]]:
        """Return bounded observed dialogue outcomes for strategy consolidation."""
        bounded = max(1, min(int(limit), 1000))
        with self._lock:
            rows = self._connection.execute(
                """SELECT evidence_id, captured_at, quality, payload_json
                FROM evidence WHERE modality='action'
                AND source_type='interaction-policy'
                ORDER BY captured_at DESC LIMIT ?""",
                (bounded,),
            ).fetchall()
        return [_row(row) for row in rows]

    def find_entity_by_source(self, source_system: str, source_profile_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                """SELECT * FROM entities WHERE json_extract(metadata_json, '$.source_system')=?
                AND json_extract(metadata_json, '$.source_profile_id')=? LIMIT 1""",
                (source_system, source_profile_id),
            ).fetchone()
        return _row(row) if row else None

    def list_claims(
        self, subject_id: str | None = None, state: str | None = "active", limit: int | None = None
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if subject_id:
            clauses.append("subject_id=?")
            values.append(subject_id)
        if state:
            clauses.append("state=?")
            values.append(state)
        values.append(limit or self.config.retrieval_limit)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM claims {where} ORDER BY created_at DESC LIMIT ?", values
            ).fetchall()
        return [_row(row) for row in rows]

    def embedding_records(
        self, modality: str | None = None, owner_type: str | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if modality:
            clauses.append("modality=?")
            values.append(modality)
        if owner_type:
            clauses.append("owner_type=?")
            values.append(owner_type)
        values.append(min(limit or self.config.graph_max_nodes, self.config.graph_max_nodes))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM embeddings {where} ORDER BY created_at DESC LIMIT ?", values
            ).fetchall()
        records = []
        for row in rows:
            record = dict(row)
            record["vector"] = np.frombuffer(record.pop("vector_blob"), dtype=np.float32).copy()
            records.append(record)
        return records

    def embedding_metadata(self, owner_id: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        values: list[Any] = []
        where = ""
        if owner_id:
            where = "WHERE owner_id=?"
            values.append(owner_id)
        values.append(limit or self.config.retrieval_limit)
        with self._lock:
            rows = self._connection.execute(
                f"""SELECT embedding_id, owner_type, owner_id, modality, model_id, dimensions, quality, created_at
                FROM embeddings {where} ORDER BY created_at DESC LIMIT ?""", values
            ).fetchall()
        return [dict(row) for row in rows]

    def graph_neighbors(self, entity_ids: list[str], max_hops: int | None = None) -> list[dict[str, Any]]:
        frontier = set(entity_ids)
        visited = set(entity_ids)
        results: list[dict[str, Any]] = []
        for hop in range(1, min(max_hops or self.config.graph_max_hops, self.config.graph_max_hops) + 1):
            if not frontier or len(visited) >= self.config.graph_max_nodes:
                break
            placeholders = ",".join("?" for _ in frontier)
            with self._lock:
                rows = self._connection.execute(
                    f"""SELECT * FROM edges WHERE state='active' AND
                    (source_id IN ({placeholders}) OR target_id IN ({placeholders}))
                    ORDER BY confidence DESC LIMIT ?""",
                    (*frontier, *frontier, self.config.graph_max_nodes - len(visited)),
                ).fetchall()
            next_frontier: set[str] = set()
            for row in rows:
                item = _row(row)
                item["hop"] = hop
                results.append(item)
                for node in (str(row["source_id"]), str(row["target_id"])):
                    if node not in visited:
                        next_frontier.add(node)
                        visited.add(node)
            frontier = next_frontier
        return results[: self.config.graph_max_nodes]

    def knowledge_graph_snapshot(self, node_limit: int = 1500) -> dict[str, object]:
        """Return a bounded, presentation-safe multimodal property graph.

        Entity nodes include people, appearance tracks, objects, categories, and
        OCR-derived content. Recent evidence, claims, and episodes remain distinct
        nodes so the browser can show cross-modal provenance instead of flattening
        everything into labels.
        """
        entity_limit = max(50, min(int(node_limit), 2000))
        evidence_limit = min(500, max(50, entity_limit // 3))
        claim_limit = min(500, max(50, entity_limit // 3))
        episode_limit = min(200, max(25, entity_limit // 8))
        link_limit = min(6000, entity_limit * 6)
        with self._lock:
            entities = self._connection.execute(
                """SELECT * FROM entities WHERE state='active' AND merged_into IS NULL
                ORDER BY updated_at DESC LIMIT ?""",
                (entity_limit,),
            ).fetchall()
            evidence = self._connection.execute(
                "SELECT * FROM evidence ORDER BY captured_at DESC LIMIT ?",
                (evidence_limit,),
            ).fetchall()
            claims = self._connection.execute(
                "SELECT * FROM claims WHERE state='active' ORDER BY created_at DESC LIMIT ?",
                (claim_limit,),
            ).fetchall()
            episodes = self._connection.execute(
                "SELECT * FROM episodes ORDER BY started_at DESC LIMIT ?",
                (episode_limit,),
            ).fetchall()
            edges = self._connection.execute(
                "SELECT * FROM edges WHERE state='active' ORDER BY confidence DESC LIMIT ?",
                (link_limit,),
            ).fetchall()
            entity_evidence = self._connection.execute(
                """SELECT ee.entity_id, ee.evidence_id, ee.role
                FROM entity_evidence ee JOIN evidence ev ON ev.evidence_id=ee.evidence_id
                ORDER BY ev.captured_at DESC LIMIT ?""",
                (link_limit,),
            ).fetchall()
            episode_entities = self._connection.execute(
                """SELECT episode_id, entity_id, role, confidence
                FROM episode_entities ORDER BY rowid DESC LIMIT ?""",
                (link_limit,),
            ).fetchall()
            episode_evidence = self._connection.execute(
                """SELECT episode_id, evidence_id, role
                FROM episode_evidence ORDER BY rowid DESC LIMIT ?""",
                (link_limit,),
            ).fetchall()

        entity_records = [_row(row) for row in entities]
        evidence_records = [_row(row) for row in evidence]
        claim_records = [_row(row) for row in claims]
        episode_records = [_row(row) for row in episodes]
        entity_ids = {str(item["entity_id"]) for item in entity_records}
        evidence_ids = {str(item["evidence_id"]) for item in evidence_records}
        episode_ids = {str(item["episode_id"]) for item in episode_records}

        nodes: list[dict[str, object]] = []
        for item in entity_records:
            label = item.get("display_name") or item["entity_id"]
            if item["entity_type"] == "appearance_track" and not item.get("display_name"):
                suffix = str(item["entity_id"]).removeprefix("person-")
                label = f"Unconfirmed observation {suffix}"
            nodes.append(
                {
                    "id": f"entity:{item['entity_id']}",
                    "source_id": item["entity_id"],
                    "kind": "entity",
                    "subtype": item["entity_type"],
                    "label": label,
                    "updated_at": item.get("updated_at"),
                    "metadata": item.get("metadata", {}),
                }
            )
        for item in evidence_records:
            payload = item.get("payload", {})
            label = (
                (payload.get("text") or payload.get("transcript"))
                if isinstance(payload, dict) else None
            ) or f"{item['modality']} evidence"
            nodes.append(
                {
                    "id": f"evidence:{item['evidence_id']}",
                    "source_id": item["evidence_id"],
                    "kind": "evidence",
                    "subtype": item["modality"],
                    "label": str(label)[:120],
                    "confidence": item.get("quality", 0.0),
                    "updated_at": item.get("captured_at"),
                    "metadata": payload,
                }
            )
        for item in claim_records:
            nodes.append(
                {
                    "id": f"claim:{item['claim_id']}",
                    "source_id": item["claim_id"],
                    "kind": "claim",
                    "subtype": item["predicate"],
                    "label": f"{item['predicate']}: {item['object_id_or_text']}",
                    "confidence": item.get("confidence", 0.0),
                    "updated_at": item.get("created_at"),
                    "metadata": item.get("metadata", {}),
                }
            )
        for item in episode_records:
            nodes.append(
                {
                    "id": f"episode:{item['episode_id']}",
                    "source_id": item["episode_id"],
                    "kind": "episode",
                    "subtype": item.get("state", "episode"),
                    "label": item.get("summary") or f"Episode {str(item['episode_id'])[:8]}",
                    "confidence": item.get("novelty", 0.0),
                    "updated_at": item.get("started_at"),
                    "metadata": {"ended_at": item.get("ended_at")},
                }
            )

        links: list[dict[str, object]] = []
        for row in edges:
            item = _row(row)
            if str(item["source_id"]) not in entity_ids or str(item["target_id"]) not in entity_ids:
                continue
            links.append(
                {
                    "id": f"edge:{item['edge_id']}",
                    "source": f"entity:{item['source_id']}",
                    "target": f"entity:{item['target_id']}",
                    "relation": item["relation"],
                    "confidence": item.get("confidence", 0.0),
                    "confirmations": item.get("confirmation_count", 1),
                }
            )
        for row in entity_evidence:
            if str(row["entity_id"]) in entity_ids and str(row["evidence_id"]) in evidence_ids:
                links.append(
                    {
                        "id": f"entity-evidence:{row['entity_id']}:{row['evidence_id']}:{row['role']}",
                        "source": f"entity:{row['entity_id']}",
                        "target": f"evidence:{row['evidence_id']}",
                        "relation": row["role"],
                        "confidence": 0.8,
                        "confirmations": 1,
                    }
                )
        for item in claim_records:
            if str(item["subject_id"]) in entity_ids:
                links.append(
                    {
                        "id": f"entity-claim:{item['claim_id']}",
                        "source": f"entity:{item['subject_id']}",
                        "target": f"claim:{item['claim_id']}",
                        "relation": item["predicate"],
                        "confidence": item.get("confidence", 0.0),
                        "confirmations": 1,
                    }
                )
        for row in episode_entities:
            if str(row["episode_id"]) in episode_ids and str(row["entity_id"]) in entity_ids:
                links.append(
                    {
                        "id": f"episode-entity:{row['episode_id']}:{row['entity_id']}:{row['role']}",
                        "source": f"episode:{row['episode_id']}",
                        "target": f"entity:{row['entity_id']}",
                        "relation": row["role"],
                        "confidence": row["confidence"],
                        "confirmations": 1,
                    }
                )
        for row in episode_evidence:
            if str(row["episode_id"]) in episode_ids and str(row["evidence_id"]) in evidence_ids:
                links.append(
                    {
                        "id": f"episode-evidence:{row['episode_id']}:{row['evidence_id']}:{row['role']}",
                        "source": f"episode:{row['episode_id']}",
                        "target": f"evidence:{row['evidence_id']}",
                        "relation": row["role"],
                        "confidence": 0.7,
                        "confirmations": 1,
                    }
                )
        return {
            "nodes": nodes,
            "links": links[:link_limit],
            "counts": {
                "entities": len(entity_records),
                "evidence": len(evidence_records),
                "claims": len(claim_records),
                "episodes": len(episode_records),
                "links": min(len(links), link_limit),
            },
            "node_limit": entity_limit,
        }

    def search_evidence(self, terms: list[str], limit: int | None = None) -> list[dict[str, Any]]:
        normalized = [term.casefold().strip() for term in terms if term.strip()]
        if not normalized:
            return []
        clauses = ["LOWER(payload_json) LIKE ?" for _ in normalized]
        values: list[Any] = [f"%{term}%" for term in normalized]
        values.append(limit or self.config.retrieval_limit)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM evidence WHERE {' OR '.join(clauses)} ORDER BY captured_at DESC LIMIT ?", values
            ).fetchall()
        return [_row(row) for row in rows]

    def conversation_history(self, limit: int = 5000) -> list[dict[str, Any]]:
        """Return durable turns plus the evidence that informed or changed them.

        Every async modality writes the originating utterance/context ID. This
        lets late vision, tool, correction, retrieval, and audio-semantic results
        enrich an existing turn instead of manufacturing duplicate messages.
        """
        bounded_limit = max(1, min(int(limit), 20000))
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM (
                    SELECT evidence_id, modality, captured_at, payload_json
                    FROM evidence
                    WHERE modality IN ('audio', 'speech', 'action', 'audio_semantics')
                    AND (
                        payload_json LIKE '%\"transcript\"%'
                        OR payload_json LIKE '%\"candidate_response\"%'
                        OR payload_json LIKE '%\"context_id\"%'
                    )
                    ORDER BY captured_at DESC
                    LIMIT ?
                ) ORDER BY captured_at ASC""",
                (bounded_limit,),
            ).fetchall()
            evidence_ids = [str(row["evidence_id"]) for row in rows]
            associations: dict[str, list[dict[str, Any]]] = defaultdict(list)
            if evidence_ids:
                for offset in range(0, len(evidence_ids), 800):
                    batch = evidence_ids[offset : offset + 800]
                    placeholders = ",".join("?" for _ in batch)
                    linked = self._connection.execute(
                        f"""SELECT entity_evidence.evidence_id, entity_evidence.role,
                        entities.entity_id, entities.entity_type, entities.display_name
                        FROM entity_evidence JOIN entities
                        ON entities.entity_id=entity_evidence.entity_id
                        WHERE entity_evidence.evidence_id IN ({placeholders})""",
                        batch,
                    ).fetchall()
                    for item in linked:
                        associations[str(item["evidence_id"])].append(
                            {
                                "id": str(item["entity_id"]),
                                "type": str(item["entity_type"]),
                                "label": str(
                                    item["display_name"] or item["entity_id"]
                                )[:120],
                                "role": str(item["role"]),
                            }
                        )

        parsed_by_id: dict[str, tuple[dict[str, Any], str]] = {}
        context_metadata: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: {"tags": [], "associations": [], "tool_calls": []}
        )

        def add_unique(target: list[dict[str, Any]], item: dict[str, Any]) -> None:
            identity = tuple(sorted((key, json.dumps(value, sort_keys=True)) for key, value in item.items()))
            if not any(
                tuple(sorted((key, json.dumps(value, sort_keys=True)) for key, value in current.items()))
                == identity
                for current in target
            ):
                target.append(item)

        for row in rows:
            try:
                payload = json.loads(str(row["payload_json"] or "{}"))
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            context_value = payload.get("context_id") or payload.get("utterance_id")
            context_id = (
                str(context_value)
                if isinstance(context_value, str) and context_value
                else f"evidence:{row['evidence_id']}"
            )
            parsed_by_id[str(row["evidence_id"])] = (payload, context_id)
            metadata = context_metadata[context_id]
            modality = str(row["modality"])
            if modality == "audio":
                add_unique(metadata["tags"], {"kind": "modality", "label": "audio"})
                if str(payload.get("asr_model") or "").casefold() == "dual":
                    add_unique(
                        metadata["tags"],
                        {"kind": "modality", "label": "dual ASR"},
                    )
                doa = payload.get("doa")
                if isinstance(doa, (int, float)):
                    add_unique(
                        metadata["tags"],
                        {"kind": "association", "label": f"DoA {round(float(doa))}°"},
                    )
            if modality == "audio_semantics":
                add_unique(
                    metadata["tags"],
                    {"kind": "tool", "label": "audio comprehension ✓"},
                )
                classifications = payload.get("classifications")
                for item in classifications[:4] if isinstance(classifications, list) else []:
                    if isinstance(item, dict) and item.get("label"):
                        add_unique(
                            metadata["tags"],
                            {
                                "kind": "modality",
                                "label": (
                                    f"{item['label']} "
                                    f"{float(item.get('confidence') or 0):.0%}"
                                )[:120],
                            },
                        )
            memory_label = None
            if payload.get("preferred_name"):
                memory_label = f"remembered name: {payload['preferred_name']}"
            elif payload.get("corrected_label"):
                memory_label = f"label updated: {payload['corrected_label']}"
            elif payload.get("predicate"):
                memory_label = f"claim updated: {payload['predicate']}"
            if memory_label:
                add_unique(
                    metadata["tags"],
                    {"kind": "memory", "label": memory_label[:120]},
                )
            retrieval = payload.get("retrieval_influences")
            if isinstance(retrieval, list) and retrieval:
                add_unique(
                    metadata["tags"],
                    {"kind": "memory", "label": f"memory recall ×{len(retrieval)}"},
                )
            tool_calls = payload.get("tool_calls")
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if not isinstance(call, dict) or not call.get("name"):
                        continue
                    normalized_call = {
                        "name": str(call["name"])[:80],
                        "success": bool(call.get("success")),
                        "duration_ms": call.get("duration_ms"),
                    }
                    add_unique(metadata["tool_calls"], normalized_call)
                    add_unique(
                        metadata["tags"],
                        {
                            "kind": "tool",
                            "label": (
                                str(call["name"]).replace("_", " ")
                                + (" ✓" if call.get("success") else " failed")
                            )[:120],
                        },
                    )
            for association in associations.get(str(row["evidence_id"]), []):
                add_unique(metadata["associations"], association)
                add_unique(
                    metadata["tags"],
                    {
                        "kind": "association",
                        "label": f"{association['type']}: {association['label']}"[:120],
                    },
                )

        history: list[dict[str, Any]] = []
        heard_by_context: dict[str, dict[str, Any]] = {}
        for row in rows:
            matched = parsed_by_id.get(str(row["evidence_id"]))
            if matched is None:
                continue
            payload, context_id = matched
            transcript = payload.get("transcript")
            response = payload.get("candidate_response")
            if isinstance(transcript, str) and transcript.strip():
                normalized = " ".join(transcript.split())[:2000]
                existing = heard_by_context.get(context_id)
                if existing is None and history and history[-1].get("role") == "heard" and history[-1].get("text") == normalized:
                    existing = history[-1]
                if existing is None:
                    existing = {
                        "id": str(row["evidence_id"]),
                        "context_id": context_id,
                        "role": "heard",
                        "text": normalized,
                        "status": "final",
                        "at": str(row["captured_at"]),
                    }
                    history.append(existing)
                heard_by_context[context_id] = existing
            elif isinstance(response, str) and response.strip():
                spoken = bool(payload.get("spoken"))
                history.append(
                    {
                        "id": str(row["evidence_id"]),
                        "context_id": context_id,
                        "role": "agent",
                        "text": " ".join(response.split())[:2000],
                        "status": "spoken" if spoken else "suppressed",
                        "at": str(row["captured_at"]),
                        "reason": str(payload.get("reason") or "")[:300],
                    }
                )
        for turn in history:
            metadata = context_metadata.get(str(turn.get("context_id")), {})
            turn["tags"] = list(metadata.get("tags", []))[:16]
            turn["associations"] = list(metadata.get("associations", []))[:12]
            turn["tool_calls"] = list(metadata.get("tool_calls", []))[:12]
        return history

    def associations_for_evidence(self, evidence_id: str) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            entities = self._connection.execute(
                """SELECT entities.*, entity_evidence.role FROM entities JOIN entity_evidence
                ON entities.entity_id=entity_evidence.entity_id WHERE entity_evidence.evidence_id=?""",
                (evidence_id,),
            ).fetchall()
            episodes = self._connection.execute(
                """SELECT episodes.*, episode_evidence.role FROM episodes JOIN episode_evidence
                ON episodes.episode_id=episode_evidence.episode_id WHERE episode_evidence.evidence_id=?""",
                (evidence_id,),
            ).fetchall()
        return {
            "entities": [_row(row) for row in entities],
            "episodes": [_row(row) for row in episodes],
        }

    def evidence_detail(self, evidence_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM evidence WHERE evidence_id=?", (evidence_id,)
            ).fetchone()
            if row is None:
                return None
            claims = self._connection.execute(
                "SELECT * FROM claims WHERE evidence_id=? ORDER BY created_at DESC",
                (evidence_id,),
            ).fetchall()
            edges = self._connection.execute(
                "SELECT * FROM edges WHERE evidence_id=? ORDER BY confidence DESC",
                (evidence_id,),
            ).fetchall()
        evidence = _row(row)
        if evidence.get("media_key"):
            evidence["artifact_url"] = f"/api/memory/evidence/{evidence_id}/media"
        return {
            "evidence": evidence,
            **self.associations_for_evidence(evidence_id),
            "claims": [_row(item) for item in claims],
            "edges": [_row(item) for item in edges],
        }

    def graph_node_detail(self, kind: str, source_id: str) -> dict[str, Any] | None:
        if kind == "entity":
            return self.entity_detail(source_id)
        if kind == "evidence":
            return self.evidence_detail(source_id)
        if kind == "episode":
            return self.episode_detail(source_id)
        if kind == "claim":
            claim = self.claim_detail(source_id)
            if claim is None:
                return None
            result: dict[str, Any] = {"claim": claim}
            result["subject"] = self.entity_detail(str(claim["subject_id"]))
            if claim.get("evidence_id"):
                result["evidence_detail"] = self.evidence_detail(str(claim["evidence_id"]))
            return result
        return None

    def episode_detail(self, episode_id: str) -> dict[str, Any] | None:
        with self._lock:
            episode = self._connection.execute(
                "SELECT * FROM episodes WHERE episode_id=?", (episode_id,)
            ).fetchone()
            if episode is None:
                return None
            evidence = self._connection.execute(
                """SELECT evidence.*, episode_evidence.role FROM evidence JOIN episode_evidence
                ON evidence.evidence_id=episode_evidence.evidence_id
                WHERE episode_evidence.episode_id=? ORDER BY evidence.captured_at""",
                (episode_id,),
            ).fetchall()
            entities = self._connection.execute(
                """SELECT entities.*, episode_entities.role, episode_entities.confidence FROM entities
                JOIN episode_entities ON entities.entity_id=episode_entities.entity_id
                WHERE episode_entities.episode_id=?""",
                (episode_id,),
            ).fetchall()
        return {
            "episode": _row(episode),
            "evidence": [_row(row) for row in evidence],
            "entities": [_row(row) for row in entities],
        }

    def episodes_without_summaries(self, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM episodes WHERE state='closed' AND (summary IS NULL OR summary='')
                ORDER BY novelty DESC, started_at DESC LIMIT ?""", (limit,)
            ).fetchall()
        return [_row(row) for row in rows]

    def set_episode_summary(self, episode_id: str, summary: str) -> None:
        with self._transaction() as connection:
            connection.execute(
                "UPDATE episodes SET summary=? WHERE episode_id=? AND (summary IS NULL OR summary='')",
                (summary, episode_id),
            )

    def conflicting_claims(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            groups = self._connection.execute(
                """SELECT subject_id, predicate, COUNT(DISTINCT object_id_or_text) AS values_count
                FROM claims WHERE state='active' GROUP BY subject_id, predicate
                HAVING values_count > 1 ORDER BY values_count DESC LIMIT ?""", (limit,)
            ).fetchall()
        return [dict(row) for row in groups]

    def expire_media_before(self, cutoff: datetime, limit: int) -> int:
        with self._transaction() as connection:
            rows = connection.execute(
                """SELECT evidence_id, media_key FROM evidence WHERE media_key IS NOT NULL AND captured_at < ?
                ORDER BY captured_at LIMIT ?""", (_timestamp(cutoff), limit)
            ).fetchall()
            for row in rows:
                path = (self.root / str(row["media_key"])).resolve()
                if self.root.resolve() in path.parents and path.is_file():
                    path.unlink()
                connection.execute("UPDATE evidence SET media_key=NULL WHERE evidence_id=?", (row["evidence_id"],))
        return len(rows)

    def delete_evidence_before(self, cutoff: datetime, limit: int) -> int:
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT evidence_id FROM evidence WHERE captured_at < ? ORDER BY captured_at LIMIT ?",
                (_timestamp(cutoff), limit),
            ).fetchall()
            evidence_ids = [str(row["evidence_id"]) for row in rows]
            for evidence_id in evidence_ids:
                connection.execute("UPDATE claims SET evidence_id=NULL WHERE evidence_id=?", (evidence_id,))
                connection.execute("UPDATE edges SET evidence_id=NULL WHERE evidence_id=?", (evidence_id,))
                connection.execute("UPDATE revisions SET evidence_id=NULL WHERE evidence_id=?", (evidence_id,))
                connection.execute("DELETE FROM evidence WHERE evidence_id=?", (evidence_id,))
        return len(evidence_ids)

    def memory_stats(self) -> dict[str, int]:
        with self._lock:
            return {
                table: int(self._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in (
                    "entities", "episodes", "claims", "edges", "evidence", "embeddings", "revisions", "jobs"
                )
            }

    def integrity_report(self) -> dict[str, object]:
        with self._lock:
            integrity = str(self._connection.execute("PRAGMA integrity_check").fetchone()[0])
            journal_mode = str(self._connection.execute("PRAGMA journal_mode").fetchone()[0])
            query_only = int(self._connection.execute("PRAGMA query_only").fetchone()[0])
            foreign_keys = [dict(row) for row in self._connection.execute("PRAGMA foreign_key_check").fetchall()]
            orphan_embeddings = int(
                self._connection.execute(
                    """SELECT COUNT(*) FROM embeddings WHERE owner_type='entity'
                    AND owner_id NOT IN (SELECT entity_id FROM entities)"""
                ).fetchone()[0]
            )
            duplicate_sources = [
                dict(row) for row in self._connection.execute(
                    """SELECT json_extract(metadata_json, '$.source_system') AS source_system,
                    json_extract(metadata_json, '$.source_profile_id') AS source_profile_id, COUNT(*) AS count
                    FROM entities WHERE json_extract(metadata_json, '$.source_system') IS NOT NULL
                    AND json_extract(metadata_json, '$.source_profile_id') IS NOT NULL
                    GROUP BY source_system, source_profile_id HAVING count > 1"""
                ).fetchall()
            ]
            open_episodes = int(
                self._connection.execute("SELECT COUNT(*) FROM episodes WHERE state='open'").fetchone()[0]
            )
        return {
            "sqlite_integrity": integrity,
            "journal_mode": journal_mode,
            "writable": query_only == 0 and self.root.is_dir() and os.access(self.root, os.W_OK),
            "database_path": str((self.root / "memory.sqlite3").resolve()),
            "foreign_key_violations": foreign_keys,
            "orphan_entity_embeddings": orphan_embeddings,
            "duplicate_legacy_sources": duplicate_sources,
            "open_episodes": open_episodes,
            "claim_conflicts": self.conflicting_claims(limit=50),
            "stats": self.memory_stats(),
            "media": self.media_integrity_report(),
        }

    def media_integrity_report(self) -> dict[str, object]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT evidence_id, media_key, checksum FROM evidence WHERE media_key IS NOT NULL"
            ).fetchall()
        missing: list[str] = []
        checksum_mismatches: list[str] = []
        for row in rows:
            path = (self.root / str(row["media_key"])).resolve()
            if self.root.resolve() not in path.parents or not path.is_file():
                missing.append(str(row["evidence_id"]))
                continue
            expected = row["checksum"]
            if expected and hashlib.sha256(path.read_bytes()).hexdigest() != str(expected):
                checksum_mismatches.append(str(row["evidence_id"]))
        return {
            "referenced": len(rows),
            "missing_evidence_ids": missing,
            "checksum_mismatch_evidence_ids": checksum_mismatches,
        }

    def create_job(self, kind: str, payload: dict[str, Any] | None = None, job_id: str | None = None) -> str:
        job_id = job_id or _identifier()
        created_at = _timestamp(datetime.now(timezone.utc))
        with self._transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO jobs (job_id, kind, state, payload_json, created_at) VALUES (?, ?, 'pending', ?, ?)",
                (job_id, kind, json.dumps(payload or {}, sort_keys=True), created_at),
            )
        return job_id

    def update_job(self, job_id: str, state: str, error: str | None = None) -> None:
        timestamp = _timestamp(datetime.now(timezone.utc))
        started_at = timestamp if state == "running" else None
        completed_at = timestamp if state in {"complete", "failed"} else None
        with self._transaction() as connection:
            connection.execute(
                """UPDATE jobs SET state=?, error=?, started_at=COALESCE(?, started_at),
                completed_at=COALESCE(?, completed_at) WHERE job_id=?""",
                (state, error, started_at, completed_at, job_id),
            )

    def list_jobs(self, state: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        values: list[Any] = []
        where = ""
        if state:
            where = "WHERE state=?"
            values.append(state)
        values.append(limit)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM jobs {where} ORDER BY created_at DESC LIMIT ?", values
            ).fetchall()
        return [_row(row) for row in rows]

    def entity_detail(self, entity_id: str) -> dict[str, Any] | None:
        with self._lock:
            entity = self._connection.execute("SELECT * FROM entities WHERE entity_id=?", (entity_id,)).fetchone()
            if entity is None:
                return None
            claims = self._connection.execute("SELECT * FROM claims WHERE subject_id=? ORDER BY created_at DESC", (entity_id,)).fetchall()
            evidence = self._connection.execute(
                """SELECT evidence.*, entity_evidence.role FROM evidence JOIN entity_evidence
                ON evidence.evidence_id=entity_evidence.evidence_id WHERE entity_evidence.entity_id=?""", (entity_id,)
            ).fetchall()
            episodes = self._connection.execute(
                """SELECT episodes.*, episode_entities.role, episode_entities.confidence FROM episodes
                JOIN episode_entities ON episodes.episode_id=episode_entities.episode_id
                WHERE episode_entities.entity_id=? ORDER BY episodes.started_at DESC LIMIT ?""",
                (entity_id, self.config.retrieval_limit),
            ).fetchall()
        return {
            "entity": _row(entity),
            "claims": [_row(row) for row in claims],
            "evidence": [_row(row) for row in evidence],
            "episodes": [_row(row) for row in episodes],
            "embeddings": self.embedding_metadata(entity_id),
        }

    def claim_detail(self, claim_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute("SELECT * FROM claims WHERE claim_id=?", (claim_id,)).fetchone()
        return _row(row) if row else None

    def delete_entity_cascade(self, entity_id: str) -> None:
        with self._transaction() as connection:
            connection.execute("DELETE FROM edges WHERE source_id=? OR target_id=?", (entity_id, entity_id))
            connection.execute("DELETE FROM embeddings WHERE owner_type='entity' AND owner_id=?", (entity_id,))
            claim_ids = [
                str(row["claim_id"]) for row in connection.execute(
                    "SELECT claim_id FROM claims WHERE subject_id=?", (entity_id,)
                ).fetchall()
            ]
            for claim_id in claim_ids:
                connection.execute(
                    "DELETE FROM revisions WHERE target_type='claim' AND target_id=?", (claim_id,)
                )
                connection.execute(
                    "DELETE FROM embeddings WHERE owner_type='claim' AND owner_id=?", (claim_id,)
                )
            connection.execute("DELETE FROM claims WHERE subject_id=?", (entity_id,))
            connection.execute("UPDATE entities SET merged_into=NULL WHERE merged_into=?", (entity_id,))
            connection.execute("DELETE FROM entities WHERE entity_id=?", (entity_id,))
