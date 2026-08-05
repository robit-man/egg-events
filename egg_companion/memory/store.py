from __future__ import annotations

import hashlib
import json
import hashlib
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from egg_companion.config import MemoryConfig
from egg_companion.memory.schema import migrate
from egg_companion.models import EvidenceRef


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
