"""World graph relations and traversal."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class WorldEdge:
    source_id: str
    relation_type: str
    target_id: str
    confidence: float
    evidence_ids: tuple[str, ...] = ()
    valid_from: str = ""
    valid_to: str | None = None
    properties: dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class WorldGraphStore:
    """Graph store for world-relevant relations with temporal validity."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection
        self._lock = threading.RLock()
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS world_edges (
                    source_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.0,
                    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                    valid_from TEXT NOT NULL,
                    valid_to TEXT,
                    properties_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (source_id, relation_type, target_id, valid_from)
                );

                CREATE INDEX IF NOT EXISTS idx_we_source ON world_edges(source_id);
                CREATE INDEX IF NOT EXISTS idx_we_target ON world_edges(target_id);
                CREATE INDEX IF NOT EXISTS idx_we_relation ON world_edges(relation_type);
                CREATE INDEX IF NOT EXISTS idx_we_valid ON world_edges(valid_from, valid_to);
                """
            )

    def add_edge(self, edge: WorldEdge) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO world_edges
                (source_id, relation_type, target_id, confidence, evidence_ids_json,
                 valid_from, valid_to, properties_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    edge.source_id, edge.relation_type, edge.target_id,
                    edge.confidence, json.dumps(list(edge.evidence_ids)),
                    edge.valid_from, edge.valid_to,
                    json.dumps(edge.properties, default=str),
                ),
            )
            self._conn.commit()

    def neighbors(self, entity_id: str, relation_type: str | None = None, max_depth: int = 1) -> list[dict[str, Any]]:
        with self._lock:
            params: list[Any] = [entity_id]
            if relation_type:
                where = "WHERE (source_id = ? OR target_id = ?) AND valid_to IS NULL AND relation_type = ?"
                params.extend([entity_id, relation_type])
            else:
                where = "WHERE (source_id = ? OR target_id = ?) AND valid_to IS NULL"
                params.append(entity_id)
            rows = self._conn.execute(
                f"SELECT source_id, relation_type, target_id, confidence FROM world_edges {where}",
                params,
            ).fetchall()
            return [
                {
                    "source_id": r[0],
                    "relation_type": r[1],
                    "target_id": r[2],
                    "confidence": r[3],
                }
                for r in rows
            ]

    def outgoing(self, entity_id: str, relation_type: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if relation_type:
                rows = self._conn.execute(
                    "SELECT source_id, relation_type, target_id, confidence FROM world_edges WHERE source_id = ? AND valid_to IS NULL AND relation_type = ?",
                    (entity_id, relation_type),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT source_id, relation_type, target_id, confidence FROM world_edges WHERE source_id = ? AND valid_to IS NULL",
                    (entity_id,),
                ).fetchall()
            return [
                {"source_id": r[0], "relation_type": r[1], "target_id": r[2], "confidence": r[3]}
                for r in rows
            ]

    def incoming(self, entity_id: str, relation_type: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if relation_type:
                rows = self._conn.execute(
                    "SELECT source_id, relation_type, target_id, confidence FROM world_edges WHERE target_id = ? AND valid_to IS NULL AND relation_type = ?",
                    (entity_id, relation_type),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT source_id, relation_type, target_id, confidence FROM world_edges WHERE target_id = ? AND valid_to IS NULL",
                    (entity_id,),
                ).fetchall()
            return [
                {"source_id": r[0], "relation_type": r[1], "target_id": r[2], "confidence": r[3]}
                for r in rows
            ]

    def find_path(self, start: str, end: str, max_depth: int = 5) -> list[str] | None:
        visited: set[str] = set()
        queue: list[tuple[str, list[str]]] = [(start, [start])]
        while queue:
            current, path = queue.pop(0)
            if current == end:
                return path
            if len(path) > max_depth:
                continue
            visited.add(current)
            for neighbor in self.outgoing(current):
                if neighbor["target_id"] not in visited:
                    queue.append((neighbor["target_id"], path + [neighbor["target_id"]]))
        return None

    def all_entity_ids(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT source_id FROM world_edges WHERE valid_to IS NULL "
                "UNION "
                "SELECT DISTINCT target_id FROM world_edges WHERE valid_to IS NULL"
            ).fetchall()
            return [r[0] for r in rows]

    def close_relation(self, source_id: str, relation_type: str, target_id: str, valid_to: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE world_edges SET valid_to = ? WHERE source_id = ? AND relation_type = ? AND target_id = ? AND valid_to IS NULL",
                (valid_to, source_id, relation_type, target_id),
            )
            self._conn.commit()

    def get_relation_history(self, source_id: str, relation_type: str, target_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT confidence, evidence_ids_json, valid_from, valid_to FROM world_edges WHERE source_id = ? AND relation_type = ? AND target_id = ? ORDER BY valid_from DESC",
                (source_id, relation_type, target_id),
            ).fetchall()
            return [
                {"confidence": r[0], "evidence_ids": json.loads(r[1]), "valid_from": r[2], "valid_to": r[3]}
                for r in rows
            ]
