"""World-model metrics and diagnostics."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class WorldMetrics:
    total_entities: int = 0
    total_assertions: int = 0
    total_relations: int = 0
    total_events: int = 0
    total_evidence: int = 0
    total_functions: int = 0
    total_action_proposals: int = 0
    total_action_executions: int = 0
    conflicts: int = 0
    ontology_types: int = 0
    identity_chains: int = 0
    avg_confidence: float = 0.0
    avg_authority: float = 0.0
    avg_evidence_per_assertion: float = 0.0


class MetricsCollector:
    """Collects and reports world-model metrics."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection
        self._lock = threading.RLock()

    def collect(self) -> WorldMetrics:
        with self._lock:
            metrics = WorldMetrics()
            try:
                row = self._conn.execute("SELECT COUNT(*) FROM current_property_state").fetchone()
                metrics.total_entities = row[0] if row else 0
            except Exception:
                pass
            try:
                row = self._conn.execute("SELECT COUNT(*) FROM world_assertions").fetchone()
                metrics.total_assertions = row[0] if row else 0
            except Exception:
                pass
            try:
                row = self._conn.execute("SELECT COUNT(*) FROM current_relation_state WHERE valid_to IS NULL").fetchone()
                metrics.total_relations = row[0] if row else 0
            except Exception:
                pass
            try:
                row = self._conn.execute("SELECT COUNT(*) FROM world_events").fetchone()
                metrics.total_events = row[0] if row else 0
            except Exception:
                pass
            try:
                row = self._conn.execute("SELECT COUNT(DISTINCT episode_id) FROM evidence").fetchone()
                metrics.total_evidence = row[0] if row else 0
            except Exception:
                pass
            try:
                row = self._conn.execute("SELECT COUNT(*) FROM function_registry").fetchone()
                metrics.total_functions = row[0] if row else 0
            except Exception:
                pass
            try:
                row = self._conn.execute("SELECT COUNT(*) FROM action_proposals").fetchone()
                metrics.total_action_proposals = row[0] if row else 0
            except Exception:
                pass
            try:
                row = self._conn.execute("SELECT COUNT(*) FROM action_executions").fetchone()
                metrics.total_action_executions = row[0] if row else 0
            except Exception:
                pass
            try:
                row = self._conn.execute("SELECT COUNT(*) FROM world_assertions WHERE state = 'conflicted'").fetchone()
                metrics.conflicts = row[0] if row else 0
            except Exception:
                pass
            try:
                row = self._conn.execute("SELECT COUNT(*) FROM ontology_types").fetchone()
                metrics.ontology_types = row[0] if row else 0
            except Exception:
                pass
            try:
                row = self._conn.execute("SELECT COUNT(*) FROM identity_chains").fetchone()
                metrics.identity_chains = row[0] if row else 0
            except Exception:
                pass
            try:
                row = self._conn.execute("SELECT AVG(confidence) FROM world_assertions WHERE valid_to IS NULL").fetchone()
                metrics.avg_confidence = row[0] if row and row[0] is not None else 0.0
            except Exception:
                pass
            try:
                row = self._conn.execute("SELECT AVG(authority) FROM world_assertions WHERE valid_to IS NULL").fetchone()
                metrics.avg_authority = row[0] if row and row[0] is not None else 0.0
            except Exception:
                pass
            try:
                row = self._conn.execute("SELECT AVG(evidence_count) FROM (SELECT json_array_length(evidence_ids_json) as evidence_count FROM world_assertions)").fetchone()
                metrics.avg_evidence_per_assertion = row[0] if row and row[0] is not None else 0.0
            except Exception:
                pass
            return metrics

    def to_dict(self, metrics: WorldMetrics | None = None) -> dict[str, Any]:
        m = metrics or self.collect()
        return {
            "total_entities": m.total_entities,
            "total_assertions": m.total_assertions,
            "total_relations": m.total_relations,
            "total_events": m.total_events,
            "total_evidence": m.total_evidence,
            "total_functions": m.total_functions,
            "total_action_proposals": m.total_action_proposals,
            "total_action_executions": m.total_action_executions,
            "conflicts": m.conflicts,
            "ontology_types": m.ontology_types,
            "identity_chains": m.identity_chains,
            "avg_confidence": round(m.avg_confidence, 4),
            "avg_authority": round(m.avg_authority, 4),
            "avg_evidence_per_assertion": round(m.avg_evidence_per_assertion, 4),
        }

    def record_metrics_snapshot(self, snapshot: dict[str, Any]) -> None:
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS metrics_history (recorded_at TEXT NOT NULL, metrics_json TEXT NOT NULL)"
            )
            self._conn.execute(
                "INSERT INTO metrics_history (recorded_at, metrics_json) VALUES (?, ?)",
                (now, json.dumps(snapshot, default=str)),
            )
            self._conn.commit()

    def history(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT recorded_at, metrics_json FROM metrics_history ORDER BY recorded_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                {"recorded_at": r[0], "metrics": json.loads(r[1])}
                for r in rows
            ]
