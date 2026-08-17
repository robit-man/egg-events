"""Action proposals, executions, and outcomes."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Any

from egg_companion.world.types import ActionExecution, ActionOutcome, ActionProposal


class ActionStore:
    """Stores action proposals, executions, and outcomes."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection
        self._lock = threading.RLock()
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS action_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    action_type TEXT NOT NULL,
                    target_entity_id TEXT NOT NULL,
                    parameters_json TEXT NOT NULL DEFAULT '{}',
                    source_evidence_id TEXT NOT NULL,
                    proposed_at TEXT NOT NULL,
                    accepted INTEGER NOT NULL DEFAULT 0,
                    rejected INTEGER NOT NULL DEFAULT 0,
                    reason TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS action_executions (
                    execution_id TEXT PRIMARY KEY,
                    proposal_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    success INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT,
                    FOREIGN KEY (proposal_id) REFERENCES action_proposals(proposal_id)
                );

                CREATE TABLE IF NOT EXISTS action_outcomes (
                    outcome_id TEXT PRIMARY KEY,
                    execution_id TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    success INTEGER NOT NULL DEFAULT 0,
                    result TEXT NOT NULL DEFAULT '',
                    side_effects_json TEXT NOT NULL DEFAULT '[]',
                    FOREIGN KEY (execution_id) REFERENCES action_executions(execution_id)
                );
                """
            )

    def propose(self, proposal: ActionProposal) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO action_proposals
                (proposal_id, action_type, target_entity_id, parameters_json,
                 source_evidence_id, proposed_at, accepted, rejected, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    proposal.proposal_id, proposal.action_type,
                    json.dumps(proposal.target_entity_ids),
                    json.dumps(proposal.inputs, default=str),
                    json.dumps(proposal.source_evidence_ids),
                    proposal.proposed_at.isoformat(),
                    int(proposal.status == "accepted"),
                    int(proposal.status == "rejected"),
                    proposal.reason,
                ),
            )
            self._conn.commit()

    def record_execution(self, execution: ActionExecution) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO action_executions
                (execution_id, proposal_id, started_at, completed_at, success, result_json)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    execution.execution_id, execution.proposal_id,
                    execution.started_at.isoformat(),
                    execution.completed_at.isoformat() if execution.completed_at else None,
                    int(execution.success) if execution.success is not None else 0,
                    json.dumps(execution.result, default=str) if execution.result else None,
                ),
            )
            self._conn.commit()

    def record_outcome(self, outcome: ActionOutcome) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO action_outcomes
                (outcome_id, execution_id, observed_at, success, result, side_effects_json)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    outcome.outcome_id, outcome.execution_id,
                    outcome.observed_at.isoformat(), int(outcome.success),
                    json.dumps(outcome.result, default=str) if outcome.result else "",
                    json.dumps(outcome.side_effects, default=str),
                ),
            )
            self._conn.commit()

    def get_proposal(self, proposal_id: str) -> ActionProposal | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT action_type, target_entity_id, parameters_json, source_evidence_id, proposed_at, accepted, rejected, reason FROM action_proposals WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
            if row is None:
                return None
            status = "rejected" if row[6] else ("accepted" if row[5] else "pending")
            return ActionProposal(
                proposal_id=proposal_id, action_type=row[0],
                target_entity_ids=tuple(json.loads(row[1])),
                inputs=json.loads(row[2]),
                source_evidence_ids=tuple(json.loads(row[3])),
                proposed_at=row[4], status=status, reason=row[7],
            )

    def pending_proposals(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT proposal_id, action_type, target_entity_id, parameters_json, proposed_at, reason FROM action_proposals WHERE accepted = 0 AND rejected = 0 ORDER BY proposed_at DESC"
            ).fetchall()
            return [
                {"proposal_id": r[0], "action_type": r[1], "target_entity_id": r[2],
                 "parameters": json.loads(r[3]), "proposed_at": r[4], "reason": r[5]}
                for r in rows
            ]

    def recent_executions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT execution_id, proposal_id, started_at, completed_at, success, result_json FROM action_executions ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                {
                    "execution_id": r[0], "proposal_id": r[1],
                    "started_at": r[2], "completed_at": r[3],
                    "success": bool(r[4]),
                    "result": json.loads(r[5]) if r[5] else None,
                }
                for r in rows
            ]

    def outcomes_for_execution(self, execution_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT outcome_id, observed_at, success, result, side_effects_json FROM action_outcomes WHERE execution_id = ? ORDER BY observed_at",
                (execution_id,),
            ).fetchall()
            return [
                {
                    "outcome_id": r[0], "observed_at": r[1],
                    "success": bool(r[2]), "result": r[3],
                    "side_effects": json.loads(r[4]),
                }
                for r in rows
            ]
