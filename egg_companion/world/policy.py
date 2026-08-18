"""Policy validation for action proposals."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any

from egg_companion.world.types import ActionProposal


@dataclass
class PolicyRule:
    rule_id: str
    name: str
    description: str
    action_type: str = "*"
    conditions_json: str = "{}"
    block: bool = False
    require_approval: bool = False
    enabled: bool = True


@dataclass
class PolicyViolation:
    rule_id: str
    rule_name: str
    proposal_id: str
    reason: str
    blocked: bool = False


class PolicyValidator:
    """Validates action proposals against safety policies."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection
        self._lock = threading.RLock()
        self._ensure_tables()
        self._register_defaults()

    def _ensure_tables(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS policy_rules (
                    rule_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    action_type TEXT NOT NULL DEFAULT '*',
                    conditions_json TEXT NOT NULL DEFAULT '{}',
                    block INTEGER NOT NULL DEFAULT 0,
                    require_approval INTEGER NOT NULL DEFAULT 0,
                    enabled INTEGER NOT NULL DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS policy_violations (
                    violation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id TEXT NOT NULL,
                    rule_name TEXT NOT NULL,
                    proposal_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    blocked INTEGER NOT NULL DEFAULT 0,
                    recorded_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS policy_action_log (
                    action_type TEXT NOT NULL,
                    proposal_id TEXT NOT NULL,
                    recorded_at TEXT NOT NULL
                );
                """
            )

    def _register_defaults(self) -> None:
        defaults = [
            PolicyRule("no_destructive", "No Destructive Actions", "Block destructive actions without explicit approval", "destructive_action", require_approval=True),
            PolicyRule("max_frequency", "Max Action Frequency", "Prevent too-frequent actions of same type", "*", conditions_json='{"max_per_minute": 10}'),
            PolicyRule("evidence_required", "Evidence Required", "Require evidence for identity-altering actions", "merge_entities", require_approval=True),
            PolicyRule("safe_zone", "Safe Zone Restrictions", "Restrict actions in safe zones", "move_object", conditions_json='{"safe_zones": ["egg_bed", "egg_nest"]}'),
        ]
        for rule in defaults:
            self.register(rule)

    def register(self, rule: PolicyRule) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO policy_rules
                (rule_id, name, description, action_type, conditions_json, block, require_approval, enabled)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rule.rule_id, rule.name, rule.description,
                    rule.action_type, rule.conditions_json,
                    int(rule.block), int(rule.require_approval), int(rule.enabled),
                ),
            )
            self._conn.commit()

    def validate(self, proposal: ActionProposal) -> list[PolicyViolation]:
        violations = []
        with self._lock:
            rules = self._conn.execute(
                "SELECT rule_id, name, action_type, conditions_json, block, require_approval FROM policy_rules WHERE enabled = 1"
            ).fetchall()
        for rule in rules:
            rule_id, name, action_type, conditions_json, block, require_approval = rule
            if action_type != "*" and action_type != proposal.action_type:
                continue
            conditions = json.loads(conditions_json)
            if "max_per_minute" in conditions:
                max_per = conditions["max_per_minute"]
                import datetime
                cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=1)).isoformat()
                with self._lock:
                    count = self._conn.execute(
                        "SELECT COUNT(*) FROM policy_action_log WHERE action_type = ? AND recorded_at >= ?",
                        (proposal.action_type, cutoff),
                    ).fetchone()[0]
                if count >= max_per:
                    violations.append(PolicyViolation(
                        rule_id=rule_id, rule_name=name,
                        proposal_id=proposal.proposal_id,
                        reason=f"Action frequency {count}/{max_per} per minute exceeded",
                        blocked=bool(block),
                    ))
            if "safe_zones" in conditions:
                safe_zones = conditions["safe_zones"]
                for target_id in proposal.target_entity_ids:
                    zone = self._located_in_zone(target_id, safe_zones)
                    if zone is not None:
                        violations.append(PolicyViolation(
                            rule_id=rule_id, rule_name=name,
                            proposal_id=proposal.proposal_id,
                            reason=f"Target entity '{target_id}' is located_in safe zone '{zone}'",
                            blocked=bool(block),
                        ))
            if require_approval:
                violations.append(PolicyViolation(
                    rule_id=rule_id, rule_name=name,
                    proposal_id=proposal.proposal_id,
                    reason=f"Action type '{proposal.action_type}' requires approval",
                    blocked=False,
                ))
        return violations

    def _located_in_zone(
        self, entity_id: str, safe_zones: list[str], max_depth: int = 5
    ) -> str | None:
        """Return the first safe zone entity_id is transitively located_in.

        Walks ``located_in`` edges in the materialized world state rather
        than substring-matching the zone name against the entity id — an
        entity named e.g. "object:egg_bedspread" must never be treated as
        being inside the "egg_bed" zone just because the string appears in
        its id.
        """

        def matches(candidate: str) -> str | None:
            for zone in safe_zones:
                if candidate == zone or candidate.endswith(f":{zone}"):
                    return zone
            return None

        with self._lock:
            visited: set[str] = set()
            frontier = [entity_id]
            for _ in range(max_depth):
                if not frontier:
                    break
                next_frontier: list[str] = []
                for current in frontier:
                    if current in visited:
                        continue
                    visited.add(current)
                    zone = matches(current)
                    if zone is not None:
                        return zone
                    try:
                        rows = self._conn.execute(
                            "SELECT target_entity_id FROM current_relation_state "
                            "WHERE source_entity_id = ? AND relation_type_id = 'located_in' "
                            "AND valid_to IS NULL",
                            (current,),
                        ).fetchall()
                    except sqlite3.OperationalError:
                        # World state tables aren't present on this connection
                        # (e.g. standalone PolicyValidator usage) — without
                        # location data we can't say the target is in a zone.
                        return None
                    next_frontier.extend(row[0] for row in rows)
                frontier = next_frontier
        return None

    def log_violation(self, violation: PolicyViolation) -> None:
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO policy_violations (rule_id, rule_name, proposal_id, reason, blocked, recorded_at) VALUES (?, ?, ?, ?, ?, ?)",
                (violation.rule_id, violation.rule_name, violation.proposal_id,
                 violation.reason, int(violation.blocked), now),
            )
            self._conn.commit()

    def recent_violations(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT rule_id, rule_name, proposal_id, reason, blocked, recorded_at FROM policy_violations ORDER BY recorded_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                {
                    "rule_id": r[0], "rule_name": r[1], "proposal_id": r[2],
                    "reason": r[3], "blocked": bool(r[4]), "recorded_at": r[5],
                }
                for r in rows
            ]

    def list_rules(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT rule_id, name, description, action_type, block, require_approval, enabled FROM policy_rules ORDER BY rule_id"
            ).fetchall()
            return [
                {
                    "rule_id": r[0], "name": r[1], "description": r[2],
                    "action_type": r[3], "block": bool(r[4]),
                    "require_approval": bool(r[5]), "enabled": bool(r[6]),
                }
                for r in rows
            ]
