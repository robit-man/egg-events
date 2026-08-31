"""Query API: unified world-model access."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from egg_companion.world.identity import IdentityGraph
from egg_companion.world.relations import WorldGraphStore
from egg_companion.world.spatial import BBox2D, SpatialState
from egg_companion.world.state import WorldStateStore


@dataclass
class EntityViewState:
    entity_id: str
    properties: dict[str, Any]
    relations: list[dict[str, Any]]
    identity_chain: list[str] | None = None
    spatial: SpatialState | None = None
    last_updated: str = ""


@dataclass
class ConflictInfo:
    entity_id: str
    property_id: str
    current_value: Any
    proposed_value: Any
    reason: str
    assertions: list[dict[str, Any]]


class WorldQuery:
    """Unified query API over the entire world model."""

    def __init__(
        self,
        state_store: WorldStateStore,
        graph_store: WorldGraphStore,
        identity_graph: IdentityGraph,
        reconciler: Any = None,
    ) -> None:
        self._state = state_store
        self._graph = graph_store
        self._identity = identity_graph
        self._reconciler = reconciler

    def entity(self, entity_id: str) -> EntityViewState | None:
        props = self._state.get_entity_state(entity_id)
        if not props:
            return None
        relations = self._state.get_entity_relations(entity_id)
        chain = self._identity.get_chain(entity_id)
        return EntityViewState(
            entity_id=entity_id,
            properties={
                k: {
                    "value": json.loads(v.value_json),
                    "value_type": v.value_type,
                    "confidence": v.confidence,
                    "authority": v.authority,
                    "epistemic_kind": v.epistemic_kind,
                    "valid_from": v.valid_from,
                    "valid_to": v.valid_to,
                    "assertion_id": v.assertion_id,
                    "revision": v.revision,
                }
                for k, v in props.items()
            },
            relations=[
                {
                    "source_entity_id": r.source_entity_id,
                    "relation_type_id": r.relation_type_id,
                    "target_entity_id": r.target_entity_id,
                    "confidence": r.confidence,
                    "authority": r.authority,
                    "valid_from": r.valid_from,
                    "valid_to": r.valid_to,
                }
                for r in relations
            ],
            identity_chain=chain.chain if chain else None,
            last_updated=max((v.updated_at for v in props.values()), default=""),
        )

    def entities(self, property_id: str | None = None, value_type: str | None = None) -> list[EntityViewState]:
        all_ids = self._state.all_entity_ids()
        result = []
        for eid in all_ids:
            ev = self.entity(eid)
            if ev is None:
                continue
            if property_id and property_id not in ev.properties:
                continue
            if value_type:
                matches = any(
                    p["value_type"] == value_type
                    for p in ev.properties.values()
                )
                if not matches:
                    continue
            result.append(ev)
        return result

    def property_value(self, entity_id: str, property_id: str) -> Any | None:
        row = self._state.get_property(entity_id, property_id)
        if row is None:
            return None
        return json.loads(row.value_json)

    def explain(self, entity_id: str, property_id: str) -> dict[str, Any]:
        return self._state.explain(entity_id, property_id)

    def explain_entity(self, entity_id: str) -> dict[str, Any] | None:
        """Full explanation of an entity: all properties with assertion history, conflicts, identity."""
        view = self.entity(entity_id)
        if view is None:
            return None
        properties_with_history: dict[str, dict[str, Any]] = {}
        for prop_id, prop_data in view.properties.items():
            entry = {**prop_data}
            if self._reconciler is not None:
                try:
                    history = self._reconciler.get_assertion_history(entity_id, prop_id)
                    entry["assertion_history"] = history
                except Exception:
                    entry["assertion_history"] = []
            else:
                entry["assertion_history"] = []
            properties_with_history[prop_id] = entry
        conflicts = self.conflicts(entity_id=entity_id)
        return {
            "entity_id": entity_id,
            "properties": properties_with_history,
            "relations": view.relations,
            "identity_chain": view.identity_chain,
            "last_updated": view.last_updated,
            "conflicts": [
                {
                    "property_id": c.property_id,
                    "current_value": c.current_value,
                    "proposed_value": c.proposed_value,
                    "reason": c.reason,
                    "assertions": c.assertions,
                }
                for c in conflicts
            ],
        }

    def sightings_for_entity(
        self,
        entity_id: str,
        history_per_entity: int = 3,
        since: str | None = None,
        until: str | None = None,
    ) -> dict[str, Any] | None:
        """Recent camera+timestamp sightings for one already-resolved entity.

        When since/until are given, the full history is filtered to that
        window before truncating to history_per_entity -- so a time-scoped
        query isn't silently limited to only the N most-recent-overall
        sightings when those happen to fall outside the requested window.
        """
        history = (
            self._reconciler.get_assertion_history(entity_id, "last_seen")
            if self._reconciler is not None
            else []
        )
        if since is not None or until is not None:
            history = [
                row for row in history
                if (since is None or row["valid_from"] >= since)
                and (until is None or row["valid_from"] <= until)
            ]
        sightings = [
            {
                "camera_id": (
                    row["source_id"].split(":", 1)[-1]
                    if ":" in row["source_id"]
                    else row["source_id"]
                ),
                "seen_at": row["valid_from"],
                "confidence": row["confidence"],
                "evidence_id": row["evidence_ids"][0] if row["evidence_ids"] else None,
            }
            for row in history[:history_per_entity]
        ]
        if not sightings:
            return None
        return {"entity_id": entity_id, "sightings": sightings}

    def recall_object_sightings(
        self,
        query: str,
        limit: int = 5,
        history_per_entity: int = 3,
        since: str | None = None,
        until: str | None = None,
    ) -> list[dict[str, Any]]:
        """Resolve a free-text object/person term to recent sightings.

        Two-step: resolve query against current label/semantic_tags to find
        candidate entities, then pull each candidate's last_seen assertion
        history for the camera and timestamp of each past sighting.
        """
        candidates = self._state.search_property_text(query, limit=limit)
        results: list[dict[str, Any]] = []
        for candidate in candidates:
            entity_id = candidate["entity_id"]
            sighting_record = self.sightings_for_entity(entity_id, history_per_entity, since, until)
            if sighting_record is None:
                continue
            label = self.property_value(entity_id, "label") or candidate["value"]
            results.append({
                "entity_id": entity_id,
                "label": label,
                "matched_property": candidate["property_id"],
                "sightings": sighting_record["sightings"],
            })
        return results

    def candidate_labels(self, limit: int = 60) -> list[dict[str, Any]]:
        """Unique current label values, for associative (embedding) recall.

        One entry per unique label text (the most-recently-updated entity
        wins ties), so the caller embeds each distinct word/phrase once
        rather than once per entity that happens to share a label.
        """
        rows = self._state.all_property_values("label", limit=limit)
        seen_labels: set[str] = set()
        candidates: list[dict[str, Any]] = []
        for row in rows:
            label = row["value"]
            if not isinstance(label, str) or not label.strip() or label in seen_labels:
                continue
            seen_labels.add(label)
            candidates.append({"entity_id": row["entity_id"], "label": label})
        return candidates

    def world_summary(self) -> dict[str, Any]:
        """Full world model summary for dashboard and debugging."""
        summary = self.summary()
        all_ids = self.all_entity_ids()
        brief_counts = self._state.entity_brief_counts()
        # Computed once and reused for both the per-entity has_conflicts
        # flag and the conflicts list below -- this used to call
        # self.conflicts() twice, each run independently doing the full
        # (now-fixed, but still non-trivial) conflict-resolution query
        # chain for no reason.
        #
        # conflicting_ids/conflict_count use the cheap candidate-pair scan
        # (no per-conflict assertion history) so the reported total and the
        # has_conflicts flags stay accurate; the detailed conflicts list is
        # capped to a representative sample -- same spirit as entities_brief
        # below being capped to the first 100 -- since fetching assertion
        # history for potentially thousands of live conflicts is real work
        # this dashboard summary doesn't need to pay for in full every time.
        conflict_rows = self._state.conflicts()
        conflicting_ids = {row.entity_id for row in conflict_rows}
        conflicts = self.conflicts(limit=100)
        entities_brief = []
        for eid in all_ids[:100]:
            info = brief_counts.get(eid, {})
            entities_brief.append({
                "entity_id": eid,
                "property_count": info.get("property_count", 0),
                "relation_count": info.get("relation_count", 0),
                "last_updated": info.get("last_updated", ""),
                "has_conflicts": eid in conflicting_ids,
            })
        return {
            **summary,
            "entities": entities_brief,
            "conflict_count": len(conflict_rows),
            "conflicts": [
                {
                    "entity_id": c.entity_id,
                    "property_id": c.property_id,
                    "current_value": c.current_value,
                    "proposed_value": c.proposed_value,
                    "reason": c.reason,
                }
                for c in conflicts
            ],
            "revision": self._state.revision,
        }

    def conflicts(
        self,
        entity_id: str | None = None,
        entity_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> list[ConflictInfo]:
        """entity_id/entity_ids scope the (expensive-ish) conflict scan to
        specific entities instead of the whole world model; limit bounds
        how many of the matched conflicts get their (also non-trivial)
        assertion history fetched and returned."""
        # Get conflicts from the assertion log via the state store
        conflict_rows = self._state.conflicts(entity_id=entity_id or "", entity_ids=entity_ids)
        seen: set[tuple[str, str]] = set()
        unique_rows = []
        for c in conflict_rows:
            key = (c.entity_id, c.property_id)
            if key in seen:
                continue
            seen.add(key)
            unique_rows.append(c)
        if limit is not None:
            unique_rows = unique_rows[:limit]

        # One bulk query for every conflict's assertion history instead of
        # one query per conflict -- with thousands of live conflicts this
        # was the dominant cost of building the dashboard's world summary.
        histories: dict[tuple[str, str], list[dict[str, Any]]] = {}
        if self._reconciler is not None and unique_rows:
            try:
                histories = self._reconciler.get_assertion_histories(
                    [(c.entity_id, c.property_id) for c in unique_rows]
                )
            except Exception:
                histories = {}

        result: list[ConflictInfo] = []
        for c in unique_rows:
            history = histories.get((c.entity_id, c.property_id), [])
            assertions = [a for a in history if a.get("state") in ("accepted", "conflicted")]
            proposed_value = None
            if len(assertions) >= 2:
                proposed_value = assertions[1].get("value")
            result.append(ConflictInfo(
                entity_id=c.entity_id,
                property_id=c.property_id,
                current_value=json.loads(c.value_json),
                proposed_value=proposed_value,
                reason=f"Multiple active assertions (authority={c.authority:.2f})",
                assertions=assertions,
            ))
        return result

    def graph_neighbors(self, entity_id: str, relation_type: str | None = None) -> list[dict[str, Any]]:
        return self._graph.neighbors(entity_id, relation_type)

    def graph_path(self, start: str, end: str) -> list[str] | None:
        return self._graph.find_path(start, end)

    def identity_chain(self, entity_id: str) -> list[str] | None:
        chain = self._identity.get_chain(entity_id)
        return chain.chain if chain else None

    def identity_history(self, entity_id: str | None = None) -> list[dict[str, Any]]:
        return self._identity.history(entity_id)

    def all_entity_ids(self) -> list[str]:
        return self._state.all_entity_ids()

    def entity_ranking_signals(self) -> dict[str, dict[str, object]]:
        return self._state.entity_ranking_signals()

    def all_current_relations(self) -> list[dict[str, Any]]:
        return [
            {
                "source_entity_id": r.source_entity_id,
                "relation_type_id": r.relation_type_id,
                "target_entity_id": r.target_entity_id,
                "confidence": r.confidence,
                "valid_from": r.valid_from,
                "valid_to": r.valid_to,
            }
            for r in self._state.all_current_relations()
        ]

    def summary(self) -> dict[str, Any]:
        entity_ids = self.all_entity_ids()
        relations = self.all_current_relations()
        return {
            "total_entities": len(entity_ids),
            "total_relations": len(relations),
            "entity_ids": entity_ids[:50],
            "relation_summary": {
                rt: sum(1 for r in relations if r["relation_type_id"] == rt)
                for rt in set(r["relation_type_id"] for r in relations)
            },
        }


TIME_PERIODS = {"any", "today", "yesterday", "this_week", "last_week", "this_month"}


def resolve_time_period(period: str | None) -> tuple[str, str] | None:
    """Resolve a fixed period keyword to a UTC (since, until) ISO bound.

    Deliberately not model-resolved free text: the model has no ground
    truth for "now" in its context, so it only classifies the kind of
    period asked about and this does the actual date arithmetic against a
    real clock. Returns None for "any"/unrecognized/missing -- no filter,
    matching current unscoped behavior.
    """
    if period is None or period == "any" or period not in TIME_PERIODS:
        return None
    now_local = datetime.now().astimezone()
    today_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "today":
        start, end = today_start, today_start + timedelta(days=1)
    elif period == "yesterday":
        start, end = today_start - timedelta(days=1), today_start
    elif period == "this_week":
        start = today_start - timedelta(days=today_start.weekday())
        end = start + timedelta(days=7)
    elif period == "last_week":
        this_week_start = today_start - timedelta(days=today_start.weekday())
        start, end = this_week_start - timedelta(days=7), this_week_start
    else:  # this_month
        start = today_start.replace(day=1)
        next_month = start.replace(day=28) + timedelta(days=4)
        end = next_month.replace(day=1)
    return (
        start.astimezone(timezone.utc).isoformat(),
        end.astimezone(timezone.utc).isoformat(),
    )
