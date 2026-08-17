"""Query API: unified world-model access."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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

    def conflicts(self) -> list[ConflictInfo]:
        # Get conflicts from the assertion log via the state store
        conflict_rows = self._state.conflicts()
        seen: set[tuple[str, str]] = set()
        result: list[ConflictInfo] = []
        for c in conflict_rows:
            key = (c.entity_id, c.property_id)
            if key in seen:
                continue
            seen.add(key)
            # Fetch assertion history to show both sides of the conflict
            assertions = []
            if self._reconciler is not None:
                try:
                    history = self._reconciler.get_assertion_history(c.entity_id, c.property_id)
                    assertions = [
                        a for a in history if a.get("state") in ("accepted", "conflicted")
                    ]
                except Exception:
                    pass
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
