"""Cognitive context: world model as context for the cognition loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from egg_companion.world.query import WorldQuery
from egg_companion.world.spatial import SpatialState


@dataclass
class ContextWindow:
    entities: list[dict[str, Any]] = field(default_factory=list)
    recent_events: list[dict[str, Any]] = field(default_factory=list)
    active_relations: list[dict[str, Any]] = field(default_factory=list)
    pending_actions: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    max_characters: int = 5000
    total_characters: int = 0


@dataclass
class EntityContext:
    entity_id: str
    label: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    relations: list[dict[str, Any]] = field(default_factory=list)
    spatial: SpatialState | None = None
    last_seen: str = ""
    confidence: float = 0.0


class CognitiveContext:
    """Builds a context window for the LLM from the world model."""

    def __init__(self, query: WorldQuery) -> None:
        self._query = query

    def build_window(
        self,
        focus_entity: str | None = None,
        max_characters: int = 5000,
        max_entities: int = 10,
        include_events: bool = True,
        include_actions: bool = True,
    ) -> ContextWindow:
        window = ContextWindow(max_characters=max_characters)
        entity_ids = self._query.all_entity_ids()
        entities = []
        total_chars = 0
        for eid in entity_ids[:max_entities]:
            ev = self._query.entity(eid)
            if ev is None:
                continue
            label = ev.properties.get("label", {}).get("value", eid) if ev.properties else eid
            if isinstance(label, dict):
                label = str(label)
            entry = {"entity_id": eid, "label": label, "properties": {}, "relations": []}
            for pid, pdata in ev.properties.items():
                entry["properties"][pid] = {
                    "value": pdata["value"],
                    "confidence": pdata["confidence"],
                    "authority": pdata["authority"],
                }
            for rel in ev.relations[:5]:
                entry["relations"].append({
                    "type": rel["relation_type_id"],
                    "target": rel["target_entity_id"],
                    "confidence": rel["confidence"],
                })
            char_count = len(json.dumps(entry, default=str))
            if total_chars + char_count > max_characters:
                break
            total_chars += char_count
            entities.append(entry)
        window.entities = entities
        window.total_characters = total_chars
        window.summary = self._query.summary()
        return window

    def for_entity(self, entity_id: str) -> EntityContext | None:
        ev = self._query.entity(entity_id)
        if ev is None:
            return None
        label = ev.properties.get("label", {}).get("value", entity_id) if ev.properties else entity_id
        if isinstance(label, dict):
            label = str(label)
        return EntityContext(
            entity_id=entity_id,
            label=label,
            properties=ev.properties,
            relations=ev.relations,
            last_seen=ev.last_updated,
            confidence=max((p["confidence"] for p in ev.properties.values()), default=0.0),
        )

    def recent_activity(self, seconds: float = 300.0) -> list[dict[str, Any]]:
        return []

    def serialize_for_llm(self, window: ContextWindow) -> str:
        parts = []
        parts.append(f"World Model Summary: {window.summary.get('total_entities', 0)} entities, {window.summary.get('total_relations', 0)} relations")
        if window.entities:
            parts.append("Entities:")
            for e in window.entities:
                props_str = ", ".join(f"{k}={v['value']}" for k, v in e.get("properties", {}).items())
                parts.append(f"  {e['entity_id']}: {e.get('label', '?')} ({props_str})")
                for rel in e.get("relations", [])[:3]:
                    parts.append(f"    -{rel['type']}->{rel['target']} (conf={rel['confidence']:.2f})")
        return "\n".join(parts)
