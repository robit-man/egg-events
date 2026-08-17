"""Bridge between the operational world model and the existing memory graph.

This module enriches the knowledge graph snapshot with:
- Ontology type annotations on entities
- Ontology-aware relation types on edges
- World model state (current properties, relations, freshness)
- Observability states
- Uncertainty decomposition
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Any

from egg_companion.world.ontology import OntologyRegistry
from egg_companion.world.state import WorldStateStore
from egg_companion.world.relations import WorldGraphStore
from egg_companion.world.types import EpistemicKind, ObservabilityState, TypedValue, ValueType


@dataclass
class EnrichedEntity:
    """Entity with ontology annotations and world state."""
    entity_id: str
    entity_type: str
    ontology_type: str | None = None
    display_name: str | None = None
    world_state: dict[str, Any] = field(default_factory=dict)
    observability: str = "unknown"
    freshness: dict[str, float] = field(default_factory=dict)
    uncertainty: dict[str, float] = field(default_factory=dict)
    relations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EnrichedEdge:
    """Edge with ontology-aware relation type."""
    edge_id: str
    source_id: str
    target_id: str
    relation: str
    ontology_relation: str | None = None
    namespace: str = "world"
    confidence: float = 0.0
    persistence: str = "observation_dependent"
    stale_after: float | None = None
    valid_from: str | None = None
    valid_to: str | None = None


class KnowledgeGraphBridge:
    """Bridges the world model ontology to the existing memory graph."""

    def __init__(
        self,
        memory_conn: sqlite3.Connection,
        world_state: WorldStateStore,
        world_graph: WorldGraphStore,
        ontology: OntologyRegistry,
    ) -> None:
        self._memory_conn = memory_conn
        self._world_state = world_state
        self._world_graph = world_graph
        self._ontology = ontology
        self._lock = threading.RLock()

    def enrich_entity(self, entity_id: str, entity_type: str) -> EnrichedEntity:
        """Enrich an entity with ontology annotations and world state."""
        ontology_type = self._ontology.get_object_type(entity_type)
        
        world_state = {}
        relations = []
        
        try:
            props = self._world_state.get_entity_state(entity_id)
            for prop_id, prop_row in props.items():
                world_state[prop_id] = {
                    "value": json.loads(prop_row.value_json),
                    "confidence": prop_row.confidence,
                    "authority": prop_row.authority,
                    "epistemic_kind": prop_row.epistemic_kind,
                    "valid_from": prop_row.valid_from,
                    "valid_to": prop_row.valid_to,
                }
        except Exception:
            pass

        try:
            world_relations = self._world_state.get_entity_relations(entity_id)
            for rel in world_relations:
                relations.append({
                    "source": rel.source_entity_id,
                    "relation": rel.relation_type_id,
                    "target": rel.target_entity_id,
                    "confidence": rel.confidence,
                    "valid_from": rel.valid_from,
                    "valid_to": rel.valid_to,
                })
        except Exception:
            pass

        observability = self._compute_observability(entity_id, world_state)
        freshness = self._compute_freshness(world_state, ontology_type)
        uncertainty = self._compute_uncertainty(world_state)

        return EnrichedEntity(
            entity_id=entity_id,
            entity_type=entity_type,
            ontology_type=ontology_type.id if ontology_type else None,
            display_name=world_state.get("label", {}).get("value") if world_state else None,
            world_state=world_state,
            observability=observability,
            freshness=freshness,
            uncertainty=uncertainty,
            relations=relations,
        )

    def enrich_edge(self, edge_id: str, source_id: str, target_id: str, relation: str) -> EnrichedEdge:
        """Enrich an edge with ontology-aware relation type."""
        ontology_relation = self._ontology.get_relation_type(relation)
        
        namespace = "world"
        if relation in {"same_person_as", "same_object_as"}:
            namespace = "identity"
        elif relation in {"co_observed_with", "heard_with", "recalled_with"}:
            namespace = "association"
        elif relation in {"expresses_theme", "informs_world_model"}:
            namespace = "narrative"
        elif relation in {"caused", "enabled", "prevented"}:
            namespace = "causal"

        return EnrichedEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            ontology_relation=relation if ontology_relation else None,
            namespace=namespace,
            persistence=ontology_relation.persistence if ontology_relation else "observation_dependent",
            stale_after=ontology_relation.stale_after if ontology_relation else None,
        )

    def _compute_observability(self, entity_id: str, world_state: dict[str, Any]) -> str:
        """Compute observability state from world state.
        
        Uses explicit observability property first, then falls back to
        last_seen age heuristic.
        """
        obs_prop = world_state.get("observability")
        if obs_prop and isinstance(obs_prop.get("value"), str):
            return obs_prop["value"]

        last_seen = world_state.get("last_seen")
        if not last_seen:
            return ObservabilityState.UNKNOWN.value
        
        import datetime
        try:
            last_seen_dt = datetime.datetime.fromisoformat(last_seen.get("value", ""))
            age_seconds = (datetime.datetime.now(datetime.timezone.utc) - last_seen_dt).total_seconds()
            
            if age_seconds < 30:
                return ObservabilityState.OBSERVED_PRESENT.value
            elif age_seconds < 300:
                return ObservabilityState.NOT_OBSERVED.value
            else:
                return ObservabilityState.UNKNOWN.value
        except Exception:
            return ObservabilityState.UNKNOWN.value

    def _compute_freshness(self, world_state: dict[str, Any], ontology_type: Any) -> dict[str, float]:
        """Compute freshness for each property based on ontology stale_after metadata."""
        freshness = {}
        
        prop_types = {}
        for pt in self._ontology.list_property_types():
            if pt.stale_after is not None:
                prop_types[pt.id] = pt
        
        for prop_id, prop_data in world_state.items():
            pt = prop_types.get(prop_id)
            if pt is None or pt.stale_after is None:
                freshness[prop_id] = 1.0
                continue
            max_age = pt.stale_after
            try:
                import datetime
                valid_from = datetime.datetime.fromisoformat(prop_data.get("valid_from", ""))
                age_seconds = (datetime.datetime.now(datetime.timezone.utc) - valid_from).total_seconds()
                if pt.decay_model == "exponential":
                    freshness[prop_id] = max(0.0, 2.0 ** (-age_seconds / max_age))
                else:
                    freshness[prop_id] = max(0.0, 1.0 - (age_seconds / max_age))
            except Exception:
                freshness[prop_id] = 0.5
        
        return freshness

    def _compute_uncertainty(self, world_state: dict[str, Any]) -> dict[str, float]:
        """Compute uncertainty decomposition for world state."""
        uncertainty = {}
        
        for prop_id, prop_data in world_state.items():
            confidence = prop_data.get("confidence", 0.0)
            authority = prop_data.get("authority", 0.0)
            
            uncertainty[prop_id] = {
                "measurement": 1.0 - confidence,
                "identity": 1.0 - authority,
                "classification": 0.1,
                "spatial": 0.2,
                "temporal": 0.1,
                "source_disagreement": 0.0,
                "staleness": 0.0,
            }
        
        return uncertainty

    def enrich_graph_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Enrich an existing knowledge graph snapshot with world model data."""
        enriched_nodes = []
        
        for node in snapshot.get("nodes", []):
            if node.get("kind") == "entity":
                entity_id = node.get("source_id", "")
                entity_type = node.get("subtype", "")
                
                try:
                    enriched = self.enrich_entity(entity_id, entity_type)
                    node["ontology_type"] = enriched.ontology_type
                    node["observability"] = enriched.observability
                    node["freshness"] = enriched.freshness
                    node["uncertainty"] = enriched.uncertainty
                    node["world_relations"] = enriched.relations[:10]
                    
                    if enriched.display_name and not node.get("label"):
                        node["label"] = enriched.display_name
                except Exception:
                    pass
            
            enriched_nodes.append(node)
        
        enriched_links = []
        
        for link in snapshot.get("links", []):
            if link.get("source", "").startswith("entity:") and link.get("target", "").startswith("entity:"):
                try:
                    enriched = self.enrich_edge(
                        link.get("id", ""),
                        link.get("source", "").removeprefix("entity:"),
                        link.get("target", "").removeprefix("entity:"),
                        link.get("relation", ""),
                    )
                    link["namespace"] = enriched.namespace
                    link["ontology_relation"] = enriched.ontology_relation
                    link["persistence"] = enriched.persistence
                except Exception:
                    pass
            
            enriched_links.append(link)
        
        snapshot["nodes"] = enriched_nodes
        snapshot["links"] = enriched_links
        
        snapshot["ontology"] = {
            "object_types": len(self._ontology.list_object_types()),
            "relation_types": len(self._ontology.list_relation_types()),
            "property_types": len(self._ontology.list_property_types()),
        }
        
        snapshot["world_model"] = {
            "total_entities": len(self._world_state.all_entity_ids()),
            "total_relations": len(self._world_state.all_current_relations()),
        }
        
        return snapshot
