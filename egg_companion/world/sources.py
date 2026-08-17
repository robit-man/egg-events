"""Source identity and property-specific authority."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SourceRecord:
    source_id: str
    source_type: str
    authority_class: str = "observation"
    model_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AuthorityPolicy:
    """Property-specific source authority evaluation.

    Authority depends on the property being assessed, not a global ranking.
    """

    # Default authority weights by (property_category, source_type, epistemic_kind)
    _DEFAULT_WEIGHTS: dict[tuple[str, str, str], float] = {
        # Person identity: user statements are strongest
        ("person.preferred_name", "user_correction", "correction"): 1.0,
        ("person.preferred_name", "user_statement", "claim"): 0.95,
        ("person.preferred_name", "llm_inference", "inference"): 0.3,
        # Object label: vision + user feedback
        ("physical_object.label", "user_correction", "correction"): 1.0,
        ("physical_object.label", "user_statement", "claim"): 0.9,
        ("physical_object.label", "detector", "observation"): 0.85,
        ("physical_object.label", "ornith_vlm", "inference"): 0.7,
        ("physical_object.label", "llm_inference", "inference"): 0.5,
        # Location: camera is strongest for current position
        ("*.current_location", "camera", "observation"): 0.9,
        ("*.current_location", "object_tracker", "observation"): 0.85,
        ("*.current_location", "user_statement", "claim"): 0.7,
        ("*.current_location", "llm_inference", "inference"): 0.4,
        # Bbox: always from detector
        ("*.bbox", "detector", "observation"): 0.95,
        # Behavior: from pose/detector
        ("*.behavior", "detector", "observation"): 0.8,
        ("*.behavior", "ornith_vlm", "inference"): 0.6,
    }

    def evaluate(
        self,
        property_type: str,
        source_type: str,
        epistemic_kind: str,
        evidence_quality: float = 1.0,
    ) -> float:
        """Return authority weight [0, 1] for this property+source combination."""
        # Try exact match
        exact = self._DEFAULT_WEIGHTS.get((property_type, source_type, epistemic_kind))
        if exact is not None:
            return exact * evidence_quality
        # Try wildcard on entity type
        parts = property_type.split(".", 1)
        if len(parts) == 2:
            wildcard = self._DEFAULT_WEIGHTS.get(("*." + parts[1], source_type, epistemic_kind))
            if wildcard is not None:
                return wildcard * evidence_quality
        # Fallback: authority_class based
        class_weights = {
            "axiom": 1.0,
            "correction": 0.95,
            "observation": 0.8,
            "claim": 0.7,
            "inference": 0.6,
            "derived": 0.5,
            "hypothesis": 0.4,
        }
        from egg_companion.world.ontology import OntologyRegistry
        # Use a default if registry not available
        return class_weights.get(epistemic_kind, 0.5) * evidence_quality
