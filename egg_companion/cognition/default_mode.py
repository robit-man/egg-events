from __future__ import annotations

import math
from datetime import datetime, timezone

from egg_companion.config import DefaultModeConfig
from egg_companion.cognition.world_model import WorldModelSynthesizer
from egg_companion.memory.store import MemoryStore


class DefaultModeNetwork:
    """Quiet-period graph replay, reflection, and bounded curiosity induction.

    This module never emits speech itself. It produces inspectable candidates
    from source-backed graph gaps; the realtime runtime separately verifies live
    relevance, conversational availability, cooldown, and question budget.
    """

    def __init__(self, store: MemoryStore, config: DefaultModeConfig) -> None:
        self.store = store
        self.config = config
        self.world_model = WorldModelSynthesizer(store, config)

    def run_once(self) -> dict[str, object]:
        job_id = self.store.create_job("default-mode-replay")
        self.store.update_job(job_id, "running")
        now = datetime.now(timezone.utc)
        result: dict[str, object] = {
            "job_id": job_id,
            "phase": "replay",
            "replayed_entity_ids": [],
            "reflections_created": 0,
            "reflection_ids": [],
            "curiosity_candidates": [],
        }
        try:
            inventory = self.store.cognitive_inventory(
                max(self.config.replay_limit * 5, self.config.replay_limit)
            )
            replayed: list[str] = []
            reflection_ids: list[str] = []
            curiosity: list[dict[str, object]] = []
            created_count = 0
            for item in inventory:
                evidence_count = int(item.get("evidence_count") or 0)
                if evidence_count < self.config.reflection_min_evidence:
                    continue
                entity_id = str(item["entity_id"])
                entity_type = str(item["entity_type"])
                label = str(item.get("display_name") or entity_id)
                edge_count = int(item.get("edge_count") or 0)
                claims = [
                    claim for claim in item.get("claims", []) if isinstance(claim, dict)
                ]
                predicates = {
                    str(claim.get("predicate")) for claim in claims if claim.get("predicate")
                }
                familiarity = 1.0 - math.exp(-evidence_count / 4.0)
                structural = 1.0 - math.exp(-edge_count / 4.0)
                metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                reliability = max(
                    0.0,
                    min(
                        1.0,
                        float(
                            metadata.get("label_confidence")
                            or metadata.get("confidence")
                            or 0.5
                        ),
                    ),
                )
                replayed.append(entity_id)

                if entity_type == "object" and "used_for" not in predicates:
                    summary = (
                        f"Recurring object {label} is source-backed but its user-confirmed "
                        "purpose remains unknown."
                    )
                    kind = "unresolved_object_purpose"
                    epistemic_value = min(
                        1.0,
                        0.35 * familiarity
                        + 0.25 * reliability
                        + 0.30
                        + 0.10 * structural,
                    )
                    if epistemic_value >= self.config.curiosity_threshold:
                        curiosity.append(
                            {
                                "subject_id": entity_id,
                                "subject_type": entity_type,
                                "subject_label": label[:120],
                                "predicate": "used_for",
                                "question": (
                                    f"I've seen the {label} a few times. What do you use it for?"
                                )[:300],
                                "epistemic_value": round(epistemic_value, 4),
                                "familiarity": round(familiarity, 4),
                                "answerability": round(reliability, 4),
                                "evidence_count": evidence_count,
                                "reason": "recurring visible object has a reducible semantic gap",
                            }
                        )
                else:
                    summary = (
                        f"Replay links {label} to {evidence_count} retained evidence items "
                        f"and {edge_count} graph relationships."
                    )
                    kind = "recurrent_association"

                reflection_id, created = self.store.record_default_mode_reflection(
                    entity_id,
                    kind,
                    summary,
                    max(reliability, familiarity),
                    {
                        "evidence_count": evidence_count,
                        "edge_count": edge_count,
                        "active_predicates": sorted(predicates),
                    },
                    now,
                )
                reflection_ids.append(reflection_id)
                created_count += int(created)
                if len(replayed) >= self.config.replay_limit:
                    break

            curiosity.sort(
                key=lambda item: (
                    float(item["epistemic_value"]), int(item["evidence_count"])
                ),
                reverse=True,
            )
            result.update(
                {
                    "phase": "complete",
                    "replayed_entity_ids": replayed,
                    "reflections_created": created_count,
                    "reflection_ids": reflection_ids,
                    "curiosity_candidates": curiosity[:3],
                    "inventory_size": len(inventory),
                }
            )
            result["meta_graph"] = self.world_model.update(
                replayed, reflection_ids, now
            )
            self.store.update_job(job_id, "complete")
            return result
        except Exception as error:
            self.store.update_job(job_id, "failed", str(error))
            raise
