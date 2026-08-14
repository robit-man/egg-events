from __future__ import annotations

from datetime import datetime, timezone

from egg_companion.config import DefaultModeConfig
from egg_companion.cognition.world_model import WorldModelSynthesizer
from egg_companion.memory.store import MemoryStore


class DefaultModeNetwork:
    """Quiet-period provenance replay feeding the model-authored dream agent."""

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
            replayed = [
                str(item["entity_id"])
                for item in inventory[: self.config.replay_limit]
            ]
            result.update(
                {
                    "phase": "complete",
                    "replayed_entity_ids": replayed,
                    "reflections_created": 0,
                    "reflection_ids": [],
                    "curiosity_candidates": [],
                    "inventory_size": len(inventory),
                }
            )
            result["meta_graph"] = self.world_model.update(replayed, [], now)
            self.store.update_job(job_id, "complete")
            return result
        except Exception as error:
            self.store.update_job(job_id, "failed", str(error))
            raise
