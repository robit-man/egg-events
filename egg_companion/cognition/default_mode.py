from __future__ import annotations

from datetime import datetime, timezone

from egg_companion.config import DefaultModeConfig, WorldPruningConfig
from egg_companion.cognition.world_model import NarrativeWorldModelSynthesizer
from egg_companion.memory.store import MemoryStore


class DefaultModeNetwork:
    """Quiet-period provenance replay feeding the model-authored dream agent."""

    def __init__(
        self,
        store: MemoryStore,
        config: DefaultModeConfig,
        pruning_config: WorldPruningConfig | None = None,
    ) -> None:
        self.store = store
        self.config = config
        self.world_model = NarrativeWorldModelSynthesizer(store, config)
        self._pruning_config = pruning_config or WorldPruningConfig()
        self._dream_count = 0

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

            # World model pruning pass — runs periodically based on config
            self._dream_count += 1
            if (
                self._pruning_config.enabled
                and self._dream_count % self._pruning_config.prune_every_n_dreams == 0
            ):
                result["pruning"] = self._prune_world_model(now)

            self.store.update_job(job_id, "complete")
            return result
        except Exception as error:
            self.store.update_job(job_id, "failed", str(error))
            raise

    def _prune_world_model(self, now: datetime) -> dict[str, object]:
        """Prune stale, low-confidence, and hallucinated entities from world model."""
        from egg_companion.world.state import WorldStateStore

        state = WorldStateStore(self.store.root / "memory.sqlite3")
        pruned_total: list[str] = []

        # 1. Remove contextually impossible entities (hallucinations)
        pruned_impossible = state.prune_contextually_impossible()
        pruned_total.extend(pruned_impossible)

        # 2. Remove low-confidence det:* entities
        pruned_low_conf = state.prune_low_confidence(
            entity_prefix="det:",
            max_confidence=self._pruning_config.min_confidence,
        )
        pruned_total.extend(pruned_low_conf)

        # 3. Remove stale entities not seen recently
        stale_cutoff = (
            now.timestamp() - self._pruning_config.stale_after_hours * 3600
        )
        from datetime import datetime as dt, timezone as tz

        stale_before = dt.fromtimestamp(stale_cutoff, tz=tz.utc).isoformat()
        pruned_stale = state.prune_stale_entities(
            stale_before=stale_before,
            entity_prefix="det:",
            min_confidence=0.7,
        )
        pruned_total.extend(pruned_stale)

        # 4. Cap total det:* entities
        det_count = state.entity_count("det:")
        if det_count > self._pruning_config.max_det_entities:
            # Remove oldest/lowest-confidence until under cap
            excess = det_count - self._pruning_config.max_det_entities
            pruned_cap = state.prune_stale_entities(
                stale_before=now.isoformat(),
                entity_prefix="det:",
                min_confidence=0.9,
            )[:excess]
            pruned_total.extend(pruned_cap)

        return {
            "pruned_impossible": len(pruned_impossible),
            "pruned_low_confidence": len(pruned_low_conf),
            "pruned_stale": len(pruned_stale),
            "pruned_cap": len(pruned_total) - len(pruned_impossible) - len(pruned_low_conf) - len(pruned_stale),
            "total_pruned": len(pruned_total),
            "det_entities_remaining": state.entity_count("det:"),
        }
