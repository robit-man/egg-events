from __future__ import annotations

from egg_companion.config import PrivacyConfig
from egg_companion.memory.retention import RetentionPlanner
from egg_companion.memory.store import MemoryStore


class MemoryConsolidator:
    """Bounded default-mode pass over durable, real evidence."""

    def __init__(self, store: MemoryStore, privacy: PrivacyConfig) -> None:
        self.store = store
        self.privacy = privacy

    def run_once(self) -> dict[str, object]:
        job_id = self.store.create_job("memory-consolidation")
        self.store.update_job(job_id, "running")
        result: dict[str, object] = {
            "job_id": job_id,
            "summarized_episodes": 0,
            "claim_conflicts": [],
            "expired_media": 0,
            "expired_evidence": 0,
        }
        try:
            for episode in self.store.episodes_without_summaries(self.store.config.consolidation_batch_size):
                detail = self.store.episode_detail(str(episode["episode_id"]))
                if detail is None:
                    continue
                self.store.set_episode_summary(str(episode["episode_id"]), self._summary(detail))
                result["summarized_episodes"] = int(result["summarized_episodes"]) + 1
            result["claim_conflicts"] = self.store.conflicting_claims(
                self.store.config.consolidation_batch_size
            )
            result.update(RetentionPlanner(self.store, self.privacy).execute())
            result["stats"] = self.store.memory_stats()
            self.store.update_job(job_id, "complete")
            return result
        except Exception as error:
            self.store.update_job(job_id, "failed", str(error))
            raise

    @staticmethod
    def _summary(detail: dict[str, object]) -> str:
        evidence = detail["evidence"]
        transcripts: list[str] = []
        labels: list[str] = []
        actions: list[str] = []
        for item in evidence:
            payload = item.get("payload", {})
            transcript = payload.get("transcript") if isinstance(payload, dict) else None
            if isinstance(transcript, str) and transcript.strip():
                transcripts.append(" ".join(transcript.split())[:180])
            detections = payload.get("detections", ()) if isinstance(payload, dict) else ()
            if isinstance(detections, list):
                labels.extend(
                    str(detection["label"]) for detection in detections
                    if isinstance(detection, dict) and detection.get("label")
                )
            if isinstance(payload, dict) and "spoken" in payload:
                actions.append("spoken" if payload["spoken"] else "speech suppressed")
        parts: list[str] = []
        if transcripts:
            parts.append("Speech: " + " | ".join(dict.fromkeys(transcripts[:3])))
        if labels:
            parts.append("Observed: " + ", ".join(dict.fromkeys(labels[:12])))
        if actions:
            parts.append("Actions: " + ", ".join(dict.fromkeys(actions)))
        if not parts:
            entity_ids = [str(entity["entity_id"]) for entity in detail["entities"]]
            parts.append("Episode involving " + (", ".join(entity_ids) if entity_ids else "unlinked sensory evidence"))
        return "; ".join(parts)[:600]
