from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
import numpy as np

from egg_companion.config import CameraConfig, EggConfig
from egg_companion.models import Observation
from egg_companion.services.scene import SceneInventory


@dataclass
class CameraTelemetry:
    camera_id: str
    source: str
    configured_rotation: int | str
    resolved_rotation: int | None = None
    frame_jpeg: bytes | None = None
    frame_sequence: int = 0
    frame_shape: tuple[int, int, int] | None = None
    fps: float | None = None
    detections: list[dict[str, object]] = field(default_factory=list)
    detection_sequence: int = 0
    inference_fps: float | None = None
    last_detection_monotonic: float | None = None
    semantic_labels: list[str] = field(default_factory=list)
    updated_at: str | None = None
    detections_updated_at: str | None = None


class RuntimeTelemetry:
    """Thread-safe, non-identifying state exposed by the local dashboard."""

    def __init__(self, config: EggConfig) -> None:
        self._lock = threading.RLock()
        self._cameras = {
            camera.id: CameraTelemetry(
                camera.id,
                camera.source,
                camera.rotation_degrees,
                camera.rotation_degrees if isinstance(camera.rotation_degrees, int) else None,
            )
            for camera in config.cameras
            if camera.enabled
        }
        self._waveform: list[float] = []
        self._waveform_sample_count = config.audio.waveform_samples
        self._waveform_sequence = 0
        self._waveform_updated_at: str | None = None
        self._audio_rms: float | None = None
        self._vad_speech = False
        self._vad_speech_ratio = 0.0
        self._vad_speech_ms = 0
        self._latest_transcript: str | None = None
        self._latest_transcript_at: str | None = None
        self._transcript_count = 0
        self._transcript_history: list[dict[str, object]] = []
        self._asr = {
            "accepted": 0,
            "rejected": 0,
            "errors": 0,
            "last_rejection": None,
            "last_rejected_at": None,
            "last_metadata": {},
        }
        self._audio_comprehension: dict[str, object] = {
            "state": "idle",
            "queued": 0,
            "completed": 0,
            "errors": 0,
            "coalesced": 0,
            "rate_limited": 0,
            "context_id": None,
            "classifications": [],
            "last_detail": None,
            "duration_ms": None,
            "updated_at": None,
        }
        self._latest_reply: str | None = None
        self._voice_runtime: dict[str, object] = {
            "floor": "listening",
            "revision": 0,
            "active_playback_id": None,
            "playback_status": None,
            "active_barge_id": None,
            "history_turns": 0,
            "last_transition_reason": "runtime_initialized",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._runtime_errors: list[dict[str, str]] = []
        self._object_learning = {
            "stable_candidates": 0,
            "duplicate_candidates": 0,
            "clip_queries": 0,
            "clip_recalls": 0,
            "vlm_requests": 0,
            "vlm_successes": 0,
            "vlm_rejections": 0,
            "vlm_errors": 0,
            "ocr_requests": 0,
            "ocr_hits": 0,
            "speech_deferrals": 0,
            "review_queued": 0,
            "audit_consistent": 0,
            "audit_flagged": 0,
            "review_queue_depth": 0,
            "last_stage": "idle",
            "last_detail": None,
            "updated_at": None,
        }
        self._ocr: dict[str, object] = {
            "queued": 0,
            "requests": 0,
            "hits": 0,
            "empty": 0,
            "errors": 0,
            "last_stage": "idle",
            "last_detail": None,
            "updated_at": None,
            "recent": [],
        }
        self._memory = {"accepted_events": 0, "closed_episodes": 0, "last_accepted": False, "last_closed": 0}
        self._attention_decisions: list[dict[str, object]] = []
        self._interaction_decisions: list[dict[str, object]] = []
        self._tool_calls: list[dict[str, object]] = []
        self._graph_activation_sequence = 0
        self._graph_activations: list[dict[str, object]] = []
        self._identity_dialogue: dict[str, object] = {"state": "idle"}
        self._identity_continuity: dict[str, object] = {
            "state": "idle",
            "queued": 0,
            "completed": 0,
            "coalesced": 0,
            "disagreements": 0,
            "errors": 0,
            "last": None,
            "recent": [],
        }
        self._consolidation: dict[str, object] = {}
        self._retrieval_hits: list[dict[str, object]] = []
        self._microphone_direction: float | None = None
        self._respeaker: dict[str, object] = {
            "device": "XVF3000 ReSpeaker USB 4-Mic Array v2.0",
            "ready": False,
        }
        self._scene = SceneInventory()
        self._brain: dict[str, object] = {}
        self._default_mode: dict[str, object] = {"state": "idle"}
        self._narrative_semantics: dict[str, object] = {"state": "idle"}
        self._gpu: dict[str, object] = {}
        self._activity: dict[str, object] = {
            "scale": 1.0,
            "state": "active",
            "last_source": None,
            "modalities": [],
        }
        self._environmental_cognition: dict[str, object] = {
            "state": "idle",
            "queued": 0,
            "coalesced": 0,
            "grounded": 0,
            "reflected": 0,
            "silent": 0,
            "spoken": 0,
            "suppressed": 0,
            "preempted": 0,
            "faded": 0,
            "stale": 0,
            "errors": 0,
            "pixel_wakeups": 0,
            "last": None,
            "recent": [],
        }

    def set_rotation(self, camera_id: str, angle: int) -> None:
        with self._lock:
            self._cameras[camera_id].resolved_rotation = angle

    def record_audio(self, rms: float, speech: bool, speech_ratio: float, speech_ms: int) -> None:
        self.record_audio_state(rms, speech, speech_ratio, speech_ms)

    def record_audio_state(self, rms: float, speech: bool, speech_ratio: float, speech_ms: int) -> None:
        with self._lock:
            self._audio_rms = round(rms, 5)
            self._vad_speech = speech
            self._vad_speech_ratio = round(speech_ratio, 3)
            self._vad_speech_ms = speech_ms

    def record_audio_comprehension(
        self,
        stage: str,
        *,
        context_id: str | None = None,
        classifications: list[dict[str, object]] | None = None,
        detail: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        with self._lock:
            counter = "errors" if stage == "error" else stage
            if counter in {
                "queued", "completed", "errors", "coalesced", "rate_limited"
            }:
                self._audio_comprehension[counter] = int(
                    self._audio_comprehension.get(counter) or 0
                ) + 1
            self._audio_comprehension["state"] = stage
            self._audio_comprehension["context_id"] = context_id
            if classifications is not None:
                self._audio_comprehension["classifications"] = [
                    dict(item) for item in classifications[:10]
                ]
            if detail is not None:
                self._audio_comprehension["last_detail"] = detail[:500]
            if duration_ms is not None:
                self._audio_comprehension["duration_ms"] = round(duration_ms, 1)
            self._audio_comprehension["updated_at"] = datetime.now(
                timezone.utc
            ).isoformat()

    def record_respeaker(self, status: dict[str, object]) -> None:
        with self._lock:
            self._respeaker = dict(status)

    def record_waveform(self, samples: np.ndarray) -> None:
        if samples.size:
            indices = np.linspace(0, samples.size - 1, num=min(self._waveform_sample_count, samples.size), dtype=int)
            waveform = samples[indices].round(4).tolist()
            rms = float(np.sqrt(np.mean(np.square(samples))))
        else:
            waveform = []
            rms = 0.0
        with self._lock:
            self._waveform = waveform
            self._audio_rms = round(rms, 5)
            self._waveform_sequence += 1
            self._waveform_updated_at = datetime.now(timezone.utc).isoformat()

    def waveform_snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "sequence": self._waveform_sequence,
                "samples": list(self._waveform),
                "rms": self._audio_rms,
                "updated_at": self._waveform_updated_at,
            }

    def record_transcript(self, transcript: str, metadata: dict[str, object] | None = None) -> None:
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            self._latest_transcript = transcript
            self._latest_transcript_at = now
            self._transcript_count += 1
            self._asr["accepted"] = int(self._asr["accepted"]) + 1
            self._asr["last_metadata"] = dict(metadata or {})
            self._transcript_history.append(
                {"text": transcript[:300], "at": now, "metadata": dict(metadata or {})}
            )
            self._transcript_history = self._transcript_history[-8:]

    def record_asr_rejection(self, reason: str, metadata: dict[str, object] | None = None) -> None:
        with self._lock:
            self._asr["rejected"] = int(self._asr["rejected"]) + 1
            self._asr["last_rejection"] = reason[:160]
            self._asr["last_rejected_at"] = datetime.now(timezone.utc).isoformat()
            self._asr["last_metadata"] = dict(metadata or {})

    def record_asr_error(self, error: BaseException) -> None:
        with self._lock:
            self._asr["errors"] = int(self._asr["errors"]) + 1
        self.record_runtime_error("asr", error)

    def record_object_learning(self, stage: str, detail: object | None = None) -> None:
        counter_by_stage = {
            "stable_candidate": "stable_candidates",
            "duplicate_candidate": "duplicate_candidates",
            "clip_query": "clip_queries",
            "clip_recall": "clip_recalls",
            "vlm_request": "vlm_requests",
            "vlm_success": "vlm_successes",
            "vlm_rejection": "vlm_rejections",
            "vlm_error": "vlm_errors",
            "ocr_request": "ocr_requests",
            "ocr_hit": "ocr_hits",
            "speech_deferral": "speech_deferrals",
            "review_queued": "review_queued",
            "audit_consistent": "audit_consistent",
            "audit_flagged": "audit_flagged",
        }
        with self._lock:
            counter = counter_by_stage.get(stage)
            if counter:
                self._object_learning[counter] = int(self._object_learning[counter]) + 1
            self._object_learning["last_stage"] = stage
            self._object_learning["last_detail"] = None if detail is None else str(detail)[:160]
            self._object_learning["updated_at"] = datetime.now(timezone.utc).isoformat()

    def set_review_queue_depth(self, depth: int) -> None:
        with self._lock:
            self._object_learning["review_queue_depth"] = int(depth)

    def record_ocr(
        self,
        stage: str,
        detail: object | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        counter_by_stage = {
            "queued": "queued",
            "request": "requests",
            "hit": "hits",
            "empty": "empty",
            "error": "errors",
        }
        with self._lock:
            counter = counter_by_stage.get(stage)
            if counter:
                self._ocr[counter] = int(self._ocr[counter]) + 1
            now = datetime.now(timezone.utc).isoformat()
            self._ocr["last_stage"] = stage
            self._ocr["last_detail"] = None if detail is None else str(detail)[:300]
            self._ocr["updated_at"] = now
            if stage == "hit":
                recent = list(self._ocr["recent"])
                recent.append({"at": now, "text": str(detail or "")[:500], **dict(metadata or {})})
                self._ocr["recent"] = recent[-12:]

    def record_reply(self, reply: str) -> None:
        with self._lock:
            self._latest_reply = reply

    def record_voice_transition(
        self, snapshot: dict[str, object], reason: str
    ) -> None:
        with self._lock:
            self._voice_runtime = {
                **dict(snapshot),
                "last_transition_reason": reason[:160],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

    def record_default_mode(self, state: dict[str, object]) -> None:
        with self._lock:
            self._default_mode = {
                **dict(state),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

    def record_narrative_semantics(self, state: dict[str, object]) -> None:
        with self._lock:
            self._narrative_semantics = {
                **dict(state),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

    def record_environmental_cognition(
        self,
        stage: str,
        *,
        stimulus_id: str | None = None,
        camera_id: str | None = None,
        salience: float | None = None,
        detail: str | None = None,
        assessment: dict[str, object] | None = None,
        deliberation: dict[str, object] | None = None,
        duration_ms: float | None = None,
    ) -> None:
        counter = {
            "queued": "queued",
            "coalesced": "coalesced",
            "grounded": "grounded",
            "reflect": "reflected",
            "reflection_queued": "reflected",
            "silent": "silent",
            "spoken": "spoken",
            "suppressed": "suppressed",
            "preempted": "preempted",
            "faded": "faded",
            "stale": "stale",
            "error": "errors",
            "pixel_novelty": "pixel_wakeups",
        }.get(stage)
        now = datetime.now(timezone.utc).isoformat()
        entry: dict[str, object] = {
            "state": stage,
            "stimulus_id": stimulus_id,
            "camera_id": camera_id,
            "updated_at": now,
        }
        if salience is not None:
            entry["salience"] = round(max(0.0, min(1.0, salience)), 4)
        if detail is not None:
            entry["detail"] = str(detail)[:1200]
        if assessment is not None:
            entry["assessment"] = dict(assessment)
        if deliberation is not None:
            entry["deliberation"] = dict(deliberation)
        if duration_ms is not None:
            entry["duration_ms"] = round(duration_ms, 1)
        with self._lock:
            if counter:
                self._environmental_cognition[counter] = int(
                    self._environmental_cognition.get(counter) or 0
                ) + 1
            self._environmental_cognition["state"] = stage
            self._environmental_cognition["last"] = entry
            recent = list(self._environmental_cognition["recent"])
            recent.append(entry)
            self._environmental_cognition["recent"] = recent[-24:]

    def record_runtime_error(self, component: str, detail: str | BaseException) -> None:
        if isinstance(detail, BaseException):
            message = str(detail).strip()
            formatted = f"{type(detail).__name__}: {message}" if message else type(detail).__name__
        else:
            formatted = detail.strip() or "unspecified error"
        with self._lock:
            self._runtime_errors.append(
                {
                    "component": component,
                    "detail": formatted[:300],
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            )
            self._runtime_errors = self._runtime_errors[-8:]

    def record_memory(
        self,
        accepted: bool,
        closed: int,
        accepted_events: int,
        closed_episodes: int,
        lifecycle: dict[str, object] | None = None,
    ) -> None:
        with self._lock:
            self._memory = {
                "accepted_events": accepted_events,
                "closed_episodes": closed_episodes,
                "last_accepted": accepted,
                "last_closed": closed,
                "lifecycle": dict(lifecycle or {}),
            }

    def record_retrieval(self, hits: list[dict[str, object]]) -> None:
        with self._lock:
            self._retrieval_hits = [dict(hit) for hit in hits]

    def record_attention(self, target_id: str, label: str, decision) -> None:
        with self._lock:
            self._attention_decisions.append(
                {
                    "target_id": target_id,
                    "label": label,
                    "capture_priority": decision.capture_priority,
                    "allow_outward_speech": decision.allow_outward_speech,
                    "components": dict(decision.components),
                    "reason": decision.reason,
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            )
            self._attention_decisions = self._attention_decisions[-20:]

    def record_interaction(self, allowed: bool, reason: str, transcript: str, response: str) -> None:
        with self._lock:
            self._interaction_decisions.append(
                {
                    "allowed": allowed,
                    "reason": reason,
                    "transcript": transcript[:160],
                    "response": response[:160],
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            )
            self._interaction_decisions = self._interaction_decisions[-20:]

    def record_tool_call(
        self,
        name: str,
        query: str,
        success: bool | None,
        detail: str,
        duration_ms: float,
        *,
        context_id: str | None = None,
    ) -> None:
        with self._lock:
            entry = {
                "name": name,
                "query": query[:300],
                "success": success,
                "status": (
                    "running" if success is None else "completed" if success else "failed"
                ),
                "detail": detail[:500],
                "duration_ms": round(duration_ms, 1),
                "context_id": context_id,
                "at": datetime.now(timezone.utc).isoformat(),
            }
            pending_index = next(
                (
                    index
                    for index in range(len(self._tool_calls) - 1, -1, -1)
                    if self._tool_calls[index].get("name") == name
                    and self._tool_calls[index].get("context_id") == context_id
                    and self._tool_calls[index].get("status") == "running"
                ),
                None,
            )
            if success is not None and pending_index is not None:
                self._tool_calls[pending_index] = entry
            else:
                self._tool_calls.append(entry)
            self._tool_calls = self._tool_calls[-20:]

    def record_graph_activation(
        self,
        source: str,
        node_ids: list[str] | tuple[str, ...],
        *,
        origin_node_ids: list[str] | tuple[str, ...] = (),
        intensity: float = 1.0,
        cascade: bool = True,
        detail: str | None = None,
    ) -> None:
        """Publish a bounded causal firing for the live knowledge graph.

        Node IDs use the same ``kind:source_id`` representation returned by the
        durable graph API, so this stream cannot manufacture graph-only facts.
        """

        allowed_kinds = {"entity", "evidence", "episode", "claim"}

        def normalized(values: list[str] | tuple[str, ...]) -> list[str]:
            result: list[str] = []
            for value in values:
                node_id = str(value).strip()
                kind, separator, source_id = node_id.partition(":")
                if not separator or kind not in allowed_kinds or not source_id:
                    continue
                if node_id not in result:
                    result.append(node_id)
            return result[:64]

        nodes = normalized(node_ids)
        origins = normalized(origin_node_ids)
        if not nodes and not origins:
            return
        with self._lock:
            self._graph_activation_sequence += 1
            self._graph_activations.append(
                {
                    "sequence": self._graph_activation_sequence,
                    "source": str(source or "cognition")[:48],
                    "node_ids": nodes,
                    "origin_node_ids": origins,
                    "intensity": round(max(0.1, min(1.0, float(intensity))), 3),
                    "cascade": bool(cascade),
                    "detail": str(detail)[:160] if detail else None,
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            )
            self._graph_activations = self._graph_activations[-64:]

    def graph_activation_snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "sequence": self._graph_activation_sequence,
                "events": [dict(event) for event in self._graph_activations],
            }

    def record_identity_dialogue(
        self,
        state: str,
        profile_id: str | None = None,
        camera_id: str | None = None,
        name: str | None = None,
    ) -> None:
        with self._lock:
            self._identity_dialogue = {
                "state": state,
                "profile_id": profile_id,
                "camera_id": camera_id,
                "name": name,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

    def record_identity_continuity(
        self,
        stage: str,
        *,
        candidate_id: str,
        entity_id: str,
        camera_id: str,
        geometry: dict[str, object] | None = None,
        analysis: dict[str, object] | None = None,
        detail: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        with self._lock:
            counter = "errors" if stage == "error" else stage
            if counter in {"queued", "completed", "coalesced", "errors"}:
                self._identity_continuity[counter] = int(
                    self._identity_continuity.get(counter) or 0
                ) + 1
            if (
                stage == "completed"
                and isinstance(analysis, dict)
                and analysis.get("same_person") is False
            ):
                self._identity_continuity["disagreements"] = int(
                    self._identity_continuity.get("disagreements") or 0
                ) + 1
            record = {
                "state": stage,
                "candidate_id": candidate_id,
                "entity_id": entity_id,
                "camera_id": camera_id,
                "geometry": dict(geometry or {}),
                "analysis": dict(analysis or {}),
                "detail": str(detail)[:500] if detail else None,
                "duration_ms": round(duration_ms, 1) if duration_ms is not None else None,
                "at": datetime.now(timezone.utc).isoformat(),
            }
            self._identity_continuity["state"] = stage
            self._identity_continuity["last"] = record
            if stage in {"completed", "error"}:
                recent = list(self._identity_continuity["recent"])
                recent.append(record)
                self._identity_continuity["recent"] = recent[-20:]

    def record_brain_tick(self, tick) -> None:
        """Sensing/cognition regions of the composed CognitiveArchitecture perceive
        pass (egg_companion/cognition/architecture.py). The memory region is not
        duplicated here; snapshot() folds in the existing self._memory state."""
        top_target = tick.targets[0] if tick.targets else None
        top_decision = tick.decisions[0][1] if tick.decisions else None
        with self._lock:
            self._brain = {
                "sensing": {
                    "target_count": len(tick.targets),
                    "top_label": top_target.detection.label if top_target else None,
                    "top_priority": round(top_target.priority, 4) if top_target else None,
                    "top_reason": top_target.reason if top_target else None,
                    "novelty": round(tick.novelty, 4),
                },
                "cognition": {
                    "capture_priority": top_decision.capture_priority if top_decision else None,
                    "allow_outward_speech": top_decision.allow_outward_speech if top_decision else None,
                    "components": dict(top_decision.components) if top_decision else {},
                    "reason": top_decision.reason if top_decision else "idle",
                },
                "graph_feedback": {
                    entity_id: {
                        "familiarity": signal.familiarity,
                        "structural_relevance": signal.structural_relevance,
                        "knowledge_gap": signal.knowledge_gap,
                        "evidence_count": signal.evidence_count,
                        "edge_count": signal.edge_count,
                    }
                    for entity_id, signal in tick.graph_feedback.items()
                },
                "observation_policy": dict(tick.observation_policy),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

    def record_gpu_state(
        self,
        ram_used_mb: float | None,
        ram_total_mb: float | None,
        gpu_load_percent: float | None,
        processes: list,
    ) -> None:
        """Real, OS-level GPU/VRAM occupancy: aggregate RAM/GPU-load from `tegrastats`
        directly, per-process breakdown from jetson-stats. Independent of any daemon's
        self-reported model state (which record_object_learning/asr telemetry cannot
        verify on its own)."""
        top_processes = sorted(
            (
                {
                    "pid": row[0],
                    "user": row[1],
                    "name": row[9],
                    "state": row[5],
                    "cpu_percent": round(float(row[6]), 1),
                    "memory_mb": round(row[7] / 1024, 1),
                    "gpu_memory_mb": round(row[8] / 1024, 1),
                }
                for row in processes
                if isinstance(row, (list, tuple)) and len(row) >= 10 and row[8]
            ),
            key=lambda item: item["gpu_memory_mb"],
            reverse=True,
        )[:8]
        with self._lock:
            self._gpu = {
                "ram_total_mb": ram_total_mb,
                "ram_used_mb": ram_used_mb,
                "gpu_load_percent": gpu_load_percent,
                "processes": top_processes,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

    def record_activity(
        self,
        scale: float,
        state: str,
        last_source: str | None,
        modalities: list[dict[str, object]],
    ) -> None:
        """Current novelty/presence/sound-driven alertness (1.0 = full
        inference rate, `idle_floor` = quiet-room falloff) and the effective
        processing rate it currently yields per perception modality."""
        with self._lock:
            self._activity = {
                "scale": round(scale, 4),
                "state": state,
                "last_source": last_source,
                "modalities": modalities,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

    def record_consolidation(self, result: dict[str, object]) -> None:
        with self._lock:
            self._consolidation = dict(result)
            self._consolidation["at"] = datetime.now(timezone.utc).isoformat()

    def record_observation(self, observation: Observation) -> None:
        detections = [
            {
                "label": detection.label,
                "confidence": round(detection.confidence, 3),
                "bbox": [round(value, 1) for value in (detection.bbox.x1, detection.bbox.y1, detection.bbox.x2, detection.bbox.y2)],
                "mask_polygon": detection.attributes.get("mask_polygon"),
                "pose_keypoints": detection.attributes.get("pose_keypoints"),
                "behavior": detection.attributes.get("behavior"),
                "identity": detection.attributes.get("identity"),
                "identity_id": detection.attributes.get("identity_id"),
                "identity_kind": detection.attributes.get("identity_kind"),
                "identity_needs_name": detection.attributes.get("identity_needs_name"),
                "identity_persistent": detection.attributes.get("identity_persistent"),
                "identity_confidence": detection.attributes.get("identity_confidence"),
                "identity_recalled": detection.attributes.get("identity_recalled"),
                "identity_sightings": detection.attributes.get("identity_sightings"),
                "identity_outcome": detection.attributes.get("identity_outcome"),
                "identity_confidence_components": detection.attributes.get("identity_confidence_components"),
                "identity_temporal_association": detection.attributes.get(
                    "identity_temporal_association"
                ),
                "base_label": detection.attributes.get("base_label"),
                "object_id": detection.attributes.get("object_id"),
                "object_recall_confidence": detection.attributes.get("object_recall_confidence"),
                "object_label_source": detection.attributes.get("object_label_source"),
                "object_evidence_count": detection.attributes.get("object_evidence_count"),
                "object_label_provenance": detection.attributes.get("object_label_provenance"),
                "object_confidence_components": detection.attributes.get("object_confidence_components"),
            }
            for detection in observation.detections
        ]
        with self._lock:
            camera = self._cameras[observation.camera_id]
            now = time.monotonic()
            if camera.last_detection_monotonic is not None:
                camera.inference_fps = round(
                    1 / max(now - camera.last_detection_monotonic, 0.001), 2
                )
            camera.last_detection_monotonic = now
            camera.detection_sequence += 1
            camera.detections = detections
            camera.semantic_labels = list(observation.semantic_labels)
            camera.detections_updated_at = datetime.now(timezone.utc).isoformat()
            self._microphone_direction = observation.microphone_direction
            self._scene.update(observation)

    def record_frame(self, camera_id: str, frame_jpeg: bytes, frame_shape: tuple[int, int, int], fps: float) -> None:
        with self._lock:
            camera = self._cameras[camera_id]
            camera.frame_jpeg = frame_jpeg
            camera.frame_sequence += 1
            camera.frame_shape = frame_shape
            camera.fps = round(fps, 1)
            camera.updated_at = datetime.now(timezone.utc).isoformat()

    def next_uncertain_observation(self) -> dict[str, object] | None:
        with self._lock:
            item = self._scene.next_uncertain()
            if item is None:
                return None
            return {
                "id": item.track_id,
                "label": item.label,
                "confidence": round(item.detection.confidence, 3),
                "object_id": item.detection.attributes.get("object_id"),
                "label_source": item.detection.attributes.get("object_label_source"),
            }

    def pending_observation(self) -> dict[str, object] | None:
        with self._lock:
            item = self._scene.pending()
            if item is None:
                return None
            return {
                "id": item.track_id,
                "label": item.label,
                "confidence": round(item.detection.confidence, 3),
                "object_id": item.detection.attributes.get("object_id"),
                "label_source": item.detection.attributes.get("object_label_source"),
            }

    def resolve_observation_correction(self, decision: str, label: str | None = None) -> dict[str, object] | None:
        with self._lock:
            item = self._scene.resolve_pending(decision, label)
            if item is None:
                return None
            return {
                "id": item.track_id,
                "label": item.label,
                "confidence": round(item.detection.confidence, 3),
                "object_id": item.detection.attributes.get("object_id"),
                "label_source": item.detection.attributes.get("object_label_source"),
            }

    def dismiss_pending_observation(self) -> None:
        with self._lock:
            self._scene.dismiss_pending()

    def record_calibration_frame(
        self, camera_id: str, frame_jpeg: bytes, frame_shape: tuple[int, int, int], fps: float
    ) -> None:
        with self._lock:
            camera = self._cameras[camera_id]
            camera.frame_jpeg = frame_jpeg
            camera.frame_sequence += 1
            camera.frame_shape = frame_shape
            camera.fps = round(fps, 1)
            camera.updated_at = datetime.now(timezone.utc).isoformat()

    def frame(self, camera_id: str) -> bytes | None:
        with self._lock:
            camera = self._cameras.get(camera_id)
            return camera.frame_jpeg if camera else None

    def frame_snapshot(self, camera_id: str) -> tuple[int, bytes] | None:
        with self._lock:
            camera = self._cameras.get(camera_id)
            if camera is None or camera.frame_jpeg is None:
                return None
            return camera.frame_sequence, camera.frame_jpeg

    def snapshot(self, config: EggConfig) -> dict[str, object]:
        with self._lock:
            cameras = [
                {
                    "id": camera.camera_id,
                    "source": camera.source,
                    "configured_rotation": camera.configured_rotation,
                    "resolved_rotation": camera.resolved_rotation,
                    "frame_shape": camera.frame_shape,
                    "fps": camera.fps,
                    "frame_sequence": camera.frame_sequence,
                    "detection_sequence": camera.detection_sequence,
                    "inference_fps": camera.inference_fps,
                    "detections": camera.detections,
                    "semantic_labels": camera.semantic_labels,
                    "updated_at": camera.updated_at,
                    "detections_updated_at": camera.detections_updated_at,
                    "raw_frame_url": f"/api/cameras/{camera.camera_id}/raw.jpg",
                    "raw_stream_url": f"/api/cameras/{camera.camera_id}/stream.mjpg",
                    "frame_url": f"/api/cameras/{camera.camera_id}/raw.jpg",
                }
                for camera in self._cameras.values()
            ]
            seen = self._scene.snapshot()
            return {
                "cameras": cameras,
                "waveform": self._waveform,
                "waveform_sequence": self._waveform_sequence,
                "waveform_updated_at": self._waveform_updated_at,
                "audio_rms": self._audio_rms,
                "microphone_direction": self._microphone_direction,
                "respeaker": dict(self._respeaker),
                "vad": {
                    "speech": self._vad_speech,
                    "speech_ratio": self._vad_speech_ratio,
                    "speech_ms": self._vad_speech_ms,
                },
                "latest_transcript": self._latest_transcript,
                "latest_transcript_at": self._latest_transcript_at,
                "transcript_count": self._transcript_count,
                "transcript_history": list(self._transcript_history),
                "asr": dict(self._asr),
                "audio_comprehension": {
                    **self._audio_comprehension,
                    "classifications": [
                        dict(item)
                        for item in self._audio_comprehension["classifications"]
                    ],
                },
                "latest_reply": self._latest_reply,
                "runtime_errors": list(self._runtime_errors),
                "object_learning": dict(self._object_learning),
                "ocr": {**self._ocr, "recent": list(self._ocr["recent"])},
                "memory": dict(self._memory),
                "brain": {**self._brain, "memory": dict(self._memory)},
                "default_mode": dict(self._default_mode),
                "narrative_semantics": dict(self._narrative_semantics),
                "gpu": dict(self._gpu),
                "activity": dict(self._activity),
                "environmental_cognition": {
                    **self._environmental_cognition,
                    "last": (
                        dict(self._environmental_cognition["last"])
                        if isinstance(self._environmental_cognition["last"], dict)
                        else None
                    ),
                    "recent": [
                        dict(item)
                        for item in self._environmental_cognition["recent"]
                    ],
                },
                "attention_decisions": list(self._attention_decisions),
                "interaction_decisions": list(self._interaction_decisions),
                "tool_calls": list(self._tool_calls),
                "graph_activations": {
                    "sequence": self._graph_activation_sequence,
                    "events": [dict(event) for event in self._graph_activations],
                },
                "identity_dialogue": dict(self._identity_dialogue),
                "identity_continuity": {
                    **self._identity_continuity,
                    "last": (
                        dict(self._identity_continuity["last"])
                        if isinstance(self._identity_continuity["last"], dict)
                        else None
                    ),
                    "recent": [
                        dict(item) for item in self._identity_continuity["recent"]
                    ],
                },
                "consolidation": dict(self._consolidation),
                "retrieval_hits": list(self._retrieval_hits),
                "seen": seen,
                "pending_observation": (
                    {"id": item.track_id, "label": item.label, "confidence": round(item.detection.confidence, 3)}
                    if (item := self._scene.pending())
                    else None
                ),
                "voice": {
                    "asr_segment_seconds": config.transcription.segment_seconds,
                    "asr_rms_threshold": config.transcription.rms_threshold,
                    "asr_target_rms": config.audio.asr_target_rms,
                    "asr_max_gain": config.audio.asr_max_gain,
                    "asr_model": config.transcription.asr_model,
                    "asr_language": config.transcription.asr_language,
                    "vad_aggressiveness": config.transcription.vad_aggressiveness,
                    "vad_input_gain": config.transcription.vad_input_gain,
                    "vad_min_voiced_rms": config.transcription.vad_min_voiced_rms,
                    "vad_min_contiguous_ms": config.transcription.vad_min_contiguous_ms,
                    "tts_model": config.omnius.voice_model,
                    "tts_voice": config.omnius.voice_name,
                    "asr_input": f"ReSpeaker DSP ASR channel {config.audio.asr_channel}",
                    "barge_in_enabled": config.audio.barge_in_enabled,
                    "vad_hangover_ms": config.transcription.vad_hangover_ms,
                    "vad_hangover_max_ms": (
                        config.transcription.vad_hangover_max_ms
                        or config.transcription.vad_hangover_ms
                    ),
                    **dict(self._voice_runtime),
                },
            }
