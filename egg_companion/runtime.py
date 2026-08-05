from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

from egg_companion.adapters.audio import ReSpeakerCapture, ReSpeakerDirection, ReSpeakerWaveformCapture
from egg_companion.adapters.camera import CameraStream
from egg_companion.adapters.omnius import OmniusClient
from egg_companion.adapters.speaker import Speaker
from egg_companion.adapters.system_service import SystemServiceClient
from egg_companion.adapters.vision import SegmentedObject, VisionEngine
from egg_companion.cognition.dialogue import DialogueClassifier, DialogueEvidence
from egg_companion.config import EggConfig
from egg_companion.core.attention import AttentionManager
from egg_companion.core.cognition import CognitiveAttentionController, InteractionPolicy
from egg_companion.memory.pipeline import MemoryPipeline
from egg_companion.memory.buffer import BufferedMediaRef, PerceptualBuffer
from egg_companion.memory.fusion import EvidenceFusion
from egg_companion.memory.migrate_legacy import LegacyMemoryMigrator
from egg_companion.memory.store import MemoryStore
from egg_companion.models import AttentionTarget, Detection, EvidenceRef, Observation, PerceptualEvent
from egg_companion.services.telemetry import RuntimeTelemetry
from egg_companion.services.identity import IdentityLibrary
from egg_companion.services.object_library import ObjectLibrary

logger = logging.getLogger(__name__)


class CompanionRuntime:
    def __init__(self, config: EggConfig) -> None:
        self.config = config
        self.telemetry = RuntimeTelemetry(config)
        self._vision: VisionEngine | None = None
        self._attention = AttentionManager(
            track_ttl_seconds=config.attention.track_ttl_seconds,
            min_priority=config.attention.min_priority,
            max_targets=config.attention.max_targets,
        )
        self._cognitive_attention = CognitiveAttentionController(
            config.cognitive_attention, config.attention.proactive_speech_enabled
        )
        self._interaction_policy = InteractionPolicy()
        self._dialogue = DialogueClassifier()
        self._direction = ReSpeakerDirection(config.audio)
        self._capture = ReSpeakerCapture(
            config.audio,
            config.transcription,
        )
        self._waveform_capture = ReSpeakerWaveformCapture(config.audio)
        self._speaker = Speaker(config.audio)
        self._omnius = OmniusClient(config.omnius)
        self._system_service = SystemServiceClient(config.system_service) if config.system_service else None
        try:
            self.identities = IdentityLibrary(config.identity)
        except Exception as error:
            logger.exception("identity library unavailable; persistent identity is degraded")
            self.telemetry.record_runtime_error("identity-library", error)
            self.identities = IdentityLibrary(config.identity.model_copy(update={"enabled": False}))
        try:
            self.objects = ObjectLibrary(config.object_learning)
        except Exception as error:
            logger.exception("object library unavailable; object recall is degraded")
            self.telemetry.record_runtime_error("object-library", error)
            self.objects = ObjectLibrary(
                config.object_learning.model_copy(update={"enabled": False})
            )
        self._observations: asyncio.Queue[Observation] = asyncio.Queue(config.runtime.event_queue_size)
        self._speech_segments: asyncio.Queue[bytes] = asyncio.Queue(config.runtime.speech_queue_size)
        self._utterances: asyncio.Queue[str] = asyncio.Queue(config.runtime.reasoning_queue_size)
        self._memory_events: asyncio.Queue[PerceptualEvent] = asyncio.Queue(config.runtime.event_queue_size)
        self._perceptual_buffer = PerceptualBuffer(config.memory)
        self._object_candidates: asyncio.Queue[
            tuple[str, Detection, SegmentedObject, str, int]
        ] = asyncio.Queue(maxsize=4)
        self._memory = None
        if config.memory.enabled:
            try:
                memory_store = MemoryStore(config.memory)
                LegacyMemoryMigrator(memory_store, self.identities, self.objects).run()
                self._memory = MemoryPipeline(config, memory_store)
            except Exception as error:
                logger.exception("cognitive memory unavailable; live sensing remains active")
                self.telemetry.record_runtime_error("cognitive-memory", error)
        self._last_greeting: datetime | None = None
        self._latest_observation: Observation | None = None
        self._speaking = False
        self._asr_holdoff_until = 0.0
        self._last_spoken_at: float | None = None
        self._camera_rotations = {camera.id: camera.rotation_degrees if isinstance(camera.rotation_degrees, int) else None for camera in config.cameras}
        self._last_rotation_attempt = {camera.id: 0.0 for camera in config.cameras}
        self._latest_frame: np.ndarray | None = None
        self._last_object_candidate_at = 0.0
        self._last_vlm_at = 0.0
        self._object_recall_lock = threading.Lock()
        self._object_recalls: dict[str, list[dict[str, object]]] = {}
        self._object_candidate_fingerprints: dict[str, float] = {}
        self._identity_name_questions: set[str] = set()
        self._last_valid_speech_at = 0.0
        self._object_candidate_tracks: dict[str, list[dict[str, object]]] = {}

    async def update_voice_config(
        self,
        segment_seconds: float | None,
        rms_threshold: float | None,
        voice_model: str | None,
        voice_name: str | None,
        asr_model: str | None,
    ) -> None:
        if segment_seconds is not None:
            self.config.transcription.segment_seconds = segment_seconds
        if rms_threshold is not None:
            self.config.transcription.rms_threshold = rms_threshold
        self._capture = ReSpeakerCapture(
            self.config.audio,
            self.config.transcription,
        )
        if voice_model and voice_model != self.config.omnius.voice_model:
            self.config.omnius.voice_model = voice_model
            await self._omnius.ensure_voice_ready()
        if voice_name is not None:
            self.config.omnius.voice_name = voice_name or None
            await self._omnius.configure_supertonic_voice(self.config.omnius.voice_name)
        if asr_model and asr_model != self.config.transcription.asr_model:
            self.config.transcription.asr_model = asr_model
            await self._omnius.ensure_asr_model(asr_model)

    def memory_snapshot(self) -> dict[str, object]:
        snapshot = self._memory.governance_snapshot() if self._memory else {}
        buffered = self._perceptual_buffer.snapshot(datetime.now(timezone.utc))
        snapshot["transient_buffer"] = {
            "camera_count": len(buffered["frames"]),
            "frame_references": sum(len(items) for items in buffered["frames"].values()),
            "audio_references": len(buffered["audio"]),
            "bytes": buffered["bytes"],
        }
        return snapshot

    def inspect_memory_entity(self, entity_id: str) -> dict[str, object] | None:
        return self._memory.inspect_entity(entity_id) if self._memory else None

    def memory_episodes(self) -> list[dict[str, object]]:
        return self._memory.episodes() if self._memory else []

    def memory_claims(self) -> list[dict[str, object]]:
        return self._memory.claims() if self._memory else []

    def add_memory_alias(self, entity_id: str, alias: str) -> dict[str, object]:
        if self._memory is None:
            raise RuntimeError("cognitive memory is disabled")
        result = self._memory.add_alias(entity_id, alias)
        profile = self.identities.name_profile(entity_id, alias)
        if profile:
            record = self.identities.profile_record(profile.profile_id)
            if record:
                self._memory.sync_identity_profile(record)
        return result

    def correct_memory_claim(self, claim_id: str, replacement: str) -> dict[str, object]:
        if self._memory is None:
            raise RuntimeError("cognitive memory is disabled")
        result = self._memory.correct_claim(claim_id, replacement)
        if result["predicate"] == "has_label":
            profile = self.objects.relabel(
                str(result["subject_id"]), replacement, 1.0, "user", "dashboard-governance",
                {"claim_id": claim_id, "corrected_at": datetime.now(timezone.utc).isoformat()},
            )
            if profile:
                record = self.objects.profile_record(profile.profile_id)
                if record:
                    self._memory.sync_object_profile(record)
        return result

    def export_memory(self) -> dict[str, object]:
        if self._memory is None:
            raise RuntimeError("cognitive memory is disabled")
        return self._memory.export()

    def export_memory_entity(self, entity_id: str) -> dict[str, object]:
        if self._memory is None:
            raise RuntimeError("cognitive memory is disabled")
        return self._memory.export_entity(entity_id)

    def revise_memory(
        self, target_type: str, target_id: str, decision: str, replacement: str | None
    ) -> dict[str, object]:
        if self._memory is None:
            raise RuntimeError("cognitive memory is disabled")
        return self._memory.revise(target_type, target_id, decision, replacement)

    def delete_memory_entity(self, entity_id: str) -> None:
        if self._memory is None:
            raise RuntimeError("cognitive memory is disabled")
        if self._memory.inspect_entity(entity_id) is None:
            raise KeyError(entity_id)
        self._memory.delete_entity(entity_id)
        self.identities.delete(entity_id)
        self.objects.delete(entity_id)

    async def run(self) -> None:
        try:
            self._direction.start()
        except Exception as error:
            logger.exception("ReSpeaker direction reader failed to start; other components remain active")
            self.telemetry.record_runtime_error("respeaker-direction", error)
        camera_tasks = [
            asyncio.create_task(
                self._run_component(
                    f"camera:{camera.id}",
                    lambda camera=camera: self._observe_camera(CameraStream(camera)),
                ),
                name=f"camera:{camera.id}",
            )
            for camera in self.config.cameras
            if camera.enabled
        ]
        component_specs = [
            ("vision-readiness", self._maintain_vision),
            ("omnius-readiness", self._maintain_omnius),
            ("attention", self._attend),
            ("audio-waveform", self._stream_waveform),
            ("speech-recognition", self._process_speech),
            ("conversation-reasoning", self._reason_about_transcript),
            ("ornith-object-labeler", self._auto_label_objects),
        ]
        tasks = camera_tasks + [
            asyncio.create_task(self._run_component(name, component), name=name)
            for name, component in component_specs
        ]
        memory_task = (
            asyncio.create_task(
                self._run_component("memory-writer", self._persist_memory_events),
                name="memory-writer",
            )
            if self._memory
            else None
        )
        consolidation_task = (
            asyncio.create_task(
                self._run_component("memory-consolidation", self._consolidate_memory),
                name="memory-consolidation",
            )
            if self._memory else None
        )
        if memory_task:
            tasks.append(memory_task)
        if consolidation_task:
            tasks.append(consolidation_task)
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self._memory:
                await asyncio.to_thread(self._memory.close, datetime.now(timezone.utc))
            await asyncio.to_thread(self._waveform_capture.close)
            self._direction.stop()

    async def _run_component(self, name: str, component) -> None:
        """Keep an independent subsystem alive without taking down healthy peers."""
        while True:
            try:
                await component()
                raise RuntimeError("component stopped unexpectedly")
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.exception("%s failed; retrying without stopping other components", name)
                self.telemetry.record_runtime_error(name, error)
                await asyncio.sleep(1)

    async def _maintain_omnius(self) -> None:
        await self._omnius.health()
        await self._omnius.ensure_voice_ready()
        await self._omnius.configure_supertonic_voice(self.config.omnius.voice_name)
        await self._omnius.ensure_asr_model(self.config.transcription.asr_model)
        while True:
            await asyncio.sleep(60)
            await self._omnius.health()

    async def _maintain_vision(self) -> None:
        if self._vision is None:
            self._vision = await asyncio.to_thread(VisionEngine, self.config.vision)
        await asyncio.Event().wait()

    async def _observe_camera(self, camera: CameraStream) -> None:
        latest_observation: Observation | None = None
        analysis_task: asyncio.Task[Observation] | None = None
        next_analysis_at = 0.0
        next_pose_at = 0.0
        next_semantic_at = 0.0
        next_preview_at = 0.0
        while True:
            previous_frame_at = time.monotonic()
            try:
                async for frame in camera.frames():
                    now = time.monotonic()
                    fps = 1 / max(now - previous_frame_at, 0.001)
                    previous_frame_at = now
                    angle = self._camera_rotations[camera.config.id]
                    vision = self._vision
                    if angle is None:
                        if vision is None:
                            if now >= next_preview_at:
                                preview = await asyncio.to_thread(self._encode_frame, frame)
                                self.telemetry.record_calibration_frame(
                                    camera.config.id, preview, frame.shape, fps
                                )
                                next_preview_at = now + 1 / self.config.vision.dashboard_fps
                            continue
                        if now - self._last_rotation_attempt[camera.config.id] < 1.0:
                            continue
                        self._last_rotation_attempt[camera.config.id] = now
                        preview = await asyncio.to_thread(self._encode_frame, frame)
                        self.telemetry.record_calibration_frame(camera.config.id, preview, frame.shape, fps)
                        angle = await asyncio.to_thread(vision.detect_rotation, frame)
                        if angle is None:
                            logger.debug("camera %s orientation calibration awaiting an upright pose", camera.config.id)
                            continue
                        if frame.shape[0] > frame.shape[1] and angle in {0, 180}:
                            logger.warning(
                                "camera %s rejected %s-degree calibration for portrait source %s",
                                camera.config.id,
                                angle,
                                frame.shape[:2],
                            )
                            continue
                        self._camera_rotations[camera.config.id] = angle
                        self.telemetry.set_rotation(camera.config.id, angle)
                        logger.info("camera %s orientation locked to %s degrees", camera.config.id, angle)
                    frame = self._rotate_frame(frame, angle)
                    self._latest_frame = frame.copy()
                    if analysis_task is not None and analysis_task.done():
                        try:
                            observation = analysis_task.result()
                        except Exception as error:
                            logger.exception("camera %s analysis failed", camera.config.id)
                            self.telemetry.record_runtime_error("vision", error)
                        else:
                            latest_observation = observation
                            self._latest_observation = observation
                            self._queue_vision_memory(observation)
                            asyncio.create_task(self._queue_object_candidate(frame.copy(), observation), name="object-candidate")
                            self.telemetry.record_observation(observation)
                            candidate = self.telemetry.next_uncertain_observation()
                            if candidate:
                                asyncio.create_task(self._ask_observation_correction(candidate), name="observation-correction")
                            if self._observations.full():
                                discarded = self._observations.get_nowait()
                                logger.debug("discarded stale observation from %s", discarded.camera_id)
                            self._observations.put_nowait(observation)
                        analysis_task = None
                    if vision is not None and analysis_task is None and now >= next_analysis_at:
                        include_pose = now >= next_pose_at
                        include_semantics = now >= next_semantic_at
                        if include_pose:
                            next_pose_at = now + 1 / self.config.vision.pose_fps
                        if include_semantics:
                            next_semantic_at = now + 1 / self.config.vision.semantic_fps
                        analysis_task = asyncio.create_task(
                            asyncio.to_thread(
                                self._analyze,
                                camera.config.id,
                                frame.copy(),
                                include_pose,
                                include_semantics,
                            ),
                            name=f"vision-analysis:{camera.config.id}",
                        )
                        next_analysis_at = now + 1 / self.config.vision.analysis_fps
                    if now >= next_preview_at:
                        raw_frame = await asyncio.to_thread(self._encode_frame, frame)
                        self.telemetry.record_frame(camera.config.id, raw_frame, frame.shape, fps)
                        self._perceptual_buffer.append_frame(
                            BufferedMediaRef(
                                camera.config.id,
                                datetime.now(timezone.utc),
                                f"volatile://{camera.config.id}/{time.monotonic_ns()}.jpg",
                                len(raw_frame),
                                {"frame_shape": list(frame.shape), "retained": False},
                            )
                        )
                        next_preview_at = now + 1 / self.config.vision.dashboard_fps
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.exception("camera %s stream failed; retrying", camera.config.id)
                self.telemetry.record_runtime_error(f"camera:{camera.config.id}", error)
                await asyncio.sleep(1)
            finally:
                if analysis_task is not None and not analysis_task.done():
                    analysis_task.cancel()
                    await asyncio.gather(analysis_task, return_exceptions=True)
                analysis_task = None

    def _analyze(
        self, camera_id: str, frame: np.ndarray, include_pose: bool = True, include_semantics: bool = True
    ) -> Observation:
        vision = self._vision
        if vision is None:
            raise RuntimeError("vision engine is not ready")
        detections, semantic_labels = vision.analyze(frame, include_pose, include_semantics)
        detections = self._apply_object_recalls(camera_id, detections)
        matches = self.identities.observe(camera_id, frame, detections, vision)
        detections = tuple(
            replace(
                detection,
                attributes={
                    **detection.attributes,
                    "identity": matches[index]["label"],
                    "identity_id": matches[index]["id"],
                    "identity_confidence": matches[index]["confidence"],
                    "identity_needs_name": matches[index]["needs_name"],
                    "identity_new": matches[index]["new"],
                    "identity_recalled": matches[index]["recalled"],
                    "identity_sightings": matches[index]["sightings"],
                    "identity_kind": matches[index]["kind"],
                    "identity_outcome": matches[index]["resolver_outcome"],
                    "identity_confidence_components": matches[index]["confidence_components"],
                },
            )
            if index in matches
            else detection
            for index, detection in enumerate(detections)
        )
        return Observation(
            camera_id=camera_id,
            timestamp=datetime.now(timezone.utc),
            detections=detections,
            semantic_labels=semantic_labels,
            microphone_direction=self._direction.latest_angle(),
        )

    async def _attend(self) -> None:
        while True:
            observation = await self._observations.get()
            for target in self._attention.select(observation):
                await self._handle_target(target, observation)

    async def _stream_waveform(self) -> None:
        pending = np.empty(0, dtype=np.float32)
        next_vad_preview_at = 0.0
        while True:
            try:
                samples = await asyncio.to_thread(self._waveform_capture.read_chunk)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.exception("ReSpeaker waveform capture failed; retrying")
                self.telemetry.record_runtime_error("audio-waveform", error)
                await asyncio.sleep(0.5)
                continue
            self.telemetry.record_waveform(samples)
            if self._speaking or time.monotonic() < self._asr_holdoff_until:
                pending = np.empty(0, dtype=np.float32)
                continue
            pending = np.concatenate((pending, samples))
            now = time.monotonic()
            if pending.size >= self.config.audio.sample_rate and now >= next_vad_preview_at:
                preview = pending[-self.config.audio.sample_rate:]
                preview_rms = await asyncio.to_thread(self._capture.analyze_samples, preview)
                self.telemetry.record_audio_state(
                    preview_rms,
                    self._capture.last_speech_detected,
                    self._capture.last_speech_ratio,
                    self._capture.last_speech_ms,
                )
                next_vad_preview_at = now + 0.1
            segment_samples = round(
                self.config.audio.sample_rate * self.config.transcription.segment_seconds
            )
            while pending.size >= segment_samples:
                segment = pending[:segment_samples]
                pending = pending[segment_samples:]
                try:
                    audio, rms = await asyncio.to_thread(self._capture.process_samples, segment)
                except Exception as error:
                    logger.exception("ReSpeaker ASR segment processing failed")
                    self.telemetry.record_runtime_error("audio-segmentation", error)
                    continue
                self.telemetry.record_audio(
                    rms,
                    self._capture.last_speech_detected,
                    self._capture.last_speech_ratio,
                    self._capture.last_speech_ms,
                )
                if rms < self.config.transcription.rms_threshold or not self._capture.last_speech_detected:
                    logger.debug(
                        "discarded non-speech capture: rms=%.5f speech_ms=%s speech_ratio=%.3f",
                        rms,
                        self._capture.last_speech_ms,
                        self._capture.last_speech_ratio,
                    )
                    continue
                self._last_valid_speech_at = time.monotonic()
                if self._speech_segments.full():
                    self._speech_segments.get_nowait()
                    logger.warning("discarded stale speech segment while Omnius is busy")
                self._speech_segments.put_nowait(audio)

    async def _process_speech(self) -> None:
        while True:
            audio = await self._speech_segments.get()
            try:
                transcript = await self._omnius.transcribe(audio)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.exception("Omnius transcription failed")
                self.telemetry.record_asr_error(error)
                continue
            if not transcript:
                metadata = dict(self._omnius.last_transcription_metadata)
                self.telemetry.record_asr_rejection(
                    str(metadata.get("rejection_reason") or "empty or ungrounded transcript"), metadata
                )
                continue
            self.telemetry.record_transcript(transcript, self._omnius.last_transcription_metadata)
            self._perceptual_buffer.append_audio(
                BufferedMediaRef(
                    "respeaker-asr",
                    datetime.now(timezone.utc),
                    f"volatile://respeaker/{time.monotonic_ns()}.wav",
                    len(audio),
                    {
                        "vad": True,
                        "rms": self._capture.last_voiced_rms,
                        "speech_ratio": self._capture.last_speech_ratio,
                        "retained": False,
                    },
                )
            )
            self._queue_speech_memory(transcript)
            if self._utterances.full():
                self._utterances.get_nowait()
                logger.warning("discarded stale utterance while Omnius reasoning is busy")
            self._utterances.put_nowait(transcript)

    def _queue_vision_memory(self, observation: Observation) -> None:
        if self._memory is None:
            return
        detections = [
            {
                "label": detection.label,
                "confidence": round(detection.confidence, 3),
                "identity_id": detection.attributes.get("identity_id"),
                "object_id": detection.attributes.get("object_id"),
                "behavior": detection.attributes.get("behavior"),
            }
            for detection in observation.detections
        ]
        entities = []
        for detection in observation.detections:
            identity_id = detection.attributes.get("identity_id")
            if identity_id:
                kind = str(detection.attributes.get("identity_kind") or "appearance")
                entities.append(
                    {
                        "id": str(identity_id),
                        "type": "person" if kind == "face" else "appearance_track",
                        "label": detection.attributes.get("identity"),
                        "confidence": detection.attributes.get("identity_confidence"),
                        "kind": kind,
                        "resolver_outcome": detection.attributes.get("identity_outcome"),
                        "face_similarity": (
                            detection.attributes.get("identity_confidence_components") or {}
                        ).get("face_similarity"),
                        "clip_similarity": (
                            detection.attributes.get("identity_confidence_components") or {}
                        ).get("clip_similarity"),
                        "source": "identity-library",
                        "camera_id": observation.camera_id,
                    }
                )
            object_id = detection.attributes.get("object_id")
            if object_id:
                entities.append(
                    {
                        "id": str(object_id),
                        "type": "object",
                        "label": detection.label,
                        "confidence": detection.attributes.get("object_recall_confidence", detection.confidence),
                        "source": detection.attributes.get("object_label_source", "object-library"),
                        "camera_id": observation.camera_id,
                        "base_label": detection.attributes.get("base_label"),
                    }
                )
        event_id = str(uuid4())
        evidence = EvidenceRef(
            evidence_id=str(uuid4()), modality="vision", captured_at=observation.timestamp,
            source_type="camera", source_id=observation.camera_id,
            quality=sum(item["confidence"] for item in detections) / max(1, len(detections)),
            metadata={"detections": detections, "semantic_labels": list(observation.semantic_labels)},
        )
        self._queue_memory_event(
            PerceptualEvent(
                event_id=event_id, event_type="vision", occurred_at=observation.timestamp, source_id=observation.camera_id,
                evidence=(evidence,),
                entity_ids=tuple(
                    str(entity_id)
                    for item in detections
                    for entity_id in (item["identity_id"], item["object_id"])
                    if entity_id
                ),
                payload={
                    "labels": [item["label"] for item in detections],
                    "scene_labels": list(observation.semantic_labels),
                    "behaviors": [item["behavior"] for item in detections if item["behavior"]],
                    "entities": entities,
                },
            )
        )

    def _queue_speech_memory(self, transcript: str) -> None:
        if self._memory is None:
            return
        now = datetime.now(timezone.utc)
        evidence = EvidenceRef(
            evidence_id=str(uuid4()), modality="speech", captured_at=now, source_type="respeaker", source_id="respeaker-asr",
            quality=self._capture.last_speech_ratio,
            metadata={
                "transcript": transcript,
                "rms": self.telemetry.snapshot(self.config)["audio_rms"],
                "vad_accepted": True,
                "vad_speech_ratio": self._capture.last_speech_ratio,
                "speech_ms": self._capture.last_speech_ms,
                "doa": self._direction.latest_angle(),
                "asr_model": self.config.transcription.asr_model,
                "asr_service": str(self.config.omnius.base_url),
                "asr_metadata": dict(self._omnius.last_transcription_metadata),
            },
        )
        self._queue_memory_event(
            PerceptualEvent(str(uuid4()), "speech", now, "respeaker-asr", (evidence,), payload={"transcript": transcript})
        )

    def _queue_memory_event(self, event: PerceptualEvent) -> None:
        if self._memory_events.full():
            self._memory_events.get_nowait()
            logger.warning("discarded stale memory event while local writer is busy")
        self._memory_events.put_nowait(event)

    async def _persist_memory_events(self) -> None:
        while True:
            event = await self._memory_events.get()
            try:
                if self._memory is None:
                    continue
                accepted, closed = await asyncio.to_thread(self._memory.ingest, event)
                self.telemetry.record_memory(
                    accepted,
                    closed,
                    self._memory.accepted_events,
                    self._memory.closed_episodes,
                    self._memory.lifecycle_snapshot(),
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.exception("cognitive memory write failed")
                self.telemetry.record_runtime_error("memory", error)

    async def _consolidate_memory(self) -> None:
        while True:
            await asyncio.sleep(self.config.memory.consolidation_interval_seconds)
            if self._speaking or not self._memory_events.empty() or not self._utterances.empty():
                continue
            try:
                if self._memory is not None:
                    result = await asyncio.to_thread(self._memory.consolidate)
                    self.telemetry.record_consolidation(result)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.exception("memory consolidation failed")
                self.telemetry.record_runtime_error("memory-consolidation", error)

    async def _queue_object_candidate(self, frame: np.ndarray, observation: Observation) -> None:
        learning = self.config.object_learning
        now = time.monotonic()
        if (
            not learning.auto_label_enabled
            or self._object_candidates.full()
            or now - self._last_object_candidate_at < learning.recall_interval_seconds
        ):
            return
        candidates = [
            detection for detection in observation.detections
            if detection.label != "person"
            and not detection.attributes.get("object_id")
            and detection.attributes.get("mask_polygon")
            and self._candidate_area_ratio(detection) <= 0.35
        ]
        candidate = min(candidates, key=lambda item: item.confidence, default=None)
        if candidate is None:
            return
        if not self._candidate_is_stable(observation.camera_id, candidate, now):
            return
        self._last_object_candidate_at = now
        vision = self._vision
        if vision is None:
            return
        segmented = await asyncio.to_thread(vision.segment_detection, frame, candidate)
        if segmented is None:
            return
        fingerprint = await asyncio.to_thread(self._segmented_fingerprint, segmented)
        self._object_candidate_fingerprints = {
            key: expires_at for key, expires_at in self._object_candidate_fingerprints.items()
            if expires_at > now
        }
        if fingerprint in self._object_candidate_fingerprints:
            self.telemetry.record_object_learning("duplicate_candidate", candidate.label)
            return
        self._object_candidate_fingerprints[fingerprint] = now + max(
            learning.auto_label_cooldown_seconds * 2, learning.recall_cache_seconds
        )
        self.telemetry.record_object_learning(
            "stable_candidate", f"{observation.camera_id}:{candidate.label}"
        )
        self._object_candidates.put_nowait((observation.camera_id, candidate, segmented, fingerprint, 0))

    async def _auto_label_objects(self) -> None:
        while self._vision is None:
            await asyncio.sleep(1)
        vision = self._vision
        for profile_id, previous_label, _ in self.objects.profiles_for_review():
            segmented = await asyncio.to_thread(self.objects.segmented_profile, profile_id)
            if segmented is None:
                continue
            await self._review_existing_object(profile_id, previous_label, segmented)
        while True:
            camera_id, detection, segmented, fingerprint, attempt = await self._object_candidates.get()
            try:
                self.telemetry.record_object_learning("clip_query", detection.label)
                recalled = await asyncio.to_thread(self.objects.match, segmented, vision)
                if recalled is not None:
                    self._cache_object_recall(camera_id, detection, recalled[0], recalled[1])
                    self.telemetry.record_object_learning(
                        "clip_recall", f"{recalled[0].label}:{recalled[1]:.3f}"
                    )
                    logger.debug("CLIP recalled object %s without Ornith VLM", recalled[0].label)
                    continue
                if time.monotonic() - self._last_vlm_at < self.config.object_learning.auto_label_cooldown_seconds:
                    continue
                speech_age = time.monotonic() - self._last_valid_speech_at
                if speech_age < self.config.object_learning.speech_priority_seconds:
                    delay = self.config.object_learning.speech_priority_seconds - speech_age
                    self.telemetry.record_object_learning("speech_deferral", f"{delay:.1f}s")
                    await asyncio.sleep(delay)
                    if not self._object_candidates.full():
                        self._object_candidates.put_nowait(
                            (camera_id, detection, segmented, fingerprint, attempt)
                        )
                    continue
                self._last_vlm_at = time.monotonic()
                image_png = await asyncio.to_thread(
                    vision.encode_segmented_object, segmented,
                    self.config.object_learning.vlm_max_image_size,
                )
                self.telemetry.record_object_learning("vlm_request", detection.label)
                result = await self._omnius.classify_masked_object(image_png, detection.label, detection.confidence)
                if result is None or result[1] < self.config.object_learning.auto_label_min_confidence:
                    detail = "invalid response" if result is None else f"{result[0]}:{result[1]:.3f}"
                    self.telemetry.record_object_learning("vlm_rejection", detail)
                    continue
                label, confidence = result
                provenance = {
                    "model_id": self.config.omnius.vision_model,
                    "detector_label": detection.label,
                    "detector_confidence": detection.confidence,
                    "mask_checksum": hashlib.sha256(image_png).hexdigest(),
                    "classified_at": datetime.now(timezone.utc).isoformat(),
                }
                profile = await asyncio.to_thread(
                    self.objects.learn, label, segmented, vision, "ornith-vlm", confidence, provenance
                )
                if profile:
                    await self._sync_object_profile(profile.profile_id)
                    self._cache_object_recall(camera_id, detection, profile, confidence)
                    self.telemetry.record_object_learning(
                        "vlm_success", f"{profile.profile_id}:{profile.label}:{confidence:.3f}"
                    )
                    logger.info("Ornith VLM learned segmented object %s at %.2f", profile.label, confidence)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.exception("Ornith object classification failed")
                self.telemetry.record_object_learning("vlm_error", error)
                self.telemetry.record_runtime_error("ornith-vlm", error)
                if attempt < self.config.object_learning.auto_label_max_retries:
                    await asyncio.sleep(self.config.object_learning.auto_label_failure_backoff_seconds * (attempt + 1))
                    if not self._object_candidates.full():
                        self._object_candidates.put_nowait(
                            (camera_id, detection, segmented, fingerprint, attempt + 1)
                        )

    async def _review_existing_object(self, profile_id: str, previous_label: str, segmented: SegmentedObject) -> None:
        vision = self._vision
        if vision is None:
            return
        try:
            image_png = await asyncio.to_thread(
                vision.encode_segmented_object, segmented,
                self.config.object_learning.vlm_max_image_size,
            )
            self.telemetry.record_object_learning("vlm_request", f"review:{previous_label}")
            result = await self._omnius.classify_masked_object(image_png, previous_label, segmented.confidence)
            if result is None or result[1] < self.config.object_learning.auto_label_min_confidence:
                self.telemetry.record_object_learning("vlm_rejection", f"review:{previous_label}")
                await asyncio.to_thread(self.objects.mark_review_failed, profile_id)
                return
            provenance = {
                "detector_label": previous_label,
                "detector_confidence": segmented.confidence,
                "mask_checksum": hashlib.sha256(image_png).hexdigest(),
                "classified_at": datetime.now(timezone.utc).isoformat(),
            }
            profile = await asyncio.to_thread(
                self.objects.relabel, profile_id, result[0], result[1], "ornith-vlm",
                self.config.omnius.vision_model, provenance,
            )
            if profile:
                await self._sync_object_profile(profile.profile_id)
                self.telemetry.record_object_learning(
                    "vlm_success", f"review:{profile.profile_id}:{profile.label}:{result[1]:.3f}"
                )
        except Exception as error:
            await asyncio.to_thread(self.objects.mark_review_failed, profile_id)
            self.telemetry.record_object_learning("vlm_error", error)
            self.telemetry.record_runtime_error("ornith-review", error)

    def _cache_object_recall(self, camera_id: str, detection: Detection, profile, similarity: float) -> None:
        expires_at = time.monotonic() + self.config.object_learning.recall_cache_seconds
        fusion = EvidenceFusion.object(similarity)
        item = {
            "bbox": detection.bbox,
            "profile_id": profile.profile_id,
            "label": profile.label,
            "similarity": float(similarity),
            "label_source": profile.label_source,
            "evidence_count": profile.samples,
            "provenance": dict(profile.label_provenance),
            "confidence_components": fusion.components,
            "expires_at": expires_at,
        }
        with self._object_recall_lock:
            active = [value for value in self._object_recalls.get(camera_id, []) if value["expires_at"] > time.monotonic()]
            active.append(item)
            self._object_recalls[camera_id] = active[-12:]

    @staticmethod
    def _segmented_fingerprint(segmented: SegmentedObject) -> str:
        import cv2

        masked = cv2.bitwise_and(segmented.image, segmented.image, mask=segmented.mask)
        preview = cv2.resize(masked, (16, 16), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(segmented.mask, (16, 16), interpolation=cv2.INTER_AREA)
        quantized = (preview // 32).astype(np.uint8)
        quantized_alpha = (alpha // 64).astype(np.uint8)
        return hashlib.sha256(quantized.tobytes() + quantized_alpha.tobytes()).hexdigest()

    @staticmethod
    def _candidate_area_ratio(detection: Detection) -> float:
        shape = detection.attributes.get("frame_shape")
        if not isinstance(shape, list) or len(shape) != 2:
            return 1.0
        return detection.bbox.area / max(float(shape[0] * shape[1]), 1.0)

    def _candidate_is_stable(self, camera_id: str, detection: Detection, now: float) -> bool:
        tracks = [
            track for track in self._object_candidate_tracks.get(camera_id, [])
            if now - float(track["last_seen"]) <= 5.0
        ]
        match = max(
            tracks,
            key=lambda track: self._bbox_iou(detection.bbox, track["bbox"]),
            default=None,
        )
        if match is None or self._bbox_iou(detection.bbox, match["bbox"]) < 0.50:
            match = {"bbox": detection.bbox, "last_seen": now, "count": 1, "last_queued": 0.0}
            tracks.append(match)
        else:
            match["bbox"] = detection.bbox
            match["last_seen"] = now
            match["count"] = int(match["count"]) + 1
        self._object_candidate_tracks[camera_id] = tracks[-12:]
        if int(match["count"]) < self.config.object_learning.stable_candidate_frames:
            return False
        if now - float(match["last_queued"]) < self.config.object_learning.recall_cache_seconds:
            return False
        match["last_queued"] = now
        return True

    def _apply_object_recalls(self, camera_id: str, detections: tuple[Detection, ...]) -> tuple[Detection, ...]:
        with self._object_recall_lock:
            recalls = [item for item in self._object_recalls.get(camera_id, []) if item["expires_at"] > time.monotonic()]
            self._object_recalls[camera_id] = recalls
        resolved: list[Detection] = []
        for detection in detections:
            match = max(recalls, key=lambda item: self._bbox_iou(detection.bbox, item["bbox"]), default=None)
            overlap = self._bbox_iou(detection.bbox, match["bbox"]) if match else 0.0
            if match and overlap >= 0.30 and detection.label != "person":
                resolved.append(
                    replace(
                        detection,
                        label=str(match["label"]),
                        attributes={
                            **detection.attributes,
                            "base_label": detection.label,
                            "object_id": match["profile_id"],
                            "object_recall_confidence": match["similarity"],
                            "object_label_source": match["label_source"],
                            "object_evidence_count": match["evidence_count"],
                            "object_label_provenance": match["provenance"],
                            "object_confidence_components": match["confidence_components"],
                        },
                    )
                )
            else:
                resolved.append(detection)
        return tuple(resolved)

    @staticmethod
    def _bbox_iou(left, right) -> float:
        intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
        intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
        intersection = intersection_width * intersection_height
        union = left.area + right.area - intersection
        return intersection / union if union > 0 else 0.0

    async def _reason_about_transcript(self) -> None:
        while True:
            transcript = await self._utterances.get()
            try:
                pending = self.telemetry.pending_observation()
                live_context = self._scene_context()
                try:
                    language = await self._omnius.reason_about_utterance(transcript, live_context)
                except Exception as error:
                    logger.warning("dialogue routing model unavailable; using sensor evidence: %s", error)
                    language = None
                dialogue = self._dialogue.classify(
                    DialogueEvidence(
                        transcript,
                        doa_aligned=(
                            self._latest_observation is not None
                            and self._latest_observation.microphone_direction is not None
                        ),
                        seconds_since_tts=(
                            time.monotonic() - self._last_spoken_at
                            if self._last_spoken_at is not None
                            else None
                        ),
                        interaction_pending=pending is not None,
                        language_directed=(
                            bool(language["directed"]) if language is not None else None
                        ),
                    )
                )
                if not dialogue.directed:
                    self.telemetry.record_interaction(
                        False, dialogue.reason, transcript, "[[SILENT]]"
                    )
                    continue
                unnamed_identity = self._visible_unnamed_identity()
                if unnamed_identity and dialogue.act == "person_naming":
                    person_name = await self._omnius.interpret_person_naming(transcript)
                    if person_name:
                        profile = await asyncio.to_thread(self.identities.name_most_recent, person_name)
                        if profile:
                            await self._sync_identity_profile(profile.profile_id)
                            self._queue_identity_name_memory(profile.profile_id, person_name, transcript)
                if dialogue.act == "object_naming":
                    try:
                        object_label = await asyncio.wait_for(
                            self._omnius.interpret_object_naming(transcript), timeout=30
                        )
                    except asyncio.TimeoutError:
                        logger.warning("held-object label interpretation timed out")
                        object_label = None
                    if object_label:
                        learned = await self._learn_held_object(object_label)
                        if learned:
                            logger.info("user-labelled segmented object as %s", learned)
                if pending:
                    feedback = await self._omnius.interpret_correction(transcript, pending)
                    if feedback and feedback["decision"] in {"confirm", "correct"}:
                        self.telemetry.resolve_observation_correction(feedback["decision"], feedback["label"] or None)
                        if feedback["decision"] == "correct" and feedback["label"] and pending.get("object_id"):
                            profile = await asyncio.to_thread(
                                self.objects.relabel, str(pending["object_id"]), feedback["label"], 1.0,
                                "user", "human-feedback",
                                {
                                    "utterance": transcript,
                                    "previous_label": pending.get("label"),
                                    "corrected_at": datetime.now(timezone.utc).isoformat(),
                                },
                            )
                            if profile:
                                await self._sync_object_profile(profile.profile_id)
                                self._queue_user_correction_memory(
                                    profile.profile_id, str(pending.get("label")), profile.label, transcript
                                )
                        await self._speak(feedback["reply"])
                        continue
                reply = await self._omnius.conversation_reply(
                    transcript, await self._cognitive_context(transcript)
                )
                decision = self._interaction_policy.evaluate(
                    transcript, reply, directed=dialogue.directed
                )
                self.telemetry.record_interaction(decision.allow_speech, decision.reason, transcript, reply)
                self._queue_interaction_memory(transcript, reply, decision.allow_speech, decision.reason)
                if decision.allow_speech:
                    await self._speak(reply)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.exception("Omnius reasoning failed; capture remains active")
                self.telemetry.record_runtime_error("reasoning", error)

    async def _learn_held_object(self, label: str) -> str | None:
        vision = self._vision
        if not self.config.object_learning.enabled or self._latest_frame is None or vision is None:
            return None
        frame = self._latest_frame.copy()
        segmented = await asyncio.to_thread(vision.segment_held_object, frame)
        if segmented is None:
            logger.info("no valid handheld object segment available for label %r", label)
            return None
        profile = await asyncio.to_thread(self.objects.learn, label, segmented, vision)
        if profile:
            await self._sync_object_profile(profile.profile_id)
        return profile.label if profile else None

    async def _sync_object_profile(self, profile_id: str) -> None:
        if self._memory is None:
            return
        profile = await asyncio.to_thread(self.objects.profile_record, profile_id)
        if profile:
            await asyncio.to_thread(self._memory.sync_object_profile, profile)

    async def _sync_identity_profile(self, profile_id: str) -> None:
        if self._memory is None:
            return
        profile = await asyncio.to_thread(self.identities.profile_record, profile_id)
        if profile:
            await asyncio.to_thread(self._memory.sync_identity_profile, profile)

    def _visible_unnamed_identity(self) -> str | None:
        latest = self._latest_observation
        if latest is None:
            return None
        for detection in latest.detections:
            if (
                detection.attributes.get("identity_needs_name")
                and detection.attributes.get("identity_kind") == "face"
                and detection.attributes.get("identity_id")
            ):
                return str(detection.attributes["identity_id"])
        return None

    @staticmethod
    def _may_name_person(transcript: str) -> bool:
        normalized_text = transcript.casefold().replace("’", "'")
        normalized = f" {' '.join(normalized_text.split())} "
        return any(phrase in normalized for phrase in (" my name is ", " i'm ", " i am ", " call me "))

    def _queue_identity_name_memory(self, profile_id: str, name: str, transcript: str) -> None:
        if self._memory is None:
            return
        now = datetime.now(timezone.utc)
        evidence = EvidenceRef(
            str(uuid4()), "speech", now, "user-correction", "respeaker-asr", quality=1.0,
            metadata={"transcript": transcript, "identity_id": profile_id, "preferred_name": name},
        )
        self._queue_memory_event(
            PerceptualEvent(
                str(uuid4()), "user_correction", now, "respeaker-asr", (evidence,), (profile_id,),
                payload={"labels": [name], "entities": [
                    {"id": profile_id, "type": "person", "label": name, "confidence": 1.0, "source": "user"}
                ]},
            )
        )

    def _queue_user_correction_memory(
        self, profile_id: str, previous_label: str, corrected_label: str, transcript: str
    ) -> None:
        if self._memory is None:
            return
        now = datetime.now(timezone.utc)
        evidence = EvidenceRef(
            str(uuid4()), "speech", now, "user-correction", "respeaker-asr", quality=1.0,
            metadata={
                "transcript": transcript,
                "object_id": profile_id,
                "previous_label": previous_label,
                "corrected_label": corrected_label,
            },
        )
        self._queue_memory_event(
            PerceptualEvent(
                str(uuid4()), "user_correction", now, "respeaker-asr", (evidence,), (profile_id,),
                payload={"labels": [corrected_label], "entities": [
                    {
                        "id": profile_id, "type": "object", "label": corrected_label,
                        "confidence": 1.0, "source": "user",
                    }
                ]},
            )
        )

    @staticmethod
    def _may_label_held_object(transcript: str) -> bool:
        words = {word.strip(".,!?;:\"'").casefold() for word in transcript.split()}
        return bool(words & {"this", "that", "object", "holding", "called", "name"})

    async def _handle_target(self, target: AttentionTarget, observation: Observation) -> None:
        decision = self._cognitive_attention.evaluate(target, observation)
        self.telemetry.record_attention(target.track_id, target.detection.label, decision)
        self._queue_attention_memory(target, decision)
        event = {
            "type": "attention.target",
            "target": {
                "track_id": target.track_id,
                "label": target.detection.label,
                "behavior": target.detection.attributes.get("behavior"),
                "novelty": target.novelty,
                "priority": target.priority,
                "reason": target.reason,
                "camera_id": target.camera_id,
                "timestamp": target.timestamp.isoformat(),
            },
            "scene_labels": list(observation.semantic_labels),
            "microphone_direction": observation.microphone_direction,
            "cognitive_decision": asdict(decision),
        }
        if target.reason == "continuing":
            logger.debug("attention: %s", event["target"])
        else:
            logger.info("attention: %s", event["target"])
        if self._system_service:
            await self._system_service.publish_event(event)
        if not decision.allow_outward_speech or target.detection.label != "person":
            return
        try:
            scene = self._describe_scene(target, observation)
            reply = await self._omnius.companion_reply(scene)
            await self._speak(reply)
            self._last_greeting = target.timestamp
        except Exception as error:
            logger.exception("proactive Omnius reply failed")
            self.telemetry.record_runtime_error("proactive-reasoning", error)

    def _queue_attention_memory(self, target: AttentionTarget, decision) -> None:
        if self._memory is None:
            return
        evidence = EvidenceRef(
            str(uuid4()), "attention", target.timestamp, "cognitive-attention", target.camera_id,
            quality=decision.capture_priority,
            metadata={
                "target_id": target.track_id,
                "label": target.detection.label,
                "decision": asdict(decision),
            },
        )
        self._queue_memory_event(
            PerceptualEvent(
                str(uuid4()), "attention", target.timestamp, target.camera_id, (evidence,),
                tuple(
                    str(value) for value in (
                        target.detection.attributes.get("identity_id"),
                        target.detection.attributes.get("object_id"),
                    ) if value
                ),
                payload={"labels": [target.detection.label], "attention_reason": decision.reason},
            )
        )

    def _queue_interaction_memory(
        self, transcript: str, response: str, allowed: bool, reason: str
    ) -> None:
        if self._memory is None:
            return
        now = datetime.now(timezone.utc)
        evidence = EvidenceRef(
            str(uuid4()), "action", now, "interaction-policy", "speech-output",
            quality=1.0 if allowed else 0.8,
            metadata={
                "input_transcript": transcript,
                "candidate_response": response,
                "spoken": allowed,
                "reason": reason,
            },
        )
        self._queue_memory_event(
            PerceptualEvent(
                str(uuid4()), "attention", now, "interaction-policy", (evidence,),
                payload={"labels": ["spoken" if allowed else "suppressed"], "attention_reason": reason},
            )
        )

    async def _ask_observation_correction(self, candidate: dict[str, object]) -> None:
        if not self._cognitive_attention.allow_uncertainty_question(datetime.now(timezone.utc)):
            self.telemetry.dismiss_pending_observation()
            self.telemetry.record_interaction(
                False, "hourly uncertainty-question budget exhausted", "", str(candidate)
            )
            return
        try:
            reply = await self._omnius.observation_question(candidate, self._scene_context())
            if reply != "[[SILENT]]":
                await self._speak(reply)
        except Exception:
            logger.exception("unable to ask for observation correction")

    def _scene_context(self, allow_name_question: bool = False) -> str:
        latest = self._latest_observation
        labels = list(latest.semantic_labels) if latest else []
        inventory = self.telemetry.snapshot(self.config)["seen"]
        objects = ", ".join(f"{item['count']} {item['label']}" for item in inventory) or "no stable objects yet"
        directions = f"; sound direction {latest.microphone_direction:.0f} degrees" if latest and latest.microphone_direction is not None else ""
        visible_people = []
        for detection in (latest.detections if latest else ()):
            identity_id = detection.attributes.get("identity_id")
            if not identity_id:
                continue
            if detection.attributes.get("identity_needs_name"):
                if allow_name_question and str(identity_id) not in self._identity_name_questions:
                    visible_people.append(
                        f"face-confirmed {identity_id} has no user-provided name; a preferred-name question is permitted once"
                    )
                    self._identity_name_questions.add(str(identity_id))
                else:
                    visible_people.append(f"unlabeled {identity_id}; do not repeat the name question")
            else:
                visible_people.append(f"visible identity {detection.attributes.get('identity')}")
        people = f"; people: {', '.join(visible_people)}" if visible_people else ""
        return (
            f"stable scene inventory: {objects}; semantic cues: {', '.join(labels) or 'none'}"
            f"{directions}{people}"
        )

    async def _cognitive_context(self, transcript: str) -> str:
        live_scene = self._scene_context(allow_name_question=True)
        if self._memory is None:
            return live_scene
        latest = self._latest_observation
        entity_ids = tuple(
            str(entity_id)
            for detection in (latest.detections if latest else ())
            for entity_id in (
                detection.attributes.get("identity_id"), detection.attributes.get("object_id")
            )
            if entity_id
        )
        vision = self._vision
        query_embedding = (
            await asyncio.to_thread(vision.embed_text, transcript) if vision is not None else None
        )
        context = await asyncio.to_thread(
            self._memory.context_for, transcript, live_scene, entity_ids, query_embedding
        )
        self.telemetry.record_retrieval(self._memory.retrieval_snapshot())
        return context

    async def _speak(self, text: str) -> bool:
        if not text.strip():
            return False
        try:
            wav_audio = await self._omnius.synthesize(text)
        except Exception as error:
            logger.exception("Omnius synthesis failed")
            self.telemetry.record_runtime_error("tts", error)
            return False
        self._speaking = True
        try:
            await self._speaker.play_wav(wav_audio)
        except Exception as error:
            logger.exception("speaker playback failed")
            self.telemetry.record_runtime_error("speaker", error)
            return False
        finally:
            self._speaking = False
            self._asr_holdoff_until = time.monotonic() + 1.5
        self.telemetry.record_reply(text)
        self._last_spoken_at = time.monotonic()
        return True

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        import cv2

        source = frame
        height, width = frame.shape[:2]
        if width > self.config.vision.dashboard_max_width:
            scale = self.config.vision.dashboard_max_width / width
            source = cv2.resize(frame, (self.config.vision.dashboard_max_width, round(height * scale)), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", source, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            raise RuntimeError("failed to encode calibration preview")
        return encoded.tobytes()

    @staticmethod
    def _rotate_frame(frame: np.ndarray, angle: int) -> np.ndarray:
        import cv2

        return {
            0: frame,
            90: cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE),
            180: cv2.rotate(frame, cv2.ROTATE_180),
            270: cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE),
        }[angle]

    def _should_greet(self, now: datetime) -> bool:
        if self._last_greeting is None:
            return True
        elapsed = (now - self._last_greeting).total_seconds()
        return elapsed >= self.config.attention.greeting_cooldown_seconds

    @staticmethod
    def _describe_scene(target: AttentionTarget, observation: Observation) -> str:
        behavior = target.detection.attributes.get("behavior")
        parts = [f"a person is {behavior}" if behavior else "a person is present"]
        if observation.semantic_labels:
            parts.append("visual context: " + ", ".join(observation.semantic_labels[:3]))
        if observation.microphone_direction is not None:
            parts.append(f"sound direction: {observation.microphone_direction:.0f} degrees")
        return "; ".join(parts)
