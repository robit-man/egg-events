import asyncio
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np

from egg_companion.adapters.vision import SegmentedObject
from egg_companion.cognition.architecture import CognitiveArchitecture
from egg_companion.config import CognitiveAttentionConfig, EggConfig, ObjectLearningConfig
from egg_companion.core.attention import AttentionManager
from egg_companion.core.cognition import CognitiveAttentionController
from egg_companion.models import BoundingBox, Detection
from egg_companion.runtime import CompanionRuntime
from egg_companion.services.object_library import ObjectLibrary
from egg_companion.services.telemetry import RuntimeTelemetry


class FakeVision:
    def embed_image(self, image: np.ndarray) -> np.ndarray:
        mean = float(image.mean())
        vector = np.array([mean + 1.0, 1.0], dtype=np.float32)
        return vector / np.linalg.norm(vector)


def segmented(value: int = 180) -> SegmentedObject:
    image = np.full((48, 48, 3), value, dtype=np.uint8)
    mask = np.full((48, 48), 255, dtype=np.uint8)
    return SegmentedObject(image, mask, 0.55)


def test_ornith_correction_and_clip_recall_survive_restart(tmp_path: Path) -> None:
    config = ObjectLearningConfig(storage_dir=str(tmp_path), similarity_threshold=0.8)
    library = ObjectLibrary(config)
    profile = library.learn("wrong base label", segmented(), FakeVision(), "detector", 0.55)
    assert profile is not None
    corrected = library.relabel(profile.profile_id, "ceramic mug", 0.93, "ornith-vlm", "robit/ornith-vision:9b")
    assert corrected is not None
    assert corrected.label_history[0]["label"] == "wrong base label"

    reopened = ObjectLibrary(config)
    recalled = reopened.match(segmented(), FakeVision())
    assert recalled is not None
    assert recalled[0].label == "ceramic mug"
    assert recalled[0].label_source == "ornith-vlm"
    assert recalled[1] >= config.similarity_threshold
    assert reopened.snapshot()[0]["label_history"][0]["label"] == "wrong base label"


def test_clip_recall_bypasses_ornith_vlm() -> None:
    async def scenario() -> None:
        config = EggConfig.model_validate(
            {
                "audio": {"input_device": "default"},
                "omnius": {"model": "test", "voice_model": "test"},
            }
        )
        profile = SimpleNamespace(
            profile_id="object-001",
            label="ceramic mug",
            label_source="ornith-vlm",
            samples=3,
            label_provenance={"model_id": "ornith-test"},
        )

        class RecallOnlyLibrary:
            @staticmethod
            def match(candidate, vision):
                return profile, 0.96

        class ForbiddenVlm:
            calls = 0

            async def classify_masked_object(self, image, label, confidence):
                self.calls += 1
                raise AssertionError("Ornith must not run after local CLIP recall")

        runtime = CompanionRuntime.__new__(CompanionRuntime)
        runtime.config = config
        runtime.objects = RecallOnlyLibrary()
        runtime._vision = object()
        runtime._omnius = ForbiddenVlm()
        runtime.telemetry = RuntimeTelemetry(config)
        runtime._brain = CognitiveArchitecture(
            AttentionManager(track_ttl_seconds=10, min_priority=0.1),
            CognitiveAttentionController(CognitiveAttentionConfig(), proactive_enabled=False),
            memory=None,
        )
        runtime._object_candidates = asyncio.Queue(maxsize=1)
        runtime._object_recall_lock = threading.Lock()
        runtime._object_recalls = {}
        runtime._last_vlm_at = 0.0
        runtime._last_valid_speech_at = 0.0
        detection = Detection("cup", 0.55, BoundingBox(0, 0, 40, 40))
        await runtime._object_candidates.put(("camera-0", detection, segmented(), "fingerprint", 0))

        task = asyncio.create_task(runtime._auto_label_objects())
        try:
            async def recalled() -> None:
                while runtime.telemetry.snapshot(config)["object_learning"]["clip_recalls"] < 1:
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(recalled(), timeout=1)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        assert runtime._omnius.calls == 0
        assert runtime._object_recalls["camera-0"][0]["profile_id"] == "object-001"

    asyncio.run(scenario())


def test_profiles_due_for_review_includes_stale_verified_profiles(tmp_path: Path) -> None:
    config = ObjectLearningConfig(storage_dir=str(tmp_path), similarity_threshold=0.8)
    library = ObjectLibrary(config)
    profile = library.learn("mug", segmented(), FakeVision(), "detector", 0.55)
    assert profile is not None

    due_when_pending = library.profiles_due_for_review(stale_after_seconds=999_999)
    assert any(item[0] == profile.profile_id for item in due_when_pending)

    library.relabel(profile.profile_id, "ceramic mug", 0.93, "ornith-vlm", "robit/ornith-vision:9b")

    due_when_fresh = library.profiles_due_for_review(stale_after_seconds=999_999)
    assert not any(item[0] == profile.profile_id for item in due_when_fresh)

    due_when_stale = library.profiles_due_for_review(stale_after_seconds=0)
    assert any(item[0] == profile.profile_id for item in due_when_stale)


class _SweepLibrary:
    def __init__(self, due: list[tuple[str, str, float]]) -> None:
        self._due = due
        self.audited: list[tuple[str, str, str | None]] = []
        self.relabeled: list[tuple[str, str, float]] = []
        self.failed: list[str] = []

    def profiles_due_for_review(self, stale_after_seconds: float):
        return self._due

    def segmented_profile(self, profile_id: str) -> SegmentedObject:
        return segmented()

    def profile_record(self, profile_id: str) -> dict[str, object]:
        return {
            "label": "mug",
            "label_confidence": 0.5,
            "samples": 2,
            "review_state": "pending",
            "label_history": [],
        }

    def mark_audited(self, profile_id: str, audit_state: str, notes: str | None) -> None:
        self.audited.append((profile_id, audit_state, notes))

    def relabel(self, profile_id, label, confidence, source, model_id, provenance=None):
        self.relabeled.append((profile_id, label, confidence))
        return SimpleNamespace(profile_id=profile_id, label=label)

    def mark_review_failed(self, profile_id: str) -> None:
        self.failed.append(profile_id)


class _FakeSweepVision:
    def encode_segmented_object(self, segmented_object, max_size):
        return b"fake-png"


def _sweep_runtime(config: EggConfig, library: _SweepLibrary, omnius) -> CompanionRuntime:
    runtime = CompanionRuntime.__new__(CompanionRuntime)
    runtime.config = config
    runtime.objects = library
    runtime._vision = _FakeSweepVision()
    runtime._omnius = omnius
    runtime._memory = None
    runtime.telemetry = RuntimeTelemetry(config)
    return runtime


def test_sweep_skips_vlm_when_audit_reports_consistent() -> None:
    async def scenario() -> None:
        config = EggConfig.model_validate(
            {
                "audio": {"input_device": "default"},
                "omnius": {"model": "test", "voice_model": "test"},
            }
        )

        class ConsistentAuditOmnius:
            async def audit_object_label(self, profile):
                return {"consistent": True, "confidence": 0.9, "reason": "history is stable"}

            async def classify_masked_object(self, image, label, confidence):
                raise AssertionError("VLM must not run when the audit reports consistent")

        library = _SweepLibrary(due=[("object-001", "mug", 0.5)])
        runtime = _sweep_runtime(config, library, ConsistentAuditOmnius())

        await runtime._sweep_object_reviews()

        assert library.audited == [("object-001", "consistent", "history is stable")]
        assert library.relabeled == []
        snapshot = runtime.telemetry.snapshot(config)["object_learning"]
        assert snapshot["audit_consistent"] == 1
        assert snapshot["audit_flagged"] == 0
        assert snapshot["review_queue_depth"] == 1

    asyncio.run(scenario())


def test_sweep_falls_back_to_vlm_when_audit_flags_or_fails() -> None:
    async def scenario() -> None:
        config = EggConfig.model_validate(
            {
                "audio": {"input_device": "default"},
                "omnius": {"model": "test", "voice_model": "test"},
            }
        )

        class FlaggingAuditOmnius:
            async def audit_object_label(self, profile):
                return {"consistent": False, "confidence": 0.3, "reason": "label history flip-flops"}

            async def classify_masked_object(self, image, label, confidence):
                return "ceramic mug", 0.95

            async def ocr_advanced(self, image_path):
                return None

        library = _SweepLibrary(due=[("object-001", "mug", 0.5)])
        runtime = _sweep_runtime(config, library, FlaggingAuditOmnius())

        await runtime._sweep_object_reviews()

        assert library.audited == []
        assert library.relabeled == [("object-001", "ceramic mug", 0.95)]
        snapshot = runtime.telemetry.snapshot(config)["object_learning"]
        assert snapshot["audit_flagged"] == 1
        assert snapshot["audit_consistent"] == 0

    asyncio.run(scenario())


def test_sweep_falls_back_to_vlm_when_audit_call_raises() -> None:
    async def scenario() -> None:
        config = EggConfig.model_validate(
            {
                "audio": {"input_device": "default"},
                "omnius": {"model": "test", "voice_model": "test"},
            }
        )

        class RaisingAuditOmnius:
            async def audit_object_label(self, profile):
                raise RuntimeError("cognition model unavailable")

            async def classify_masked_object(self, image, label, confidence):
                return "ceramic mug", 0.95

            async def ocr_advanced(self, image_path):
                return None

        library = _SweepLibrary(due=[("object-001", "mug", 0.5)])
        runtime = _sweep_runtime(config, library, RaisingAuditOmnius())

        await runtime._sweep_object_reviews()

        assert library.relabeled == [("object-001", "ceramic mug", 0.95)]

    asyncio.run(scenario())
