import asyncio
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np

from egg_companion.adapters.vision import SegmentedObject
from egg_companion.cognition.architecture import CognitiveArchitecture
from egg_companion.config import CognitiveAttentionConfig, EggConfig, ObjectLearningConfig
from egg_companion.core.activity import ActivityGovernor
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
    summary = reopened.summary_snapshot()[0]
    assert summary["appearance_description"] == ""
    assert summary["label_history_count"] == 1
    assert "adjudication_history" not in summary


def test_label_trust_ranks_user_over_vla_over_unverified() -> None:
    assert ObjectLibrary.label_trust("user") > ObjectLibrary.label_trust("ornith-vlm")
    assert ObjectLibrary.label_trust("ornith-vlm") > ObjectLibrary.label_trust("detector")
    assert ObjectLibrary.label_trust(None) == ObjectLibrary.label_trust("detector") == 0


def test_duplicate_merge_keeps_the_vla_confirmed_label_even_with_fewer_samples(
    tmp_path: Path,
) -> None:
    """Reproduces the historical bug: an object seen many times under a wrong
    generic detector label must not out-vote a freshly VLA-confirmed profile
    of the same physical item just because it has more accumulated samples --
    future mask reads should show the corrected label, not the stale one."""
    config = ObjectLearningConfig(storage_dir=str(tmp_path), similarity_threshold=0.8)
    library = ObjectLibrary(config)
    stale = library.learn("object", segmented(150), FakeVision(), "detector", 0.4)
    for _ in range(9):
        library.learn("object", segmented(150), FakeVision(), "detector", 0.4)
    assert stale is not None
    stale_record = library.profile_record(stale.profile_id)
    assert stale_record["samples"] == 10

    corrected = library.learn(
        "ceramic mug", segmented(151), FakeVision(), "ornith-vlm", 0.9, force_new=True
    )
    assert corrected is not None
    corrected_record = library.profile_record(corrected.profile_id)
    assert corrected_record["samples"] == 1

    stale_rank = (
        ObjectLibrary.label_trust(stale_record["label_source"]),
        stale_record["samples"],
    )
    corrected_rank = (
        ObjectLibrary.label_trust(corrected_record["label_source"]),
        corrected_record["samples"],
    )
    canonical_id, alias_id = (
        (stale.profile_id, corrected.profile_id)
        if stale_rank >= corrected_rank
        else (corrected.profile_id, stale.profile_id)
    )
    assert canonical_id == corrected.profile_id
    assert alias_id == stale.profile_id

    merged = library.merge_profiles(canonical_id, alias_id, 0.95, {"same_instance": True})
    assert merged is not None
    assert merged.label == "ceramic mug"
    assert merged.samples == 11

    recalled = library.match(segmented(151), FakeVision())
    assert recalled is not None
    assert recalled[0].label == "ceramic mug"


def test_clip_recall_is_only_a_proposal_until_ornith_confirms_pixels() -> None:
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
            appearance_description="a ceramic mug with a dark handle",
        )

        class RecallOnlyLibrary:
            confirmed = 0

            @staticmethod
            def match(candidate, vision):
                return profile, 0.96

            @staticmethod
            def thumbnail(profile_id):
                return b"prior-png"

            @staticmethod
            def profile_record(profile_id):
                return {
                    "profile_id": profile_id,
                    "label": "ceramic mug",
                    "appearance_description": "a ceramic mug with a dark handle",
                    "samples": 3,
                    "review_state": "vlm_verified",
                }

            @classmethod
            def confirm_match(cls, profile_id, label, candidate, vision, confidence, **kwargs):
                cls.confirmed += 1
                return profile

        class ConfirmingVlm:
            calls = 0

            async def compare_masked_object_candidate(self, *args):
                self.calls += 1
                return {
                    "object_present": True,
                    "same_instance": True,
                    "confidence": 0.95,
                    "label": "ceramic mug",
                    "appearance_description": "a ceramic mug with a dark handle",
                    "detector_supported": True,
                    "analysis": "The distinctive handle and body agree.",
                    "visible_correspondences": ["dark handle"],
                    "visible_conflicts": [],
                    "visible_text": False,
                    "text_regions": [],
                }

        runtime = CompanionRuntime.__new__(CompanionRuntime)
        runtime.config = config
        runtime.objects = RecallOnlyLibrary()
        runtime._vision = object()
        runtime._omnius = ConfirmingVlm()
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
        runtime._memory = None
        runtime._background_visual_tasks = set()
        runtime._sync_object_profile = lambda profile_id: asyncio.sleep(0)  # type: ignore[method-assign]
        runtime._queue_object_adjudication_memory = lambda *args: None  # type: ignore[method-assign]
        runtime._run_advanced_ocr = lambda image: asyncio.sleep(0, result=None)  # type: ignore[method-assign]
        runtime._vision = SimpleNamespace(
            encode_segmented_object=lambda candidate, max_size: b"current-png"
        )
        detection = Detection("cup", 0.55, BoundingBox(0, 0, 40, 40))
        await runtime._object_candidates.put(("camera-0", detection, segmented(), "fingerprint", 0))

        task = asyncio.create_task(runtime._auto_label_objects())
        try:
            async def recalled() -> None:
                while RecallOnlyLibrary.confirmed < 1:
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(recalled(), timeout=1)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        assert runtime._omnius.calls == 1
        assert RecallOnlyLibrary.confirmed == 1
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

    def relabel(self, profile_id, label, confidence, source, model_id, provenance=None, **kwargs):
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
    runtime._activity = ActivityGovernor(config.activity)
    runtime.telemetry = RuntimeTelemetry(config)
    return runtime


def test_sweep_never_lets_text_only_audit_skip_pixel_review() -> None:
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
                return "ceramic mug", 0.95

        library = _SweepLibrary(due=[("object-001", "mug", 0.5)])
        runtime = _sweep_runtime(config, library, ConsistentAuditOmnius())

        await runtime._sweep_object_reviews()

        assert library.audited == []
        assert library.relabeled == [("object-001", "ceramic mug", 0.95)]
        snapshot = runtime.telemetry.snapshot(config)["object_learning"]
        assert snapshot["audit_consistent"] == 0
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
        assert snapshot["audit_flagged"] == 0
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
