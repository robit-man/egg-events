import asyncio
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np

from egg_companion.adapters.vision import SegmentedObject
from egg_companion.config import EggConfig, ObjectLearningConfig
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
            def profiles_for_review():
                return []

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
