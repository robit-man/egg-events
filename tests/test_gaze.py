import unittest

from egg_companion.core.gaze import classify_gaze


def _make_face_keypoints(
    nose: tuple[float, float, float] = (0.5, 0.4, 0.9),
    left_eye: tuple[float, float, float] = (0.46, 0.35, 0.9),
    right_eye: tuple[float, float, float] = (0.54, 0.35, 0.9),
    left_ear: tuple[float, float, float] = (0.4, 0.36, 0.7),
    right_ear: tuple[float, float, float] = (0.6, 0.36, 0.7),
) -> list[list[float]]:
    points: list[list[float]] = [[0.5, 0.5, 0.5] for _ in range(17)]
    points[0] = list(nose)
    points[1] = list(left_eye)
    points[2] = list(right_eye)
    points[3] = list(left_ear)
    points[4] = list(right_ear)
    return points


class GazeTests(unittest.TestCase):
    def test_nose_centered_between_eyes_is_facing_camera(self) -> None:
        points = _make_face_keypoints()
        result = classify_gaze(points)
        assert result is not None
        self.assertEqual(result["state"], "facing_camera")
        self.assertAlmostEqual(result["yaw_offset"], 0.0, delta=0.05)

    def test_nose_shifted_left_of_eye_midpoint_is_looking_left(self) -> None:
        points = _make_face_keypoints(nose=(0.47, 0.4, 0.9))
        result = classify_gaze(points)
        assert result is not None
        self.assertEqual(result["state"], "looking_left")
        self.assertLess(result["yaw_offset"], 0.0)

    def test_nose_shifted_right_of_eye_midpoint_is_looking_right(self) -> None:
        points = _make_face_keypoints(nose=(0.53, 0.4, 0.9))
        result = classify_gaze(points)
        assert result is not None
        self.assertEqual(result["state"], "looking_right")
        self.assertGreater(result["yaw_offset"], 0.0)

    def test_extreme_offset_is_looking_away(self) -> None:
        points = _make_face_keypoints(nose=(0.7, 0.4, 0.9))
        result = classify_gaze(points)
        assert result is not None
        self.assertEqual(result["state"], "looking_away")

    def test_only_one_eye_visible_is_looking_away(self) -> None:
        points = _make_face_keypoints(right_eye=(0.54, 0.35, 0.05))
        result = classify_gaze(points)
        assert result is not None
        self.assertEqual(result["state"], "looking_away")

    def test_low_confidence_nose_returns_none(self) -> None:
        points = _make_face_keypoints(nose=(0.5, 0.4, 0.1))
        self.assertIsNone(classify_gaze(points))

    def test_neither_eye_visible_returns_none(self) -> None:
        points = _make_face_keypoints(
            left_eye=(0.46, 0.35, 0.05), right_eye=(0.54, 0.35, 0.05),
        )
        self.assertIsNone(classify_gaze(points))

    def test_none_keypoints_returns_none(self) -> None:
        self.assertIsNone(classify_gaze(None))

    def test_insufficient_keypoints_returns_none(self) -> None:
        self.assertIsNone(classify_gaze([[0.5, 0.5, 0.9] for _ in range(3)]))

    def test_confidence_is_min_of_nose_and_eyes(self) -> None:
        points = _make_face_keypoints(nose=(0.5, 0.4, 0.55))
        result = classify_gaze(points)
        assert result is not None
        self.assertEqual(result["confidence"], 0.55)


class TestGazeReachesWorldModel(unittest.TestCase):
    """Regression test for the same bug class already caught twice this
    session (the OCR payload and the vision "detections" key): a field
    computed upstream (vision.py attaching attributes["gaze"] to a
    Detection) that the downstream payload-builder
    (_queue_vision_memory's hand-picked field list) silently drops. gaze
    must reach current_property_state through the real
    _queue_vision_memory code path, not just through a hand-rolled
    payload dict that assumes the wiring is correct."""

    def test_queue_vision_memory_includes_gaze_and_it_reaches_world_state(self) -> None:
        import asyncio
        import tempfile
        from datetime import datetime, timezone

        from egg_companion.config import EggConfig
        from egg_companion.memory.pipeline import MemoryPipeline
        from egg_companion.memory.store import MemoryStore
        from egg_companion.models import BoundingBox, Detection, Observation
        from egg_companion.runtime import CompanionRuntime

        with tempfile.TemporaryDirectory() as tmp:
            config = EggConfig.model_validate({
                "audio": {"input_device": "default", "doa_mode": "disabled"},
                "omnius": {"model": "test", "voice_model": "test"},
                "identity": {"enabled": False},
                "object_learning": {"enabled": False},
                "camera_discovery": {"enabled": False},
                "memory": {"storage_dir": f"{tmp}/memory"},
            })
            store = MemoryStore(config.memory)
            pipeline = MemoryPipeline(config, store)

            runtime = object.__new__(CompanionRuntime)
            runtime._memory = pipeline
            runtime._last_visual_evidence_at = {}
            # Bypass the async persistence queue (_memory_events) that a
            # fully-constructed CompanionRuntime would drain in the
            # background -- persist synchronously like the queue consumer
            # eventually would, so the test can assert on the result.
            runtime._queue_memory_event = pipeline._persist_event

            detection = Detection(
                "person", 0.9, BoundingBox(10, 10, 50, 50),
                {
                    "identity_id": "person-gaze-test",
                    "identity_persistent": True,
                    "identity_kind": "face",
                    "gaze": {
                        "state": "facing_camera", "yaw_offset": 0.01, "confidence": 0.8,
                    },
                },
            )
            observation = Observation(
                "camera-video1", datetime.now(timezone.utc), (detection,)
            )

            asyncio.run(runtime._queue_vision_memory(observation, frame=None))

            value = pipeline._world_query.property_value(
                "person-gaze-test", "gaze_state"
            )
            self.assertEqual(value, "facing_camera")
