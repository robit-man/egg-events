from __future__ import annotations

import tempfile
import unittest

import numpy as np

from egg_companion.adapters.vision import FaceCrop, VisionEngine
from egg_companion.config import IdentityConfig
from egg_companion.models import BoundingBox, Detection
from egg_companion.services.identity import IdentityLibrary


class _Vision:
    def __init__(self, face_embedding: np.ndarray) -> None:
        self._face_embedding = face_embedding

    def face_crops(self, frame: np.ndarray, detection: Detection) -> tuple[FaceCrop, ...]:
        return (FaceCrop(frame, 0.95, "face", self._face_embedding),)

    @staticmethod
    def embed_image(frame: np.ndarray) -> np.ndarray:
        return np.array((0.25, 0.5, 0.75), dtype=np.float32)


class _AppearanceVision:
    @staticmethod
    def face_crops(frame: np.ndarray, detection: Detection) -> tuple[FaceCrop, ...]:
        return (FaceCrop(frame, 0.92, "appearance", None),)

    @staticmethod
    def embed_image(frame: np.ndarray) -> np.ndarray:
        raise AssertionError("CLIP must not be used to create a person identity")

    def segment_detection(
        self, frame: np.ndarray, detection: Detection
    ):
        return VisionEngine.segment_detection(self, frame, detection)  # type: ignore[arg-type]

    encode_segmented_object = staticmethod(VisionEngine.encode_segmented_object)


class IdentityLibraryTests(unittest.TestCase):
    @staticmethod
    def _enroll(
        library: IdentityLibrary, camera_id: str, embedding: np.ndarray
    ) -> dict[str, object]:
        vision = _Vision(embedding)
        frame = np.zeros((96, 96, 3), dtype=np.uint8)
        detection = Detection("person", 0.95, BoundingBox(0, 0, 96, 96))
        result: dict[str, object] = {}
        for _ in range(3):
            result = library.observe(camera_id, frame, (detection,), vision)[0]
        return result

    def test_repeated_appearance_frames_remain_one_transient_track(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library = IdentityLibrary(IdentityConfig(storage_dir=directory))
            vision = _AppearanceVision()
            frame = np.zeros((96, 96, 3), dtype=np.uint8)
            first_detection = Detection("person", 0.95, BoundingBox(0, 0, 80, 96))
            moved_detection = Detection("person", 0.95, BoundingBox(3, 0, 83, 96))

            first = library.observe("front", frame, (first_detection,), vision)[0]
            repeated = library.observe("front", frame, (moved_detection,), vision)[0]

            self.assertEqual(first["id"], repeated["id"])
            self.assertTrue(first["id"].startswith("track-"))
            self.assertFalse(first["persistent"])
            self.assertEqual(library.migration_profiles(), [])
            self.assertEqual(library.summary()["legacy_appearance_fragments"], 0)

    def test_overlapping_instance_masks_bridge_dislocated_person_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library = IdentityLibrary(
                IdentityConfig(
                    storage_dir=directory,
                    temporal_vlm_cooldown_seconds=0,
                )
            )
            vision = _AppearanceVision()
            frame = np.full((100, 100, 3), 96, dtype=np.uint8)
            first_detection = Detection(
                "person",
                0.95,
                BoundingBox(0, 0, 30, 95),
                {
                    "mask_polygon": [[10, 4], [70, 4], [70, 94], [10, 94]],
                    "frame_shape": [100, 100, 3],
                },
            )
            dislocated_detection = Detection(
                "person",
                0.95,
                BoundingBox(60, 0, 90, 95),
                {
                    "mask_polygon": [[15, 4], [75, 4], [75, 94], [15, 94]],
                    "frame_shape": [100, 100, 3],
                },
            )

            first = library.observe("front", frame, (first_detection,), vision)[0]
            continued = library.observe(
                "front", frame, (dislocated_detection,), vision
            )[0]

            self.assertEqual(continued["id"], first["id"])
            self.assertEqual(continued["temporal_association"]["basis"], "mask_overlap")
            self.assertEqual(continued["temporal_association"]["bbox_iou"], 0)
            self.assertGreater(continued["temporal_association"]["mask_iou"], 0.8)
            comparison = continued["_temporal_comparison"]
            self.assertEqual(comparison["entity_id"], first["id"])
            self.assertEqual(comparison["prior_entity_id"], first["id"])
            self.assertEqual(
                comparison["geometry"]["centroid_dx_pixels"], 60.0
            )
            self.assertFalse(
                comparison["geometry"]["simultaneous_distinct_mask_conflict"]
            )

    def test_simultaneous_person_masks_never_reuse_one_track_twice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library = IdentityLibrary(IdentityConfig(storage_dir=directory))
            vision = _AppearanceVision()
            frame = np.full((100, 100, 3), 96, dtype=np.uint8)

            def person(left: int, right: int) -> Detection:
                return Detection(
                    "person",
                    0.95,
                    BoundingBox(left, 0, right, 95),
                    {
                        "mask_polygon": [
                            [left, 4], [right, 4], [right, 94], [left, 94]
                        ],
                        "frame_shape": [100, 100, 3],
                    },
                )

            prior = library.observe("front", frame, (person(5, 85),), vision)[0]
            current = library.observe(
                "front", frame, (person(8, 82), person(12, 88)), vision
            )

            self.assertEqual(current[0]["id"], prior["id"])
            self.assertNotEqual(current[1]["id"], prior["id"])
            self.assertEqual(
                current[1]["temporal_association"]["basis"], "new_track"
            )

    def test_face_profile_is_recalled_after_database_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = IdentityConfig(storage_dir=directory, face_similarity_threshold=0.8, sample_interval_seconds=3600)
            face_embedding = np.array((0.1, 0.2, 0.3), dtype=np.float32)
            vision = _Vision(face_embedding)
            frame = np.zeros((96, 96, 3), dtype=np.uint8)
            detection = Detection("person", 0.95, BoundingBox(0, 0, 96, 96))

            library = IdentityLibrary(config)
            first_candidate = library.observe("front", frame, (detection,), vision)[0]
            second_candidate = library.observe("front", frame, (detection,), vision)[0]
            first = library.observe("front", frame, (detection,), vision)[0]
            recalled = IdentityLibrary(config).observe("side", frame, (detection,), vision)[0]

            self.assertFalse(first_candidate["persistent"])
            self.assertFalse(second_candidate["persistent"])
            self.assertEqual(first_candidate["id"], second_candidate["id"])
            self.assertEqual(first["resolver_outcome"], "new")
            self.assertEqual(first["id"], recalled["id"])
            self.assertTrue(recalled["recalled"])
            self.assertEqual(recalled["sightings"], 2)
            self.assertEqual(recalled["resolver_outcome"], "recalled")
            self.assertGreaterEqual(recalled["confidence_components"]["face_similarity"], 0.99)

    def test_one_face_frame_never_creates_a_durable_person(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library = IdentityLibrary(IdentityConfig(storage_dir=directory))
            vision = _Vision(np.array((0.1, 0.2, 0.3), dtype=np.float32))
            frame = np.zeros((96, 96, 3), dtype=np.uint8)
            detection = Detection("person", 0.95, BoundingBox(0, 0, 96, 96))

            result = library.observe("front", frame, (detection,), vision)[0]

            self.assertFalse(result["persistent"])
            self.assertEqual(result["enrollment_observations"], 1)
            self.assertEqual(library.migration_profiles(), [])

    def test_retroactive_aliases_stack_profiles_and_recall_the_canonical_person(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = IdentityConfig(
                storage_dir=directory,
                face_similarity_threshold=0.90,
                retroactive_merge_similarity=0.80,
                sample_interval_seconds=3600,
            )
            first_vector = np.array((1.0, 0.0, 0.0), dtype=np.float32)
            second_vector = np.array((0.82, 0.57236, 0.0), dtype=np.float32)
            library = IdentityLibrary(config)
            first = self._enroll(library, "front", first_vector)
            second = self._enroll(library, "side", second_vector)

            aliases = library.coalesce_profiles()
            recalled = library.observe(
                "rear",
                np.zeros((96, 96, 3), dtype=np.uint8),
                (Detection("person", 0.95, BoundingBox(0, 0, 96, 96)),),
                _Vision(second_vector),
            )[0]

            self.assertNotEqual(first["id"], second["id"])
            self.assertEqual(len(aliases), 1)
            self.assertEqual(aliases[0]["canonical_id"], first["id"])
            self.assertEqual(aliases[0]["alias_id"], second["id"])
            self.assertEqual(recalled["id"], first["id"])
            self.assertEqual(recalled["resolver_outcome"], "coalesced_recall")
            rows = library.snapshot()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["stack_count"], 2)
            self.assertEqual(library.summary()["canonical_people"], 1)
            self.assertEqual(library.summary()["coalesced_aliases"], 1)

            reopened = IdentityLibrary(config)
            self.assertEqual(reopened.alias_mappings(), aliases)

    def test_retroactive_alias_never_crosses_coobservation_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = IdentityConfig(
                storage_dir=directory,
                face_similarity_threshold=0.90,
                retroactive_merge_similarity=0.80,
            )
            library = IdentityLibrary(config)
            first = self._enroll(library, "front", np.array((1.0, 0.0, 0.0), dtype=np.float32))
            second = self._enroll(
                library, "side", np.array((0.82, 0.57236, 0.0), dtype=np.float32)
            )

            aliases = library.coalesce_profiles(
                {tuple(sorted((str(first["id"]), str(second["id"]))))}
            )

            self.assertEqual(aliases, [])
            self.assertEqual(library.summary()["canonical_people"], 2)

    def test_rejected_face_samples_are_quarantined_without_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library = IdentityLibrary(IdentityConfig(storage_dir=directory))
            enrolled = self._enroll(
                library, "front", np.array((1.0, 0.0, 0.0), dtype=np.float32)
            )
            sample = library.face_sample_snapshot()[0]

            result = library.apply_face_validation(
                {str(sample["sample_id"]): False}, "test-validator"
            )

            self.assertEqual(result, {"accepted": 0, "rejected": 1})
            self.assertEqual(library.summary()["canonical_people"], 0)
            self.assertEqual(library.summary()["quarantined_face_samples"], 1)
            self.assertEqual(library.face_sample_snapshot(), [])
            self.assertIsNotNone(
                library.face_sample(str(enrolled["id"]), str(sample["sample_id"]))
            )
