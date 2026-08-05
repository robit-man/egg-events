from __future__ import annotations

import tempfile
import unittest

import numpy as np

from egg_companion.adapters.vision import FaceCrop
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


class IdentityLibraryTests(unittest.TestCase):
    def test_face_profile_is_recalled_after_database_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = IdentityConfig(storage_dir=directory, face_similarity_threshold=0.8, sample_interval_seconds=3600)
            face_embedding = np.array((0.1, 0.2, 0.3), dtype=np.float32)
            vision = _Vision(face_embedding)
            frame = np.zeros((96, 96, 3), dtype=np.uint8)
            detection = Detection("person", 0.95, BoundingBox(0, 0, 96, 96))

            first = IdentityLibrary(config).observe("front", frame, (detection,), vision)[0]
            recalled = IdentityLibrary(config).observe("side", frame, (detection,), vision)[0]

            self.assertEqual(first["id"], recalled["id"])
            self.assertTrue(recalled["recalled"])
            self.assertEqual(recalled["sightings"], 2)
            self.assertEqual(recalled["resolver_outcome"], "recalled")
            self.assertGreaterEqual(recalled["confidence_components"]["face_similarity"], 0.99)
