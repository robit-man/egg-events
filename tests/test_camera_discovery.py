from __future__ import annotations

import unittest
from unittest.mock import patch

from egg_companion.config import EggConfig, _discover_cameras


class CameraDiscoveryTests(unittest.TestCase):
    def test_discovers_v4l2_nodes_without_a_fixed_inventory(self) -> None:
        config = EggConfig.model_validate(
            {
                "cameras": [],
                "audio": {"input_device": "default"},
                "omnius": {"model": "test", "voice_model": "test"},
            }
        )
        with patch("egg_companion.config.glob", return_value=["/dev/video12", "/dev/video2"]), patch(
            "egg_companion.config.Path.exists", return_value=True
        ), patch("egg_companion.config.Path.is_char_device", return_value=True):
            cameras = _discover_cameras(config)
        self.assertEqual([camera.id for camera in cameras], ["camera-video2", "camera-video12"])
        self.assertTrue(all(camera.rotation_degrees == "auto" for camera in cameras))
