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

    def test_discovered_cameras_inherit_capture_resolution_from_discovery_config(self) -> None:
        config = EggConfig.model_validate(
            {
                "cameras": [],
                "camera_discovery": {"capture_width": 3840, "capture_height": 2160},
                "audio": {"input_device": "default"},
                "omnius": {"model": "test", "voice_model": "test"},
            }
        )
        with patch("egg_companion.config.glob", return_value=["/dev/video0"]), patch(
            "egg_companion.config.Path.exists", return_value=True
        ), patch("egg_companion.config.Path.is_char_device", return_value=True):
            cameras = _discover_cameras(config)
        self.assertEqual(cameras[0].capture_width, 3840)
        self.assertEqual(cameras[0].capture_height, 2160)

    def test_discovered_cameras_default_to_no_capture_resolution_override(self) -> None:
        config = EggConfig.model_validate(
            {
                "cameras": [],
                "audio": {"input_device": "default"},
                "omnius": {"model": "test", "voice_model": "test"},
            }
        )
        with patch("egg_companion.config.glob", return_value=["/dev/video0"]), patch(
            "egg_companion.config.Path.exists", return_value=True
        ), patch("egg_companion.config.Path.is_char_device", return_value=True):
            cameras = _discover_cameras(config)
        self.assertIsNone(cameras[0].capture_width)
        self.assertIsNone(cameras[0].capture_height)
