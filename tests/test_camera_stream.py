from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from egg_companion.adapters.camera import CameraStream
from egg_companion.config import CameraConfig


class CameraStreamCaptureResolutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_requests_configured_capture_resolution(self) -> None:
        """capture_width/capture_height being set must request that
        resolution from the device -- confirmed against the real rig
        (v4l2-ctl --list-formats-ext) that these sensors expose 4K
        natively under their existing pixel format, so this must NOT
        force an unverified fourcc like MJPG, which could break
        negotiation on hardware that doesn't support it."""
        config = CameraConfig(
            id="camera-video0", source="/dev/video0",
            capture_width=3840, capture_height=2160,
        )
        fake_capture = MagicMock()
        fake_capture.isOpened.return_value = True
        fake_capture.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 3840.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 2160.0,
        }.get(prop, 0.0)
        fake_capture.read.return_value = (True, np.zeros((2160, 3840, 3), dtype=np.uint8))

        stream = CameraStream(config)
        with patch("cv2.VideoCapture", return_value=fake_capture):
            agen = stream.frames()
            frame = await agen.__anext__()
            await agen.aclose()

        fake_capture.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        fake_capture.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        fourcc_calls = [call for call in fake_capture.set.call_args_list if call.args[0] == cv2.CAP_PROP_FOURCC]
        self.assertEqual(fourcc_calls, [])
        self.assertEqual(frame.shape, (2160, 3840, 3))

    async def test_leaves_capture_resolution_untouched_when_not_configured(self) -> None:
        config = CameraConfig(id="camera-video0", source="/dev/video0")
        fake_capture = MagicMock()
        fake_capture.isOpened.return_value = True
        fake_capture.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))

        stream = CameraStream(config)
        with patch("cv2.VideoCapture", return_value=fake_capture):
            agen = stream.frames()
            await agen.__anext__()
            await agen.aclose()

        fake_capture.set.assert_not_called()


if __name__ == "__main__":
    unittest.main()
