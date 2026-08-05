from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import cv2
import numpy as np

from egg_companion.config import CameraConfig

logger = logging.getLogger(__name__)


class CameraStream:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._capture: cv2.VideoCapture | None = None

    async def frames(self) -> AsyncIterator[np.ndarray]:
        source = self.config.source.removeprefix("v4l2://")
        self._capture = cv2.VideoCapture(source, cv2.CAP_V4L2 if source.startswith("/dev/") else cv2.CAP_FFMPEG)
        if not self._capture.isOpened():
            raise RuntimeError(f"camera {self.config.id} cannot open {self.config.source}")
        interval = 1 / self.config.fps
        try:
            while True:
                started = asyncio.get_running_loop().time()
                ok, frame = await asyncio.to_thread(self._capture.read)
                if not ok or frame is None:
                    raise RuntimeError(f"camera {self.config.id} returned no frame")
                yield frame
                await asyncio.sleep(max(0, interval - (asyncio.get_running_loop().time() - started)))
        finally:
            self._capture.release()
