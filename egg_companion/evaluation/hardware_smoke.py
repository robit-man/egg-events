from __future__ import annotations

import os
import platform
from pathlib import Path


def hardware_smoke() -> dict[str, object]:
    """Non-destructive local resource inventory for evaluation reports."""
    cameras = sorted(str(path) for path in Path("/dev").glob("video*") if path.exists())
    load = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "camera_nodes": cameras,
        "camera_count": len(cameras),
        "load_average": [round(value, 3) for value in load],
    }
