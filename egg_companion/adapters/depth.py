"""On-demand monocular metric depth via a bounded-lifetime subprocess.

Depth Anything 3 (DA3METRIC-LARGE) already exists, fully downloaded and
working, in a sibling project's own Python environment
(/home/egg/Depth-Anything-3/venv) -- this repo does not vendor it. Each
call spawns scripts/depth_worker.py in that venv, waits for it to write
its result to a temp directory, and reads it back. The subprocess boundary
is deliberate: this Jetson doesn't have the ~4GB of headroom to keep the
model resident alongside YOLO/pose/CLIP/Whisper/the local LLM, so paying a
~15-30s cold-load cost per call in exchange for guaranteed full memory
reclaim on process exit is the right tradeoff for an occupancy map that
only needs to update every tens of seconds, not every frame.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from egg_companion.config import OccupancyConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DepthResult:
    depth: np.ndarray  # HxW float32, best-effort metric meters
    confidence: np.ndarray | None  # HxW float32 in [0, 1], or None if unavailable
    model: str


class DepthEstimator:
    """Runs depth inference as a subprocess in the Depth Anything 3 venv."""

    def __init__(self, config: OccupancyConfig, repo_root: Path | None = None) -> None:
        self.config = config
        self._repo_root = repo_root or Path(__file__).resolve().parents[2]

    def _resolve_worker_script(self) -> Path:
        worker_script = Path(self.config.depth_worker_script)
        if not worker_script.is_absolute():
            worker_script = self._repo_root / worker_script
        return worker_script

    async def estimate(self, image_png: bytes) -> DepthResult | None:
        if not self.config.enabled:
            return None
        venv_python = Path(self.config.depth_venv_python)
        worker_script = self._resolve_worker_script()
        if not venv_python.exists():
            logger.debug("depth estimator unavailable: no venv python at %s", venv_python)
            return None
        if not worker_script.exists():
            logger.debug("depth estimator unavailable: no worker script at %s", worker_script)
            return None

        with TemporaryDirectory(prefix="egg-depth-") as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "frame.png"
            image_path.write_bytes(image_png)
            output_dir = tmp_path / "out"

            try:
                process = await asyncio.create_subprocess_exec(
                    str(venv_python), str(worker_script),
                    str(image_path), str(output_dir),
                    "--model", self.config.model_name,
                    "--process-res", str(self.config.process_res),
                    cwd=self.config.depth_repo_dir,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            except OSError as error:
                logger.warning("could not start depth worker: %s", error)
                return None

            try:
                await asyncio.wait_for(
                    process.wait(), timeout=self.config.subprocess_timeout_seconds,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                logger.warning(
                    "depth worker timed out after %.0fs",
                    self.config.subprocess_timeout_seconds,
                )
                return None

            metadata_path = output_dir / "metadata.json"
            if not metadata_path.exists():
                logger.warning(
                    "depth worker produced no metadata (exit=%s)", process.returncode
                )
                return None
            try:
                metadata = json.loads(metadata_path.read_text())
            except (json.JSONDecodeError, OSError) as error:
                logger.warning("depth worker metadata unreadable: %s", error)
                return None
            if metadata.get("error"):
                logger.warning("depth worker failed: %s", metadata["error"])
                return None

            depth_path = output_dir / "depth.npy"
            if not depth_path.exists():
                return None
            depth = np.load(depth_path)

            confidence = None
            conf_path = output_dir / "conf.npy"
            if metadata.get("has_conf") and conf_path.exists():
                confidence = np.load(conf_path)

            return DepthResult(
                depth=depth,
                confidence=confidence,
                model=str(metadata.get("model", self.config.model_name)),
            )
