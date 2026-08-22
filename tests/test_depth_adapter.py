"""Tests for DepthEstimator's subprocess orchestration.

Uses a small fake worker script (plain stdlib, no torch/CUDA) run under
the current interpreter instead of the real Depth Anything 3 venv/model --
this exercises the real subprocess lifecycle (spawn, timeout, argument
passing, output-file parsing) without needing a GPU or a ~4GB model load.
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from egg_companion.adapters.depth import DepthEstimator
from egg_companion.config import OccupancyConfig

_FAKE_WORKER_SOURCE = textwrap.dedent(
    """
    import json
    import sys
    from pathlib import Path

    import numpy as np

    # argv: [image_path, output_dir, "--model", model_name, "--process-res", res]
    mode = sys.argv[4] if len(sys.argv) > 4 else "success"
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "hang":
        import time
        time.sleep(60)

    if mode == "fail":
        (output_dir / "metadata.json").write_text(
            json.dumps({"error": "SomeError: synthetic failure"})
        )
        sys.exit(1)

    depth = np.full((4, 4), 2.5, dtype=np.float32)
    np.save(output_dir / "depth.npy", depth)
    conf = np.full((4, 4), 0.8, dtype=np.float32)
    np.save(output_dir / "conf.npy", conf)
    (output_dir / "metadata.json").write_text(json.dumps({
        "error": None, "model": "fake-model", "has_conf": True,
        "depth_shape": [4, 4],
    }))
    """
)


@pytest.fixture
def fake_worker(tmp_path) -> Path:
    script = tmp_path / "fake_worker.py"
    script.write_text(_FAKE_WORKER_SOURCE)
    return script


def _config(fake_worker: Path, **overrides) -> OccupancyConfig:
    # Defaults to disabled on the real robot (memory pressure); these
    # tests are specifically exercising DepthEstimator's own behavior, so
    # they need it on except where a test explicitly overrides it.
    defaults = dict(
        enabled=True,
        depth_venv_python=sys.executable,
        depth_worker_script=str(fake_worker),
        depth_repo_dir=str(fake_worker.parent),
        subprocess_timeout_seconds=10.0,
    )
    defaults.update(overrides)
    return OccupancyConfig.model_validate(defaults)


class TestDepthEstimator:
    def test_disabled_returns_none_without_touching_filesystem(self, fake_worker) -> None:
        estimator = DepthEstimator(_config(fake_worker, enabled=False))
        result = asyncio.run(estimator.estimate(b"not a real png"))
        assert result is None

    def test_missing_venv_python_returns_none(self, fake_worker) -> None:
        estimator = DepthEstimator(_config(fake_worker, depth_venv_python="/nonexistent/python"))
        result = asyncio.run(estimator.estimate(b"data"))
        assert result is None

    def test_missing_worker_script_returns_none(self, fake_worker, tmp_path) -> None:
        estimator = DepthEstimator(_config(fake_worker, depth_worker_script=str(tmp_path / "nope.py")))
        result = asyncio.run(estimator.estimate(b"data"))
        assert result is None


class TestDepthEstimatorSubprocessLifecycle:
    """DepthEstimator always passes `--model <name>` positionally; the fake
    worker repurposes that value as its own behavior mode (success/fail/
    hang) since DepthEstimator has no other hook to steer a stand-in
    script's behavior without changing its real argument contract."""

    async def _run(self, fake_worker: Path, model_as_mode: str, **overrides) -> "object | None":
        estimator = DepthEstimator(_config(fake_worker, model_name=model_as_mode, **overrides))
        return await estimator.estimate(b"fake png bytes")

    def test_successful_run_returns_depth_and_confidence(self, fake_worker) -> None:
        result = asyncio.run(self._run(fake_worker, "success"))
        assert result is not None
        assert result.model == "fake-model"
        assert result.depth.shape == (4, 4)
        assert np.allclose(result.depth, 2.5)
        assert result.confidence is not None
        assert np.allclose(result.confidence, 0.8)

    def test_worker_reported_error_returns_none(self, fake_worker) -> None:
        result = asyncio.run(self._run(fake_worker, "fail"))
        assert result is None

    def test_worker_timeout_is_killed_and_returns_none(self, fake_worker) -> None:
        result = asyncio.run(
            self._run(fake_worker, "hang", subprocess_timeout_seconds=10.0)
        )
        assert result is None
