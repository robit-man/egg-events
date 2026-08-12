#!/usr/bin/env python3
"""Keep Omnius' isolated Jetson audio venv aligned with host CUDA 12.2.

Omnius creates this venv with system-site access. On this device that also sees
the user's cuda-python 13 namespace, which no longer exports ``cuda.cudart``.
The 1.0.629 TensorRT worker imports that API, so pin the matching CUDA 12.2
binding inside the isolated Omnius venv whenever it has been created.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


RUNTIME_PYTHON = Path.home() / ".omnius/runtimes/audio/venv/bin/python"
CUDA_PYTHON = "cuda-python==12.2.0"


def bindings_ready(python: Path) -> bool:
    result = subprocess.run(
        [
            str(python),
            "-c",
            (
                "from cuda import cudart; "
                "status,version=cudart.cudaRuntimeGetVersion(); "
                "assert int(status)==0 and version==12020"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.returncode == 0


def main() -> int:
    if not RUNTIME_PYTHON.exists():
        print("Omnius audio venv not created yet; CUDA binding repair deferred")
        return 0
    if bindings_ready(RUNTIME_PYTHON):
        print("Omnius audio CUDA 12.2 Python binding is ready")
        return 0
    install = subprocess.run(
        [
            str(RUNTIME_PYTHON),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--force-reinstall",
            CUDA_PYTHON,
        ],
        check=False,
        timeout=300,
    )
    if install.returncode != 0 or not bindings_ready(RUNTIME_PYTHON):
        print(
            "warning: Omnius audio CUDA binding repair failed; daemon will "
            "remain available but audio classifier readiness may be degraded",
            file=sys.stderr,
        )
        return 0
    print(f"Pinned {CUDA_PYTHON} in the isolated Omnius audio runtime")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
