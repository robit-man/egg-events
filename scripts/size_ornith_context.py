#!/usr/bin/env python3
"""Dynamically size Ornith's num_ctx from memory actually available right now.

The model's Modelfile declares its own num_ctx, which Ollama honors over
the server-wide OLLAMA_CONTEXT_LENGTH default -- and llama.cpp reserves the
*entire* KV cache for that context size up front, at model-load time,
regardless of how much of it a given request actually uses. A context
sized for worst-case use on a shared, memory-constrained Jetson (this
model previously shipped with num_ctx=262144) can alone reserve most of
the system's RAM before egg_companion's own processes, cameras, or ASR
ever get a chance to run -- confirmed directly on this device: dropping
262144 -> 4096 freed roughly 9.4GB.

This recomputes a context size from current MemAvailable every time Ollama
starts (wired in as an ExecStartPost, matching the existing
repair_omnius_*.py pattern in this directory) rather than hardcoding one
number, so it adapts to whatever this device can actually spare right now
-- generous when memory is free, conservative when it isn't -- instead of
either starving the model or starving everything else.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "egg.yaml"

# Empirically derived on this device (not from model architecture math --
# Ornith's exact KV-cache-per-token footprint isn't published): observed
# ~9.4GB freed moving num_ctx 262144 -> 4096, i.e. roughly 9.4GiB /
# 258048 tokens. Rounded up slightly to bias the estimate conservative
# (better to under-allocate context than to size this too optimistically
# and reintroduce the same memory crunch it exists to prevent).
BYTES_PER_TOKEN = 40_000

# Fraction of *currently available* memory (not total) this may claim for
# context, split across OLLAMA_NUM_PARALLEL slots. Chosen so num_ctx scales
# gradually across the actual range this device operates in rather than
# jumping straight to MAX_CTX the moment resizing is attempted at all --
# ~4096 near the resize floor below, ~16384 (MAX_CTX) only once ~12GiB is
# genuinely free. This runs once at daemon startup, and available memory
# only drops from there as cameras/ASR/the rest of the stack come up, so
# err toward leaving headroom rather than using it.
AVAILABLE_MEMORY_FRACTION = 0.05

# Below this, don't resize at all -- leave whatever num_ctx is already set.
# A resize forces a model reload, and reloading while memory is already
# this tight is how a transient dip turns into a real incident (confirmed
# directly: a resize attempted at ~2.9GiB available briefly cascaded into
# a load-timeout/cancel loop before the system recovered on its own).
MIN_AVAILABLE_BYTES_TO_RESIZE = 4 * 1024**3

MIN_CTX = 4096  # floor: never below what egg_companion itself requests per turn
MAX_CTX = 16384  # ceiling: conservative given how little headroom this device has
ROUND_TO = 1024

OLLAMA_READY_TIMEOUT_SECONDS = 60


def mem_available_bytes() -> int:
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not found in /proc/meminfo")


def compute_num_ctx(available_bytes: int, num_parallel: int) -> int:
    budget = (available_bytes * AVAILABLE_MEMORY_FRACTION) / max(1, num_parallel)
    raw_ctx = int(budget / BYTES_PER_TOKEN)
    bounded = max(MIN_CTX, min(MAX_CTX, raw_ctx))
    return (bounded // ROUND_TO) * ROUND_TO


def wait_for_ollama_ready(timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = subprocess.run(["ollama", "list"], check=False, capture_output=True, timeout=10)
        if result.returncode == 0:
            return True
        time.sleep(1)
    return False


def current_num_ctx(modelfile_lines: list[str]) -> int | None:
    for line in modelfile_lines:
        stripped = line.strip()
        if stripped.startswith("PARAMETER num_ctx "):
            try:
                return int(stripped.split()[-1])
            except ValueError:
                return None
    return None


def resize_model(model: str, num_ctx: int) -> bool:
    show = subprocess.run(
        ["ollama", "show", model, "--modelfile"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if show.returncode != 0:
        print(
            f"warning: could not read Modelfile for {model}; leaving its "
            f"context size as-is: {show.stderr.strip()}",
            file=sys.stderr,
        )
        return False
    lines = show.stdout.splitlines()
    if current_num_ctx(lines) == num_ctx:
        print(f"{model} already at num_ctx={num_ctx}, no reload needed")
        return True
    if not any(line.strip().startswith("PARAMETER num_ctx ") for line in lines):
        lines.append(f"PARAMETER num_ctx {num_ctx}")
    rewritten = "\n".join(
        f"PARAMETER num_ctx {num_ctx}" if line.strip().startswith("PARAMETER num_ctx ") else line
        for line in lines
    )
    tmp_modelfile = Path(f"/tmp/{model.replace('/', '_').replace(':', '_')}.Modelfile")
    tmp_modelfile.write_text(rewritten)
    try:
        create = subprocess.run(
            ["ollama", "create", model, "-f", str(tmp_modelfile)],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        tmp_modelfile.unlink(missing_ok=True)
    if create.returncode != 0:
        print(
            f"warning: resizing {model} to num_ctx={num_ctx} failed: {create.stderr.strip()}",
            file=sys.stderr,
        )
        return False
    return True


def main() -> int:
    if not CONFIG_PATH.exists():
        print(f"{CONFIG_PATH} not found; context sizing deferred")
        return 0
    config = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    omnius_config = config.get("omnius", {}) or {}
    models = sorted(
        {str(omnius_config[key]) for key in ("model", "vision_model") if omnius_config.get(key)}
    )
    if not models:
        print("no omnius model configured; context sizing deferred")
        return 0

    if not wait_for_ollama_ready(OLLAMA_READY_TIMEOUT_SECONDS):
        print(
            "warning: ollama did not become ready in time; context sizing skipped",
            file=sys.stderr,
        )
        return 0

    available = mem_available_bytes()
    if available < MIN_AVAILABLE_BYTES_TO_RESIZE:
        print(
            f"only {available / (1024**3):.1f}GiB available (below "
            f"{MIN_AVAILABLE_BYTES_TO_RESIZE / (1024**3):.0f}GiB floor); "
            "leaving existing context size alone rather than risk a reload "
            "under pressure"
        )
        return 0

    num_parallel = int(os.environ.get("OLLAMA_NUM_PARALLEL", "1") or "1")
    num_ctx = compute_num_ctx(available, num_parallel)
    print(
        f"sizing context: {available / (1024**3):.1f}GiB available, "
        f"num_parallel={num_parallel} -> num_ctx={num_ctx}"
    )
    for model in models:
        if resize_model(model, num_ctx):
            print(f"set {model} num_ctx={num_ctx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
