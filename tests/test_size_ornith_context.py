from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).parents[1] / "scripts" / "size_ornith_context.py"
    spec = importlib.util.spec_from_file_location("size_ornith_context", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = _load_module()
GiB = 1024**3


def test_num_ctx_scales_gradually_with_available_memory() -> None:
    small = module.compute_num_ctx(int(4 * GiB), 1)
    medium = module.compute_num_ctx(int(8 * GiB), 1)
    large = module.compute_num_ctx(int(20 * GiB), 1)
    assert module.MIN_CTX <= small < medium < large <= module.MAX_CTX


def test_num_ctx_never_below_floor_or_above_ceiling() -> None:
    assert module.compute_num_ctx(0, 1) == module.MIN_CTX
    assert module.compute_num_ctx(int(1000 * GiB), 1) == module.MAX_CTX


def test_num_ctx_splits_budget_across_parallel_slots() -> None:
    one_slot = module.compute_num_ctx(int(20 * GiB), 1)
    two_slots = module.compute_num_ctx(int(20 * GiB), 2)
    assert two_slots < one_slot


def test_num_ctx_is_rounded() -> None:
    ctx = module.compute_num_ctx(int(7 * GiB), 1)
    assert ctx % module.ROUND_TO == 0


def test_current_num_ctx_reads_existing_parameter() -> None:
    lines = ["FROM some/blob", "PARAMETER num_ctx 8192", "PARAMETER temperature 1"]
    assert module.current_num_ctx(lines) == 8192


def test_current_num_ctx_missing_parameter_returns_none() -> None:
    assert module.current_num_ctx(["FROM some/blob", "PARAMETER temperature 1"]) is None
