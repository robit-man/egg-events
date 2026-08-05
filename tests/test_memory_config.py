from pathlib import Path

import pytest

from egg_companion.config import EggConfig, load_config


def test_memory_defaults_are_bounded() -> None:
    config = EggConfig.model_validate({"audio": {"input_device": "default"}, "omnius": {"model": "x", "voice_model": "x"}})
    assert config.memory.graph_max_hops == 2
    assert config.memory.migration_mode == "dual_write"


def test_memory_rejects_unbounded_graph_walk() -> None:
    with pytest.raises(ValueError):
        EggConfig.model_validate({
            "audio": {"input_device": "default"}, "omnius": {"model": "x", "voice_model": "x"},
            "memory": {"graph_max_hops": 7},
        })


def test_project_memory_yaml_loads() -> None:
    config = load_config(Path("config/egg.yaml"))
    assert config.memory.enabled
    assert config.privacy.deletion_enabled
