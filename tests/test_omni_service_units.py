"""Deployment contracts that keep Omni weight loads under admission control."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_adapter_service_starts_no_model_weights_or_smoke_generations() -> None:
    unit = (ROOT / "deploy/egg-omni-adapters.service").read_text(encoding="utf-8")

    assert "OMNI_ENABLE_COMPREHENSION=0" in unit
    assert "OMNI_TTS_PERSISTENT=0" in unit
    assert "OMNI_STARTUP_SMOKE=0" in unit
    assert "OMNI_LANGUAGE_API=openai" in unit
    assert "Wants=egg-omni-comprehension.service" not in unit
    assert "LLAMA_ARG_POLL=0" in unit


def test_comprehension_is_demand_only_and_does_not_busy_poll() -> None:
    unit = (ROOT / "deploy/egg-omni-comprehension.service").read_text(
        encoding="utf-8"
    )

    assert "WantedBy=default.target" not in unit
    assert "--poll 0 --poll-batch 0" in unit
    assert "comprehension_launcher.py" in unit
    assert "-c {context}" in unit
    assert "OMNI_COMPREHENSION_MEMORY_RESERVE_GIB=3.0" in unit
    assert "OMNI_COMPREHENSION_CONTEXT_FILE=%t/" in unit
    assert "MemoryMax=26G" in unit


def test_adapter_reads_the_context_the_launcher_actually_selected() -> None:
    unit = (ROOT / "deploy/egg-omni-adapters.service").read_text(encoding="utf-8")

    assert "OMNI_COMPREHENSION_CONTEXT_TOKENS=@CONTEXT_TOKENS@" in unit
    assert "OMNI_COMPREHENSION_CONTEXT_FILE=%t/" in unit


def test_bootstrap_removes_the_legacy_boot_dependency() -> None:
    script = (ROOT / "scripts/bootstrap-omni-adapters.sh").read_text(
        encoding="utf-8"
    )

    assert "deploy/egg-omni-adapters.service" in script
    assert "deploy/egg-omni-comprehension.service" in script
    assert "20-comprehension.conf" in script
    assert "disable egg-omni-comprehension.service" in script
