"""Every generation must be charged to the memory budget.

Egg's replies end at the same Ollama runner whether they are addressed to it
directly or reached through the Omnius daemon. While those loads were
invisible to the residency manager, a spoken turn could admit the 16.7 GiB
comprehension worker for transcription and then pull 14.3 GiB of language
weights in beside it -- over-committing a 30 GiB module and getting the
companion killed mid-turn, which is what "it crashes when I speak to it"
looked like from outside.
"""

from __future__ import annotations

import asyncio

import pytest

from egg_companion.adapters.omnius import OmniusClient
from egg_companion.config import EggConfig
from egg_companion.services.residency import (
    LANGUAGE_COMPONENT,
    Component,
    ResidencyRefused,
    WeightResidencyManager,
    available_memory_gib,
)

MODEL = "robit/ornith-1.5-omni:q4km"


def client_with(manager: WeightResidencyManager | None) -> OmniusClient:
    config = EggConfig.model_validate(
        {
            "audio": {"input_device": "default", "doa_mode": "disabled"},
            "omnius": {"model": MODEL, "voice_model": "supertonic"},
            "identity": {"enabled": False},
            "object_learning": {"enabled": False},
            "memory": {"enabled": False},
            "camera_discovery": {"enabled": False},
        }
    )
    return OmniusClient(config.omnius, None, manager)


def budget(monkeypatch, free_gib: float) -> tuple[WeightResidencyManager, dict, list]:
    from egg_companion.services import residency as residency_module

    manager = WeightResidencyManager(total_gib=30.0, reserve_gib=2.0)
    free = {"gib": free_gib}
    evicted: list[str] = []
    loaded = {LANGUAGE_COMPONENT: False, "omni_comprehension": True}
    costs = {LANGUAGE_COMPONENT: 14.3, "omni_comprehension": 16.7}

    def component(name: str, priority: int) -> Component:
        async def load() -> None:
            loaded[name] = True
            free["gib"] -= costs[name]

        async def unload() -> None:
            evicted.append(name)
            loaded[name] = False
            free["gib"] += costs[name]

        async def is_loaded() -> bool:
            return loaded[name]

        return Component(
            name=name,
            cost_gib=costs[name],
            load=load,
            unload=unload,
            is_loaded=is_loaded,
            priority=priority,
        )

    manager.register(component(LANGUAGE_COMPONENT, 8))
    manager.register(component("omni_comprehension", 10))
    monkeypatch.setattr(residency_module, "available_memory_gib", lambda: free["gib"])
    monkeypatch.setattr(manager, "settle_seconds", 0.0, raising=False)
    return manager, loaded, evicted


def test_a_generation_evicts_comprehension_instead_of_stacking_on_it(monkeypatch) -> None:
    manager, loaded, evicted = budget(monkeypatch, free_gib=11.0)
    client = client_with(manager)
    held: list[bool] = []

    async def generate() -> None:
        async with client._language_lane(client._conversational_gate):
            held.append(loaded[LANGUAGE_COMPONENT])

    asyncio.run(generate())

    assert held == [True]
    # 14.3 + 16.7 + 2.0 reserve does not fit in 30, so one had to give way.
    assert evicted == ["omni_comprehension"]


def test_a_generation_that_cannot_fit_fails_rather_than_over_committing(monkeypatch) -> None:
    """Failing the turn beats taking the machine down with it."""

    manager, _, _ = budget(monkeypatch, free_gib=1.0)
    manager._components["omni_comprehension"]._in_use = 1  # pinned mid-request
    client = client_with(manager)

    async def generate() -> None:
        async with client._language_lane(client._conversational_gate):
            pass

    with pytest.raises(ResidencyRefused):
        asyncio.run(generate())


def test_an_unmanaged_deployment_is_unaffected() -> None:
    """Without a manager the client behaves exactly as it did before."""

    client = client_with(None)
    ran: list[bool] = []

    async def generate() -> None:
        async with client._language_lane(client._conversational_gate):
            ran.append(True)

    asyncio.run(generate())
    assert ran == [True]


def test_the_gate_is_still_exclusive(monkeypatch) -> None:
    """The lane must not weaken the serialisation the gate provided."""

    manager, _, _ = budget(monkeypatch, free_gib=20.0)
    client = client_with(manager)
    order: list[str] = []

    async def use(tag: str) -> None:
        async with client._language_lane(client._conversational_gate):
            order.append(f"enter-{tag}")
            await asyncio.sleep(0.01)
            order.append(f"exit-{tag}")

    async def both() -> None:
        await asyncio.gather(use("a"), use("b"))

    asyncio.run(both())
    # No interleaving: each generation completes before the next begins.
    assert order in (
        ["enter-a", "exit-a", "enter-b", "exit-b"],
        ["enter-b", "exit-b", "enter-a", "exit-a"],
    )


def test_omni_mode_does_not_configure_the_services_it_replaced() -> None:
    """Exclusive omni mode must leave the discrete voice stack alone.

    Whisper and Supertonic are stopped deliberately, so asking them for a
    model switch fails the readiness component against something that is
    absent by design -- a connection refused to 11436 retried forever -- and
    any call that did succeed would bring a replaced backend back.
    """

    config = EggConfig.model_validate(
        {
            "audio": {"input_device": "default", "doa_mode": "disabled"},
            "omnius": {"model": MODEL, "voice_model": "supertonic"},
            "omni_adapter": {"mode": "omni", "model": MODEL},
            "identity": {"enabled": False},
            "object_learning": {"enabled": False},
            "memory": {"enabled": False},
            "camera_discovery": {"enabled": False},
        }
    )
    assert config.omni_adapter.silences_discrete_voice is True

    traditional = config.model_copy(
        update={"omni_adapter": config.omni_adapter.model_copy(update={"mode": "traditional"})}
    )
    assert traditional.omni_adapter.silences_discrete_voice is False

    shared = config.model_copy(
        update={"omni_adapter": config.omni_adapter.model_copy(update={"exclusive": False})}
    )
    assert shared.omni_adapter.silences_discrete_voice is False


# -- Whisper's heuristics must not judge the omni package ------------------


def test_a_short_qwen_transcript_over_a_max_window_is_kept() -> None:
    """The exact rejection that made Egg look deaf.

    The VAD ran to its 6 s cap, the speaker said something short, and a
    heuristic written for Whisper -- at least 1.75 alphanumeric characters
    per second -- threw the transcript away. The audio was at 0.24 RMS, so
    nothing about the recording was in doubt.
    """

    from egg_companion.adapters.omnius import OMNI_BACKEND, OmniusClient

    payload = {"text": "Hey Egg.", "duration": 6.0, "language": "en", "segments": []}
    evidence = {
        "duration": 6.0,
        "boundary_reason": "max_utterance",
        "requested_language": "en",
        "wav_rms": 0.236,
    }

    assert (
        OmniusClient.transcription_rejection_reason(payload, evidence, engine=OMNI_BACKEND)
        is None
    )
    # Unchanged for the engine it was written for.
    assert (
        OmniusClient.transcription_rejection_reason(payload, evidence)
        == "sparse transcript over max-length acoustic window"
    )


def test_engine_agnostic_guards_still_apply_to_omni() -> None:
    """Skipping Whisper's tests must not mean trusting anything at all."""

    from egg_companion.adapters.omnius import OMNI_BACKEND, OmniusClient

    evidence = {"duration": 6.0, "requested_language": "en"}
    looping = {
        "text": "Allah Allah Allah Allah Allah Allah Allah Allah",
        "duration": 6.0,
        "segments": [],
    }
    assert (
        OmniusClient.transcription_rejection_reason(looping, evidence, engine=OMNI_BACKEND)
        is not None
    )

    # And an adapter that reports its own reason is still believed.
    refused = {"text": "hi", "rejection_reason": "no speech detected", "segments": []}
    assert (
        OmniusClient.transcription_rejection_reason(refused, evidence, engine=OMNI_BACKEND)
        == "no speech detected"
    )
