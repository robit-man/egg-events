"""The voice page must name the backend that actually serves each stage.

In omni mode the Whisper and Supertonic stages are stopped on purpose. The
voice daemon keeps reporting their state regardless, so passing it straight
through makes the page claim ASR is not ready and that Supertonic is
speaking -- while the omni package is in fact doing both.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from egg_companion.adapters.omni import OmniAdapterClient, OmniAdapterError
from egg_companion.adapters.omnius import OMNI_BACKEND, OmniusClient
from egg_companion.config import EggConfig

MODEL = "robit/ornith-1.5-omni:q4km"

# What the voice daemon reports once omni mode has stopped its own stages.
STOPPED_STACK_STATE = {
    "asrEngineId": "openai-whisper",
    "asrModelId": "base",
    "asrBackend": "openai-whisper",
    "asrReady": False,
    "voiceModelId": "supertonic",
    "voiceReady": True,
    "listenActive": True,
}


def omni_client(**overrides) -> OmniusClient:
    config = EggConfig.model_validate(
        {
            "audio": {"input_device": "default", "doa_mode": "disabled"},
            "omnius": {"model": MODEL, "voice_model": "supertonic"},
            "omni_adapter": {"mode": "omni", "model": MODEL, **overrides},
            "identity": {"enabled": False},
            "object_learning": {"enabled": False},
            "memory": {"enabled": False},
            "camera_discovery": {"enabled": False},
        }
    )
    return OmniusClient(config.omnius, OmniAdapterClient(config.omni_adapter))


def test_serving_omni_is_reported_as_the_asr_and_voice_backend() -> None:
    client = omni_client()
    state = client._with_omni_service_state(dict(STOPPED_STACK_STATE), True)

    assert state["asrBackend"] == OMNI_BACKEND
    assert state["asrModelId"] == MODEL
    assert state["asrReady"] is True
    assert state["voiceModelId"] == MODEL
    # Untouched fields still come from the daemon that owns them.
    assert state["listenActive"] is True


def test_the_page_is_not_told_the_stopped_stack_is_ready() -> None:
    client = omni_client()
    state = client._with_omni_service_state(dict(STOPPED_STACK_STATE), False)

    # Exclusive mode: omni owns the stage whether or not it is answering, so
    # a failure has to surface as omni failing rather than as Supertonic
    # quietly taking the credit.
    assert state["voiceModelId"] == MODEL
    assert state["voiceReady"] is False
    assert state["asrReady"] is False


def test_a_non_exclusive_fallback_still_names_the_daemon_that_answers() -> None:
    client = omni_client(exclusive=False)
    state = client._with_omni_service_state(dict(STOPPED_STACK_STATE), False)

    # Here Whisper and Supertonic really are the ones serving.
    assert state["asrBackend"] == "openai-whisper"
    assert state["voiceModelId"] == "supertonic"


def test_catalog_entries_carry_the_readiness_the_page_reads() -> None:
    client = omni_client()
    tts, asr = client._with_omni_catalog_entries({"models": []}, {"models": []}, True)

    assert asr["models"][0]["readiness"]["weightsReady"] is True
    assert tts["models"][0]["readiness"]["weightsReady"] is True
    assert asr["models"][0]["id"] == MODEL


def test_readiness_ignores_the_routing_health_cache() -> None:
    """The badge must show what is true now, not the cached routing verdict.

    Routing keeps a healthy window and a failure cooldown on purpose. Read
    for a status display they invert it: the page claimed the adapter was up
    for the whole TTL after it was stopped, and down for the whole cooldown
    after it came back.
    """

    import time

    client = omni_client()
    adapter = client._omni
    calls: list[int] = []
    alive = False

    async def health(timeout_seconds=None):
        calls.append(1)
        if not alive:
            raise OmniAdapterError("connection refused")
        return {"ok": True}

    adapter.health = health

    # A cached healthy window must not survive the adapter going away.
    adapter._healthy_until = time.monotonic() + 3600
    assert asyncio.run(adapter.probe()) is False

    # Nor may the failure cooldown outlive its recovery.
    alive = True
    adapter._cooldown_until = time.monotonic() + 3600
    assert asyncio.run(adapter.probe()) is True

    # Every check is a real question put to the adapter.
    assert len(calls) == 2


def test_a_cached_catalog_is_still_told_the_truth_about_readiness() -> None:
    """The five-minute catalog cache must not cache readiness with it.

    Returning the cached catalog early once skipped the omni rewrite
    entirely, so the page showed the stopped Whisper stage for up to five
    minutes after the adapter came up.
    """

    client = omni_client()
    serving = True

    async def probe() -> bool:
        return serving

    client._omni.probe = probe
    client._voice_catalog_cache = {"tts": {"models": []}, "asr": {"models": []}}
    client._voice_catalog_cached_at = float("inf")

    async def state_is(expected_ready: bool) -> dict:
        view = await client._omni_voice_view(
            client._voice_catalog_cache, dict(STOPPED_STACK_STATE)
        )
        assert view["state"]["asrReady"] is expected_ready
        assert view["asr"]["models"][0]["id"] == MODEL
        assert view["asr"]["models"][0]["readiness"]["weightsReady"] is expected_ready
        return view

    asyncio.run(state_is(True))
    serving = False
    # Same cache entry, adapter now down: the page must follow, not lag.
    asyncio.run(state_is(False))


def test_a_manager_released_stage_reads_as_standby_not_as_broken() -> None:
    """The residency manager stops the speech unit to reclaim memory.

    Reporting that as a failure is what put "ASR ... not ready" on the voice
    page while the pipeline was in fact working: the manager loads the unit
    again the moment a turn asks for it.
    """

    client = omni_client()
    adapter = client._omni

    async def health(timeout_seconds=None):
        raise OmniAdapterError("connection refused")

    adapter.health = health
    adapter._residency = SimpleNamespace(manages=lambda name: True)

    available = asyncio.run(adapter.availability())
    assert available["state"] == "standby"
    assert available["ready"] is True

    state = client._with_omni_service_state(dict(STOPPED_STACK_STATE), available)
    assert state["asrState"] == "standby"
    assert state["asrModelId"] == MODEL


def test_an_unmanaged_dead_adapter_is_still_reported_as_unavailable() -> None:
    """Standby is only honest when something is going to start it again."""

    client = omni_client()
    adapter = client._omni

    async def health(timeout_seconds=None):
        raise OmniAdapterError("connection refused")

    adapter.health = health
    adapter._residency = None

    available = asyncio.run(adapter.availability())
    assert available["state"] == "unavailable"
    assert available["ready"] is False
    assert "connection refused" in str(available["detail"])


def test_polling_status_does_not_park_the_adapter() -> None:
    """Observing must not degrade the thing observed.

    The probe used to open the routing failure cooldown, so simply looking
    at the voice page while the manager had the unit released made the next
    turn refuse against an adapter that was about to be started.
    """

    client = omni_client()
    adapter = client._omni

    async def health(timeout_seconds=None):
        raise OmniAdapterError("connection refused")

    adapter.health = health
    for _ in range(3):
        assert asyncio.run(adapter.probe()) is False

    assert adapter._cooldown_until == 0.0
    # The reason is still recorded for display.
    assert "connection refused" in str(adapter._last_error)


def test_a_managed_reload_clears_a_stale_failure_cooldown() -> None:
    """Evicting a unit must not leave a verdict that outlives the reload.

    The manager stops and starts the speech unit as ordinary operation. The
    failure that eviction causes used to park the adapter for the whole
    cooldown, so the turn that triggered the reload was refused against a
    component the manager had just brought back up.
    """

    import contextlib
    import time

    client = omni_client()
    adapter = client._omni

    @contextlib.asynccontextmanager
    async def require(name):
        yield SimpleNamespace(name=name)

    async def ensure_headroom(gib, exclude=""):
        return None

    adapter._residency = SimpleNamespace(
        require=require,
        ensure_headroom=ensure_headroom,
        manages=lambda name: True,
        reserve_gib=2.0,
    )
    adapter._cooldown_until = time.monotonic() + 3600
    adapter._last_error = "ClientConnectorError: Cannot connect"

    async def enter() -> None:
        async with adapter._resident(adapter.SPEECH_COMPONENT):
            pass

    asyncio.run(enter())
    assert adapter._cooldown_until == 0.0
    # Health is cleared too: the next call asks rather than assumes.
    assert adapter._healthy_until == 0.0


def test_comprehension_does_not_evict_the_daemon_it_speaks_through(monkeypatch) -> None:
    """The adapter daemon is the transport for every task, not a peer.

    The speech component manages the unit serving port 8910, and every
    request -- transcription included -- goes through it. Holding only the
    comprehension worker let the manager pick that daemon as its eviction
    candidate, tearing down the HTTP server the same request was about to
    use: the manager logged a successful load and the turn then failed with
    "Cannot connect to host 127.0.0.1:8910".
    """

    from egg_companion.services import residency as residency_module
    from egg_companion.services.residency import Component, WeightResidencyManager

    manager = WeightResidencyManager(total_gib=30.0, reserve_gib=2.0)
    evicted: list[str] = []
    loaded = {"omni_speech": True, "omni_comprehension": False}
    # Only room for comprehension once something has been given up.
    free = {"gib": 9.0}
    costs = {"omni_speech": 6.4, "omni_comprehension": 16.7}

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

    # Ollama stands in for the component that *should* be given up: it is not
    # the transport, so losing it costs the turn a reload and nothing more.
    costs["ollama_language"] = 15.6
    loaded["ollama_language"] = True

    # The real wiring registers the daemon as a small, always-resident
    # transport: stopping it reclaims nothing, because its speech worker
    # already exits after each utterance.
    costs["omni_speech"] = 0.25
    transport = component("omni_speech", 5)
    object.__setattr__(transport, "always_resident", True)
    manager.register(transport)
    manager.register(component("omni_comprehension", 10))
    manager.register(component("ollama_language", 8))

    monkeypatch.setattr(residency_module, "available_memory_gib", lambda: free["gib"])
    monkeypatch.setattr(manager, "settle_seconds", 0.0, raising=False)

    client = OmniAdapterClient(
        omni_client()._omni.config.model_copy(update={"speech_headroom_gib": 0.1}),
        manager,
    )

    held: list[bool] = []

    async def use_comprehension() -> None:
        async with client._resident(client.COMPREHENSION_COMPONENT):
            # The daemon is up and serving for the whole request. It is not
            # pinned by this call: its own always_resident flag protects it,
            # and pinning it here would reserve the speech worker's budget
            # for the daemon's lifetime -- which deadlocked the comprehension
            # load against memory nothing was actually using.
            held.append(loaded["omni_speech"])

    asyncio.run(use_comprehension())

    # The transport was held up throughout, and the reload was paid for by
    # the language model instead.
    assert held == [True]
    assert "omni_speech" not in evicted
    assert "ollama_language" in evicted
