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


# -- only the omni weights, when omni is selected --------------------------


def omni_config(**overrides):
    return EggConfig.model_validate(
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


def test_the_voice_daemon_is_stopped_only_once_chat_left_it() -> None:
    """It holds Whisper and YAMNet resident and will not release them.

    Stopping it is safe only because replies now come from the adapter, so
    the two are separate settings rather than one implying the other.
    """

    assert omni_config().omni_adapter.silences_voice_daemon is True
    assert omni_config(silence_voice_daemon=False).omni_adapter.silences_voice_daemon is False
    # Not exclusive: the daemon is still the one answering, so it stays up.
    assert omni_config(exclusive=False).omni_adapter.silences_voice_daemon is False


def test_web_search_survives_stopping_the_daemon() -> None:
    """The adapter carries the tool suite, so nothing is traded away.

    These tools came from Omnius and run without loading any model, so the
    adapter can execute them itself. Web search stays available with the
    daemon stopped, rather than being withdrawn along with it.
    """

    from egg_companion.adapters.omni import OmniAdapterClient
    from egg_companion.adapters.omnius import OmniusClient

    config = omni_config()
    client = OmniusClient(config.omnius, OmniAdapterClient(config.omni_adapter))
    names = {
        definition["function"]["name"]
        for definition in client._realtime_tool_definitions()
    }
    assert "search_current_web" in names


def test_web_search_reads_a_page_rather_than_returning_bare_links() -> None:
    """Discovery returns titles and URLs; most snippets come back empty.

    Answering a question needs the page itself, so the search is followed by
    a fetch the way the portal's own loop does it. Evidence made of link text
    alone is what "I cannot pull a headline from those results" sounds like.
    """

    from egg_companion.adapters.omni import OmniAdapterClient
    from egg_companion.adapters.omnius import OmniusClient

    config = omni_config()
    adapter = OmniAdapterClient(config.omni_adapter)
    client = OmniusClient(config.omnius, adapter)
    calls: list[tuple[str, dict]] = []

    async def execute_tool(name, arguments):
        calls.append((name, dict(arguments)))
        if name == "web_search":
            return {
                "results": [
                    {"title": "Starship", "url": "https://example.test/s", "snippet": ""},
                    {"title": "Launches", "url": "https://example.test/l", "snippet": ""},
                ]
            }
        return {"text": "Starship completed its eleventh flight test on Tuesday."}

    adapter.execute_tool = execute_tool
    evidence = asyncio.run(client.web_search("starship latest flight", num_results=4))

    assert [name for name, _ in calls] == ["web_search", "web_fetch"]
    assert calls[0][1]["query"] == "starship latest flight"
    assert calls[1][1]["url"] == "https://example.test/s"
    assert "https://example.test/s" in evidence
    assert "eleventh flight test" in evidence


def test_a_page_that_will_not_load_does_not_fail_the_search() -> None:
    """One bad result must not cost the turn its evidence."""

    from egg_companion.adapters.omni import OmniAdapterClient
    from egg_companion.adapters.omnius import OmniusClient

    config = omni_config()
    adapter = OmniAdapterClient(config.omni_adapter)
    client = OmniusClient(config.omnius, adapter)

    async def execute_tool(name, arguments):
        if name == "web_search":
            return {"results": [{"title": "Starship", "url": "https://example.test/s"}]}
        raise RuntimeError("connection reset")

    adapter.execute_tool = execute_tool
    evidence = asyncio.run(client.web_search("starship", num_results=2))

    assert "https://example.test/s" in evidence


def test_replies_are_generated_through_the_adapter_in_omni_mode() -> None:
    """With the daemon stopped, the adapter has to be the one answering."""

    from egg_companion.adapters.omni import OmniAdapterClient
    from egg_companion.adapters.omnius import OmniusClient

    config = omni_config()
    adapter = OmniAdapterClient(config.omni_adapter)
    client = OmniusClient(config.omnius, adapter)
    seen: dict[str, object] = {}

    async def chat(messages, **kwargs):
        seen["messages"] = messages
        seen.update(kwargs)
        return {"role": "assistant", "content": "Hello, wonderful friend!"}

    adapter.chat = chat
    reply = asyncio.run(
        client._realtime_chat([{"role": "user", "content": "hi"}], allow_tool_requests=False)
    )

    assert reply == "Hello, wonderful friend!"
    assert seen["messages"] == [{"role": "user", "content": "hi"}]
    # Reasoning stays off so tokens are not spent before the reply.
    assert seen["think"] is False


# -- one worker answers everything -----------------------------------------


def test_language_is_charged_to_the_worker_that_holds_the_weights() -> None:
    """Comprehension and language are the same weights in the same process.

    Budgeting a separate Ollama slot for them means accounting for two copies
    of one model. They do not both fit, so the manager swaps between hearing
    and answering on every turn and each turn pays a full reload.
    """

    assert omni_config().omni_adapter.language_component == "omni_comprehension"
    assert (
        omni_config(language_stage="ollama").omni_adapter.language_component
        == "ollama_language"
    )


def test_no_separate_ollama_slot_is_budgeted_for_the_same_model(monkeypatch) -> None:
    from egg_companion.services.residency_wiring import build_residency_manager

    manager = build_residency_manager(omni_config())
    assert manager is not None
    assert "ollama_language" not in manager._components
    assert {"omni_comprehension", "omni_speech"} <= set(manager._components)


def test_speech_can_always_be_admitted_for_a_reply(monkeypatch) -> None:
    """Comprehension must stay evictable, or a turn ends without a voice.

    Measured on this module: comprehension 16.7 GiB and speech 6.7 GiB come
    to 23.4 of 23.6 usable, the desktop session holding the rest. Pinning
    comprehension -- tempting, since it answers both hearing and language --
    means speech can never be admitted and every reply is silent.
    """

    from egg_companion.services.residency_wiring import build_residency_manager

    manager = build_residency_manager(omni_config())
    assert manager is not None
    comprehension = manager._components["omni_comprehension"]
    assert comprehension.always_resident is False
    assert comprehension.pinned is False

    # Speech outranks nothing; it is reclaimed first when idle. What matters
    # is that the expensive worker can give way to it at all.
    assert comprehension.priority > manager._components["omni_speech"].priority


def test_a_busy_worker_is_not_mistaken_for_an_absent_one() -> None:
    """llama.cpp answers 503 both while loading and while every slot is busy.

    Reading "busy" as "unloaded" made the manager try to load a component
    that was already resident -- and a pinned component cannot be evicted to
    make room for itself, so the turn was refused with "only 6.2 GiB is
    available (pinned: omni_comprehension)" against weights that were there.
    """

    import aiohttp
    from aiohttp import web

    from egg_companion.services.residency_wiring import _http_ready

    async def scenario() -> None:
        async def busy(_request):
            return web.json_response({"error": {"message": "no slot available"}}, status=503)

        async def loading(_request):
            return web.json_response({"error": {"message": "Loading model"}}, status=503)

        app = web.Application()
        app.router.add_get("/busy", busy)
        app.router.add_get("/loading", loading)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        try:
            assert await _http_ready(f"http://127.0.0.1:{port}/busy")() is True
            assert await _http_ready(f"http://127.0.0.1:{port}/loading")() is False
        finally:
            await runner.cleanup()

    asyncio.run(scenario())


def test_a_long_prompt_is_trimmed_rather_than_refused() -> None:
    """llama.cpp refuses an over-long prompt; Ollama quietly truncated it.

    A turn that overran the worker's context got no reply at all:
    "request (4108 tokens) exceeds the available context size (4096)".
    Growing the worker is not the answer here -- 4096 to 6144 tokens per slot
    was measured at 3.5 GiB, half the room the speech worker needs.
    """

    from egg_companion.adapters.omni import OmniAdapterClient

    system = {"role": "system", "content": "You are Egg."}
    old = [
        {"role": "user", "content": "x" * 6000},
        {"role": "assistant", "content": "y" * 6000},
    ]
    latest = {"role": "user", "content": "What did you just hear?"}

    fitted = OmniAdapterClient._fit_to_context(
        [system, *old, latest], context_tokens=4096, reply_tokens=160
    )

    # The system prompt and the live question always survive.
    assert fitted[0] == system
    assert fitted[-1] == latest
    assert OmniAdapterClient._estimated_tokens(fitted) <= 4096 - 160


def test_a_prompt_that_already_fits_is_left_alone() -> None:
    from egg_companion.adapters.omni import OmniAdapterClient

    messages = [
        {"role": "system", "content": "You are Egg."},
        {"role": "user", "content": "Hello."},
    ]
    assert (
        OmniAdapterClient._fit_to_context(messages, 4096, 160) == messages
    )


def test_an_oversized_system_prompt_is_cut_back() -> None:
    """History alone is not always the problem.

    Egg's system message carries its instructions followed by accumulated
    world context, and that block alone can overrun the worker. Trimming only
    conversation left a 4152-token request against a 4096-token worker and no
    reply at all.
    """

    from egg_companion.adapters.omni import OmniAdapterClient

    system = {"role": "system", "content": "INSTRUCTIONS. " + ("world detail. " * 2000)}
    latest = {"role": "user", "content": "What did you just hear?"}
    tools = [
        {
            "type": "function",
            "function": {"name": "inspect_current_camera", "description": "d" * 400},
        }
    ]

    fitted = OmniAdapterClient._fit_to_context(
        [system, latest], context_tokens=4096, reply_tokens=160, tools=tools
    )

    assert OmniAdapterClient._estimated_tokens(fitted, tools) <= 4096 - 160
    # The instructions at the head survive; the question is untouched.
    assert str(fitted[0]["content"]).startswith("INSTRUCTIONS.")
    assert fitted[-1] == latest


def test_tool_schemas_count_against_the_context_budget() -> None:
    """They are rendered into the prompt, so ignoring them understates it."""

    from egg_companion.adapters.omni import OmniAdapterClient

    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "t", "description": "d" * 3000}}]

    assert OmniAdapterClient._estimated_tokens(messages, tools) > (
        OmniAdapterClient._estimated_tokens(messages) + 500
    )
