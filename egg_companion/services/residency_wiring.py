"""Build the residency manager for this deployment's heavy components.

Kept separate from the manager itself so the manager stays a general
admission-control mechanism with no knowledge of Ollama, systemd units, or
Egg's particular models.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import aiohttp

from egg_companion.config import EggConfig
from egg_companion.services.residency import (
    LANGUAGE_COMPONENT,
    Component,
    WeightResidencyManager,
    systemd_component,
)

logger = logging.getLogger(__name__)


def _http_ready(url: str, timeout_seconds: float = 2.0):
    """Probe a worker's health endpoint for "its weights are usable".

    A llama.cpp server answers 503 for two very different situations: the
    model is still loading, and every slot is busy right now. Only the first
    means the weights are absent. Treating a busy server as unloaded makes
    the manager try to load a component that is already resident -- and when
    that component is pinned it cannot be evicted to make room for itself, so
    the turn is refused against weights that were there the whole time.
    """

    async def ready() -> bool:
        try:
            timeout = aiohttp.ClientTimeout(total=timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status < 400:
                        return True
                    if response.status != 503:
                        return False
                    body = (await response.text())[:300].lower()
        except Exception:  # noqa: BLE001 - an unreachable probe means not ready
            return False
        # Busy is usable; still loading is not.
        return "loading" not in body

    return ready


def ollama_component(
    base_url: str,
    model: str,
    cost_gib: float,
    *,
    priority: int,
    keep_alive: str = "30m",
) -> Component:
    """Ollama serving one tag, as something the budget can account for.

    Ollama owns its own process, so "load" and "unload" here mean residency of
    the model rather than lifetime of the server: a zero keep-alive releases
    the weights, and a minimal generation brings them back. Registering it
    matters even though Egg does not supervise it -- an unregistered consumer
    of the same unified memory is a hole in the guarantee.
    """

    root = base_url.rstrip("/")

    async def _post(path: str, payload: dict, timeout_seconds: float) -> dict | None:
        try:
            timeout = aiohttp.ClientTimeout(total=timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(f"{root}{path}", json=payload) as response:
                    if response.status >= 400:
                        return None
                    return await response.json()
        except Exception as error:  # noqa: BLE001
            logger.debug("ollama %s failed: %s", path, error)
            return None

    async def load() -> None:
        # An empty prompt asks Ollama to load the model and nothing else.
        await _post(
            "/api/generate",
            {"model": model, "keep_alive": keep_alive},
            timeout_seconds=600,
        )

    async def unload() -> None:
        await _post(
            "/api/generate",
            {"model": model, "keep_alive": 0},
            timeout_seconds=60,
        )

    async def is_loaded() -> bool:
        try:
            timeout = aiohttp.ClientTimeout(total=3)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{root}/api/ps") as response:
                    if response.status >= 400:
                        return False
                    payload = await response.json()
        except Exception:  # noqa: BLE001
            return False
        return any(
            str(item.get("name") or "") == model
            for item in (payload.get("models") or [])
        )

    return Component(
        name=LANGUAGE_COMPONENT,
        cost_gib=cost_gib,
        load=load,
        unload=unload,
        is_loaded=is_loaded,
        priority=priority,
        load_timeout_seconds=600,
    )


def _warn_on_context_drift(settings) -> None:
    """Complain if a legacy fixed unit disagrees with the configured ceiling.

    Current units use ``{context}`` and select from MemAvailable at launch. The
    check remains for old installations that still bake a numeric ``-c`` into
    the unit, because those can silently drift from configuration.
    """

    unit_path = (
        Path.home() / ".config/systemd/user" / settings.comprehension_unit
    )
    try:
        text = unit_path.read_text(encoding="utf-8")
    except OSError:
        return
    if "comprehension_launcher.py" in text and "-c {context}" in text:
        return
    match = re.search(r"-c\s+(\d+)", text)
    if not match:
        return
    started_with = int(match.group(1))
    if started_with != settings.comprehension_context_tokens:
        logger.warning(
            "residency: %s starts the comprehension worker with -c %d but the "
            "budget assumes %d; regenerate the unit or the worker may be "
            "OOM-killed mid-turn",
            settings.comprehension_unit,
            started_with,
            settings.comprehension_context_tokens,
        )


def _warn_on_language_stage_drift(config: EggConfig) -> None:
    """Complain if the adapter answers language somewhere the budget ignores.

    The budget follows `omni_adapter.language_stage`; the adapter follows
    OMNI_LANGUAGE_API in its unit. When they disagree the weights are real but
    unaccounted, which is the failure that over-commits the module and takes
    the companion down mid-turn rather than refusing a turn.
    """

    unit_path = (
        Path.home() / ".config/systemd/user" / config.omni_adapter.autostart_unit
    )
    try:
        text = unit_path.read_text(encoding="utf-8")
    except OSError:
        return
    on_comprehension = "OMNI_LANGUAGE_API=openai" in text
    expected = config.omni_adapter.language_stage == "comprehension"
    if on_comprehension != expected:
        logger.warning(
            "residency: %s answers language on %s but the budget assumes %s; "
            "set omni_adapter.language_stage and the unit's OMNI_LANGUAGE_API "
            "to agree, or the weights it loads will be unaccounted",
            config.omni_adapter.autostart_unit,
            "the comprehension worker" if on_comprehension else "Ollama",
            config.omni_adapter.language_stage,
        )


def build_residency_manager(config: EggConfig) -> WeightResidencyManager | None:
    """Register every heavy component this deployment can load.

    Returns None when residency management is switched off, in which case
    callers behave as they did before: whoever started the components is
    responsible for their having fitted.
    """

    settings = config.residency
    if not settings.enabled:
        logger.info("residency management is disabled; component sizing is unmanaged")
        return None

    manager = WeightResidencyManager(reserve_gib=settings.reserve_gib)
    _warn_on_context_drift(settings)
    _warn_on_language_stage_drift(config)
    adapter_base = str(config.omni_adapter.base_url).rstrip("/")

    manager.register(
        systemd_component(
            "omni_comprehension",
            settings.comprehension_unit,
            settings.comprehension_cost_gib,
            priority=settings.comprehension_priority,
            # Readiness is the worker answering, not the unit being active: an
            # active unit whose model is still loading is not something a
            # request can be routed to.
            ready=_http_ready("http://127.0.0.1:8901/health"),
            load_timeout_seconds=settings.comprehension_load_timeout_seconds,
            # Releasing this one on idle buys nothing. It answers hearing,
            # vision and language, so the only thing that ever wants its
            # memory is the speech worker -- which takes it by eviction when
            # it needs it. Letting it go after three quiet minutes just means
            # the next thing said pays a 16.7 GiB reload before Egg can even
            # transcribe it.
            idle_release_seconds=(
                0.0
                if config.omni_adapter.language_component == "omni_comprehension"
                else settings.comprehension_idle_release_seconds
            ),
            # Not always_resident: if the transient TTS worker cannot fit
            # beside comprehension, speech may evict it after ASR and language
            # have both used this same process. That is at most one necessary
            # transition, not a reload between every stage.
            always_resident=False,
        )
    )
    manager.register(
        systemd_component(
            "omni_speech",
            settings.speech_unit,
            # This unit is only the adapter/controller. Qwen3-TTS is a
            # non-persistent child whose measured peak is admitted separately
            # immediately before synthesis.
            settings.adapter_service_cost_gib,
            priority=settings.speech_priority,
            ready=_http_ready(f"{adapter_base}/healthz"),
            load_timeout_seconds=settings.speech_load_timeout_seconds,
            idle_release_seconds=settings.speech_idle_release_seconds,
            # Stopping this tiny transport cannot reclaim the TTS weights --
            # they already exit after each utterance -- and restarting it used
            # to rerun heavyweight startup smoke. Keep the controller, not the
            # model, resident.
            always_resident=True,
        )
    )
    if config.omni_adapter.language_component == LANGUAGE_COMPONENT:
        manager.register(
            ollama_component(
                str(config.omnius.vision_base_url),
                config.omnius.model,
                settings.language_cost_gib,
                priority=settings.language_priority,
                keep_alive=config.omnius.chat_keep_alive,
            )
        )
    else:
        # Language runs on the comprehension worker, which is already
        # registered and already holds these weights. Registering a second
        # Ollama slot for the same model would budget for two copies and
        # invite the manager to swap between them every turn.
        logger.info(
            "residency: language runs on %s; no separate Ollama slot is budgeted",
            config.omni_adapter.language_component,
        )
    logger.info(
        "residency: %.1f GiB total, %.1f GiB reserved, components %s",
        manager.total_gib,
        manager.reserve_gib,
        ", ".join(sorted(manager._components)),
    )
    return manager
