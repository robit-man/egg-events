"""Build the residency manager for this deployment's heavy components.

Kept separate from the manager itself so the manager stays a general
admission-control mechanism with no knowledge of Ollama, systemd units, or
Egg's particular models.
"""

from __future__ import annotations

import logging

import aiohttp

from egg_companion.config import EggConfig
from egg_companion.services.residency import (
    Component,
    WeightResidencyManager,
    systemd_component,
)

logger = logging.getLogger(__name__)


def _http_ready(url: str, timeout_seconds: float = 2.0):
    async def ready() -> bool:
        try:
            timeout = aiohttp.ClientTimeout(total=timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    return response.status < 400
        except Exception:  # noqa: BLE001 - an unreachable probe means not ready
            return False

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
        name="ollama_language",
        cost_gib=cost_gib,
        load=load,
        unload=unload,
        is_loaded=is_loaded,
        priority=priority,
        load_timeout_seconds=600,
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
        )
    )
    manager.register(
        systemd_component(
            "omni_speech",
            settings.speech_unit,
            settings.speech_cost_gib,
            priority=settings.speech_priority,
            ready=_http_ready(f"{adapter_base}/healthz"),
            load_timeout_seconds=settings.speech_load_timeout_seconds,
        )
    )
    manager.register(
        ollama_component(
            str(config.omnius.vision_base_url),
            config.omnius.model,
            settings.language_cost_gib,
            priority=settings.language_priority,
            keep_alive=config.omnius.chat_keep_alive,
        )
    )
    logger.info(
        "residency: %.1f GiB total, %.1f GiB reserved, components %s",
        manager.total_gib,
        manager.reserve_gib,
        ", ".join(sorted(manager._components)),
    )
    return manager
