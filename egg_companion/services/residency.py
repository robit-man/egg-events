"""Memory-bounded residency for the heavy model components.

This is the parent manager: it owns which large weights are loaded, and its
first duty is that the machine never runs out of memory. Unified memory on a
Jetson has no separate VRAM to overflow into -- an over-commit does not merely
kill the offending process, it can take the desktop and the companion with it.
That happened on this host before this module existed.

Measured on a 32 GB AGX Orin, 29.98 GiB usable:

    baseline (OS, desktop, companion)          ~8 GiB
    Qwen3-Omni comprehension, 8K context       16.8 GiB
    Qwen3-TTS worker while cloning              6.7 GiB   (~4 without)
    Ollama language on the logical tag          5.6 GiB

Any three of those together exceed the module. A conversational turn does not
need them simultaneously, though -- it needs them in sequence: perceive, think,
speak. So components are declared with their measured cost and a priority, and
acquiring one evicts lower-priority components until it genuinely fits.

Two rules make this safe rather than optimistic:

* Never load on hope. A component is admitted only when free memory minus the
  reserve can actually hold it. If eviction cannot make room, the request is
  refused and the caller degrades -- a missing capability is recoverable, an
  OOM is not.
* Never evict under someone's feet. A component in use is pinned for the
  duration, so a long synthesis cannot have its weights pulled mid-utterance.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_MEMINFO = "/proc/meminfo"


def _meminfo_gib(field_name: str) -> float:
    try:
        with open(_MEMINFO, encoding="utf-8") as handle:
            for line in handle:
                key, _, value = line.partition(":")
                if key == field_name:
                    return float(value.strip().split()[0]) / 1048576
    except OSError:
        pass
    return 0.0


def total_memory_gib() -> float:
    return _meminfo_gib("MemTotal")


def available_memory_gib() -> float:
    """Memory the kernel believes can be handed out without swapping.

    MemAvailable, not MemFree: page cache is reclaimable and counting it as
    used would refuse loads that would actually succeed.
    """

    return _meminfo_gib("MemAvailable")


class ResidencyRefused(RuntimeError):
    """Raised when a component cannot be made to fit within the budget.

    Callers treat this the way they treat an unavailable adapter: degrade to
    whatever path does fit, never retry into an OOM.
    """


@dataclass
class Component:
    """One heavy set of weights the manager can load and unload."""

    name: str
    # Measured resident cost, in GiB. Deliberately measured rather than taken
    # from the file size: on Tegra the GPU allocation is not the weight file.
    cost_gib: float
    load: Callable[[], Awaitable[None]]
    unload: Callable[[], Awaitable[None]]
    is_loaded: Callable[[], Awaitable[bool]]
    # Higher survives eviction longer. The component that is most expensive to
    # reload should outrank the ones that are cheap to bring back.
    priority: int = 0
    load_timeout_seconds: float = 600.0

    _in_use: int = field(default=0, init=False)
    _loaded_at: float = field(default=0.0, init=False)
    _last_used: float = field(default=0.0, init=False)

    @property
    def pinned(self) -> bool:
        return self._in_use > 0


class WeightResidencyManager:
    """Admit heavy components only when they demonstrably fit."""

    def __init__(
        self,
        *,
        reserve_gib: float = 3.0,
        total_gib: float | None = None,
    ) -> None:
        # Headroom the manager will not spend. The OS, page cache, and the
        # companion's own allocations move around underneath us; without a
        # reserve, a load that exactly fits at admission time is an OOM a
        # second later.
        self.reserve_gib = reserve_gib
        self.total_gib = total_gib if total_gib is not None else total_memory_gib()
        self._components: dict[str, Component] = {}
        self._lock = asyncio.Lock()

    def register(self, component: Component) -> None:
        self._components[component.name] = component

    def status(self) -> dict[str, object]:
        return {
            "total_gib": round(self.total_gib, 2),
            "available_gib": round(available_memory_gib(), 2),
            "reserve_gib": self.reserve_gib,
            "components": {
                name: {
                    "cost_gib": item.cost_gib,
                    "priority": item.priority,
                    "pinned": item.pinned,
                    "in_use": item._in_use,
                    "last_used": item._last_used,
                }
                for name, item in self._components.items()
            },
        }

    async def _loaded_components(self) -> list[Component]:
        loaded = []
        for component in self._components.values():
            try:
                if await component.is_loaded():
                    loaded.append(component)
            except Exception as error:  # noqa: BLE001 - probing must not raise
                logger.debug("residency probe failed for %s: %s", component.name, error)
        return loaded

    async def _make_room(self, target: Component) -> None:
        """Evict until ``target`` fits, or refuse.

        Eviction order is lowest priority first, then least recently used, so
        the component that is cheapest to bring back goes first.
        """

        needed = target.cost_gib + self.reserve_gib
        if available_memory_gib() >= needed:
            return

        candidates = [
            item
            for item in await self._loaded_components()
            if item.name != target.name and not item.pinned
        ]
        candidates.sort(key=lambda item: (item.priority, item._last_used))

        # Refuse before evicting anything when even a full sweep could not
        # make room. Otherwise a request that was always impossible tears down
        # perfectly good components on its way to failing anyway -- which is
        # strictly worse than having declined it up front.
        reclaimable = sum(item.cost_gib for item in candidates)
        if available_memory_gib() + reclaimable < needed:
            pinned = [item.name for item in self._components.values() if item.pinned]
            raise ResidencyRefused(
                f"{target.name} needs {target.cost_gib:.1f} GiB plus a "
                f"{self.reserve_gib:.1f} GiB reserve; only "
                f"{available_memory_gib():.1f} GiB is available and at most "
                f"{reclaimable:.1f} GiB can be reclaimed"
                + (f" (pinned: {', '.join(pinned)})" if pinned else "")
            )

        for candidate in candidates:
            logger.info(
                "residency: evicting %s (%.1f GiB) to admit %s (%.1f GiB); "
                "available %.1f GiB, need %.1f GiB",
                candidate.name,
                candidate.cost_gib,
                target.name,
                target.cost_gib,
                available_memory_gib(),
                needed,
            )
            try:
                await candidate.unload()
            except Exception as error:  # noqa: BLE001 - eviction is best effort
                logger.warning("residency: could not unload %s: %s", candidate.name, error)
                continue
            # Releasing GPU memory is not instantaneous on Tegra; give the
            # driver a moment before judging whether it worked.
            for _ in range(20):
                if available_memory_gib() >= needed:
                    return
                await asyncio.sleep(0.5)

        if available_memory_gib() < needed:
            pinned = [item.name for item in self._components.values() if item.pinned]
            raise ResidencyRefused(
                f"{target.name} needs {target.cost_gib:.1f} GiB plus a "
                f"{self.reserve_gib:.1f} GiB reserve, but only "
                f"{available_memory_gib():.1f} GiB is available"
                + (f" (pinned: {', '.join(pinned)})" if pinned else "")
            )

    @contextlib.asynccontextmanager
    async def require(self, name: str) -> AsyncIterator[Component]:
        """Ensure ``name`` is resident for the duration of the block.

        The component is pinned while the block runs, so a concurrent request
        for something larger cannot evict it mid-use.
        """

        component = self._components.get(name)
        if component is None:
            raise ResidencyRefused(f"no such component: {name}")

        async with self._lock:
            if not await component.is_loaded():
                await self._make_room(component)
                logger.info(
                    "residency: loading %s (%.1f GiB); available %.1f GiB",
                    component.name,
                    component.cost_gib,
                    available_memory_gib(),
                )
                try:
                    await asyncio.wait_for(
                        component.load(), timeout=component.load_timeout_seconds
                    )
                except asyncio.TimeoutError as error:
                    raise ResidencyRefused(
                        f"{component.name} did not load within "
                        f"{component.load_timeout_seconds:g}s"
                    ) from error
                component._loaded_at = time.monotonic()
            component._in_use += 1
            component._last_used = time.monotonic()

        try:
            yield component
        finally:
            async with self._lock:
                component._in_use = max(0, component._in_use - 1)
                component._last_used = time.monotonic()


# -- component factories -------------------------------------------------


def systemd_component(
    name: str,
    unit: str,
    cost_gib: float,
    *,
    priority: int = 0,
    ready: Callable[[], Awaitable[bool]] | None = None,
    load_timeout_seconds: float = 600.0,
) -> Component:
    """A component whose lifetime is a systemd user unit.

    ``ready`` is the real readiness test -- an active unit is not the same as
    a loaded model, and admitting one as the other is how a caller ends up
    talking to a port that is listening but not serving.
    """

    async def systemctl(*args: str) -> int:
        if shutil.which("systemctl") is None:
            return 1
        process = await asyncio.create_subprocess_exec(
            "systemctl", "--user", *args,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        return await process.wait()

    async def load() -> None:
        await systemctl("start", unit)
        if ready is None:
            return
        while not await ready():
            await asyncio.sleep(2)

    async def unload() -> None:
        await systemctl("stop", unit)

    async def is_loaded() -> bool:
        if ready is not None:
            return await ready()
        return await systemctl("is-active", "--quiet", unit) == 0

    return Component(
        name=name,
        cost_gib=cost_gib,
        load=load,
        unload=unload,
        is_loaded=is_loaded,
        priority=priority,
        load_timeout_seconds=load_timeout_seconds,
    )
