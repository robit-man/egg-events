"""Tests for the memory-bounded residency manager.

The manager's first duty is that the machine never over-commits, so most of
these assert refusal and eviction rather than success.
"""

from __future__ import annotations

import asyncio

import pytest

from egg_companion.services import residency
from egg_companion.services.residency import (
    Component,
    ResidencyRefused,
    WeightResidencyManager,
)


def _component(name: str, cost: float, state: dict[str, bool], **overrides) -> Component:
    async def load() -> None:
        state[name] = True

    async def unload() -> None:
        state[name] = False

    async def is_loaded() -> bool:
        return state.get(name, False)

    return Component(
        name=name, cost_gib=cost, load=load, unload=unload,
        is_loaded=is_loaded, **overrides,
    )


def _manager(monkeypatch, available: float, **kwargs) -> WeightResidencyManager:
    """A manager over a simulated memory pool that shrinks as things load."""

    monkeypatch.setattr(residency, "available_memory_gib", lambda: available)
    return WeightResidencyManager(total_gib=30.0, settle_seconds=0, **kwargs)


def test_a_component_that_fits_is_loaded(monkeypatch) -> None:
    state: dict[str, bool] = {}
    manager = _manager(monkeypatch, available=25.0, reserve_gib=3.0)
    manager.register(_component("comprehension", 16.8, state))

    async def scenario() -> None:
        async with manager.require("comprehension"):
            assert state["comprehension"] is True

    asyncio.run(scenario())


def test_a_component_that_cannot_fit_is_refused_not_attempted(monkeypatch) -> None:
    """Refusing is recoverable; an OOM on unified memory is not."""

    state: dict[str, bool] = {}
    manager = _manager(monkeypatch, available=4.0, reserve_gib=3.0)
    manager.register(_component("comprehension", 16.8, state))

    async def scenario() -> None:
        async with manager.require("comprehension"):
            pass

    with pytest.raises(ResidencyRefused, match="16.8 GiB"):
        asyncio.run(scenario())
    # The load must never have been attempted.
    assert state.get("comprehension") is not True


def test_the_reserve_is_never_spent(monkeypatch) -> None:
    state: dict[str, bool] = {}
    # 18 GiB free would fit a 16.8 GiB component, but not with a 3 GiB reserve.
    manager = _manager(monkeypatch, available=18.0, reserve_gib=3.0)
    manager.register(_component("comprehension", 16.8, state))

    async def scenario() -> None:
        async with manager.require("comprehension"):
            pass

    with pytest.raises(ResidencyRefused):
        asyncio.run(scenario())


def test_a_lower_priority_component_is_evicted_to_make_room(monkeypatch) -> None:
    state = {"language": True}
    pool = {"free": 6.0}
    monkeypatch.setattr(residency, "available_memory_gib", lambda: pool["free"])
    manager = WeightResidencyManager(total_gib=30.0, reserve_gib=3.0, settle_seconds=0)

    async def unload_language() -> None:
        state["language"] = False
        pool["free"] += 5.6

    language = _component("language", 5.6, state, priority=0)
    language.unload = unload_language
    manager.register(language)
    manager.register(_component("comprehension", 8.0, state, priority=10))

    async def scenario() -> None:
        async with manager.require("comprehension"):
            assert state["language"] is False
            assert state["comprehension"] is True

    asyncio.run(scenario())


def test_a_pinned_component_is_never_evicted(monkeypatch) -> None:
    """Weights must not be pulled out from under an in-flight utterance."""

    state = {"tts": True}
    monkeypatch.setattr(residency, "available_memory_gib", lambda: 5.0)
    manager = WeightResidencyManager(total_gib=30.0, reserve_gib=3.0, settle_seconds=0)
    manager.register(_component("tts", 4.0, state, priority=0))
    manager.register(_component("comprehension", 16.8, state, priority=10))

    async def scenario() -> None:
        async with manager.require("tts"):
            # tts is pinned here, so admitting comprehension must fail rather
            # than evict it.
            with pytest.raises(ResidencyRefused, match="pinned: tts"):
                async with manager.require("comprehension"):
                    pass
            assert state["tts"] is True

    asyncio.run(scenario())


def test_eviction_prefers_the_cheapest_to_bring_back(monkeypatch) -> None:
    state = {"low": True, "high": True}
    pool = {"free": 4.0}
    monkeypatch.setattr(residency, "available_memory_gib", lambda: pool["free"])
    manager = WeightResidencyManager(total_gib=30.0, reserve_gib=1.0, settle_seconds=0)
    evicted: list[str] = []

    def make(name: str, priority: int, frees: float) -> Component:
        async def unload() -> None:
            evicted.append(name)
            state[name] = False
            pool["free"] += frees

        component = _component(name, frees, state, priority=priority)
        component.unload = unload
        return component

    manager.register(make("low", 0, 3.0))
    manager.register(make("high", 10, 3.0))
    manager.register(_component("target", 5.0, state, priority=5))

    async def scenario() -> None:
        async with manager.require("target"):
            pass

    asyncio.run(scenario())
    # Only the low-priority one was needed, and it went first.
    assert evicted == ["low"]
    assert state["high"] is True


def test_an_already_loaded_component_is_not_reloaded(monkeypatch) -> None:
    state = {"comprehension": True}
    loads = {"count": 0}
    monkeypatch.setattr(residency, "available_memory_gib", lambda: 1.0)
    manager = WeightResidencyManager(total_gib=30.0, reserve_gib=3.0, settle_seconds=0)

    async def load() -> None:
        loads["count"] += 1

    component = _component("comprehension", 16.8, state)
    component.load = load
    manager.register(component)

    async def scenario() -> None:
        # Available memory is far below the cost, but it is already resident,
        # so no admission check applies.
        async with manager.require("comprehension"):
            pass

    asyncio.run(scenario())
    assert loads["count"] == 0


def test_a_load_that_hangs_is_refused_rather_than_waited_on_forever(monkeypatch) -> None:
    state: dict[str, bool] = {}
    monkeypatch.setattr(residency, "available_memory_gib", lambda: 25.0)
    manager = WeightResidencyManager(total_gib=30.0, reserve_gib=3.0, settle_seconds=0)

    async def never_finishes() -> None:
        await asyncio.sleep(60)

    component = _component("comprehension", 16.8, state, load_timeout_seconds=0.2)
    component.load = never_finishes
    manager.register(component)

    async def scenario() -> None:
        async with manager.require("comprehension"):
            pass

    with pytest.raises(ResidencyRefused, match="did not load"):
        asyncio.run(scenario())


def test_an_unknown_component_is_refused(monkeypatch) -> None:
    manager = _manager(monkeypatch, available=25.0)

    async def scenario() -> None:
        async with manager.require("nonexistent"):
            pass

    with pytest.raises(ResidencyRefused, match="no such component"):
        asyncio.run(scenario())


def test_status_reports_the_budget_and_components(monkeypatch) -> None:
    state: dict[str, bool] = {}
    manager = _manager(monkeypatch, available=12.0, reserve_gib=3.0)
    manager.register(_component("comprehension", 16.8, state, priority=10))

    status = manager.status()

    assert status["total_gib"] == 30.0
    assert status["available_gib"] == 12.0
    assert status["reserve_gib"] == 3.0
    assert status["components"]["comprehension"]["cost_gib"] == 16.8
    assert status["components"]["comprehension"]["pinned"] is False


def test_memory_readings_come_from_meminfo() -> None:
    """MemAvailable, not MemFree: page cache is reclaimable."""

    total = residency.total_memory_gib()
    available = residency.available_memory_gib()

    assert total > 0
    assert 0 <= available <= total


def test_an_impossible_request_evicts_nothing(monkeypatch) -> None:
    """A request that could never fit must not tear down good components."""

    state = {"comprehension": True}
    monkeypatch.setattr(residency, "available_memory_gib", lambda: 4.0)
    manager = WeightResidencyManager(total_gib=30.0, reserve_gib=3.0, settle_seconds=0)
    evicted: list[str] = []

    async def unload() -> None:
        evicted.append("comprehension")
        state["comprehension"] = False

    resident = _component("comprehension", 16.8, state, priority=10)
    resident.unload = unload
    manager.register(resident)
    # Larger than the whole machine, even after reclaiming everything.
    manager.register(_component("oversized", 99.0, state, priority=0))

    async def scenario() -> None:
        async with manager.require("oversized"):
            pass

    with pytest.raises(ResidencyRefused, match="can be reclaimed"):
        asyncio.run(scenario())
    assert evicted == []
    assert state["comprehension"] is True


def test_an_idle_component_is_released(monkeypatch) -> None:
    """Holding weights while idle is indistinguishable from a leak."""

    state = {"comprehension": True}
    monkeypatch.setattr(residency, "available_memory_gib", lambda: 4.0)
    manager = WeightResidencyManager(total_gib=30.0, reserve_gib=3.0, settle_seconds=0)
    component = _component(
        "comprehension", 16.8, state, idle_release_seconds=0.05
    )
    manager.register(component)

    async def scenario() -> None:
        async with manager.require("comprehension"):
            pass
        # Still resident immediately after use.
        assert await manager.release_idle() == []
        assert state["comprehension"] is True
        await asyncio.sleep(0.1)
        assert await manager.release_idle() == ["comprehension"]
        assert state["comprehension"] is False

    asyncio.run(scenario())


def test_a_pinned_component_is_never_released_as_idle(monkeypatch) -> None:
    state = {"comprehension": True}
    monkeypatch.setattr(residency, "available_memory_gib", lambda: 25.0)
    manager = WeightResidencyManager(total_gib=30.0, reserve_gib=3.0, settle_seconds=0)
    manager.register(
        _component("comprehension", 16.8, state, idle_release_seconds=0.01)
    )

    async def scenario() -> None:
        async with manager.require("comprehension"):
            await asyncio.sleep(0.05)
            assert await manager.release_idle() == []
            assert state["comprehension"] is True

    asyncio.run(scenario())


def test_a_component_without_an_idle_window_is_held(monkeypatch) -> None:
    """Zero means hold until evicted, for things that manage their own life."""

    state = {"language": True}
    monkeypatch.setattr(residency, "available_memory_gib", lambda: 25.0)
    manager = WeightResidencyManager(total_gib=30.0, reserve_gib=3.0, settle_seconds=0)
    manager.register(_component("language", 15.6, state, idle_release_seconds=0))

    async def scenario() -> None:
        async with manager.require("language"):
            pass
        await asyncio.sleep(0.05)
        assert await manager.release_idle() == []
        assert state["language"] is True

    asyncio.run(scenario())


def test_memory_released_during_the_settle_window_avoids_eviction(monkeypatch) -> None:
    """A worker that just exited frees its pages asynchronously on Tegra."""

    state = {"language": True}
    readings = iter([4.0, 4.0, 25.0, 25.0, 25.0])
    monkeypatch.setattr(
        residency, "available_memory_gib", lambda: next(readings, 25.0)
    )
    manager = WeightResidencyManager(
        total_gib=30.0, reserve_gib=3.0, settle_seconds=2.0
    )
    evicted: list[str] = []

    async def unload() -> None:
        evicted.append("language")

    language = _component("language", 15.6, state, priority=0)
    language.unload = unload
    manager.register(language)
    manager.register(_component("comprehension", 16.7, state, priority=10))

    async def scenario() -> None:
        async with manager.require("comprehension"):
            pass

    asyncio.run(scenario())
    # The memory arrived while settling, so nothing had to be torn down.
    assert evicted == []
    assert state["language"] is True
