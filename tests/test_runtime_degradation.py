import asyncio

from egg_companion.config import EggConfig
from egg_companion.runtime import CompanionRuntime


def degraded_config() -> EggConfig:
    return EggConfig.model_validate(
        {
            "audio": {"input_device": "default", "doa_mode": "disabled"},
            "omnius": {"model": "test", "voice_model": "test"},
            "identity": {"enabled": False},
            "object_learning": {"enabled": False},
            "memory": {"enabled": False},
            "camera_discovery": {"enabled": False},
        }
    )


def test_runtime_construction_does_not_require_vision_or_any_camera() -> None:
    runtime = CompanionRuntime(degraded_config())

    assert runtime._vision is None
    assert runtime.telemetry.snapshot(runtime.config)["cameras"] == []


def test_failed_component_retries_without_stopping_supervisor() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(degraded_config())
        calls = 0
        recovered = asyncio.Event()

        async def component() -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("transient failure")
            recovered.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(runtime._run_component("test-component", component))
        try:
            await asyncio.wait_for(recovered.wait(), timeout=2)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        assert calls == 2
        assert runtime.telemetry.snapshot(runtime.config)["runtime_errors"][-1][
            "component"
        ] == "test-component"

    asyncio.run(scenario())
