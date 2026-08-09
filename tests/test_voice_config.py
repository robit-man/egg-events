import asyncio

from egg_companion.config import EggConfig
from egg_companion.runtime import CompanionRuntime


def degraded_config() -> EggConfig:
    return EggConfig.model_validate(
        {
            "audio": {"input_device": "default", "doa_mode": "disabled"},
            "omnius": {"model": "test", "voice_model": "supertonic"},
            "identity": {"enabled": False},
            "object_learning": {"enabled": False},
            "memory": {"enabled": False},
            "camera_discovery": {"enabled": False},
        }
    )


def test_rejected_asr_switch_does_not_mutate_config() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(degraded_config())
        original_model = runtime.config.transcription.asr_model

        async def failing_ensure_asr_model(model_id: str) -> None:
            raise RuntimeError("Omnius ASR model switch HTTP 400: rejected")

        runtime._omnius.ensure_asr_model = failing_ensure_asr_model

        raised = False
        try:
            await runtime.update_voice_config(None, None, None, None, "some-other-model")
        except RuntimeError:
            raised = True

        assert raised
        assert runtime.config.transcription.asr_model == original_model

    asyncio.run(scenario())


def test_rejected_voice_model_switch_does_not_mutate_config() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(degraded_config())
        original_voice_model = runtime.config.omnius.voice_model

        async def failing_ensure_voice_ready(model_id: str | None = None) -> None:
            raise RuntimeError("Omnius voice model switch HTTP 400: rejected")

        runtime._omnius.ensure_voice_ready = failing_ensure_voice_ready

        raised = False
        try:
            await runtime.update_voice_config(None, None, "some-other-voice", None, None)
        except RuntimeError:
            raised = True

        assert raised
        assert runtime.config.omnius.voice_model == original_voice_model

    asyncio.run(scenario())
