import asyncio

from egg_companion.cognition.conversation import AudioTurn
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


def test_matching_asr_selection_is_still_reconciled_with_backend() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(degraded_config())
        requested: list[str] = []

        async def ensure_asr_model(model_id: str) -> None:
            requested.append(model_id)

        runtime._omnius.ensure_asr_model = ensure_asr_model
        current = runtime.config.transcription.asr_model

        await runtime.update_voice_config(None, None, None, None, current)

        assert requested == [current]

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


def test_pending_heard_audio_prevents_stale_playback_publication() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(degraded_config())
        playback_started = False

        async def synthesize(_text: str) -> bytes:
            return b"RIFF speculative audio"

        class FailingIfPlayedSpeaker:
            is_playing = False

            async def play_wav(self, *_args, **_kwargs):
                nonlocal playback_started
                playback_started = True
                raise AssertionError("pending ingress must block speaker publication")

        runtime._omnius.synthesize = synthesize
        runtime._speaker = FailingIfPlayedSpeaker()
        runtime._conversation_turns.speech_started()

        spoken = await runtime._speak("This response is already obsolete.", expected_revision=0)

        assert not spoken
        assert not playback_started

    asyncio.run(scenario())


def test_live_voice_config_updates_asr_normalization_and_vad_gain() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(degraded_config())

        await runtime.update_voice_config(
            None, None, None, None, None,
            asr_target_rms=0.12,
            asr_max_gain=48,
            vad_input_gain=2.5,
            asr_language="en",
        )

        assert runtime.config.audio.asr_target_rms == 0.12
        assert runtime.config.audio.asr_max_gain == 48
        assert runtime.config.transcription.vad_input_gain == 2.5
        assert runtime.config.transcription.asr_language == "en"
        assert runtime._capture.audio.asr_target_rms == 0.12
        assert runtime._segmenter.transcription.vad_input_gain == 2.5

    asyncio.run(scenario())


def test_superseded_reasoning_is_not_mistaken_for_component_shutdown_on_python_310() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(degraded_config())
        first_started = asyncio.Event()
        second_finished = asyncio.Event()

        async def handle(turn: AudioTurn) -> None:
            if turn.revision == 1:
                first_started.set()
                await asyncio.Event().wait()
            second_finished.set()

        runtime._handle_audio_turn = handle  # type: ignore[method-assign]
        worker = asyncio.create_task(runtime._reason_about_transcript())
        first = AudioTurn("one", 1, "first", 1.0, 2.0)
        second = AudioTurn("two", 2, "second", 2.0, 3.0)
        runtime._utterances.put_nowait(first)
        await asyncio.wait_for(first_started.wait(), timeout=1)

        assert runtime._cancel_stale_reasoning(second.revision)
        runtime._utterances.put_nowait(second)
        await asyncio.wait_for(second_finished.wait(), timeout=1)

        assert not worker.done()
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(scenario())
