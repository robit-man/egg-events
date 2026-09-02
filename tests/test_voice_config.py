import asyncio
from types import SimpleNamespace

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


def test_update_model_selection_sets_both_conversational_and_vision_roles() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(degraded_config())

        async def model_catalog():
            return [{"name": "some/other-model:9b"}]

        runtime._omnius.model_catalog = model_catalog  # type: ignore[method-assign]

        await runtime.update_model_selection("some/other-model:9b")

        assert runtime.config.omnius.model == "some/other-model:9b"
        assert runtime.config.omnius.vision_model == "some/other-model:9b"

    asyncio.run(scenario())


def test_update_model_selection_rejects_a_model_not_in_the_local_catalog() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(degraded_config())
        original_model = runtime.config.omnius.model
        original_vision_model = runtime.config.omnius.vision_model

        async def model_catalog():
            return [{"name": "some/other-model:9b"}]

        runtime._omnius.model_catalog = model_catalog  # type: ignore[method-assign]

        raised = False
        try:
            await runtime.update_model_selection("not-pulled/nonexistent:1b")
        except ValueError:
            raised = True

        assert raised
        assert runtime.config.omnius.model == original_model
        assert runtime.config.omnius.vision_model == original_vision_model

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


def test_join_utterance_texts_reads_as_separate_sentences() -> None:
    join = CompanionRuntime._join_utterance_texts
    assert join(["only one"]) == "only one"
    assert join(["wait", "no check the news instead"]) == "wait. no check the news instead"
    assert join(["already ends here.", "next fragment"]) == "already ends here. next fragment"
    assert join(["is that right?", "yes"]) == "is that right? yes"


def test_queued_burst_of_utterances_is_consolidated_into_one_reasoning_turn() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(degraded_config())
        handled: list[AudioTurn] = []
        finished = asyncio.Event()

        async def handle(turn: AudioTurn) -> None:
            handled.append(turn)
            finished.set()

        runtime._handle_audio_turn = handle  # type: ignore[method-assign]
        first = AudioTurn("one", 1, "wait", 1.0, 1.5, barge_id=None)
        second = AudioTurn("two", 2, "no check the news", 1.5, 2.0, barge_id="barge-1")
        third = AudioTurn("three", 3, "instead", 2.0, 2.5, barge_id="barge-1")
        for utterance_id in ("one", "two", "three"):
            runtime._turn_visual_snapshots[utterance_id] = object()  # type: ignore[assignment]
            runtime._turn_acoustic_context[utterance_id] = {"dummy": True}

        # Queue all three before the reasoning loop gets a chance to run
        # get() -- asyncio.create_task doesn't start executing until the
        # caller yields, so this reliably reproduces "already queued when
        # the loop wakes up" without needing to fake cancellation timing.
        runtime._utterances.put_nowait(first)
        runtime._utterances.put_nowait(second)
        runtime._utterances.put_nowait(third)
        worker = asyncio.create_task(runtime._reason_about_transcript())
        await asyncio.wait_for(finished.wait(), timeout=1)

        assert len(handled) == 1
        merged = handled[0]
        assert merged.text == "wait. no check the news. instead"
        assert merged.revision == 3
        assert merged.utterance_id == "three"
        assert merged.ended_at == 2.5
        assert merged.barge_id == "barge-1"
        assert merged.started_at == 1.0

        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        assert runtime._turn_visual_snapshots == {}
        assert runtime._turn_acoustic_context == {}

    asyncio.run(scenario())


def test_queue_text_turn_enqueues_text_origin_turn() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(degraded_config())

        utterance_id = runtime.queue_text_turn("what did you see")

        queued = runtime._utterances.get_nowait()
        assert queued.origin == "text"
        assert queued.text == "what did you see"
        assert queued.utterance_id == utterance_id
        assert queued.barge_id is None

    asyncio.run(scenario())


def test_queue_text_turn_writes_heard_transcript_evidence() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(degraded_config())
        runtime._memory = object()  # bypass the "memory disabled" guard

        utterance_id = runtime.queue_text_turn("what did you see")

        event = runtime._memory_events.get_nowait()
        assert event.event_type == "speech"
        assert event.payload["transcript"] == "what did you see"
        assert event.evidence[0].modality == "speech"
        assert event.evidence[0].metadata["transcript"] == "what did you see"
        assert event.evidence[0].metadata["context_id"] == utterance_id
        assert event.evidence[0].metadata["utterance_id"] == utterance_id

    asyncio.run(scenario())


def test_queue_text_turn_rejects_empty_text() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(degraded_config())

        raised = False
        try:
            runtime.queue_text_turn("   ")
        except ValueError:
            raised = True

        assert raised
        assert runtime._utterances.empty()

    asyncio.run(scenario())


def test_deliver_reply_routes_text_origin_without_speak() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(degraded_config())

        async def failing_speak(*_args, **_kwargs):
            raise AssertionError("text-origin replies must never call _speak")

        runtime._speak = failing_speak  # type: ignore[method-assign]

        delivered = await runtime._deliver_reply(
            "text", "here is your answer", expected_revision=0
        )

        assert delivered

    asyncio.run(scenario())


def test_deliver_reply_routes_voice_origin_through_speak() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(degraded_config())
        called: list[tuple[str, int | None]] = []

        async def fake_speak(text, expected_revision=None):
            called.append((text, expected_revision))
            return True

        runtime._speak = fake_speak  # type: ignore[method-assign]

        delivered = await runtime._deliver_reply("voice", "hello", expected_revision=0)

        assert delivered
        assert called == [("hello", 0)]

    asyncio.run(scenario())


def test_deliver_text_reply_never_synthesizes_or_plays_audio() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(degraded_config())

        async def failing_synthesize(_text: str) -> bytes:
            raise AssertionError("text replies must not synthesize audio")

        class FailingSpeaker:
            is_playing = False

            async def play_wav(self, *_args, **_kwargs):
                raise AssertionError("text replies must not reach the speaker")

        runtime._omnius.synthesize = failing_synthesize
        runtime._speaker = FailingSpeaker()

        delivered = await runtime._deliver_text_reply("answer", expected_revision=0)

        assert delivered

    asyncio.run(scenario())


def test_queue_interaction_memory_tags_text_origin_replies() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(degraded_config())
        runtime._memory = SimpleNamespace(retrieval_snapshot=lambda: [])

        runtime._queue_interaction_memory(
            "what did you see", "nothing yet", True, "typed message answered",
            context_id="turn-1", origin="text",
        )

        event = runtime._memory_events.get_nowait()
        assert event.evidence[0].metadata["origin"] == "text"
        assert event.evidence[0].metadata["spoken"] is True

    asyncio.run(scenario())


def test_queue_interaction_memory_defaults_to_voice_origin() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(degraded_config())
        runtime._memory = SimpleNamespace(retrieval_snapshot=lambda: [])

        runtime._queue_interaction_memory(
            "hello", "hi there", True, "conversational reply", context_id="turn-1"
        )

        event = runtime._memory_events.get_nowait()
        assert event.evidence[0].metadata["origin"] == "voice"

    asyncio.run(scenario())


def test_deliver_text_reply_rejects_stale_revision() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(degraded_config())
        runtime._conversation_turns.revision = 5

        delivered = await runtime._deliver_text_reply("stale answer", expected_revision=1)

        assert not delivered

    asyncio.run(scenario())
