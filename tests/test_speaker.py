import asyncio
import io
import wave

from egg_companion.adapters.speaker import Speaker
from egg_companion.config import AudioConfig


def _wav(seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    payload = io.BytesIO()
    with wave.open(payload, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(round(seconds * sample_rate) * 2))
    return payload.getvalue()


def test_wav_tail_slice_does_not_restart_from_frame_zero() -> None:
    audio = _wav(1.0)

    tail = Speaker.slice_wav(audio, 0.4)

    assert 0.59 <= Speaker.wav_duration(tail) <= 0.61
    assert len(tail) < len(audio)


def test_playback_interruption_is_identity_bound_and_resumable(monkeypatch) -> None:
    class FakeProcess:
        def __init__(self, wait_for_interrupt: bool) -> None:
            self.returncode = None
            self.wait_for_interrupt = wait_for_interrupt
            self.released = asyncio.Event()
            self.payload = b""

        async def communicate(self, payload: bytes):
            self.payload = payload
            if self.wait_for_interrupt:
                await self.released.wait()
            self.returncode = -15 if self.wait_for_interrupt else 0
            return b"", b""

        def terminate(self) -> None:
            self.returncode = -15
            self.released.set()

        def kill(self) -> None:
            self.returncode = -9
            self.released.set()

        async def wait(self) -> int:
            await self.released.wait()
            return int(self.returncode or 0)

    async def scenario() -> None:
        processes: list[FakeProcess] = []

        async def create_process(*_args, **_kwargs):
            process = FakeProcess(wait_for_interrupt=not processes)
            processes.append(process)
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
        speaker = Speaker(
            AudioConfig(
                input_device="default",
                playback_resume_rewind_ms=0,
            )
        )
        original = _wav(1.0)
        playing = asyncio.create_task(
            speaker.play_wav(original, playback_id="playback-1")
        )
        await asyncio.sleep(0.01)

        assert await speaker.interrupt("stale-playback") is None
        interrupted = await speaker.interrupt("playback-1")
        first_result = await playing

        assert interrupted is not None
        assert first_result.outcome == "interrupted"
        assert speaker.paused_playback_id == "playback-1"

        resumed = await speaker.resume("playback-1")

        assert resumed is not None
        assert resumed.outcome == "completed"
        assert len(processes) == 2
        assert len(processes[1].payload) < len(original)

    asyncio.run(scenario())
