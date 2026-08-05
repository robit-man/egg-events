from __future__ import annotations

import asyncio

from egg_companion.config import AudioConfig


class Speaker:
    def __init__(self, config: AudioConfig) -> None:
        self.config = config

    async def play_wav(self, audio: bytes) -> None:
        process = await asyncio.create_subprocess_exec(
            "aplay",
            "-q",
            "-D",
            self.config.output_device,
            "-",
            stdin=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(process.communicate(audio), timeout=30)
        except asyncio.TimeoutError as error:
            process.kill()
            await process.wait()
            raise RuntimeError("audio playback timed out") from error
        exit_code = process.returncode
        if exit_code:
            detail = stderr.decode("utf-8", errors="replace").strip() if stderr else ""
            raise RuntimeError(f"audio playback exited with {exit_code}: {detail}")
