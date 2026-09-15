"""Client for the Qwen Omni adapter (``robit.ollama.omni-adapter.v1``).

The adapter fronts one logical Ollama tag -- ``robit/ornith-1.5-omni:q4km`` --
and routes a single Ollama-shaped request through Qwen3-Omni comprehension,
the Ornith language graph, and Qwen3-TTS. Egg uses it for the perceptual
stages that its existing Omnius backend answers less precisely:

* **Speech versus sound.** One comprehension pass returns the verbatim
  transcript and the non-speech acoustic observation as *separately tagged*
  evidence. Egg already refuses to let room noise become something the user
  said; the adapter enforces that split at the model boundary instead of
  leaving it to a downstream heuristic.
* **Environmental audio.** A tagged observation describes ambience, activity,
  and timing in language, which the YAMNet classifier -- a fixed 521-class
  taxonomy with no temporal structure -- cannot express.
* **Video.** Bounded clip understanding with frame sampling, which has no
  equivalent on the current path at all.
* **Speech output.** Qwen3-TTS at 24 kHz, including a request-local voice
  reference for cloning.

Every method here is optional by construction. The adapter runs as a separate
supervised process; when it is absent, unhealthy, or slow, callers fall back
to the existing Omnius paths rather than losing the capability. That is why
this module owns a health gate with a failure cooldown: a realtime companion
must not pay an adapter connection timeout on every spoken turn while the
service is down.

See https://github.com/robit-man/qwen-omni-adapters for the wire contract.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import re
import time
import wave
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np

from egg_companion.config import OmniAdapterConfig

logger = logging.getLogger(__name__)

ADAPTER_SCHEMA = "robit.ollama.omni-adapter.v1"

# The adapter's audio contract: 16 kHz mono PCM16 in, 24 kHz mono PCM16 out.
INPUT_SAMPLE_RATE_HZ = 16000
INPUT_CHANNELS = 1
INPUT_SAMPLE_WIDTH_BYTES = 2
OUTPUT_SAMPLE_RATE_HZ = 24000

_SPEECH_TRANSCRIPT_BLOCK = re.compile(
    r"<speech_transcript>(.*?)</speech_transcript>", re.IGNORECASE | re.DOTALL
)
_AUDIO_OBSERVATION_BLOCK = re.compile(
    r"<audio_observation>(.*?)</audio_observation>", re.IGNORECASE | re.DOTALL
)
_VISUAL_OBSERVATION_BLOCK = re.compile(
    r"<visual_observation>(.*?)</visual_observation>", re.IGNORECASE | re.DOTALL
)

# Asking one comprehension pass for both channels is what makes the split
# authoritative: the model decides what was speech, and everything it did not
# attribute to a speaker stays out of the transcript by construction.
_AUDIO_PERCEPTION_INSTRUCTION = (
    "Analyze the supplied audio. Output exactly two XML elements and nothing "
    "else: <speech_transcript>verbatim speech, or empty if no speech is "
    "intelligible</speech_transcript><audio_observation>objective non-speech "
    "sounds, ambience, music, speaker activity, temporal changes, and "
    "uncertainty; do not repeat the transcript</audio_observation>. Do not "
    "answer the speech."
)


class OmniAdapterError(RuntimeError):
    """Raised when the Omni adapter cannot answer a request."""


class OmniAdapterUnavailable(OmniAdapterError):
    """Raised when the adapter is disabled, unreachable, or in its cooldown.

    Callers treat this as "use the existing path", never as a hard failure.
    """


def normalize_input_wav(wav_audio: bytes) -> bytes:
    """Return ``wav_audio`` as the 16 kHz mono PCM16 WAV the contract requires.

    Egg captures at 16 kHz mono already, so this is normally a no-op check.
    It exists because the adapter rejects a mismatched container outright, and
    a rejected utterance is a lost turn rather than a degraded one.
    """

    if not wav_audio.startswith(b"RIFF"):
        raise OmniAdapterError("audio payload is not a RIFF/WAVE container")
    try:
        with wave.open(io.BytesIO(wav_audio), "rb") as source:
            if source.getcomptype() != "NONE":
                raise OmniAdapterError("audio payload must contain uncompressed PCM")
            channels = source.getnchannels()
            width = source.getsampwidth()
            rate = source.getframerate()
            frames = source.readframes(source.getnframes())
    except wave.Error as error:
        raise OmniAdapterError(f"invalid WAV container: {error}") from error
    if (
        channels == INPUT_CHANNELS
        and width == INPUT_SAMPLE_WIDTH_BYTES
        and rate == INPUT_SAMPLE_RATE_HZ
    ):
        return wav_audio
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise OmniAdapterError(f"unsupported WAV sample width: {width * 8} bits")
    samples = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if width == 1:
        samples = (samples - 128.0) / 128.0
    else:
        samples /= float(1 << (width * 8 - 1))
    if channels > INPUT_CHANNELS:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != INPUT_SAMPLE_RATE_HZ and samples.size:
        target_count = max(1, round(samples.size * INPUT_SAMPLE_RATE_HZ / rate))
        samples = np.interp(
            np.linspace(0.0, 1.0, target_count, endpoint=False),
            np.linspace(0.0, 1.0, samples.size, endpoint=False),
            samples,
        ).astype(np.float32)
    pcm = np.clip(samples * 32767.0, -32768.0, 32767.0).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(INPUT_CHANNELS)
        target.setsampwidth(INPUT_SAMPLE_WIDTH_BYTES)
        target.setframerate(INPUT_SAMPLE_RATE_HZ)
        target.writeframes(pcm.tobytes())
    return output.getvalue()


def _tagged(pattern: re.Pattern[str], text: str) -> str | None:
    blocks = [match.group(1).strip() for match in pattern.finditer(text)]
    joined = "\n".join(block for block in blocks if block)
    return joined or None


class OmniAdapterClient:
    """Speak ``robit.ollama.omni-adapter.v1`` to a local adapter server."""

    def __init__(self, config: OmniAdapterConfig) -> None:
        self.config = config
        self._healthy_until = 0.0
        self._cooldown_until = 0.0
        self._last_error: str | None = None
        self._contract: dict[str, Any] | None = None
        # The adapter admits a bounded number of concurrent GPU lanes and Egg
        # shares its GPU with vision, ASR, and the language model. One in-flight
        # comprehension request at a time keeps an environmental-audio job from
        # queueing behind -- or ahead of -- a spoken turn.
        self._gate = asyncio.Semaphore(max(1, config.max_concurrent_requests))

    # -- plumbing --------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def _base_url(self) -> str:
        return str(self.config.base_url).rstrip("/")

    def _headers(self) -> dict[str, str]:
        if not self.config.bearer_token_env:
            return {}
        token = os.getenv(self.config.bearer_token_env)
        if not token:
            raise OmniAdapterError(
                f"required token environment variable is unset: {self.config.bearer_token_env}"
            )
        return {"Authorization": f"Bearer {token}"}

    def status(self) -> dict[str, object]:
        """Describe the adapter for the dashboard and the startup audit."""

        now = time.monotonic()
        return {
            "enabled": self.enabled,
            "base_url": self._base_url(),
            "model": self.config.model,
            "healthy": self.enabled and now < self._healthy_until,
            "cooling_down": now < self._cooldown_until,
            "last_error": self._last_error,
            "transcription": self.config.transcription_enabled,
            "audio_scene": self.config.audio_scene_enabled,
            "speech": self.config.speech_enabled,
        }

    def _note_failure(self, error: BaseException) -> None:
        """Park the adapter after a failure so turns stop paying its timeout."""

        self._healthy_until = 0.0
        self._cooldown_until = time.monotonic() + self.config.failure_cooldown_seconds
        self._last_error = f"{type(error).__name__}: {error}"

    async def health(self) -> dict[str, object]:
        """Probe the adapter, raising on any failure. Used by the audit."""

        timeout = aiohttp.ClientTimeout(total=self.config.health_timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                f"{self._base_url()}/healthz", headers=self._headers()
            ) as response:
                if response.status >= 400:
                    detail = (await response.text())[:300]
                    raise OmniAdapterError(
                        f"Omni adapter health HTTP {response.status}: {detail}"
                    )
                payload = await response.json()
        if not isinstance(payload, dict):
            raise OmniAdapterError("Omni adapter health is not an object")
        self._healthy_until = time.monotonic() + self.config.health_ttl_seconds
        self._cooldown_until = 0.0
        self._last_error = None
        return payload

    async def contract(self) -> dict[str, Any]:
        """Return the adapter's own versioned wire contract."""

        if self._contract is not None:
            return self._contract
        timeout = aiohttp.ClientTimeout(total=self.config.health_timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                f"{self._base_url()}/api/omni/adapter/contract", headers=self._headers()
            ) as response:
                response.raise_for_status()
                payload = await response.json()
        if not isinstance(payload, dict) or payload.get("schema") != ADAPTER_SCHEMA:
            raise OmniAdapterError(
                f"Omni adapter does not advertise {ADAPTER_SCHEMA}"
            )
        self._contract = payload
        return payload

    async def _ensure_available(self) -> None:
        """Raise ``OmniAdapterUnavailable`` unless the adapter is usable now."""

        if not self.enabled:
            raise OmniAdapterUnavailable("Omni adapter is disabled")
        now = time.monotonic()
        if now < self._healthy_until:
            return
        if now < self._cooldown_until:
            raise OmniAdapterUnavailable(
                f"Omni adapter is in its failure cooldown: {self._last_error}"
            )
        try:
            await self.health()
        except (aiohttp.ClientError, asyncio.TimeoutError, OmniAdapterError, OSError) as error:
            self._note_failure(error)
            raise OmniAdapterUnavailable(
                f"Omni adapter is unreachable: {self._last_error}"
            ) from error

    async def _post(
        self, payload: dict[str, Any], *, timeout_seconds: float
    ) -> dict[str, Any]:
        await self._ensure_available()
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        try:
            async with self._gate:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{self._base_url()}/api/chat",
                        json=payload,
                        headers=self._headers(),
                    ) as response:
                        if response.status >= 400:
                            detail = (await response.text())[:500]
                            raise OmniAdapterError(
                                f"Omni adapter HTTP {response.status}: {detail}"
                            )
                        result = await response.json()
        except asyncio.CancelledError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as error:
            self._note_failure(error)
            raise OmniAdapterUnavailable(
                f"Omni adapter request failed: {self._last_error}"
            ) from error
        if not isinstance(result, dict):
            raise OmniAdapterError("Omni adapter returned a non-object response")
        return result

    def _request(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        response_modalities: list[str] | None = None,
        speech_mode: str = "never",
        speech: dict[str, Any] | None = None,
        require_speech: bool = False,
        include_audio_from_video: bool = True,
        think: bool = False,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        omni: dict[str, Any] = {"schema": ADAPTER_SCHEMA, "task": task}
        if require_speech:
            omni["require_speech"] = True
        if task in {"chat", "describe"}:
            omni["include_audio_from_video"] = include_audio_from_video
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "omni": omni,
            "response_modalities": response_modalities or ["text"],
            "speech_mode": speech_mode,
            # Adapter v1's portable route is non-streaming by contract.
            "stream": False,
            # Egg's perceptual stages are grounding, not deliberation; hidden
            # reasoning here costs a full extra generation before every turn.
            "think": think,
            "keep_alive": self.config.keep_alive,
        }
        if speech:
            payload["speech"] = speech
        if options:
            payload["options"] = options
        return payload

    @staticmethod
    def _message(result: dict[str, Any]) -> dict[str, Any]:
        message = result.get("message")
        if not isinstance(message, dict):
            raise OmniAdapterError("Omni adapter response contains no message object")
        return message

    @staticmethod
    def _adapter_metadata(result: dict[str, Any]) -> dict[str, Any]:
        metadata = result.get("adapter")
        return metadata if isinstance(metadata, dict) else {}

    @staticmethod
    def _audio_envelope(wav_audio: bytes) -> dict[str, str]:
        return {
            "mime_type": "audio/wav",
            "encoding": "base64",
            "data": base64.b64encode(wav_audio).decode("ascii"),
        }

    # -- perception ------------------------------------------------------

    async def perceive_audio(self, wav_audio: bytes) -> dict[str, object]:
        """Return separately tagged speech and non-speech evidence for one clip.

        This is the nuanced replacement for a bare ASR call: a single
        comprehension pass yields ``transcript`` (only what a speaker actually
        said) and ``audio_observation`` (everything else that was audible).
        Neither channel can be silently promoted into the other, which is the
        property Egg's memory pipeline depends on.
        """

        audio = normalize_input_wav(wav_audio)
        payload = self._request(
            task="describe",
            messages=[
                {
                    "role": "user",
                    "content": _AUDIO_PERCEPTION_INSTRUCTION,
                    "audios": [self._audio_envelope(audio)],
                }
            ],
        )
        result = await self._post(payload, timeout_seconds=self.config.timeout_seconds)
        metadata = self._adapter_metadata(result)
        observation = str(
            metadata.get("observation") or self._message(result).get("content") or ""
        ).strip()
        transcript = metadata.get("input_transcript")
        if not isinstance(transcript, str) or not transcript.strip():
            transcript = _tagged(_SPEECH_TRANSCRIPT_BLOCK, observation)
        audio_observation = metadata.get("audio_observation")
        if not isinstance(audio_observation, str) or not audio_observation.strip():
            audio_observation = _tagged(_AUDIO_OBSERVATION_BLOCK, observation)
        if transcript is None and audio_observation is None and observation:
            # An untagged answer describes the clip; it is an acoustic
            # observation, and attributing it to a speaker would be a
            # fabricated transcript.
            audio_observation = observation
        return {
            "transcript": (transcript or "").strip() or None,
            "audio_observation": (audio_observation or "").strip() or None,
            "observation": observation or None,
            "model": self.config.model,
            "backend": "qwen3-omni",
            "route": list(metadata.get("route") or []),
        }

    async def transcribe(self, wav_audio: bytes) -> str | None:
        """Transcribe speech only, with no language or TTS stage."""

        audio = normalize_input_wav(wav_audio)
        payload = self._request(
            task="transcribe",
            messages=[{"role": "user", "content": "", "audios": [self._audio_envelope(audio)]}],
            require_speech=True,
        )
        result = await self._post(payload, timeout_seconds=self.config.timeout_seconds)
        metadata = self._adapter_metadata(result)
        transcript = metadata.get("input_transcript")
        if not isinstance(transcript, str) or not transcript.strip():
            transcript = str(self._message(result).get("content") or "")
        transcript = _tagged(_SPEECH_TRANSCRIPT_BLOCK, transcript) or transcript
        return transcript.strip() or None

    async def describe_audio(self, wav_audio: bytes) -> str | None:
        """Describe a clip's acoustic content as prose, without transcribing it."""

        perceived = await self.perceive_audio(wav_audio)
        observation = perceived.get("audio_observation")
        return str(observation) if observation else None

    async def describe_video(
        self,
        video: bytes,
        *,
        mime_type: str = "video/mp4",
        prompt: str | None = None,
        fps: float | None = None,
        max_frames: int | None = None,
        include_audio: bool | None = None,
    ) -> dict[str, object]:
        """Describe a bounded video clip, optionally using its own audio track."""

        sampling: dict[str, Any] = {}
        resolved_fps = self.config.video_fps if fps is None else fps
        resolved_frames = self.config.video_max_frames if max_frames is None else max_frames
        resolved_audio = (
            self.config.video_include_audio if include_audio is None else include_audio
        )
        if resolved_fps:
            sampling["fps"] = float(resolved_fps)
        if resolved_frames:
            sampling["max_frames"] = int(resolved_frames)
        sampling["include_audio"] = bool(resolved_audio)
        envelope: dict[str, Any] = {
            "mime_type": mime_type,
            "encoding": "base64",
            "data": base64.b64encode(video).decode("ascii"),
            "sampling": sampling,
        }
        payload = self._request(
            task="describe",
            messages=[
                {
                    "role": "user",
                    "content": prompt or "",
                    "videos": [envelope],
                }
            ],
            include_audio_from_video=bool(resolved_audio),
        )
        result = await self._post(
            payload, timeout_seconds=self.config.video_timeout_seconds
        )
        metadata = self._adapter_metadata(result)
        observation = str(
            metadata.get("observation") or self._message(result).get("content") or ""
        ).strip()
        return {
            "observation": observation or None,
            "visual_observation": _tagged(_VISUAL_OBSERVATION_BLOCK, observation)
            or observation
            or None,
            "transcript": metadata.get("input_transcript") or None,
            "audio_observation": metadata.get("audio_observation") or None,
            "model": self.config.model,
            "backend": "qwen3-omni",
        }

    # -- speech ----------------------------------------------------------

    def _speech_settings(self) -> dict[str, Any]:
        speech: dict[str, Any] = {}
        if self.config.voice:
            speech["voice"] = self.config.voice
        if self.config.voice_language:
            speech["language"] = self.config.voice_language
        if self.config.voice_style:
            speech["style"] = self.config.voice_style
        reference = self.config.voice_reference_path
        if reference:
            path = Path(reference).expanduser()
            try:
                raw = path.read_bytes()
            except OSError as error:
                raise OmniAdapterError(
                    f"voice reference is unreadable: {path}: {error}"
                ) from error
            speech["speaker_audio"] = base64.b64encode(
                normalize_input_wav(raw)
            ).decode("ascii")
        return speech

    async def synthesize(self, text: str) -> bytes:
        """Synthesize ``text`` with Qwen3-TTS and return a 24 kHz mono WAV."""

        spoken = text.strip()
        if not spoken:
            raise OmniAdapterError("synthesis requires non-empty text")
        payload = self._request(
            task="synthesize",
            messages=[{"role": "user", "content": spoken}],
            response_modalities=["text", "audio"],
            speech_mode="always",
            speech=self._speech_settings(),
        )
        result = await self._post(
            payload, timeout_seconds=self.config.speech_timeout_seconds
        )
        audio = self._message(result).get("audio")
        if not isinstance(audio, dict) or not audio.get("data"):
            raise OmniAdapterError("Omni adapter returned no synthesized audio")
        try:
            wav_audio = base64.b64decode(str(audio["data"]), validate=True)
        except (ValueError, TypeError) as error:
            raise OmniAdapterError("synthesized audio is not valid base64") from error
        if not wav_audio.startswith(b"RIFF"):
            raise OmniAdapterError("synthesized audio is not a WAV payload")
        return wav_audio
