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
import contextlib
import io
import json
import logging
import os
import re
import time
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np

from egg_companion.config import OmniAdapterConfig
from egg_companion.services.residency import (
    LANGUAGE_COMPONENT as _LANGUAGE_COMPONENT,
    ResidencyRefused,
    WeightResidencyManager,
)

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

    # Which registered component each route needs resident. The names match
    # what the runtime registers with the residency manager.
    COMPREHENSION_COMPONENT = "omni_comprehension"
    SPEECH_COMPONENT = "omni_speech"
    LANGUAGE_COMPONENT = _LANGUAGE_COMPONENT

    def __init__(
        self,
        config: OmniAdapterConfig,
        residency: WeightResidencyManager | None = None,
    ) -> None:
        self.config = config
        # The parent manager that guarantees these weights fit before they are
        # loaded. Without one the client behaves as before and assumes whoever
        # started the adapter sized it.
        self._residency = residency
        self._healthy_until = 0.0
        self._cooldown_until = 0.0
        self._last_error: str | None = None
        self._contract: dict[str, Any] | None = None
        self._voice_profile_cache: tuple[dict[str, Any], Path] | dict[str, Any] | None = None
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
            "mode": self.config.mode,
            "enabled": self.enabled,
            "base_url": self._base_url(),
            "model": self.config.model,
            "healthy": self.enabled and now < self._healthy_until,
            "cooling_down": now < self._cooldown_until,
            "last_error": self._last_error,
            "transcription": self.config.uses_transcription,
            "audio_scene": self.config.uses_audio_scene,
            "video": self.config.uses_video,
            "speech": self.config.uses_speech,
        }

    def _clear_failure_state(self) -> None:
        """Forget an earlier failure so the next call re-checks for itself.

        Health is left unasserted rather than assumed: this clears the
        cooldown, it does not claim the adapter is up.
        """

        self._cooldown_until = 0.0
        self._healthy_until = 0.0

    def _note_failure(self, error: BaseException) -> None:
        """Park the adapter after a failure so turns stop paying its timeout."""

        self._healthy_until = 0.0
        self._cooldown_until = time.monotonic() + self.config.failure_cooldown_seconds
        self._last_error = f"{type(error).__name__}: {error}"

    async def health(self, timeout_seconds: float | None = None) -> dict[str, object]:
        """Probe the adapter, raising on any failure. Used by the audit."""

        timeout = aiohttp.ClientTimeout(
            total=timeout_seconds
            if timeout_seconds is not None
            else self.config.health_timeout_seconds
        )
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

    async def probe(self) -> bool:
        """Ask the adapter directly, ignoring the cached health window.

        Routing deliberately has hysteresis: a healthy TTL keeps turns off
        the timeout path, and a failure cooldown keeps them off an adapter
        known to be down. That is right for choosing where to send work and
        wrong for a status display, which has to show what is true now --
        reusing the cache made the voice page report health for the whole
        TTL after the adapter died, and failure for the whole cooldown after
        it came back.

        It says nothing about whether a worker's weights are resident: those
        are load-on-demand, and a swap in progress is ordinary operation.
        """

        if not self.enabled:
            return False
        try:
            await self.health()
        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            OmniAdapterError,
            OSError,
        ) as error:
            # Deliberately not _note_failure: a status poll must not park the
            # adapter. Letting it open the failure cooldown meant that merely
            # looking at the voice page made the next turn refuse, and while
            # the manager idle-releases this unit as a matter of course, the
            # poll that saw it down would keep it down.
            logger.debug("omni adapter probe failed: %s", error)
            self._last_error = f"{type(error).__name__}: {error}"
            return False
        return True

    async def availability(self, component: str = SPEECH_COMPONENT) -> dict[str, object]:
        """How a status display should describe this stage.

        Three states, because up-or-down is the wrong question for a managed
        component. The residency manager stops the speech unit to reclaim
        memory and starts it again on demand, so an adapter that is not
        answering this second is usually on standby rather than broken. A
        badge that goes red on every idle release teaches the reader to
        ignore it just as surely as one that never goes red at all.
        """

        if not self.enabled:
            return {"state": "disabled", "ready": False, "detail": None}
        if await self.probe():
            return {"state": "ready", "ready": True, "detail": None}
        if self._residency is not None and self._residency.manages(component):
            # Released or still loading. It answers when a turn asks for it;
            # if admission genuinely fails, that turn records the reason here
            # and the next poll reports unavailable.
            return {"state": "standby", "ready": True, "detail": self._last_error}
        return {"state": "unavailable", "ready": False, "detail": self._last_error}

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

    @contextlib.asynccontextmanager
    async def _resident(self, component: str) -> AsyncIterator[None]:
        """Hold ``component`` resident for this request.

        A refusal is surfaced as ``OmniAdapterUnavailable`` so it degrades
        through the same fallback as an unhealthy adapter: not having the
        capability this turn is recoverable, and an over-commit is not.
        """

        if self._residency is None:
            yield
            return
        try:
            async with contextlib.AsyncExitStack() as stack:
                if component != self.SPEECH_COMPONENT and self._residency.manages(
                    self.SPEECH_COMPONENT
                ):
                    # Every task reaches the weights through the adapter
                    # daemon, and that daemon is the unit the speech component
                    # manages. Holding only the comprehension worker let the
                    # manager evict the daemon to make room for it, tearing
                    # down the HTTP server the very same request was about to
                    # use -- which surfaced as the adapter being unreachable
                    # moments after the manager reported a successful load.
                    await stack.enter_async_context(
                        self._residency.require(self.SPEECH_COMPONENT)
                    )
                elif component == self.SPEECH_COMPONENT:
                    # The TTS worker is non-persistent: the service being up
                    # says nothing about whether the worker can spawn. Reserve
                    # the room it actually needs, or fail over before the
                    # attempt rather than after it.
                    await self._residency.ensure_headroom(
                        self.config.speech_headroom_gib, exclude=component
                    )
                await stack.enter_async_context(self._residency.require(component))
                # The manager has just brought this component up, which is
                # newer information than any earlier failure. Without this the
                # cooldown opened by the eviction outlived the reload and the
                # turn was refused against an adapter that was already back.
                self._clear_failure_state()
                yield
        except ResidencyRefused as error:
            logger.info("omni %s is not resident and will not fit: %s", component, error)
            raise OmniAdapterUnavailable(
                f"{component} cannot be made resident: {error}"
            ) from error

    # -- language --------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        think: bool = False,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        temperature: float = 0.0,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Generate one reply through the adapter's language stage.

        This exists so a conversation needs nothing but the single weights
        package. The adapter reaches the same Ollama tag Egg used to address
        through the voice daemon, so no additional weights are loaded -- but
        the daemon stops being required to hold a conversation, which is what
        lets it be stopped along with the Whisper and Supertonic weights it
        keeps resident and never offers a way to release.

        Returns the Ollama-shaped message, so ``tool_calls`` reach the caller
        unchanged. Adapter v1 is non-streaming by contract, so a caller that
        wants deltas gets the finished reply in one piece.
        """

        payload = self._request(
            task="chat",
            messages=messages,
            think=think,
            options={
                key: value
                for key, value in (
                    ("num_ctx", num_ctx),
                    ("num_predict", num_predict),
                    ("temperature", temperature),
                )
                if value is not None
            }
            or None,
        )
        if tools:
            payload["tools"] = tools
        async with self._resident(self.LANGUAGE_COMPONENT):
            result = await self._post(
                payload,
                timeout_seconds=(
                    timeout_seconds
                    if timeout_seconds is not None
                    else self.config.timeout_seconds
                ),
            )
        return self._message(result)

    def _portal_token(self) -> str:
        """Read the token the daemon minted for this run of the portal.

        It is rewritten on every daemon start, so it is read per call rather
        than cached: the residency manager stops and starts that unit as a
        matter of course, and a cached token would be stale from the first
        eviction onwards.
        """

        path = Path(self.config.portal_token_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise OmniAdapterUnavailable(
                f"omni portal token is unavailable at {path}: {error}"
            ) from error

    async def execute_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Run one of the adapter's own tools and return its result.

        These are the suite the portal carries -- web search and fetch,
        document and session search, scratch memory, bounded arithmetic. Egg
        keeps its own tool loop because most of its tools are cameras and
        memory, so it runs these individually rather than handing the
        conversation over to the portal's agentic loop.
        """

        token = self._portal_token()
        base = str(self.config.portal_base_url).rstrip("/")
        timeout = aiohttp.ClientTimeout(total=self.config.portal_timeout_seconds)
        async with self._resident(self.SPEECH_COMPONENT):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{base}/api/tools/{name}/call",
                        json={"arguments": arguments},
                        headers={"Authorization": f"Bearer {token}"},
                    ) as response:
                        if response.status >= 400:
                            detail = (await response.text())[:300]
                            raise OmniAdapterError(
                                f"omni tool {name} HTTP {response.status}: {detail}"
                            )
                        payload = await response.json()
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as error:
                raise OmniAdapterUnavailable(
                    f"omni tool {name} is unreachable: {error}"
                ) from error
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, dict):
            raise OmniAdapterError(f"omni tool {name} returned no result object")
        return result

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
        async with self._resident(self.COMPREHENSION_COMPONENT):
            result = await self._post(
                payload, timeout_seconds=self.config.timeout_seconds
            )
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
        async with self._resident(self.COMPREHENSION_COMPONENT):
            result = await self._post(
                payload, timeout_seconds=self.config.timeout_seconds
            )
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
        async with self._resident(self.COMPREHENSION_COMPONENT):
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

    def _voice_profile(self) -> tuple[dict[str, Any], Path] | None:
        """Load the adapter's own voice profile, so Egg speaks its presets.

        The adapter checkout ships the same ``voice-profile.json`` the portal
        uses. Reading it here is what makes Egg's default voice the project's
        Female preset rather than whatever the backend happens to pick.
        """

        if self._voice_profile_cache is not None:
            return self._voice_profile_cache or None
        path = Path(self.config.voice_profile_path).expanduser()
        try:
            profile = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            logger.info("voice profile is unavailable (%s); using backend defaults", error)
            self._voice_profile_cache = {}
            return None
        if not isinstance(profile, dict):
            self._voice_profile_cache = {}
            return None
        self._voice_profile_cache = (profile, path.parent)
        return self._voice_profile_cache

    def _selected_preset(
        self, profile: dict[str, Any]
    ) -> dict[str, Any] | None:
        presets = profile.get("presets")
        if not isinstance(presets, list):
            return None
        candidates = [item for item in presets if isinstance(item, dict)]
        requested = self.config.voice_preset
        if requested:
            for preset in candidates:
                if str(preset.get("id") or "").casefold() == requested.casefold():
                    return preset
            logger.warning(
                "voice preset %r is not in %s; using the profile default",
                requested,
                self.config.voice_profile_path,
            )
        for preset in candidates:
            if preset.get("default"):
                return preset
        return candidates[0] if candidates else None

    def _speaker_reference(self) -> bytes | None:
        """Return the speaker WAV to clone, preferring an explicit override."""

        override = self.config.voice_reference_path
        if override:
            path = Path(override).expanduser()
            try:
                return path.read_bytes()
            except OSError as error:
                raise OmniAdapterError(
                    f"voice reference is unreadable: {path}: {error}"
                ) from error
        loaded = self._voice_profile()
        if loaded is None:
            return None
        profile, root = loaded
        preset = self._selected_preset(profile)
        speaker_file = (preset or {}).get("speaker_file") or profile.get("speaker_file")
        if not isinstance(speaker_file, str) or not speaker_file:
            return None
        path = (root / speaker_file).expanduser()
        try:
            return path.read_bytes()
        except OSError as error:
            logger.warning("voice preset reference is unreadable: %s: %s", path, error)
            return None

    def _speech_settings(self) -> dict[str, Any]:
        speech: dict[str, Any] = {}
        loaded = self._voice_profile()
        profile = loaded[0] if loaded else {}
        preset = self._selected_preset(profile) if profile else None
        # Profile values first, explicit config second: the profile carries the
        # project's tuned sampling, config exists to deviate from it.
        for key, profile_key, override in (
            ("language", "language", self.config.voice_language),
            ("temperature", "temperature", self.config.voice_temperature),
            ("top_k", "top_k", self.config.voice_top_k),
            ("top_p", "top_p", self.config.voice_top_p),
            ("seed", "seed", self.config.voice_seed),
            ("max_frames", "max_frames", self.config.voice_max_frames),
        ):
            value = override if override is not None else profile.get(profile_key)
            if value is not None:
                speech[key] = value
        voice = self.config.voice or (preset or {}).get("id")
        if voice:
            speech["voice"] = voice
        if self.config.voice_style:
            speech["style"] = self.config.voice_style
        reference = self._speaker_reference()
        if reference:
            speech["speaker_audio"] = base64.b64encode(
                normalize_input_wav(reference)
            ).decode("ascii")
        return speech

    async def stream_speech(self, text: str) -> AsyncIterator[bytes]:
        """Yield Qwen3-TTS decoder PCM windows as they are produced.

        This is the portal's NDJSON extension (`/api/chat/stream`), not the
        portable v1 route. It exists for one reason: the backend emits about
        160 ms of PCM per decoder window, so playback can start on the first
        window instead of after the whole utterance has been generated.

        Chunks are 24 kHz mono PCM16 with no WAV header, in order. The stream's
        sequence numbers are continuous across blocks and are checked here: a
        gap means audio was lost, and silently concatenating across it would
        produce a subtly wrong utterance rather than an obvious failure.
        """

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
        payload["stream"] = True
        await self._ensure_available()
        timeout = aiohttp.ClientTimeout(
            total=self.config.speech_timeout_seconds,
            sock_read=self.config.speech_timeout_seconds,
        )
        expected = 0
        produced = False
        try:
            # Pinned for the whole stream: weights must not be evicted between
            # decoder windows, which would cut an utterance in half.
            async with self._resident(self.SPEECH_COMPONENT), self._gate:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{self._base_url()}/api/chat/stream",
                        json=payload,
                        headers=self._headers(),
                    ) as response:
                        if response.status >= 400:
                            detail = (await response.text())[:500]
                            raise OmniAdapterError(
                                f"Omni adapter stream HTTP {response.status}: {detail}"
                            )
                        async for raw in response.content:
                            line = raw.strip()
                            if not line:
                                continue
                            try:
                                event = json.loads(line)
                            except ValueError as error:
                                raise OmniAdapterError(
                                    "Omni adapter stream returned invalid NDJSON"
                                ) from error
                            if not isinstance(event, dict):
                                continue
                            kind = event.get("type")
                            if kind == "error":
                                raise OmniAdapterError(
                                    f"Omni adapter stream failed: {event.get('error')}"
                                )
                            if kind != "audio_delta":
                                continue
                            audio = event.get("audio")
                            if not isinstance(audio, dict) or not audio.get("data"):
                                continue
                            sequence = audio.get("sequence")
                            if isinstance(sequence, int) and sequence != expected:
                                raise OmniAdapterError(
                                    "Omni adapter PCM stream lost a window: expected "
                                    f"sequence {expected}, received {sequence}"
                                )
                            expected += 1
                            try:
                                chunk = base64.b64decode(str(audio["data"]), validate=True)
                            except (ValueError, TypeError) as error:
                                raise OmniAdapterError(
                                    "Omni adapter PCM window is not valid base64"
                                ) from error
                            if len(chunk) % 2:
                                raise OmniAdapterError(
                                    "Omni adapter PCM window ended on a partial sample"
                                )
                            if chunk:
                                produced = True
                                yield chunk
        except asyncio.CancelledError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as error:
            self._note_failure(error)
            raise OmniAdapterUnavailable(
                f"Omni adapter speech stream failed: {self._last_error}"
            ) from error
        if not produced:
            raise OmniAdapterError("Omni adapter stream returned no audio")

    async def chat_with_speech(
        self, messages: list[dict[str, Any]], *, think: bool = False
    ) -> dict[str, object]:
        """Answer a conversation and speak the reply in one adapter round trip.

        The adapter routes language then TTS itself, so the spoken audio is
        generated from exactly the text that is returned -- there is no window
        in which a caller could play one reply while displaying another.
        """

        if not messages:
            raise OmniAdapterError("chat requires at least one message")
        payload = self._request(
            task="chat",
            messages=[
                {"role": str(item.get("role") or "user"), "content": str(item.get("content") or "")}
                for item in messages
            ],
            response_modalities=["text", "audio"],
            speech_mode="always",
            speech=self._speech_settings(),
            think=think,
        )
        async with self._resident(self.SPEECH_COMPONENT):
            result = await self._post(
                payload, timeout_seconds=self.config.speech_timeout_seconds
            )
        message = self._message(result)
        text = str(message.get("content") or "").strip()
        audio_envelope = message.get("audio")
        audio: bytes | None = None
        if isinstance(audio_envelope, dict) and audio_envelope.get("data"):
            try:
                decoded = base64.b64decode(str(audio_envelope["data"]), validate=True)
            except (ValueError, TypeError) as error:
                raise OmniAdapterError("spoken reply is not valid base64") from error
            if not decoded.startswith(b"RIFF"):
                raise OmniAdapterError("spoken reply is not a WAV payload")
            audio = decoded
        metadata = self._adapter_metadata(result)
        return {
            "text": text,
            "audio": audio,
            "route": list(metadata.get("route") or []),
            "thinking": message.get("thinking"),
        }

    @staticmethod
    def pcm_to_wav(pcm: bytes, sample_rate: int = OUTPUT_SAMPLE_RATE_HZ) -> bytes:
        """Wrap streamed PCM windows in the WAV container callers expect."""

        output = io.BytesIO()
        with wave.open(output, "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(sample_rate)
            target.writeframes(pcm)
        return output.getvalue()

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
        async with self._resident(self.SPEECH_COMPONENT):
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
