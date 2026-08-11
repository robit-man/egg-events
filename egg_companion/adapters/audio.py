from __future__ import annotations

import io
import json
import math
import re
import subprocess
import struct
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np

from egg_companion.config import AudioConfig, TranscriptionConfig

_respeaker_usb_lock = threading.RLock()

_RESPEAKER_DSP_PARAMETERS: dict[str, tuple[int, int, str]] = {
    "doa_angle": (21, 0, "int"),
    "voice_activity": (19, 32, "int"),
    "speech_detected": (19, 22, "int"),
    "agc_gain": (19, 3, "float"),
    "aec_far_end_silence": (18, 31, "int"),
    "rt60_seconds": (18, 26, "float"),
}
_RESPEAKER_LED_COMMANDS = {"trace": 0, "listen": 2, "speak": 3, "think": 4, "spin": 5}

_SPEECH_BAND_LOW_HZ = 160.0
_SPEECH_BAND_LOW_TRANSITION_HZ = 40.0
_SPEECH_BAND_HIGH_HZ = 7600.0
_SPEECH_BAND_HIGH_TRANSITION_HZ = 400.0


def condition_speech_band(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Remove DC, chassis/fan rumble, and near-Nyquist energy before VAD/ASR.

    The ReSpeaker DSP stream on this unit has a strong stationary component near
    117 Hz. WebRTC VAD classifies that component as voice, so applying gain to
    the unfiltered window turns room tone into plausible Whisper text. A smooth
    FFT-domain speech-band filter avoids an optional DSP dependency and works on
    both 30 ms VAD frames and complete utterances.
    """
    source = np.asarray(samples, dtype=np.float32)
    if source.ndim != 1 or not source.size:
        return source.copy()
    centered = source - float(np.mean(source))
    spectrum = np.fft.rfft(centered)
    frequencies = np.fft.rfftfreq(centered.size, d=1.0 / sample_rate)
    weights = np.ones_like(frequencies)

    low_stop = max(0.0, _SPEECH_BAND_LOW_HZ - _SPEECH_BAND_LOW_TRANSITION_HZ)
    weights[frequencies < low_stop] = 0.0
    low_transition = (frequencies >= low_stop) & (frequencies < _SPEECH_BAND_LOW_HZ)
    if np.any(low_transition):
        phase = (frequencies[low_transition] - low_stop) / max(
            _SPEECH_BAND_LOW_HZ - low_stop, 1.0
        )
        weights[low_transition] = 0.5 - 0.5 * np.cos(np.pi * phase)

    nyquist = sample_rate / 2
    high_stop = min(nyquist, _SPEECH_BAND_HIGH_HZ)
    high_start = max(_SPEECH_BAND_LOW_HZ, high_stop - _SPEECH_BAND_HIGH_TRANSITION_HZ)
    high_transition = (frequencies > high_start) & (frequencies <= high_stop)
    if np.any(high_transition):
        phase = (frequencies[high_transition] - high_start) / max(
            high_stop - high_start, 1.0
        )
        weights[high_transition] = 0.5 + 0.5 * np.cos(np.pi * phase)
    weights[frequencies > high_stop] = 0.0
    return np.fft.irfft(spectrum * weights, n=centered.size).astype(np.float32)


@dataclass(frozen=True)
class UtteranceBoundary:
    kind: Literal["started", "ended"]
    at_monotonic: float
    samples: np.ndarray | None = None
    reason: str | None = None
    voiced_ms: int = 0
    silence_target_ms: int = 0
    continuation_count: int = 0


def resolve_pulse_source(audio: AudioConfig) -> tuple[str, int]:
    if audio.input_device != "default":
        return audio.input_device, audio.channels
    result = subprocess.run(
        ["pactl", "get-default-source"],
        check=False,
        capture_output=True,
        text=True,
    )
    source = result.stdout.strip()
    if result.returncode or not source:
        detail = result.stderr.strip()
        raise RuntimeError(f"cannot resolve PulseAudio input source: {detail or result.returncode}")
    sources = subprocess.run(
        ["pactl", "list", "sources", "short"], check=False, capture_output=True, text=True
    )
    for line in sources.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[1] == source:
            match = re.search(r"\b(\d+)ch\b", line)
            if match:
                return source, int(match.group(1))
    return source, audio.channels


def _respeaker_device(config: AudioConfig):
    try:
        import usb.core
    except ImportError as error:
        raise RuntimeError("PyUSB is required for ReSpeaker XVF3000 controls") from error
    device = usb.core.find(
        idVendor=config.respeaker_vendor_id,
        idProduct=config.respeaker_product_id,
    )
    if device is None:
        raise RuntimeError(
            f"ReSpeaker USB device {config.respeaker_vendor_id:04x}:"
            f"{config.respeaker_product_id:04x} not found"
        )
    return device


def _read_respeaker_parameter(device, parameter: tuple[int, int, str]) -> float:
    import usb.util

    parameter_id, offset, value_type = parameter
    command = 0x80 | offset | (0x40 if value_type == "int" else 0)
    response = device.ctrl_transfer(
        usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
        0, command, parameter_id, 8, 100000,
    )
    mantissa, exponent = struct.unpack("ii", bytes(response))
    return float(mantissa if value_type == "int" else mantissa * (2.0 ** exponent))


def read_respeaker_dsp_status(config: AudioConfig, *, diagnostics: bool = True) -> dict[str, object]:
    """Read the XVF3000's native VAD, DoA, AEC, AGC, and room estimate."""

    device = _respeaker_device(config)
    names = list(_RESPEAKER_DSP_PARAMETERS) if diagnostics else [
        "doa_angle", "voice_activity", "speech_detected"
    ]
    with _respeaker_usb_lock:
        values = {
            name: _read_respeaker_parameter(device, _RESPEAKER_DSP_PARAMETERS[name])
            for name in names
        }
    values["doa_angle"] = values["doa_angle"] % 360
    for name in ("voice_activity", "speech_detected", "aec_far_end_silence"):
        if name in values:
            values[name] = bool(values[name])
    values.update(
        {
            "device": "XVF3000 ReSpeaker USB 4-Mic Array v2.0",
            "sample_rate": config.sample_rate,
            "capture_channels": config.channels,
            "asr_channel": config.asr_channel,
            "asr_stream": "processed AEC/beamformed/denoised channel 0",
            "updated_at": time.time(),
        }
    )
    return values


def read_respeaker_direction(config: AudioConfig) -> float:
    angle = float(read_respeaker_dsp_status(config, diagnostics=False)["doa_angle"])
    if not 0 <= angle < 360:
        raise RuntimeError(f"ReSpeaker returned invalid direction: {angle}")
    return angle


def write_respeaker_led(config: AudioConfig, state: str) -> None:
    if not config.respeaker_led_enabled:
        return
    import usb.util

    device = _respeaker_device(config)
    normalized = state.strip().lower()
    if normalized == "off":
        command, data = 1, [0, 0, 0, 0]
    elif normalized in _RESPEAKER_LED_COMMANDS:
        command, data = _RESPEAKER_LED_COMMANDS[normalized], [0]
    else:
        raise ValueError(f"unsupported ReSpeaker LED state: {state}")
    with _respeaker_usb_lock:
        device.ctrl_transfer(
            usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0, 0x20, 0x1C, [config.respeaker_led_brightness], 8000,
        )
        device.ctrl_transfer(
            usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0, command, 0x1C, data, 8000,
        )


class ReSpeakerDirection:
    """Reads DOA from the ReSpeaker 4 Mic Array's native USB tuning interface."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self._angles: deque[float] = deque(maxlen=20)
        self._thread: threading.Thread | None = None
        self._failure: BaseException | None = None
        self._stop_event = threading.Event()
        self._status: dict[str, object] = {
            "device": "XVF3000 ReSpeaker USB 4-Mic Array v2.0",
            "ready": False,
            "led_state": "off",
        }

    def start(self) -> None:
        if self.config.doa_mode == "disabled":
            return
        if self.config.doa_mode not in {"respeaker_usb", "serial"}:
            raise ValueError(f"unsupported ReSpeaker DOA mode: {self.config.doa_mode}")
        if self.config.doa_mode == "respeaker_usb" and self.config.respeaker_led_enabled:
            self.set_led_state("trace")
        self._thread = threading.Thread(target=self._read, name="respeaker-doa", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.config.doa_mode == "respeaker_usb" and self.config.respeaker_led_enabled:
            try:
                self.set_led_state("off")
            except Exception:
                pass

    def latest_angle(self) -> float | None:
        if self._failure:
            raise RuntimeError("ReSpeaker DOA reader failed") from self._failure
        return self._angles[-1] if self._angles else None

    def latest_status(self) -> dict[str, object]:
        return dict(self._status)

    def set_led_state(self, state: str) -> None:
        write_respeaker_led(self.config, state)
        self._status = {**self._status, "led_state": state, "led_brightness": self.config.respeaker_led_brightness}

    def try_set_led_state(self, state: str) -> bool:
        try:
            self.set_led_state(state)
            return True
        except Exception:
            return False

    def _read(self) -> None:
        try:
            if self.config.doa_mode == "respeaker_usb":
                self._read_respeaker_usb()
                return
            self._read_serial()
        except BaseException as error:
            self._failure = error

    def _read_respeaker_usb(self) -> None:
        last_diagnostics = 0.0
        while not self._stop_event.is_set():
            diagnostics = time.monotonic() - last_diagnostics >= 1.0
            status = {**self._status, **read_respeaker_dsp_status(self.config, diagnostics=diagnostics)}
            if diagnostics:
                last_diagnostics = time.monotonic()
            status["ready"] = True
            self._status = status
            self._angles.append(float(status["doa_angle"]))
            self._stop_event.wait(0.1)

    def _read_serial(self) -> None:
        if not self.config.doa_serial_device:
            raise RuntimeError("doa_serial_device is required when doa_mode is serial")
        import serial

        with serial.Serial(self.config.doa_serial_device, baudrate=115200, timeout=1) as port:
            while not self._stop_event.is_set():
                line = port.readline().decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                payload = json.loads(line)
                angle = payload.get("doa_angle", payload.get("angle", payload.get("direction")))
                if isinstance(angle, (int, float)):
                    self._angles.append(float(angle) % 360)


class UtteranceSegmenter:
    """Buffers continuous mono audio into complete, VAD-bounded utterances instead
    of a fixed-length tumbling window. A blind fixed window routinely truncates
    speech mid-word at its boundary, which is a well-documented trigger for
    Whisper repetition-loop hallucinations; this instead starts an utterance on
    confirmed voice onset (with a pre-roll so the first syllable isn't clipped
    while onset is being confirmed) and ends it on a trailing-silence hangover or
    a hard `segment_seconds` cap, so speech is never split mid-utterance."""

    _FRAME_MS = 30  # webrtcvad requires exact 10/20/30ms frames

    def __init__(self, audio: AudioConfig, transcription: TranscriptionConfig) -> None:
        import webrtcvad

        self.audio = audio
        self.transcription = transcription
        self._detector = webrtcvad.Vad(transcription.vad_aggressiveness)
        self._frame_samples = audio.sample_rate * self._FRAME_MS // 1000
        pre_roll_frames = max(1, round(transcription.vad_pre_roll_ms / self._FRAME_MS))
        self._pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_frames)
        self._onset_frames_required = max(1, round(transcription.vad_min_contiguous_ms / self._FRAME_MS))
        self._max_frames = max(
            self._onset_frames_required,
            round(transcription.segment_seconds * 1000 / self._FRAME_MS),
        )
        self._carry = np.empty(0, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self._recording = False
        self._utterance_frames: list[np.ndarray] = []
        self._voiced_streak = 0
        self._silence_streak = 0
        self._voiced_frames = 0
        self._continuation_count = 0
        self._pre_roll.clear()
        self._carry = np.empty(0, dtype=np.float32)

    def feed(self, samples: np.ndarray) -> list[np.ndarray]:
        """Consume newly captured raw samples; return zero or more completed
        utterances (raw, ungained mono float32 arrays) finalized by this call."""
        return [
            event.samples
            for event in self.feed_events(samples)
            if event.kind == "ended" and event.samples is not None
        ]

    def feed_events(self, samples: np.ndarray) -> list[UtteranceBoundary]:
        """Consume samples and expose ordered onset/end events for barge-in."""
        combined = np.concatenate((self._carry, samples.astype(np.float32, copy=False)))
        frame_count = combined.size // self._frame_samples
        self._carry = combined[frame_count * self._frame_samples:]
        events: list[UtteranceBoundary] = []
        for index in range(frame_count):
            frame = combined[index * self._frame_samples: (index + 1) * self._frame_samples]
            events.extend(self._consume_frame(frame))
        return events

    def _consume_frame(self, frame: np.ndarray) -> list[UtteranceBoundary]:
        voiced = self._is_voiced(frame)
        if not self._recording:
            self._pre_roll.append(frame)
            self._voiced_streak = self._voiced_streak + 1 if voiced else 0
            if self._voiced_streak < self._onset_frames_required:
                return []
            self._recording = True
            self._utterance_frames = list(self._pre_roll)
            self._silence_streak = 0
            self._voiced_frames = self._onset_frames_required
            self._continuation_count = 0
            return [
                UtteranceBoundary(
                    "started",
                    time.monotonic(),
                    voiced_ms=self._voiced_frames * self._FRAME_MS,
                    silence_target_ms=self._hangover_target_ms(),
                )
            ]
        self._utterance_frames.append(frame)
        if voiced:
            if self._silence_streak:
                self._continuation_count += 1
            self._silence_streak = 0
            self._voiced_frames += 1
        else:
            self._silence_streak += 1
        target_frames = max(1, round(self._hangover_target_ms() / self._FRAME_MS))
        if self._silence_streak >= target_frames:
            return [self._finalize("silence")]
        if len(self._utterance_frames) >= self._max_frames:
            return [self._finalize("max_utterance")]
        return []

    def _hangover_target_ms(self) -> int:
        base_ms = float(self.transcription.vad_hangover_ms)
        configured_max = self.transcription.vad_hangover_max_ms
        max_ms = max(base_ms, float(configured_max if configured_max is not None else base_ms))
        if max_ms <= base_ms:
            return round(base_ms)
        growth_ms = float(self.transcription.vad_hangover_growth_ms)
        effective_voiced_ms = (
            self._voiced_frames * self._FRAME_MS
            + self._continuation_count
            * growth_ms
            * float(self.transcription.vad_continuation_growth)
        )
        extension = (max_ms - base_ms) * (1 - math.exp(-effective_voiced_ms / growth_ms))
        return round(min(max_ms, max(base_ms, base_ms + extension)))

    def _finalize(self, reason: str) -> UtteranceBoundary:
        utterance = (
            np.concatenate(self._utterance_frames)
            if self._utterance_frames
            else np.empty(0, dtype=np.float32)
        )
        event = UtteranceBoundary(
            "ended",
            time.monotonic(),
            samples=utterance,
            reason=reason,
            voiced_ms=self._voiced_frames * self._FRAME_MS,
            silence_target_ms=self._hangover_target_ms(),
            continuation_count=self._continuation_count,
        )
        self._recording = False
        self._utterance_frames = []
        self._voiced_streak = 0
        self._silence_streak = 0
        self._voiced_frames = 0
        self._continuation_count = 0
        self._pre_roll.clear()
        return event

    def _is_voiced(self, frame: np.ndarray) -> bool:
        speech_band = condition_speech_band(frame, self.audio.sample_rate)
        gained = np.clip(speech_band * self.transcription.vad_input_gain, -1, 1)
        rms = float(np.sqrt(np.mean(np.square(gained))))
        if rms < self.transcription.vad_min_voiced_rms:
            return False
        pcm = (gained * 32767).astype("<i2")
        return self._detector.is_speech(pcm.tobytes(), self.audio.sample_rate)


class ReSpeakerCapture:
    """Captures a real voiced ReSpeaker segment as a mono 16-bit WAV payload."""

    def __init__(self, audio: AudioConfig, transcription: TranscriptionConfig) -> None:
        self.audio = audio
        self.transcription = transcription
        self.last_speech_ratio = 0.0
        self.last_speech_ms = 0
        self.last_voiced_rms = 0.0
        self.last_speech_detected = False
        self.last_conditioned_rms = 0.0
        self.last_applied_gain = 1.0

    def capture_once(self) -> bytes | None:
        wav_audio, rms = self.capture_segment()
        return wav_audio if rms >= self.transcription.rms_threshold else None

    def capture_segment(self) -> tuple[bytes, float]:
        frame_count = int(self.audio.sample_rate * self.transcription.segment_seconds)
        source, capture_channels = self._resolve_pulse_source()
        process = subprocess.Popen(
            [
                "parecord",
                f"--device={source}",
                "--raw",
                "--format=s16le",
                f"--rate={self.audio.sample_rate}",
                f"--channels={capture_channels}",
                "--latency-msec=50",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            raw_audio, stderr = process.communicate(timeout=self.transcription.segment_seconds)
        except subprocess.TimeoutExpired:
            process.terminate()
            raw_audio, stderr = process.communicate(timeout=2)
        if process.returncode not in {0, -15}:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"parecord capture failed: {detail or process.returncode}")
        samples = np.frombuffer(raw_audio, dtype="<i2")
        complete_samples = samples.size - samples.size % capture_channels
        audio = samples[:complete_samples].reshape(-1, capture_channels).astype(np.float32) / 32768
        if audio.shape[0] < max(1, frame_count // 2):
            raise RuntimeError(f"ReSpeaker capture was too short: {audio.shape[0]} frames")
        if audio.ndim != 2 or not audio.shape[1]:
            raise RuntimeError("ReSpeaker capture returned no input channels")
        if self.audio.asr_channel >= audio.shape[1]:
            raise RuntimeError(
                f"ReSpeaker ASR channel {self.audio.asr_channel} is unavailable in {audio.shape[1]}-channel capture"
            )
        mono = audio[:, self.audio.asr_channel]
        return self.process_samples(mono)

    def process_samples(self, mono: np.ndarray) -> tuple[bytes, float]:
        rms = self.analyze_samples(mono)
        conditioned = condition_speech_band(mono, self.audio.sample_rate)
        reference_rms = self.last_voiced_rms or self.last_conditioned_rms
        gain = min(
            self.audio.asr_target_rms / max(reference_rms, 1e-6),
            self.audio.asr_max_gain,
        )
        self.last_applied_gain = float(gain)
        normalized = np.clip(conditioned * gain, -1, 1)
        pcm = (normalized * 32767).astype("<i2")
        payload = io.BytesIO()
        with wave.open(payload, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.audio.sample_rate)
            wav.writeframes(pcm.tobytes())
        return payload.getvalue(), rms

    def analyze_samples(self, mono: np.ndarray) -> float:
        if mono.ndim != 1 or not mono.size:
            raise RuntimeError("ReSpeaker ASR processing received no mono samples")
        mono = mono.astype(np.float32, copy=False)
        rms = float(np.sqrt(np.mean(np.square(mono))))
        speech_band = condition_speech_band(mono, self.audio.sample_rate)
        self.last_conditioned_rms = float(np.sqrt(np.mean(np.square(speech_band))))
        vad_input = np.clip(speech_band * self.transcription.vad_input_gain, -1, 1)
        self.last_speech_detected = self._detect_speech(vad_input)
        return rms

    def _detect_speech(self, mono: np.ndarray) -> bool:
        try:
            import webrtcvad
        except ImportError as error:
            raise RuntimeError("webrtcvad-wheels is required for ReSpeaker speech detection") from error
        frame_ms = 30
        frame_samples = self.audio.sample_rate * frame_ms // 1000
        frame_count = mono.size // frame_samples
        if frame_count == 0:
            self.last_speech_ratio = 0.0
            self.last_speech_ms = 0
            return False
        pcm = (np.clip(mono[: frame_count * frame_samples], -1, 1) * 32767).astype("<i2")
        detector = webrtcvad.Vad(self.transcription.vad_aggressiveness)
        frames = pcm.reshape(frame_count, frame_samples)
        frame_rms = np.sqrt(np.mean(np.square(frames.astype(np.float32) / 32768), axis=1))
        voiced = [
            detector.is_speech(frame.tobytes(), self.audio.sample_rate)
            and rms >= self.transcription.vad_min_voiced_rms
            for frame, rms in zip(frames, frame_rms)
        ]
        voiced_frames = int(sum(voiced))
        self.last_speech_ratio = float(voiced_frames / frame_count)
        self.last_speech_ms = int(voiced_frames * frame_ms)
        self.last_voiced_rms = float(frame_rms[voiced].mean()) if voiced_frames else 0.0
        longest_run = 0
        run = 0
        for is_voiced in voiced:
            run = run + 1 if is_voiced else 0
            longest_run = max(longest_run, run)
        return (
            self.last_speech_ms >= self.transcription.vad_min_speech_ms
            and self.last_speech_ratio >= self.transcription.vad_min_speech_ratio
            and longest_run * frame_ms >= self.transcription.vad_min_contiguous_ms
        )

    def _resolve_pulse_source(self) -> tuple[str, int]:
        return resolve_pulse_source(self.audio)


class ReSpeakerWaveformCapture:
    """Persistent ReSpeaker reader for dashboard waveform data independent of ASR windows."""

    def __init__(self, audio: AudioConfig) -> None:
        self.audio = audio
        self._process: subprocess.Popen[bytes] | None = None
        self._channels = audio.channels
        self._lock = threading.RLock()

    def read_chunk(self) -> np.ndarray:
        with self._lock:
            self._ensure_process()
            if self._process is None or self._process.stdout is None:
                raise RuntimeError("ReSpeaker waveform process is unavailable")
            frame_count = max(1, self.audio.sample_rate // self.audio.waveform_fps)
            expected_bytes = frame_count * self._channels * 2
            payload = self._process.stdout.read(expected_bytes)
            if len(payload) != expected_bytes:
                self.close()
                raise RuntimeError(f"ReSpeaker waveform capture was short: {len(payload)}/{expected_bytes} bytes")
        samples = np.frombuffer(payload, dtype="<i2").reshape(-1, self._channels).astype(np.float32) / 32768
        if self.audio.asr_channel >= samples.shape[1]:
            raise RuntimeError(f"ReSpeaker ASR channel {self.audio.asr_channel} is unavailable in waveform capture")
        return samples[:, self.audio.asr_channel]

    def close(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
            if process is None:
                return
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)

    def _ensure_process(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        source, self._channels = resolve_pulse_source(self.audio)
        self._process = subprocess.Popen(
            [
                "parecord",
                f"--device={source}",
                "--raw",
                "--format=s16le",
                f"--rate={self.audio.sample_rate}",
                f"--channels={self._channels}",
                "--latency-msec=20",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
