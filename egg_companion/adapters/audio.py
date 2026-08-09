from __future__ import annotations

import json
import io
import re
import subprocess
import struct
import threading
import time
import wave
from collections import deque

import numpy as np

from egg_companion.config import AudioConfig, TranscriptionConfig

_respeaker_usb_lock = threading.RLock()


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


def read_respeaker_direction(config: AudioConfig) -> float:
    try:
        import usb.core
        import usb.util
    except ImportError as error:
        raise RuntimeError("PyUSB is required for ReSpeaker USB direction-of-arrival") from error
    with _respeaker_usb_lock:
        device = usb.core.find(
            idVendor=config.respeaker_vendor_id,
            idProduct=config.respeaker_product_id,
        )
        if device is None:
            raise RuntimeError(
                f"ReSpeaker USB device {config.respeaker_vendor_id:04x}:"
                f"{config.respeaker_product_id:04x} not found"
            )
        response = device.ctrl_transfer(
            usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0,
            0xC0,
            21,
            8,
            100000,
        )
    angle, _ = struct.unpack("ii", bytes(response))
    if not 0 <= angle < 360:
        raise RuntimeError(f"ReSpeaker returned invalid direction: {angle}")
    return float(angle)


class ReSpeakerDirection:
    """Reads DOA from the ReSpeaker 4 Mic Array's native USB tuning interface."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self._angles: deque[float] = deque(maxlen=20)
        self._thread: threading.Thread | None = None
        self._failure: BaseException | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self.config.doa_mode == "disabled":
            return
        if self.config.doa_mode not in {"respeaker_usb", "serial"}:
            raise ValueError(f"unsupported ReSpeaker DOA mode: {self.config.doa_mode}")
        self._thread = threading.Thread(target=self._read, name="respeaker-doa", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def latest_angle(self) -> float | None:
        if self._failure:
            raise RuntimeError("ReSpeaker DOA reader failed") from self._failure
        return self._angles[-1] if self._angles else None

    def _read(self) -> None:
        try:
            if self.config.doa_mode == "respeaker_usb":
                self._read_respeaker_usb()
                return
            self._read_serial()
        except BaseException as error:
            self._failure = error

    def _read_respeaker_usb(self) -> None:
        while not self._stop_event.is_set():
            self._angles.append(read_respeaker_direction(self.config))
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
        self._hangover_frames_required = max(1, round(transcription.vad_hangover_ms / self._FRAME_MS))
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
        self._pre_roll.clear()
        self._carry = np.empty(0, dtype=np.float32)

    def feed(self, samples: np.ndarray) -> list[np.ndarray]:
        """Consume newly captured raw samples; return zero or more completed
        utterances (raw, ungained mono float32 arrays) finalized by this call."""
        combined = np.concatenate((self._carry, samples.astype(np.float32, copy=False)))
        frame_count = combined.size // self._frame_samples
        self._carry = combined[frame_count * self._frame_samples:]
        completed: list[np.ndarray] = []
        for index in range(frame_count):
            frame = combined[index * self._frame_samples: (index + 1) * self._frame_samples]
            utterance = self._consume_frame(frame)
            if utterance is not None:
                completed.append(utterance)
        return completed

    def _consume_frame(self, frame: np.ndarray) -> np.ndarray | None:
        voiced = self._is_voiced(frame)
        if not self._recording:
            self._pre_roll.append(frame)
            self._voiced_streak = self._voiced_streak + 1 if voiced else 0
            if self._voiced_streak < self._onset_frames_required:
                return None
            self._recording = True
            self._utterance_frames = list(self._pre_roll)
            self._silence_streak = 0
            return None
        self._utterance_frames.append(frame)
        self._silence_streak = 0 if voiced else self._silence_streak + 1
        if self._silence_streak >= self._hangover_frames_required or len(self._utterance_frames) >= self._max_frames:
            return self._finalize()
        return None

    def _finalize(self) -> np.ndarray:
        utterance = (
            np.concatenate(self._utterance_frames) if self._utterance_frames else np.empty(0, dtype=np.float32)
        )
        self._recording = False
        self._utterance_frames = []
        self._voiced_streak = 0
        self._silence_streak = 0
        self._pre_roll.clear()
        return utterance

    def _is_voiced(self, frame: np.ndarray) -> bool:
        gained = np.clip(frame * self.transcription.vad_input_gain, -1, 1)
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
        gain = min(self.audio.asr_target_rms / max(rms, 1e-6), self.audio.asr_max_gain)
        normalized = np.clip(mono * gain, -1, 1)
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
        vad_input = np.clip(mono * self.transcription.vad_input_gain, -1, 1)
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
