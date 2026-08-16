#!/usr/bin/env python3
"""Jetson-native dual Whisper service for Egg.

Runs inside the pinned dustynv/whisper JetPack container so PyTorch, CUDA, and
the Tegra driver ABI agree. A tiny English pass rejects silence cheaply; a base
English pass verifies admitted speech and supplies the final transcript.
"""

from __future__ import annotations

import io
import json
import os
import re
import threading
import time
import wave
from difflib import SequenceMatcher
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np


HOST = os.getenv("EGG_WHISPER_HOST", "127.0.0.1")
PORT = int(os.getenv("EGG_WHISPER_PORT", "11436"))
MODEL_CACHE = os.getenv("EGG_WHISPER_MODEL_CACHE", "/models")
FAST_MODEL = os.getenv("EGG_WHISPER_FAST_MODEL", "tiny.en")
ACCURATE_MODEL = os.getenv("EGG_WHISPER_ACCURATE_MODEL", "base.en")
MAX_AUDIO_BYTES = 16 * 1024 * 1024
def normalize_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(re.sub(r"[^\w\s]", " ", value.casefold()).split())


def transcript_agreement(left: str, right: str) -> float:
    left_normalized, right_normalized = normalize_text(left), normalize_text(right)
    if not left_normalized or not right_normalized:
        return 0.0
    return SequenceMatcher(
        None, left_normalized.split(), right_normalized.split()
    ).ratio()


def rejection_reason(result: dict[str, Any]) -> str | None:
    text = normalize_text(result.get("text"))
    if not text:
        return "empty transcript"
    segments = [
        segment for segment in result.get("segments", []) if isinstance(segment, dict)
    ]
    no_speech = [
        float(segment["no_speech_prob"])
        for segment in segments
        if isinstance(segment.get("no_speech_prob"), (int, float))
    ]
    log_prob = [
        float(segment["avg_logprob"])
        for segment in segments
        if isinstance(segment.get("avg_logprob"), (int, float))
    ]
    compression = [
        float(segment["compression_ratio"])
        for segment in segments
        if isinstance(segment.get("compression_ratio"), (int, float))
    ]
    if no_speech and sum(no_speech) / len(no_speech) >= 0.55:
        return "high no-speech probability"
    if log_prob and sum(log_prob) / len(log_prob) <= -1.0:
        return "low average token probability"
    if compression and max(compression) >= 2.4:
        return "repetitive transcript compression"
    return None


def _quality(result: dict[str, Any]) -> tuple[float, float]:
    segments = [
        segment for segment in result.get("segments", []) if isinstance(segment, dict)
    ]
    log_probs = [
        float(segment["avg_logprob"])
        for segment in segments
        if isinstance(segment.get("avg_logprob"), (int, float))
    ]
    no_speech = [
        float(segment["no_speech_prob"])
        for segment in segments
        if isinstance(segment.get("no_speech_prob"), (int, float))
    ]
    return (
        sum(log_probs) / len(log_probs) if log_probs else -2.0,
        sum(no_speech) / len(no_speech) if no_speech else 1.0,
    )


def choose_dual_result(
    fast: dict[str, Any], accurate: dict[str, Any]
) -> tuple[dict[str, Any], str | None, float]:
    """Prefer base, but require independent evidence when its confidence is weak."""

    fast_reason, accurate_reason = rejection_reason(fast), rejection_reason(accurate)
    agreement = transcript_agreement(str(fast.get("text") or ""), str(accurate.get("text") or ""))
    if fast_reason is None and accurate_reason is None:
        base_log_prob, base_no_speech = _quality(accurate)
        if agreement < 0.22 and (base_log_prob < -0.72 or base_no_speech > 0.32):
            return {}, "dual Whisper disagreement on weak base decode", agreement
        return accurate, None, agreement
    if accurate_reason is None:
        base_log_prob, base_no_speech = _quality(accurate)
        if base_log_prob >= -0.55 and base_no_speech <= 0.22:
            return accurate, None, agreement
    if fast_reason is None:
        fast_log_prob, fast_no_speech = _quality(fast)
        if fast_log_prob >= -0.45 and fast_no_speech <= 0.15:
            return fast, None, agreement
    return {}, accurate_reason or fast_reason or "dual Whisper rejected transcript", agreement


class DualWhisperRuntime:
    def __init__(self) -> None:
        self.fast = None
        self.accurate = None
        self.ready = False
        self.error: str | None = None
        self.loaded_at: str | None = None
        self.mode = "dual"
        self.device = "unloaded"
        self._gate = threading.Lock()
        threading.Thread(target=self._load, name="whisper-model-loader", daemon=True).start()

    def _load(self) -> None:
        try:
            import torch
            import whisper

            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"Jetson CUDA unavailable to PyTorch {torch.__version__} ({torch.version.cuda})"
                )
            self.device = f"cuda:0 ({torch.cuda.get_device_name(0)})"
            self.fast = whisper.load_model(
                FAST_MODEL, device="cuda", download_root=MODEL_CACHE
            )
            self.accurate = whisper.load_model(
                ACCURATE_MODEL, device="cuda", download_root=MODEL_CACHE
            )
            self.loaded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.ready = True
        except Exception as error:  # pragma: no cover - hardware startup path
            self.error = f"{type(error).__name__}: {error}"

    @staticmethod
    def _decode_wav(payload: bytes) -> tuple[np.ndarray, float]:
        with wave.open(io.BytesIO(payload), "rb") as source:
            channels = source.getnchannels()
            width = source.getsampwidth()
            rate = source.getframerate()
            count = source.getnframes()
            frames = source.readframes(count)
        if channels != 1 or width != 2 or rate != 16000:
            raise ValueError("ASR input must be mono 16-bit PCM WAV at 16000 Hz")
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        return audio, count / rate

    @staticmethod
    def _infer(model: Any, audio: np.ndarray, language: str) -> dict[str, Any]:
        selected_language = None if language == "auto" else language
        return model.transcribe(
            audio,
            language=selected_language,
            task="transcribe",
            fp16=True,
            verbose=False,
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.55,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )

    def transcribe(self, payload: bytes, language: str) -> dict[str, Any]:
        if not self.ready or self.fast is None or self.accurate is None:
            raise RuntimeError(self.error or "dual Whisper models are still loading")
        audio, duration = self._decode_wav(payload)
        with self._gate:
            fast_result = self._infer(self.fast, audio, language)
            if self.mode == FAST_MODEL:
                reason = rejection_reason(fast_result)
                return {
                    "text": "" if reason else str(fast_result.get("text") or "").strip(),
                    "duration": duration,
                    "language": fast_result.get("language"),
                    "segments": [] if reason else fast_result.get("segments", []),
                    "rejection_reason": reason,
                    "backend": "jetson-whisper-cuda",
                    "engines": {FAST_MODEL: {"accepted": reason is None}},
                }
            # A grounded fast-pass decode is admission evidence, never the sole
            # final answer unless the base pass itself fails with high confidence.
            if rejection_reason(fast_result) is not None:
                return {
                    "text": "",
                    "duration": duration,
                    "language": fast_result.get("language"),
                    "segments": [],
                    "rejection_reason": rejection_reason(fast_result),
                    "backend": "jetson-dual-whisper",
                    "engines": {FAST_MODEL: {"accepted": False}},
                }
            accurate_result = self._infer(self.accurate, audio, language)
            if self.mode == ACCURATE_MODEL:
                reason = rejection_reason(accurate_result)
                return {
                    "text": "" if reason else str(accurate_result.get("text") or "").strip(),
                    "duration": duration,
                    "language": accurate_result.get("language"),
                    "segments": [] if reason else accurate_result.get("segments", []),
                    "rejection_reason": reason,
                    "backend": "jetson-whisper-cuda",
                    "engines": {ACCURATE_MODEL: {"accepted": reason is None}},
                }
        selected, reason, agreement = choose_dual_result(fast_result, accurate_result)
        return {
            "text": str(selected.get("text") or "").strip(),
            "duration": duration,
            "language": selected.get("language") or accurate_result.get("language"),
            "segments": selected.get("segments", []),
            "rejection_reason": reason,
            "backend": "jetson-dual-whisper",
            "agreement": round(agreement, 4),
            "engines": {
                FAST_MODEL: {"accepted": rejection_reason(fast_result) is None},
                ACCURATE_MODEL: {"accepted": rejection_reason(accurate_result) is None},
            },
        }

    def state(self) -> dict[str, Any]:
        return {
            "asrEngineId": "jetson-dual-whisper",
            "asrModelId": self.mode,
            "asrBackend": "dustynv-whisper-cuda",
            "asrPhase": "listening" if self.ready else "loading",
            "asrReady": self.ready,
            "device": self.device,
            "loadedAt": self.loaded_at,
            "lastError": self.error,
        }

    def catalog(self) -> dict[str, Any]:
        models = []
        for model_id, label in (
            ("dual", f"Dual {FAST_MODEL} → {ACCURATE_MODEL}"),
            (FAST_MODEL, FAST_MODEL),
            (ACCURATE_MODEL, ACCURATE_MODEL),
        ):
            models.append(
                {
                    "id": model_id,
                    "engineId": "jetson-dual-whisper",
                    "label": label,
                    "detail": "Pinned JetPack CUDA PyTorch inside dustynv/whisper:r36.2.0",
                    "readiness": {
                        "engineId": "jetson-dual-whisper",
                        "modelId": model_id,
                        "installed": True,
                        "weightsReady": self.ready,
                        "active": self.ready and model_id == self.mode,
                        "device": self.device,
                        "lastError": self.error,
                    },
                    "selected": model_id == self.mode,
                    "isActive": self.ready and model_id == self.mode,
                }
            )
        return {
            "current": self.mode,
            "models": models,
            "engines": [
                {
                    "id": "jetson-dual-whisper",
                    "label": "Jetson Dual Whisper",
                    "detail": "CUDA tiny admission followed by CUDA base verification",
                    "provider": "OpenAI / NVIDIA Jetson Containers",
                    "runtime": "PyTorch CUDA 12.2",
                    "models": models,
                    "availability": "available" if self.ready else "loading",
                    "selected": True,
                }
            ],
        }


RUNTIME: DualWhisperRuntime | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "EggJetsonWhisper/1"

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        assert RUNTIME is not None
        path = urlparse(self.path).path
        if path == "/health":
            self._json(HTTPStatus.OK if RUNTIME.ready else HTTPStatus.SERVICE_UNAVAILABLE, RUNTIME.state())
        elif path == "/v1/voice/state":
            self._json(HTTPStatus.OK, RUNTIME.state())
        elif path == "/v1/voice/asr-models":
            self._json(HTTPStatus.OK, RUNTIME.catalog())
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        assert RUNTIME is not None
        parsed = urlparse(self.path)
        if parsed.path == "/v1/voice/asr-models/switch":
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length) or b"{}")
            model_id = str(request.get("modelId") or "")
            if model_id not in {"dual", FAST_MODEL, ACCURATE_MODEL}:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "unknown_model"})
                return
            RUNTIME.mode = model_id
            self._json(HTTPStatus.OK, RUNTIME.state())
            return
        if parsed.path != "/v1/voice/transcribe":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_AUDIO_BYTES:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_audio_size"})
            return
        try:
            payload = self.rfile.read(length)
            language = parse_qs(parsed.query).get("language", ["auto"])[0]
            result = RUNTIME.transcribe(payload, language)
            self._json(HTTPStatus.OK, result)
        except ValueError as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception as error:
            self._json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "transcribe_unavailable", "message": str(error)},
            )

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.log_date_time_string()} {format % args}", flush=True)


if __name__ == "__main__":
    RUNTIME = DualWhisperRuntime()
    print(
        f"Egg Jetson dual Whisper listening on http://{HOST}:{PORT}; "
        f"models {FAST_MODEL} -> {ACCURATE_MODEL}",
        flush=True,
    )
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
