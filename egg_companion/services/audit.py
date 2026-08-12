from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from egg_companion.adapters.omnius import OmniusClient
from egg_companion.adapters.audio import read_respeaker_direction
from egg_companion.adapters.system_service import SystemServiceClient
from egg_companion.config import EggConfig
from egg_companion.memory.store import MemoryStore


@dataclass(frozen=True)
class AuditCheck:
    name: str
    status: str
    detail: str


def _error_detail(error: Exception) -> str:
    """Keep timeout and cancellation failures legible in dashboard diagnostics."""
    message = str(error).strip()
    return message or type(error).__name__


async def _command_check(name: str, command: list[str]) -> AuditCheck:
    if not shutil.which(command[0]):
        return AuditCheck(name, "fail", f"command unavailable: {command[0]}")
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    output, _ = await process.communicate()
    detail = output.decode("utf-8", errors="replace").strip()[:1000]
    return AuditCheck(name, "pass" if process.returncode == 0 else "fail", detail)


async def _camera_check(camera_source: str, camera_id: str) -> AuditCheck:
    if camera_source.startswith("/dev/video"):
        if not Path(camera_source).exists():
            return AuditCheck(f"camera:{camera_id}", "fail", f"missing: {camera_source}")
        try:
            import cv2

            def capture_frame() -> tuple[bool, object | None]:
                capture = cv2.VideoCapture(camera_source, cv2.CAP_V4L2)
                try:
                    if not capture.isOpened():
                        raise RuntimeError("cannot open V4L2 stream")
                    return capture.read()
                finally:
                    capture.release()

            ok, frame = await asyncio.wait_for(asyncio.to_thread(capture_frame), timeout=5.0)
            if not ok or frame is None:
                raise RuntimeError("V4L2 stream returned no frame")
            return AuditCheck(f"camera:{camera_id}", "pass", f"{camera_source}; frame={frame.shape}")
        except Exception as error:
            in_use = subprocess.run(
                ["fuser", camera_source], check=False, capture_output=True, text=True
            )
            if in_use.returncode == 0 and in_use.stdout.strip():
                return AuditCheck(
                    f"camera:{camera_id}",
                    "pass",
                    f"{camera_source}; active companion stream (pid {in_use.stdout.strip()})",
                )
            return AuditCheck(f"camera:{camera_id}", "fail", str(error))
    return AuditCheck(f"camera:{camera_id}", "pass", f"configured network source: {camera_source}")


def _module_check(module: str) -> AuditCheck:
    installed = importlib.util.find_spec(module) is not None
    return AuditCheck(
        f"python:{module}",
        "pass" if installed else "fail",
        "installed" if installed else "install project dependencies in the active environment",
    )


def _cuda_check() -> AuditCheck:
    try:
        import torch
    except ImportError:
        return AuditCheck("cuda", "fail", "PyTorch is not installed")
    if not torch.cuda.is_available():
        return AuditCheck("cuda", "fail", "PyTorch cannot access CUDA")
    return AuditCheck("cuda", "pass", f"{torch.cuda.get_device_name(0)}; CUDA {torch.version.cuda}")


def _gpu_runtime_pm_check() -> AuditCheck:
    for path in (
        Path("/sys/devices/platform/17000000.gpu/power"),
        Path("/sys/devices/platform/gpu.0/power"),
    ):
        if not path.exists():
            continue
        status = (path / "runtime_status").read_text().strip()
        control = (path / "control").read_text().strip()
        if status in {"error", "unsupported"}:
            return AuditCheck("gpu-runtime-pm", "fail", f"kernel runtime PM is {status}")
        if control not in {"auto", "on"}:
            return AuditCheck(
                "gpu-runtime-pm", "fail", f"runtime control is {control}, expected auto or on"
            )
        return AuditCheck("gpu-runtime-pm", "pass", f"status={status}; control={control}")
    return AuditCheck("gpu-runtime-pm", "fail", "Jetson GPU power controls are unavailable")


def _checkpoint_check(name: str, configured_path: str) -> AuditCheck:
    checkpoint = Path(configured_path)
    return AuditCheck(
        name,
        "pass" if checkpoint.is_file() and checkpoint.stat().st_size > 1_000_000 else "fail",
        str(checkpoint),
    )


def _memory_check(config: EggConfig) -> AuditCheck:
    try:
        store = MemoryStore(config.memory)
        try:
            report = store.integrity_report()
        finally:
            store.close()
        ready = (
            report["sqlite_integrity"] == "ok"
            and report["journal_mode"] == "wal"
            and report["writable"] is True
            and not report["foreign_key_violations"]
        )
        return AuditCheck(
            "cognitive-memory",
            "pass" if ready else "fail",
            f"integrity={report['sqlite_integrity']}; journal={report['journal_mode']}; "
            f"writable={report['writable']}; foreign_keys={len(report['foreign_key_violations'])}",
        )
    except Exception as error:
        return AuditCheck("cognitive-memory", "fail", f"{type(error).__name__}: {error}")


def _respeaker_audio_io_check(config: EggConfig) -> AuditCheck:
    try:
        import sounddevice as sound

        source = subprocess.run(
            ["pactl", "get-default-source"], check=True, capture_output=True, text=True
        ).stdout.strip()
        sink = subprocess.run(
            ["pactl", "get-default-sink"], check=True, capture_output=True, text=True
        ).stdout.strip()
        if "respeaker" not in source.lower() or "respeaker" not in sink.lower():
            raise RuntimeError(f"ReSpeaker hardware DSP route is not active: source={source}; sink={sink}")
        frames = sound.rec(
            int(config.audio.sample_rate * 0.25),
            samplerate=config.audio.sample_rate,
            channels=config.audio.channels,
            dtype="float32",
            device=config.audio.input_device,
        )
        sound.wait()
        with sound.OutputStream(
            samplerate=config.audio.sample_rate,
            channels=2,
            dtype="float32",
            device=config.audio.output_device,
        ):
            pass
        return AuditCheck("respeaker-audio-io", "pass", f"{source}; {sink}; capture={frames.shape}")
    except Exception as error:
        return AuditCheck("respeaker-audio-io", "fail", str(error))


def _degrade_unavailable_cameras(checks: list[AuditCheck]) -> list[AuditCheck]:
    if not any(check.status == "pass" for check in checks):
        return checks
    return [
        check
        if check.status == "pass"
        else AuditCheck(
            check.name,
            "warn",
            f"{check.detail}; unavailable camera ignored because another camera is ready",
        )
        for check in checks
    ]


async def audit_hardware(config: EggConfig) -> list[AuditCheck]:
    checks: list[AuditCheck] = [
        await _command_check("jetson-release", ["cat", "/etc/nv_tegra_release"]),
        await _command_check("video-devices", ["v4l2-ctl", "--list-devices"]),
        await _command_check("capture-devices", ["arecord", "-l"]),
        await _command_check("playback-devices", ["aplay", "-l"]),
        await _command_check("audio-playback", ["aplay", "--version"]),
        await _command_check("gpu-pm-guard", ["systemctl", "is-active", "egg-gpu-pm-guard.service"]),
        await _command_check("ornith-model", ["ollama", "show", config.omnius.vision_model]),
        *[
            _module_check(module)
            for module in ("cv2", "torch", "ultralytics", "open_clip", "sounddevice", "usb", "webrtcvad")
        ],
        _gpu_runtime_pm_check(),
        _cuda_check(),
        _checkpoint_check("detector-checkpoint", config.vision.detector_model),
        _checkpoint_check("pose-checkpoint", config.vision.pose_model),
        _checkpoint_check("sam-checkpoint", config.vision.sam_model),
        _checkpoint_check("sface-checkpoint", config.vision.sface_model),
        _memory_check(config),
    ]
    camera_checks = list(
        await asyncio.gather(
            *[
                _camera_check(camera.source, camera.id)
                for camera in config.cameras
                if camera.enabled
            ]
        )
    )
    camera_checks = _degrade_unavailable_cameras(camera_checks)
    checks.extend(camera_checks)
    checks.append(await asyncio.to_thread(_respeaker_audio_io_check, config))
    if config.audio.doa_mode == "respeaker_usb":
        device_path = Path(f"/sys/bus/usb/devices")
        expected = f"{config.audio.respeaker_vendor_id:04x}:{config.audio.respeaker_product_id:04x}"
        found = any(
            (candidate / "idVendor").exists()
            and (candidate / "idProduct").exists()
            and f"{(candidate / 'idVendor').read_text().strip()}:{(candidate / 'idProduct').read_text().strip()}" == expected
            for candidate in device_path.iterdir()
        )
        checks.append(AuditCheck("respeaker-usb", "pass" if found else "fail", expected))
        if found:
            try:
                angle = await asyncio.to_thread(read_respeaker_direction, config.audio)
                checks.append(AuditCheck("respeaker-direction", "pass", f"{angle:.0f} degrees"))
            except Exception as error:
                checks.append(AuditCheck("respeaker-direction", "fail", str(error)))
    elif config.audio.doa_mode == "serial" and config.audio.doa_serial_device:
        exists = Path(config.audio.doa_serial_device).exists()
        checks.append(
            AuditCheck(
                "respeaker-doa",
                "pass" if exists else "fail",
                config.audio.doa_serial_device,
            )
        )
    omnius = OmniusClient(config.omnius)
    omnius_ready = False
    try:
        await omnius.health()
        omnius_ready = True
        checks.append(AuditCheck("omnius", "pass", str(config.omnius.base_url)))
    except Exception as error:
        checks.append(AuditCheck("omnius", "fail", _error_detail(error)))
    if omnius_ready:
        try:
            catalog = await omnius.voice_catalog()
            tts_models = catalog.get("tts", {}).get("models", []) if isinstance(catalog.get("tts"), dict) else []
            asr_models = catalog.get("asr", {}).get("models", []) if isinstance(catalog.get("asr"), dict) else []
            checks.append(
                AuditCheck(
                    "omnius-voice-catalog",
                    "pass" if tts_models and asr_models else "fail",
                    f"tts={len(tts_models)}; asr={len(asr_models)}",
                )
            )
        except Exception as error:
            checks.append(AuditCheck("omnius-voice-catalog", "fail", _error_detail(error)))
        try:
            await omnius.ensure_voice_ready()
            voice_state = await omnius.voice_state()
            voice_ready = voice_state.get("voiceReady") is True
            checks.append(
                AuditCheck(
                    "omnius-voice",
                    "pass" if voice_ready else "fail",
                    f"ready={voice_state.get('voiceReady')}; model={voice_state.get('voiceModelId')}; error={voice_state.get('lastError')}",
                )
            )
        except Exception as error:
            checks.append(AuditCheck("omnius-voice", "fail", _error_detail(error)))
        try:
            cognition = await omnius.chat_contract_probe()
            checks.append(AuditCheck("omnius-cognition", "pass", cognition[:120]))
        except Exception as error:
            checks.append(AuditCheck("omnius-cognition", "fail", _error_detail(error)))
    if config.system_service:
        try:
            await SystemServiceClient(config.system_service).health()
            checks.append(AuditCheck("system-service", "pass", str(config.system_service.base_url)))
        except Exception as error:
            checks.append(AuditCheck("system-service", "fail", str(error)))
    return checks


def format_audit(checks: list[AuditCheck]) -> str:
    return json.dumps([asdict(check) for check in checks], indent=2)


def readiness_passes(checks: list[AuditCheck]) -> bool:
    """Return whether checks permit runtime startup; warnings describe degraded capacity."""
    return bool(checks) and all(check.status != "fail" for check in checks)
