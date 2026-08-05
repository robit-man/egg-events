from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from egg_companion.services.audit import (
    AuditCheck,
    _degrade_unavailable_cameras,
    _cuda_check,
    _gpu_runtime_pm_check,
    readiness_passes,
)


def test_gpu_runtime_pm_accepts_kernel_auto_control() -> None:
    def sysfs_value(path: Path) -> str:
        return {"runtime_status": "suspended", "control": "auto"}[path.name]

    with patch.object(Path, "exists", return_value=True), patch.object(
        Path, "read_text", autospec=True, side_effect=sysfs_value
    ):
        check = _gpu_runtime_pm_check()

    assert check == AuditCheck("gpu-runtime-pm", "pass", "status=suspended; control=auto")


def test_cuda_probe_is_direct_and_not_suppressed_by_runtime_pm() -> None:
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _: "Test GPU",
        ),
        version=SimpleNamespace(cuda="12.6"),
    )
    with patch.dict("sys.modules", {"torch": torch}), patch(
        "egg_companion.services.audit._gpu_runtime_pm_check",
        return_value=AuditCheck("gpu-runtime-pm", "fail", "unrelated PM diagnostic"),
    ) as pm_check:
        check = _cuda_check()

    assert check == AuditCheck("cuda", "pass", "Test GPU; CUDA 12.6")
    pm_check.assert_not_called()


def test_readiness_accepts_degraded_camera_warning() -> None:
    checks = [
        AuditCheck("cuda", "pass", "available"),
        AuditCheck("camera:camera-0", "pass", "frame available"),
        AuditCheck("camera:camera-1", "warn", "camera unavailable"),
    ]

    assert readiness_passes(checks)


def test_readiness_rejects_real_failure() -> None:
    assert not readiness_passes([AuditCheck("cuda", "fail", "unavailable")])


def test_one_ready_camera_downgrades_other_camera_failures() -> None:
    checks = _degrade_unavailable_cameras(
        [
            AuditCheck("camera:ready", "pass", "frame available"),
            AuditCheck("camera:missing", "fail", "cannot open"),
        ]
    )

    assert [check.status for check in checks] == ["pass", "warn"]
    assert readiness_passes(checks)


def test_zero_ready_cameras_remains_a_diagnostic_failure() -> None:
    checks = _degrade_unavailable_cameras(
        [AuditCheck("camera:missing", "fail", "cannot open")]
    )

    assert checks[0].status == "fail"
