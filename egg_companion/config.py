from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, HttpUrl, field_validator


class CameraConfig(BaseModel):
    id: str
    source: str
    fps: float = Field(default=8.0, gt=0, le=60)
    rotation_degrees: int | Literal["auto"] = "auto"
    enabled: bool = True

    @field_validator("source")
    @classmethod
    def validate_source(cls, source: str) -> str:
        if not source.startswith(("/dev/video", "rtsp://", "v4l2://")):
            raise ValueError("source must be a /dev/video*, v4l2://, or rtsp:// endpoint")
        return source

    @field_validator("rotation_degrees")
    @classmethod
    def validate_rotation(cls, rotation: int | str) -> int | str:
        if rotation != "auto" and rotation not in {0, 90, 180, 270}:
            raise ValueError("rotation_degrees must be auto, 0, 90, 180, or 270")
        return rotation


class CameraDiscoveryConfig(BaseModel):
    enabled: bool = True
    source_glob: str = "/dev/video*"
    fps: float = Field(default=8.0, gt=0, le=60)
    rotation_degrees: int | Literal["auto"] = "auto"

    @field_validator("source_glob")
    @classmethod
    def validate_source_glob(cls, source_glob: str) -> str:
        if not source_glob.startswith("/dev/"):
            raise ValueError("camera discovery source_glob must target /dev")
        return source_glob

    @field_validator("rotation_degrees")
    @classmethod
    def validate_rotation(cls, rotation: int | str) -> int | str:
        if rotation != "auto" and rotation not in {0, 90, 180, 270}:
            raise ValueError("rotation_degrees must be auto, 0, 90, 180, or 270")
        return rotation


class VisionConfig(BaseModel):
    detector_model: str = "models/yoloe-11s-seg-pf.pt"
    pose_model: str = "models/yolo11n-pose.pt"
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    device: str = "cuda"
    confidence_threshold: float = Field(default=0.45, ge=0, le=1)
    sam_model: str = "models/sam2.1_t.pt"
    sam_image_size: int = Field(default=640, ge=320, le=1280)
    sface_model: str = "models/face_recognition_sface_2021dec.onnx"
    analysis_fps: float = Field(default=2.0, gt=0, le=15)
    pose_fps: float = Field(default=0.75, gt=0, le=15)
    semantic_fps: float = Field(default=0.25, gt=0, le=15)
    dashboard_fps: float = Field(default=8.0, gt=0, le=15)
    dashboard_max_width: int = Field(default=960, ge=320, le=1920)
    max_instances: int = Field(default=24, ge=1, le=100)
    minimum_detector_classes: int = Field(default=1000, ge=80, le=10000)
    semantic_prompts: list[str] = Field(
        default_factory=lambda: [
            "a person approaching the device",
            "a person speaking to the device",
            "a person waving",
            "a seated person",
            "a group of people",
            "a pet or animal",
            "an unattended package",
            "a door opening",
        ]
    )


class AudioConfig(BaseModel):
    input_device: str
    output_device: str = "default"
    doa_mode: str = "respeaker_usb"
    respeaker_vendor_id: int = Field(default=0x2886, ge=0, le=65535)
    respeaker_product_id: int = Field(default=0x0018, ge=0, le=65535)
    doa_serial_device: str | None = None
    sample_rate: int = Field(default=16000, gt=0)
    channels: int = Field(default=1, ge=1, le=8)
    asr_channel: int = Field(default=0, ge=0, le=7)
    asr_target_rms: float = Field(default=0.08, gt=0, le=1)
    asr_max_gain: float = Field(default=24.0, ge=1, le=48)
    waveform_fps: int = Field(default=30, ge=10, le=60)
    waveform_samples: int = Field(default=256, ge=64, le=1024)


class TranscriptionConfig(BaseModel):
    # segment_seconds is the hard cap on a single utterance's length, not a fixed
    # capture window: utterances are bounded by VAD onset/hangover (see
    # vad_min_contiguous_ms / vad_hangover_ms) so speech is never chopped mid-word.
    segment_seconds: float = Field(default=3.0, gt=0, le=15)
    rms_threshold: float = Field(default=0.012, gt=0, le=1)
    asr_model: str = "medium"
    vad_aggressiveness: int = Field(default=2, ge=0, le=3)
    vad_input_gain: float = Field(default=10.0, ge=1, le=32)
    vad_min_speech_ms: int = Field(default=240, ge=30, le=3000)
    vad_min_speech_ratio: float = Field(default=0.12, ge=0, le=1)
    vad_min_contiguous_ms: int = Field(default=180, ge=30, le=3000)
    vad_min_voiced_rms: float = Field(default=0.008, gt=0, le=1)
    vad_pre_roll_ms: float = Field(default=300, ge=0, le=2000)
    vad_hangover_ms: float = Field(default=600, ge=100, le=5000)


class OmniusConfig(BaseModel):
    base_url: HttpUrl = "http://127.0.0.1:11435"
    model: str
    vision_model: str = "robit/ornith-vision:9b"
    vision_base_url: HttpUrl = "http://127.0.0.1:11434"
    voice_model: str
    voice_name: str | None = None
    bearer_token_env: str | None = None
    timeout_seconds: float = Field(default=20, gt=0, le=120)


class SystemServiceConfig(BaseModel):
    base_url: HttpUrl
    status_path: str = "/health"
    event_path: str = "/events"
    bearer_token_env: str | None = None


class AttentionConfig(BaseModel):
    max_targets: int = Field(default=1, ge=1, le=5)
    track_ttl_seconds: float = Field(default=10, gt=0)
    min_priority: float = Field(default=0.35, ge=0, le=1)
    greeting_cooldown_seconds: float = Field(default=45, gt=0)
    proactive_speech_enabled: bool = False


class IdentityConfig(BaseModel):
    enabled: bool = True
    storage_dir: str = "data/identity-library"
    similarity_threshold: float = Field(default=0.88, ge=0, le=1)
    face_similarity_threshold: float = Field(default=0.45, ge=0, le=1)
    sample_interval_seconds: float = Field(default=15, gt=0)


class ObjectLearningConfig(BaseModel):
    enabled: bool = True
    storage_dir: str = "data/object-library"
    similarity_threshold: float = Field(default=0.86, ge=0, le=1)
    auto_label_enabled: bool = True
    auto_label_confidence_threshold: float = Field(default=0.68, ge=0, le=1)
    auto_label_min_confidence: float = Field(default=0.70, ge=0, le=1)
    auto_label_cooldown_seconds: float = Field(default=6, gt=0, le=600)
    recall_interval_seconds: float = Field(default=0.75, gt=0, le=60)
    recall_cache_seconds: float = Field(default=12, gt=0, le=300)
    auto_label_max_retries: int = Field(default=2, ge=0, le=8)
    auto_label_failure_backoff_seconds: float = Field(default=10, gt=0, le=600)
    vlm_max_image_size: int = Field(default=512, ge=224, le=1024)
    stable_candidate_frames: int = Field(default=3, ge=2, le=20)
    speech_priority_seconds: float = Field(default=8, ge=0, le=120)
    review_sweep_interval_seconds: float = Field(default=900, gt=0, le=86400)
    review_stale_after_seconds: float = Field(default=21600, gt=0, le=604800)
    confidence_audit_enabled: bool = True
    confidence_audit_batch_size: int = Field(default=5, ge=1, le=50)


class MemoryConfig(BaseModel):
    enabled: bool = True
    storage_dir: str = "data/cognitive-memory"
    migration_mode: Literal["legacy", "dual_write", "graph"] = "dual_write"
    retain_raw_media: bool = True
    raw_media_retention_hours: int = Field(default=72, ge=0, le=8760)
    episode_min_seconds: float = Field(default=1.0, gt=0, le=60)
    episode_max_seconds: float = Field(default=90.0, gt=1, le=3600)
    retrieval_limit: int = Field(default=12, ge=1, le=100)
    graph_max_hops: int = Field(default=2, ge=1, le=6)
    graph_max_nodes: int = Field(default=80, ge=4, le=1000)
    context_max_characters: int = Field(default=5000, ge=500, le=20000)
    consolidation_interval_seconds: float = Field(default=300, ge=30, le=86400)
    consolidation_batch_size: int = Field(default=20, ge=1, le=200)
    buffer_frames_per_camera: int = Field(default=24, ge=1, le=240)
    buffer_audio_segments: int = Field(default=16, ge=1, le=128)
    buffer_ttl_seconds: float = Field(default=120, gt=0, le=3600)
    buffer_max_bytes: int = Field(default=16_777_216, ge=1_048_576, le=536_870_912)


class EventSegmentationConfig(BaseModel):
    inactivity_seconds: float = Field(default=8.0, gt=0, le=300)
    entity_change_threshold: float = Field(default=0.45, ge=0, le=1)
    semantic_change_threshold: float = Field(default=0.45, ge=0, le=1)
    doa_change_degrees: float = Field(default=35.0, gt=0, le=180)
    speech_boundary_seconds: float = Field(default=1.0, gt=0, le=30)


class CognitiveAttentionConfig(BaseModel):
    new_entity_weight: float = Field(default=0.35, ge=0, le=1)
    action_change_weight: float = Field(default=0.20, ge=0, le=1)
    speech_weight: float = Field(default=0.30, ge=0, le=1)
    prediction_error_weight: float = Field(default=0.15, ge=0, le=1)
    interruption_threshold: float = Field(default=0.75, ge=0, le=1)
    proactive_rate_limit_seconds: float = Field(default=90, gt=0, le=3600)
    uncertainty_question_budget_per_hour: int = Field(default=4, ge=0, le=100)


class PrivacyConfig(BaseModel):
    persistent_identity_enabled: bool = True
    profile_retention_days: int = Field(default=30, ge=0, le=3650)
    evidence_retention_days: int = Field(default=30, ge=0, le=3650)
    export_enabled: bool = True
    deletion_enabled: bool = True


class RuntimeConfig(BaseModel):
    event_queue_size: int = Field(default=8, ge=1, le=100)
    speech_queue_size: int = Field(default=4, ge=1, le=32)
    reasoning_queue_size: int = Field(default=4, ge=1, le=32)
    log_level: str = "INFO"


class EggConfig(BaseModel):
    cameras: list[CameraConfig] = Field(default_factory=list)
    camera_discovery: CameraDiscoveryConfig = Field(default_factory=CameraDiscoveryConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    audio: AudioConfig
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    omnius: OmniusConfig
    system_service: SystemServiceConfig | None = None
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    object_learning: ObjectLearningConfig = Field(default_factory=ObjectLearningConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    event_segmentation: EventSegmentationConfig = Field(default_factory=EventSegmentationConfig)
    cognitive_attention: CognitiveAttentionConfig = Field(default_factory=CognitiveAttentionConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @field_validator("cameras")
    @classmethod
    def require_enabled_camera(cls, cameras: list[CameraConfig]) -> list[CameraConfig]:
        ids = [camera.id for camera in cameras]
        if len(ids) != len(set(ids)):
            raise ValueError("camera ids must be unique")
        return cameras


def _device_sort_key(source: str) -> tuple[int, str]:
    suffix = Path(source).name.removeprefix("video")
    return (int(suffix), source) if suffix.isdecimal() else (10**9, source)


def _discover_cameras(config: EggConfig) -> list[CameraConfig]:
    if not config.camera_discovery.enabled:
        return []
    configured_sources = {camera.source for camera in config.cameras}
    cameras: list[CameraConfig] = []
    for source in sorted(glob(config.camera_discovery.source_glob), key=_device_sort_key):
        path = Path(source)
        if source in configured_sources or not path.exists() or not path.is_char_device():
            continue
        cameras.append(
            CameraConfig(
                id=f"camera-{path.name}",
                source=source,
                fps=config.camera_discovery.fps,
                rotation_degrees=config.camera_discovery.rotation_degrees,
            )
        )
    return cameras


def load_config(path: str | Path) -> EggConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)
    if not isinstance(raw_config, dict):
        raise ValueError(f"configuration {config_path} must be a YAML mapping")
    config = EggConfig.model_validate(raw_config)
    config.cameras.extend(_discover_cameras(config))
    if not any(camera.enabled for camera in config.cameras):
        raise ValueError("no enabled cameras were configured or discovered")
    return config
