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
    # Unset (default) leaves the device at whatever resolution its driver
    # opens with -- many UVC webcams default to a low mode (e.g. 640x480)
    # until a higher one is explicitly requested. Set both to opt this
    # camera into a higher native capture resolution (e.g. 3840x2160 for
    # a 4K-capable sensor) so the occupancy/depth pipeline has real detail
    # to work with instead of a driver-default low-res frame.
    capture_width: int | None = Field(default=None, ge=320, le=7680)
    capture_height: int | None = Field(default=None, ge=240, le=4320)

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
    # Applied to every auto-discovered camera -- see CameraConfig.
    # capture_width/capture_height for what these do.
    capture_width: int | None = Field(default=None, ge=320, le=7680)
    capture_height: int | None = Field(default=None, ge=240, le=4320)

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
    # CLIP is background semantic/recall work. It can live on CPU while the
    # detector stays on CUDA, preserving GPU headroom for realtime language,
    # speech, and explicitly requested VLM work.
    clip_device: str | None = None
    pose_enabled: bool = True
    semantic_enabled: bool = True
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
    respeaker_led_enabled: bool = True
    respeaker_led_brightness: int = Field(default=8, ge=0, le=31)
    sample_rate: int = Field(default=16000, gt=0)
    channels: int = Field(default=1, ge=1, le=8)
    asr_channel: int = Field(default=0, ge=0, le=7)
    asr_target_rms: float = Field(default=0.08, gt=0, le=1)
    asr_max_gain: float = Field(default=24.0, ge=1, le=48)
    barge_in_enabled: bool = True
    playback_resume_rewind_ms: int = Field(default=80, ge=0, le=500)
    playback_timeout_seconds: float = Field(default=30, gt=0, le=300)
    waveform_fps: int = Field(default=30, ge=10, le=60)
    waveform_samples: int = Field(default=256, ge=64, le=1024)


class TranscriptionConfig(BaseModel):
    # segment_seconds is the hard cap on a single utterance's length, not a fixed
    # capture window: utterances are bounded by VAD onset/hangover (see
    # vad_min_contiguous_ms / vad_hangover_ms) so speech is never chopped mid-word.
    segment_seconds: float = Field(default=12.0, gt=0, le=15)
    rms_threshold: float = Field(default=0.012, gt=0, le=1)
    asr_model: str = "medium"
    asr_language: str = Field(default="en", pattern=r"^(auto|[a-z]{2,3}(?:-[A-Z]{2})?)$")
    vad_aggressiveness: int = Field(default=2, ge=0, le=3)
    vad_input_gain: float = Field(default=10.0, ge=1, le=32)
    vad_min_speech_ms: int = Field(default=240, ge=30, le=3000)
    vad_min_speech_ratio: float = Field(default=0.12, ge=0, le=1)
    vad_min_contiguous_ms: int = Field(default=180, ge=30, le=3000)
    vad_min_voiced_rms: float = Field(default=0.008, gt=0, le=1)
    vad_pre_roll_ms: float = Field(default=300, ge=0, le=2000)
    vad_hangover_ms: float = Field(default=600, ge=100, le=5000)
    # When set above vad_hangover_ms, trailing silence grows toward this bound
    # as voiced duration and natural pause continuations accumulate.
    vad_hangover_max_ms: float | None = Field(default=None, ge=100, le=5000)
    vad_hangover_growth_ms: float = Field(default=1600, gt=0, le=20000)
    vad_continuation_growth: float = Field(default=1.0, ge=0, le=8)


class AudioComprehensionConfig(BaseModel):
    """Bounded, non-blocking semantic analysis of admitted room audio."""

    enabled: bool = True
    queue_size: int = Field(default=1, ge=1, le=8)
    minimum_interval_seconds: float = Field(default=15.0, ge=0, le=3600)
    minimum_confidence: float = Field(default=0.12, ge=0, le=1)
    top_k: int = Field(default=5, ge=1, le=20)
    context_ttl_seconds: float = Field(default=90.0, gt=0, le=3600)


class OmniusConfig(BaseModel):
    base_url: HttpUrl = "http://127.0.0.1:11435"
    asr_base_url: HttpUrl | None = None
    model: str = "robit/ornith-1.5:9b"
    vision_model: str = "robit/ornith-1.5:9b"
    vision_base_url: HttpUrl = "http://127.0.0.1:11434"
    voice_model: str
    voice_name: str | None = None
    bearer_token_env: str | None = None
    timeout_seconds: float = Field(default=20, gt=0, le=120)
    # Omnius forwards this bounded Ollama context for realtime chat. Without
    # an explicit value, namespaced Ornith manifests request their full 262K
    # training window and consume memory needed by ASR/TTS/VLM runtimes.
    chat_num_ctx: int = Field(default=4096, ge=2048, le=32768)
    # Keep enough alternating heard/agent messages for natural follow-ups while
    # leaving most of the bounded prompt window to grounded cognitive context.
    chat_history_messages: int = Field(default=12, ge=4, le=24)
    # Retain the single multimodal model between spoken turns so visual work
    # and replies do not repeatedly pay a multi-second load/eviction penalty.
    chat_keep_alive: str = "30m"
    # Spoken turns use Omnius's direct realtime backend and explicitly disable
    # hidden reasoning. The separate LLM router is optional because it adds a
    # full serial generation before every reply.
    reasoning_enabled: bool = False
    dialogue_router_enabled: bool = False
    visual_snapshot_max_age_seconds: float = Field(default=2.5, gt=0, le=15)
    visual_snapshot_max_cameras: int = Field(default=4, ge=1, le=16)
    visual_contact_sheet_size: int = Field(default=768, ge=512, le=1536)


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
    # Identity calibration is deliberately separate from generic proactive
    # commentary. A stable face may be asked once for a preferred name even
    # when unsolicited scene narration is disabled.
    identity_question_enabled: bool = True
    # A persistent profile already required IdentityConfig's multi-frame face
    # enrollment, so its first durable sighting is sufficiently grounded.
    identity_question_min_sightings: int = Field(default=1, ge=1, le=100)
    identity_question_timeout_seconds: float = Field(default=300, gt=0, le=3600)
    identity_question_cooldown_seconds: float = Field(default=120, ge=0, le=86400)


class ActivityConfig(BaseModel):
    """Novelty/presence/sound-driven falloff for perception frequency.

    Inference (detection, pose, semantics, OCR) runs at its full configured
    rate while the room holds activity, and decays toward an idle floor after
    `decay_seconds` of a genuinely empty, silent scene -- the same way a quiet
    room needs less visual/auditory vigilance than a busy one. Any new
    novelty, presence, or speech snaps the rate back to full immediately.
    """

    enabled: bool = True
    idle_floor: float = Field(default=0.15, ge=0.02, le=1.0)
    decay_seconds: float = Field(default=20.0, gt=0, le=600)
    novelty_threshold: float = Field(default=0.12, ge=0, le=1)


class IdentityConfig(BaseModel):
    enabled: bool = True
    storage_dir: str = "data/identity-library"
    # Whole-body CLIP features describe appearance/semantics; they are not
    # permitted to create durable people.  They remain available elsewhere for
    # scene and object understanding.
    similarity_threshold: float = Field(default=0.88, ge=0, le=1)
    face_similarity_threshold: float = Field(default=0.45, ge=0, le=1)
    face_match_margin: float = Field(default=0.04, ge=0, le=1)
    minimum_face_quality: float = Field(default=0.75, ge=0, le=1)
    enrollment_min_face_observations: int = Field(default=3, ge=2, le=20)
    enrollment_face_consistency: float = Field(default=0.65, ge=0, le=1)
    retroactive_coalescing_enabled: bool = True
    retroactive_merge_similarity: float = Field(default=0.80, ge=0, le=1)
    track_ttl_seconds: float = Field(default=8.0, gt=0, le=120)
    track_iou_threshold: float = Field(default=0.18, ge=0, le=1)
    track_center_distance: float = Field(default=0.65, gt=0, le=3)
    track_mask_iou_threshold: float = Field(default=0.30, ge=0, le=1)
    track_mask_containment_threshold: float = Field(default=0.70, ge=0, le=1)
    track_mask_max_gap_seconds: float = Field(default=8.0, gt=0, le=30)
    temporal_vlm_comparison_enabled: bool = True
    temporal_vlm_queue_size: int = Field(default=2, ge=1, le=16)
    temporal_vlm_cooldown_seconds: float = Field(default=15.0, ge=0, le=3600)
    sample_interval_seconds: float = Field(default=15, gt=0)
    gallery_max_samples: int = Field(default=8, ge=2, le=32)
    gallery_diversity_similarity: float = Field(default=0.985, ge=0, le=1)


class DreamsConfig(BaseModel):
    """Idle-time, bounded offline learning and identity consolidation."""

    enabled: bool = True
    model_path: str = "models/cvlface_adaface_ir18_webface4m"
    model_id: str = "minchul/cvlface_adaface_ir18_webface4m"
    model_revision: str = "0dd53f188fa27968b0a1326970ebf4aeb37ce2ca"
    device: str = "cuda"
    batch_size: int = Field(default=64, ge=1, le=256)
    use_half_precision: bool = True
    idle_seconds: float = Field(default=45, ge=5, le=3600)
    interval_min_seconds: float = Field(default=600, ge=30, le=86400)
    interval_max_seconds: float = Field(default=1800, ge=30, le=172800)
    convergence_interval_seconds: float = Field(default=60, ge=15, le=3600)
    # Raised from the original defaults (0.35/0.40/0.24/0.30/...) after
    # live evidence: those thresholds let a 3-model "consensus" merge 86 of
    # 95 total identity aliases at 0.35-0.65 average similarity, including
    # absorbing a well-established named profile (1327 samples) into an
    # unnamed fragment at just 0.41 similarity. Cosine similarity between
    # face embeddings of genuinely the same person is typically well above
    # 0.6 for these model families; scores in the 0.3-0.5 band are common
    # between DIFFERENT people who simply share generic facial structure,
    # pose, or lighting. These values plus the VLM confirmation gate below
    # (see IdentityDreamEngine.run's verifier param) are the actual fix.
    proposal_similarity: float = Field(default=0.55, ge=-1, le=1)
    modern_merge_similarity: float = Field(default=0.62, ge=-1, le=1)
    modern_strong_similarity: float = Field(default=0.72, ge=-1, le=1)
    legacy_merge_similarity: float = Field(default=0.45, ge=-1, le=1)
    legacy_strong_similarity: float = Field(default=0.70, ge=-1, le=1)
    legacy_similarity_floor: float = Field(default=0.30, ge=-1, le=1)
    comparison_model_path: str | None = None
    comparison_model_id: str = "insightface/buffalo_s-w600k_mbf"
    comparison_merge_similarity: float = Field(default=0.50, ge=-1, le=1)
    comparison_strong_similarity: float = Field(default=0.68, ge=-1, le=1)
    comparison_similarity_floor: float = Field(default=0.32, ge=-1, le=1)
    minimum_model_votes: int = Field(default=2, ge=2, le=3)
    separated_modern_similarity: float = Field(default=0.58, ge=-1, le=1)
    separated_legacy_floor: float = Field(default=0.32, ge=-1, le=1)
    mutual_neighbor_margin: float = Field(default=0.025, ge=0, le=1)
    reciprocal_neighbor_rank: int = Field(default=8, ge=1, le=20)
    coobservation_min_confirmations: int = Field(default=3, ge=1, le=100)
    auto_merge_enabled: bool = True
    # Minimum Ornith confidence (see IdentityDreamEngine.run's verifier
    # param / OmniusClient.compare_identity_profiles) to accept a same_person
    # confirmation for an embedding-consensus merge proposal.
    vlm_confirmation_min_confidence: float = Field(default=0.6, ge=0, le=1)


class ObjectLearningConfig(BaseModel):
    enabled: bool = True
    storage_dir: str = "data/object-library"
    similarity_threshold: float = Field(default=0.86, ge=0, le=1)
    auto_label_enabled: bool = True
    adjudication_queue_size: int = Field(default=12, ge=1, le=64)
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
    duplicate_proposal_similarity: float = Field(default=0.94, ge=0, le=1)
    duplicate_adjudication_batch_size: int = Field(default=1, ge=0, le=8)


class OcrRefinementConfig(BaseModel):
    enabled: bool = True
    local_confidence_threshold: float = Field(default=0.72, ge=0.1, le=1.0)
    min_text_length_for_refinement: int = Field(default=6, ge=1, le=100)
    max_refinements_per_minute: int = Field(default=20, ge=1, le=120)


class OcrDedupConfig(BaseModel):
    enabled: bool = True
    window_seconds: float = Field(default=300.0, ge=30.0, le=3600.0)
    hash_size: int = Field(default=8, ge=4, le=16)


class OcrBackfillConfig(BaseModel):
    enabled: bool = True
    scan_interval_seconds: float = Field(default=60.0, ge=10.0, le=600.0)
    batch_size: int = Field(default=4, ge=1, le=16)


class OcrConfig(BaseModel):
    enabled: bool = True
    local_multipass_enabled: bool = True
    omnius_refinement_enabled: bool = True
    vlm_text_detection_enabled: bool = True
    full_frame_interval_seconds: float = Field(default=20, ge=2, le=3600)
    text_object_interval_seconds: float = Field(default=8, ge=1, le=3600)
    queue_size: int = Field(default=8, ge=1, le=64)
    max_image_size: int = Field(default=1280, ge=320, le=2560)
    min_text_characters: int = Field(default=2, ge=1, le=100)
    max_fragments: int = Field(default=8, ge=1, le=32)
    max_region_refinements: int = Field(default=2, ge=0, le=8)
    vlm_text_check_interval: float = Field(default=5.0, ge=1.0, le=30.0)
    ledger_db_path: str = "data/ocr-jobs.sqlite3"
    refinement: OcrRefinementConfig = Field(default_factory=OcrRefinementConfig)
    dedup: OcrDedupConfig = Field(default_factory=OcrDedupConfig)
    backfill: OcrBackfillConfig = Field(default_factory=OcrBackfillConfig)


class OccupancyConfig(BaseModel):
    """Fused voxel occupancy mapping via on-demand monocular metric depth.

    The four cameras are a known panoramic array (see
    camera_yaw_degrees), so every camera's depth is rotated into one
    shared "egg frame" grid rather than kept in disconnected per-camera
    frames -- see core/occupancy.py's module docstring for the exact
    fusion geometry. There's still no calibrated per-camera intrinsics
    (assumed_hfov_degrees is an estimate, not a measurement), so this is
    accurate multi-view *reconstruction* within that limit, not
    navigation-grade metric precision.

    The depth model runs as a subprocess in a separate, pre-existing
    Python environment (not this project's venv) and is loaded fresh and
    torn down on every cycle -- this hardware doesn't have the memory
    headroom to keep a ~4GB model resident alongside everything else.
    """

    # Defaults to disabled: live testing on this hardware showed swap usage
    # climb noticeably from just a handful of on-demand depth cycles on a
    # system that already runs with well under 1GB of free memory. Built,
    # tested, and verified working end-to-end against the real model --
    # opt in via config/egg.yaml once comfortable with the memory tradeoff.
    enabled: bool = False
    depth_venv_python: str = "/home/egg/Depth-Anything-3/venv/bin/python"
    depth_worker_script: str = "scripts/depth_worker.py"
    depth_repo_dir: str = "/home/egg/Depth-Anything-3"
    model_name: str = "depth-anything/DA3METRIC-LARGE"
    # Default (504) is the memory-safe value this hardware was validated
    # against (see update_interval_seconds' comment on swap pressure) --
    # the ceiling is raised to 3840 so a deployment with the memory/GPU
    # headroom for it can opt into processing true full-4K input end to
    # end (this is DA3's own internal working resolution, independent of
    # occupancy.max_input_width below, which controls what resolution the
    # source frame is encoded at before being handed to the model at all).
    process_res: int = Field(default=504, ge=128, le=3840)
    subprocess_timeout_seconds: float = Field(default=90.0, ge=10.0, le=600.0)
    # Conservative default: live testing on this hardware showed swap usage
    # climb noticeably (5Gi -> 8.9Gi) across a handful of depth cycles on a
    # system that already runs with well under 1GB of free memory. With 4
    # cameras staggered at this interval, a full sweep happens roughly
    # every 4 minutes rather than contending for memory every ~15s.
    update_interval_seconds: float = Field(default=180.0, ge=10.0, le=3600.0)
    voxel_size_meters: float = Field(default=0.1, gt=0.01, le=2.0)
    max_range_meters: float = Field(default=6.0, gt=0.5, le=50.0)
    min_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    # No calibrated intrinsics exist for any camera; assumed_hfov_degrees
    # is a coarse stand-in used only to back-project depth into a rough
    # volume -- fine for "is space near X roughly occupied" reasoning, not
    # for anything requiring true metric precision. Defaults to the same
    # 60 degrees the array is described as stitching at, since adjacent
    # cameras' edges meeting implies each one's own FOV is close to that.
    assumed_hfov_degrees: float = Field(default=60.0, ge=20.0, le=170.0)
    # Degrees between adjacent cameras in the array, positive = counter-
    # clockwise from above. Every camera's yaw is auto-computed from its
    # parsed trailing index among whatever cameras are currently live
    # (see core/occupancy.py's resolve_camera_yaw_degrees), evenly spaced
    # by this and centered on the array midpoint -- so a camera-video4/
    # video5 discovered later is automatically placed, and the whole
    # array recenters correctly rather than defaulting new cameras to
    # yaw=0 or needing a hardcoded per-camera-count mapping. The default
    # (60.0) reproduces the physically-mounted 4-camera rig's exact
    # -90/-30/30/90 spacing for N=4, so this changes no existing behavior.
    camera_array_spacing_degrees: float = Field(default=60.0, ge=1.0, le=180.0)
    # camera_id -> yaw in degrees: an explicit override for a specific
    # camera, taking priority over the auto-computed value above (e.g. if
    # the real mounting isn't perfectly evenly spaced). Empty by default.
    camera_yaw_degrees: dict[str, float] = Field(default_factory=dict)
    # Every Nth depth-map pixel (in both row and column) gets back-
    # projected into a voxel per integration cycle -- lower means more of
    # DA3's actual per-frame points get used (denser reconstruction, more
    # CPU per cycle), higher means fewer (coarser, cheaper). Adjustable
    # live from the dashboard's Resolution +/- control.
    sample_stride: int = Field(default=8, ge=1, le=32)
    # Cap on the frame width handed to the depth model, applied only to
    # the occupancy pipeline's own encode of the source camera frame --
    # NOT the same as vision.dashboard_max_width, which caps the lossy
    # JPEG preview stream shown in the dashboard. Defaults to true 4K
    # width so a camera capturing at 3840px (see CameraConfig.
    # capture_width) is passed through to depth estimation untouched
    # rather than silently cropped down to a preview-sized frame.
    max_input_width: int = Field(default=3840, ge=320, le=7680)
    max_voxels: int = Field(default=60_000, ge=1_000, le=1_000_000)
    stale_after_seconds: float = Field(default=1800.0, ge=60.0, le=86400.0)


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
    epistemic_value_weight: float = Field(default=0.18, ge=0, le=1)
    observation_policy_weight: float = Field(default=0.22, ge=0, le=1)
    graph_familiarity_discount: float = Field(default=0.85, ge=0, le=1)
    irreducible_uncertainty_discount: float = Field(default=0.65, ge=0, le=1)
    interruption_threshold: float = Field(default=0.75, ge=0, le=1)
    communicative_action_threshold: float = Field(default=0.35, ge=0, le=1)
    proactive_rate_limit_seconds: float = Field(default=90, gt=0, le=3600)
    # Generic low-confidence label interrogation is disabled by default; the
    # source-backed default-mode curiosity contract is the proactive path.
    uncertainty_question_budget_per_hour: int = Field(default=0, ge=0, le=100)


class DefaultModeConfig(BaseModel):
    """Bounded quiet-period provenance replay and model-dream controls."""

    enabled: bool = True
    idle_seconds: float = Field(default=45, ge=5, le=3600)
    interval_min_seconds: float = Field(default=60, ge=15, le=86400)
    interval_max_seconds: float = Field(default=180, ge=15, le=172800)
    replay_limit: int = Field(default=8, ge=1, le=100)
    proactive_budget_per_hour: int = Field(default=2, ge=0, le=20)
    proactive_cooldown_seconds: float = Field(default=300, ge=0, le=86400)
    question_timeout_seconds: float = Field(default=240, gt=0, le=3600)
    meta_graph_limit: int = Field(default=24, ge=1, le=200)
    entity_summary_limit: int = Field(default=12, ge=1, le=100)
    document_context_characters: int = Field(default=1800, ge=400, le=8000)
    narrative_timezone: str = "local"
    narrative_replay_max_days: int = Field(default=30, ge=1, le=3650)
    narrative_bucket_minutes: int = Field(default=15, ge=1, le=180)
    narrative_max_entries: int = Field(default=96, ge=8, le=288)


class SocialCognitionConfig(BaseModel):
    """Evidence-bound conversational affect and interaction reflection."""

    enabled: bool = True
    queue_size: int = Field(default=8, ge=1, le=64)
    history_turns: int = Field(default=24, ge=4, le=200)
    profile_context_characters: int = Field(default=1800, ge=400, le=8000)


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


class WorldPruningConfig(BaseModel):
    """World model pruning and hallucination removal settings."""
    enabled: bool = True
    min_confidence: float = Field(default=0.45, ge=0, le=1)
    stale_after_hours: float = Field(default=24.0, ge=1, le=168)
    max_det_entities: int = Field(default=200, ge=10, le=2000)
    prune_every_n_dreams: int = Field(default=3, ge=1, le=20)
    impossible_labels_file: str = ""


class EggConfig(BaseModel):
    cameras: list[CameraConfig] = Field(default_factory=list)
    camera_discovery: CameraDiscoveryConfig = Field(default_factory=CameraDiscoveryConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    audio: AudioConfig
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    audio_comprehension: AudioComprehensionConfig = Field(
        default_factory=AudioComprehensionConfig
    )
    omnius: OmniusConfig
    system_service: SystemServiceConfig | None = None
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    activity: ActivityConfig = Field(default_factory=ActivityConfig)
    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    dreams: DreamsConfig = Field(default_factory=DreamsConfig)
    object_learning: ObjectLearningConfig = Field(default_factory=ObjectLearningConfig)
    ocr: OcrConfig = Field(default_factory=OcrConfig)
    occupancy: OccupancyConfig = Field(default_factory=OccupancyConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    event_segmentation: EventSegmentationConfig = Field(default_factory=EventSegmentationConfig)
    cognitive_attention: CognitiveAttentionConfig = Field(default_factory=CognitiveAttentionConfig)
    default_mode: DefaultModeConfig = Field(default_factory=DefaultModeConfig)
    social_cognition: SocialCognitionConfig = Field(
        default_factory=SocialCognitionConfig
    )
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    world_pruning: WorldPruningConfig = Field(default_factory=WorldPruningConfig)

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
                capture_width=config.camera_discovery.capture_width,
                capture_height=config.camera_discovery.capture_height,
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
