from __future__ import annotations

import logging
from glob import glob
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

logger = logging.getLogger(__name__)


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
    # The local discrete-model perception stack: YOLOE detection/segmentation,
    # YOLO pose, SAM, CLIP, and the ONNX face detector/recognizer. Omni mode
    # switches this off wholesale -- see OmniAdapterConfig.exclusive -- because
    # the point of the single weights package is that none of these load.
    enabled: bool = True
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
    # `model` and `vision_model` name the same underlying Ollama-loaded
    # instance in this deployment (one GPU, OLLAMA_NUM_PARALLEL=1 -- a
    # single serving slot). Every call against that slot -- realtime chat,
    # environmental grounding, OCR, object/identity comparison, visual
    # question answering -- MUST request this identical num_ctx: llama.cpp
    # reloads the entire model (multi-second, gigabytes of tensors) any
    # time a request asks for a different context size than what's
    # currently loaded. Confirmed directly on this device: letting
    # environmental grounding request a larger context than the
    # conversational path caused 48 full model reloads in two hours --
    # audible "churning", 40-90s+ turn latency, and dropped replies --
    # simply from the two call types alternating. Without an explicit
    # value, namespaced Ornith manifests also request their full 262K
    # training window and consume memory needed by ASR/TTS runtimes, so
    # this is deliberately bounded, not left to the model's default.
    # Sized to also cover environmental grounding's heavier prompt (a
    # multi-camera contact-sheet image plus a long text prompt) --
    # smaller values here truncated that JSON mid-object even with an
    # unbounded num_predict, because num_ctx is the real ceiling on
    # prompt+completion combined.
    model_num_ctx: int = Field(default=16384, ge=4096, le=131072)
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


class ResidencyConfig(BaseModel):
    """Memory budget for the heavy model components.

    Unified memory has no separate VRAM to overflow into, so an over-commit
    does not merely kill the offending process -- on this device it froze the
    machine. The manager admits a component only when its measured cost plus
    this reserve genuinely fits, and refuses otherwise.

    The costs below are measured on a 32 GB AGX Orin, not derived from file
    size: on Tegra the GPU allocation is not the weight file.
    """

    enabled: bool = True
    # Headroom the manager will not spend. The OS, page cache, and the
    # companion's own allocations move underneath a load that was sized at
    # admission time.
    reserve_gib: float = Field(default=2.0, ge=0.5, le=16)
    # Qwen3-Omni comprehension at an 8K window. The most expensive to reload,
    # so it outranks everything else and is evicted last.
    # Measured at a 4096 context: 16.7 GiB, against 16.8 at 8192. The KV is
    # not the driver here, the weights are -- halving the window buys almost
    # nothing, so do not expect context to be the lever for fitting this.
    comprehension_cost_gib: float = Field(default=16.7, gt=0, le=64)
    comprehension_unit: str = "egg-omni-comprehension.service"
    # The context the comprehension worker is started with. The manager budgets
    # against this, so the unit's -c must match it. They drifted once -- the
    # manager sized a 4096 window while the unit ran 8192 -- and the worker was
    # OOM-killed by its own cgroup cap mid-utterance, which looked from outside
    # like the assistant simply disconnecting.
    comprehension_context_tokens: int = Field(default=4096, ge=1024, le=131072)
    comprehension_priority: int = 10
    comprehension_load_timeout_seconds: float = Field(default=420, gt=0, le=3600)
    # Release comprehension after this long unused. Holding 16.8 GiB while
    # idle is indistinguishable from a leak to everything else on the module:
    # it squats memory the language model needs and nothing asks it to leave.
    comprehension_idle_release_seconds: float = Field(default=180, ge=0, le=86400)
    # The Qwen3-TTS worker, which is non-persistent and so only holds this
    # while actually speaking.
    # Measured while cloning: 6.4 GiB for the worker, not the 1.4 GiB of
    # weights -- the speaker-embedding encoder is most of it.
    # Measured on a 30 GiB Orin, sampling MemAvailable through a synthesis
    # with the comprehension worker resident: the speech worker peaks at 4.0
    # GiB. Earlier figures of 6.4 and 6.7 were read against a moving baseline
    # and were high by two thirds, which mattered: the manager evicted 16.7
    # GiB of comprehension every turn to reserve room for something that
    # needed far less, and the turn then paid to load it back.
    speech_cost_gib: float = Field(default=4.0, gt=0, le=32)
    # The adapter HTTP/TTS wrapper holds no model weights while idle. Keep its
    # small controller footprint separate from the transient TTS worker cost,
    # or merely starting the endpoint is budgeted as if speech were already
    # generating and needlessly evicts comprehension.
    adapter_service_cost_gib: float = Field(default=0.25, gt=0, le=4)
    speech_unit: str = "egg-omni-adapters.service"
    speech_priority: int = 5
    speech_load_timeout_seconds: float = Field(default=300, gt=0, le=3600)
    # The speech service is small and its TTS worker is already
    # non-persistent, so it stays up by default.
    speech_idle_release_seconds: float = Field(default=0, ge=0, le=86400)
    # Ollama serving the logical tag. Measured by unloading it and watching
    # MemAvailable: 15.6 GiB, not the 5.6 GiB `ollama ps` reports. On Tegra
    # the nvmap allocation is roughly 10 GiB beyond the reported model size,
    # and registering the reported figure makes the manager believe it cannot
    # reclaim enough to admit anything larger.
    #
    # Registered so the budget accounts for Ollama even though Ollama owns its
    # own lifetime -- an unregistered consumer of the same pool is a hole in
    # the guarantee.
    language_cost_gib: float = Field(default=15.6, gt=0, le=64)
    language_priority: int = 8
    # Ollama manages its own keep-alive, so the manager only evicts it under
    # pressure rather than on a timer.
    language_idle_release_seconds: float = Field(default=0, ge=0, le=86400)
    # How often the runtime sweeps for idle components.
    sweep_interval_seconds: float = Field(default=30, gt=0, le=3600)


class OmniAdapterConfig(BaseModel):
    """Qwen Omni adapter (``robit.ollama.omni-adapter.v1``) routing.

    The adapter is a separately supervised process from
    https://github.com/robit-man/qwen-omni-adapters that fronts one logical
    Ollama tag with Qwen3-Omni comprehension and Qwen3-TTS speech. Egg uses it
    for the perceptual stages that benefit from a genuinely multimodal model:
    speech and non-speech sound separated at the model boundary, environmental
    audio described in language rather than a fixed 521-class taxonomy, bounded
    video understanding, and 24 kHz speech with an optional voice reference.

    Every capability below is individually switchable and every one falls back
    to the existing Omnius path when the adapter is absent or unhealthy, so
    enabling this never removes a capability Egg already had.
    """

    # The one switch. `omni` routes perception and speech through the all-in-one
    # package; `traditional` keeps the Omnius ASR service, the YAMNet
    # classifier, and Supertonic TTS. Individual capabilities below can still
    # be pinned, but the mode is what an operator is expected to change.
    mode: Literal["omni", "traditional"] = "omni"
    # The whole point of omni mode: one weights package answers everything, so
    # every other discrete model is silenced rather than left resident. That
    # means no YOLOE/SAM/CLIP/pose/face ONNX stack, no separate Whisper ASR, no
    # Supertonic voice, and no YAMNet classifier -- image and video
    # understanding come from the logical tag, audio and speech from its
    # sidecar. On a 32 GB module this is not only cleaner, it is the only way
    # the comprehension component fits at all.
    #
    # Set false to run the adapter *beside* the existing stack, which needs a
    # host with memory for both.
    exclusive: bool = True
    base_url: HttpUrl = "http://127.0.0.1:8910"
    # The logical Omni tag. Its comprehension and TTS components live in the
    # tag's custom sidecar layer, and its *standard* layers are the Ornith
    # language model itself -- stock Ollama runs the tag directly for text,
    # vision, and tools.
    model: str = "robit/ornith-1.5-omni:q4km"
    # Deliberately the same tag. `robit/ornith-1.5-omni:q4km` and
    # `robit/ornith-1.5:9b` resolve to byte-identical base and projector blobs
    # -- verified with `ollama show --modelfile` on this device -- so the
    # language model is already baked into the logical tag and nothing has to
    # be loaded beside it. Ollama keys a loaded runner by tag *name*, though,
    # so naming a second tag would still load a second resident copy of the
    # same weights; with OLLAMA_MAX_LOADED_MODELS=1 the two evict each other on
    # every alternating call. One tag, one slot. See `share_ollama_slot`.
    language_model: str = "robit/ornith-1.5-omni:q4km"
    # Point Egg's own chat and vision calls at `language_model` too, so the
    # companion, the vision path, and the adapter's language stage all address
    # one Ollama runner. Turning this off is only correct on a host that can
    # afford several models resident at once; on this single-slot device it
    # reintroduces exactly the multi-second reload churn documented on
    # OmniusConfig.model_num_ctx.
    share_ollama_slot: bool = True
    bearer_token_env: str | None = None
    # Comprehension of a bounded speech segment on the integrated Jetson GPU.
    timeout_seconds: float = Field(default=45, gt=0, le=300)
    speech_timeout_seconds: float = Field(default=90, gt=0, le=600)
    # Memory used by the non-persistent Qwen3-TTS worker. Reserved before each
    # utterance in addition to the residency manager's safety reserve, because
    # the lightweight speech service being up is not the same as its worker
    # fitting. The measured 4.0 GiB peak plus a small load-time margin. Set too
    # high, this evicts the worker that answers everything on every turn; set
    # too low, the TTS process is admitted into memory it does not fit and
    # dies with a CUDA allocation failure part-way through speaking.
    speech_headroom_gib: float = Field(default=4.2, ge=1, le=32)
    video_timeout_seconds: float = Field(default=180, gt=0, le=900)
    health_timeout_seconds: float = Field(default=3, gt=0, le=30)
    # The startup audit probes once, while the machine is at its busiest
    # bringing every service up. Holding it to the per-turn budget reports a
    # healthy adapter as degraded for no better reason than contention.
    audit_health_timeout_seconds: float = Field(default=20, gt=0, le=120)
    # A passed health probe is trusted for this long, so ordinary turns do not
    # pay an extra round trip before every perception call.
    health_ttl_seconds: float = Field(default=30, gt=0, le=600)
    # After a failure the adapter is skipped entirely for this long. Without
    # it, a stopped adapter would add its full connection timeout to every
    # spoken turn instead of failing over to Omnius immediately.
    failure_cooldown_seconds: float = Field(default=60, ge=0, le=3600)
    # Egg shares one GPU between vision, ASR, language, and speech. More than
    # one in-flight adapter request only queues work behind a spoken turn.
    max_concurrent_requests: int = Field(default=1, ge=1, le=4)
    keep_alive: str = "30m"

    # Start the adapter service alongside the companion, so `omni` mode is
    # self-contained rather than depending on an operator having started a
    # second unit by hand.
    autostart: bool = True
    autostart_unit: str = "egg-omni-adapters.service"
    autostart_timeout_seconds: float = Field(default=900, ge=0, le=3600)
    # Separate model services that exclusive omni mode makes redundant. The
    # CUDA Whisper container holds its own ASR weights on the same unified
    # memory the comprehension component needs, and omni mode transcribes from
    # the single package instead.
    silence_units: list[str] = Field(
        default_factory=lambda: ["egg-whisper.service"]
    )
    # Stop the voice daemon too once the omni package answers language.
    #
    # The daemon keeps a Whisper worker and a YAMNet classifier resident --
    # about 1.1 GiB of weights the single package replaces -- and exposes no
    # way to release them short of stopping it. It is only safe to stop once
    # replies are generated through the adapter, which is why this is separate
    # from `exclusive` rather than implied by it.
    #
    # The cost is the daemon's own tool execution: `search_current_web` runs
    # there, and is withdrawn from the model's choices while it is stopped
    # rather than offered as a function that cannot run.
    silence_voice_daemon: bool = True
    voice_daemon_unit: str = "omnius-daemon.service"
    # The adapter's portal carries a tool harness lifted from Omnius -- web
    # search, fetch and crawl, document and session search, scratch memory,
    # bounded arithmetic. Egg drives its own tool loop, so it executes these
    # one at a time; running them here is what lets the voice daemon go
    # without taking web search with it.
    portal_base_url: HttpUrl = HttpUrl("http://127.0.0.1:8920")
    # The daemon mints a fresh portal token on every start and writes it here.
    portal_token_path: str = (
        "vendor/qwen-omni-adapters/runtime-data/state/access-token.txt"
    )
    portal_timeout_seconds: float = Field(default=45, gt=0, le=300)
    # The context the language worker actually has per request. Prompts are
    # trimmed to fit it: a llama.cpp server refuses an over-long prompt
    # outright rather than truncating, so a turn that overruns gets no reply.
    # Must match the comprehension worker's per-slot context; the runtime
    # warns when they disagree.
    language_context_tokens: int = Field(default=4096, ge=1024, le=131072)
    # Which process answers language for the adapter.
    #
    # "comprehension" points the adapter's language stage at the comprehension
    # worker, which is already serving an OpenAI-shaped /v1/chat/completions
    # and already holds these weights. That is the whole point of a single
    # weights package: the alternative, "ollama", loads the same model a
    # second time in a second runtime, and 16.7 + 15.6 + 6.4 GiB does not fit
    # in 30, so every turn pays a full reload to swap between hearing and
    # answering.
    #
    # This must match OMNI_LANGUAGE_API/OMNI_LANGUAGE_URL in the adapter unit;
    # the runtime warns when the two disagree, because the mismatch is
    # invisible until the budget is wrong.
    language_stage: Literal["comprehension", "ollama"] = "comprehension"

    @property
    def language_component(self) -> str:
        """The residency component whose weights answer a generation."""

        return (
            "omni_comprehension"
            if self.mode == "omni" and self.language_stage == "comprehension"
            else "ollama_language"
        )

    # Per-capability pins. None follows `mode`; True/False override it, which is
    # how a host runs (say) Qwen3-TTS speech while leaving comprehension on the
    # traditional path. Each capability independently falls back to the Omnius
    # path whenever the adapter cannot answer.
    transcription_enabled: bool | None = None
    audio_scene_enabled: bool | None = None
    video_enabled: bool | None = None
    speech_enabled: bool | None = None

    # Qwen3-TTS voice selection. The adapter checkout ships the same
    # `voice-profile.json` the portal uses, so Egg speaks with the project's
    # own Female/Male presets rather than an unrelated default. `voice_preset`
    # names a preset id from that file; unset uses the one marked default.
    # `voice_reference_path` overrides the preset with any local WAV, which is
    # the request-local speaker embedding Qwen3-TTS clones from.
    voice_profile_path: str = "vendor/qwen-omni-adapters/portal/voice-profile.json"
    voice_preset: str | None = None
    voice: str | None = None
    voice_language: str | None = None
    voice_style: str | None = None
    voice_reference_path: str | None = None
    # Synthesis sampling. None follows the voice profile, which is where the
    # project's tuned values live.
    voice_temperature: float | None = Field(default=None, ge=0, le=2)
    voice_top_k: int | None = Field(default=None, ge=1, le=200)
    voice_top_p: float | None = Field(default=None, gt=0, le=1)
    voice_seed: int | None = None
    voice_max_frames: int | None = Field(default=None, ge=1, le=4096)
    # Stream decoder PCM windows and start playback on the first one, instead
    # of waiting for the whole utterance to synthesize. This is the portal's
    # behaviour and it is the difference between speech starting in a few
    # hundred milliseconds and starting after a full generation.
    stream_speech: bool = True

    # Video sampling bounds for describe_video, and the rolling per-camera clip
    # buffer the `video` tool draws from. Frames are downscaled before
    # buffering: a few seconds of full-resolution frames per camera would cost
    # more memory than the comprehension it feeds.
    video_fps: float = Field(default=2.0, gt=0, le=30)
    video_max_frames: int = Field(default=48, ge=1, le=1024)
    video_include_audio: bool = True
    video_buffer_seconds: float = Field(default=6.0, gt=0, le=30)
    video_buffer_max_width: int = Field(default=640, ge=160, le=1920)

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_enabled_flag(cls, data: object) -> object:
        """Map a pre-`mode` `enabled:` boolean onto the mode switch.

        `enabled` is now derived from `mode`. Pydantic would otherwise ignore
        the stale key and silently turn omni on for a deployment that had
        explicitly turned the adapter off, which is the one migration outcome
        nobody wants.
        """

        if isinstance(data, dict) and "enabled" in data:
            data = dict(data)
            legacy = data.pop("enabled")
            if isinstance(legacy, bool):
                data.setdefault("mode", "omni" if legacy else "traditional")
        return data

    @property
    def enabled(self) -> bool:
        return self.mode == "omni"

    def _capability(self, pinned: bool | None) -> bool:
        return self.enabled if pinned is None else (pinned and self.enabled)

    @property
    def uses_transcription(self) -> bool:
        return self._capability(self.transcription_enabled)

    @property
    def uses_audio_scene(self) -> bool:
        return self._capability(self.audio_scene_enabled)

    @property
    def uses_video(self) -> bool:
        return self._capability(self.video_enabled)

    @property
    def uses_speech(self) -> bool:
        return self._capability(self.speech_enabled)

    @property
    def silences_voice_daemon(self) -> bool:
        """Whether the voice daemon is stopped along with its weights."""

        return self.silences_discrete_voice and self.silence_voice_daemon

    @property
    def silences_discrete_voice(self) -> bool:
        """Whether the discrete speech and ASR services are stopped on purpose.

        When this holds, configuring those services is not merely wasted: the
        calls fail against something deliberately absent, and any that succeed
        bring a replaced backend back into the turn.
        """

        return self.enabled and self.exclusive


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


class EnvironmentalCognitionConfig(BaseModel):
    """Event-driven visual grounding, reflection, and optional social outreach.

    Numeric bounds here govern inference cost and evidence freshness. They do
    not map scene labels, gestures, phrases, or identities to an action; Ornith
    decides whether an admitted perceptual change warrants silence, reflection,
    a statement, or a question.
    """

    enabled: bool = True
    outward_speech_enabled: bool = True
    queue_size: int = Field(default=1, ge=1, le=8)
    minimum_salience: float = Field(default=0.18, ge=0, le=1)
    salience_half_life_seconds: float = Field(default=45.0, gt=0, le=3600)
    habituation_half_life_seconds: float = Field(default=600.0, gt=0, le=86400)
    current_evidence_max_age_seconds: float = Field(default=30.0, gt=0, le=300)
    raw_frame_width: int = Field(default=64, ge=24, le=320)
    raw_novelty_minimum: float = Field(default=0.045, ge=0.001, le=1)
    raw_surprise_sigma: float = Field(default=3.0, ge=0.5, le=10)
    raw_reference_blend: float = Field(default=0.08, gt=0, le=1)
    raw_probe_min_interval_seconds: float = Field(default=4.0, ge=0, le=300)
    reflection_characters: int = Field(default=900, ge=200, le=4000)


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
    omni_adapter: OmniAdapterConfig = Field(default_factory=OmniAdapterConfig)
    residency: ResidencyConfig = Field(default_factory=ResidencyConfig)
    system_service: SystemServiceConfig | None = None
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    activity: ActivityConfig = Field(default_factory=ActivityConfig)
    environmental_cognition: EnvironmentalCognitionConfig = Field(
        default_factory=EnvironmentalCognitionConfig
    )
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

    @model_validator(mode="after")
    def silence_discrete_models_in_omni_mode(self) -> EggConfig:
        """In exclusive omni mode, load nothing but the single weights package.

        Egg's traditional stack keeps six independent model families resident
        -- YOLOE, SAM, CLIP, YOLO pose, ONNX face detection/recognition, and
        the identity-dream embedders -- plus a separate Whisper ASR service and
        the Supertonic voice. Omni mode replaces all of them with one logical
        Ollama tag: its standard layers answer language and image
        understanding, its sidecar answers audio, video, and speech.

        Leaving the discrete models loaded would defeat the purpose and, on a
        32 GB module, leave no room for the 18.5 GiB comprehension component.
        The capabilities that exist only as consumers of discrete-model output
        -- identity galleries, object learning, voxel occupancy, local OCR,
        identity dreams -- are switched off with it rather than left running
        against an empty detector.
        """

        if not (self.omni_adapter.enabled and self.omni_adapter.exclusive):
            return self
        silenced: list[str] = []
        for section, reason in (
            ("vision", "YOLOE/SAM/CLIP/pose/face ONNX"),
            ("identity", "face-gallery identity (needs the discrete face stack)"),
            ("object_learning", "object library (needs discrete detections)"),
            ("occupancy", "voxel occupancy (needs the depth subprocess)"),
            ("dreams", "identity dreams (separate embedding models)"),
            ("ocr", "local OCR models"),
        ):
            current = getattr(self, section)
            if getattr(current, "enabled", False):
                silenced.append(f"{section} ({reason})")
                object.__setattr__(self, section, current.model_copy(update={"enabled": False}))
        if silenced:
            logger.info(
                "omni_adapter.exclusive: the single weights package answers everything; "
                "silenced %s",
                "; ".join(silenced),
            )
        return self

    @model_validator(mode="after")
    def share_one_ollama_slot(self) -> EggConfig:
        """Address one Ollama runner when the Omni adapter is enabled.

        The logical Omni tag carries the Ornith language model in its standard
        layers, so `robit/ornith-1.5-omni:q4km` and `robit/ornith-1.5:9b` are
        the same weights under two names. Ollama keys a loaded runner by name,
        and this device runs OLLAMA_MAX_LOADED_MODELS=1, so leaving Egg's chat
        and vision on one name while the adapter's language stage uses the
        other makes every alternating call a full multi-second model reload --
        the exact failure OmniusConfig.model_num_ctx documents.

        Reconciling them here rather than asking the operator to keep three
        settings in sync keeps that footgun closed by default.
        """

        if not self.omni_adapter.enabled or not self.omni_adapter.share_ollama_slot:
            return self
        shared = self.omni_adapter.language_model
        changed = {
            field: getattr(self.omnius, field)
            for field in ("model", "vision_model")
            if getattr(self.omnius, field) != shared
        }
        if changed:
            logger.info(
                "omni_adapter.share_ollama_slot: repointing %s at %s so Egg and the "
                "adapter's language stage share one Ollama runner",
                ", ".join(f"omnius.{field}={value}" for field, value in changed.items()),
                shared,
            )
            object.__setattr__(
                self,
                "omnius",
                self.omnius.model_copy(
                    update={field: shared for field in changed}
                ),
            )
        return self


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
