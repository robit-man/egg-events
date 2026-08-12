# Omnius audio comprehension handoff for the Egg Jetson

Date: 2026-08-12  
Audience: Omnius implementer  
Goal: make `audio_analyze/classify` a reliable, warm, offline, hardware-compatible service without affecting realtime ASR, conversation, or vision.

## Executive summary

Egg's realtime ASR is healthy because it was separated into a persistent, JetPack-matched CUDA service with pinned models and a stable REST boundary. Omnius 1.0.627 had the opposite shape: every `audio_analyze/classify` call started Python, imported a generic CPU-only TensorFlow wheel, loaded YAMNet from a TF-Hub URL/cache, performed inference, printed JSON through a text wrapper, and exited. Omnius' direct-tool executor also defaulted to 30 seconds unless the caller supplied `timeout_ms`.

The live symptom is:

```text
RuntimeError: Omnius audio_analyze failed:
Tool 'audio_analyze' timed out after 30000ms
```

Egg explicitly sends the executor timeout, which removes the accidental 30-second cutoff. Omnius 1.0.629 now provides the requested persistent, pre-warmed, offline TensorRT runtime. The deployment details below explain the JetPack compatibility settings required to make that runtime actually warm on this device.

## Omnius 1.0.628 verification update

Omnius 1.0.628 adds the requested daemon-owned interfaces:

- `GET /v1/audio/classify/health`
- `POST /v1/audio/classify`
- `POST /v1/audio/classify/setup`

Egg now prefers the structured `/v1/audio/classify` result and retains the old
`audio_analyze` tool call only as a compatibility fallback. The dashboard also
reports `omnius-audio` independently from base daemon, voice, and cognition
health.

The first live check on this device returned HTTP 503 with:

```text
ready=false
backend=tensorrt-fp16
device=jetson-cuda:0
weights_ready=false
warmed=false
last_error=JetPack CUDA Torch preflight failed: CUDA unavailable
           (torch=2.10.0+cpu, torch_cuda=None)
```

This confirms that the REST/lifecycle contract is present, but the managed audio
runtime is still resolving a CPU-only Torch build. Setup must use a JetPack
R36.3-compatible CUDA runtime without replacing Egg's working CUDA Torch or the
isolated Whisper service. The acceptance tests below remain the definition of
done; route availability alone is not readiness.

## Omnius 1.0.629 working deployment

The 1.0.629 runtime is now live and passes the intended contract after three
JetPack-specific corrections:

1. Set `OMNIUS_AUDIO_PYTHON` to Egg's verified JetPack interpreter
   (`.venv/bin/python`: Torch 2.2.0, CUDA 12.2, Orin SM 8.7). Omnius then creates
   its own isolated `~/.omnius/runtimes/audio/venv` with system-site access; it
   does not modify Egg's environment.
2. Put Egg's `scripts/compat-bin` before `/usr/bin` in the Omnius service `PATH`.
   JetPack R36.3's `/usr/bin/tegrastats` does not support the `--count` option
   used by Omnius 1.0.629; the wrapper reads the requested number of genuine
   samples and terminates the real process.
3. Pin `cuda-python==12.2.0` inside the isolated Omnius audio venv. The inherited
   `cuda-python==13.2.0` namespace does not expose `from cuda import cudart`,
   while the TensorRT worker requires that API and the host runtime is CUDA 12.2.

The active systemd drop-in is reproducible from
`config/systemd/omnius-daemon-audio-cuda.conf`; the pre-start repair and
`tegrastats` compatibility implementation live under `scripts/` and survive
future Omnius npm updates.

Observed readiness after setup:

```text
ready=true; backend=tensorrt-fp16; device=Orin; compute_capability=8.7
weights_ready=true; warmed=true; model_load_ms=528.602; errors=0
```

A retained six-second ReSpeaker WAV completed through Egg's structured adapter
in 125 ms and returned five grounded AudioSet classifications. This meets the
persistent-worker and latency goals; the legacy TensorFlow/TF-Hub path is now
only a compatibility fallback for older Omnius releases.

## Exact hardware and working voice stack

| Component | Live value |
|---|---|
| Host | NVIDIA Jetson AGX Orin, `aarch64` |
| Jetson Linux | R36.3, kernel `5.15.136-tegra` |
| JetPack CUDA | CUDA 12.2 |
| Application Python | 3.10.12 |
| CUDA Torch | 2.2.0, JetPack-matched, `torch.cuda.is_available() == True` |
| Microphone | Seeed ReSpeaker USB 4-Mic Array v2.0, USB `2886:0018`, XVF3000 |
| Pulse source | `alsa_input.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.multichannel-input` |
| Physical stream | Six channels; channel 0 is processed AEC/beamformed audio |
| Egg audio sent to Omnius | RIFF/WAV, mono PCM, 16 kHz, normally six seconds or shorter |
| Conditioning | 160 Hz speech-band high-pass, voiced-RMS AGC, target RMS `0.16`, bounded gain `48x` |
| ASR | Persistent CUDA dual Whisper service on `127.0.0.1:11436` |
| Omnius | 1.0.629 on `127.0.0.1:11435` |
| Ollama/Ornith | `127.0.0.1:11434`; shared unified memory, so audio must not depend on an LLM |

The working ASR deployment details are in [the Egg README](../README.md#jetsonarm64-voice-stack-what-made-realtime-asr-work). Preserve its CUDA Torch installation. Do not let audio-comprehension dependency resolution replace the JetPack-matched Torch packages.

## What Egg already guarantees

Egg owns physical capture and turn admission. For `action=classify`, Omnius must analyze the supplied file and must not open `default`, invoke `arecord`, or start another ReSpeaker capture process.

Before calling Omnius, Egg has already:

1. Selected XVF3000 channel 0 from the six-channel USB stream.
2. Applied speech-band filtering and bounded gain.
3. Run WebRTC VAD and rejected ungrounded/silent ASR windows.
4. Written a valid mono, 16 kHz WAV.
5. Retained the WAV under cognitive-memory storage and passed a temporary local path to Omnius.
6. Put comprehension behind ASR in a queue of size one; newer audio coalesces rather than delaying speech recognition.

Do not normalize the input a second time merely to make it resemble microphone capture. Classification may calculate and return acoustic measurements, but labels must be derived from the supplied samples.

## Current Omnius implementation and measured failure

The installed implementation is in the compiled `AudioAnalyzeTool.classifyAudio` path. It currently does all of the following per request:

```text
ensureVenv([tensorflow, tensorflow-hub, numpy, soundfile, resampy, ...])
spawn Python
import TensorFlow + tensorflow_hub
hub.load("https://tfhub.dev/google/yamnet/1")
load/read/resample WAV
run YAMNet
print JSON
exit Python
```

Live facts from this device:

- `~/.omnius/audio-ml-venv` is 6.2 GB.
- It contains TensorFlow 2.21.0, tensorflow-hub 0.16.1, NumPy 2.2.6, SoundFile 0.14.0, and resampy 0.4.3.
- That TensorFlow build reports `tf.test.is_built_with_cuda() == False` and exposes no GPU.
- The YAMNet TF-Hub artifact is only cached under `/tmp/tfhub_modules` (18 MB), an ephemeral location.
- A retained six-second Egg WAV classifies correctly as `Speech 0.60`, `Vehicle 0.15`, but the process frequently crosses the direct executor's former 30-second limit under concurrent vision/cognition load.
- The direct executor times out by racing the tool promise. This does not prove that the Python subprocess was terminated. Timed-out inference can therefore continue consuming CPU/RAM and overlap a later request.
- The tool returns human-formatted text containing embedded JSON. Egg must search for the first `{` and parse the remainder.

This is a lifecycle/deployment problem, not an input-level or ReSpeaker gain problem.

## Required Omnius architecture

### 1. Persistent isolated runtime

Run audio classification in one long-lived worker or sidecar. Load the model once during startup and warm it with a short zero/noise tensor. Keep this process isolated from:

- the JetPack CUDA dual-Whisper ASR service;
- Ollama cognition and Ornith vision model residency;
- Omnius' managed ASR virtual environment;
- Egg's application virtual environment.

A practical layout is:

```text
~/.omnius/runtimes/audio/
  venv/
  models/yamnet/
  class_map.csv
  worker.py
```

The worker can be supervised by Omnius or systemd, but Omnius health must report its real readiness.

### 2. Offline pinned model

Download and checksum YAMNet during installation/bootstrap, not during inference. Store it under `~/.omnius/runtimes/audio/models`, never `/tmp`, and set an explicit cache/model path. Runtime requests must perform zero external network fetches.

Pin the class map with the model. Return the model digest and taxonomy version in readiness and inference results.

### 3. Jetson-compatible backend

The minimum-risk implementation is a persistent CPU worker using a pinned TFLite or ONNX YAMNet export. YAMNet is small enough that a warm six-second inference should not require a GPU, and keeping it off CUDA avoids unified-memory contention with Whisper, Ornith, and cognition.

If a GPU backend is chosen, build it specifically for JetPack R36.3/CUDA 12.2 (for example, TensorRT from a validated ONNX export). Do not install a generic PyPI CUDA wheel and do not modify the system/application Torch installation. Backend choice must be visible in `/health` as `cpu-tflite`, `cpu-onnx`, or `tensorrt`, not inferred from package version strings.

### 4. Bounded concurrency and cancellation

- Serialize classifier execution per loaded model.
- Permit at most one waiting classification; coalesce or reject older contextual jobs with a typed `busy` response.
- A timed-out REST request must cancel/kill its underlying work.
- Never leave detached TensorFlow/Python processes after ToolExecutor timeout.
- Model initialization must be single-flight: concurrent first requests await the same warmup task.

### 5. Structured REST result

Keep the existing direct-tool contract for compatibility:

```http
POST /v1/tools/audio_analyze/call
Authorization: Bearer …
Content-Type: application/json
```

```json
{
  "args": {
    "action": "classify",
    "file": "/tmp/egg-audio-comprehension-….wav",
    "top_k": 5
  },
  "timeout_ms": 90000
}
```

Add a structured object to `result.data` (or a documented equivalent). Do not require callers to parse a prose prefix:

```json
{
  "tool": "audio_analyze",
  "result": {
    "success": true,
    "output": "Speech 0.607, Vehicle 0.155",
    "data": {
      "schema_version": 1,
      "action": "classify",
      "model": "yamnet",
      "model_digest": "sha256:…",
      "backend": "cpu-tflite",
      "taxonomy": "AudioSet-521",
      "sample_rate_hz": 16000,
      "duration_seconds": 6.0,
      "classifications": [
        {"label": "Speech", "confidence": 0.607},
        {"label": "Vehicle", "confidence": 0.155}
      ],
      "timings_ms": {
        "queue": 0.4,
        "decode": 2.1,
        "inference": 34.8,
        "total": 38.0
      }
    },
    "durationMs": 38.0
  }
}
```

Recommended follow-up: add `POST /v1/audio/classify` accepting WAV bytes or multipart upload. Local filesystem paths work while Egg and Omnius share a host/user, but byte upload is a cleaner service boundary and removes path lifetime/security ambiguity.

### 6. Readiness and warmup contract

Expose audio readiness independently of general Omnius health:

```json
{
  "ready": true,
  "model": "yamnet",
  "backend": "cpu-tflite",
  "device": "aarch64",
  "weights_ready": true,
  "warmed": true,
  "model_load_ms": 412.0,
  "last_inference_ms": 38.0,
  "queue_depth": 0,
  "errors": 0
}
```

General `/health` returning 200 is not evidence that audio classification is ready. A cold, missing, downloading, mock, or scaffold backend must say so explicitly.

## Grounding and semantic policy

`classify` and `comprehend` must remain distinct:

- `classify` is the deterministic AudioSet classifier. It returns only model scores and measured acoustic facts.
- `comprehend` may fuse specialized audio roles, but it must mark whether each role is live, mock, or unavailable.
- Never emit `mock semantic scaffold` as observed evidence.
- Do not ask an LLM to invent audio labels when the classifier fails.
- Empty/quiet input should return a valid low-information result, not a plausible speech/event hallucination.
- Preserve raw scores. Thresholding into memories remains Egg policy.

## Acceptance tests on this Egg

The Omnius change is complete only when all of these pass:

1. **Offline boot:** disconnect network after installation; audio readiness becomes true and classification succeeds.
2. **Warm latency:** after one warmup, classify a six-second, mono 16 kHz WAV in under 500 ms p95 for 100 sequential calls. If the selected backend cannot meet this, document the measured bound and keep it below two seconds.
3. **Cold latency:** service start through first ready state is bounded and reported; no request performs dependency installation or model download.
4. **Process stability:** 100 calls do not increase Python worker count or leave child processes after timeout/cancellation.
5. **Concurrency:** submit three jobs simultaneously; at most one runs and one waits, with deterministic busy/coalesced behavior.
6. **Hardware fixture:** a retained Egg podcast window produces meaningful `Speech` probability. A silent fixture does not produce a confident speech/event label.
7. **No ASR regression:** `curl http://127.0.0.1:11436/health` remains healthy and dual-Whisper latency does not materially change during classification.
8. **No cognition starvation:** Omnius chat and Ornith can make progress while audio classification is active.
9. **Structured schema:** callers consume `result.data.classifications` without parsing human text.
10. **Health recovery:** force the worker down, observe audio readiness false, restart it, and observe readiness auto-return true without restarting Omnius or Egg.

## Egg integration locations

- Audio capture/conditioning: `egg_companion/adapters/audio.py`
- Omnius tool invocation/parser: `egg_companion/adapters/omnius.py`
- Non-blocking comprehension queue: `egg_companion/runtime.py`
- Live telemetry: `egg_companion/services/telemetry.py`
- Current working values: `config/egg.yaml`
- Jetson bootstrap and dependency rules: `scripts/bootstrap-jetson.sh`
- CUDA dual-Whisper service: `scripts/jetson_whisper_server.py` and `deploy/egg-whisper.service`

## Definition of done

Audio comprehension is not done when the Python packages merely import or when a one-off request eventually succeeds. It is done when a preloaded local model repeatedly classifies Egg's already-conditioned WAVs with bounded latency, explicit backend/readiness metadata, no network dependency, no leaked work after timeouts, and no measurable degradation of ASR or conversational inference.
