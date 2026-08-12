# Omnius defect report: nemotron-streaming activation refused despite ready catalog state

Filed from the `egg_companion` client after live-probing the Omnius daemon running on this device at `http://127.0.0.1:11435`. Originally reproduced on v1.0.616 and reproduced again after upgrading the live daemon to v1.0.628 on 2026-08-12.

## 1. `nemotron-streaming` cannot be activated, and the catalog gives no way to detect this in advance

`GET /v1/asr/engines` reports the engine as fully ready:

```json
{
  "id": "nemotron-streaming",
  "selected": false,
  "readiness": { "installed": true, "weightsReady": true, "active": false }
}
```

But activating it fails unconditionally:

```
POST /v1/asr/activate
{"engineId":"nemotron-streaming","modelId":"nemotron-speech-streaming-en-0.6b"}

-> 500
{"error":"asr_activation_failed","message":"Nemotron is catalogued but its legacy self-installing worker is disabled until it is migrated to the managed fail-closed runtime.","requested":{"engineId":"nemotron-streaming","modelId":"nemotron-speech-streaming-en-0.6b"}}
```

The compatibility alias `POST /v1/voice/asr-models/switch` gives an even less useful error when only `modelId` is supplied (no `engineId`), silently assuming the *currently active* engine instead of the model's own engine:

```
POST /v1/voice/asr-models/switch {"modelId":"nemotron-speech-streaming-en-0.6b"}
-> 400 {"error":"invalid_asr_selection","message":"Unknown ASR model nemotron-speech-streaming-en-0.6b for engine transcribe-cli"}
```

**Ask:** add a `readiness.disabled` (or similar) flag — distinct from `installed`/`weightsReady` — so catalog consumers can detect a deliberately-gated engine programmatically instead of parsing a free-text `message` string. Right now `installed: true, weightsReady: true` actively signals "safe to activate" when it isn't.

## 2. `/v1/ocr/advanced` returns a contradictory error on a successful call

```
POST /v1/ocr/advanced {"imagePath":"<valid, existing path>"}
-> 200
{
  "success": true,
  "imagePath": "<same path echoed back>",
  "ocrText": "",
  "ocrError": "image path is required",
  "visionDescription": "",
  "visionUsed": false,
  "visionModel": null,
  "contextBlock": "[Image at <path> — OCR found no text; treat as UNCOMPREHENDED]"
}
```

`imagePath` was present and correct (it's echoed back verbatim), yet `ocrError` claims it was missing. This looks like the OCR subprocess/tool is being invoked with the path under a different argument name than the one the outer handler validated, so the internal call fails with its own generic "missing" error instead of a real "no text found" result. `success: true` alongside a populated `ocrError` is also inconsistent — should probably be `success: false` (or `ocrError: null` when the true outcome is "ran fine, found no text").

**Ask:** fix the internal argument binding so `ocrError` reflects the actual OCR failure mode (or is `null` on a genuine "no text found" result), and make `success` reflect whether OCR itself succeeded.

The v1.0.628 retest used both a full camera frame and a perspective-rectified crop of a detected monitor. The full-frame call took 122.84 seconds and returned the same contradictory response. Supplying all three compatibility keys (`imagePath`, `path`, and `file`) still timed out after 35 seconds, so this was not repairable by changing the Egg request field.

### Omnius 1.0.629 status

The argument-binding defect is resolved by a new canonical contract:

```json
{"args":{"image":"/absolute/image.png","language":"eng","regions":true},"timeout_ms":120000}
```

Egg now consumes structured `result.data`. The endpoint is not operational on
this JetPack R36.3 host yet because its bootstrap performs `sudo -n apt-get` and
the service has no passwordless sudo. More importantly, Ubuntu Jammy arm64 has a
`python3-opencv` package but no `python3-pytesseract` candidate, so granting sudo
would not make the current all-APT command succeed. Omnius should install only
native/system packages through APT and install the pure-Python `pytesseract`
package in its isolated `~/.omnius/runtimes/vision/ocr-advanced` venv, or expose
an explicit pre-provisioned OCR Python override. Egg retains bounded local OCR
while this managed runtime is unavailable.

## Environment

- `omnius@1.0.629` (current retest; originally filed against 1.0.616)
- Daemon started via `~/.config/systemd/user/omnius-daemon.service`
- Jetson AGX Orin, JetPack R36.3
