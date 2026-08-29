#!/usr/bin/env python3
"""Repair Omnius' generated CUDA ASR probe for Python 3.10.

Some Omnius releases embed an f-string containing an escaped quote in generated
Python source.  Python 3.10 rejects that source before the ASR worker can start.
This targeted, idempotent rewrite retains the same diagnostic while using %-style
formatting, which is valid on every supported Python version.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import tempfile


BAD = (
    r"f'CUDA-only ASR is enabled but torch cannot use CUDA "
    r"(torch.version.cuda={getattr(torch.version, \"cuda\", None)}, "
    r"device_count={torch.cuda.device_count()})'"
)
GOOD = (
    r"'CUDA-only ASR is enabled but torch cannot use CUDA "
    r"(torch.version.cuda=%s, device_count=%s)' % "
    r'''(getattr(torch.version, "cuda", None), torch.cuda.device_count())'''
)
ESCAPED_GOOD = GOOD.replace('"cuda"', r'\"cuda\"')
REPLACEMENTS = (
    (BAD, GOOD),
    (BAD.replace(r'\"', r'\\"'), GOOD),
    (ESCAPED_GOOD, GOOD),
    (ESCAPED_GOOD.replace(r'\"', r'\\"'), GOOD),
)
BAD_SCRIPT_LOOKUP = (
    'const candidates = [\n'
    '    join134(MODULE_DIR, "..", "scripts", name10),'
)
GOOD_SCRIPT_LOOKUP = (
    'const candidates = [\n'
    '    join134(MODULE_DIR, "scripts", name10),\n'
    '    join134(MODULE_DIR, "..", "scripts", name10),'
)
HTTP_TRANSCRIBE_START = (
    '        const result = selection.engineId === "vibevoice-transformers" ? await (async () => {'
)
HTTP_TRANSCRIBE_END = "        try {\n          fs14.unlinkSync(tmpPath);"
HTTP_TRANSCRIBE_FAST_V1 = (
    "        const transcribed = await listen.transcribeFile(tmpPath, void 0, {\n"
    "          context: context2,\n"
    "          setup: true\n"
    "        });\n"
    "        if (!transcribed) throw new Error(`Selected ASR runtime returned no result for ${selection.engineId}/${selection.modelId}`);\n"
    "        const result = {\n"
    "          text: transcribed.text,\n"
    "          duration: transcribed.duration,\n"
    "          segments: transcribed.segments ?? [],\n"
    "          speakers: transcribed.speakers ?? [],\n"
    "          engineId: transcribed.engineId ?? selection.engineId,\n"
    "          modelId: transcribed.modelId ?? selection.modelId,\n"
    "          rawText: transcribed.rawText,\n"
    "          warnings: transcribed.warnings ?? []\n"
    "        };\n"
)
HTTP_TRANSCRIBE_GOOD = (
    "        const transcribed = await listen.transcribeFile(tmpPath, void 0, {\n"
    "          context: context2,\n"
    '          language: urlObj.searchParams.get("language") || "auto",\n'
    "          setup: true\n"
    "        });\n"
    "        if (!transcribed) throw new Error(`Selected ASR runtime returned no result for ${selection.engineId}/${selection.modelId}`);\n"
    "        const result = {\n"
    "          text: transcribed.text,\n"
    "          duration: transcribed.duration,\n"
    "          language: transcribed.language,\n"
    "          segments: transcribed.segments ?? [],\n"
    "          speakers: transcribed.speakers ?? [],\n"
    "          engineId: transcribed.engineId ?? selection.engineId,\n"
    "          modelId: transcribed.modelId ?? selection.modelId,\n"
    "          rawText: transcribed.rawText,\n"
    "          warnings: transcribed.warnings ?? []\n"
    "        };\n"
)
TRANSCRIBE_CLI_OPTIONS_BAD = (
    '              model: this.config.model,\n'
    '              format: "json",\n'
    "              diarize: false,\n"
    "              wordTimestamps: false\n"
)
TRANSCRIBE_CLI_OPTIONS_GOOD = (
    '              model: this.config.model,\n'
    '              format: "json",\n'
    "              diarize: false,\n"
    '              language: options2.language ?? "auto",\n'
    "              wordTimestamps: false\n"
)
TRANSCRIBE_CLI_RESULT_BAD = (
    "              text: result.text,\n"
    "              duration: result.duration,\n"
    "              speakers: result.speakers,"
)
TRANSCRIBE_CLI_RESULT_GOOD = (
    "              text: result.text,\n"
    "              duration: result.duration,\n"
    "              language: result.language,\n"
    "              speakers: result.speakers,"
)
HTTP_LANGUAGE_RESPONSE_BAD = "          language: null,\n          engineId: result.engineId,"
HTTP_LANGUAGE_RESPONSE_GOOD = (
    "          language: result.language ?? null,\n          engineId: result.engineId,"
)
DIRECT_CHAT_MODEL_BAD = '  const cleanModel = model.replace(/^[a-z]+\\//, "");'
DIRECT_CHAT_MODEL_GOOD = (
    '  const cleanModel = isVllm ? model.replace(/^[a-z]+\\//, "") : model;'
)
CHAT_NUM_CTX_PASS_BAD = (
    '              ...chatBody["max_tokens"] !== void 0 ? '
    '{ max_tokens: chatBody["max_tokens"] } : {},\n'
    '              ...chatBody["response_format"] !== void 0'
)
CHAT_NUM_CTX_PASS_GOOD = (
    '              ...chatBody["max_tokens"] !== void 0 ? '
    '{ max_tokens: chatBody["max_tokens"] } : {},\n'
    '              ...chatBody["num_ctx"] !== void 0 ? '
    '{ num_ctx: chatBody["num_ctx"] } : {},\n'
    '              ...chatBody["response_format"] !== void 0'
)
OLLAMA_NUM_CTX_BAD = (
    '    if (typeof ef["max_tokens"] === "number")\n'
    '      ollamaOpts["num_predict"] = ef["max_tokens"];\n'
    '    if (typeof ef["seed"] === "number")'
)
OLLAMA_NUM_CTX_GOOD = (
    '    if (typeof ef["max_tokens"] === "number")\n'
    '      ollamaOpts["num_predict"] = ef["max_tokens"];\n'
    '    if (typeof ef["num_ctx"] === "number")\n'
    '      ollamaOpts["num_ctx"] = ef["num_ctx"];\n'
    '    if (typeof ef["seed"] === "number")'
)
CHAT_KEEP_ALIVE_PASS_BAD = (
    '              ...chatBody["num_ctx"] !== void 0 ? '
    '{ num_ctx: chatBody["num_ctx"] } : {},\n'
    '              ...chatBody["response_format"] !== void 0'
)
CHAT_KEEP_ALIVE_PASS_GOOD = (
    '              ...chatBody["num_ctx"] !== void 0 ? '
    '{ num_ctx: chatBody["num_ctx"] } : {},\n'
    '              ...chatBody["keep_alive"] !== void 0 ? '
    '{ keep_alive: chatBody["keep_alive"] } : {},\n'
    '              ...chatBody["response_format"] !== void 0'
)
OLLAMA_KEEP_ALIVE_BAD = (
    '      think: false,\n'
    '      ...hasTools ? { tools: ef["tools"] } : {},'
)
OLLAMA_KEEP_ALIVE_GOOD = (
    '      think: false,\n'
    '      ...ef["keep_alive"] !== void 0 ? { keep_alive: ef["keep_alive"] } : {},\n'
    '      ...hasTools ? { tools: ef["tools"] } : {},'
)
NEMO_DEPENDENCY_BAD = (
    "    except ImportError:\n"
    '        emit_status("Installing nemo_toolkit[asr] (large — may take a few minutes)...")'
)
NEMO_DEPENDENCY_UNSAFE = (
    "    except Exception as e:\n"
    '        emit_status(f"NeMo import unavailable ({e}) — validating install before transformers fallback")'
)
NEMO_DEPENDENCY_GOOD = (
    "    except Exception as e:\n"
    '        emit_status(f"NeMo import unavailable ({e}) — using transformers fallback")\n'
    "        return"
)
NEMO_LOADER_BAD = (
    "    except ImportError:\n"
    "        return (None, None)\n"
    "    try:\n"
    "        device = select_asr_device()"
)
NEMO_LOADER_GOOD = (
    "    except Exception as e:\n"
    '        emit_status(f"NeMo import failed ({e}) — using transformers fallback")\n'
    "        return (None, None)\n"
    "    try:\n"
    "        device = select_asr_device()"
)
NEMO_MODEL_BAD = "    args = parser.parse_args()\n\n    if args.check:"
NEMO_MODEL_GOOD = (
    "    args = parser.parse_args()\n"
    "    if args.model and \"/\" not in args.model:\n"
    "        args.model = f\"nvidia/{args.model}\"\n\n"
    "    if args.check:"
)


def repair_nemotron(path: Path, source: str) -> str:
    repaired = source
    occurrences = 0
    for broken, compatible in (
        (NEMO_DEPENDENCY_BAD, NEMO_DEPENDENCY_GOOD),
        (NEMO_LOADER_BAD, NEMO_LOADER_GOOD),
        (NEMO_MODEL_BAD, NEMO_MODEL_GOOD),
    ):
        count = repaired.count(broken)
        occurrences += count
        repaired = repaired.replace(broken, compatible)
    # An earlier repair retained the upstream generic pip install. Remove the
    # complete install block: on Jetson it can replace NVIDIA's CUDA 12.2 torch
    # with an incompatible PyPI CUDA build merely because optional NeMo failed.
    unsafe_install = (
        f"{NEMO_DEPENDENCY_UNSAFE}\n"
        "        try:\n"
        "            subprocess.check_call(\n"
        '                [sys.executable, "-m", "pip", "install", "nemo_toolkit[asr]"],\n'
        "                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,\n"
        "                timeout=600,\n"
        "            )\n"
        "        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:\n"
        '            emit_status(f"NeMo install skipped ({e}) — will use transformers fallback")'
    )
    unsafe_count = repaired.count(unsafe_install)
    occurrences += unsafe_count
    repaired = repaired.replace(unsafe_install, NEMO_DEPENDENCY_GOOD)
    if occurrences == 0:
        if all(
            compatible in source
            for compatible in (NEMO_DEPENDENCY_GOOD, NEMO_LOADER_GOOD, NEMO_MODEL_GOOD)
        ):
            return "already-compatible"
        raise RuntimeError(f"expected Omnius Nemotron runtime was not found in {path}")
    mode = path.stat().st_mode
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as output:
        output.write(repaired)
        temporary = Path(output.name)
    os.chmod(temporary, mode)
    os.replace(temporary, path)
    return f"repaired-{occurrences}"


def repair(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    if "def _load_nemo_model" in source:
        return repair_nemotron(path, source)
    repaired = source
    occurrences = 0
    for broken, compatible in REPLACEMENTS:
        count = repaired.count(broken)
        occurrences += count
        repaired = repaired.replace(broken, compatible)
    # In the bundled single-file build MODULE_DIR is already `dist`, so the
    # generated managed-ASR helper must inspect `dist/scripts` before trying
    # source-tree and process-working-directory fallbacks. Without this the
    # shipped live-nemotron.py exists but resolves as ~/live-nemotron.py.
    script_lookup_count = repaired.count(BAD_SCRIPT_LOOKUP)
    occurrences += script_lookup_count
    repaired = repaired.replace(BAD_SCRIPT_LOOKUP, GOOD_SCRIPT_LOOKUP)
    # esbuild's numeric suffixes change between Omnius releases (join134,
    # join135, name10, ...). Repair the generated lookup structurally so an
    # innocuous bundle renumbering cannot prevent the daemon from starting.
    script_pattern = re.compile(
        r'(const candidates = \[\n)(\s+)(join\d+)\(MODULE_DIR, "\.\.", "scripts", (name\d+)\),'
    )

    def add_dist_script_candidate(match: re.Match[str]) -> str:
        return (
            f"{match.group(1)}{match.group(2)}{match.group(3)}"
            f'(MODULE_DIR, "scripts", {match.group(4)}),\n'
            f"{match.group(2)}{match.group(3)}"
            f'(MODULE_DIR, "..", "scripts", {match.group(4)}),'
        )

    repaired, structural_script_count = script_pattern.subn(
        add_dist_script_candidate, repaired, count=1
    )
    occurrences += structural_script_count
    route_start = repaired.find(HTTP_TRANSCRIBE_START)
    if route_start >= 0:
        route_end = repaired.find(HTTP_TRANSCRIBE_END, route_start)
        if route_end < 0:
            raise RuntimeError(f"Omnius HTTP transcription route terminator was not found in {path}")
        repaired = repaired[:route_start] + HTTP_TRANSCRIBE_GOOD + repaired[route_end:]
        occurrences += 1
    for broken, compatible in (
        (HTTP_TRANSCRIBE_FAST_V1, HTTP_TRANSCRIBE_GOOD),
        (TRANSCRIBE_CLI_OPTIONS_BAD, TRANSCRIBE_CLI_OPTIONS_GOOD),
        (TRANSCRIBE_CLI_RESULT_BAD, TRANSCRIBE_CLI_RESULT_GOOD),
        (HTTP_LANGUAGE_RESPONSE_BAD, HTTP_LANGUAGE_RESPONSE_GOOD),
        (DIRECT_CHAT_MODEL_BAD, DIRECT_CHAT_MODEL_GOOD),
        (CHAT_NUM_CTX_PASS_BAD, CHAT_NUM_CTX_PASS_GOOD),
        (OLLAMA_NUM_CTX_BAD, OLLAMA_NUM_CTX_GOOD),
        (CHAT_KEEP_ALIVE_PASS_BAD, CHAT_KEEP_ALIVE_PASS_GOOD),
        (OLLAMA_KEEP_ALIVE_BAD, OLLAMA_KEEP_ALIVE_GOOD),
    ):
        count = repaired.count(broken)
        occurrences += count
        repaired = repaired.replace(broken, compatible)
    if occurrences == 0:
        cuda_compatible = (
            "CUDA-only ASR is enabled" not in source
            or "torch.version.cuda=%s" in source
        )
        script_compatible = (
            "function locateScript" not in source
            or re.search(
                r'join\d+\(MODULE_DIR, "scripts", name\d+\)', source
            )
            is not None
        )
        language_compatible = (
            "/v1/voice/transcribe" not in source
            or (
                'language: options2.language ?? "auto"' in source
                and "language: result.language" in source
            )
        )
        chat_compatible = (
            "async function directChatBackend" not in source
            or (
                DIRECT_CHAT_MODEL_GOOD in source
                and '{ num_ctx: chatBody["num_ctx"] }' in source
                and 'ollamaOpts["num_ctx"] = ef["num_ctx"]' in source
                and '{ keep_alive: chatBody["keep_alive"] }' in source
                and '{ keep_alive: ef["keep_alive"] }' in source
            )
        )
        if cuda_compatible and script_compatible and language_compatible and chat_compatible:
            return "already-compatible"
        raise RuntimeError(f"expected Omnius ASR probe was not found in {path}")
    mode = path.stat().st_mode
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as output:
        output.write(repaired)
        temporary = Path(output.name)
    os.chmod(temporary, mode)
    os.replace(temporary, path)
    return f"repaired-{occurrences}"


def default_targets() -> list[Path]:
    targets = list(
        Path.home().glob(".nvm/versions/node/*/lib/node_modules/omnius/dist/index.js")
    )
    if not targets:
        return []

    def node_version(path: Path) -> tuple[int, ...]:
        version = next(
            (part for part in path.parts if re.fullmatch(r"v\d+(?:\.\d+)+", part)),
            "v0",
        )
        return tuple(int(item) for item in version[1:].split("."))

    # NVM leaves older global installations behind. Only the newest Node tree
    # can own the active launcher; patching stale bundles can fail a valid
    # daemon start after Omnius changes its generated code shape.
    return [max(targets, key=node_version)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    targets = args.paths or default_targets()
    if not targets:
        parser.error("no Omnius installation found")
    expanded_targets: list[Path] = []
    for target in targets:
        expanded_targets.append(target)
        bundled_nemotron = target.parent / "scripts" / "live-nemotron.py"
        if target.name == "index.js" and bundled_nemotron.is_file():
            expanded_targets.append(bundled_nemotron)
    for target in dict.fromkeys(expanded_targets):
        print(f"{target}: {repair(target)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
