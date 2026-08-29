from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_repair_is_complete_and_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "index.js"
    target.write_text(
        "const probe = `else: raise RuntimeError("
        "f'CUDA-only ASR is enabled but torch cannot use CUDA "
        "(torch.version.cuda={getattr(torch.version, \\\"cuda\\\", None)}, "
        "device_count={torch.cuda.device_count()})')`;\n"
        "function locateScript(name10) {\n"
        "  const candidates = [\n"
        "    join134(MODULE_DIR, \"..\", \"scripts\", name10),\n"
        "  ];\n"
        "}\n",
        encoding="utf-8",
    )
    script = Path(__file__).parents[1] / "scripts" / "repair_omnius_asr_runtime.py"

    first = subprocess.run(
        [sys.executable, str(script), str(target)], check=True, capture_output=True, text=True
    )
    second = subprocess.run(
        [sys.executable, str(script), str(target)], check=True, capture_output=True, text=True
    )

    source = target.read_text(encoding="utf-8")
    assert "repaired-2" in first.stdout
    assert "already-compatible" in second.stdout
    assert "RuntimeError(f'CUDA-only" not in source
    assert "device_count=%s)' %" in source
    assert 'getattr(torch.version, "cuda", None)' in source
    assert r'\"cuda\"' not in source
    assert 'join134(MODULE_DIR, "scripts", name10)' in source


def test_repair_makes_bundled_nemotron_fail_over_and_normalize_model_id(tmp_path: Path) -> None:
    target = tmp_path / "live-nemotron.py"
    target.write_text(
        "def _ensure_deps():\n"
        "    try:\n"
        "        import nemo.collections.asr  # noqa: F401\n"
        "    except ImportError:\n"
        '        emit_status("Installing nemo_toolkit[asr] (large — may take a few minutes)...")\n'
        "def _load_nemo_model():\n"
        "    try:\n"
        "        import nemo.collections.asr as nemo_asr\n"
        "    except ImportError:\n"
        "        return (None, None)\n"
        "    try:\n"
        "        device = select_asr_device()\n"
        "def main():\n"
        "    args = parser.parse_args()\n\n"
        "    if args.check:\n"
        "        return 0\n",
        encoding="utf-8",
    )
    script = Path(__file__).parents[1] / "scripts" / "repair_omnius_asr_runtime.py"

    first = subprocess.run(
        [sys.executable, str(script), str(target)], check=True, capture_output=True, text=True
    )
    second = subprocess.run(
        [sys.executable, str(script), str(target)], check=True, capture_output=True, text=True
    )

    source = target.read_text(encoding="utf-8")
    assert "repaired-3" in first.stdout
    assert "already-compatible" in second.stdout
    assert "except Exception as e:" in source
    assert 'args.model = f"nvidia/{args.model}"' in source


def test_repair_accepts_renumbered_esbuild_script_symbols(tmp_path: Path) -> None:
    target = tmp_path / "index.js"
    target.write_text(
        '''function locateScript(name10) {
  const candidates = [
    join135(MODULE_DIR, "..", "scripts", name10),
    join135(process.cwd(), "scripts", name10)
  ];
}
const cuda = "CUDA-only ASR is enabled torch.version.cuda=%s";
const options = { language: options2.language ?? "auto" };
const result = { language: result.language };
''',
        encoding="utf-8",
    )
    script = Path(__file__).parents[1] / "scripts" / "repair_omnius_asr_runtime.py"

    first = subprocess.run(
        [sys.executable, str(script), str(target)], check=True, capture_output=True, text=True
    )
    second = subprocess.run(
        [sys.executable, str(script), str(target)], check=True, capture_output=True, text=True
    )

    source = target.read_text(encoding="utf-8")
    assert "repaired-1" in first.stdout
    assert "already-compatible" in second.stdout
    assert 'join135(MODULE_DIR, "scripts", name10)' in source


def test_repair_bounds_direct_ollama_chat_and_preserves_namespaced_models(
    tmp_path: Path,
) -> None:
    target = tmp_path / "index.js"
    target.write_text(
        '''async function directChatBackend(opts) {
  const isVllm = false;
  const cleanModel = model.replace(/^[a-z]+\\//, "");
  const ef = extraFields || {};
  const ollamaOpts = {};
    if (typeof ef["max_tokens"] === "number")
      ollamaOpts["num_predict"] = ef["max_tokens"];
    if (typeof ef["seed"] === "number") ollamaOpts["seed"] = ef["seed"];
  const reqBody = JSON.stringify({
      think: false,
      ...hasTools ? { tools: ef["tools"] } : {},
  });
}
const extraFields = {
              ...chatBody["max_tokens"] !== void 0 ? { max_tokens: chatBody["max_tokens"] } : {},
              ...chatBody["response_format"] !== void 0 ? { response_format: chatBody["response_format"] } : {},
};
''',
        encoding="utf-8",
    )
    script = Path(__file__).parents[1] / "scripts" / "repair_omnius_asr_runtime.py"

    first = subprocess.run(
        [sys.executable, str(script), str(target)],
        check=True,
        capture_output=True,
        text=True,
    )
    second = subprocess.run(
        [sys.executable, str(script), str(target)],
        check=True,
        capture_output=True,
        text=True,
    )

    source = target.read_text(encoding="utf-8")
    assert "repaired-5" in first.stdout
    assert "already-compatible" in second.stdout
    assert 'const cleanModel = isVllm ? model.replace' in source
    assert '{ num_ctx: chatBody["num_ctx"] }' in source
    assert 'ollamaOpts["num_ctx"] = ef["num_ctx"]' in source
    assert '{ keep_alive: chatBody["keep_alive"] }' in source
    assert '{ keep_alive: ef["keep_alive"] }' in source
