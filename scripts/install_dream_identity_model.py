from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


REPOSITORY = "minchul/cvlface_adaface_ir18_webface4m"
REVISION = "0dd53f188fa27968b0a1326970ebf4aeb37ce2ca"
COMPARISON_REPOSITORY = "deepghs/insightface"
COMPARISON_FILE = "buffalo_s/w600k_mbf.onnx"
COMPARISON_SHA256 = "9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f"


def main() -> None:
    import hashlib
    import shutil

    root = Path(__file__).resolve().parents[1]
    destination = root / "models" / "cvlface_adaface_ir18_webface4m"
    resolved = snapshot_download(
        REPOSITORY,
        revision=REVISION,
        local_dir=destination,
        allow_patterns=[
            "README.md",
            "config.json",
            "wrapper.py",
            "model.safetensors",
            "models/**",
        ],
    )
    print(f"Pinned offline identity model installed at {resolved}")
    comparison_source = Path(
        hf_hub_download(COMPARISON_REPOSITORY, filename=COMPARISON_FILE)
    )
    digest = hashlib.sha256(comparison_source.read_bytes()).hexdigest()
    if digest != COMPARISON_SHA256:
        raise RuntimeError(
            f"unexpected MobileFaceNet checksum {digest}; expected {COMPARISON_SHA256}"
        )
    comparison_destination = (
        Path.home() / ".insightface" / "models" / "buffalo_s" / "w600k_mbf.onnx"
    )
    comparison_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(comparison_source, comparison_destination)
    print(f"Pinned offline comparison model installed at {comparison_destination}")


if __name__ == "__main__":
    main()
