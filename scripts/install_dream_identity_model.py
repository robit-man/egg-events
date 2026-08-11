from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


REPOSITORY = "minchul/cvlface_adaface_ir18_webface4m"
REVISION = "0dd53f188fa27968b0a1326970ebf4aeb37ce2ca"


def main() -> None:
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


if __name__ == "__main__":
    main()
