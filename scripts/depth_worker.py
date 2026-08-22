#!/usr/bin/env python3
"""Single-shot monocular metric depth inference for the occupancy pipeline.

Runs in a separate, pre-existing Python environment (the Depth Anything 3
project's own venv), not egg_companion's -- this repo does not vendor the
depth_anything_3 package. Invoked as a subprocess exactly once per occupancy
update: load the model, run inference on one image, write results to disk,
exit. The whole point of the subprocess boundary is that process exit
reclaims all GPU/CPU memory unconditionally, since this Jetson does not have
enough headroom to keep a ~4GB model resident alongside the rest of Egg's
perception stack.

Usage:
    python3 depth_worker.py <image_path> <output_dir> [--model NAME] [--process-res N]

Writes to <output_dir>:
    depth.npy       float32 HxW depth map (meters, best-effort metric scale)
    conf.npy        float32 HxW confidence map, if the model provides one
    metadata.json   {"depth_shape": [h, w], "has_conf": bool, "is_metric": ...,
                      "model": str, "error": str|null}

Exit code 0 on success (metadata.json always written); non-zero on failure,
with "error" set in metadata.json when possible so the caller gets a reason
without having to parse stdout/stderr, which the model itself writes
unstructured log lines to.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("output_dir")
    parser.add_argument("--model", default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--process-res", type=int, default=504)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"

    try:
        import numpy as np
        import torch
        from PIL import Image
        from depth_anything_3.api import DepthAnything3

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = DepthAnything3.from_pretrained(args.model).to(device)

        image = Image.open(args.image_path).convert("RGB")
        with torch.no_grad():
            prediction = model.inference(image=[image], process_res=args.process_res)

        depth = np.asarray(prediction.depth[0], dtype=np.float32)
        np.save(output_dir / "depth.npy", depth)

        conf = getattr(prediction, "conf", None)
        has_conf = conf is not None and len(conf) > 0
        if has_conf:
            np.save(output_dir / "conf.npy", np.asarray(conf[0], dtype=np.float32))

        metadata = {
            "error": None,
            "model": args.model,
            "depth_shape": list(depth.shape),
            "depth_min": float(depth.min()),
            "depth_max": float(depth.max()),
            "depth_mean": float(depth.mean()),
            "has_conf": has_conf,
            "is_metric": bool(getattr(prediction, "is_metric", None)),
            "intrinsics": (
                np.asarray(prediction.intrinsics[0]).tolist()
                if getattr(prediction, "intrinsics", None) is not None
                else None
            ),
        }
        metadata_path.write_text(json.dumps(metadata))
        return 0
    except Exception as error:  # noqa: BLE001 -- must always report, never crash silently
        metadata_path.write_text(json.dumps({
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }))
        return 1


if __name__ == "__main__":
    sys.exit(main())
