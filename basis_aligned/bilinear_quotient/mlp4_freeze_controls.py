#!/usr/bin/env python3
"""Freeze the only fit-derived validation control without exposing fit at eval."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
FIT = HERE / "mlp4_z4_fit_artifact.pt"
FIT_MANIFEST = HERE / "mlp4_z4_fit_artifact_manifest.json"
OUTPUT = HERE / "mlp4_z4_validation_controls.pt"
MANIFEST = HERE / "mlp4_z4_validation_controls_manifest.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    fit_manifest = json.loads(FIT_MANIFEST.read_text())
    assert sha(FIT) == fit_manifest["artifact_sha256"]
    fit = torch.load(FIT, map_location="cpu", weights_only=False)
    mean = fit["linear"]["ymean"].float().contiguous()
    assert mean.shape == (1152,) and torch.isfinite(mean).all()
    torch.save({"schema_version": 1, "control_id": "fit-mean-mlp4-output",
                "fit_artifact_sha256": sha(FIT), "mean_output": mean}, OUTPUT)
    manifest = {
        "schema_version": 1,
        "artifact": OUTPUT.name,
        "artifact_sha256": sha(OUTPUT),
        "fit_artifact_sha256": sha(FIT),
        "mean_output_tensor_sha256": hashlib.sha256(
            mean.numpy().astype("<f4", copy=False).tobytes()).hexdigest(),
        "shape": list(mean.shape),
        "behavioral_selection": False,
        "validation_opened": False,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"wrote {OUTPUT.name} {manifest['artifact_sha256']}")


if __name__ == "__main__":
    main()
