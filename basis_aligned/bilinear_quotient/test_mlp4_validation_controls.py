import hashlib
import json

import torch

from . import mlp4_freeze_controls as controls


def test_frozen_control_is_minimal_and_bound():
    manifest = json.loads(controls.MANIFEST.read_text())
    artifact = torch.load(controls.OUTPUT, map_location="cpu", weights_only=False)
    assert set(artifact) == {"schema_version", "control_id",
                            "fit_artifact_sha256", "mean_output"}
    assert controls.sha(controls.OUTPUT) == manifest["artifact_sha256"]
    assert artifact["fit_artifact_sha256"] == manifest["fit_artifact_sha256"]
    mean = artifact["mean_output"]
    assert mean.shape == (1152,) and mean.dtype == torch.float32
    assert hashlib.sha256(mean.numpy().astype("<f4", copy=False).tobytes()
                          ).hexdigest() == manifest["mean_output_tensor_sha256"]
    assert not manifest["behavioral_selection"] and not manifest["validation_opened"]
