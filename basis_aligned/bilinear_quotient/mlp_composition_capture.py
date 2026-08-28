#!/usr/bin/env python3
"""Failure-closed validator for paired MLP composition-state captures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CONTRACT = HERE / "mlp_composition_capture_contract.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_contract(path=CONTRACT):
    result = json.loads(Path(path).read_text())
    if result.get("schema_version") != 1 or not result.get("contract_id"):
        raise ValueError("unsupported or unidentified capture contract")
    if result.get("gpu_authorized") is not False \
            or result.get("validation_data_authorized") is not False:
        raise ValueError("template must not authorize data or GPU execution")
    return result


def _require_manifest(contract, manifest):
    missing = set(contract["required_manifest_fields"])-set(manifest)
    if missing:
        raise ValueError(f"manifest missing fields: {sorted(missing)}")
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported manifest schema")
    layer = manifest.get("layer")
    expected_object = f"blocks.{layer}.mlp"
    expected_interface = f"{expected_object}.rmsnorm_input"
    if manifest["object_id"] != expected_object \
            or manifest["interface"] != expected_interface:
        raise ValueError("object, layer, and typed interface disagree")
    if not isinstance(manifest["state_width"], int) or manifest["state_width"] <= 0:
        raise ValueError("state_width must be positive")
    for field in ("checkpoint_sha256", "rows_artifact_sha256",
                  "upstream_program_sha256", "capture_source_sha256",
                  "artifact_sha256"):
        value = manifest[field]
        if not isinstance(value, str) or len(value) != 64 \
                or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{field} is not a lowercase SHA256")


def load_and_validate(manifest_path, artifact_path, contract_path=CONTRACT):
    contract = load_contract(contract_path)
    manifest = json.loads(Path(manifest_path).read_text())
    _require_manifest(contract, manifest)
    if sha256(artifact_path) != manifest["artifact_sha256"]:
        raise ValueError("capture artifact hash mismatch")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    metadata = ("schema_version", "contract_id", "object_id", "interface",
                "checkpoint_sha256", "rows_artifact_sha256",
                "upstream_program_sha256")
    if not isinstance(artifact, dict) or any(key not in artifact for key in metadata):
        raise ValueError("capture artifact metadata incomplete")
    if artifact["schema_version"] != 1 \
            or artifact["contract_id"] != contract["contract_id"]:
        raise ValueError("capture artifact contract mismatch")
    for key in metadata[2:]:
        if artifact[key] != manifest[key]:
            raise ValueError(f"capture metadata mismatch: {key}")
    ids = artifact.get("observation_ids")
    live = artifact.get("z_live")
    composed = artifact.get("z_composed")
    width = manifest["state_width"]
    if not torch.is_tensor(ids) or ids.dtype != torch.int64 or ids.ndim != 2 \
            or ids.shape[1] != 2 or ids.shape[0] == 0:
        raise ValueError("observation_ids must be nonempty int64 [N,2]")
    if not torch.is_tensor(live) or not torch.is_tensor(composed) \
            or live.dtype != torch.float32 or composed.dtype != torch.float32 \
            or live.shape != composed.shape or live.shape != (ids.shape[0], width):
        raise ValueError("paired states must be aligned float32 [N,state_width]")
    if not torch.isfinite(live).all() or not torch.isfinite(composed).all():
        raise ValueError("paired states contain nonfinite values")
    if bool((ids < 0).any()):
        raise ValueError("observation IDs must be nonnegative")
    keys = ids[:, 0]*(int(ids[:, 1].max())+1)+ids[:, 1]
    if ids.shape[0] > 1 and not bool((keys[1:] > keys[:-1]).all()):
        raise ValueError("observation IDs must be unique and lexicographically increasing")
    return manifest, artifact
