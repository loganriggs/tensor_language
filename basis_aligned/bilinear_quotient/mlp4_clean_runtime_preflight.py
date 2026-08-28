#!/usr/bin/env python3
"""Synthetic-only dynamic identity check before MLP4 validation may open rows."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

from . import bilin18_clean_runtime as runtime
from . import mlp4_z4_validation as validation
from . import bilin18_reference_forward as reference

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "mlp4_clean_runtime_preflight_protocol.json"
OUTPUT = HERE / "mlp4_clean_runtime_preflight_result.json"


def tensor_hash(value):
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy()
                          .astype("<f4", copy=False).tobytes()).hexdigest()


def telemetry():
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    rows = [[int(part.strip()) for part in line.split(",")]
            for line in query.stdout.splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("no GPU telemetry row")
    return {"maximum_temperature_c": max(row[0] for row in rows),
            "maximum_device_memory_used_mib": max(row[1] for row in rows)}


@torch.no_grad()
def main():
    started = time.time()
    protocol = json.loads(PROTOCOL.read_text())
    tokens = torch.arange(protocol["synthetic_input"]["token_count"],
                          device=runtime.DEV, dtype=torch.long).reshape(1, -1)
    candidate_logits = validation.forward_inline(tokens).float()
    reference_logits = reference.reference_forward(runtime.m, tokens).float()
    difference = candidate_logits-reference_logits
    resource = telemetry()
    resource["peak_torch_allocated_gib"] = \
        torch.cuda.max_memory_allocated(runtime.DEV)/2**30
    parameter_count = sum(parameter.numel() for parameter in runtime.m.parameters())
    result = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "synthetic_only": True,
        "natural_rows_opened": False,
        "checkpoint_blob_sha256": runtime.sha(runtime.CHECKPOINT_PATH),
        "config_sha256": runtime.sha(runtime.CONFIG_PATH),
        "parameter_count": parameter_count,
        "input_token_ids": tokens.cpu().flatten().tolist(),
        "candidate_logits_sha256": tensor_hash(candidate_logits),
        "reference_logits_sha256": tensor_hash(reference_logits),
        "maximum_absolute_logit_error": float(difference.abs().max()),
        "root_mean_square_logit_error": float(difference.square().mean().sqrt()),
        "resources": {**resource, "runtime_s": time.time()-started},
    }
    result["gates"] = {
        "checkpoint": result["checkpoint_blob_sha256"] ==
            protocol["checkpoint"]["blob_sha256"],
        "config": result["config_sha256"] == protocol["checkpoint"]["config_sha256"],
        "parameter_count": parameter_count == protocol["checkpoint"]["parameter_count"],
        "forward_max_error": result["maximum_absolute_logit_error"] <=
            protocol["gates"]["maximum_absolute_logit_error"],
        "forward_rms_error": result["root_mean_square_logit_error"] <=
            protocol["gates"]["root_mean_square_logit_error"],
        "memory": resource["peak_torch_allocated_gib"] <=
            protocol["resources"]["hard_abort_peak_gib"],
        "temperature": resource["maximum_temperature_c"] <=
            protocol["resources"]["hard_abort_temperature_c"],
    }
    result["passes"] = all(result["gates"].values())
    OUTPUT.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)
    if not result["passes"]:
        raise RuntimeError("clean runtime preflight failed")


if __name__ == "__main__":
    main()
