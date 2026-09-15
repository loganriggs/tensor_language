#!/usr/bin/env python3
"""Freeze L11H3's native top output singular vector as a subject write axis."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OPS = ROOT / "basis_aligned/bilinear_quotient/ops"
sys.path.insert(0, str(OPS))
import fastload
import torch

RANK = HERE / "SUBJECT_NUMBER_DIRECTION_CARDINALITY_RANK1_V1_ARTIFACT.json"
OUT = HERE / "SUBJECT_NUMBER_NATIVE_WEIGHT_AXIS_V1_ARTIFACT.json"
WEIGHTS_SHA256 = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def main():
    config, checkpoint, _ = fastload._paths()
    if sha(checkpoint) != WEIGHTS_SHA256: raise ValueError("checkpoint changed")
    rank = json.loads(RANK.read_text())
    if rank["terminal"] != "rank1_frozen_weights_only": raise ValueError("rank artifact changed")
    model = fastload.load_model_fast()
    matrix = model.transformer.h[11].attn.c_proj.weight.detach().double()[:, 3 * 128:4 * 128]
    u, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
    prior = torch.tensor(rank["axis"], dtype=torch.float64); prior /= prior.norm()
    axis = u[:, 0]
    if float(axis @ prior) < 0: axis = -axis
    overlap = float(axis @ prior)
    payload = {
        "schema": "subject_number_native_weight_axis_v1_artifact", "layer": 11, "head": 3,
        "construction": "top_left_singular_vector_of_native_head_output_projection",
        "axis": axis.float().tolist(), "axis_norm": float(axis.norm()),
        "prior_axis_cosine": overlap, "prior_axis_squared_overlap": overlap ** 2,
        "top_singular_value": float(singular[0]), "second_singular_value": float(singular[1]),
        "rank_artifact_sha256": sha(RANK), "checkpoint_weights_sha256": WEIGHTS_SHA256,
        "causal_outcomes_read": False, "behavior_fit": False, "axis_combination_fit": False,
        "rank_sweep": False, "parameter_updates": 0, "terminal": "native_weight_axis_frozen",
    }
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: payload[k] for k in ("terminal", "prior_axis_cosine", "prior_axis_squared_overlap", "top_singular_value", "second_singular_value")}, indent=2))


if __name__ == "__main__": main()
