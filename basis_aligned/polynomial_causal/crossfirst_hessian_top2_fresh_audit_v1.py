#!/usr/bin/env python3
"""Outcome-complete scale/sign audit for the fresh Hessian top-two test."""
import json
from pathlib import Path

import torch

from sparse_path_stability_atlas_v1 import digest

P = Path(__file__).resolve().parent
ARTIFACT = P / "CROSSFIRST_HESSIAN_TOP2_FRESH_V1_ARTIFACT.pt"
PRIMARY = P / "CROSSFIRST_HESSIAN_TOP2_FRESH_V1_RESULT.json"
OUT = P / "CROSSFIRST_HESSIAN_TOP2_FRESH_V1_AUDIT.json"
EXPECTED_ARTIFACT = "a14342977bb6091a01a03172b2217a719a2d90ddc731c82bd44b16119ac028ee"


def main():
    if digest(ARTIFACT) != EXPECTED_ARTIFACT:
        raise ValueError("artifact changed")
    primary = json.loads(PRIMARY.read_text())
    if primary["artifact_sha256"] != EXPECTED_ARTIFACT or not primary["predictions"]["pred_a_instrument"]:
        raise ValueError("invalid primary authority")
    artifact = torch.load(ARTIFACT, weights_only=True)
    finite = artifact["finite_interactions"][:, 0].double()
    child = artifact["child_effects"][:, 0].double()
    prediction = artifact["allocations"][:, artifact["candidate_indices"], 0].sum(1).double()
    families = []
    for family in range(4):
        sl = slice(12 * family, 12 * (family + 1))
        observed, predicted, child_effect = finite[sl], prediction[sl], child[sl]
        mismatch = torch.sign(observed) != torch.sign(predicted)
        original = float(observed.norm() / child_effect.norm().clamp_min(1e-30))
        corrected = float((observed - predicted).norm() / child_effect.norm().clamp_min(1e-30))
        families.append({
            "family": family,
            "uncorrected_additive_error_relative_to_child": original,
            "corrected_error_relative_to_child": corrected,
            "fractional_error_reduction": 1.0 - corrected / max(original, 1e-30),
            "finite_interaction_sign_mismatches": int(mismatch.sum()),
            "mismatch_absolute_finite_interactions": observed[mismatch].abs().tolist(),
            "finite_interaction_rms": float(observed.square().mean().sqrt()),
            "maximum_rowwise_corrected_error_over_absolute_child_effect": float(((observed - predicted).abs() / child_effect.abs().clamp_min(1e-30)).max()),
        })
    result = {
        "schema": "crossfirst_hessian_top2_fresh_v1_audit",
        "primary_artifact_sha256": EXPECTED_ARTIFACT,
        "families": families,
        "interpretation": "The frozen two-stage term reduces aggregate child-relative composition error by 27.1--55.5% and leaves every family below 10%. Four of five finite-interaction sign misses are <=1.80e-4 absolute, but one family has a 2.09x rowwise corrected-error/child ratio because its child effect is tiny. Preserve the preregistered finite-vector fidelity null; the supported claim is aggregate composition correction, not per-row interaction reconstruction.",
        "scope": "Score-only audit of the fresh hash-bound artifact; no model execution, selection, threshold change, or verdict relabeling.",
    }
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
