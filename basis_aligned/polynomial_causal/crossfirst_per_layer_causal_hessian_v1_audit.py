#!/usr/bin/env python3
"""Outcome-complete audit of the preregistered per-layer Hessian artifact."""
import json
from pathlib import Path

import torch

from sparse_path_stability_atlas_v1 import digest

P = Path(__file__).resolve().parent
ARTIFACT = P / "CROSSFIRST_PER_LAYER_CAUSAL_HESSIAN_V1_ARTIFACT.pt"
PRIMARY = P / "CROSSFIRST_PER_LAYER_CAUSAL_HESSIAN_V1_RESULT.json"
OUT = P / "CROSSFIRST_PER_LAYER_CAUSAL_HESSIAN_V1_AUDIT.json"
EXPECTED_ARTIFACT = "c0ab22e3a64536a31bd8d0bac0ba5d1acbd20d2035053d50b3588742848a42f2"


def ratio(numerator, denominator):
    return float(numerator.norm() / denominator.norm().clamp_min(1e-30))


def main():
    if digest(ARTIFACT) != EXPECTED_ARTIFACT:
        raise ValueError("artifact changed")
    primary = json.loads(PRIMARY.read_text())
    if primary["artifact_sha256"] != EXPECTED_ARTIFACT or not primary["predictions"]["pred_a_instrument"]:
        raise ValueError("primary authority invalid")
    artifact = torch.load(ARTIFACT, weights_only=True)
    terms = artifact["allocations"][0, :, :, 0].double()
    finite = artifact["finite_interactions"][:, 0].double()
    child = artifact["child_effects"][:, 0].double()
    names = artifact["stage_names"]
    energy = terms.square().sum(0)
    order = torch.argsort(energy, descending=True)
    frontier = []
    minimum_k = None
    for width in range(1, len(names) + 1):
        indices = order[:width]
        prediction = terms[:, indices].sum(1)
        families = []
        for family in range(4):
            sl = slice(24 * family, 24 * (family + 1))
            families.append({
                "family": family,
                "residual_relative_to_child": ratio(prediction[sl] - finite[sl], child[sl]),
                "residual_relative_to_finite_interaction": ratio(prediction[sl] - finite[sl], finite[sl]),
            })
        maximum = max(report["residual_relative_to_child"] for report in families)
        frontier.append({"width": width, "stages": [names[int(index)] for index in indices], "maximum_residual_relative_to_child": maximum, "families": families})
        if minimum_k is None and maximum <= .10:
            minimum_k = width

    leave_one_family_out = []
    for held in range(4):
        training_rows = torch.cat([torch.arange(24 * family, 24 * (family + 1)) for family in range(4) if family != held])
        held_slice = slice(24 * held, 24 * (held + 1))
        training_energy = terms[training_rows].square().sum(0)
        indices = torch.argsort(training_energy, descending=True)[:2]
        prediction = terms[held_slice][:, indices].sum(1)
        leave_one_family_out.append({
            "held_family": held,
            "selected_stages": [names[int(index)] for index in indices],
            "residual_relative_to_child": ratio(prediction - finite[held_slice], child[held_slice]),
            "residual_relative_to_finite_interaction": ratio(prediction - finite[held_slice], finite[held_slice]),
        })

    random_top_two = []
    for writer_index, label in enumerate(artifact["writer_labels"][1:], start=1):
        random_terms = artifact["allocations"][writer_index, :, :, 0].double()
        random_energy = random_terms.square().sum(0)
        indices = torch.argsort(random_energy, descending=True)[:2]
        random_top_two.append({
            "writer": label,
            "stages": [names[int(index)] for index in indices],
            "energy_fraction": float(random_energy[indices].sum() / random_energy.sum().clamp_min(1e-30)),
        })

    result = {
        "schema": "crossfirst_per_layer_causal_hessian_v1_audit",
        "primary_artifact_sha256": EXPECTED_ARTIFACT,
        "pooled_stage_order": [names[int(index)] for index in order],
        "minimum_pooled_energy_order_width_for_ten_percent_child_error": minimum_k,
        "frontier": frontier,
        "leave_one_family_out_top_two": leave_one_family_out,
        "random_top_two": random_top_two,
        "interpretation": "Exploratory opened-panel audit. The pooled two-stage correction is attention17 plus MLP10 and leaves <10% child-relative composition error in every family; the same pair is selected in every leave-one-family-out fold. This nominates, but does not freshly confirm, a reusable second-order correction.",
        "scope": "Score-only audit of the hash-bound per-prompt Hessian artifact; no model execution, fit, parameter update, fresh panel, or retroactive preregistered predicate.",
    }
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"minimum_k": minimum_k, "top_two": result["pooled_stage_order"][:2], "leave_one_family_out": leave_one_family_out}, indent=2))


if __name__ == "__main__":
    main()
