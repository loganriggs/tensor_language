#!/usr/bin/env python3
"""Complete augmentation cube around the three-MLP response core on v12."""

# BQGATE: EXPERIMENT pred_a_authority_replay_finiteness_and_exact_price pred_b_full_ten_site_pool_remains_faithful pred_c_a_six_site_or_smaller_program_exists pred_d_selector_is_deterministic_and_eligible pred_e_complete_zero_fit_augmentation_lattice
import hashlib
import json
import os
from pathlib import Path

import run_temporal_iswas_three_mlp_response_program_fresh_v12_v1 as experiment

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v12_omitted_response_augmentation_cube_v1.json"
FRESH_RESULT = ROOT / "circuits/followups/temporal_iswas_three_mlp_response_program_fresh_v12_v1_result.json"
FRESH_RUNNER = ROOT / "ops/run_temporal_iswas_three_mlp_response_program_fresh_v12_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v12_omitted_response_augmentation_cube_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas_v12_omitted_response_augmentation_cube_v1"
CORE = ("MLP13", "MLP15", "MLP16")
OPTIONAL = ("L13H6", "L15H1", "L15H5", "L17H2", "MLP12", "MLP14", "MLP17")
FRESH_EXPECTED = {
    "result": "ff831c87408078d1408463ff00a3452c7d8a5b2c7780c355ccc2d6c23376044c",
    "runner": "a7f3907731cb35e588ace7ba5316edb2fd418bdef92790a04cad1c89f21761c7",
}
MAX_FORWARDS, MAX_EVALUATIONS = 146, 7884


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cube_sites(mask):
    return CORE + tuple(site for bit, site in enumerate(OPTIONAL) if mask & (1 << bit))


def eligible(metrics):
    return (
        metrics["behavior_direction_fraction"] >= 0.90
        and metrics["worst_six_cell_residual"] <= 0.20
        and all(metrics["tasks"][task]["behavior"]["signed_projection"] >= 0.80
                and metrics["tasks"][task]["mode1"]["signed_projection"] >= 0.60
                and metrics["tasks"][task]["mode2"]["signed_projection"] >= 0.60
                for task in ("temporal", "iswas"))
    )


def configure():
    if {"result": sha(FRESH_RESULT), "runner": sha(FRESH_RUNNER)} != FRESH_EXPECTED:
        raise RuntimeError("fresh transfer-null authority changed")
    fresh = json.loads(FRESH_RESULT.read_text())
    if (fresh.get("terminal") != "response_program_transfer_null"
            or fresh.get("predictions", {}).get("pred_b_selected_three_mlp_program_transfers") is not False
            or fresh.get("predictions", {}).get("pred_c_full_ten_site_pool_is_faithful") is not True):
        raise RuntimeError("fresh transfer-null decision changed")
    aliases = {
        "direct": (),
        "selected": CORE,
        "full_ten_site_pool": CORE + OPTIONAL,
        "minus_MLP13": ("MLP15", "MLP16"),
        "minus_MLP15": ("MLP13", "MLP16"),
        "minus_MLP16": ("MLP13", "MLP15"),
    }
    cube = {f"mask_{mask:03d}": cube_sites(mask) for mask in range(128)}
    experiment.PRIOR = PRIOR
    experiment.OUT = OUT
    experiment.CANDIDATE_ID = CANDIDATE_ID
    experiment.ARMS = {**aliases, **cube}
    experiment.MAX_FORWARDS = MAX_FORWARDS
    experiment.MAX_EVALUATIONS = MAX_EVALUATIONS
    experiment.EXPECTED = {
        **experiment.EXPECTED,
        "prior": "80927a25bbdbbe868d03bf9cb5bb87c7bc24753171e9f366d8d168638f9be2c4",
    }
    original_write = experiment.atomic_create_json

    def write_result(_path, result):
        cube_metrics = {str(mask): result["arm_metrics"][f"mask_{mask:03d}"]
                        for mask in range(128)}
        eligible_items = [(mask, metrics) for mask, metrics in cube_metrics.items()
                          if eligible(metrics)]
        if eligible_items:
            selected_mask, selected_metrics = min(
                eligible_items,
                key=lambda item: (
                    int(item[0]).bit_count(),
                    item[1]["worst_six_cell_residual"],
                    int(item[0]),
                ),
            )
            selected_mask = int(selected_mask)
            selected_optional = [site for bit, site in enumerate(OPTIONAL)
                                 if selected_mask & (1 << bit)]
            selected_sites = list(cube_sites(selected_mask))
        else:
            selected_mask = None
            selected_metrics = None
            selected_optional = []
            selected_sites = []
        full = cube_metrics["127"]
        inherited_a = result["predictions"][
            "pred_a_authority_capability_replay_self_clamp_finiteness_price"]
        pred_a = bool(inherited_a and result["price"]["model_forwards"] == MAX_FORWARDS
                      and result["price"]["example_evaluations"] == MAX_EVALUATIONS)
        pred_b = bool(full["behavior_direction_fraction"] >= 0.95
                      and full["worst_six_cell_residual"] <= 0.02)
        pred_c = bool(selected_metrics is not None and len(selected_optional) <= 3)
        pred_d = bool(
            selected_metrics is not None
            and eligible(selected_metrics)
            and selected_metrics["sites"] == selected_sites
            and selected_mask == min(
                (int(mask) for mask, metrics in cube_metrics.items() if eligible(metrics)),
                key=lambda mask: (
                    mask.bit_count(),
                    cube_metrics[str(mask)]["worst_six_cell_residual"],
                    mask,
                ),
            )
        )
        pred_e = bool(
            set(cube_metrics) == {str(mask) for mask in range(128)}
            and result["price"]["fit_updates"] == 0
            and result["price"]["model_updates"] == 0
            and result["price"]["transformer_backwards"] == 0
        )
        changed_predictions = {
            "pred_a_authority_replay_finiteness_and_exact_price": pred_a,
            "pred_b_full_ten_site_pool_remains_faithful": pred_b,
            "pred_c_a_six_site_or_smaller_program_exists": pred_c,
            "pred_d_selector_is_deterministic_and_eligible": pred_d,
            "pred_e_complete_zero_fit_augmentation_lattice": pred_e,
        }
        result["schema"] = "temporal_iswas_v12_omitted_response_augmentation_cube_result_v1"
        result["candidate_id"] = CANDIDATE_ID
        result["dryrun"]["candidate_id"] = CANDIDATE_ID
        result["discovery_scope"] = "outcome_adaptive_v12_requires_new_text_disjoint_confirmation"
        result["core"] = list(CORE)
        result["optional_sites"] = list(OPTIONAL)
        result["cube_arm_metrics"] = cube_metrics
        result["eligible_cube_arm_count"] = len(eligible_items)
        result["selected_mask"] = selected_mask
        result["selected_optional_sites"] = selected_optional
        result["selected_sites"] = selected_sites
        result["selected_metrics"] = selected_metrics
        result["predictions"] = changed_predictions
        result["terminal"] = (
            "invalid" if not pred_a
            else "compact_augmented_response_discovery" if all(changed_predictions.values())
            else "distributed_response_program"
        )
        original_write(OUT, result)
        print(json.dumps({key: result[key] for key in (
            "candidate_id", "eligible_cube_arm_count", "selected_mask",
            "selected_optional_sites", "selected_metrics", "predictions", "terminal", "price")},
            sort_keys=True))

    experiment.atomic_create_json = write_result


def main():
    configure()
    experiment.main()


if __name__ == "__main__":
    main()
