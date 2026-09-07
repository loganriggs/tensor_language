#!/usr/bin/env python3
"""Fresh-temporal confirmation of the four/five-MLP response programs."""

# BQGATE: EXPERIMENT pred_a_authority_replay_self_clamp_finiteness_and_price pred_b_four_site_mlp17_repair_transfers pred_c_five_site_program_meets_strict_cells pred_d_all_five_sites_have_mode2_support pred_e_mlp12_is_a_reproducible_fidelity_term
import hashlib
import json
import math
from pathlib import Path

import circuit_candidate_temporal_auxiliary_fresh_cues_v13 as temporal
import run_temporal_iswas_three_mlp_response_program_fresh_v12_v1 as experiment

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_v13_iswas_v12_four_five_mlp_program_v1.json"
AUGMENTATION = ROOT / "circuits/followups/temporal_iswas_v12_omitted_response_augmentation_cube_v1_result.json"
AUGMENTATION_RUNNER = ROOT / "ops/run_temporal_iswas_v12_omitted_response_augmentation_cube_v1.py"
PARENT_RUNNER = ROOT / "ops/run_temporal_iswas_three_mlp_response_program_fresh_v12_v1.py"
TEMPORAL_BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v13.py"
TEMPORAL_CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v13_capability_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_v13_iswas_v12_four_five_mlp_program_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_v13_iswas_v12_four_five_mlp_program_v1"
CORE = ("MLP13", "MLP15", "MLP16")
FOUR = CORE + ("MLP17",)
FIVE = ("MLP12",) + CORE + ("MLP17",)
POOL = ("L13H6", "L15H1", "L15H5", "L17H2", "MLP12", "MLP13", "MLP14", "MLP15", "MLP16", "MLP17")
MAX_FORWARDS, MAX_EVALUATIONS = 22, 1188
EXPECTED_EXTERNAL = {
    "augmentation": "144e1c001a9c186daa7e2fa426b702a390f938b6fb2f2cf27880812d366301ed",
    "augmentation_runner": "6c7c277e0bcb286a7c60e13fc636bec1388382f77fe9abc902ac7606632a0c61",
    "parent_runner": "a7f3907731cb35e588ace7ba5316edb2fd418bdef92790a04cad1c89f21761c7",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_residual(metrics, task):
    return sum(metrics["tasks"][task][family]["squared_residual"]
               for family in ("behavior", "mode1", "mode2")) / 3.0


def configure():
    external = {
        "augmentation": sha(AUGMENTATION),
        "augmentation_runner": sha(AUGMENTATION_RUNNER),
        "parent_runner": sha(PARENT_RUNNER),
    }
    if external != EXPECTED_EXTERNAL:
        raise RuntimeError("four/five-MLP discovery authority changed")
    augmentation = json.loads(AUGMENTATION.read_text())
    if (augmentation.get("terminal") != "compact_augmented_response_discovery"
            or augmentation.get("selected_mask") != 80
            or set(augmentation.get("selected_sites", ())) != set(FIVE)):
        raise RuntimeError("five-MLP discovery decision changed")
    arms = {
        "direct": (),
        "core_three": CORE,
        "four_site": FOUR,
        "selected": FIVE,
        "full_ten_site_pool": POOL,
        **{f"minus_{site}": tuple(value for value in FIVE if value != site) for site in FIVE},
    }
    experiment.temporal = temporal
    experiment.PRIOR = PRIOR
    experiment.OUT = OUT
    experiment.CANDIDATE_ID = CANDIDATE_ID
    experiment.SELECTED = FIVE
    experiment.POOL = POOL
    experiment.ARMS = arms
    experiment.MAX_FORWARDS = MAX_FORWARDS
    experiment.MAX_EVALUATIONS = MAX_EVALUATIONS
    experiment.TEMPORAL_BUILDER = TEMPORAL_BUILDER
    experiment.TEMPORAL_CAPABILITY = TEMPORAL_CAPABILITY
    experiment.EXPECTED = {
        **experiment.EXPECTED,
        "prior": "73475d5435325fff4a8c3ae334b8ac864eacdefe4a887936467ac51f40ce969d",
        "temporal_builder": "3f738bf2fb2d4a5425dba85eaf948d7ed888e4b891666ff77a51c8638acd2509",
        "temporal_capability": "e053d3381680ce5a933356a060448466d7567e3079c4a2b9a5bff262bd98b9c1",
    }
    original_write = experiment.atomic_create_json

    def write_result(_path, result):
        four = result["arm_metrics"]["four_site"]
        five = result["arm_metrics"]["selected"]
        decrements = result["mode2_signed_projection_decrements"]
        instrument = result["instrument"]
        pred_a = bool(
            all(instrument[key] for key in ("physical_reader_hash_ok",))
            and instrument["orientation_max_abs"] <= 1e-6
            and instrument["identity_self_and_live_replay_max_abs"] <= 1e-4
            and instrument["attention_reconstruction_max_abs"] <= 5e-4
            and result["full_live_replay_metrics"]["worst_six_cell_residual"] <= 1e-10
            and result["price"]["model_forwards"] == MAX_FORWARDS
            and result["price"]["example_evaluations"] == MAX_EVALUATIONS
            and all(math.isfinite(value) for by_task in decrements.values()
                    for value in by_task.values())
        )
        pred_b = bool(
            four["behavior_direction_fraction"] >= 0.90
            and all(four["tasks"][task]["behavior"]["signed_projection"] >= 0.75
                    and four["tasks"][task]["mode1"]["signed_projection"] >= 0.60
                    and four["tasks"][task]["mode2"]["signed_projection"] >= 0.60
                    and max(four["tasks"][task][family]["squared_residual"]
                            for family in ("behavior", "mode1", "mode2")) <= 0.20
                    for task in ("temporal", "iswas"))
        )
        pred_c = bool(
            five["behavior_direction_fraction"] >= 0.90
            and five["worst_six_cell_residual"] <= 0.20
            and all(five["tasks"][task]["behavior"]["signed_projection"] >= 0.80
                    and five["tasks"][task]["mode1"]["signed_projection"] >= 0.60
                    and five["tasks"][task]["mode2"]["signed_projection"] >= 0.60
                    for task in ("temporal", "iswas"))
        )
        pred_d = bool(
            all(decrements[site][task] >= 0.02 for site in FIVE
                for task in ("temporal", "iswas"))
            and decrements["MLP17"]["temporal"] >= 0.15
        )
        fidelity_gain = {
            task: mean_residual(four, task) - mean_residual(five, task)
            for task in ("temporal", "iswas")
        }
        iswas_behavior_gain = (
            five["tasks"]["iswas"]["behavior"]["signed_projection"]
            - four["tasks"]["iswas"]["behavior"]["signed_projection"]
        )
        pred_e = bool(all(value > 0 for value in fidelity_gain.values())
                      and iswas_behavior_gain >= 0.005)
        predictions = {
            "pred_a_authority_replay_self_clamp_finiteness_and_price": pred_a,
            "pred_b_four_site_mlp17_repair_transfers": pred_b,
            "pred_c_five_site_program_meets_strict_cells": pred_c,
            "pred_d_all_five_sites_have_mode2_support": pred_d,
            "pred_e_mlp12_is_a_reproducible_fidelity_term": pred_e,
        }
        result["schema"] = "temporal_v13_iswas_v12_four_five_mlp_program_result_v1"
        result["candidate_id"] = CANDIDATE_ID
        result["dryrun"]["candidate_id"] = CANDIDATE_ID
        result["evidence_scope"] = {
            "temporal": "genuinely_fresh_v13_confirmation",
            "iswas": "v12_replay_control_not_fresh_confirmation",
        }
        result["four_site_metrics"] = four
        result["five_site_metrics"] = five
        result["fidelity_gain_four_minus_five_mean_residual"] = fidelity_gain
        result["iswas_behavior_signed_projection_gain_from_MLP12"] = iswas_behavior_gain
        result["predictions"] = predictions
        result["terminal"] = (
            "invalid" if not pred_a
            else "temporal_fresh_five_mlp_response_program" if all(predictions.values())
            else "temporal_response_program_transfer_null"
        )
        original_write(OUT, result)
        print(json.dumps({key: result[key] for key in (
            "candidate_id", "evidence_scope", "four_site_metrics", "five_site_metrics",
            "mode2_signed_projection_decrements", "fidelity_gain_four_minus_five_mean_residual",
            "predictions", "terminal", "price")}, sort_keys=True))

    experiment.atomic_create_json = write_result


def main():
    configure()
    experiment.main()


if __name__ == "__main__":
    main()
