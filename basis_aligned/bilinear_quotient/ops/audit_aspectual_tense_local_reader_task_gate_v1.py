#!/usr/bin/env python3
"""Zero-forward held-out local-reader task gate and causal dispatch audit."""

# BQLANE: cpu
# BQGATE: AUDIT pred_a_hash_rows_split_and_margin_identity pred_b_frozen_three_way_gate pred_c_heldout_construction_task_routing pred_d_dispatched_A_P_causal_preservation pred_e_dispatched_C_selectivity pred_f_exact_coverage_and_zero_forward_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import circuit_candidate_aspectual_fresh_lexicon_v5 as has_builder
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v6 as is_builder
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_local_reader_task_gate_v1.json"
ALIGNED = ROOT / "circuits/followups/aspectual_tense_joint_upstream_program_composition_v1_result.json"
PAIRED = ROOT / "circuits/followups/aspectual_tense_paired_upstream_reader_state_audit_v1_result.json"
HAS_SCALE = ROOT / "circuits/followups/aspectual_anchor_resid10_frozen_gain_fresh_lexicon_v2_result.json"
IS_SCALE = ROOT / "circuits/followups/tense_auxiliary_is_was_selective_das_resid18_rank1_v1_result.json"
HAS_BUILDER = ROOT / "ops/circuit_candidate_aspectual_fresh_lexicon_v5.py"
IS_BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v6.py"
OUT = ROOT / "circuits/followups/aspectual_tense_local_reader_task_gate_v1_result.json"
CANDIDATE_ID = "aspectual_tense.local_reader_task_gate_v1"
EXPECTED_PRIOR_SHA256 = "fc0fe72c3065457a493d662794fc4637ac1e109814d9e7d1ecebf5c9c43ba0eb"
EXPECTED = {
    ALIGNED: "46479986f81751af6141e8fcbaf19d4413198b119171711715414d2869f43e08",
    PAIRED: "be5802364a1b5cfa6efb3e747eb63d0ca3372565ef7071bedeed7a85892a7efd",
    HAS_SCALE: "a4cee1818acd6a28e999eda26d0447f94d080b913f4061d0e6dab4914cb3802c",
    IS_SCALE: "36f80c04e4a1b2c6a7e0594126cd71e8781aab987521f8865d7e6de842346c02",
    HAS_BUILDER: "ae624913c5adfe07cf028acf6549cd5fe2debd4b090c71659218fe158089fe2c",
    IS_BUILDER: "b8541360334bd2793a02fae525a94dda05ce600fd4de5b6c3d953063d4c6b0ae",
}
EXPECTED_ROWS = {
    "has_had": "296c2186f477a6d450bbbb87fda5ba89b999eb4d3ac0dc18e31496ca47d5caf7",
    "is_was": "4eee90d9f39f6997c4926a0e7f6baecc4134c06535fe307d0a38f936b75defd5",
}
FEATURES = ("has_contrast", "is_contrast")
CLASSES = ("has_had", "is_was", "abstain")


class AuditError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    values = list(values)
    if not values or any(not math.isfinite(value) for value in values):
        raise AuditError("empty or nonfinite metric")
    return statistics.fmean(values)


def summarize_signed(values):
    values = list(values)
    return {
        "count": len(values),
        "mean": mean(values),
        "mean_absolute": mean(abs(value) for value in values),
        "direction_fraction": sum(value > 0.0 for value in values) / len(values),
    }


def main():
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps({"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False, "model_forwards": 0, "training_records": 48, "test_records": 80, "fitted_scalars": 10}, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256 or any(sha(path) != digest for path, digest in EXPECTED.items()):
        raise AuditError("prior or authority hash changed")

    prior = json.loads(PRIOR.read_text())
    aligned = json.loads(ALIGNED.read_text())
    paired = json.loads(PAIRED.read_text())
    has_scale_result = json.loads(HAS_SCALE.read_text())
    is_scale_result = json.loads(IS_SCALE.read_text())
    rows_by_bank = {"has_had": has_builder.build_rows(), "is_was": is_builder.build_rows()}
    row_hashes = {"has_had": has_builder.validate_rows(rows_by_bank["has_had"]), "is_was": is_builder.validate_rows(rows_by_bank["is_was"])}
    row_maps = {bank: {str(row["row_id"]): int(row["group_number"]) for row in rows} for bank, rows in rows_by_bank.items()}
    scales = {
        "has_had": float(has_scale_result["score"]["target_scale"]),
        "is_was": float(is_scale_result["score"]["families"]["target_scale"]),
    }
    records = []
    margin_keys = {"base", "own", "partner", "joint"}
    identity_ok = True
    for source in aligned["intervention_records"]:
        record = dict(source)
        bank, row_id = record.get("bank"), str(record.get("row_id"))
        finite_values = [record.get(feature) for feature in FEATURES]
        finite_values += list(record.get("margins", {}).values())
        identity_ok = identity_ok and bank in row_maps and row_id in row_maps.get(bank, {}) and margin_keys == set(record.get("margins", {}))
        identity_ok = identity_ok and all(isinstance(value, (int, float)) and math.isfinite(value) for value in finite_values)
        record["group_number"] = row_maps.get(bank, {}).get(row_id, -1)
        records.append(record)

    train, test = [], []
    for record in records:
        if record["family"] == "A1":
            item = dict(record, true_class=record["bank"], split="train")
            train.append(item)
        elif record["family"] == "C" and record["group_number"] <= 7:
            item = dict(record, true_class="abstain", split="train")
            train.append(item)
        elif record["family"] in ("A2", "P"):
            item = dict(record, true_class=record["bank"], split="test")
            test.append(item)
        elif record["family"] == "C" and record["group_number"] >= 8:
            item = dict(record, true_class="abstain", split="test")
            test.append(item)
        else:
            identity_ok = False

    means = {feature: mean(item[feature] for item in train) for feature in FEATURES}
    standard_deviations = {
        feature: math.sqrt(mean((item[feature] - means[feature]) ** 2 for item in train)) for feature in FEATURES
    }
    fit_ok = all(value > 0.0 and math.isfinite(value) for value in standard_deviations.values())

    def standardized(item):
        return tuple((item[feature] - means[feature]) / standard_deviations[feature] for feature in FEATURES)

    centroids = {}
    for class_name in CLASSES:
        selected = [standardized(item) for item in train if item["true_class"] == class_name]
        centroids[class_name] = tuple(mean(point[index] for point in selected) for index in range(len(FEATURES)))

    classified = []
    minimum_gap = math.inf
    for item in test:
        point = standardized(item)
        distances = {
            class_name: sum((point[index] - centroid[index]) ** 2 for index in range(len(FEATURES)))
            for class_name, centroid in centroids.items()
        }
        ranked = sorted(CLASSES, key=lambda class_name: (distances[class_name], CLASSES.index(class_name)))
        gap = distances[ranked[1]] - distances[ranked[0]]
        minimum_gap = min(minimum_gap, gap)
        predicted = ranked[0]
        margins = item["margins"]
        dispatched = margins["base"] if predicted == "abstain" else (margins["own"] if predicted == item["bank"] else margins["partner"])
        output = {
            "bank": item["bank"], "family": item["family"], "row_id": item["row_id"], "group_number": item["group_number"],
            "true_class": item["true_class"], "predicted_class": predicted, "correct": predicted == item["true_class"],
            "standardized_features": {feature: point[index] for index, feature in enumerate(FEATURES)},
            "squared_distances": distances, "distance_gap": gap, "base_margin": margins["base"], "dispatched_margin": dispatched,
        }
        if item["family"] == "A2":
            denominator = item["donor_reference_margin"] - margins["base"]
            if denominator == 0.0:
                raise AuditError("zero A2 recovery denominator")
            output["dispatched_recovery"] = (dispatched - margins["base"]) / denominator
        elif item["family"] == "P":
            denominator = -2.0 * margins["base"]
            if denominator == 0.0:
                raise AuditError("zero P reflection denominator")
            output["dispatched_reflection"] = (dispatched - margins["base"]) / denominator
        else:
            output["dispatched_normalized_unrelated_effect"] = abs(dispatched - margins["base"]) / scales[item["bank"]]
        classified.append(output)

    routing = {}
    for bank in ("has_had", "is_was"):
        routing[bank] = {}
        for family in ("A2", "P"):
            selected = [item for item in classified if item["bank"] == bank and item["family"] == family]
            routing[bank][family] = {"count": len(selected), "accuracy": sum(item["correct"] for item in selected) / len(selected)}
    c_test = [item for item in classified if item["family"] == "C"]
    routing["heldout_C_abstain"] = {"count": len(c_test), "accuracy": sum(item["correct"] for item in c_test) / len(c_test)}

    dispatched = {}
    for bank in ("has_had", "is_was"):
        a2 = [item["dispatched_recovery"] for item in classified if item["bank"] == bank and item["family"] == "A2"]
        p = [item["dispatched_reflection"] for item in classified if item["bank"] == bank and item["family"] == "P"]
        dispatched[bank] = {"A2": summarize_signed(a2), "P": summarize_signed(p)}
    c_effects = [item["dispatched_normalized_unrelated_effect"] for item in c_test]
    dispatched["heldout_C"] = {"count": len(c_effects), "mean_normalized_unrelated_effect": mean(c_effects), "max_normalized_unrelated_effect": max(c_effects)}

    expected_authorities = {
        "aligned_joint_result_sha256": EXPECTED[ALIGNED], "paired_reader_audit_sha256": EXPECTED[PAIRED],
        "q_has_scale_authority_sha256": EXPECTED[HAS_SCALE], "q_is_scale_authority_sha256": EXPECTED[IS_SCALE],
        "q_has_builder_sha256": EXPECTED[HAS_BUILDER], "q_is_builder_sha256": EXPECTED[IS_BUILDER],
    }
    pred_a = (
        prior.get("candidate_id") == CANDIDATE_ID and prior.get("authorities") == expected_authorities
        and aligned.get("terminal") == "null" and paired.get("terminal") == "null"
        and row_hashes == EXPECTED_ROWS and len(records) == 128 and identity_ok
        and scales == prior["frozen_gate"]["normalization_scales"]
    )
    pred_b = fit_ok and minimum_gap > 1e-12 and all(len([item for item in train if item["true_class"] == class_name]) == 16 for class_name in CLASSES)
    pred_c = all(routing[bank][family]["accuracy"] >= 0.75 for bank in ("has_had", "is_was") for family in ("A2", "P")) and routing["heldout_C_abstain"]["accuracy"] >= 0.75
    pred_d = all(dispatched[bank][family]["mean"] >= 0.75 and dispatched[bank][family]["direction_fraction"] >= 0.75 for bank in ("has_had", "is_was") for family in ("A2", "P"))
    pred_e = dispatched["heldout_C"]["mean_normalized_unrelated_effect"] <= 0.20
    pred_f = len(train) == 48 and len(test) == 80 and len(classified) == 80 and len({item["row_id"] for item in train}) == 48 and len({item["row_id"] for item in test}) == 80 and not ({item["row_id"] for item in train} & {item["row_id"] for item in test})
    predictions = {
        "pred_a_hash_rows_split_and_margin_identity": pred_a,
        "pred_b_frozen_three_way_gate": pred_b,
        "pred_c_heldout_construction_task_routing": pred_c,
        "pred_d_dispatched_A_P_causal_preservation": pred_d,
        "pred_e_dispatched_C_selectivity": pred_e,
        "pred_f_exact_coverage_and_zero_forward_price": pred_f,
    }
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_f else "invalid")
    reason = {"screen": "local_reader_task_gate_supported", "null": "local_reader_task_gate_misses", "invalid": "authority_fit_identity_split_or_coverage_invalid"}[terminal]
    result = {
        "schema": "aspectual_tense_local_reader_task_gate_result_v1", "candidate_id": CANDIDATE_ID,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "authority_sha256": expected_authorities, "rows_sha256": row_hashes,
        "gate": {"features": list(FEATURES), "classes_in_tie_order": list(CLASSES), "means": means, "population_standard_deviations": standard_deviations, "standardized_centroids": {key: list(value) for key, value in centroids.items()}, "minimum_test_distance_gap": minimum_gap, "fitted_scalars": 10},
        "routing": routing, "dispatched_causal_metrics": dispatched, "test_predictions": classified,
        "predictions": predictions,
        "price": {"model_forwards": 0, "example_evaluations": 0, "training_records": len(train), "test_records": len(test), "fitted_scalars": 10, "feature_dimensions": 2, "classes": 3, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0},
        "terminal": terminal, "reason": reason,
        "next_action": "compile the automatic dual-program gate and prospectively validate it on a disjoint lexicon" if terminal == "screen" else "seek a raw-text or earlier-state task branch signal without expanding writer rank",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "gate", "routing", "dispatched_causal_metrics", "predictions", "price", "terminal", "reason", "next_action")}, sort_keys=True))


if __name__ == "__main__":
    main()
