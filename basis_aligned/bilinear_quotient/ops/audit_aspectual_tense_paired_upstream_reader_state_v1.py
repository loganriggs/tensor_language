#!/usr/bin/env python3
"""Zero-forward shared-state versus task-gated reader audit."""

# BQLANE: cpu
# BQGATE: AUDIT pred_a_hash_schema_identity_and_exact_coverage pred_b_within_bank_shared_temporal_observation pred_c_bidirectional_cross_bank_affine_transport pred_d_target_over_control_separation pred_e_zero_forward_price_and_no_postselection
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_paired_upstream_reader_state_audit_v1.json"
ALIGNED = ROOT / "circuits/followups/aspectual_tense_joint_upstream_program_composition_v1_result.json"
OPPOSED = ROOT / "circuits/followups/aspectual_tense_opposed_command_program_composition_v1_result.json"
OUT = ROOT / "circuits/followups/aspectual_tense_paired_upstream_reader_state_audit_v1_result.json"
CANDIDATE_ID = "aspectual_tense.paired_upstream_reader_state_audit_v1"
EXPECTED_PRIOR_SHA256 = "459b8870901f16c6330e432ca4db09686ffea778ba37f899415998389b1c2c3e"
EXPECTED = {ALIGNED: "46479986f81751af6141e8fcbaf19d4413198b119171711715414d2869f43e08", OPPOSED: "cb25f6a064565ec7d59e6315e099c715b83d9850c195a4d522e84ca93cbf5d9c"}


class AuditError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        raise AuditError("invalid correlation vectors")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        raise AuditError("degenerate correlation vector")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def fit_line(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    denominator = sum((x - mx) ** 2 for x in xs)
    if denominator <= 0.0:
        raise AuditError("degenerate fit feature")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denominator
    return {"intercept": my - slope * mx, "slope": slope, "train_count": len(xs)}


def score_line(fit, xs, ys):
    predictions = [fit["intercept"] + fit["slope"] * x for x in xs]
    mean_y = statistics.fmean(ys)
    denominator = sum((y - mean_y) ** 2 for y in ys)
    if denominator <= 0.0:
        raise AuditError("degenerate held-out target")
    return {"test_count": len(xs), "r2": 1.0 - sum((y - prediction) ** 2 for y, prediction in zip(ys, predictions)) / denominator, "rmse": math.sqrt(statistics.fmean((y - prediction) ** 2 for y, prediction in zip(ys, predictions)))}


def main():
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps({"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False, "model_forwards": 0, "target_records": 96, "control_records": 32, "ols_fit_parameters": 16}, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256 or any(sha(path) != digest for path, digest in EXPECTED.items()):
        raise AuditError("prior or authority hash changed")
    prior, aligned, opposed = json.loads(PRIOR.read_text()), json.loads(ALIGNED.read_text()), json.loads(OPPOSED.read_text())
    records = aligned["intervention_records"]
    finite_identity = all(set(("bank", "family", "row_id", "direction", "has_contrast", "is_contrast")) <= set(record) and math.isfinite(record["has_contrast"]) and math.isfinite(record["is_contrast"]) for record in records)
    target = [record for record in records if record["family"] in ("A1", "A2", "P")]
    controls = [record for record in records if record["family"] == "C"]
    authority_ok = (
        prior.get("candidate_id") == CANDIDATE_ID
        and aligned.get("terminal") == "null"
        and aligned["reason"] == "joint_preservation_or_composition_law_misses"
        and all(aligned["predictions"]["pred_" + suffix] for suffix in ("a_authority_capability_basis_and_exact_heads", "b_frozen_dual_program_identity_and_own_route", "c_joint_has_had_program_preservation", "d_joint_is_was_program_preservation", "f_exact_coverage_and_price"))
        and opposed.get("terminal") == "invalid"
        and opposed["predictions"]["pred_" + "e_additive_law_and_live_opposition"] is False
        and len(records) == 128 and len(target) == 96 and len(controls) == 32 and finite_identity
    )

    within = {"target": {}, "control": {}}
    for label, population in (("target", target), ("control", controls)):
        for bank in ("has_had", "is_was"):
            within[label][bank] = {}
            for direction in ("past_to_present", "present_to_past"):
                selected = [record for record in population if record["bank"] == bank and record["direction"] == direction]
                within[label][bank][direction] = {"count": len(selected), "pearson_has_is": pearson([record["has_contrast"] for record in selected], [record["is_contrast"] for record in selected])}

    transports = []
    for direction in ("past_to_present", "present_to_past"):
        by_bank = {bank: [record for record in target if record["bank"] == bank and record["direction"] == direction] for bank in ("has_had", "is_was")}
        for train_bank, test_bank in (("has_had", "is_was"), ("is_was", "has_had")):
            train, test = by_bank[train_bank], by_bank[test_bank]
            for source_key, target_key in (("has_contrast", "is_contrast"), ("is_contrast", "has_contrast")):
                fit = fit_line([record[source_key] for record in train], [record[target_key] for record in train])
                score = score_line(fit, [record[source_key] for record in test], [record[target_key] for record in test])
                transports.append({"direction": direction, "train_bank": train_bank, "test_bank": test_bank, "source": source_key, "target": target_key, **fit, **score})

    target_correlations = [abs(cell["pearson_has_is"]) for bank in within["target"].values() for cell in bank.values()]
    control_correlations = [abs(cell["pearson_has_is"]) for bank in within["control"].values() for cell in bank.values()]
    separation = statistics.median(target_correlations) - statistics.median(control_correlations)
    pred_a = authority_ok
    pred_b = all(value >= 0.70 for value in target_correlations)
    pred_c = all(item["r2"] >= 0.50 for item in transports)
    pred_d = separation >= 0.20
    pred_e = len(transports) == 8 and all(item["train_count"] == item["test_count"] == 24 for item in transports)
    predictions = {
        "pred_a_hash_schema_identity_and_exact_coverage": pred_a,
        "pred_b_within_bank_shared_temporal_observation": pred_b,
        "pred_c_bidirectional_cross_bank_affine_transport": pred_c,
        "pred_d_target_over_control_separation": pred_d,
        "pred_e_zero_forward_price_and_no_postselection": pred_e,
    }
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_e else "invalid")
    reason = {"screen": "shared_upstream_temporal_scalar_supported", "null": "paired_readers_require_task_gating", "invalid": "authority_identity_coverage_finiteness_or_price_invalid"}[terminal]
    value = {
        "schema": "aspectual_tense_paired_upstream_reader_state_audit_result_v1", "candidate_id": CANDIDATE_ID,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "authority_sha256": {"aligned_joint_result": EXPECTED[ALIGNED], "opposed_instrument_result": EXPECTED[OPPOSED]},
        "within_bank": within, "cross_bank_transports": transports,
        "separation": {"target_median_absolute_correlation": statistics.median(target_correlations), "control_median_absolute_correlation": statistics.median(control_correlations), "difference": separation},
        "predictions": predictions, "price": {"model_forwards": 0, "example_evaluations": 0, "target_records": len(target), "control_records": len(controls), "ols_fit_parameters": 16, "feature_dimensions": 2, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0},
        "terminal": terminal, "reason": reason,
        "next_action": "compile shared temporal state with task-specific observation and writers" if terminal == "screen" else "retain task-gated local readers and design a naturally opposed multi-variable authority",
    }
    atomic_create_json(OUT, value)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "within_bank": within, "cross_bank_transports": transports, "separation": value["separation"], "price": value["price"], "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
