#!/usr/bin/env python3
# BQLANE: cpu
# BQGATE: frozen A-E zero-forward source-compression release audit.
"""Audit release of preserved source-compression rows after an indexed diagnostic."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_attention11h3_15h5_source_compression_release_v1.json"
SCIENCE = ROOT / "circuits/followups/aspectual_anchor_attention11h3_15h5_source_compression_split_v1_result.json"
SCIENCE_PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_attention11h3_15h5_source_compression_split_v1.json"
SCIENCE_RUNNER = ROOT / "ops/run_aspectual_anchor_attention11h3_15h5_source_compression_split_v1.py"
DIAGNOSTIC = ROOT / "circuits/followups/aspectual_anchor_attention11h3_15h5_source_projection_query_diagnostic_v1_result.json"
DIAGNOSTIC_PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_attention11h3_15h5_source_projection_query_diagnostic_v1.json"
DIAGNOSTIC_RUNNER = ROOT / "ops/run_aspectual_anchor_attention11h3_15h5_source_projection_query_diagnostic_v1.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_attention11h3_15h5_source_compression_release_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.attention11h3_15h5_source_compression_release_v1"
EXPECTED = {
    PRIOR: "ad5051a746b0ecd422223f6b396396d5f1a51f77927624956518b63167705b66",
    SCIENCE: "7b4ba19260f4d311ca959331e60f29c64136f416566582d510591c96d5243adb",
    SCIENCE_PRIOR: "6cfd9e357cc3c7797c35ef206316918fbb832c6f5792d2c52a62e9022a688a96",
    SCIENCE_RUNNER: "419c0be9a1c9cebf594225c492b9402af7240d7ffd95d7987320d3e2ecb18a30",
    DIAGNOSTIC: "e14a47c743fcb31b23591c0ff9579a5282aa9d1f98c35f272a32929951cd9ba8",
    DIAGNOSTIC_PRIOR: "fc9a4c39adaef15c6dbc13da297a6184c25be16e49c2e2f06ce98d5269605863",
    DIAGNOSTIC_RUNNER: "83bdfbeb850c51f32c09316085571cce8394cdbc2c9fcc6089fbbb8466699db2",
}
BOUNDARIES = ("11", "15")


class AuditError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def main() -> None:
    dryrun = {
        "schema": "aspectual_anchor_attention11h3_15h5_source_compression_release_dryrun_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "cpu_only",
        "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
        "model_forwards": 0, "example_evaluations": 0, "model_backwards": 0,
        "model_updates": 0, "fit_parameters": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    hash_checks = {str(path.relative_to(ROOT)): sha256(path) == digest for path, digest in EXPECTED.items()}
    prior = json.loads(PRIOR.read_text())
    science = json.loads(SCIENCE.read_text())
    science_prior = json.loads(SCIENCE_PRIOR.read_text())
    diagnostic = json.loads(DIAGNOSTIC.read_text())
    records = science.get("intervention_logits", [])
    science_prediction_values = science.get("predictions", {})

    pred_a = (
        all(hash_checks.values()) and prior.get("candidate_id") == CANDIDATE_ID
        and science.get("terminal") == "invalid" and diagnostic.get("terminal") == "screen"
        and tuple(science_prediction_values) == tuple(science_prior.get("predictions", {}))
        and tuple(science_prediction_values.values()) == (False, True, True, True, True)
        and all(diagnostic.get("predictions", {}).values())
    )
    unique_records = {
        (record.get("phase"), str(record.get("boundary")), record.get("arm_id"), record.get("row_id"))
        for record in records
    }
    finite_records = all(
        all(math.isfinite(float(record[field])) for field in ("answer_logit", "foil_logit", "recovery"))
        for record in records
    )
    pred_b = (
        science_prior.get("created_utc", "") < science.get("started_utc", "")
        and science.get("prior_art_sha256") == EXPECTED[SCIENCE_PRIOR]
        and science.get("selection_row_ids_sha256") == science_prior["population_split"]["selection_row_ids_sha256"]
        and science.get("confirmation_row_ids_sha256") == science_prior["population_split"]["confirmation_row_ids_sha256"]
        and len(records) == len(unique_records) == 576 and finite_records
        and science["score"]["forward_calls"] <= science_prior["price"]["model_forwards_max"]
        and science["score"]["example_evaluations"] <= science_prior["price"]["example_evaluations_max"]
        and science["score"]["fit_parameters"] == 0
    )
    diagnostic_score = diagnostic.get("score", {})
    pred_c = (
        all(float(diagnostic_score["query_projection_max_abs"][boundary]) <= 0.04 for boundary in BOUNDARIES)
        and all(float(diagnostic_score["attention_term_reconstruction_max_abs"][boundary]) <= 1.0e-4 for boundary in BOUNDARIES)
        and all(
            float(diagnostic_score["full_sequence_projection_max_abs"][boundary]) > 1.0
            and abs(
                float(diagnostic_score["full_sequence_projection_max_abs"][boundary])
                - float(diagnostic_score["off_query_projection_max_abs"][boundary])
            ) <= 1.0e-6 for boundary in BOUNDARIES
        )
        and diagnostic_score.get("partition_rows") == 16
    )
    selection = science.get("score", {}).get("selection", {})
    confirmation = science.get("score", {}).get("confirmation", {})
    pred_d = True
    released_banks = {}
    for boundary in BOUNDARIES:
        selected = selection[boundary]["selected_roles"]
        family_increments = confirmation[boundary]["selected_family_increments"]
        valid = (
            len(selected) == len(set(selected)) == 3
            and all(role in science_prior["frozen_design"]["source_roles"] for role in selected)
            and selection[boundary]["all_minus_none_increment"] > 0.0
            and confirmation[boundary]["selected_roles"] == selected
            and confirmation[boundary]["selected_to_all_source_fraction"] >= 0.70
            and all(float(family_increments[family]) > 0.0 for family in ("A1", "A2"))
        )
        pred_d = pred_d and valid
        released_banks[boundary] = {
            "head": {"11": 3, "15": 5}[boundary], "source_roles": selected,
            "confirmation_retained_fraction": confirmation[boundary]["selected_to_all_source_fraction"],
            "confirmation_family_increments": family_increments,
        }
    pred_e = (
        dryrun["model_forwards"] == dryrun["example_evaluations"] == dryrun["fit_parameters"] == 0
        and prior["price"] == {
            "model_forwards": 0, "example_evaluations": 0, "model_backwards": 0,
            "model_updates": 0, "fit_parameters": 0,
        }
        and "post-outcome" in science.get("next_action", "") + diagnostic.get("scope_boundary", "").lower()
        and science.get("terminal") == "invalid"
    )
    predictions = {
        "pred_a_authority_and_failure_localization": pred_a,
        "pred_b_scientific_design_integrity": pred_b,
        "pred_c_query_instrument_validity": pred_c,
        "pred_d_compression_claim": pred_d,
        "pred_e_scope_and_price": pred_e,
    }
    terminal = "release" if all(predictions.values()) else "withhold"
    result = {
        "schema": "aspectual_anchor_attention11h3_15h5_source_compression_release_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "cpu_only",
        "created_utc": utc_now(), "prior_art_sha256": EXPECTED[PRIOR],
        "authority_hash_checks": hash_checks, "dryrun": dryrun,
        "predictions": predictions, "released_banks": released_banks if terminal == "release" else {},
        "evidence_class": "prospective_scientific_arms_with_post_outcome_instrument_audit_repair",
        "immutable_v1_terminal": science.get("terminal"),
        "scope_boundary": (
            "Paired-causal source banks for the two validated aspectual constructions; checkpoint, paired states, "
            "full carried/MLP boundary deltas, and native suffix remain required. This is not standalone, free-form, "
            "new-construction, native-margin, full-logit, or whole-model prediction."
        ),
        "price": prior["price"], "terminal": terminal,
        "reason": "source_banks_released_with_explicit_audit_repair" if terminal == "release" else "release_audit_failed",
        "next_action": "compile the released source banks into the executable transparent path program" if terminal == "release" else "retain dominant heads without source-bank promotion",
    }
    from circuit_fast_screen_managed_runner import atomic_create_json
    atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID, "terminal": terminal,
        "predictions": predictions, "released_banks": result["released_banks"],
        "result": str(OUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
