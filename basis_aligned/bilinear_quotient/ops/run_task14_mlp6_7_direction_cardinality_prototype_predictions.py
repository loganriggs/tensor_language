#!/usr/bin/env python3
"""Seal third-corpus predictions from fixed upstream prototypes and readers."""

# BQGATE: EXPERIMENT pred_a_capability_and_artifacts pred_b_one_thousand_twenty_four_predictions_sealed pred_c_no_target_exact_or_causal_outcomes
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import circuit_fast_screen_candidate_task14_cardinality_prototype_transfer as authority
import circuit_fast_screen_managed_runner as managed
import native_capability_license as licensing
import run_task14_cardinality_prototype_transfer_native_capability as capability
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as factor_gate


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_mlp6_7_fixed_direction_cardinality_upstream_program_v1.json"
PROTOTYPES = ROOT / "circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_artifact_v1.json"
OUT = ROOT / "circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_predictions_v1.json"
PRIOR_ART_SHA256 = "075c1f83f5801e2eb874d6df55b6070d56a6a0271716dd15e99d044e4f2c2f2d"
PROTOTYPE_SHA256 = "cce00d8f2309b0f9e0329c094238505819f2cf8a21c04e276af2085e068c1d07"
CAPABILITY_RESULT_SHA256 = "b5db9ccd55b8e244458cdbeada7246fb45eaa95c6abc19cdeeba1f7bd41e6a1c"
CAPABILITY_LICENSE_SHA256 = "c595bd0edf7e92b659f3d209836bec0c6d68524b0255c120d7651f71923f5af1"
SUBSETS = factor_gate.BACKGROUND_SUBSETS
PRED_KEYS = (
    "pred_a_capability_and_artifacts",
    "pred_b_one_thousand_twenty_four_predictions_sealed",
    "pred_c_no_target_exact_or_causal_outcomes",
)


class PredictionError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_prototypes() -> dict[str, object]:
    if _sha256(PROTOTYPES) != PROTOTYPE_SHA256:
        raise PredictionError("prototype artifact changed")
    artifact = json.loads(PROTOTYPES.read_text())
    if artifact.get("terminal") != "prototype_artifact" or not all(artifact.get("predictions", {}).values()) or len(artifact.get("prototypes", {})) != 12:
        raise PredictionError("prototype artifact is invalid")
    return artifact


def validate_preflight() -> None:
    if _sha256(PRIOR_ART) != PRIOR_ART_SHA256:
        raise PredictionError("prior art changed")
    for path, expected, label in (
        (capability.RESULT, CAPABILITY_RESULT_SHA256, "capability result"),
        (capability.LICENSE, CAPABILITY_LICENSE_SHA256, "capability license"),
    ):
        if _sha256(path) != expected:
            raise PredictionError(f"{label} changed")
    licensing.validate_causal_preflight(
        capability.build_gate(), capability.RESULT, capability.LICENSE,
        expected_license_sha256=CAPABILITY_LICENSE_SHA256,
        causal_candidate_id=authority.CAUSAL_CANDIDATE_ID,
    )
    _load_prototypes()


def derive_price() -> dict[str, int]:
    return {
        "physical_model_forwards": 0, "example_evaluations": 0,
        "causal_interventions": 0, "backwards": 0, "parameter_updates": 0,
        "sealed_predictions": 1024,
    }


def compile_plan() -> dict[str, object]:
    validate_preflight()
    return {
        "schema": "task14_mlp6_7_direction_cardinality_prototype_prediction_plan_v1",
        "candidate_id": authority.CAUSAL_CANDIDATE_ID,
        "split": "THIRD_CORPUS_PREDICTION_ONLY_BEFORE_CAUSAL_LATTICE",
        "row_count": 32, "background_subsets": list(SUBSETS),
        "prior_art_sha256": PRIOR_ART_SHA256,
        "prototype_artifact_sha256": PROTOTYPE_SHA256,
        "capability_license_sha256": CAPABILITY_LICENSE_SHA256,
        "target_exact_displacements_consumed": 0,
        "causal_outcomes_opened": False,
        "predictions": dict(zip(PRED_KEYS, (
            "candidate-scoped capability and frozen prototype artifact validate",
            "exactly 512 cardinality and 512 direction-only predictions are sealed",
            "prediction uses no third-corpus exact displacement, forward, intervention, outcome, fit, scale, or offset",
        ))),
        "price": derive_price(),
    }


def build_evidence(rows: list[dict[str, object]], prototypes: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    evidence = []
    for row in rows:
        direction = row["direction_id"]
        for subset in SUBSETS:
            cardinality_key = f"{direction}.cardinality_{len(subset)}"
            direction_key = f"{direction}.direction_only"
            evidence.append({
                "row_id": row["row_id"], "direction": direction,
                "template": row["template_id"], "background": subset,
                "cardinality": len(subset),
                "cardinality_prototype_key": cardinality_key,
                "direction_only_prototype_key": direction_key,
                "cardinality_reader_q": float(prototypes[cardinality_key]["frozen_reader_q"]),
                "direction_only_reader_q": float(prototypes[direction_key]["frozen_reader_q"]),
            })
    return evidence


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    if OUT.exists():
        raise PredictionError(f"refusing overwrite {OUT}")
    artifact = _load_prototypes()
    evidence = build_evidence(authority.build_rows(), artifact["prototypes"])
    unique = {(item["row_id"], item["background"]) for item in evidence}
    predictions = dict(zip(PRED_KEYS, (
        True,
        len(evidence) == 512 and len(unique) == 512 and 2 * len(evidence) == 1024,
        plan["target_exact_displacements_consumed"] == 0 and plan["causal_outcomes_opened"] is False,
    )))
    terminal = "sealed_prediction" if all(predictions.values()) else "invalid"
    payload = managed.atomic_create_json(OUT, {
        "schema": "task14_mlp6_7_direction_cardinality_prototype_predictions_v1",
        "candidate_id": authority.CAUSAL_CANDIDATE_ID, "terminal": terminal,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "plan": plan, "predictions": predictions, "evidence": evidence,
        "target_exact_displacements_consumed": 0, "causal_outcomes_opened": False,
    })
    print(json.dumps({"terminal": terminal, "predictions": predictions, "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
