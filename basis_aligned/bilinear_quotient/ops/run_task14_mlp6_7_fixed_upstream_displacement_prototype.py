#!/usr/bin/env python3
"""Score held-row-free upstream displacement prototypes through frozen readers."""

# BQGATE: EXPERIMENT pred_a_instrument_and_complete_join pred_b_exact_subset_prototype_predicts_all_effects pred_c_intermediate_compositions_transfer pred_d_each_direction_template_cell_transfers pred_e_factor_identity_beats_cardinality_control pred_f_target_free_provenance
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import circuit_fast_screen_managed_runner as managed


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_mlp6_7_fixed_upstream_displacement_prototype_v1.json"
SEALED = ROOT / "circuits/fast_screens/task14_fixed_direction_reader_cross_corpus_transfer_v1_predictions.json"
CAUSAL = ROOT / "circuits/fast_screens/task14_fixed_direction_reader_cross_corpus_transfer_v1_result.json"
READER = ROOT / "circuits/fast_screens/task14_mlp6_7_fixed_direction_reader_artifact_v2_result.json"
OUT = ROOT / "circuits/fast_screens/task14_mlp6_7_fixed_upstream_displacement_prototype_v1_result.json"
PRIOR_ART_SHA256 = "95f8633beee234d1c1a2cf80163fa2df53c49a08d290026cfb271d9b6734e9ff"
SEALED_SHA256 = "1d6a5ce082efb6f59b492b5d80c690979e1d3df1bd532d73edf49caaefc8cc81"
CAUSAL_SHA256 = "e53160cff6407c27dfd3a0e6b15740984db0d8b4468fcb96a10d32cd4a5f13b9"
READER_SHA256 = "9db4eefe16498cb65fb9c21ea3f2475c790c89ebb2e65a70e8ad6b7886f2ae57"
CANDIDATE_ID = "subject_verb.number_agreement.mlp6_7_fixed_upstream_displacement_prototype_v1"
PRED_KEYS = (
    "pred_a_instrument_and_complete_join",
    "pred_b_exact_subset_prototype_predicts_all_effects",
    "pred_c_intermediate_compositions_transfer",
    "pred_d_each_direction_template_cell_transfers",
    "pred_e_factor_identity_beats_cardinality_control",
    "pred_f_target_free_provenance",
)
BARS = {
    "minimum_overall_cosine": 0.80,
    "maximum_overall_relative_l2_error": 0.65,
    "minimum_overall_sign_agreement": 0.80,
    "minimum_intermediate_cosine": 0.75,
    "maximum_intermediate_relative_l2_error": 0.70,
    "minimum_intermediate_sign_agreement": 0.75,
    "minimum_cell_cosine": 0.65,
    "maximum_cell_relative_l2_error": 0.85,
    "minimum_cell_sign_agreement": 0.65,
    "minimum_sse_reduction_over_cardinality": 0.10,
}


class PrototypeError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_price() -> dict[str, int]:
    return {
        "physical_model_forwards": 0,
        "example_evaluations": 0,
        "causal_interventions": 0,
        "backwards": 0,
        "parameter_updates": 0,
        "sealed_scalar_inputs": 512,
        "causal_scalar_targets": 512,
    }


def compile_plan() -> dict[str, object]:
    for path, expected, label in (
        (PRIOR_ART, PRIOR_ART_SHA256, "prior art"),
        (SEALED, SEALED_SHA256, "sealed predictions"),
        (CAUSAL, CAUSAL_SHA256, "causal validation"),
        (READER, READER_SHA256, "reader artifact"),
    ):
        if _sha256(path) != expected:
            raise PrototypeError(f"{label} changed")
    reader = json.loads(READER.read_text())
    if reader.get("terminal") != "reader_artifact" or any(
        len(item.get("coordinates", [])) != 1152
        or not all(math.isfinite(float(x)) for x in item["coordinates"])
        for item in reader.get("readers", {}).values()
    ):
        raise PrototypeError("reader artifact is not two finite 1152-D vectors")
    return {
        "schema": "task14_mlp6_7_fixed_upstream_displacement_prototype_plan_v1",
        "candidate_id": CANDIDATE_ID,
        "split": "RETROSPECTIVE_SECOND_CORPUS_HELD_ROW_FREE",
        "prior_art_sha256": PRIOR_ART_SHA256,
        "sealed_prediction_sha256": SEALED_SHA256,
        "causal_validation_sha256": CAUSAL_SHA256,
        "reader_artifact_sha256": READER_SHA256,
        "prototype": "mean q_hat over other same-direction rows at exact factor subset",
        "control": "mean q_hat over other same-direction rows and all equal-cardinality subsets",
        "activation_reconstruction_is_verdict": False,
        "bars": dict(BARS),
        "price": derive_price(),
    }


def _mean(values: list[float]) -> float:
    if not values:
        raise PrototypeError("empty prototype pool")
    return sum(values) / len(values)


def build_predictions(sealed_evidence: list[dict[str, object]]) -> list[dict[str, object]]:
    if len(sealed_evidence) != 512:
        raise PrototypeError("sealed evidence must contain 512 rows")
    unique = {(x["row_id"], x["background"]) for x in sealed_evidence}
    if len(unique) != 512:
        raise PrototypeError("sealed evidence keys are not unique")
    predictions = []
    for held in sealed_evidence:
        exact_pool = [
            float(x["fixed_reader_q"])
            for x in sealed_evidence
            if x["row_id"] != held["row_id"]
            and x["direction"] == held["direction"]
            and x["background"] == held["background"]
        ]
        cardinality_pool = [
            float(x["fixed_reader_q"])
            for x in sealed_evidence
            if x["row_id"] != held["row_id"]
            and x["direction"] == held["direction"]
            and int(x["cardinality"]) == int(held["cardinality"])
        ]
        expected_control = 15 * math.comb(4, int(held["cardinality"]))
        if len(exact_pool) != 15 or len(cardinality_pool) != expected_control:
            raise PrototypeError("prototype pool does not enforce held-row exclusion")
        predictions.append({
            "row_id": held["row_id"],
            "background": held["background"],
            "direction": held["direction"],
            "template": held["template"],
            "cardinality": int(held["cardinality"]),
            "exact_subset_prototype_q": _mean(exact_pool),
            "cardinality_control_q": _mean(cardinality_pool),
            "exact_pool_rows": len(exact_pool),
            "cardinality_pool_values": len(cardinality_pool),
            "held_row_excluded": True,
        })
    return predictions


def _stats(items: list[dict[str, object]], prediction_field: str) -> dict[str, float | int]:
    actual = [float(x["actual_q"]) for x in items]
    predicted = [float(x[prediction_field]) for x in items]
    dot = sum(x * y for x, y in zip(actual, predicted))
    actual_norm = math.sqrt(sum(x * x for x in actual))
    predicted_norm = math.sqrt(sum(x * x for x in predicted))
    return {
        "count": len(items),
        "cosine": dot / max(actual_norm * predicted_norm, 1e-30),
        "relative_l2_error": math.sqrt(sum((x - y) ** 2 for x, y in zip(actual, predicted))) / max(actual_norm, 1e-30),
        "sign_agreement": sum((x > 0) == (y > 0) for x, y in zip(actual, predicted)) / len(items),
        "sse": sum((x - y) ** 2 for x, y in zip(actual, predicted)),
    }


def score(
    predictions: list[dict[str, object]], causal_evidence: list[dict[str, object]], bars: dict[str, float] = BARS,
) -> dict[str, object]:
    causal = {(x["row_id"], x["background"]): x for x in causal_evidence}
    joined = []
    for prediction in predictions:
        outcome = causal.get((prediction["row_id"], prediction["background"]))
        if outcome is None:
            raise PrototypeError("missing causal target")
        for key in ("direction", "template", "cardinality"):
            if prediction[key] != outcome[key]:
                raise PrototypeError(f"joined metadata mismatch: {key}")
        joined.append({**prediction, "actual_q": float(outcome["actual_q"])})
    finite = len(joined) == 512 and all(
        math.isfinite(float(x[key]))
        for x in joined
        for key in ("actual_q", "exact_subset_prototype_q", "cardinality_control_q")
    )
    overall = {name: _stats(joined, field) for name, field in (
        ("exact_subset", "exact_subset_prototype_q"),
        ("cardinality_control", "cardinality_control_q"),
    )}
    intermediate_rows = [x for x in joined if x["background"] not in {"", "EAUW"}]
    intermediate = _stats(intermediate_rows, "exact_subset_prototype_q")
    cells = {
        f"{direction}.{template}": _stats(
            [x for x in joined if x["direction"] == direction and x["template"] == template],
            "exact_subset_prototype_q",
        )
        for direction in ("plural_to_singular", "singular_to_plural")
        for template in ("above_inside", "inside_above")
    }
    exact = overall["exact_subset"]
    reduction = 1.0 - float(exact["sse"]) / max(float(overall["cardinality_control"]["sse"]), 1e-30)
    pred_a = finite and len({(x["row_id"], x["background"]) for x in joined}) == 512
    pred_b = (
        float(exact["cosine"]) >= bars["minimum_overall_cosine"]
        and float(exact["relative_l2_error"]) <= bars["maximum_overall_relative_l2_error"]
        and float(exact["sign_agreement"]) >= bars["minimum_overall_sign_agreement"]
    )
    pred_c = (
        float(intermediate["cosine"]) >= bars["minimum_intermediate_cosine"]
        and float(intermediate["relative_l2_error"]) <= bars["maximum_intermediate_relative_l2_error"]
        and float(intermediate["sign_agreement"]) >= bars["minimum_intermediate_sign_agreement"]
    )
    pred_d = all(
        float(cell["cosine"]) >= bars["minimum_cell_cosine"]
        and float(cell["relative_l2_error"]) <= bars["maximum_cell_relative_l2_error"]
        and float(cell["sign_agreement"]) >= bars["minimum_cell_sign_agreement"]
        for cell in cells.values()
    )
    pred_f = all(
        x["held_row_excluded"] is True
        and x["exact_pool_rows"] == 15
        and x["cardinality_pool_values"] == 15 * math.comb(4, int(x["cardinality"]))
        for x in joined
    )
    verdicts = (pred_a, pred_a and pred_b, pred_a and pred_c, pred_a and pred_d, pred_a and reduction >= bars["minimum_sse_reduction_over_cardinality"], pred_a and pred_f)
    return {
        "overall": overall,
        "intermediate_only": intermediate,
        "by_direction_template": cells,
        "sse_reduction_over_cardinality_control": reduction,
        "provenance": {
            "held_row_displacement_consumed": False,
            "prediction_reads_causal_outcomes": False,
            "fit_operations": 0,
            "activation_reconstruction_is_verdict": False,
        },
        "predictions": dict(zip(PRED_KEYS, map(bool, verdicts))),
        "joined_evidence": joined,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    if OUT.exists():
        raise PrototypeError(f"refusing overwrite {OUT}")
    sealed = json.loads(SEALED.read_text())
    causal = json.loads(CAUSAL.read_text())
    predictions = build_predictions(sealed["evidence"])
    scored = score(predictions, causal["causal_evidence"])
    terminal = "valid_retrospective_screen" if scored["predictions"][PRED_KEYS[0]] else "invalid"
    payload = managed.atomic_create_json(OUT, {
        "schema": "task14_mlp6_7_fixed_upstream_displacement_prototype_result_v1",
        "candidate_id": CANDIDATE_ID,
        "terminal": terminal,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "plan": plan,
        "score": scored,
    })
    print(json.dumps({
        "terminal": terminal,
        "predictions": scored["predictions"],
        "result_sha256": hashlib.sha256(payload).hexdigest(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
