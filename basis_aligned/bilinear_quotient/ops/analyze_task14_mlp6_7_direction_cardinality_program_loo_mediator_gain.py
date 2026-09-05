#!/usr/bin/env python3
"""LOO scalar composition from fixed program reader to MLP15/17 mediation."""

# BQGATE: EXPERIMENT pred_a_exact_join_and_row_holdout pred_b_mlp15_predictive pred_c_mlp17_predictive pred_d_joint_mediation_predictive pred_e_cardinality_is_needed pred_f_fixed_price_and_scope
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
PRIOR_ART = ROOT / "circuits/prior_art/task14_mlp6_7_direction_cardinality_program_loo_mediator_gain_v1.json"
PROGRAM = ROOT / "circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_causal_validation_v1_result.json"
MEDIATION = ROOT / "circuits/followups/task14_mlp6_7_direction_cardinality_program_mlp15_vs_mlp17_mediation_v1_result.json"
OUT = ROOT / "circuits/followups/task14_mlp6_7_direction_cardinality_program_loo_mediator_gain_v1_result.json"
PRIOR_ART_SHA256 = "0a957acb7e3851a8e57cf89ef1134bd67f02e11e8d10f3e80f7daf97dd92c1a8"
PROGRAM_SHA256 = "9a488259efdb85477f35c612696368e8a3e372338a11253615a7b53650c88fe0"
MEDIATION_SHA256 = "9c43f964aa0976b91925e188c6f66bd585891915ca64c32411c6ad9cf75660e5"
CANDIDATE_ID = "subject_verb.number_agreement.mlp6_7_direction_cardinality_program_loo_mediator_gain_v1"
COMPONENTS = ("m15", "m17", "interaction")
BARS = {"m15_min_cosine": .75, "m15_max_relative_l2": .75, "m15_min_sign": .65, "m17_min_cosine": .90, "m17_max_relative_l2": .40, "m17_min_sign": .90, "joint_min_cosine": .90, "joint_max_relative_l2": .40, "joint_min_sign": .90, "group_joint_min_cosine": .85, "minimum_sse_reduction": .10}
PRED_KEYS = ("pred_a_exact_join_and_row_holdout", "pred_b_mlp15_predictive", "pred_c_mlp17_predictive", "pred_d_joint_mediation_predictive", "pred_e_cardinality_is_needed", "pred_f_fixed_price_and_scope")


class GainError(ValueError):
    pass


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load():
    for path, expected, label in ((PRIOR_ART, PRIOR_ART_SHA256, "prior art"), (PROGRAM, PROGRAM_SHA256, "program"), (MEDIATION, MEDIATION_SHA256, "mediation")):
        if _sha256(path) != expected:
            raise GainError(f"{label} changed")
    program = json.loads(PROGRAM.read_text())
    mediation = json.loads(MEDIATION.read_text())
    readers = {(x["row_id"], x["background"]): float(x["sealed_cardinality_reader_q"]) for x in program["score"]["joined_evidence"]}
    rows = []
    for item in mediation["evidence"]:
        key = (item["row_id"], item["background"])
        if key not in readers:
            raise GainError("mediation cell lacks sealed reader prediction")
        rows.append({**item, "reader_q": readers[key]})
    if len(rows) != 512 or len(readers) != 512 or len({(x["row_id"], x["background"]) for x in rows}) != 512:
        raise GainError("immutable sources do not join one-to-one")
    return rows


def compile_plan():
    rows = _load()
    return {"schema": "task14_mlp6_7_direction_cardinality_program_loo_mediator_gain_plan_v1", "candidate_id": CANDIDATE_ID, "row_count": len(rows), "components": list(COMPONENTS), "model": "component_hat=a[component,direction,cardinality]*sealed_reader_q", "fold": "exclude every cell sharing the held row_id", "control": "a[component,direction] under identical row folds", "stored_full_data_scalar_count": 30, "prior_art_sha256": PRIOR_ART_SHA256, "program_sha256": PROGRAM_SHA256, "mediation_sha256": MEDIATION_SHA256, "bars": dict(BARS), "price": {"cpu_only": True, "predictions": 512, "gpu_model_forwards": 0, "backwards": 0, "parameter_updates": 0}}


def _slope(rows, component):
    den = sum(x["reader_q"] ** 2 for x in rows)
    if den <= 1e-30:
        raise GainError("gain fit has zero reader support")
    return sum(x["reader_q"] * x[component] for x in rows) / den


def _stats(rows, actual, predicted):
    a = [float(x[actual]) for x in rows]
    p = [float(x[predicted]) for x in rows]
    an = math.sqrt(sum(x * x for x in a)); pn = math.sqrt(sum(x * x for x in p))
    return {"count": len(rows), "cosine": sum(x * y for x, y in zip(a, p)) / max(an * pn, 1e-30), "relative_l2_error": math.sqrt(sum((x-y)**2 for x,y in zip(a,p))) / max(an, 1e-30), "sign_agreement": sum((x > 0) == (y > 0) for x,y in zip(a,p)) / len(a), "sse": sum((x-y)**2 for x,y in zip(a,p))}


def evaluate():
    rows = _load()
    predicted = []
    for held in rows:
        train = [x for x in rows if x["row_id"] != held["row_id"]]
        exact_group = [x for x in train if x["direction"] == held["direction"] and x["cardinality"] == held["cardinality"]]
        direction_group = [x for x in train if x["direction"] == held["direction"]]
        if not exact_group or not direction_group or any(x["row_id"] == held["row_id"] for x in exact_group + direction_group):
            raise GainError("row-held-out fold is empty or contaminated")
        values = {}
        for component in COMPONENTS:
            values[f"gain_{component}"] = _slope(exact_group, component)
            values[f"pred_{component}"] = values[f"gain_{component}"] * held["reader_q"]
            values[f"control_gain_{component}"] = _slope(direction_group, component)
            values[f"control_pred_{component}"] = values[f"control_gain_{component}"] * held["reader_q"]
        values["pred_joint"] = sum(values[f"pred_{c}"] for c in COMPONENTS)
        values["control_pred_joint"] = sum(values[f"control_pred_{c}"] for c in COMPONENTS)
        predicted.append({**held, **values})
    full_gains = {f"{component}/{direction}/cardinality_{k}": _slope([x for x in rows if x["direction"] == direction and x["cardinality"] == k], component) for component in COMPONENTS for direction in ("singular_to_plural", "plural_to_singular") for k in range(5)}
    return predicted, full_gains


def score(rows, full_gains):
    component_stats = {c: _stats(rows, c, f"pred_{c}") for c in COMPONENTS}
    joint = _stats(rows, "m_both", "pred_joint")
    control = _stats(rows, "m_both", "control_pred_joint")
    groups = {f"{d}/{t}": _stats([x for x in rows if x["direction"] == d and x["template"] == t], "m_both", "pred_joint") for d in ("singular_to_plural", "plural_to_singular") for t in ("near_beyond", "beyond_near")}
    reduction = 1 - joint["sse"] / max(control["sse"], 1e-30)
    pred_a = len(rows) == 512 and len(full_gains) == 30 and all(sum(x["row_id"] != held["row_id"] and x["direction"] == held["direction"] and x["cardinality"] == held["cardinality"] for x in rows) > 0 for held in rows)
    a, b = component_stats["m15"], component_stats["m17"]
    pred_b = a["cosine"] >= BARS["m15_min_cosine"] and a["relative_l2_error"] <= BARS["m15_max_relative_l2"] and a["sign_agreement"] >= BARS["m15_min_sign"]
    pred_c = b["cosine"] >= BARS["m17_min_cosine"] and b["relative_l2_error"] <= BARS["m17_max_relative_l2"] and b["sign_agreement"] >= BARS["m17_min_sign"]
    pred_d = joint["cosine"] >= BARS["joint_min_cosine"] and joint["relative_l2_error"] <= BARS["joint_max_relative_l2"] and joint["sign_agreement"] >= BARS["joint_min_sign"] and all(x["cosine"] >= BARS["group_joint_min_cosine"] for x in groups.values())
    pred_e = reduction >= BARS["minimum_sse_reduction"]
    predictions = dict(zip(PRED_KEYS, (pred_a, pred_b, pred_c, pred_d, pred_e, True)))
    terminal = "invalid" if not (pred_a and predictions[PRED_KEYS[5]]) else "prototype_gain_screen" if all(predictions.values()) else "null" if not pred_d else "inconclusive"
    return {"component_stats": component_stats, "joint_mediation": joint, "direction_only_control": control, "joint_by_direction_template": groups, "cardinality_sse_reduction_over_direction_only": reduction, "full_data_gains": full_gains, "predictions": predictions, "terminal": terminal}


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true"); args = parser.parse_args(argv)
    plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True)); return
    if OUT.exists(): raise GainError(f"refusing overwrite {OUT}")
    rows, gains = evaluate(); scored = score(rows, gains)
    payload = managed.atomic_create_json(OUT, {"schema": "task14_mlp6_7_direction_cardinality_program_loo_mediator_gain_result_v1", "candidate_id": CANDIDATE_ID, "terminal": scored["terminal"], "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "plan": plan, "score": scored, "evidence": rows})
    print(json.dumps({"terminal": scored["terminal"], "predictions": scored["predictions"], "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__": main()
