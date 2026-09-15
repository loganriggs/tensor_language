#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: 5forwards80seq; subject-number rank-one two-site composition;0fits;0updates;noquantization.
"""Test additive composition of two frozen subject-number rank-one writes."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

RUNNER = Path(__file__).resolve()
OPS = RUNNER.parent
ROOT = RUNNER.parents[3]
POLY = ROOT / "basis_aligned/polynomial_causal"
ROWS = POLY / "SUBJECT_NUMBER_TWO_SITE_COMPOSITION_V2_ROWS.json"
RANK1 = POLY / "SUBJECT_NUMBER_DIRECTION_CARDINALITY_RANK1_V1_ARTIFACT.json"
PREREG = POLY / "SUBJECT_NUMBER_TWO_SITE_COMPOSITION_V2_PREREGISTRATION.md"
FRESH_RESULT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/subject_number_rank1_fresh_confirmation_v1_result.json"
BINDING = POLY / "SUBJECT_NUMBER_TWO_SITE_COMPOSITION_V2_BINDING.json"
OUT = POLY / "SUBJECT_NUMBER_TWO_SITE_COMPOSITION_V2_RESULT.json"
REQUESTED_DRY = bool(os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"))
LAYER = 11
HEAD = 3
SITE_POSITIONS = (5, 14)
ARMS = ("base", "zero", "site1", "site2", "both")
BARS = {
    "maximum_zero_replay_logit_error": 1e-5,
    "minimum_native_accuracy": 0.75,
    "minimum_single_site_effect_rms": 0.01,
    "minimum_single_site_positive_fraction": 0.75,
    "maximum_anticausal_absolute_effect": 1e-5,
    "maximum_unrelated_control_fraction": 0.75,
    "minimum_composition_cosine": 0.95,
    "maximum_composition_relative_l2": 0.25,
    "minimum_composition_sign": 0.90,
    "minimum_composition_norm_ratio": 0.80,
    "maximum_composition_norm_ratio": 1.20,
    "minimum_live_interaction_fraction": 0.20,
}
PRICE = {"forwards": 5, "sequences": 80, "fits": 0, "backwards": 0, "updates": 0}
PREDICTION_REGISTRY = {
    "pred_a_exact_instrument_and_capability": None,
    "pred_b_single_site_writes_live_and_selective": None,
    "pred_c_two_site_additive_composition": None,
    "pred_d_two_site_interaction_live": None,
}
PRED_KEYS = tuple(PREDICTION_REGISTRY)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"rows": ROWS, "rank1": RANK1, "preregistration": PREREG, "fresh_result": FRESH_RESULT}
    for name, expected in binding["files"].items():
        if digest(paths[name]) != expected:
            raise RuntimeError(f"bound {name} changed")
    rows, rank = json.loads(ROWS.read_text()), json.loads(RANK1.read_text())
    if binding["row_manifest_sha256"] != rows["row_manifest_sha256"]:
        raise RuntimeError("row manifest changed")
    if binding["bars"] != BARS or binding["price"] != PRICE or binding["site_positions"] != list(SITE_POSITIONS):
        raise RuntimeError("bound design changed")
    coefficients = {key: rank["coefficients"][key] for key in binding["coefficient_keys"]}
    if coefficients != binding["coefficients"] or rank["rank"] != 1 or rank["rank_sweep"]:
        raise RuntimeError("rank-one artifact changed")
    return binding, rows, rank


def plan():
    binding, rows, rank = load_bound()
    return {
        "schema": "subject_number_two_site_composition_v2_plan",
        "model_loaded": False,
        "gpu_accessed": False,
        "queue_touched": False,
        "rows": rows["row_count"],
        "sites": rows["site_count"],
        "site_positions": list(SITE_POSITIONS),
        "arms": list(ARMS),
        "rank": rank["rank"],
        "coefficients": binding["coefficients"],
        "bars": BARS,
        "price": PRICE,
        "predicates": list(PRED_KEYS),
    }


def metrics(predicted, actual):
    pn, an = float(np.linalg.norm(predicted)), float(np.linalg.norm(actual))
    return {
        "count": len(actual),
        "cosine": float(np.dot(predicted, actual) / (pn * an)) if pn and an else 0.0,
        "relative_l2_error": float(np.linalg.norm(predicted - actual) / an) if an else float("inf"),
        "sign_agreement": float(np.mean(np.sign(predicted) == np.sign(actual))),
        "predicted_to_actual_norm_ratio": pn / an if an else float("inf"),
    }


def composition_passes(report):
    return bool(
        report["cosine"] >= BARS["minimum_composition_cosine"]
        and report["relative_l2_error"] <= BARS["maximum_composition_relative_l2"]
        and report["sign_agreement"] >= BARS["minimum_composition_sign"]
        and BARS["minimum_composition_norm_ratio"] <= report["predicted_to_actual_norm_ratio"] <= BARS["maximum_composition_norm_ratio"]
    )


def main():
    binding, frozen, rank = load_bound()
    if REQUESTED_DRY:
        print(json.dumps(plan(), indent=2, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)
    signal.alarm(600)
    sys.path.insert(0, str(OPS))
    import torch
    import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent
    import circuit_fast_screen_managed_runner as managed

    torch.set_num_threads(2)
    torch_lib, F, facade = tangent.parent.factors._dependencies()
    if torch_lib is not torch:
        raise RuntimeError("torch dependency mismatch")
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    device = next(model.parameters()).device
    rows = frozen["rows"]
    tokens = torch.tensor([row["token_ids"] for row in rows], dtype=torch.long, device=device)
    axis = torch.tensor(rank["axis"], dtype=torch.float32, device=device)
    deltas = []
    for site_index in range(2):
        deltas.append(torch.stack([
            float(rank["coefficients"][f"{row['sites'][site_index]['direction']}.cardinality_4"]) * axis
            for row in rows
        ]))
    counts = {"forwards": 0, "sequences": 0}

    def run(delta1=None, delta2=None):
        counts["forwards"] += 1; counts["sequences"] += len(tokens)
        def attention(event):
            write, first_value = event.block.attn(event.state, event.first_value)
            if event.site == LAYER:
                write = write.clone()
                if delta1 is not None:
                    write[:, SITE_POSITIONS[0]] += delta1.to(write.dtype)
                if delta2 is not None:
                    write[:, SITE_POSITIONS[1]] += delta2.to(write.dtype)
            return write, first_value
        def mlp(event):
            return event.block.mlp(event.state)
        with torch.no_grad():
            return facade.forward_with_dispatch(model, tokens, attention, mlp, require_production=False).float().cpu().numpy()

    started = time.perf_counter()
    logits = {
        "base": run(),
        "zero": run(torch.zeros_like(deltas[0]), torch.zeros_like(deltas[1])),
        "site1": run(deltas[0], None),
        "site2": run(None, deltas[1]),
        "both": run(deltas[0], deltas[1]),
    }
    zero_error = float(np.max(np.abs(logits["zero"] - logits["base"])))

    number_margins = {arm: np.zeros((len(rows), 2), dtype=np.float64) for arm in ARMS}
    control_margins = {arm: np.zeros((len(rows), 2), dtype=np.float64) for arm in ARMS}
    native_correct = np.zeros((len(rows), 2), dtype=bool)
    for row_index, row in enumerate(rows):
        can_id, will_id = row["control_token_ids"]["can"], row["control_token_ids"]["will"]
        for site_index, site in enumerate(row["sites"]):
            position = int(site["position"])
            native_correct[row_index, site_index] = logits["base"][row_index, position, int(site["native_answer_id"])] > logits["base"][row_index, position, int(site["opposite_answer_id"])]
            for arm in ARMS:
                values = logits[arm][row_index, position]
                number_margins[arm][row_index, site_index] = values[int(site["opposite_answer_id"])] - values[int(site["native_answer_id"])]
                control_margins[arm][row_index, site_index] = values[can_id] - values[will_id]

    number_effects = {arm: number_margins[arm] - number_margins["base"] for arm in ("site1", "site2", "both")}
    control_effects = {arm: control_margins[arm] - control_margins["base"] for arm in ("site1", "site2", "both")}
    capability = {}
    for site_index in range(2):
        for number in ("singular",):
            for template in frozen["templates"]:
                indices = [i for i, row in enumerate(rows) if row["sites"][site_index]["native_number"] == number and row["template_id"] == template]
                capability[f"site{site_index + 1}|{number}|{template}"] = float(np.mean(native_correct[indices, site_index]))

    single_reports = {}
    single_pass = True
    for site_index, arm in enumerate(("site1", "site2")):
        own = number_effects[arm][:, site_index]
        for direction in ("singular_to_plural",):
            indices = [i for i, row in enumerate(rows) if row["sites"][site_index]["direction"] == direction]
            values = own[indices]
            control = control_effects[arm][indices, site_index]
            rms = float(np.sqrt(np.mean(values ** 2)))
            report = {
                "count": len(values),
                "effect_rms": rms,
                "positive_fraction": float(np.mean(values > 0)),
                "unrelated_control_fraction": float(np.sqrt(np.mean(control ** 2)) / rms) if rms else float("inf"),
            }
            report["passes"] = bool(
                report["effect_rms"] >= BARS["minimum_single_site_effect_rms"]
                and report["positive_fraction"] >= BARS["minimum_single_site_positive_fraction"]
                and report["unrelated_control_fraction"] <= BARS["maximum_unrelated_control_fraction"]
            )
            single_reports[f"site{site_index + 1}|{direction}"] = report
            single_pass &= report["passes"]

    anticausal = float(np.max(np.abs(number_effects["site2"][:, 0])))
    predicted = number_effects["site1"] + number_effects["site2"]
    actual = number_effects["both"]
    composition = {"overall": metrics(predicted.ravel(), actual.ravel())}
    for site_index in range(2):
        composition[f"site{site_index + 1}"] = metrics(predicted[:, site_index], actual[:, site_index])
    for report in composition.values():
        report["passes"] = composition_passes(report)
    interaction = actual - predicted
    joint_rms = float(np.sqrt(np.mean(actual ** 2)))
    interaction_fraction = float(np.sqrt(np.mean(interaction ** 2)) / joint_rms) if joint_rms else float("inf")
    joint_control_rms = float(np.sqrt(np.mean(control_effects["both"] ** 2)))
    joint_control_fraction = joint_control_rms / joint_rms if joint_rms else float("inf")

    pred_a = bool(zero_error <= BARS["maximum_zero_replay_logit_error"] and all(value >= BARS["minimum_native_accuracy"] for value in capability.values()) and counts == {"forwards": 5, "sequences": 80})
    pred_b = bool(pred_a and single_pass and anticausal <= BARS["maximum_anticausal_absolute_effect"] and joint_control_fraction <= BARS["maximum_unrelated_control_fraction"])
    pred_c = bool(pred_b and all(report["passes"] for report in composition.values()))
    pred_d = bool(pred_b and not pred_c and interaction_fraction >= BARS["minimum_live_interaction_fraction"])
    terminal = "invalid_or_dead_single_site" if not pred_b else ("two_site_additive_composition" if pred_c else ("two_site_interaction_live" if pred_d else "two_site_composition_null"))
    result = {
        "schema": "subject_number_two_site_composition_v2_result",
        "terminal": terminal,
        "predictions": dict(zip(PRED_KEYS, map(bool, (pred_a, pred_b, pred_c, pred_d)))),
        "instrument": {"zero_replay_max_logit_error": zero_error, "native_capability": capability},
        "single_site": single_reports,
        "anticausal_site2_to_site1_max_absolute_effect": anticausal,
        "composition": composition,
        "interaction_over_joint_rms": interaction_fraction,
        "joint_unrelated_control_fraction": joint_control_fraction,
        "records": [
            {
                "row_id": row["row_id"],
                "number_pair": row["number_pair"],
                "template": row["template_id"],
                "site1_effect": number_effects["site1"][i].tolist(),
                "site2_effect": number_effects["site2"][i].tolist(),
                "joint_effect": actual[i].tolist(),
                "interaction": interaction[i].tolist(),
            }
            for i, row in enumerate(rows)
        ],
        "bars": BARS,
        "price": {**PRICE, "observed_forwards": counts["forwards"], "observed_sequences": counts["sequences"]},
        "row_manifest_sha256": frozen["row_manifest_sha256"],
        "binding_sha256": digest(BINDING),
        "runner_sha256": digest(RUNNER),
        "checkpoint_sha256": checkpoint.weights_sha256,
        "quantized": False,
        "claim_boundary": "two additive frozen L11H3 rank-one write installations through native downstream background; upstream selector and all other model state remain external",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "wall_seconds": time.perf_counter() - started,
    }
    managed.atomic_create_json(OUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
