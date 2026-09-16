#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_position_generalization_and_capability pred_b_generated_single_sites_live pred_c_generated_two_site_composition
"""Compose two independently executed response-weighted subject-number generators."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import circuit_fast_screen_managed_runner as managed
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent


RUNNER = Path(__file__).resolve()
ROOT = RUNNER.parents[3]
POLY = ROOT / "basis_aligned/polynomial_causal"
ROWS = POLY / "SUBJECT_NUMBER_TWO_SITE_COMPOSITION_V2_ROWS.json"
ARTIFACT = POLY / "SUBJECT_NUMBER_RESPONSE_WEIGHTED_PROTOTYPE_FROZEN_V1_ARTIFACT.json"
REMOVAL = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/subject_number_response_weighted_removal_v2_result.json"
FIXED_RESULT = POLY / "SUBJECT_NUMBER_TWO_SITE_COMPOSITION_V2_RESULT.json"
PREREG = POLY / "SUBJECT_NUMBER_RESPONSE_WEIGHTED_TWO_SITE_COMPOSITION_V1_PREREGISTRATION.md"
BINDING = POLY / "SUBJECT_NUMBER_RESPONSE_WEIGHTED_TWO_SITE_COMPOSITION_V1_BINDING.json"
OUT = POLY / "SUBJECT_NUMBER_RESPONSE_WEIGHTED_TWO_SITE_COMPOSITION_V1_RESULT.json"
SITE_POSITIONS = (5, 14)
ARMS = ("base", "zero", "site1", "site2", "both")
PRICE = {"physical_model_forwards": 7, "sequences": 112,
         "generator_executions": 32, "fits": 0, "backwards": 0, "updates": 0}
PREDICTION_REGISTRY = {"pred_a_position_generalization_and_capability": None,
                       "pred_b_generated_single_sites_live": None,
                       "pred_c_generated_two_site_composition": None}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(predicted, actual):
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    pn, an = np.linalg.norm(predicted), np.linalg.norm(actual)
    return {"count": int(actual.size), "cosine": float(predicted.ravel() @ actual.ravel() / max(pn * an, 1e-30)),
            "relative_l2_error": float(np.linalg.norm(predicted - actual) / max(an, 1e-30)),
            "sign_agreement": float(np.mean(np.sign(predicted) == np.sign(actual))),
            "predicted_to_actual_norm_ratio": float(pn / max(an, 1e-30))}


def composition_passes(report):
    return report["cosine"] >= .95 and report["relative_l2_error"] <= .25 \
        and report["sign_agreement"] >= .90 \
        and .80 <= report["predicted_to_actual_norm_ratio"] <= 1.20


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"rows": ROWS, "artifact": ARTIFACT, "removal": REMOVAL,
             "fixed_composition": FIXED_RESULT, "preregistration": PREREG}
    if binding["files"] != {key: sha(path) for key, path in paths.items()} \
            or binding["price"] != PRICE or binding["site_positions"] != list(SITE_POSITIONS) \
            or binding["arms"] != list(ARMS):
        raise ValueError("binding changed")
    rows, artifact, removal, fixed = (json.loads(path.read_text()) for path in (ROWS, ARTIFACT, REMOVAL, FIXED_RESULT))
    if rows["row_count"] != 16 or artifact["terminal"] != "response_weighted_prototypes_frozen_opened_only" \
            or removal["terminal"] != "response_weighted_selective_removal_held" \
            or fixed["terminal"] != "two_site_additive_composition":
        raise ValueError("parent status changed")
    return binding, rows, artifact, fixed


def plan():
    binding, rows, artifact, _ = load_bound()
    return {"schema": "subject_number_response_weighted_two_site_composition_v1_plan",
            "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
            "rows": rows["row_count"], "sites": rows["site_count"],
            "site_positions": list(SITE_POSITIONS), "arms": list(ARMS),
            "ports_per_generator": ["native_mlp8_state", "frozen_head_weights", "counterfactual_direction"],
            "prototype_sha256": artifact["prototype_audit"]["singular_to_plural"]["float32_sha256"],
            "price": PRICE, "binding_sha256": sha(BINDING), "bound_files": sorted(binding["files"])}


def head_function_at(model, captured, projection, position, attention, torch, F):
    mlp = model.transformer.h[tangent.parent.MLP_LAYER].mlp
    conditional = tangent.parent.grandparent.parent
    value_module = tangent.parent.grandparent.value_v2

    def function(raw_subject):
        normalized = F.rms_norm(raw_subject, (raw_subject.shape[-1],))
        product = F.linear(normalized, mlp.Left.weight) * F.linear(normalized, mlp.Right.weight)
        output = F.linear(product, mlp.Down.weight) + mlp.Down_bias
        propagated = tangent.parent.polarized_v2._sequentially_propagate(model, output, captured["M8"].dtype)
        slot = captured["M8"].clone(); slot[:, position] = propagated
        high = torch.zeros_like(slot)
        for layer in conditional.LAYERS:
            high = high + (slot if layer == tangent.parent.MLP_LAYER else captured[f"M{layer}"])
        high = high + captured["HR"]
        current = conditional._current_from_high(captured, high, attention, torch, F)
        value = value_module._project_once(current, captured["cached_pre"], projection, F)
        terms = captured["p"].unsqueeze(-1) * captured["u"]
        mask = torch.arange(terms.shape[1], device=terms.device) != position
        return terms[:, mask].sum(1) + captured["p"][:, position].unsqueeze(-1) * value[:, position]
    return function


@np.errstate(all="raise")
def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)
    binding, frozen, artifact, fixed = load_bound()
    torch, F, facade = tangent.parent.factors._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    device = next(model.parameters()).device; rows = frozen["rows"]
    tokens = torch.tensor([row["token_ids"] for row in rows], dtype=torch.long, device=device)
    axis = torch.tensor(artifact["native_axis"], dtype=torch.float32, device=device)
    prototype = torch.tensor(artifact["prototypes"]["singular_to_plural"], dtype=torch.float32, device=device)
    beta = torch.tensor(artifact["interaction_beta"], dtype=torch.float64, device=device)
    writes, alphas, head_replay_errors, closure_errors = [], [], [], []
    counts = {"physical_model_forwards": 0, "sequences": 0, "generator_executions": 0,
              "fits": 0, "backwards": 0, "updates": 0}
    attention = model.transformer.h[tangent.parent.LAYER].attn
    for position in SITE_POSITIONS:
        finals = torch.full((len(rows),), position, dtype=torch.long, device=device)
        with torch.no_grad():
            _, captured, projection, closure, inputs = tangent.parent._decomposed_forward(model, tokens, finals, torch, F, facade)
            counts["physical_model_forwards"] += 1; counts["sequences"] += len(rows)
            function = head_function_at(model, captured, projection, position, attention, torch, F)
            x = inputs["raw_state"][:, position]
            h0, hp = function(x), function(x + prototype)
            z = (h0 @ axis).double(); s = ((hp - h0) @ axis).double()
            alpha = torch.stack([torch.ones_like(z), z, s, z * s], dim=1) @ beta
            writes.append(alpha.to(axis.dtype).unsqueeze(-1) * axis)
            alphas.append(alpha.detach().cpu().numpy())
            head_replay_errors.append(float((h0 - captured["head"]).abs().max()))
            closure_errors.extend([closure["input_state_closure_max_absolute_error"],
                                   closure["input_normalized_closure_max_absolute_error"]])
            counts["generator_executions"] += len(rows)

    def run(delta1=None, delta2=None):
        counts["physical_model_forwards"] += 1; counts["sequences"] += len(rows)
        def attention_dispatch(event):
            write, first_value = event.block.attn(event.state, event.first_value)
            if event.site == tangent.parent.LAYER:
                write = write.clone()
                if delta1 is not None: write[:, SITE_POSITIONS[0]] += delta1.to(write.dtype)
                if delta2 is not None: write[:, SITE_POSITIONS[1]] += delta2.to(write.dtype)
            return write, first_value
        with torch.no_grad():
            return facade.forward_with_dispatch(model, tokens, attention_dispatch,
                lambda event: event.block.mlp(event.state), require_production=False).float().cpu().numpy()

    logits = {"base": run(), "zero": run(torch.zeros_like(writes[0]), torch.zeros_like(writes[1])),
              "site1": run(writes[0], None), "site2": run(None, writes[1]), "both": run(writes[0], writes[1])}
    zero_error = float(np.max(np.abs(logits["zero"] - logits["base"])))
    number_margins = {arm: np.zeros((len(rows), 2)) for arm in ARMS}
    control_margins = {arm: np.zeros((len(rows), 2)) for arm in ARMS}
    native_correct = np.zeros((len(rows), 2), dtype=bool)
    for i, row in enumerate(rows):
        can_id, will_id = row["control_token_ids"]["can"], row["control_token_ids"]["will"]
        for j, site in enumerate(row["sites"]):
            position = site["position"]
            native_correct[i, j] = logits["base"][i, position, site["native_answer_id"]] > logits["base"][i, position, site["opposite_answer_id"]]
            for arm in ARMS:
                values = logits[arm][i, position]
                number_margins[arm][i, j] = values[site["opposite_answer_id"]] - values[site["native_answer_id"]]
                control_margins[arm][i, j] = values[can_id] - values[will_id]
    number_effects = {arm: number_margins[arm] - number_margins["base"] for arm in ("site1", "site2", "both")}
    control_effects = {arm: control_margins[arm] - control_margins["base"] for arm in ("site1", "site2", "both")}
    capability = {}
    for j in range(2):
        for template in frozen["templates"]:
            ids = [i for i, row in enumerate(rows) if row["template_id"] == template]
            capability[f"site{j+1}|{template}"] = float(np.mean(native_correct[ids, j]))
    single_reports, single_pass = {}, True
    for j, arm in enumerate(("site1", "site2")):
        own = number_effects[arm][:, j]; control = control_effects[arm][:, j]
        rms = float(np.sqrt(np.mean(own ** 2)))
        report = {"effect_rms": rms, "positive_fraction": float(np.mean(own > 0)),
                  "unrelated_control_fraction": float(np.sqrt(np.mean(control ** 2)) / max(rms, 1e-30))}
        report["passes"] = report["effect_rms"] >= .01 and report["positive_fraction"] >= .75 and report["unrelated_control_fraction"] <= .75
        single_reports[f"site{j+1}"] = report; single_pass &= report["passes"]
    anticausal = float(np.max(np.abs(number_effects["site2"][:, 0])))
    predicted = number_effects["site1"] + number_effects["site2"]
    actual = number_effects["both"]
    composition = {"overall": metrics(predicted, actual),
                   "site1": metrics(predicted[:, 0], actual[:, 0]),
                   "site2": metrics(predicted[:, 1], actual[:, 1])}
    for report in composition.values(): report["passes"] = composition_passes(report)
    interaction = actual - predicted
    interaction_fraction = float(np.sqrt(np.mean(interaction ** 2)) / max(np.sqrt(np.mean(actual ** 2)), 1e-30))
    instrument = (max(head_replay_errors) <= 5e-5 and max(closure_errors) <= 5e-5 and zero_error <= 1e-5
                  and all(value >= .75 for value in capability.values()) and all(np.isfinite(x).all() for x in alphas)
                  and counts == PRICE)
    pred_b = instrument and single_pass and anticausal <= 1e-5
    pred_c = pred_b and all(report["passes"] for report in composition.values())
    predictions = {"pred_a_position_generalization_and_capability": bool(instrument),
                   "pred_b_generated_single_sites_live": bool(pred_b),
                   "pred_c_generated_two_site_composition": bool(pred_c)}
    terminal = "response_weighted_generator_two_site_composition" if all(predictions.values()) else "invalid" if not instrument else "response_weighted_generator_two_site_null"
    result = {"schema": "subject_number_response_weighted_two_site_composition_v1_result",
              "terminal": terminal, "predictions": predictions,
              "instrument": {"head_replay_max_absolute_error": max(head_replay_errors),
                             "input_closure_max_absolute_error": max(closure_errors),
                             "zero_replay_max_logit_error": zero_error, "native_capability": capability,
                             "counts": counts},
              "generated_coefficients": {f"site{j+1}": {"minimum": float(value.min()), "maximum": float(value.max()),
                                                          "mean": float(value.mean()), "std": float(value.std())}
                                         for j, value in enumerate(alphas)},
              "single_site": single_reports, "anticausal_site2_to_site1_max_absolute_effect": anticausal,
              "composition": composition, "interaction_over_joint_rms": interaction_fraction,
              "records": [{"row_id": row["row_id"], "template": row["template_id"],
                           "site1_alpha": float(alphas[0][i]), "site2_alpha": float(alphas[1][i]),
                           "site1_effect": number_effects["site1"][i].tolist(),
                           "site2_effect": number_effects["site2"][i].tolist(),
                           "joint_effect": actual[i].tolist(), "interaction": interaction[i].tolist()}
                          for i, row in enumerate(rows)],
              "price": PRICE, "row_manifest_sha256": frozen["row_manifest_sha256"],
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "binding_sha256": sha(BINDING), "runner_sha256": sha(RUNNER),
              "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "scope": "Two independent executions of the frozen response-weighted generator at positions 5 and 14, followed by native downstream joint installation; no donor state or coefficient refit."}
    managed.atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("terminal", "predictions", "instrument",
                                                    "generated_coefficients", "single_site",
                                                    "anticausal_site2_to_site1_max_absolute_effect",
                                                    "composition", "interaction_over_joint_rms")}, indent=2))
    assert instrument


if __name__ == "__main__":
    main()
