#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_factorial_instrument pred_b_sparse_port_transfer pred_c_matched_subset_null
"""Discover a sparse topology-aware upstream port subset for subject number."""
from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import circuit_fast_screen_managed_runner as managed
import run_subject_number_topology_response_prototype_discovery_v2 as parent


RUNNER = Path(__file__).resolve()
ROOT = RUNNER.parents[3]
POLY = ROOT / "basis_aligned/polynomial_causal"
PREREG = POLY / "SUBJECT_NUMBER_TOPOLOGY_PORT_FACTORIAL_DISCOVERY_V1_PREREGISTRATION.md"
CORRECTION = POLY / "SUBJECT_NUMBER_TOPOLOGY_PORT_FACTORIAL_DISCOVERY_V1_IMPLEMENTATION_CORRECTION.md"
BINDING = POLY / "SUBJECT_NUMBER_TOPOLOGY_PORT_FACTORIAL_DISCOVERY_V1_BINDING.json"
PARENT_RESULT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/subject_number_topology_response_prototype_discovery_v2_result.json"
OUT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/subject_number_topology_port_factorial_discovery_v1_result.json"
FAMILIES = "EAUWYZ"
SUBSETS = tuple("".join(FAMILIES[bit] for bit in range(len(FAMILIES)) if mask & (1 << bit))
                for mask in range(1 << len(FAMILIES)))
PRICE = {"physical_model_forwards": 2, "sequences": 64,
         "offline_head_evaluations": 2048, "fits": 0,
         "backwards": 0, "updates": 0}
PREDICTION_REGISTRY = {"pred_a_exact_factorial_instrument": None,
                       "pred_b_sparse_port_transfer": None,
                       "pred_c_matched_subset_null": None}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    an, pn = np.linalg.norm(actual), np.linalg.norm(predicted)
    return {"count": int(actual.size),
            "cosine": float(actual @ predicted / max(an * pn, 1e-30)),
            "relative_l2_error": float(np.linalg.norm(actual - predicted) / max(an, 1e-30)),
            "sign_agreement": float(np.mean(np.sign(actual) == np.sign(predicted))),
            "rms_actual": float(np.sqrt(np.mean(actual ** 2))),
            "rms_predicted": float(np.sqrt(np.mean(predicted ** 2)))}


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"rows": parent.base.ROWS, "artifact": parent.base.ARTIFACT,
             "fixed_binding": parent.base.FIXED_BINDING, "parent_result": PARENT_RESULT,
             "preregistration": PREREG, "correction": CORRECTION}
    subset_sha256 = hashlib.sha256("|".join(SUBSETS).encode()).hexdigest()
    if binding["files"] != {key: sha(path) for key, path in paths.items()} \
            or binding["families"] != FAMILIES or binding["subsets_sha256"] != subset_sha256 \
            or binding["site_positions"] != list(parent.base.SITE_POSITIONS) \
            or binding["price"] != PRICE:
        raise ValueError("binding changed")
    rows = json.loads(parent.base.ROWS.read_text())
    artifact = json.loads(parent.base.ARTIFACT.read_text())
    fixed = json.loads(parent.base.FIXED_BINDING.read_text())
    prior = json.loads(PARENT_RESULT.read_text())
    if prior["terminal"] != "topology_response_prototype_null" \
            or rows["row_count"] != 16 \
            or artifact["terminal"] != "response_weighted_prototypes_frozen_opened_only":
        raise ValueError("parent status changed")
    return binding, rows, artifact, fixed


def plan():
    binding, rows, _, fixed = load_bound()
    return {"schema": "subject_number_topology_port_factorial_discovery_v1_plan",
            "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
            "opened_authority": True, "rows": rows["row_count"],
            "templates": rows["templates"], "families": FAMILIES,
            "subset_count": len(SUBSETS), "site_positions": list(parent.base.SITE_POSITIONS),
            "fixed_coefficient": fixed["coefficients"]["singular_to_plural.cardinality_4"],
            "selection": "training qualifiers error<=.50 sign>=.75; min cardinality,error,mask",
            "price": PRICE, "binding_sha256": sha(BINDING),
            "bound_files": sorted(binding["files"])}


def mobius(values):
    result = np.asarray(values, dtype=np.float64).copy()
    for bit in range(len(FAMILIES)):
        for mask in range(len(SUBSETS)):
            if mask & (1 << bit):
                result[mask] -= result[mask ^ (1 << bit)]
    return result


@np.errstate(all="raise")
def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)
    binding, frozen, artifact, fixed = load_bound()
    torch, F, facade = parent.base.prior_runner.tangent.parent.factors._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                             verify_weights_sha256=True)
    device = next(model.parameters()).device
    rows = frozen["rows"]; n = len(rows)
    native_tokens = torch.tensor([row["token_ids"] for row in rows], dtype=torch.long, device=device)
    plurals = parent.base.plural_token_ids(rows)
    axis = torch.tensor(artifact["native_axis"], dtype=torch.float32, device=device)
    beta = torch.tensor(artifact["interaction_beta"], dtype=torch.float64, device=device)
    fixed_alpha = float(fixed["coefficients"]["singular_to_plural.cardinality_4"])
    attention = model.transformer.h[parent.base.prior_runner.tangent.parent.LAYER].attn
    templates = np.asarray([row["template_id"] for row in rows])
    all_alpha = np.zeros((2, len(SUBSETS), n), dtype=np.float64)
    all_s = np.zeros_like(all_alpha)
    head_errors = []; hybrid_errors = []; closure_errors = []
    sequence_count = 0; offline_count = 0

    for site_index, position in enumerate(parent.base.SITE_POSITIONS):
        opposite_tokens = native_tokens.clone()
        for row_index, row in enumerate(rows):
            opposite_tokens[row_index, position] = plurals[row["sites"][site_index]["subject"]]
        tokens = torch.cat([native_tokens, opposite_tokens])
        finals = torch.full((2 * n,), position, dtype=torch.long, device=device)
        with torch.no_grad():
            _, captured, projection, closure, inputs = parent.base.prior_runner.tangent.parent._decomposed_forward(
                model, tokens, finals, torch, F, facade)
        sequence_count += len(tokens)
        native_captured = parent.base.slice_values(captured, slice(0, n))
        native_inputs = parent.base.slice_values(inputs, slice(0, n))
        opposite_inputs = parent.base.slice_values(inputs, slice(n, 2 * n))
        function = parent.base.prior_runner.head_function_at(
            model, native_captured, projection, position, attention, torch, F)
        x = native_inputs["raw_state"][:, position]
        with torch.no_grad():
            h0 = function(x)
        head_errors.append(float((h0 - native_captured["head"]).abs().max()))
        z = (h0 @ axis).double()
        for mask, subset in enumerate(SUBSETS):
            if subset:
                _, raw, hybrid_error = parent.base.prior_runner.tangent.parent._hybrid_input(
                    native_inputs, opposite_inputs, subset, F)
            else:
                raw, hybrid_error = native_inputs["raw_state"], 0.0
            hybrid_errors.append(float(hybrid_error))
            with torch.no_grad():
                head = function(raw[:, position])
            s = ((head - h0) @ axis).double()
            alpha = torch.stack([torch.ones_like(z), z, s, z * s], dim=1) @ beta
            all_s[site_index, mask] = s.cpu().numpy()
            all_alpha[site_index, mask] = alpha.cpu().numpy()
            offline_count += n
        closure_errors.extend([closure["input_state_closure_max_absolute_error"],
                               closure["input_normalized_closure_max_absolute_error"]])

    held_actual = []; held_predicted = []; fold_reports = {}
    null_predictions = []
    selections = []
    for site_index in range(2):
        for held_out in sorted(set(templates)):
            train = np.flatnonzero(templates != held_out)
            test = np.flatnonzero(templates == held_out)
            target_train = np.full(len(train), fixed_alpha)
            scored = []
            for mask, subset in enumerate(SUBSETS):
                report = stats(target_train, all_alpha[site_index, mask, train])
                qualifies = report["relative_l2_error"] <= .50 and report["sign_agreement"] >= .75
                scored.append((not qualifies, len(subset), report["relative_l2_error"], mask,
                               subset, qualifies, report))
            selected = min(scored)
            _, cardinality, _, mask, subset, qualifies, train_report = selected
            prediction = all_alpha[site_index, mask, test]
            target_test = np.full(len(test), fixed_alpha)
            held_actual.extend(target_test); held_predicted.extend(prediction)
            alternatives = [m for m, value in enumerate(SUBSETS)
                            if len(value) == cardinality and m != mask]
            alternative_metrics = [stats(target_test, all_alpha[site_index, m, test])
                                   for m in alternatives]
            null_predictions.append(alternative_metrics)
            test_report = stats(target_test, prediction)
            key = f"site{site_index + 1}|{held_out}"
            fold_reports[key] = {"selected_subset": subset, "selected_mask": mask,
                                 "cardinality": cardinality, "training_qualified": qualifies,
                                 "training_metrics": train_report, "held_out_metrics": test_report,
                                 "matched_alternative_count": len(alternatives),
                                 "matched_alternative_median_error": float(np.median(
                                     [value["relative_l2_error"] for value in alternative_metrics]))
                                     if alternative_metrics else None}
            selections.append((subset, cardinality, qualifies, test_report,
                               fold_reports[key]["matched_alternative_median_error"]))

    selected_metrics = stats(held_actual, held_predicted)
    yz_mask = SUBSETS.index("YZ"); full_mask = SUBSETS.index(FAMILIES)
    target_all = np.full(2 * n, fixed_alpha)
    baselines = {"YZ": stats(target_all, np.concatenate([all_alpha[0, yz_mask], all_alpha[1, yz_mask]])),
                 "EAUWYZ": stats(target_all, np.concatenate([all_alpha[0, full_mask], all_alpha[1, full_mask]]))}
    matched_fold_medians = [value for *_, value in selections if value is not None]
    matched_median_error = float(np.median(matched_fold_medians)) \
        if len(matched_fold_medians) == 4 else None
    null_advantage = matched_median_error - selected_metrics["relative_l2_error"] \
        if matched_median_error is not None else None

    coefficients = np.concatenate([all_alpha[0], all_alpha[1]], axis=1)
    allocated = mobius(coefficients)
    allocation_rms = np.sqrt(np.mean(allocated ** 2, axis=1))
    order = np.argsort(-allocation_rms)
    allocation = [{"subset": SUBSETS[index], "mask": int(index),
                   "order": len(SUBSETS[index]), "coefficient_rms": float(allocation_rms[index]),
                   "fraction_of_total_mobius_rms": float(allocation_rms[index]
                       / max(np.sqrt(np.sum(allocation_rms ** 2)), 1e-30))}
                  for index in order]

    instrument = (sequence_count == PRICE["sequences"]
                  and offline_count == PRICE["offline_head_evaluations"]
                  and max(head_errors) <= 5e-5 and max(hybrid_errors) <= 5e-5
                  and max(closure_errors) <= 5e-5 and np.isfinite(all_alpha).all())
    sparse_transfer = (all(value[2] for value in selections)
                       and all(value[1] <= 3 for value in selections)
                       and selected_metrics["relative_l2_error"] <= .50
                       and selected_metrics["sign_agreement"] >= .90)
    null_pass = null_advantage is not None and null_advantage >= .10
    predictions = {"pred_a_exact_factorial_instrument": bool(instrument),
                   "pred_b_sparse_port_transfer": bool(instrument and sparse_transfer),
                   "pred_c_matched_subset_null": bool(instrument and sparse_transfer and null_pass)}
    terminal = "invalid" if not instrument else "topology_sparse_port_factorial_held" \
        if all(predictions.values()) else "topology_sparse_port_factorial_null"
    result = {"schema": "subject_number_topology_port_factorial_discovery_v1_result",
              "terminal": terminal, "predictions": predictions,
              "selected_program_metrics": selected_metrics,
              "selected_folds": fold_reports, "baselines": baselines,
              "matched_cardinality_median_error": matched_median_error,
              "matched_cardinality_advantage": null_advantage,
              "mobius_allocation": allocation,
              "instrument": {"native_head_replay_max_absolute_error": max(head_errors),
                             "hybrid_input_closure_max_absolute_error": max(hybrid_errors),
                             "input_closure_max_absolute_error": max(closure_errors),
                             "counts": {**PRICE, "observed_sequences": sequence_count,
                                        "observed_offline_head_evaluations": offline_count}},
              "outcome_access": {"behavioral_effects": False, "answer_logits": False,
                                 "downstream_causal_outcomes": False, "new_text": False,
                                 "opened_typed_input_ports": True},
              "price": PRICE, "binding_sha256": sha(BINDING), "runner_sha256": sha(RUNNER),
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "scope": "Opened two-clause six-port MLP8-input factorial with reciprocal leave-template-out sparse subset selection and Boolean-cube coefficient allocation; no behavioral or fresh-OOD claim."}
    managed.atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("terminal", "predictions",
                                                    "selected_program_metrics", "selected_folds",
                                                    "baselines", "matched_cardinality_median_error",
                                                    "matched_cardinality_advantage", "mobius_allocation",
                                                    "instrument")}, indent=2))
    assert instrument


if __name__ == "__main__":
    main()
