#!/usr/bin/env python3
"""Contextual endpoint/midpoint JVP readout of the Task14 MLP6--7 unit."""

# BQGATE: EXPERIMENT pred_a_instrument_and_parent_closure pred_b_midpoint_quadratic_readout pred_c_endpoint_local_readout pred_d_material_nonquadratic_transport pred_e_context_changes_readout pred_f_parent_level_lexical_specificity

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import circuit_fast_screen_managed_runner as managed
import run_task14_head11_3_fresh_matched_subject_mlp8_mlp6_7_source_factorial as parent


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_mlp6_7_contextual_midpoint_tangent_readout_v1.json"
PARENT_RESULT = ROOT / "circuits/fast_screens/task14_head11_3_fresh_matched_subject_mlp8_mlp6_7_source_factorial_v1_result.json"
OUT = ROOT / "circuits/fast_screens/task14_mlp6_7_contextual_midpoint_tangent_readout_v1_result.json"
PRIOR_ART_SHA256 = "2a51f5d3acaf07ebcb115eecc0a9636cd81b7dafaf421bd2df28f3a18e1453a8"
PARENT_RESULT_SHA256 = "eff2b9e7ab76b4335733e8bde6708435a18e6b0da14073373e211d9588994efa"
CANDIDATE_ID = "subject_verb.number_agreement.mlp6_7_contextual_midpoint_tangent_readout_v1"
SOURCES = ("opposite", "lexical")
BACKGROUNDS = ("recipient", "donor_context")
METHODS = ("base", "exact", "endpoint", "midpoint")
CONDITIONS = tuple(f"{source}_{background}_{method}" for source in SOURCES
                   for background in BACKGROUNDS for method in METHODS)
BARS = {
    "maximum_numerical_absolute_error": 5e-5,
    "minimum_midpoint_cosine": .95,
    "maximum_midpoint_relative_error": .25,
    "minimum_midpoint_task_recovery": .75,
    "maximum_midpoint_task_recovery": 1.25,
    "minimum_endpoint_cosine": .90,
    "maximum_endpoint_relative_error": .35,
    "minimum_endpoint_task_recovery": .65,
    "maximum_endpoint_task_recovery": 1.35,
    "material_midpoint_relative_error": .35,
    "minimum_material_task_recovery": .60,
    "maximum_material_task_recovery": 1.40,
    "minimum_background_error_gap": .15,
    "maximum_lexical_ratio": .25,
}


class ContextualTangentError(ValueError):
    pass


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_rows():
    return parent.build_rows()


def validate_preflight():
    if _sha256(PRIOR_ART) != PRIOR_ART_SHA256:
        raise ContextualTangentError("prior-art receipt changed")
    if _sha256(PARENT_RESULT) != PARENT_RESULT_SHA256:
        raise ContextualTangentError("parent result changed")
    result = json.loads(PARENT_RESULT.read_text())
    pred = result.get("score", {}).get("predictions", {})
    if result.get("terminal") != "valid_causal_screen" \
            or pred.get("pred_a_instrument_and_parent_closure") is not True \
            or pred.get("pred_g_number_specific") is not False:
        raise ContextualTangentError("parent no longer licenses grouped-unit readout")


def compile_plan():
    validate_preflight()
    return {
        "schema": "task14_mlp6_7_contextual_midpoint_tangent_readout_plan_v1",
        "candidate_id": CANDIDATE_ID,
        "prior_art_sha256": PRIOR_ART_SHA256,
        "parent_result_sha256": PARENT_RESULT_SHA256,
        "row_count": 16,
        "conditions": list(CONDITIONS),
        "background_subsets": {"recipient": ["", "YZ"],
                               "donor_context": ["EAUW", "EAUWYZ"]},
        "bars": dict(BARS),
        "price": {"physical_model_forwards": 4, "example_evaluations": 608,
                  "causal_interventions": 192, "backwards": "JVP only; no optimization",
                  "parameter_updates": 0},
        "closed_claims": ["semantic_uniqueness", "rank", "compression",
                          "activation_reconstruction", "new_independent_text",
                          "necessity_outside_fixed_L11H3_interface"],
    }


def _role_slice(values, start, stop):
    return {key: value[start:stop] for key, value in values.items()}


def _directional_jvps(function, base, source, torch):
    delta = source - base
    midpoint = base + .5 * delta
    primal, endpoint = torch.autograd.functional.jvp(
        function, base, delta, create_graph=False, strict=True)
    _, centered = torch.autograd.functional.jvp(
        function, midpoint, delta, create_graph=False, strict=True)
    return primal.detach(), function(source).detach(), endpoint.detach(), centered.detach()


def _raw_pair(recipient, source, background, F):
    if background == "recipient":
        base = recipient["raw_state"]
        _, changed, _ = parent._hybrid_input(recipient, source, "YZ", F)
    elif background == "donor_context":
        _, base, _ = parent._hybrid_input(recipient, source, "EAUW", F)
        _, changed, _ = parent._hybrid_input(recipient, source, "EAUWYZ", F)
    else:
        raise ContextualTangentError("unknown background")
    return base[:, parent.SUBJECT_POSITION], changed[:, parent.SUBJECT_POSITION]


def _head_function(model, recipient, opposite, attention, projection, torch, F):
    mlp = model.transformer.h[parent.MLP_LAYER].mlp

    def function(raw_subject):
        normalized = F.rms_norm(raw_subject, (raw_subject.shape[-1],))
        product = F.linear(normalized, mlp.Left.weight) \
            * F.linear(normalized, mlp.Right.weight)
        output = F.linear(product, mlp.Down.weight) + mlp.Down_bias
        propagated = parent.polarized_v2._sequentially_propagate(
            model, output, recipient["M8"].dtype)
        slot = recipient["M8"].clone()
        slot[:, parent.SUBJECT_POSITION] = propagated
        return parent.grandparent._head_from_slot(
            recipient, opposite, slot, attention, projection, torch, F)

    return function


def _compile_patch(tokens, heads, rows, torch):
    indices, replacements, masks, specs = [], [], [], []
    for row_index, row in enumerate(rows):
        for condition in CONDITIONS:
            indices.append(row_index)
            replacements.append(heads[condition][row_index])
            masks.append(condition in {"opposite_recipient_base",
                                       "lexical_recipient_base"})
            specs.append((row_index, condition,
                          f"{row['direction_id']}__{row['template_id']}"))
    index = torch.tensor(indices, dtype=torch.long, device=tokens.device)
    return {"tokens": tokens[index],
            "finals": torch.full_like(index, parent.SUBJECT_POSITION),
            "replacement_heads": torch.stack(replacements),
            "native_reinstall_mask": torch.tensor(masks, dtype=torch.bool,
                                                    device=tokens.device),
            "specs": specs}


def evaluate(model, torch, F, facade):
    rows = build_rows(); n = len(rows); device = next(model.parameters()).device
    tokens, finals = parent.downstream.depth.parent.v1._role_batch(rows, torch, device)
    native_roles = parent.factors._native_logits(model, tokens, torch, F)
    replay, captured, projection, role_closure, inputs = parent._decomposed_forward(
        model, tokens, finals, torch, F, facade)
    roles = {"recipient": _role_slice(captured, 0, n),
             "opposite": _role_slice(captured, n, 2*n),
             "lexical": _role_slice(captured, 2*n, 3*n)}
    input_roles = {"recipient": _role_slice(inputs, 0, n),
                   "opposite": _role_slice(inputs, n, 2*n),
                   "lexical": _role_slice(inputs, 2*n, 3*n)}
    attention = model.transformer.h[parent.LAYER].attn
    function = _head_function(model, roles["recipient"], roles["opposite"],
                              attention, projection, torch, F)
    heads, geometry = {}, {}
    for source in SOURCES:
        geometry[source] = {}
        for background in BACKGROUNDS:
            x0, x1 = _raw_pair(input_roles["recipient"], input_roles[source],
                               background, F)
            base, exact, endpoint_delta, midpoint_delta = _directional_jvps(
                function, x0, x1, torch)
            heads[f"{source}_{background}_base"] = base
            heads[f"{source}_{background}_exact"] = exact
            heads[f"{source}_{background}_endpoint"] = base + endpoint_delta
            heads[f"{source}_{background}_midpoint"] = base + midpoint_delta
            geometry[source][background] = {
                "exact_delta": (exact - base).cpu(),
                "endpoint_delta": endpoint_delta.cpu(),
                "midpoint_delta": midpoint_delta.cpu(),
            }
    patch = _compile_patch(tokens[:n], heads, rows, torch)
    native_patch = parent.factors._native_logits(model, patch["tokens"], torch, F)
    patched, _, _, patch_closure = parent.downstream._decomposed_forward(
        model, patch["tokens"], patch["finals"], torch, F, facade,
        replacement_heads=patch["replacement_heads"],
        native_reinstall_mask=patch["native_reinstall_mask"])
    mask = patch["native_reinstall_mask"]
    exactness = {
        "native_role_replay_max_absolute_logit_error": float((native_roles-replay).abs().max()),
        "role_state_closure_max_absolute_error": role_closure["input_state_closure_max_absolute_error"],
        "role_normalized_closure_max_absolute_error": role_closure["input_normalized_closure_max_absolute_error"],
        "downstream_state_closure_max_absolute_error": patch_closure["state_sum_max_absolute_error"],
        "downstream_normalized_closure_max_absolute_error": patch_closure["normalized_state_max_absolute_error"],
        "recipient_noop_full_logit_max_absolute_error": float((patched[mask]-native_patch[mask]).abs().max()),
    }
    evidence = []
    for out_index, (row_index, condition, cell_id) in enumerate(patch["specs"]):
        # The explicit parser avoids ambiguity in the donor_context token.
        source = "opposite" if condition.startswith("opposite_") else "lexical"
        suffix = condition[len(source)+1:]
        background = "donor_context" if suffix.startswith("donor_context_") else "recipient"
        method = suffix[len(background)+1:]
        metrics = parent.grandparent._both_metrics(
            patched[out_index, parent.SUBJECT_POSITION], rows[row_index], torch)
        native_metrics = parent.grandparent._both_metrics(
            native_patch[out_index, parent.SUBJECT_POSITION], rows[row_index], torch)
        evidence.append({"row_id": rows[row_index]["row_id"], "cell_id": cell_id,
                         "source": source, "background": background, "method": method,
                         "target_margin": metrics[f"{source}_target_margin"],
                         "target_CE": metrics[f"{source}_target_CE"],
                         "target_margin_improvement":
                             metrics[f"{source}_target_margin"]
                             - native_metrics[f"{source}_target_margin"],
                         "target_CE_improvement":
                             native_metrics[f"{source}_target_CE"]
                             - metrics[f"{source}_target_CE"]})
    return evidence, exactness, geometry


def _vector_stats(exact, predicted, torch):
    exact = exact.double().reshape(-1); predicted = predicted.double().reshape(-1)
    return {"cosine": float(torch.dot(exact, predicted) /
                            (exact.norm()*predicted.norm()).clamp_min(1e-30)),
            "relative_error": float((predicted-exact).norm()/exact.norm().clamp_min(1e-30))}


def score(evidence, exactness, geometry, torch, bars=BARS):
    expected = {(row["row_id"], source, background, method)
                for row in build_rows() for source in SOURCES
                for background in BACKGROUNDS for method in METHODS}
    observed = {(item["row_id"], item["source"], item["background"], item["method"])
                for item in evidence}
    if len(evidence) != len(expected) or observed != expected:
        raise ContextualTangentError("incomplete or duplicate evidence lattice")
    numeric_keys = ("target_margin", "target_CE", "target_margin_improvement",
                    "target_CE_improvement")
    if not all(math.isfinite(float(item[key])) for item in evidence for key in numeric_keys):
        raise ContextualTangentError("non-finite task evidence")
    if not all(bool(torch.isfinite(tensor).all())
               for by_background in geometry.values()
               for tensors in by_background.values() for tensor in tensors.values()):
        raise ContextualTangentError("non-finite JVP geometry")
    grouped = defaultdict(lambda: defaultdict(dict))
    for item in evidence:
        key = (item["source"], item["background"])
        grouped[item["cell_id"]][key].setdefault(item["method"], []).append(item)
    cells = {}
    parent_result = json.loads(PARENT_RESULT.read_text())
    parent_by = {(x["row_id"], x["condition"]): x for x in parent_result["evidence"]}
    parent_margin_error = 0.0
    parent_ce_error = 0.0
    for cell_id, pairs in grouped.items():
        cells[cell_id] = {}
        direction_rows = [row for row in build_rows()
                          if f"{row['direction_id']}__{row['template_id']}" == cell_id]
        row_ids = {row["row_id"] for row in direction_rows}
        for (source, background), methods in pairs.items():
            entry = {}
            for method, items in methods.items():
                entry[method] = {metric: statistics.fmean(x[metric] for x in items)
                                 for metric in ("target_margin", "target_CE")}
            for method in ("exact", "endpoint", "midpoint"):
                exact_margin = entry["exact"]["target_margin"]-entry["base"]["target_margin"]
                method_margin = entry[method]["target_margin"]-entry["base"]["target_margin"]
                entry[method]["margin_effect"] = method_margin
                entry[method]["margin_recovery"] = method_margin/max(abs(exact_margin),1e-12) \
                    * (1 if exact_margin >= 0 else -1)
            indices = [i for i,row in enumerate(build_rows()) if row["row_id"] in row_ids]
            geo = geometry[source][background]
            for method in ("endpoint", "midpoint"):
                entry[method].update(_vector_stats(
                    geo["exact_delta"][indices], geo[f"{method}_delta"][indices], torch))
            base_subset, exact_subset = (("", "YZ") if background == "recipient"
                                         else ("EAUW", "EAUWYZ"))
            for method, subset in (("base", base_subset), ("exact", exact_subset)):
                condition = "recipient" if not subset else f"{source}_{subset}_full"
                for item in methods[method]:
                    prior = parent_by[(item["row_id"], condition)]
                    parent_margin_error = max(parent_margin_error, abs(
                        item["target_margin_improvement"]
                        - prior[f"{source}_target_margin_improvement"]))
                    parent_ce_error = max(parent_ce_error, abs(
                        item["target_CE_improvement"]
                        - prior[f"{source}_target_CE_improvement"]))
            cells[cell_id][f"{source}:{background}"] = entry
    exactness["parent_finite_margin_max_absolute_error"] = parent_margin_error
    exactness["parent_finite_CE_max_absolute_error"] = parent_ce_error
    instrument = all(value <= bars["maximum_numerical_absolute_error"]
                     for value in exactness.values())
    opposite = [entry for cell in cells.values() for key,entry in cell.items()
                if key.startswith("opposite:")]
    midpoint = instrument and all(
        e["midpoint"]["cosine"] >= bars["minimum_midpoint_cosine"]
        and e["midpoint"]["relative_error"] <= bars["maximum_midpoint_relative_error"]
        and bars["minimum_midpoint_task_recovery"] <= e["midpoint"]["margin_recovery"]
            <= bars["maximum_midpoint_task_recovery"] for e in opposite)
    endpoint = instrument and all(
        e["endpoint"]["cosine"] >= bars["minimum_endpoint_cosine"]
        and e["endpoint"]["relative_error"] <= bars["maximum_endpoint_relative_error"]
        and bars["minimum_endpoint_task_recovery"] <= e["endpoint"]["margin_recovery"]
            <= bars["maximum_endpoint_task_recovery"] for e in opposite)
    nonquadratic = instrument and not midpoint and any(
        e["midpoint"]["relative_error"] > bars["material_midpoint_relative_error"]
        or not (bars["minimum_material_task_recovery"] <= e["midpoint"]["margin_recovery"]
                <= bars["maximum_material_task_recovery"]) for e in opposite)
    gaps = []
    for cell in cells.values():
        gaps.append(abs(cell["opposite:recipient"]["midpoint"]["relative_error"]
                        - cell["opposite:donor_context"]["midpoint"]["relative_error"]))
    context = instrument and max(gaps) >= bars["minimum_background_error_gap"]
    lexical_ratios = []
    for cell in cells.values():
        for background in BACKGROUNDS:
            scale = abs(cell[f"opposite:{background}"]["exact"]["margin_effect"])
            for method in ("exact", "midpoint"):
                lexical_ratios.append(abs(cell[f"lexical:{background}"][method]["margin_effect"])
                                      / max(scale, 1e-12))
    lexical = instrument and max(lexical_ratios) <= bars["maximum_lexical_ratio"]
    return {**exactness, "cells": cells, "maximum_lexical_ratio": max(lexical_ratios),
            "maximum_background_midpoint_error_gap": max(gaps), "predictions": {
        "pred_a_instrument_and_parent_closure": bool(instrument),
        "pred_b_midpoint_quadratic_readout": bool(midpoint),
        "pred_c_endpoint_local_readout": bool(endpoint),
        "pred_d_material_nonquadratic_transport": bool(nonquadratic),
        "pred_e_context_changes_readout": bool(context),
        "pred_f_parent_level_lexical_specificity": bool(lexical)}}


def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args(argv); plan=compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":
        print(json.dumps(plan,sort_keys=True)); return
    if OUT.exists(): raise ContextualTangentError(f"refusing to overwrite {OUT}")
    torch,F,facade=parent.factors._dependencies()
    model,checkpoint=facade.load_bilin18(device="cuda",dtype=torch.float32,verify_weights_sha256=True)
    evidence,exactness,geometry=evaluate(model,torch,F,facade)
    scored=score(evidence,exactness,geometry,torch)
    terminal="valid_causal_screen" if scored["predictions"]["pred_a_instrument_and_parent_closure"] else "invalid"
    result={"schema":"task14_mlp6_7_contextual_midpoint_tangent_readout_result_v1",
            "candidate_id":CANDIDATE_ID,"terminal":terminal,"plan":plan,
            "checkpoint_weights_sha256":checkpoint.weights_sha256,"score":scored,
            "evidence":evidence,"evaluated_splits":["LICENSED_HOLDOUT_REUSED_TEXT"],
            "forbidden_splits_opened":[]}
    payload=managed.atomic_create_json(OUT,result)
    print(json.dumps({"terminal":terminal,"predictions":scored["predictions"],
                      "result_sha256":hashlib.sha256(payload).hexdigest()},sort_keys=True))


if __name__ == "__main__": main()
