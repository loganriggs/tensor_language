#!/usr/bin/env python3
"""Off-grid manipulation test of the Task14 grouped MLP6--7 quadratic law."""

# BQGATE: EXPERIMENT pred_a_instrument_closure pred_b_quadratic_head_prediction pred_c_quadratic_task_manipulation pred_d_extrapolation_stable pred_e_lexical_effect_bounded

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
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_mlp6_7_quadratic_gain_manipulation_v1.json"
PARENT_RESULT = ROOT / "circuits/fast_screens/task14_mlp6_7_contextual_midpoint_tangent_readout_v1_result.json"
OUT = ROOT / "circuits/fast_screens/task14_mlp6_7_quadratic_gain_manipulation_v1_result.json"
PRIOR_ART_SHA256 = "5a96aa03a9b412c63a437376b6d9bf055cb80196710f98c5a1781a25c67b518b"
PARENT_RESULT_SHA256 = "48c72ea08c2573d520e639bbd34805ce6b60f4ec10d420fbbebed8e6112a65aa"
CANDIDATE_ID = "subject_verb.number_agreement.mlp6_7_quadratic_gain_manipulation_v1"
GAINS = (-0.5, 0.5, 1.5)
METHODS = ("exact", "predicted")
BARS = {
    "maximum_numerical_absolute_error": 5e-5,
    "minimum_head_cosine": .98,
    "maximum_head_relative_error": .10,
    "minimum_task_recovery": .80,
    "maximum_task_recovery": 1.20,
    "maximum_lexical_ratio": .25,
}


class QuadraticGainError(ValueError):
    pass


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _gain_id(gain):
    return ("m" if gain < 0 else "p") + str(abs(gain)).replace(".", "p")


def validate_preflight():
    if _sha256(PRIOR_ART) != PRIOR_ART_SHA256:
        raise QuadraticGainError("prior-art receipt changed")
    if _sha256(PARENT_RESULT) != PARENT_RESULT_SHA256:
        raise QuadraticGainError("parent result changed")
    result = json.loads(PARENT_RESULT.read_text())
    predictions = result.get("score", {}).get("predictions", {})
    if result.get("terminal") != "valid_causal_screen" \
            or predictions.get("pred_a_instrument_and_parent_closure") is not True \
            or predictions.get("pred_b_midpoint_quadratic_readout") is not True:
        raise QuadraticGainError("parent no longer licenses quadratic manipulation")


def compile_plan():
    validate_preflight()
    return {
        "schema": "task14_mlp6_7_quadratic_gain_manipulation_plan_v1",
        "candidate_id": CANDIDATE_ID,
        "prior_art_sha256": PRIOR_ART_SHA256,
        "parent_result_sha256": PARENT_RESULT_SHA256,
        "row_count": 16,
        "sources": list(tangent.SOURCES),
        "backgrounds": list(tangent.BACKGROUNDS),
        "gains": list(GAINS),
        "methods": list(METHODS),
        "bars": dict(BARS),
        "price": {"physical_model_forwards": 4, "example_evaluations": 816,
                  "causal_interventions": 384,
                  "backwards": "two JVPs per source/background; no optimization",
                  "parameter_updates": 0},
        "closed_claims": ["semantic_uniqueness", "rank", "compression",
                          "activation_reconstruction", "new_independent_text",
                          "necessity_outside_fixed_L11H3_interface"],
    }


def _compile_patch(tokens, heads, rows, torch):
    indices, replacements, specs = [], [], []
    for row_index, row in enumerate(rows):
        for source in tangent.SOURCES:
            for background in tangent.BACKGROUNDS:
                for gain in GAINS:
                    for method in METHODS:
                        key = (source, background, gain, method)
                        indices.append(row_index)
                        replacements.append(heads[key][row_index])
                        specs.append((row_index, source, background, gain, method,
                                      f"{row['direction_id']}__{row['template_id']}"))
    index = torch.tensor(indices, dtype=torch.long, device=tokens.device)
    return {"tokens": tokens[index],
            "finals": torch.full_like(index, tangent.parent.SUBJECT_POSITION),
            "replacement_heads": torch.stack(replacements), "specs": specs}


def evaluate(model, torch, F, facade):
    parent = tangent.parent
    rows = tangent.build_rows(); n = len(rows); device = next(model.parameters()).device
    tokens, finals = parent.downstream.depth.parent.v1._role_batch(rows, torch, device)
    native_roles = parent.factors._native_logits(model, tokens, torch, F)
    replay, captured, projection, role_closure, inputs = parent._decomposed_forward(
        model, tokens, finals, torch, F, facade)
    roles = {"recipient": tangent._role_slice(captured, 0, n),
             "opposite": tangent._role_slice(captured, n, 2*n)}
    input_roles = {"recipient": tangent._role_slice(inputs, 0, n),
                   "opposite": tangent._role_slice(inputs, n, 2*n),
                   "lexical": tangent._role_slice(inputs, 2*n, 3*n)}
    function = tangent._head_function(
        model, roles["recipient"], roles["opposite"],
        model.transformer.h[parent.LAYER].attn, projection, torch, F)
    heads, geometry = {}, {}
    for source in tangent.SOURCES:
        for background in tangent.BACKGROUNDS:
            x0, x1 = tangent._raw_pair(input_roles["recipient"], input_roles[source],
                                       background, F)
            base, _, endpoint_delta, midpoint_delta = tangent._directional_jvps(
                function, x0, x1, torch)
            delta = x1 - x0
            curvature = midpoint_delta - endpoint_delta
            for gain in GAINS:
                exact = function(x0 + gain * delta).detach()
                predicted = base + gain * endpoint_delta + gain * gain * curvature
                heads[(source, background, gain, "exact")] = exact
                heads[(source, background, gain, "predicted")] = predicted
                geometry[(source, background, gain)] = {
                    "exact_delta": (exact - base).cpu(),
                    "predicted_delta": (predicted - base).cpu(),
                }
    patch = _compile_patch(tokens[:n], heads, rows, torch)
    native_patch = parent.factors._native_logits(model, patch["tokens"], torch, F)
    patched, _, _, patch_closure = parent.downstream._decomposed_forward(
        model, patch["tokens"], patch["finals"], torch, F, facade,
        replacement_heads=patch["replacement_heads"])
    exactness = {
        "native_role_replay_max_absolute_logit_error": float((native_roles-replay).abs().max()),
        "role_state_closure_max_absolute_error": role_closure["input_state_closure_max_absolute_error"],
        "role_normalized_closure_max_absolute_error": role_closure["input_normalized_closure_max_absolute_error"],
        "downstream_state_closure_max_absolute_error": patch_closure["state_sum_max_absolute_error"],
        "downstream_normalized_closure_max_absolute_error": patch_closure["normalized_state_max_absolute_error"],
        "bound_parent_recipient_noop_full_logit_max_absolute_error":
            json.loads(PARENT_RESULT.read_text())["score"]["recipient_noop_full_logit_max_absolute_error"],
    }
    evidence = []
    for out_index, (row_index, source, background, gain, method, cell_id) in enumerate(patch["specs"]):
        metrics = parent.grandparent._both_metrics(
            patched[out_index, parent.SUBJECT_POSITION], rows[row_index], torch)
        native = parent.grandparent._both_metrics(
            native_patch[out_index, parent.SUBJECT_POSITION], rows[row_index], torch)
        evidence.append({"row_id": rows[row_index]["row_id"], "cell_id": cell_id,
                         "source": source, "background": background, "gain": gain,
                         "method": method,
                         "target_margin": metrics[f"{source}_target_margin"],
                         "target_CE": metrics[f"{source}_target_CE"],
                         "native_target_margin": native[f"{source}_target_margin"],
                         "native_target_CE": native[f"{source}_target_CE"]})
    return evidence, exactness, geometry


def _vector_stats(exact, predicted, torch):
    exact = exact.double().reshape(-1); predicted = predicted.double().reshape(-1)
    return {"cosine": float(torch.dot(exact, predicted) /
                            (exact.norm()*predicted.norm()).clamp_min(1e-30)),
            "relative_error": float((predicted-exact).norm()/exact.norm().clamp_min(1e-30))}


def score(evidence, exactness, geometry, torch, bars=BARS):
    expected = {(row["row_id"], source, background, gain, method)
                for row in tangent.build_rows() for source in tangent.SOURCES
                for background in tangent.BACKGROUNDS for gain in GAINS for method in METHODS}
    observed = {(x["row_id"], x["source"], x["background"], x["gain"], x["method"])
                for x in evidence}
    if len(evidence) != len(expected) or observed != expected:
        raise QuadraticGainError("incomplete or duplicate evidence lattice")
    numeric = ("target_margin", "target_CE", "native_target_margin", "native_target_CE")
    if not all(math.isfinite(float(x[k])) for x in evidence for k in numeric):
        raise QuadraticGainError("non-finite task evidence")
    if not all(bool(torch.isfinite(v).all()) for tensors in geometry.values()
               for v in tensors.values()):
        raise QuadraticGainError("non-finite gain geometry")
    parent_result = json.loads(PARENT_RESULT.read_text())
    parent_base = {(x["row_id"], x["source"], x["background"]): x
                   for x in parent_result["evidence"] if x["method"] == "base"}
    grouped = defaultdict(lambda: defaultdict(dict))
    for item in evidence:
        grouped[item["cell_id"]][(item["source"], item["background"], item["gain"])][item["method"]] = item
    cells = {}
    for cell_id, entries in grouped.items():
        cells[cell_id] = {}
        row_ids = {row["row_id"] for row in tangent.build_rows()
                   if f"{row['direction_id']}__{row['template_id']}" == cell_id}
        indices = [i for i, row in enumerate(tangent.build_rows()) if row["row_id"] in row_ids]
        for (source, background, gain), by_method in entries.items():
            # Four lexical variants are summarized only after rowwise effects are formed.
            rows_by_method = defaultdict(list)
            for item in evidence:
                if (item["cell_id"], item["source"], item["background"], item["gain"]) \
                        == (cell_id, source, background, gain):
                    base = parent_base[(item["row_id"], source, background)]
                    rows_by_method[item["method"]].append({
                        "margin_effect": item["target_margin"] - base["target_margin"],
                        "CE_effect": base["target_CE"] - item["target_CE"],
                    })
            entry = {method: {metric: statistics.fmean(x[metric] for x in items)
                              for metric in ("margin_effect", "CE_effect")}
                     for method, items in rows_by_method.items()}
            exact_effect = entry["exact"]["margin_effect"]
            entry["predicted"]["margin_recovery"] = (
                entry["predicted"]["margin_effect"] / max(abs(exact_effect), 1e-12)
                * (1 if exact_effect >= 0 else -1))
            entry["predicted"].update(_vector_stats(
                geometry[(source, background, gain)]["exact_delta"][indices],
                geometry[(source, background, gain)]["predicted_delta"][indices], torch))
            cells[cell_id][f"{source}:{background}:{_gain_id(gain)}"] = entry
    instrument = all(float(v) <= bars["maximum_numerical_absolute_error"]
                     for v in exactness.values())
    opposite = [(key, entry) for cell in cells.values() for key, entry in cell.items()
                if key.startswith("opposite:")]
    head = instrument and all(
        e["predicted"]["cosine"] >= bars["minimum_head_cosine"]
        and e["predicted"]["relative_error"] <= bars["maximum_head_relative_error"]
        for _, e in opposite)
    task = instrument and all(
        bars["minimum_task_recovery"] <= e["predicted"]["margin_recovery"]
        <= bars["maximum_task_recovery"] for _, e in opposite)
    extrapolation = head and task and all(
        (":m0p5" in key or ":p1p5" in key) and
        e["predicted"]["cosine"] >= bars["minimum_head_cosine"] and
        e["predicted"]["relative_error"] <= bars["maximum_head_relative_error"] and
        bars["minimum_task_recovery"] <= e["predicted"]["margin_recovery"]
        <= bars["maximum_task_recovery"]
        for key, e in opposite if ":p0p5" not in key)
    lexical_ratios = []
    for cell in cells.values():
        for background in tangent.BACKGROUNDS:
            for gain in GAINS:
                suffix = f"{background}:{_gain_id(gain)}"
                scale = abs(cell[f"opposite:{suffix}"]["exact"]["margin_effect"])
                for method in METHODS:
                    lexical_ratios.append(abs(cell[f"lexical:{suffix}"][method]["margin_effect"])
                                          / max(scale, 1e-12))
    lexical = instrument and max(lexical_ratios) <= bars["maximum_lexical_ratio"]
    return {**exactness, "cells": cells,
            "minimum_opposite_predicted_cosine": min(e["predicted"]["cosine"] for _,e in opposite),
            "maximum_opposite_predicted_relative_error": max(e["predicted"]["relative_error"] for _,e in opposite),
            "minimum_opposite_task_recovery": min(e["predicted"]["margin_recovery"] for _,e in opposite),
            "maximum_opposite_task_recovery": max(e["predicted"]["margin_recovery"] for _,e in opposite),
            "maximum_lexical_ratio": max(lexical_ratios), "predictions": {
                "pred_a_instrument_closure": bool(instrument),
                "pred_b_quadratic_head_prediction": bool(head),
                "pred_c_quadratic_task_manipulation": bool(task),
                "pred_d_extrapolation_stable": bool(extrapolation),
                "pred_e_lexical_effect_bounded": bool(lexical)}}


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv); plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True)); return
    if OUT.exists():
        raise QuadraticGainError(f"refusing to overwrite {OUT}")
    torch, F, facade = tangent.parent.factors._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                            verify_weights_sha256=True)
    evidence, exactness, geometry = evaluate(model, torch, F, facade)
    scored = score(evidence, exactness, geometry, torch)
    terminal = "valid_causal_screen" if scored["predictions"]["pred_a_instrument_closure"] else "invalid"
    result = {"schema": "task14_mlp6_7_quadratic_gain_manipulation_result_v1",
              "candidate_id": CANDIDATE_ID, "terminal": terminal, "plan": plan,
              "checkpoint_weights_sha256": checkpoint.weights_sha256, "score": scored,
              "evidence": evidence, "evaluated_splits": ["LICENSED_HOLDOUT_REUSED_TEXT"],
              "forbidden_splits_opened": []}
    payload = managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": scored["predictions"],
                      "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
