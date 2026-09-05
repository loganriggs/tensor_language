#!/usr/bin/env python3
"""Continuously manipulate the E/A/U/W gate controlling grouped MLP6--7."""

# BQGATE: EXPERIMENT pred_a_instrument_and_binary_endpoint_closure pred_b_midpoint_head_and_task_prediction pred_c_extrapolated_head_and_task_prediction pred_d_directional_gate_is_manipulable pred_e_template_stable pred_f_lexical_collateral_absolutely_small
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
import run_task14_fresh_fronted_mlp6_7_background_composition_transfer as causal
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as gate
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_fresh_fronted_mlp6_7_continuous_background_gain_manipulation_v1.json"
AMENDMENT = ROOT / "circuits/prior_art/task14_fresh_fronted_mlp6_7_continuous_background_gain_manipulation_v1_price_amendment.json"
PARENT_RESULT = ROOT / "circuits/fast_screens/task14_fresh_fronted_mlp6_7_background_composition_transfer_v1_result.json"
OUT = ROOT / "circuits/fast_screens/task14_fresh_fronted_mlp6_7_continuous_background_gain_manipulation_v1_result.json"
PRIOR_ART_SHA256 = "8a137c17d54b6d865c5288b3ebe0709fc0efb1203c8d967744186d39043ae032"
AMENDMENT_SHA256 = "4c0c418ce999deba01177486dc38b59a60c15e92feb3ba180467934245ca0388"
PARENT_RESULT_SHA256 = "2685549927059d3623f68961c4cc8102d24bda3aeb02c7d8b22b2d747edb7a9a"
CANDIDATE_ID = "subject_verb.number_agreement.fresh_fronted_mlp6_7_continuous_background_gain_manipulation_v1"
ENDPOINT_GAINS = (0.0, 1.0)
NEW_GAINS = (-0.5, 0.5, 1.5)
ALL_GAINS = (-0.5, 0.0, 0.5, 1.0, 1.5)
PATCH_CHUNK_ROWS = 256
BARS = {"maximum_numerical_absolute_error": 5e-5,
        "minimum_midpoint_head_cosine": .98,
        "maximum_midpoint_head_relative_error": .15,
        "minimum_midpoint_task_recovery": .80,
        "maximum_midpoint_task_recovery": 1.20,
        "minimum_extrapolated_head_cosine": .95,
        "maximum_extrapolated_head_relative_error": .30,
        "minimum_extrapolated_task_recovery": .70,
        "maximum_extrapolated_task_recovery": 1.30,
        "minimum_live_margin_effect": .01,
        "maximum_template_q_difference": .04,
        "maximum_absolute_lexical_margin_effect": .02}


class ContinuousBackgroundError(ValueError):
    pass


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _methods(gain):
    return ("base", "exact") if gain in ENDPOINT_GAINS else ("base", "exact", "predicted")


def compile_specs(row_count=32):
    return tuple((row, source, gain, method) for row in range(row_count)
                 for source in tangent.SOURCES for gain in ALL_GAINS
                 for method in _methods(gain))


def derive_price(row_count=32):
    installations = len(compile_specs(row_count))
    chunks = math.ceil(installations / PATCH_CHUNK_ROWS)
    role_rows = row_count * 3
    return {"physical_model_forwards": 2 + 2 * chunks,
            "example_evaluations": 2 * role_rows + 2 * installations,
            "causal_installations": installations, "backwards": 0,
            "parameter_updates": 0, "maximum_patch_chunk_rows": PATCH_CHUNK_ROWS,
            "patch_chunks": chunks}


def validate_preflight():
    for path, expected, label in ((PRIOR_ART, PRIOR_ART_SHA256, "prior-art"),
                                  (AMENDMENT, AMENDMENT_SHA256, "price amendment"),
                                  (PARENT_RESULT, PARENT_RESULT_SHA256, "parent result")):
        if _sha256(path) != expected:
            raise ContinuousBackgroundError(f"{label} changed")
    causal.validate_preflight()
    result = json.loads(PARENT_RESULT.read_text())
    if result.get("terminal") != "valid_causal_screen" or not all(
            result.get("score", {}).get("predictions", {}).values()):
        raise ContinuousBackgroundError("parent does not license continuous gate test")
    if derive_price() != {"physical_model_forwards": 10,
            "example_evaluations": 1856, "causal_installations": 832,
            "backwards": 0, "parameter_updates": 0,
            "maximum_patch_chunk_rows": 256, "patch_chunks": 4}:
        raise ContinuousBackgroundError("derived price changed")


def compile_plan():
    validate_preflight()
    return {"schema": "task14_fresh_fronted_mlp6_7_continuous_background_gain_manipulation_plan_v1",
            "candidate_id": CANDIDATE_ID,
            "split": "FRESH_FRONTED_BINARY_LATTICE_REUSE_NEW_CONTINUOUS_BACKGROUND_GAINS",
            "row_count": 32, "sources": list(tangent.SOURCES),
            "endpoint_gains": list(ENDPOINT_GAINS), "new_gains": list(NEW_GAINS),
            "prior_art_sha256": PRIOR_ART_SHA256,
            "price_amendment_sha256": AMENDMENT_SHA256,
            "parent_result_sha256": PARENT_RESULT_SHA256,
            "bars": dict(BARS), "price": derive_price()}


def _head_metrics(exact, predicted, torch):
    exact_flat = exact.flatten().double(); predicted_flat = predicted.flatten().double()
    exact_norm = float(torch.linalg.vector_norm(exact_flat))
    predicted_norm = float(torch.linalg.vector_norm(predicted_flat))
    cosine = float(torch.dot(exact_flat, predicted_flat) /
                   max(exact_norm * predicted_norm, 1e-30))
    error = float(torch.linalg.vector_norm(predicted_flat - exact_flat)) / max(exact_norm, 1e-30)
    return {"cosine": cosine, "relative_error": error,
            "exact_norm": exact_norm, "predicted_norm": predicted_norm}


def evaluate(model, torch, F, facade):
    rows = causal.build_rows(); n = len(rows); parent = tangent.parent
    device = next(model.parameters()).device
    tokens, finals = parent.downstream.depth.parent.v1._role_batch(rows, torch, device)
    native_roles = parent.factors._native_logits(model, tokens, torch, F)
    replay, captured, projection, role_closure, inputs = parent._decomposed_forward(
        model, tokens, finals, torch, F, facade)
    roles = {"recipient": tangent._role_slice(captured, 0, n),
             "opposite": tangent._role_slice(captured, n, 2*n)}
    input_roles = {"recipient": tangent._role_slice(inputs, 0, n),
                   "opposite": tangent._role_slice(inputs, n, 2*n),
                   "lexical": tangent._role_slice(inputs, 2*n, 3*n)}
    function = tangent._head_function(model, roles["recipient"], roles["opposite"],
        model.transformer.h[parent.LAYER].attn, projection, torch, F)
    heads = {}; head_summary = {}
    with torch.no_grad():
        for source in tangent.SOURCES:
            base0 = gate._raw_for(input_roles["recipient"], input_roles[source], "", F)
            base1 = gate._raw_for(input_roles["recipient"], input_roles[source], "EAUW", F)
            exact0 = gate._raw_for(input_roles["recipient"], input_roles[source], "YZ", F)
            exact1 = gate._raw_for(input_roles["recipient"], input_roles[source], "EAUWYZ", F)
            delta0 = function(exact0) - function(base0)
            delta1 = function(exact1) - function(base1)
            for gain in ALL_GAINS:
                base_raw = base0 + gain * (base1 - base0)
                exact_raw = exact0 + gain * (exact1 - exact0)
                base_head = function(base_raw).detach()
                exact_head = function(exact_raw).detach()
                heads[(source, gain, "base")] = base_head
                heads[(source, gain, "exact")] = exact_head
                if gain in NEW_GAINS:
                    predicted = (base_head + delta0 + gain * (delta1 - delta0)).detach()
                    heads[(source, gain, "predicted")] = predicted
                    for cell_id in sorted({f"{r['direction_id']}__{r['template_id']}" for r in rows}):
                        indices = [i for i, row in enumerate(rows)
                                   if f"{row['direction_id']}__{row['template_id']}" == cell_id]
                        head_summary[f"{cell_id}::{source}::{gain}"] = _head_metrics(
                            (exact_head-base_head)[indices], (predicted-base_head)[indices], torch)
    specs = compile_specs(n); indices = torch.tensor([x[0] for x in specs], device=device)
    patch_tokens = tokens[:n][indices]
    patch_finals = torch.full_like(indices, parent.SUBJECT_POSITION)
    replacements = torch.stack([heads[(source, gain, method)] [row]
                                for row, source, gain, method in specs])
    noop_mask = torch.tensor([gain == 0.0 and method == "base"
                              for _, _, gain, method in specs], dtype=torch.bool, device=device)
    native_chunks, patched_chunks, closures = [], [], []
    noop_error = 0.0
    for start in range(0, len(specs), PATCH_CHUNK_ROWS):
        stop = min(start + PATCH_CHUNK_ROWS, len(specs))
        native = parent.factors._native_logits(model, patch_tokens[start:stop], torch, F)
        patched, _, _, closure = parent.downstream._decomposed_forward(
            model, patch_tokens[start:stop], patch_finals[start:stop], torch, F, facade,
            replacement_heads=replacements[start:stop],
            native_reinstall_mask=noop_mask[start:stop])
        mask = noop_mask[start:stop]
        if bool(mask.any()): noop_error = max(noop_error, float((patched[mask]-native[mask]).abs().max()))
        native_chunks.append(native[:, parent.SUBJECT_POSITION].detach().cpu())
        patched_chunks.append(patched[:, parent.SUBJECT_POSITION].detach().cpu())
        closures.append(closure)
    native_patch = torch.cat(native_chunks); patched = torch.cat(patched_chunks)
    evidence = []
    for i, (row_index, source, gain, method) in enumerate(specs):
        base = parent.grandparent._both_metrics(native_patch[i], rows[row_index], torch)
        value = parent.grandparent._both_metrics(patched[i], rows[row_index], torch)
        evidence.append({"row_id": rows[row_index]["row_id"],
            "cell_id": f"{rows[row_index]['direction_id']}__{rows[row_index]['template_id']}",
            "source": source, "gain": gain, "method": method,
            "target_margin_improvement": value[f"{source}_target_margin"]-base[f"{source}_target_margin"],
            "target_CE_improvement": base[f"{source}_target_CE"]-value[f"{source}_target_CE"]})
    exactness = {"native_role_replay_max_absolute_logit_error": float((native_roles-replay).abs().max()),
        "role_state_closure_max_absolute_error": role_closure["input_state_closure_max_absolute_error"],
        "role_normalized_closure_max_absolute_error": role_closure["input_normalized_closure_max_absolute_error"],
        "downstream_state_closure_max_absolute_error": max(x["state_sum_max_absolute_error"] for x in closures),
        "downstream_normalized_closure_max_absolute_error": max(x["normalized_state_max_absolute_error"] for x in closures),
        "recipient_noop_full_logit_max_absolute_error": noop_error}
    return evidence, head_summary, exactness


def score(evidence, head_summary, exactness, bars=BARS):
    expected = {(row["row_id"], f"{row['direction_id']}__{row['template_id']}", source, gain, method)
        for row in causal.build_rows() for source in tangent.SOURCES for gain in ALL_GAINS
        for method in _methods(gain)}
    observed = [(x["row_id"], x["cell_id"], x["source"], x["gain"], x["method"]) for x in evidence]
    if len(observed) != len(expected) or set(observed) != expected or len(set(observed)) != len(expected):
        raise ContinuousBackgroundError("continuous lattice incomplete or duplicated")
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for item in evidence:
        grouped[item["cell_id"]][item["source"]][float(item["gain"])][item["method"]].append(item)
    cells = {}; endpoint_error = 0.0
    parent_cells = json.loads(PARENT_RESULT.read_text())["score"]["cells"]
    for cell_id, sources in grouped.items():
        cells[cell_id] = {}
        for source, gains in sources.items():
            cells[cell_id][source] = {}
            for gain, methods in gains.items():
                base = methods["base"]
                summary = {}
                for method in _methods(gain):
                    if method == "base": continue
                    summary[method] = {metric: statistics.fmean(
                        float(a[f"target_{metric}_improvement"])-float(b[f"target_{metric}_improvement"])
                        for a, b in zip(methods[method], base)) for metric in ("margin", "CE")}
                if "predicted" in summary:
                    summary["recovery"] = {metric: summary["predicted"][metric] /
                        (summary["exact"][metric] if abs(summary["exact"][metric]) >= 1e-30 else 1e-30)
                        for metric in ("margin", "CE")}
                cells[cell_id][source][str(gain)] = summary
            for gain, background in ((0.0, ""), (1.0, "EAUW")):
                for metric in ("margin", "CE"):
                    endpoint_error = max(endpoint_error, abs(cells[cell_id][source][str(gain)]["exact"][metric]
                        - float(parent_cells[cell_id][source][metric]["q"][background])))
    exactness = {**exactness, "binary_endpoint_margin_CE_max_absolute_error": endpoint_error}
    instrument = all(float(x) <= bars["maximum_numerical_absolute_error"] for x in exactness.values())
    def gain_pass(gain, midpoint):
        head_cos = bars["minimum_midpoint_head_cosine"] if midpoint else bars["minimum_extrapolated_head_cosine"]
        head_err = bars["maximum_midpoint_head_relative_error"] if midpoint else bars["maximum_extrapolated_head_relative_error"]
        lo = bars["minimum_midpoint_task_recovery"] if midpoint else bars["minimum_extrapolated_task_recovery"]
        hi = bars["maximum_midpoint_task_recovery"] if midpoint else bars["maximum_extrapolated_task_recovery"]
        heads_ok = all(value["cosine"] >= head_cos and value["relative_error"] <= head_err
            for key, value in head_summary.items() if key.endswith(f"::opposite::{gain}"))
        tasks = [cell["opposite"][str(gain)] for cell in cells.values()]
        live = [task for task in tasks if abs(task["exact"]["margin"]) >= bars["minimum_live_margin_effect"]]
        tasks_ok = bool(live) and all(lo <= task["recovery"]["margin"] <= hi for task in live)
        return heads_ok and tasks_ok
    midpoint = gain_pass(0.5, True)
    extrapolated = all(gain_pass(gain, False) for gain in (-0.5, 1.5))
    direction_values = defaultdict(dict)
    for direction in ("plural_to_singular", "singular_to_plural"):
        for gain in ALL_GAINS:
            direction_values[direction][str(gain)] = statistics.fmean(
                cell["opposite"][str(gain)]["exact"]["margin"]
                for cell_id, cell in cells.items() if cell_id.startswith(direction + "__"))
    p = [direction_values["plural_to_singular"][str(g)] for g in ALL_GAINS]
    s = [direction_values["singular_to_plural"][str(g)] for g in ALL_GAINS]
    manip = all(a > b for a, b in zip(p, p[1:])) and all(a < b for a, b in zip(s, s[1:])) \
        and direction_values["plural_to_singular"]["1.5"] < 0 \
        and direction_values["singular_to_plural"]["1.5"] > 0
    template_max = 0.0
    for direction in direction_values:
        selected = [cell for cell_id, cell in cells.items() if cell_id.startswith(direction + "__")]
        for gain in ALL_GAINS:
            template_max = max(template_max, abs(selected[0]["opposite"][str(gain)]["exact"]["margin"]
                - selected[1]["opposite"][str(gain)]["exact"]["margin"]))
    lexical_max = max(abs(cell["lexical"][str(gain)]["exact"]["margin"])
                      for cell in cells.values() for gain in ALL_GAINS)
    return {**exactness, "cells": cells, "head_predictions": head_summary,
            "direction_mean_exact_margin_q": direction_values,
            "maximum_template_q_difference": template_max,
            "maximum_absolute_lexical_margin_effect": lexical_max,
            "predictions": {"pred_a_instrument_and_binary_endpoint_closure": bool(instrument),
                "pred_b_midpoint_head_and_task_prediction": bool(instrument and midpoint),
                "pred_c_extrapolated_head_and_task_prediction": bool(instrument and extrapolated),
                "pred_d_directional_gate_is_manipulable": bool(instrument and manip),
                "pred_e_template_stable": bool(instrument and template_max <= bars["maximum_template_q_difference"]),
                "pred_f_lexical_collateral_absolutely_small": bool(instrument and lexical_max <= bars["maximum_absolute_lexical_margin_effect"])}}


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv); plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True)); return
    if OUT.exists(): raise ContinuousBackgroundError(f"refusing to overwrite {OUT}")
    torch, F, facade = tangent.parent.factors._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                            verify_weights_sha256=True)
    evidence, heads, exactness = evaluate(model, torch, F, facade)
    scored = score(evidence, heads, exactness)
    terminal = "valid_causal_screen" if scored["predictions"][
        "pred_a_instrument_and_binary_endpoint_closure"] else "invalid"
    result = {"schema": "task14_fresh_fronted_mlp6_7_continuous_background_gain_manipulation_result_v1",
        "candidate_id": CANDIDATE_ID, "terminal": terminal, "plan": plan,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "score": scored, "evidence": evidence,
        "evaluated_splits": ["FRESH_FRONTED_BINARY_LATTICE_REUSE_NEW_CONTINUOUS_BACKGROUND_GAINS"],
        "forbidden_splits_opened": []}
    payload = managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": scored["predictions"],
                      "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__": main()
