#!/usr/bin/env python3
"""Fresh-text confirmation of Task14 fronted natural-QK number specificity."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_joint_fresh_number_specificity pred_c_qk1_fresh_number_specificity pred_d_qk2_fresh_number_specificity

from __future__ import annotations

from collections import defaultdict
import json
import math
import os
import statistics
import sys

import circuit_fast_screen_candidate_task14_fresh_fronted_natural_qk_number_specificity as authority
import run_task14_head11_3_ood_fronted_natural_qk_number_specificity as prior


ROOT = prior.ROOT
OUT = ROOT / "circuits/fast_screens/task14_head11_3_fresh_fronted_natural_qk_number_specificity_v1_result.json"
PRIOR_ART_SHA256 = "e85974a4b7ef69bbc958e49acb3979bdf629795c9413cbaf4f8fda0afff3a7df"
CONDITIONS = prior.CONDITIONS


def build_rows():
    return authority.build_rows()


def _expected_sign(cell_id):
    return 1 if cell_id.startswith("singular_to_plural") else -1


def compile_plan():
    rows = build_rows()
    return {"schema": "task14_head11_3_fresh_fronted_natural_qk_number_specificity_plan_v1",
            "candidate_id": "subject_verb.number_agreement.head11_3_fresh_fronted_natural_qk_number_specificity",
            "split": "FRESH_TEXT", "screen_tier": "BASIC", "row_count": len(rows),
            "authority_sha256": authority.validate_rows(rows), "conditions": list(CONDITIONS),
            "price": {"model_forwards": 3, "example_evaluations": 416,
                      "backwards": 0, "parameter_updates": 0},
            "bars": {"minimum_native_correct_each_role_each_cell": 7,
                     "maximum_native_replay_absolute_logit_error": 7e-5,
                     "maximum_source_term_identity_absolute_error": 5e-5,
                     "maximum_direct_score_identity_absolute_error": 5e-6,
                     "maximum_installed_term_absolute_error": 5e-5,
                     "minimum_opposite_joint_absolute_margin_effect": .05,
                     "minimum_expected_row_sign_fraction": .75,
                     "maximum_same_over_opposite_joint_margin_leakage": .25,
                     "maximum_same_over_opposite_joint_ce_leakage": .25,
                     "minimum_branch_signed_fraction_of_joint": .20,
                     "maximum_same_over_opposite_branch_margin_leakage": .25},
            "scope": "FRESH_TEXT_NEW_AUTHORITY_NEW_INTERVENTION",
            "closed_claims": ["syntax_general_semantics", "individual_query_key_semantics",
                              "selectivity", "completeness", "upstream_writers"]}


def _condition_effect(grouped, group_key, condition, expected):
    baseline = grouped[(group_key, "recipient_score")]
    target = grouped[(group_key, condition)]
    margins = [right["donor_margin"] - left["donor_margin"]
               for left, right in zip(baseline, target)]
    ces = [left["donor_ce"] - right["donor_ce"] for left, right in zip(baseline, target)]
    return {"mean_margin_effect": statistics.fmean(margins),
            "mean_ce_effect": statistics.fmean(ces),
            "expected_margin_sign_fraction": sum(value * expected > 0 for value in margins) / len(margins)}


def score(evidence, capability, exactness, bars):
    grouped = defaultdict(list)
    for row in evidence:
        grouped[(row["cell_id"], row["condition"])].append(row)
    cells = {}
    for cell_id, native_counts in capability.items():
        expected = _expected_sign(cell_id)
        effects = {condition: _condition_effect(grouped, cell_id, condition, expected)
                   for condition in CONDITIONS[1:]}
        joint = effects["opposite_joint"]
        effects["opposite_joint"]["same_margin_leakage"] = (
            abs(effects["same_joint"]["mean_margin_effect"]) / abs(joint["mean_margin_effect"]))
        effects["opposite_joint"]["same_ce_leakage"] = (
            abs(effects["same_joint"]["mean_ce_effect"]) / abs(joint["mean_ce_effect"])
            if abs(joint["mean_ce_effect"]) > 1e-12 else None)
        for branch in ("qk1", "qk2"):
            opposite = effects[f"opposite_{branch}"]
            same = effects[f"same_{branch}"]
            opposite["signed_fraction_of_joint"] = (
                opposite["mean_margin_effect"] / joint["mean_margin_effect"])
            opposite["same_margin_leakage"] = (
                abs(same["mean_margin_effect"]) / abs(opposite["mean_margin_effect"])
                if abs(opposite["mean_margin_effect"]) > 1e-12 else None)
        cells[cell_id] = {"native_correct_of_8": native_counts,
                          "expected_score_effect_sign": expected, "effects": effects}
    diagnostic = {}
    diagnostic_grouped = defaultdict(list)
    for row in evidence:
        diagnostic_grouped[(row["diagnostic_cell_id"], row["condition"])].append(row)
    for diagnostic_id in sorted({row["diagnostic_cell_id"] for row in evidence}):
        expected = _expected_sign(diagnostic_id)
        diagnostic[diagnostic_id] = {
            condition: _condition_effect(diagnostic_grouped, diagnostic_id, condition, expected)
            for condition in ("same_joint", "opposite_joint", "same_qk1", "opposite_qk1",
                              "same_qk2", "opposite_qk2")}
    exact_live = (exactness["native_replay_max_absolute_logit_error"] <=
                  bars["maximum_native_replay_absolute_logit_error"] and
                  exactness["source_term_identity_max_absolute_error"] <=
                  bars["maximum_source_term_identity_absolute_error"] and
                  exactness["direct_score_identity_max_absolute_error"] <=
                  bars["maximum_direct_score_identity_absolute_error"] and
                  exactness["installed_term_max_absolute_error"] <=
                  bars["maximum_installed_term_absolute_error"] and
                  all(math.isfinite(row[key]) for row in evidence
                      for key in ("donor_margin", "donor_ce")))
    native_live = all(min(counts.values()) >= bars["minimum_native_correct_each_role_each_cell"]
                      for counts in capability.values())
    joint_live = all(
        abs(cell["effects"]["opposite_joint"]["mean_margin_effect"]) >=
            bars["minimum_opposite_joint_absolute_margin_effect"] and
        cell["effects"]["opposite_joint"]["mean_margin_effect"] * cell["expected_score_effect_sign"] > 0 and
        cell["effects"]["opposite_joint"]["mean_ce_effect"] * cell["expected_score_effect_sign"] > 0 and
        cell["effects"]["opposite_joint"]["expected_margin_sign_fraction"] >=
            bars["minimum_expected_row_sign_fraction"] for cell in cells.values())
    instrument = exact_live and native_live and joint_live
    joint_specific = instrument and all(
        cell["effects"]["opposite_joint"]["same_margin_leakage"] <=
            bars["maximum_same_over_opposite_joint_margin_leakage"] and
        cell["effects"]["opposite_joint"]["same_ce_leakage"] is not None and
        cell["effects"]["opposite_joint"]["same_ce_leakage"] <=
            bars["maximum_same_over_opposite_joint_ce_leakage"] for cell in cells.values())

    def branch_specific(branch):
        return instrument and all(
            cell["effects"][f"opposite_{branch}"]["signed_fraction_of_joint"] >=
                bars["minimum_branch_signed_fraction_of_joint"] and
            cell["effects"][f"opposite_{branch}"]["expected_margin_sign_fraction"] >=
                bars["minimum_expected_row_sign_fraction"] and
            cell["effects"][f"opposite_{branch}"]["same_margin_leakage"] is not None and
            cell["effects"][f"opposite_{branch}"]["same_margin_leakage"] <=
                bars["maximum_same_over_opposite_branch_margin_leakage"] and
            cell["effects"][f"opposite_{branch}"]["mean_ce_effect"] *
                cell["expected_score_effect_sign"] > 0 for cell in cells.values())
    return {**exactness, "cells": cells, "direction_attractor_diagnostics": diagnostic,
            "predictions": {"pred_a_instrument_live": instrument,
                "pred_b_joint_fresh_number_specificity": joint_specific,
                "pred_c_qk1_fresh_number_specificity": branch_specific("qk1"),
                "pred_d_qk2_fresh_number_specificity": branch_specific("qk2")}}


def _native_capability(rows, native, count, torch):
    counts = {cell: {"recipient": 0, "same": 0, "opposite": 0}
              for cell in sorted({row["cell_id"] for row in rows})}
    for index, row in enumerate(rows):
        q = 8; cell = row["cell_id"]
        for role, offset, answer, foil in (
            ("recipient", 0, row["base_answer_id"], row["base_foil_id"]),
            ("same", count, row["same_answer_id"], row["donor_answer_id"]),
            ("opposite", 2*count, row["opposite_answer_id"], row["base_answer_id"]),
        ):
            counts[cell][role] += int(float(native[offset+index, q, answer]
                                                   - native[offset+index, q, foil]) > 0)
    return counts


def evaluate(model, torch, F, facade, plan):
    rows = build_rows(); count = len(rows); device = next(model.parameters()).device
    pad = prior.qk.parent.atlas.test_atlas._pad
    base_tokens, base_finals = pad(rows, "base_ids", 9, torch, device)
    same_tokens, same_finals = pad(rows, "same_ids", 9, torch, device)
    opposite_tokens, opposite_finals = pad(rows, "opposite_ids", 9, torch, device)
    tokens = torch.cat((base_tokens, same_tokens, opposite_tokens))
    finals = torch.cat((base_finals, same_finals, opposite_finals))
    native = prior.qk.parent.atlas.test_atlas.factor_parent._native_logits(model, tokens, torch, F)
    replay, factors = prior.qk._qk_factor_forward(model, tokens, finals, torch, F, facade)
    base = {name: value[:count] for name, value in factors.items()}
    same = {name: value[count:2*count] for name, value in factors.items()}
    opposite = {name: value[2*count:] for name, value in factors.items()}
    patch = prior._compile_patch_batch(base_tokens, base_finals, base, same, opposite, rows, torch)
    patched, patched_factors = prior.qk._qk_factor_forward(
        model, patch["tokens"], patch["finals"], torch, F, facade,
        replacement_heads=patch["replacement_heads"])
    exactness = {
        "native_replay_max_absolute_logit_error": float((native-replay).abs().max()),
        "source_term_identity_max_absolute_error": max(float((
            torch.einsum("bk,bkd->bd", side["p"], side["u"]) - side["head"]).abs().max())
            for side in (base, same, opposite, patched_factors)),
        "direct_score_identity_max_absolute_error": max(prior._direct_score_error(side, torch)
                                                          for side in (base, same, opposite)),
        "installed_term_max_absolute_error": 0.0,
    }
    evidence = []
    for output_index, (row_index, condition, _cell, _group) in enumerate(patch["specs"]):
        row = rows[row_index]; q = 8
        native_margin, native_ce = prior.qk.parent.atlas.test_atlas._donor_metrics(
            replay[row_index], row, q, torch)
        margin, ce = prior.qk.parent.atlas.test_atlas._donor_metrics(
            patched[output_index], row, q, torch)
        evidence.append({"row_id": row["row_id"], "group_id": row["group_id"],
            "cell_id": row["cell_id"], "diagnostic_cell_id": row["diagnostic_cell_id"],
            "condition": condition, "native_donor_margin": native_margin,
            "donor_margin": margin, "margin_delta": margin-native_margin,
            "native_donor_ce": native_ce, "donor_ce": ce, "donor_ce_gain": native_ce-ce})
        exactness["installed_term_max_absolute_error"] = max(
            exactness["installed_term_max_absolute_error"], float((
                patch["replacement_heads"][output_index] - base["head"][row_index]
                + base["p"][row_index, 8] * base["u"][row_index, 8]
                - patch["installed_scalars"][output_index] * opposite["u"][row_index, 8]
            ).abs().max()))
    return evidence, _native_capability(rows, native, count, torch), exactness


def main():
    plan = compile_plan()
    if "--dry-run" in sys.argv or os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    if OUT.exists(): raise RuntimeError(f"refusing to overwrite {OUT}")
    torch, F, facade = prior.qk.parent.atlas.test_atlas.factor_parent._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                            verify_weights_sha256=True)
    with torch.no_grad(): evidence, capability, exactness = evaluate(model, torch, F, facade, plan)
    scored = score(evidence, capability, exactness, plan["bars"]); predictions = scored["predictions"]
    terminal = ("invalid" if not predictions["pred_a_instrument_live"] else
                "fresh_joint_number_specificity" if predictions["pred_b_joint_fresh_number_specificity"] else
                "fresh_natural_qk_number_specificity_null")
    result = {"schema": "task14_head11_3_fresh_fronted_natural_qk_number_specificity_result_v1",
              "plan": plan, "prior_art_sha256": PRIOR_ART_SHA256,
              "checkpoint_weights_sha256": checkpoint.weights_sha256, "terminal": terminal,
              "score": scored, "evidence": evidence, "evaluated_splits": ["FRESH_TEXT"],
              "forbidden_splits_opened": [], "model_forwards": 3}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(result, indent=1)+"\n")
    print(json.dumps({"result": str(OUT), "terminal": terminal,
                      "predictions": predictions}, indent=2))


if __name__ == "__main__": main()
