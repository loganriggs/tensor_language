#!/usr/bin/env python3
"""Matched natural-state number specificity of Task14's fronted self QK score."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_joint_natural_score_number_specificity pred_c_qk1_natural_score_number_specificity pred_d_qk2_natural_score_number_specificity

from __future__ import annotations

from collections import defaultdict
import json
import math
import os
import statistics
import sys

import run_task14_head11_3_ood_fronted_self_qk_factorial as qk


ROOT = qk.ROOT
OUT = ROOT / "circuits/fast_screens/task14_head11_3_ood_fronted_natural_qk_number_specificity_v1_result.json"
PRIOR_ART_SHA256 = "f78a318377ce42bbd1c067244e5eb32afa5f6c9836f13bf9a89f4e211a94d85e"
CONDITIONS = ("recipient_score", "same_qk1", "opposite_qk1", "same_qk2",
              "opposite_qk2", "same_joint", "opposite_joint")
EXPECTED_SIGN = {"fronted_singular_to_fronted_plural": 1,
                 "fronted_plural_to_fronted_singular": -1}


def build_triples():
    recipients = qk.build_rows()
    source = {row["group_id"]: row for row in
              qk.parent.atlas.authority._CANDIDATE._source_rows()
              if row["transform_id"] == "A1"}
    by_category = defaultdict(list)
    for row in recipients:
        raw = source[row["group_id"]]
        category = (row["base_subject_number"], bool(raw["base_attractor_plural"]),
                    bool(raw["base_second_attractor_plural"]))
        by_category[category].append(row)
    donor_for = {}
    for category, rows in by_category.items():
        ordered = sorted(rows, key=lambda row: int(source[row["group_id"]]["group_number"]))
        if len(ordered) < 2:
            raise ValueError(f"cannot cycle foreign groups in {category}")
        for index, row in enumerate(ordered):
            donor_for[row["row_id"]] = ordered[(index + 1) % len(ordered)]
    output = []
    for row in recipients:
        foreign = donor_for[row["row_id"]]
        raw = source[foreign["group_id"]]
        same_ids, opposite_ids = list(raw["base_ids"]), list(raw["donor_ids"])
        differences = [index for index, pair in enumerate(zip(same_ids, opposite_ids))
                       if pair[0] != pair[1]]
        augmented = dict(row)
        augmented.update(
            specificity_row_id=qk.parent.atlas.test_atlas._canonical(
                ["task14_fronted_natural_qk_number_specificity_v1", row["row_id"], foreign["group_id"]]),
            foreign_group_id=foreign["group_id"], same_ids=same_ids,
            opposite_ids=opposite_ids, same_text=raw["base_text"],
            opposite_text=raw["donor_text"],
            attractor_plural=bool(raw["base_attractor_plural"]),
            second_attractor_plural=bool(raw["base_second_attractor_plural"]),
        )
        if foreign["group_id"] == row["group_id"] or differences != [8]:
            raise ValueError("foreign same/opposite pair is not a token-8-only counterfactual")
        if same_ids[8] == row["base_ids"][8] or same_ids[:8] == row["base_ids"][:8]:
            raise ValueError("foreign donor does not change both noun and context")
        if raw["base_subject_number"] != row["base_subject_number"] \
                or raw["donor_subject_number"] == row["base_subject_number"]:
            raise ValueError("foreign donor number relation changed")
        recipient_raw = source[row["group_id"]]
        if (raw["base_attractor_plural"], raw["base_second_attractor_plural"]) != (
                recipient_raw["base_attractor_plural"], recipient_raw["base_second_attractor_plural"]):
            raise ValueError("foreign pairing does not preserve both attractor pluralities")
        output.append(augmented)
    if len(output) != 32 or len({row["foreign_group_id"] for row in output}) != 32:
        raise ValueError("cyclic foreign pairing is not a 32-row permutation")
    return output


def compile_plan():
    rows = build_triples()
    return {"schema": "task14_head11_3_ood_fronted_natural_qk_number_specificity_plan_v1",
            "candidate_id": "subject_verb.number_agreement.head11_3_ood_fronted_natural_qk_number_specificity",
            "split": "OOD_TEXT_REUSE_NEW_INTERVENTION", "screen_tier": "BASIC",
            "row_count": len(rows), "authority_sha256": qk.parent.atlas.test_atlas._canonical(rows),
            "pairing": "cyclic foreign A1 group within recipient number and both attractor plurality states",
            "conditions": list(CONDITIONS),
            "price": {"model_forwards": 3, "example_evaluations": 416,
                      "backwards": 0, "parameter_updates": 0},
            "bars": {"minimum_native_accuracy_each_side_each_cell": .85,
                     "maximum_native_replay_absolute_logit_error": 7e-5,
                     "maximum_source_term_identity_absolute_error": 5e-5,
                     "maximum_direct_score_identity_absolute_error": 5e-6,
                     "maximum_installed_term_absolute_error": 5e-5,
                     "minimum_live_absolute_joint_effect": .05,
                     "minimum_joint_row_sign_fraction": .75,
                     "maximum_same_over_opposite_effect": .25,
                     "minimum_branch_signed_fraction_of_joint": .20,
                     "minimum_branch_row_sign_fraction": .75},
            "scope": "OOD_TEXT_REUSE_NEW_INTERVENTION",
            "closed_claims": ["pristine_OOD_confirmation", "individual_query_key_semantics",
                              "syntax_general_semantics", "selectivity", "completeness"]}


def _branch_scores(factors, torch):
    width = factors["q"].shape[-1]
    a = (factors["q"] * factors["k"][:, 8]).sum(-1) / width
    b = (factors["q2"] * factors["k2"][:, 8]).sum(-1) / width
    return a, b


def _compile_patch_batch(tokens, finals, base, same, opposite, rows, torch):
    ba, bb = _branch_scores(base, torch)
    sa, sb = _branch_scores(same, torch)
    oa, ob = _branch_scores(opposite, torch)
    score_by_condition = {
        "recipient_score": ba * bb, "same_qk1": sa * bb, "opposite_qk1": oa * bb,
        "same_qk2": ba * sb, "opposite_qk2": ba * ob,
        "same_joint": sa * sb, "opposite_joint": oa * ob,
    }
    indices, heads, specs, scalars = [], [], [], []
    for row_index, row in enumerate(rows):
        native_term = base["p"][row_index, 8] * base["u"][row_index, 8]
        for condition in CONDITIONS:
            scalar = score_by_condition[condition][row_index]
            term = scalar * opposite["u"][row_index, 8]
            indices.append(row_index); scalars.append(scalar)
            heads.append(base["head"][row_index] - native_term + term)
            specs.append((row_index, condition, row["atlas_cell_id"], row["group_id"]))
    index = torch.tensor(indices, device=tokens.device)
    return {"tokens": tokens[index], "finals": finals[index],
            "replacement_heads": torch.stack(heads), "installed_scalars": torch.stack(scalars),
            "specs": specs}


def _direct_score_error(factors, torch):
    a, b = _branch_scores(factors, torch)
    return float((a * b - factors["p"][:, 8]).abs().max())


def _sign(value):
    return 1 if value > 0 else -1 if value < 0 else 0


def score(evidence, capability, exactness, bars):
    grouped = defaultdict(list)
    for row in evidence:
        grouped[(row["atlas_cell_id"], row["condition"])].append(row)
    cells = {}
    for cell_id, accuracy in capability.items():
        expected = EXPECTED_SIGN[cell_id]
        baseline = grouped[(cell_id, "recipient_score")]

        def effect(condition):
            target = grouped[(cell_id, condition)]
            margins = [right["donor_margin"] - left["donor_margin"]
                       for left, right in zip(baseline, target)]
            ces = [left["donor_ce"] - right["donor_ce"] for left, right in zip(baseline, target)]
            return {"mean_margin_effect": statistics.fmean(margins),
                    "mean_ce_effect": statistics.fmean(ces),
                    "row_expected_sign_fraction": sum(value * expected > 0 for value in margins) / len(margins)}
        effects = {condition: effect(condition) for condition in CONDITIONS[1:]}
        joint = effects["opposite_joint"]
        for prefix in ("joint", "qk1", "qk2"):
            same_name = "same_joint" if prefix == "joint" else f"same_{prefix}"
            opposite_name = "opposite_joint" if prefix == "joint" else f"opposite_{prefix}"
            denominator = effects[opposite_name]["mean_margin_effect"]
            effects[opposite_name]["same_number_leakage_ratio"] = (
                abs(effects[same_name]["mean_margin_effect"]) / abs(denominator)
                if abs(denominator) > 1e-12 else None)
            if prefix != "joint":
                effects[opposite_name]["signed_fraction_of_joint"] = (
                    denominator / joint["mean_margin_effect"]
                    if abs(joint["mean_margin_effect"]) > 1e-12 else None)
        ordered = sorted({row["group_id"] for row in baseline})
        halves = []
        for groups in (set(ordered[:8]), set(ordered[8:])):
            half = {}
            for condition in CONDITIONS:
                values = [row for row in grouped[(cell_id, condition)] if row["group_id"] in groups]
                half[condition] = statistics.fmean(row["donor_margin"] for row in values)
            halves.append({"group_ids": sorted(groups),
                           "same_joint_effect": half["same_joint"] - half["recipient_score"],
                           "opposite_joint_effect": half["opposite_joint"] - half["recipient_score"]})
        cells[cell_id] = {"native_accuracy": accuracy, "expected_score_effect_sign": expected,
                          "effects": effects, "lexical_halves": halves}
    exact_live = (exactness["native_replay_max_absolute_logit_error"] <=
                  bars["maximum_native_replay_absolute_logit_error"] and
                  exactness["source_term_identity_max_absolute_error"] <=
                  bars["maximum_source_term_identity_absolute_error"] and
                  exactness["direct_score_identity_max_absolute_error"] <=
                  bars["maximum_direct_score_identity_absolute_error"] and
                  exactness["installed_term_max_absolute_error"] <=
                  bars["maximum_installed_term_absolute_error"] and
                  exactness["recipient_baseline_duplicate_max_absolute_error"] == 0 and
                  all(math.isfinite(row[key]) for row in evidence
                      for key in ("donor_margin", "donor_ce")))
    joint_live = all(
        min(cell["native_accuracy"].values()) >= bars["minimum_native_accuracy_each_side_each_cell"] and
        abs(cell["effects"]["opposite_joint"]["mean_margin_effect"]) >= bars["minimum_live_absolute_joint_effect"] and
        cell["effects"]["opposite_joint"]["mean_margin_effect"] * cell["expected_score_effect_sign"] > 0 and
        cell["effects"]["opposite_joint"]["mean_ce_effect"] * cell["expected_score_effect_sign"] > 0 and
        cell["effects"]["opposite_joint"]["row_expected_sign_fraction"] >= bars["minimum_joint_row_sign_fraction"]
        for cell in cells.values())
    instrument = exact_live and joint_live
    joint_specific = instrument and all(
        cell["effects"]["opposite_joint"]["same_number_leakage_ratio"] <=
        bars["maximum_same_over_opposite_effect"] for cell in cells.values())

    def branch_specific(name):
        opposite = f"opposite_{name}"
        return instrument and all(
            cell["effects"][opposite]["signed_fraction_of_joint"] >=
                bars["minimum_branch_signed_fraction_of_joint"] and
            cell["effects"][opposite]["row_expected_sign_fraction"] >=
                bars["minimum_branch_row_sign_fraction"] and
            cell["effects"][opposite]["same_number_leakage_ratio"] <=
                bars["maximum_same_over_opposite_effect"]
            for cell in cells.values())
    return {**exactness, "cells": cells, "predictions": {
        "pred_a_instrument_live": instrument,
        "pred_b_joint_natural_score_number_specificity": joint_specific,
        "pred_c_qk1_natural_score_number_specificity": branch_specific("qk1"),
        "pred_d_qk2_natural_score_number_specificity": branch_specific("qk2")}}


def evaluate(model, torch, F, facade, plan):
    rows = build_triples(); count = len(rows); device = next(model.parameters()).device
    pad = qk.parent.atlas.test_atlas._pad
    base_tokens, base_finals = pad(rows, "base_ids", 9, torch, device)
    same_tokens, same_finals = pad(rows, "same_ids", 9, torch, device)
    opposite_tokens, opposite_finals = pad(rows, "opposite_ids", 9, torch, device)
    tokens = torch.cat((base_tokens, same_tokens, opposite_tokens))
    finals = torch.cat((base_finals, same_finals, opposite_finals))
    native = qk.parent.atlas.test_atlas.factor_parent._native_logits(model, tokens, torch, F)
    replay, factors = qk._qk_factor_forward(model, tokens, finals, torch, F, facade)
    base = {name: value[:count] for name, value in factors.items()}
    same = {name: value[count:2*count] for name, value in factors.items()}
    opposite = {name: value[2*count:] for name, value in factors.items()}
    patch = _compile_patch_batch(base_tokens, base_finals, base, same, opposite, rows, torch)
    patched, patched_factors = qk._qk_factor_forward(
        model, patch["tokens"], patch["finals"], torch, F, facade,
        replacement_heads=patch["replacement_heads"])
    exactness = {
        "native_replay_max_absolute_logit_error": float((native - replay).abs().max()),
        "source_term_identity_max_absolute_error": max(float((
            torch.einsum("bk,bkd->bd", side["p"], side["u"]) - side["head"]).abs().max())
            for side in (base, same, opposite, patched_factors)),
        "direct_score_identity_max_absolute_error": max(_direct_score_error(side, torch)
                                                          for side in (base, same, opposite)),
        "installed_term_max_absolute_error": 0.0,
        "recipient_baseline_duplicate_max_absolute_error": 0.0,
    }
    evidence = []
    for output_index, (row_index, condition, cell, group_id) in enumerate(patch["specs"]):
        row = rows[row_index]; q = int(base_finals[row_index])
        native_margin, native_ce = qk.parent.atlas.test_atlas._donor_metrics(replay[row_index], row, q, torch)
        margin, ce = qk.parent.atlas.test_atlas._donor_metrics(patched[output_index], row, q, torch)
        evidence.append({"row_id": row["specificity_row_id"], "group_id": group_id,
            "foreign_group_id": row["foreign_group_id"], "atlas_cell_id": cell,
            "condition": condition, "native_donor_margin": native_margin,
            "donor_margin": margin, "margin_delta": margin-native_margin,
            "native_donor_ce": native_ce, "donor_ce": ce, "donor_ce_gain": native_ce-ce})
        exactness["installed_term_max_absolute_error"] = max(
            exactness["installed_term_max_absolute_error"], float((
                patch["replacement_heads"][output_index] - base["head"][row_index]
                + base["p"][row_index, 8] * base["u"][row_index, 8]
                - patch["installed_scalars"][output_index] * opposite["u"][row_index, 8]
            ).abs().max()))
    capability_rows = [dict(row, donor_ids=row["opposite_ids"]) for row in rows]
    capability = qk.parent.atlas.test_atlas._capability(
        capability_rows, native[:count], native[2*count:], base_finals, opposite_finals)
    return evidence, capability, exactness


def main():
    plan = compile_plan()
    if "--dry-run" in sys.argv or os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    torch, F, facade = qk.parent.atlas.test_atlas.factor_parent._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                            verify_weights_sha256=True)
    with torch.no_grad():
        evidence, capability, exactness = evaluate(model, torch, F, facade, plan)
    scored = score(evidence, capability, exactness, plan["bars"])
    predictions = scored["predictions"]
    terminal = ("invalid" if not predictions["pred_a_instrument_live"] else
                "joint_natural_score_number_specificity" if predictions["pred_b_joint_natural_score_number_specificity"] else
                "natural_qk_number_specificity_null")
    result = {"schema": "task14_head11_3_ood_fronted_natural_qk_number_specificity_result_v1",
              "plan": plan, "prior_art_sha256": PRIOR_ART_SHA256,
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "terminal": terminal, "score": scored, "evidence": evidence,
              "evaluated_splits": ["OOD_TEXT_REUSE_NEW_INTERVENTION"],
              "forbidden_splits_opened": [], "model_forwards": 3}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"result": str(OUT), "terminal": terminal,
                      "predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
