#!/usr/bin/env python3
"""Exact 16-corner Q/K/Q2/K2 factorial for Task14's fronted self score."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_qk1_pair_sufficiency pred_c_qk2_pair_sufficiency pred_d_branch_composition_dependence

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
import statistics
import sys

import run_task14_head11_3_ood_fronted_score_role_factorial as parent


ROOT = parent.ROOT
OUT = ROOT / "circuits/fast_screens/task14_head11_3_ood_fronted_self_qk_factorial_v1_result.json"
PRIOR_ART_SHA256 = "b91e8cf7b2df08458d79229093f05669e1be22b7eda0a480940e3df06b28baa7"
PARENT_RESULT_SHA256 = "39269428f6948ceae892721c7f79906e1223b97c6eb0033de0c95be545ac1f8e"
FACTORS = ("q", "k", "q2", "k2")
FULL_MASK = 15
SELF_POSITION = 8


def build_rows():
    rows = parent.build_rows()
    if len(rows) != 32 or any(row["subject_position"] != SELF_POSITION for row in rows):
        raise ValueError("fronted self-QK authority changed")
    return rows


def compile_plan():
    rows = build_rows()
    return {
        "schema": "task14_head11_3_ood_fronted_self_qk_factorial_plan_v1",
        "candidate_id": "subject_verb.number_agreement.head11_3_ood_fronted_self_qk_factorial",
        "split": "OOD_TEXT_REUSE_NEW_INTERVENTION", "screen_tier": "BASIC",
        "row_count": len(rows), "authority_sha256": parent.atlas.test_atlas._canonical(rows),
        "design": "all 16 recipient/donor q,k,q2,k2 corners at final-query self score; donor u8; all other source terms native",
        "factor_bit_order": list(FACTORS),
        "price": {"model_forwards": 3, "example_evaluations": 4 * len(rows) + 16 * len(rows),
                  "backwards": 0, "parameter_updates": 0},
        "outcomes": ["donor_directed_is_are_margin", "donor_answer_ce",
                     "four_factor_mobius", "four_factor_shapley"],
        "bars": {
            "minimum_native_accuracy_each_side_each_cell": .85,
            "maximum_native_replay_absolute_logit_error": 5e-5,
            "maximum_source_term_sum_absolute_error": 5e-5,
            "maximum_pre_subject_value_absolute_error": 5e-5,
            "maximum_endpoint_metric_reproduction_error": 7e-5,
            "maximum_installed_term_absolute_error": 5e-5,
            "maximum_algebra_absolute_error": 1e-10,
            "minimum_factor_or_score_norm": 1e-8,
            "minimum_recipient_donor_factor_or_score_difference": 1e-8,
            "minimum_endpoint_direction_fraction_each_cell": .75,
            "minimum_pair_signed_recovery_each_cell": .70,
            "minimum_pair_row_direction_fraction_each_cell": .75,
            "minimum_branch_interaction_absolute_fraction": .10,
        },
        "scope": "OOD_TEXT_REUSE_NEW_INTERVENTION",
        "closed_claims": ["pristine_OOD_confirmation", "natural_QK_variables",
                          "selectivity", "completeness", "semantic_naming"],
    }


def _self_score(factors, row, mask, torch):
    terms = []
    for bit, name in enumerate(FACTORS):
        source = factors["donor"] if mask & (1 << bit) else factors["base"]
        value = source[name][row]
        if name in ("k", "k2"):
            value = value[SELF_POSITION]
        terms.append(value)
    q, k, q2, k2 = terms
    width = q.shape[-1]
    return (q * k).sum() / width * (q2 * k2).sum() / width


def _compile_patch_batch(tokens, finals, base, donor, rows, torch):
    indices, heads, specs, scalars = [], [], [], []
    factors = {"base": base, "donor": donor}
    for row_index, row in enumerate(rows):
        native_term = base["p"][row_index, SELF_POSITION] * base["u"][row_index, SELF_POSITION]
        for mask in range(16):
            scalar = _self_score(factors, row_index, mask, torch)
            term = scalar * donor["u"][row_index, SELF_POSITION]
            indices.append(row_index)
            heads.append(base["head"][row_index] - native_term + term)
            scalars.append(scalar)
            specs.append((row_index, mask, row["atlas_cell_id"], row["group_id"]))
    index = torch.tensor(indices, device=tokens.device)
    return {"tokens": tokens[index], "finals": finals[index],
            "replacement_heads": torch.stack(heads), "installed_scalars": torch.stack(scalars),
            "specs": specs}


def _qk_factor_forward(model, tokens, finals, torch, F, facade, replacement_heads=None):
    captured = {}
    primitive = parent.atlas.test_atlas.factor_parent.source_factor
    layer = parent.atlas.test_atlas.factor_parent.LAYER
    head = parent.atlas.test_atlas.factor_parent.HEAD

    def attention(event):
        if event.site != layer:
            return event.block.attn(event.state, event.first_value)
        write, factors = primitive.replay_attention_with_source_factors(
            event.state, event.first_value, event.block.attn, finals, head, torch, F,
            include_qk_factors=True,
        )
        captured.update({name: value.detach().clone() for name, value in factors.items()})
        if replacement_heads is not None:
            rows = torch.arange(tokens.shape[0], device=tokens.device)
            write[rows, finals] += (replacement_heads - factors["head"]).to(write.dtype)
        return write, event.first_value

    logits = facade.forward_with_dispatch(
        model, tokens, attention, lambda event: event.block.mlp(event.state),
        require_production=False,
    ).float()
    if set(captured) != {"p", "u", "head", *FACTORS}:
        raise RuntimeError("failed to capture normalized rotary QK factors")
    return logits, captured


def _split(factors, count):
    return ({name: value[:count] for name, value in factors.items()},
            {name: value[count:] for name, value in factors.items()})


def _mobius_shapley(values):
    if set(values) != set(range(16)):
        raise ValueError("four-factor accounting requires all 16 corners")
    mobius = {mask: sum((-1.0 if (mask.bit_count() - sub.bit_count()) % 2 else 1.0)
                        * values[sub] for sub in range(16) if sub & ~mask == 0)
              for mask in range(16)}
    reconstructed = {mask: sum(mobius[sub] for sub in range(16) if sub & ~mask == 0)
                     for mask in range(16)}
    shapley = {}
    for bit, name in enumerate(FACTORS):
        value = 0.0
        for mask in range(16):
            if mask & (1 << bit):
                continue
            size = mask.bit_count()
            value += math.factorial(size) * math.factorial(3 - size) / math.factorial(4) * (
                values[mask | (1 << bit)] - values[mask])
        shapley[name] = value
    return (mobius, shapley,
            max(abs(reconstructed[mask] - values[mask]) for mask in range(16)),
            abs(sum(shapley.values()) - (values[15] - values[0])))


def _parent_endpoints():
    if hashlib.sha256(parent.OUT.read_bytes()).hexdigest() != PARENT_RESULT_SHA256:
        raise RuntimeError("fronted score-role parent changed")
    result = json.loads(parent.OUT.read_text())
    wanted = {"C_native_scores_donor_value", "group_corner_100"}
    return {(row["row_id"], row["condition"]): row for row in result["evidence"]
            if row["condition"] in wanted}


def _sign(value):
    return 1 if value > 0 else -1 if value < 0 else 0


def score(evidence, capability, exactness, liveness, bars):
    grouped = defaultdict(list)
    for row in evidence:
        grouped[(row["atlas_cell_id"], row["mask"])].append(row)
    cells, algebra = {}, 0.0
    for cell_id, accuracy in capability.items():
        margins = {mask: statistics.fmean(row["margin_delta"] for row in grouped[(cell_id, mask)])
                   for mask in range(16)}
        ces = {mask: statistics.fmean(row["donor_ce_gain"] for row in grouped[(cell_id, mask)])
               for mask in range(16)}
        mm, ms, mr, me = _mobius_shapley(margins)
        cm, cs, cr, ce = _mobius_shapley(ces)
        algebra = max(algebra, mr, me, cr, ce)
        total_m, total_ce = margins[15] - margins[0], ces[15] - ces[0]

        def route(mask):
            contributions = [right["donor_margin"] - left["donor_margin"]
                             for left, right in zip(grouped[(cell_id, 0)], grouped[(cell_id, mask)])]
            ce_contributions = [left["donor_ce"] - right["donor_ce"]
                                for left, right in zip(grouped[(cell_id, 0)], grouped[(cell_id, mask)])]
            return {"mean_margin_contribution": statistics.fmean(contributions),
                    "signed_recovery": statistics.fmean(contributions) / total_m,
                    "row_direction_fraction": sum(value * total_m > 0 for value in contributions) / len(contributions),
                    "mean_ce_contribution": statistics.fmean(ce_contributions),
                    "ce_same_sign_as_total": statistics.fmean(ce_contributions) * total_ce > 0}

        interaction = margins[15] - margins[3] - margins[12] + margins[0]
        ordered = sorted({row["group_id"] for row in grouped[(cell_id, 0)]})
        half_signs = []
        for groups in (set(ordered[:8]), set(ordered[8:])):
            half = {mask: statistics.fmean(row["margin_delta"] for row in grouped[(cell_id, mask)]
                                           if row["group_id"] in groups)
                    for mask in (0, 3, 12, 15)}
            half_signs.append(_sign(half[15] - half[3] - half[12] + half[0]))
        endpoint_direction = sum(row["margin_delta"] > 0 for row in grouped[(cell_id, 15)]) / 16
        cells[cell_id] = {
            "native_accuracy": accuracy, "corners": {f"{m:04b}": {
                "mean_margin_delta": margins[m], "mean_donor_ce_gain": ces[m]} for m in range(16)},
            "total": {"margin": total_m, "ce": total_ce,
                      "endpoint_direction_fraction": endpoint_direction},
            "qk1_pair": route(3), "qk2_pair": route(12),
            "branch_interaction": {"margin": interaction,
                "absolute_fraction_of_total": abs(interaction) / abs(total_m),
                "sign": _sign(interaction), "lexical_half_signs": half_signs},
            "mobius_margin": {f"{m:04b}": mm[m] for m in range(16)},
            "mobius_ce": {f"{m:04b}": cm[m] for m in range(16)},
            "shapley_margin": ms, "shapley_ce": cs,
        }
    exact_live = all((exactness[key] <= bars[bar]) for key, bar in (
        ("native_replay_max_absolute_logit_error", "maximum_native_replay_absolute_logit_error"),
        ("source_term_sum_max_absolute_error", "maximum_source_term_sum_absolute_error"),
        ("pre_subject_value_max_absolute_error", "maximum_pre_subject_value_absolute_error"),
        ("endpoint_metric_reproduction_max_absolute_error", "maximum_endpoint_metric_reproduction_error"),
        ("installed_term_max_absolute_error", "maximum_installed_term_absolute_error"),
    )) and algebra <= bars["maximum_algebra_absolute_error"]
    factor_live = (liveness["minimum_factor_or_score_norm"] >= bars["minimum_factor_or_score_norm"] and
                   liveness["minimum_recipient_donor_factor_or_score_difference"] >=
                   bars["minimum_recipient_donor_factor_or_score_difference"] and
                   liveness["all_finite"])
    capability_live = all(min(cell["native_accuracy"].values()) >=
                          bars["minimum_native_accuracy_each_side_each_cell"] and
                          cell["total"]["endpoint_direction_fraction"] >=
                          bars["minimum_endpoint_direction_fraction_each_cell"] and
                          abs(cell["total"]["margin"]) > 1e-12 and abs(cell["total"]["ce"]) > 1e-12
                          for cell in cells.values())
    instrument = exact_live and factor_live and capability_live

    def sufficient(name):
        return instrument and all(cell[name]["signed_recovery"] >=
            bars["minimum_pair_signed_recovery_each_cell"] and
            cell[name]["row_direction_fraction"] >= bars["minimum_pair_row_direction_fraction_each_cell"] and
            cell[name]["ce_same_sign_as_total"] for cell in cells.values())
    signs = [cell["branch_interaction"]["sign"] for cell in cells.values()]
    dependence = instrument and all(cell["branch_interaction"]["absolute_fraction_of_total"] >=
        bars["minimum_branch_interaction_absolute_fraction"] and
        all(sign == cell["branch_interaction"]["sign"] for sign in cell["branch_interaction"]["lexical_half_signs"])
        for cell in cells.values()) and signs[0] != 0 and len(set(signs)) == 1
    return {**exactness, "factor_liveness": liveness, "algebra_max_absolute_error": algebra,
            "cells": cells, "predictions": {
                "pred_a_instrument_live": instrument,
                "pred_b_qk1_pair_sufficiency": sufficient("qk1_pair"),
                "pred_c_qk2_pair_sufficiency": sufficient("qk2_pair"),
                "pred_d_branch_composition_dependence": dependence}}


def evaluate(model, torch, F, facade, plan):
    rows = build_rows(); count = len(rows); device = next(model.parameters()).device
    base_tokens, base_finals = parent.atlas.test_atlas._pad(rows, "base_ids", 9, torch, device)
    donor_tokens, donor_finals = parent.atlas.test_atlas._pad(rows, "donor_ids", 9, torch, device)
    tokens = torch.cat((base_tokens, donor_tokens)); finals = torch.cat((base_finals, donor_finals))
    native = parent.atlas.test_atlas.factor_parent._native_logits(model, tokens, torch, F)
    replay, factors = _qk_factor_forward(model, tokens, finals, torch, F, facade)
    base, donor = _split(factors, count)
    patch = _compile_patch_batch(base_tokens, base_finals, base, donor, rows, torch)
    patched, patched_factors = _qk_factor_forward(
        model, patch["tokens"], patch["finals"], torch, F, facade,
        replacement_heads=patch["replacement_heads"],
    )
    exactness = {
        "native_replay_max_absolute_logit_error": float((native - replay).abs().max()),
        "source_term_sum_max_absolute_error": max(float((torch.einsum("bk,bkd->bd", side["p"], side["u"])
                                                          - side["head"]).abs().max())
                                                     for side in (base, donor, patched_factors)),
        "pre_subject_value_max_absolute_error": float((base["u"][:, :8] - donor["u"][:, :8]).abs().max()),
        "endpoint_metric_reproduction_max_absolute_error": 0.0,
        "installed_term_max_absolute_error": 0.0,
    }
    endpoints = _parent_endpoints(); evidence = []
    for output_index, (row_index, mask, cell, group_id) in enumerate(patch["specs"]):
        row = rows[row_index]; q = int(base_finals[row_index])
        native_margin, native_ce = parent.atlas.test_atlas._donor_metrics(replay[row_index], row, q, torch)
        margin, ce = parent.atlas.test_atlas._donor_metrics(patched[output_index], row, q, torch)
        item = {"row_id": row["row_id"], "group_id": group_id, "atlas_cell_id": cell,
                "mask": mask, "condition": f"corner_{mask:04b}",
                "native_donor_margin": native_margin, "donor_margin": margin,
                "margin_delta": margin - native_margin, "native_donor_ce": native_ce,
                "donor_ce": ce, "donor_ce_gain": native_ce - ce,
                "installed_self_score": float(patch["installed_scalars"][output_index])}
        evidence.append(item)
        exactness["installed_term_max_absolute_error"] = max(
            exactness["installed_term_max_absolute_error"],
            float((patch["replacement_heads"][output_index] - base["head"][row_index]
                   + base["p"][row_index, 8] * base["u"][row_index, 8]
                   - patch["installed_scalars"][output_index] * donor["u"][row_index, 8]).abs().max()))
        if mask in (0, 15):
            condition = "C_native_scores_donor_value" if mask == 0 else "group_corner_100"
            prior = endpoints[(row["row_id"], condition)]
            exactness["endpoint_metric_reproduction_max_absolute_error"] = max(
                exactness["endpoint_metric_reproduction_max_absolute_error"],
                *(abs(item[key] - prior[key]) for key in
                  ("native_donor_margin", "donor_margin", "margin_delta", "native_donor_ce", "donor_ce", "donor_ce_gain")))
    norms, differences = [], []
    all_finite = True
    for row_index in range(count):
        for name in FACTORS:
            left, right = base[name][row_index], donor[name][row_index]
            if name in ("k", "k2"):
                left, right = left[8], right[8]
            norms.extend((float(left.norm()), float(right.norm())))
            differences.append(float((left - right).norm()))
            all_finite &= bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        for names in (("q", "k"), ("q2", "k2")):
            scores = []
            for side in (base, donor):
                scores.append((side[names[0]][row_index] * side[names[1]][row_index, 8]).sum()
                              / side[names[0]].shape[-1])
            norms.extend(abs(float(value)) for value in scores)
            differences.append(abs(float(scores[0] - scores[1])))
            all_finite &= all(math.isfinite(float(value)) for value in scores)
    liveness = {"minimum_factor_or_score_norm": min(norms),
                "minimum_recipient_donor_factor_or_score_difference": min(differences),
                "all_finite": all_finite}
    capability = parent.atlas.test_atlas._capability(
        rows, native[:count], native[count:], base_finals, donor_finals)
    return evidence, capability, exactness, liveness


def main():
    plan = compile_plan()
    if "--dry-run" in sys.argv or os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    torch, F, facade = parent.atlas.test_atlas.factor_parent._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                            verify_weights_sha256=True)
    with torch.no_grad():
        evidence, capability, exactness, liveness = evaluate(model, torch, F, facade, plan)
    scored = score(evidence, capability, exactness, liveness, plan["bars"])
    predictions = scored["predictions"]
    terminal = ("invalid" if not predictions["pred_a_instrument_live"] else
                "both_qk_pairs_sufficient" if predictions["pred_b_qk1_pair_sufficiency"] and predictions["pred_c_qk2_pair_sufficiency"] else
                "qk1_pair_sufficient" if predictions["pred_b_qk1_pair_sufficiency"] else
                "qk2_pair_sufficient" if predictions["pred_c_qk2_pair_sufficiency"] else
                "branch_composition_dependence" if predictions["pred_d_branch_composition_dependence"] else
                "fronted_self_qk_factorial_null")
    result = {"schema": "task14_head11_3_ood_fronted_self_qk_factorial_result_v1",
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
