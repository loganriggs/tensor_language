#!/usr/bin/env python3
"""Exact fronted-OOD L11H3 score-role factorial under the final-subject value."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_earlier_score_sufficiency pred_c_self_score_sufficiency pred_d_redundant_score_routes

from __future__ import annotations

from collections import defaultdict
import hashlib
import itertools
import json
import os
import statistics
import sys

import run_task14_head11_3_ood_same_syntax_source_value_atlas as atlas


ROOT = atlas.ROOT
OUT = ROOT / "circuits/fast_screens/task14_head11_3_ood_fronted_score_role_factorial_v2_result.json"
PRIOR_ART_SHA256 = "60db6a5733520724db12185b1ca4149ae1e27715bb817a160f5b135dcbed9a99"
PARENT_RESULT_SHA256 = "e0cdff4a7e71713d3ae8ab58dda72de83ee58930516d13faafbdc279a405ed66"
WEAK_CELL = "fronted_singular_to_fronted_plural"
GROUPS = {"E": tuple(range(7)), "D": (7,), "S": (8,)}
GLOBAL = ("A_native_scores_native_value", "B_donor_scores_native_value",
          "C_native_scores_donor_value", "D_donor_scores_donor_value")


def build_rows():
    rows = [row for row in atlas.build_rows() if row["target_family"] == "A1"]
    if len(rows) != 32 or any(len(row["base_ids"]) != 9 or row["subject_position"] != 8
                              for row in rows):
        raise ValueError("fronted A1 authority changed")
    cells = defaultdict(int)
    for row in rows:
        cells[row["atlas_cell_id"]] += 1
    if len(cells) != 2 or set(cells.values()) != {16}:
        raise ValueError(f"fronted direction cells lost balance: {dict(cells)}")
    return rows


def compile_plan():
    rows = build_rows()
    patched_per_row = 4 + 2 * 9 + 8
    return {
        "schema": "task14_head11_3_ood_fronted_score_role_factorial_plan_v1",
        "candidate_id": "subject_verb.number_agreement.head11_3_ood_fronted_score_role_factorial",
        "split": "OOD_TEXT_REUSE_NEW_INTERVENTION", "screen_tier": "BASIC",
        "row_count": len(rows), "authority_sha256": atlas.test_atlas._canonical(rows),
        "groups": {name: list(positions) for name, positions in GROUPS.items()},
        "design": {
            "global_factorial": "native/donor whole score vector x native/donor final-subject value",
            "score_singletons": "each donor p_k under both native and donor final-subject values",
            "group_factorial": "all eight E,D,S score corners with donor final-subject value",
        },
        "price": {"model_forwards": 3,
                  "example_evaluations": 4 * len(rows) + patched_per_row * len(rows),
                  "backwards": 0, "parameter_updates": 0},
        "outcomes": ["donor_directed_is_are_margin", "donor_answer_ce",
                     "three_factor_mobius", "three_factor_shapley"],
        "bars": {
            "minimum_native_accuracy_each_side_each_cell": .85,
            "maximum_native_replay_absolute_logit_error": 5e-5,
            "maximum_source_term_identity_absolute_error": 5e-5,
            "maximum_pre_subject_value_absolute_error": 5e-5,
            "maximum_native_corner_absolute_logit_error": 7e-5,
            "maximum_complete_head_vector_absolute_error": 5e-5,
            "maximum_group_endpoint_absolute_logit_error": 5e-5,
            "maximum_parent_value_only_reproduction_error": 5e-5,
            "maximum_algebra_closure_error": 1e-10,
            "minimum_complete_head_direction_fraction_each_cell": .75,
            "minimum_directional_score_need_recovery": .30,
            "minimum_score_route_recovery": .70,
            "minimum_score_route_direction_fraction": .75,
        },
        "scope": "OOD_TEXT_REUSE_NEW_INTERVENTION",
        "closed_claims": ["pristine_OOD_confirmation", "QK_branch_identification",
                          "selectivity", "completeness"],
    }


def _head(score, native_u, donor_u, donor_value, torch):
    values = native_u.clone()
    if donor_value:
        values[8] = donor_u[8]
    return torch.einsum("k,kd->d", score, values)


def _compile_patch_batch(tokens, finals, base, donor, rows, torch):
    batch_rows, sources, terms, heads, specs = [], [], [], [], []
    for row_index, _row in enumerate(rows):
        bp, bu = base["p"][row_index], base["u"][row_index]
        dp, du = donor["p"][row_index], donor["u"][row_index]
        native_term = bp[0] * bu[0]

        def add(condition, score, donor_value, kind, position=None, corner=None):
            batch_rows.append(row_index); sources.append(0); terms.append(native_term)
            heads.append(_head(score, bu, du, donor_value, torch))
            specs.append((row_index, condition, kind, position, corner, donor_value))

        add(GLOBAL[0], bp, False, "global")
        add(GLOBAL[1], dp, False, "global")
        add(GLOBAL[2], bp, True, "global")
        add(GLOBAL[3], dp, True, "global")
        for donor_value in (False, True):
            for position in range(9):
                score = bp.clone(); score[position] = dp[position]
                add(f"singleton_p{position}_{'donor' if donor_value else 'native'}_value",
                    score, donor_value, "singleton", position=position)
        for corner in range(8):
            score = bp.clone()
            for bit, name in enumerate(("E", "D", "S")):
                if corner & (1 << bit):
                    score[list(GROUPS[name])] = dp[list(GROUPS[name])]
            add(f"group_corner_{corner:03b}", score, True, "group", corner=corner)
    indices = torch.tensor(batch_rows, dtype=torch.long, device=tokens.device)
    return {
        "tokens": tokens[indices], "finals": finals[indices],
        "source_positions": torch.tensor(sources, dtype=torch.long, device=tokens.device),
        "replacement_terms": torch.stack(terms), "replacement_heads": torch.stack(heads),
        "specs": specs,
    }


def _parent_value_rows():
    if hashlib.sha256(atlas.OUT.read_bytes()).hexdigest() != PARENT_RESULT_SHA256:
        raise RuntimeError("OOD value-atlas parent changed")
    result = json.loads(atlas.OUT.read_text())
    return {row["row_id"]: row for row in result["evidence"]
            if row["condition"] == "joint_all_values" and row["target_family"] == "A1"}


def _mobius(values):
    coefficients = {}
    for subset in range(8):
        total = 0.0
        members = subset.bit_count()
        for inner in range(8):
            if inner & ~subset == 0:
                total += (-1.0 if (members - inner.bit_count()) % 2 else 1.0) * values[inner]
        coefficients[subset] = total
    reconstructed = {subset: sum(coefficients[inner] for inner in range(8)
                                  if inner & ~subset == 0) for subset in range(8)}
    shapley = {bit: sum(value / subset.bit_count() for subset, value in coefficients.items()
                        if subset & (1 << bit)) for bit in range(3)}
    reconstruction_error = max(abs(reconstructed[key] - values[key]) for key in values)
    efficiency_error = abs(sum(shapley.values()) - (values[7] - values[0]))
    return coefficients, shapley, reconstruction_error, efficiency_error


def score(evidence, capability, exactness, bars):
    grouped = defaultdict(list)
    for row in evidence:
        grouped[(row["atlas_cell_id"], row["condition"])].append(row)
    cells = {}
    algebra_error = exactness["global_closure_max_absolute_error"]
    for cell_id, accuracy in capability.items():
        summaries = {}
        for condition in GLOBAL:
            rows = grouped[(cell_id, condition)]
            summaries[condition] = {
                "mean_margin_delta": statistics.fmean(row["margin_delta"] for row in rows),
                "mean_donor_ce_gain": statistics.fmean(row["donor_ce_gain"] for row in rows),
            }
        a, b, c, d = (summaries[name] for name in GLOBAL)
        complete_margin = d["mean_margin_delta"]
        complete_ce = d["mean_donor_ce_gain"]
        complete_direction = sum(row["margin_delta"] > 0 for row in grouped[(cell_id, GLOBAL[3])]) / 16
        all_score_margin = d["mean_margin_delta"] - c["mean_margin_delta"]
        all_score_ce = d["mean_donor_ce_gain"] - c["mean_donor_ce_gain"]
        global_factorial = {
            "corners": summaries,
            "score_main_margin": b["mean_margin_delta"] - a["mean_margin_delta"],
            "value_main_margin": c["mean_margin_delta"] - a["mean_margin_delta"],
            "interaction_margin": d["mean_margin_delta"] - b["mean_margin_delta"]
                - c["mean_margin_delta"] + a["mean_margin_delta"],
            "all_score_conditional_margin": all_score_margin,
            "all_score_conditional_ce": all_score_ce,
            "directional_score_need_recovery": all_score_margin / complete_margin,
            "complete_direction_fraction": complete_direction,
        }
        corner_margin = {corner: statistics.fmean(
            row["margin_delta"] for row in grouped[(cell_id, f"group_corner_{corner:03b}")]
        ) for corner in range(8)}
        corner_ce = {corner: statistics.fmean(
            row["donor_ce_gain"] for row in grouped[(cell_id, f"group_corner_{corner:03b}")]
        ) for corner in range(8)}
        mm, ms, mr, me = _mobius(corner_margin)
        cm, cs, cr, ce = _mobius(corner_ce)
        algebra_error = max(algebra_error, mr, me, cr, ce)

        def route(corner):
            target = grouped[(cell_id, f"group_corner_{corner:03b}")]
            empty = grouped[(cell_id, "group_corner_000")]
            margins = [right["donor_margin"] - left["donor_margin"]
                       for left, right in zip(empty, target)]
            ces = [left["donor_ce"] - right["donor_ce"] for left, right in zip(empty, target)]
            mean_margin = statistics.fmean(margins)
            return {"mean_margin_contribution": mean_margin,
                    "recovery_of_all_score_conditional_margin": mean_margin / all_score_margin
                        if all_score_margin > 0 else None,
                    "direction_fraction": sum(value > 0 for value in margins) / len(margins),
                    "mean_ce_contribution": statistics.fmean(ces)}

        earlier, self_route = route(3), route(4)
        singleton_scores = {}
        for donor_value, baseline_name, full_name in (
            (False, GLOBAL[0], GLOBAL[1]), (True, GLOBAL[2], GLOBAL[3]),
        ):
            state = "donor_value" if donor_value else "native_value"
            baseline_rows = grouped[(cell_id, baseline_name)]
            full_score_effect = (summaries[full_name]["mean_margin_delta"]
                                 - summaries[baseline_name]["mean_margin_delta"])
            singleton_scores[state] = []
            for position in range(9):
                target_rows = grouped[(cell_id, f"singleton_p{position}_{'donor' if donor_value else 'native'}_value")]
                margins = [right["donor_margin"] - left["donor_margin"]
                           for left, right in zip(baseline_rows, target_rows)]
                ces = [left["donor_ce"] - right["donor_ce"]
                       for left, right in zip(baseline_rows, target_rows)]
                mean_margin = statistics.fmean(margins)
                singleton_scores[state].append({
                    "source_position": position,
                    "semantic_group": "E" if position <= 6 else "D" if position == 7 else "S",
                    "mean_margin_contribution": mean_margin,
                    "recovery_of_same_value_all_score_correction": (
                        mean_margin / full_score_effect if abs(full_score_effect) > 1e-12 else None
                    ),
                    "direction_fraction": sum(value > 0 for value in margins) / len(margins),
                    "mean_ce_contribution": statistics.fmean(ces),
                })
        cells[cell_id] = {
            "native_accuracy": accuracy, "global_factorial": global_factorial,
            "earlier_E_plus_D_route": earlier, "self_S_route": self_route,
            "score_singletons": singleton_scores,
            "group_corners": {f"{key:03b}": {"mean_margin_delta": corner_margin[key],
                                              "mean_donor_ce_gain": corner_ce[key]}
                              for key in range(8)},
            "mobius": {"margin": {f"{key:03b}": value for key, value in mm.items()},
                        "ce": {f"{key:03b}": value for key, value in cm.items()}},
            "shapley": {"margin": {name: ms[index] for index, name in enumerate(("E", "D", "S"))},
                        "ce": {name: cs[index] for index, name in enumerate(("E", "D", "S"))}},
        }
    exact_live = all(value <= bars[key] for value, key in (
        (exactness["native_replay_max_absolute_logit_error"], "maximum_native_replay_absolute_logit_error"),
        (exactness["source_term_identity_max_absolute_error"], "maximum_source_term_identity_absolute_error"),
        (exactness["pre_subject_value_max_absolute_error"], "maximum_pre_subject_value_absolute_error"),
        (exactness["native_corner_max_absolute_logit_error"], "maximum_native_corner_absolute_logit_error"),
        (exactness["complete_head_vector_max_absolute_error"], "maximum_complete_head_vector_absolute_error"),
        (exactness["group_endpoint_max_absolute_logit_error"], "maximum_group_endpoint_absolute_logit_error"),
        (exactness["parent_value_only_max_absolute_reproduction_error"], "maximum_parent_value_only_reproduction_error"),
        (algebra_error, "maximum_algebra_closure_error"),
    ))
    capability_live = all(
        min(cell["native_accuracy"].values()) >= bars["minimum_native_accuracy_each_side_each_cell"] and
        cell["global_factorial"]["corners"][GLOBAL[3]]["mean_margin_delta"] > 0 and
        cell["global_factorial"]["corners"][GLOBAL[3]]["mean_donor_ce_gain"] > 0 and
        cell["global_factorial"]["complete_direction_fraction"] >=
        bars["minimum_complete_head_direction_fraction_each_cell"] for cell in cells.values()
    )
    need = cells[WEAK_CELL]["global_factorial"]
    directional_need = (need["all_score_conditional_margin"] > 0 and
                        need["all_score_conditional_ce"] > 0 and
                        need["directional_score_need_recovery"] >=
                        bars["minimum_directional_score_need_recovery"])
    instrument = exact_live and capability_live and directional_need

    def passes(route):
        return (route["recovery_of_all_score_conditional_margin"] is not None and
                route["recovery_of_all_score_conditional_margin"] >=
                bars["minimum_score_route_recovery"] and
                route["direction_fraction"] >= bars["minimum_score_route_direction_fraction"] and
                route["mean_ce_contribution"] > 0)

    earlier = instrument and passes(cells[WEAK_CELL]["earlier_E_plus_D_route"])
    self_sufficient = instrument and passes(cells[WEAK_CELL]["self_S_route"])
    return {
        **exactness, "algebra_max_absolute_error": algebra_error,
        "directional_score_need_reproduced": directional_need,
        "cells": cells,
        "predictions": {
            "pred_a_instrument_live": instrument,
            "pred_b_earlier_score_sufficiency": earlier,
            "pred_c_self_score_sufficiency": self_sufficient,
            "pred_d_redundant_score_routes": earlier and self_sufficient,
        },
    }


def evaluate(model, torch, F, facade, plan):
    rows = build_rows(); count = len(rows)
    device = next(model.parameters()).device
    base_tokens, base_finals = atlas.test_atlas._pad(rows, "base_ids", 9, torch, device)
    donor_tokens, donor_finals = atlas.test_atlas._pad(rows, "donor_ids", 9, torch, device)
    tokens, finals = torch.cat((base_tokens, donor_tokens)), torch.cat((base_finals, donor_finals))
    native = atlas.test_atlas.factor_parent._native_logits(model, tokens, torch, F)
    replay, factors = atlas.test_atlas.factor_parent._factor_forward(
        model, tokens, finals, torch, F, facade,
    )
    base, donor = atlas.test_atlas._split_factors(factors, count)
    patch = _compile_patch_batch(base_tokens, base_finals, base, donor, rows, torch)
    patched, patched_factors = atlas.test_atlas.factor_parent._factor_forward(
        model, patch["tokens"], patch["finals"], torch, F, facade,
        source_positions=patch["source_positions"], replacement_terms=patch["replacement_terms"],
        replacement_heads=patch["replacement_heads"],
    )
    exactness = {
        "native_replay_max_absolute_logit_error": float((replay - native).abs().max()),
        "source_term_identity_max_absolute_error": max(float((
            torch.einsum("bk,bkd->bd", item["p"], item["u"]) - item["head"]
        ).abs().max()) for item in (base, donor, patched_factors)),
        "pre_subject_value_max_absolute_error": float((base["u"][:, :8] - donor["u"][:, :8]).abs().max()),
        "native_corner_max_absolute_logit_error": 0.0,
        "complete_head_vector_max_absolute_error": 0.0,
        "group_endpoint_max_absolute_logit_error": 0.0,
        "parent_value_only_max_absolute_reproduction_error": 0.0,
        "global_closure_max_absolute_error": 0.0,
    }
    evidence, outputs = [], {}
    parent = _parent_value_rows()
    for output_index, spec in enumerate(patch["specs"]):
        row_index, condition, kind, position, corner, donor_value = spec
        row = rows[row_index]; q = int(base_finals[row_index])
        native_margin, native_ce = atlas.test_atlas._donor_metrics(replay[row_index], row, q, torch)
        margin, ce = atlas.test_atlas._donor_metrics(patched[output_index], row, q, torch)
        item = {"row_id": row["row_id"], "atlas_cell_id": row["atlas_cell_id"],
                "condition": condition, "kind": kind, "source_position": position,
                "corner": corner, "donor_value": donor_value,
                "native_donor_margin": native_margin, "donor_margin": margin,
                "margin_delta": margin - native_margin, "native_donor_ce": native_ce,
                "donor_ce": ce, "donor_ce_gain": native_ce - ce}
        evidence.append(item); outputs[(row_index, condition)] = patched[output_index]
        if condition == GLOBAL[2]:
            prior = parent[row["row_id"]]
            exactness["parent_value_only_max_absolute_reproduction_error"] = max(
                exactness["parent_value_only_max_absolute_reproduction_error"],
                *(abs(item[key] - prior[key]) for key in
                  ("native_donor_margin", "donor_margin", "margin_delta", "donor_ce")))
    for row_index in range(count):
        exactness["native_corner_max_absolute_logit_error"] = max(
            exactness["native_corner_max_absolute_logit_error"],
            float((outputs[(row_index, GLOBAL[0])] - replay[row_index]).abs().max()))
        exactness["group_endpoint_max_absolute_logit_error"] = max(
            exactness["group_endpoint_max_absolute_logit_error"],
            float((outputs[(row_index, "group_corner_000")] - outputs[(row_index, GLOBAL[2])]).abs().max()),
            float((outputs[(row_index, "group_corner_111")] - outputs[(row_index, GLOBAL[3])]).abs().max()))
        exactness["complete_head_vector_max_absolute_error"] = max(
            exactness["complete_head_vector_max_absolute_error"],
            float((_head(donor["p"][row_index], base["u"][row_index], donor["u"][row_index], True, torch)
                   - donor["head"][row_index]).abs().max()))
        for metric in ("margin_delta", "donor_ce_gain"):
            vals = {name: next(item[metric] for item in evidence
                               if item["row_id"] == rows[row_index]["row_id"] and item["condition"] == name)
                    for name in GLOBAL}
            closure = (vals[GLOBAL[3]] - vals[GLOBAL[0]]) - (
                (vals[GLOBAL[1]] - vals[GLOBAL[0]]) + (vals[GLOBAL[2]] - vals[GLOBAL[0]]) +
                (vals[GLOBAL[3]] - vals[GLOBAL[1]] - vals[GLOBAL[2]] + vals[GLOBAL[0]]))
            exactness["global_closure_max_absolute_error"] = max(
                exactness["global_closure_max_absolute_error"], abs(closure))
    capability = atlas.test_atlas._capability(
        rows, native[:count], native[count:], base_finals, donor_finals,
    )
    return evidence, capability, exactness


def main():
    plan = compile_plan()
    if "--dry-run" in sys.argv or os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    torch, F, facade = atlas.test_atlas.factor_parent._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                            verify_weights_sha256=True)
    with torch.no_grad():
        evidence, capability, exactness = evaluate(model, torch, F, facade, plan)
    scored = score(evidence, capability, exactness, plan["bars"])
    predictions = scored["predictions"]
    terminal = ("invalid" if not predictions["pred_a_instrument_live"] else
                "redundant_score_routes_screen" if predictions["pred_d_redundant_score_routes"] else
                "earlier_score_sufficiency_screen" if predictions["pred_b_earlier_score_sufficiency"] else
                "self_score_sufficiency_screen" if predictions["pred_c_self_score_sufficiency"] else
                "fronted_score_role_factorial_null")
    result = {"schema": "task14_head11_3_ood_fronted_score_role_factorial_result_v1",
              "plan": plan, "prior_art_sha256": PRIOR_ART_SHA256,
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "terminal": terminal, "score": scored, "evidence": evidence,
              "evaluated_splits": ["OOD_TEXT_REUSE_NEW_INTERVENTION"],
              "forbidden_splits_opened": [], "model_forwards": 3}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"result": str(OUT), "terminal": terminal, "score": scored}, indent=2))


if __name__ == "__main__":
    main()
