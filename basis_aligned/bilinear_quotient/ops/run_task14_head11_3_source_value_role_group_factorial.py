#!/usr/bin/env python3
"""Exact four-group source-value factorial for Task14 L11H3."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_SI_sufficient_all_cells pred_c_bridge_repairs_failed_SI_cells pred_d_SxI_interaction_shared

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

import run_task14_head11_3_same_syntax_source_value_atlas as atlas


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits/fast_screens/task14_head11_3_source_value_role_group_factorial_v1_result.json"
PRIOR_ART_SHA256 = "dac329efb08006e4ab7f99f757fce0516c02f8b9097fc5cac4b9bd31d2794c47"
ATLAS_RESULT_SHA256 = "6c76345138e4b8e727bc0bbbfd37fa0a09b156c51c8082c3c227e8b043b5341f"
GROUPS = ("S", "I", "B", "A")
ALL_MASK = (1 << len(GROUPS)) - 1


def _group_positions(row):
    if row["target_family"] == "A1":
        groups = {"S": (1,), "I": (2,), "B": (3, 4), "A": (5, 6)}
    elif row["target_family"] == "A2":
        groups = {"S": (1,), "I": (2,), "B": (3, 4, 5), "A": (6, 7)}
    else:
        raise ValueError("factorial only accepts A1/A2 recipient syntaxes")
    positions = [position for group in GROUPS for position in groups[group]]
    if sorted(positions) != list(range(1, len(row["base_ids"]))) \
            or len(positions) != len(set(positions)):
        raise ValueError("source groups are not a disjoint exhaustive partition of positions 1..final")
    return groups


def _subset_label(mask):
    members = "".join(group for index, group in enumerate(GROUPS) if mask & (1 << index))
    return members or "empty"


def compile_plan():
    rows = atlas.build_rows()
    for row in rows:
        _group_positions(row)
    return {
        "schema": "task14_head11_3_source_value_role_group_factorial_plan_v1",
        "candidate_id": "subject_verb.number_agreement.head11_3_source_value_role_group_factorial",
        "split": "TEST_REUSE_NEW_INTERVENTION", "screen_tier": "BASIC",
        "row_count": len(rows), "paired_authority_sha256": atlas._canonical(rows),
        "groups": {"S": {"pp": [1], "relative": [1]},
                   "I": {"pp": [2], "relative": [2]},
                   "B": {"pp": [3, 4], "relative": [3, 4, 5]},
                   "A": {"pp": [5, 6], "relative": [6, 7]}},
        "conditions": [_subset_label(mask) for mask in range(16)] + ["complete_head"],
        "intervention": "recipient scores times donor/native values selected by semantic group",
        "price": {"model_forwards": 3, "example_evaluations": 1344,
                  "backwards": 0, "parameter_updates": 0},
        "outcomes": ["donor_directed_is_are_margin_primary", "donor_answer_ce_support",
                     "exact_margin_mobius_dividends", "margin_shapley_by_group"],
        "bars": {
            "minimum_native_accuracy_each_side_each_cell": .85,
            "maximum_native_replay_absolute_logit_error": 5e-5,
            "maximum_source_term_identity_absolute_error": 5e-5,
            "maximum_empty_subset_absolute_logit_error": 5e-5,
            "maximum_atlas_full_subset_reproduction_error": 5e-5,
            "maximum_atlas_complete_head_reproduction_error": 5e-5,
            "maximum_mobius_reconstruction_error": 1e-8,
            "maximum_shapley_efficiency_error": 1e-8,
            "minimum_complete_head_direction_fraction_each_cell": .75,
            "minimum_SI_recovery_of_joint_each_cell": .70,
            "minimum_SIB_recovery_of_joint_each_cell": .80,
            "minimum_conditional_bridge_fraction_of_joint": .10,
            "minimum_conditional_bridge_row_direction_fraction": .75,
            "minimum_SxI_dividend_fraction_of_joint_each_cell": .10,
        },
        "closed_claims": ["OOD", "selectivity", "completeness", "upstream_writer_identity"],
        "limits": "Möbius terms describe task nonadditivity for this intervention family, not ontology labels.",
    }


def _compile_patch_batch(base_tokens, base_finals, base, donor, rows, torch):
    batch_rows, source_positions, replacement_terms, replacement_heads, specs = [], [], [], [], []
    for row_index, row in enumerate(rows):
        q = int(base_finals[row_index])
        groups = _group_positions(row)
        native_term_zero = base["p"][row_index, 0] * base["u"][row_index, 0]
        for mask in range(16):
            selected = {position for group_index, group in enumerate(GROUPS)
                        if mask & (1 << group_index) for position in groups[group]}
            values = torch.stack([
                donor["u"][row_index, position] if position in selected
                else base["u"][row_index, position]
                for position in range(q + 1)
            ])
            head = torch.einsum("k,kd->d", base["p"][row_index, :q + 1], values)
            batch_rows.append(row_index)
            source_positions.append(0)
            replacement_terms.append(native_term_zero)
            replacement_heads.append(head)
            specs.append((row_index, "subset", mask))
        batch_rows.append(row_index)
        source_positions.append(0)
        replacement_terms.append(native_term_zero)
        replacement_heads.append(donor["head"][row_index])
        specs.append((row_index, "complete_head", None))
    indices = torch.tensor(batch_rows, dtype=torch.long, device=base_tokens.device)
    return {
        "tokens": base_tokens[indices], "finals": base_finals[indices],
        "source_positions": torch.tensor(source_positions, dtype=torch.long,
                                         device=base_tokens.device),
        "replacement_terms": torch.stack(replacement_terms),
        "replacement_heads": torch.stack(replacement_heads), "specs": specs,
    }


def _mobius_and_shapley(values):
    if set(values) != set(range(16)):
        raise ValueError("Möbius transform requires all sixteen subset values")
    dividends = {}
    for mask in range(16):
        dividends[mask] = sum(
            (-1) ** ((mask.bit_count() - submask.bit_count())) * values[submask]
            for submask in range(16) if submask & ~mask == 0
        )
    shapley = {}
    factorial = math.factorial
    for group_index, group in enumerate(GROUPS):
        bit = 1 << group_index
        contribution = 0.0
        for mask in range(16):
            if mask & bit:
                continue
            size = mask.bit_count()
            weight = factorial(size) * factorial(3 - size) / factorial(4)
            contribution += weight * (values[mask | bit] - values[mask])
        shapley[group] = contribution
    return dividends, shapley


def _atlas_reproduction_evidence():
    path = atlas.OUT
    if hashlib.sha256(path.read_bytes()).hexdigest() != ATLAS_RESULT_SHA256:
        raise RuntimeError("source-value atlas result changed")
    result = json.loads(path.read_text())
    return {
        condition: {row["row_id"]: row for row in result["evidence"]
                    if row["condition"] == condition}
        for condition in ("joint_all_values", "complete_head")
    }


def _donor_metrics(logits, row, q, torch):
    donor, recipient = int(row["donor_answer_id"]), int(row["base_answer_id"])
    return (float(logits[q, donor] - logits[q, recipient]),
            float(-torch.log_softmax(logits[q], dim=-1)[donor]))


def score(evidence, capability, replay_error, identity_error, empty_error,
          atlas_joint_error, atlas_complete_error, bars):
    grouped = defaultdict(list)
    for item in evidence:
        grouped[(item["atlas_cell_id"], item["subset_mask"], item["condition"])].append(item)
    cells = {}
    mobius_error = shapley_error = 0.0
    for cell_id, accuracy in capability.items():
        complete_rows = grouped[(cell_id, None, "complete_head")]
        complete_margin = statistics.fmean(row["margin_delta"] for row in complete_rows)
        complete_ce = statistics.fmean(row["donor_ce_gain"] for row in complete_rows)
        complete_direction = sum(row["margin_delta"] > 0 for row in complete_rows) / len(complete_rows)
        raw_margin, raw_ce, subsets = {}, {}, {}
        for mask in range(16):
            values = grouped[(cell_id, mask, "subset")]
            margin = statistics.fmean(row["margin_delta"] for row in values)
            ce = statistics.fmean(row["donor_ce_gain"] for row in values)
            raw_margin[mask] = margin
            raw_ce[mask] = ce
            subsets[_subset_label(mask)] = {
                "subset_mask": mask, "groups": [GROUPS[index] for index in range(4) if mask & (1 << index)],
                "mean_donor_margin": statistics.fmean(row["donor_margin"] for row in values),
                "mean_donor_ce": statistics.fmean(row["donor_ce"] for row in values),
                "mean_margin_delta": margin,
                "margin_direction_fraction": sum(row["margin_delta"] > 0 for row in values) / len(values),
                "mean_donor_ce_gain": ce,
            }
        game_margin = {mask: raw_margin[mask] - raw_margin[0] for mask in range(16)}
        game_ce = {mask: raw_ce[mask] - raw_ce[0] for mask in range(16)}
        joint_margin, joint_ce = game_margin[ALL_MASK], game_ce[ALL_MASK]
        for mask in range(16):
            subset = subsets[_subset_label(mask)]
            subset["game_margin_from_empty"] = game_margin[mask]
            subset["game_ce_gain_from_empty"] = game_ce[mask]
            subset["margin_recovery_of_joint_values"] = \
                game_margin[mask] / joint_margin if joint_margin > 0 else None
            subset["margin_recovery_of_complete_head"] = \
                game_margin[mask] / complete_margin if complete_margin > 0 else None
            subset["ce_recovery_of_joint_values"] = \
                game_ce[mask] / joint_ce if joint_ce > 0 else None
            subset["ce_recovery_of_complete_head"] = \
                game_ce[mask] / complete_ce if complete_ce > 0 else None
        dividends, shapley = _mobius_and_shapley(game_margin)
        cell_mobius_error = max(abs(sum(dividends[submask] for submask in range(16)
                                            if submask & ~mask == 0) - game_margin[mask])
                                for mask in range(16))
        cell_shapley_error = abs(sum(shapley.values()) - joint_margin)
        mobius_error = max(mobius_error, cell_mobius_error)
        shapley_error = max(shapley_error, cell_shapley_error)
        dividend_rows = []
        for mask, value in dividends.items():
            fraction = abs(value) / abs(joint_margin) if abs(joint_margin) > 1e-12 else None
            dividend_rows.append({"subset": _subset_label(mask), "subset_mask": mask,
                                  "order": mask.bit_count(), "margin_dividend": value,
                                  "absolute_fraction_of_joint_margin": fraction})
        si = (1 << 0) | (1 << 1)
        sib = si | (1 << 2)
        ba = (1 << 2) | (1 << 3)
        si_recovery = game_margin[si] / joint_margin if joint_margin > 0 else None
        sib_recovery = game_margin[sib] / joint_margin if joint_margin > 0 else None
        bridge_fraction = (game_margin[sib] - game_margin[si]) / joint_margin \
            if joint_margin > 0 else None
        bridge_ce = game_ce[sib] - game_ce[si]
        by_row = defaultdict(dict)
        for mask in (si, sib):
            for row in grouped[(cell_id, mask, "subset")]:
                by_row[row["row_id"]][mask] = row["margin_delta"]
        bridge_direction = sum(values[sib] - values[si] > 0 for values in by_row.values()) / len(by_row)
        attractor_fraction = (game_margin[ALL_MASK] - game_margin[sib]) / joint_margin \
            if joint_margin > 0 else None
        high_order = [row for row in dividend_rows if row["order"] >= 2]
        largest_high_order = max(high_order, key=lambda row: row["absolute_fraction_of_joint_margin"])
        half_interactions = []
        ordered_groups = sorted({row["group_id"] for row in grouped[(cell_id, 0, "subset")]})
        for half_index, half_groups in enumerate((set(ordered_groups[:8]), set(ordered_groups[8:]))):
            half_game = {}
            for mask in range(16):
                values = [row for row in grouped[(cell_id, mask, "subset")]
                          if row["group_id"] in half_groups]
                half_game[mask] = statistics.fmean(row["margin_delta"] for row in values)
            empty = half_game[0]
            half_game = {mask: value - empty for mask, value in half_game.items()}
            half_dividends, _unused = _mobius_and_shapley(half_game)
            half_interactions.append({"half": half_index, "group_ids": sorted(half_groups),
                                      "SxI_dividend": half_dividends[si],
                                      "sign": 1 if half_dividends[si] > 0 else
                                              (-1 if half_dividends[si] < 0 else 0)})
        cells[cell_id] = {
            "row_count": len(complete_rows), "native_accuracy": accuracy,
            "complete_head": {"mean_margin_delta": complete_margin,
                              "margin_direction_fraction": complete_direction,
                              "mean_donor_ce_gain": complete_ce},
            "subsets": subsets,
            "summary": {"SI_recovery_of_joint": si_recovery,
                        "SIB_recovery_of_joint": sib_recovery,
                        "conditional_B_fraction_of_joint": bridge_fraction,
                        "conditional_B_ce_gain": bridge_ce,
                        "conditional_B_row_direction_fraction": bridge_direction,
                        "A_addition_fraction_of_joint": attractor_fraction,
                        "SI_shapley_fraction_of_joint":
                            (shapley["S"] + shapley["I"]) / joint_margin,
                        "SI_addition_to_BA_fraction_of_joint":
                            (game_margin[ALL_MASK] - game_margin[ba]) / joint_margin,
                        "largest_high_order_dividend": largest_high_order},
            "joint_values": {"game_margin": joint_margin, "game_ce_gain": joint_ce},
            "margin_mobius_dividends": dividend_rows,
            "mobius_reconstruction_max_absolute_error": cell_mobius_error,
            "shapley_efficiency_absolute_error": cell_shapley_error,
            "SxI_interaction": {"margin_dividend": dividends[si],
                               "absolute_fraction_of_joint_margin": abs(dividends[si]) / abs(joint_margin),
                               "sign": 1 if dividends[si] > 0 else (-1 if dividends[si] < 0 else 0),
                               "lexical_halves": half_interactions},
            "margin_shapley": {group: {"value": value,
                                        "fraction_of_joint_margin": value / joint_margin
                                        if abs(joint_margin) > 1e-12 else None}
                               for group, value in shapley.items()},
        }
    instrument = (
        replay_error <= bars["maximum_native_replay_absolute_logit_error"] and
        identity_error <= bars["maximum_source_term_identity_absolute_error"] and
        empty_error <= bars["maximum_empty_subset_absolute_logit_error"] and
        atlas_joint_error <= bars["maximum_atlas_full_subset_reproduction_error"] and
        atlas_complete_error <= bars["maximum_atlas_complete_head_reproduction_error"] and
        mobius_error <= bars["maximum_mobius_reconstruction_error"] and
        shapley_error <= bars["maximum_shapley_efficiency_error"] and
        all(min(cell["native_accuracy"].values()) >=
            bars["minimum_native_accuracy_each_side_each_cell"] and
            cell["complete_head"]["mean_margin_delta"] > 0 and
            cell["complete_head"]["mean_donor_ce_gain"] > 0 and
            cell["complete_head"]["margin_direction_fraction"] >=
            bars["minimum_complete_head_direction_fraction_each_cell"] and
            cell["joint_values"]["game_margin"] > 0
            for cell in cells.values())
    )
    si_sufficient = instrument and all(
        cell["summary"]["SI_recovery_of_joint"] >=
        bars["minimum_SI_recovery_of_joint_each_cell"] for cell in cells.values())
    failed_si = [cell for cell in cells.values() if cell["summary"]["SI_recovery_of_joint"] <
                 bars["minimum_SI_recovery_of_joint_each_cell"]]
    bridge_repairs = instrument and bool(failed_si) and all(
        cell["summary"]["SIB_recovery_of_joint"] >=
            bars["minimum_SIB_recovery_of_joint_each_cell"] and
        cell["summary"]["conditional_B_fraction_of_joint"] >=
            bars["minimum_conditional_bridge_fraction_of_joint"] and
        cell["summary"]["conditional_B_ce_gain"] > 0 and
        cell["summary"]["conditional_B_row_direction_fraction"] >=
            bars["minimum_conditional_bridge_row_direction_fraction"]
        for cell in failed_si)
    interaction_signs = {cell["SxI_interaction"]["sign"] for cell in cells.values()}
    shared_sxi = instrument and len(interaction_signs) == 1 and 0 not in interaction_signs and all(
        cell["SxI_interaction"]["absolute_fraction_of_joint_margin"] >=
            bars["minimum_SxI_dividend_fraction_of_joint_each_cell"] and
        all(half["sign"] == cell["SxI_interaction"]["sign"]
            for half in cell["SxI_interaction"]["lexical_halves"])
        for cell in cells.values())
    return {
        "native_replay_max_absolute_logit_error": replay_error,
        "source_term_identity_max_absolute_error": identity_error,
        "empty_subset_max_absolute_logit_error": empty_error,
        "atlas_full_subset_max_absolute_reproduction_error": atlas_joint_error,
        "atlas_complete_head_max_absolute_reproduction_error": atlas_complete_error,
        "mobius_reconstruction_max_absolute_error": mobius_error,
        "shapley_efficiency_max_absolute_error": shapley_error,
        "cells": cells,
        "predictions": {
            "pred_a_instrument_live": instrument,
            "pred_b_SI_sufficient_all_cells": si_sufficient,
            "pred_c_bridge_repairs_failed_SI_cells": bridge_repairs,
            "pred_d_SxI_interaction_shared": shared_sxi,
        },
    }


def evaluate(model, torch, F, facade, plan):
    rows = atlas.build_rows()
    device = next(model.parameters()).device
    length = max(len(row["base_ids"]) for row in rows)
    base_tokens, base_finals = atlas._pad(rows, "base_ids", length, torch, device)
    donor_tokens, donor_finals = atlas._pad(rows, "donor_ids", length, torch, device)
    combined_tokens = torch.cat((base_tokens, donor_tokens))
    combined_finals = torch.cat((base_finals, donor_finals))
    native = atlas.factor_parent._native_logits(model, combined_tokens, torch, F)
    replay, factors = atlas.factor_parent._factor_forward(
        model, combined_tokens, combined_finals, torch, F, facade,
    )
    count = len(rows)
    base, donor = atlas._split_factors(factors, count)
    patch = _compile_patch_batch(base_tokens, base_finals, base, donor, rows, torch)
    patched, patched_factors = atlas.factor_parent._factor_forward(
        model, patch["tokens"], patch["finals"], torch, F, facade,
        source_positions=patch["source_positions"],
        replacement_terms=patch["replacement_terms"],
        replacement_heads=patch["replacement_heads"],
    )
    replay_error = float((replay - native).abs().max())
    identity_error = max(
        float((torch.einsum("bk,bkd->bd", item["p"], item["u"])-item["head"]).abs().max())
        for item in (base, donor, patched_factors)
    )
    capability = atlas._capability(
        rows, native[:count], native[count:], base_finals, donor_finals,
    )
    atlas_reproduction = _atlas_reproduction_evidence()
    evidence, empty_error, atlas_joint_error, atlas_complete_error = [], 0.0, 0.0, 0.0
    for patched_index, (row_index, condition, mask) in enumerate(patch["specs"]):
        row = rows[row_index]
        q = int(base_finals[row_index])
        native_margin, native_ce = _donor_metrics(replay[row_index], row, q, torch)
        margin, ce = _donor_metrics(patched[patched_index], row, q, torch)
        item = {"row_id": row["row_id"], "group_id": row["group_id"],
                "target_family": row["target_family"],
                "atlas_cell_id": row["atlas_cell_id"],
                "condition": condition, "subset_mask": mask,
                "subset": _subset_label(mask) if mask is not None else None,
                "native_donor_margin": native_margin, "donor_margin": margin,
                "margin_delta": margin - native_margin,
                "native_donor_ce": native_ce, "donor_ce": ce,
                "donor_ce_gain": native_ce - ce}
        evidence.append(item)
        if condition == "subset" and mask == 0:
            empty_error = max(empty_error,
                              float((patched[patched_index] - replay[row_index]).abs().max()))
        if condition == "subset" and mask == ALL_MASK:
            parent = atlas_reproduction["joint_all_values"][row["row_id"]]
            atlas_joint_error = max(atlas_joint_error,
                                    abs(item["margin_delta"] - parent["margin_delta"]),
                                    abs(item["donor_ce"] - parent["donor_ce"]),
                                    abs(item["native_donor_ce"] - parent["native_donor_ce"]))
        if condition == "complete_head":
            parent = atlas_reproduction["complete_head"][row["row_id"]]
            atlas_complete_error = max(atlas_complete_error,
                                       abs(item["margin_delta"] - parent["margin_delta"]),
                                       abs(item["donor_ce"] - parent["donor_ce"]),
                                       abs(item["native_donor_ce"] - parent["native_donor_ce"]))
    return (evidence, capability, replay_error, identity_error, empty_error,
            atlas_joint_error, atlas_complete_error)


def main():
    plan = compile_plan()
    if "--dry-run" in sys.argv or os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    torch, F, facade = atlas.factor_parent._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True,
    )
    with torch.no_grad():
        values = evaluate(model, torch, F, facade, plan)
    scored = score(*values, plan["bars"])
    if not scored["predictions"]["pred_a_instrument_live"]:
        terminal = "invalid"
    else:
        terminal = "role_group_factorial_screen"
    result = {
        "schema": "task14_head11_3_source_value_role_group_factorial_result_v1",
        "plan": plan, "prior_art_sha256": PRIOR_ART_SHA256,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "terminal": terminal, "score": scored, "evidence": values[0],
        "evaluated_splits": ["TEST_REUSE_NEW_INTERVENTION"],
        "forbidden_splits_opened": [], "model_forwards": 3,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"result": str(OUT), "terminal": terminal, "score": scored}, indent=2))


if __name__ == "__main__":
    main()
