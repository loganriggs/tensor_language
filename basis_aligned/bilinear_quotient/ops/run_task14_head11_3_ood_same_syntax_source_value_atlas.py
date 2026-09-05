#!/usr/bin/env python3
"""OOD-text-reuse same-syntax source-value atlas for Task14 L11H3."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_joint_value_sufficiency pred_c_fronted_final_subject_sufficiency pred_d_two_attractor_relative_later_relay

from __future__ import annotations

from collections import defaultdict
import json
import os
import statistics
import sys

import circuit_fast_screen_candidate_task14_ood_cross_syntax as authority
import run_task14_head11_3_same_syntax_source_value_atlas as test_atlas


ROOT = test_atlas.ROOT
OUT = ROOT / "circuits/fast_screens/task14_head11_3_ood_same_syntax_source_value_atlas_v1_result.json"
PRIOR_ART_SHA256 = "ed372fd4186871e126023b021aceeae404b51882feb68d5891c8f5cb0cc8a884"
SPECIAL_CONDITIONS = test_atlas.SPECIAL_CONDITIONS
SUBJECT_POSITION = {"A1": 8, "A2": 1}
ROLE_BY_FAMILY = {
    "A1": ("fronted_preposition", "first_attractor_determiner", "first_attractor",
           "fronted_preposition", "second_attractor_determiner", "second_attractor",
           "punctuation", "subject_determiner", "subject"),
    "A2": ("subject_determiner", "subject", "relative_marker", "relative_pronoun",
           "relative_verb", "preposition", "first_attractor_determiner",
           "first_attractor", "preposition", "second_attractor_determiner",
           "second_attractor"),
}


def build_rows():
    parents = authority.build_rows()
    authority.validate_rows(parents)
    source_rows = {(str(row["group_id"]), str(row["transform_id"])): row
                   for row in authority._CANDIDATE._source_rows()}
    output = []
    for row in parents:
        family = str(row["target_family"])
        source = source_rows[(row["group_id"], family)]
        donor = authority._CANDIDATE._endpoint(source, "donor")
        subject_position = SUBJECT_POSITION[family]
        expected_length = 9 if family == "A1" else 11
        if len(row["base_ids"]) != expected_length or len(donor["ids"]) != expected_length:
            raise ValueError(f"OOD {family} length changed")
        differences = [position for position, pair in
                       enumerate(zip(row["base_ids"], donor["ids"])) if pair[0] != pair[1]]
        if differences != [subject_position]:
            raise ValueError(f"OOD {family} pair differs at {differences}, not subject")
        if donor["subject_number"] == row["base_subject_number"] \
                or donor["answer_id"] != row["base_foil_id"]:
            raise ValueError("same-syntax OOD donor does not reverse subject number and answer")
        construction = "fronted" if family == "A1" else "two_attractor_relative"
        augmented = dict(row)
        augmented.update(
            donor_ids=donor["ids"], donor_text=donor["text"],
            donor_answer_id=donor["answer_id"], donor_foil_id=donor["foil_id"],
            donor_subject_number=donor["subject_number"],
            donor_semantic_position=donor["position"],
            atlas_cell_id=(f"{construction}_{row['base_subject_number']}_to_"
                           f"{construction}_{donor['subject_number']}"),
            subject_position=subject_position,
            source_roles=list(ROLE_BY_FAMILY[family]),
            source_token_text=[test_atlas.task14.ENCODING.decode([token])
                               for token in row["base_ids"]],
            donor_source_token_text=[test_atlas.task14.ENCODING.decode([token])
                                     for token in donor["ids"]],
        )
        output.append(augmented)
    cells = defaultdict(int)
    for row in output:
        cells[row["atlas_cell_id"]] += 1
    if len(output) != 64 or len(cells) != 4 or set(cells.values()) != {16}:
        raise ValueError(f"OOD atlas lost four balanced cells: {dict(cells)}")
    return output


def compile_plan():
    rows = build_rows()
    patched = sum(len(row["base_ids"]) + len(SPECIAL_CONDITIONS) for row in rows)
    return {
        "schema": "task14_head11_3_ood_same_syntax_source_value_atlas_plan_v1",
        "candidate_id": "subject_verb.number_agreement.head11_3_ood_same_syntax_source_value_atlas",
        "split": "OOD_TEXT_REUSE_NEW_INTERVENTION", "screen_tier": "BASIC",
        "row_count": len(rows), "paired_authority_sha256": test_atlas._canonical(rows),
        "design": {
            "A1": "fronted two-attractor prompt, length 9, subject and final query at position 8",
            "A2": "two-attractor relative clause, length 11, subject at position 1",
            "pairing": "same group, same construction and lemma, opposite number; only subject token changes",
        },
        "site": {"layer": test_atlas.factor_parent.LAYER,
                 "head": test_atlas.factor_parent.HEAD,
                 "query": "final_prediction_position",
                 "sources": "every valid position through the final query; padding excluded"},
        "interventions": {
            "single_source_value": "recipient p_k times same-syntax donor u_k",
            "joint_all_values": "sum of recipient p_k times donor u_k over every valid source",
            "complete_head": "complete same-syntax donor head write",
        },
        "price": {"model_forwards": 3,
                  "example_evaluations": 4 * len(rows) + patched,
                  "backwards": 0, "parameter_updates": 0},
        "outcomes": ["donor_directed_is_are_margin", "donor_answer_ce"],
        "bars": {
            "minimum_native_accuracy_each_side_each_cell": .85,
            "maximum_native_replay_absolute_logit_error": 5e-5,
            "maximum_source_term_identity_absolute_error": 5e-5,
            "maximum_native_noop_absolute_logit_error": 5e-5,
            "maximum_fronted_pre_subject_noop_absolute_logit_error": 5e-5,
            "maximum_fronted_subject_joint_absolute_logit_error": 5e-5,
            "minimum_complete_head_direction_fraction_each_cell": .75,
            "minimum_joint_margin_recovery_each_cell": .80,
            "minimum_joint_direction_fraction_each_cell": .75,
            "minimum_fronted_subject_margin_recovery_each_cell": .80,
            "minimum_relative_later_source_margin_recovery": .10,
            "minimum_relative_later_source_direction_fraction": .75,
        },
        "registered_screens": {
            "joint_value_sufficiency": "joint values recover >=.80 of complete-head margin in every cell",
            "fronted_final_subject_sufficiency": (
                "all positions before the changed final subject are exact no-ops; final-subject "
                "singleton reproduces joint and recovers >=.80 in both A1 cells"
            ),
            "two_attractor_relative_later_relay": (
                "each A2 direction cell has a later non-subject singleton with >=.10 recovery, "
                ">=.75 donor direction, and positive CE"
            ),
        },
        "scope": "OOD_TEXT_REUSE_NEW_INTERVENTION",
        "closed_claims": ["pristine_OOD_confirmation", "cross_syntax_generalization",
                          "selectivity", "completeness", "upstream_writer_identity"],
    }


def score(evidence, capability, replay_error, identity_error, noop_error,
          fronted_pre_error, fronted_subject_joint_error, bars):
    grouped = defaultdict(list)
    for item in evidence:
        grouped[(item["atlas_cell_id"], item["condition"], item["source_position"])].append(item)
    cells = {}
    joint_passes, fronted_passes, relative_passes = [], [], []
    for cell_id, accuracy in capability.items():
        complete_rows = grouped[(cell_id, "complete_head", None)]
        complete_margin = statistics.fmean(row["margin_delta"] for row in complete_rows)
        complete_ce = statistics.fmean(row["donor_ce_gain"] for row in complete_rows)
        complete_direction = sum(row["margin_delta"] > 0 for row in complete_rows) / len(complete_rows)
        source_summaries = []
        positions = sorted(key[2] for key in grouped if key[0] == cell_id
                           and key[1] == "single_source_value")
        for position in positions:
            values = grouped[(cell_id, "single_source_value", position)]
            margin = statistics.fmean(row["margin_delta"] for row in values)
            ce = statistics.fmean(row["donor_ce_gain"] for row in values)
            source_summaries.append({
                "source_position": position, "semantic_role": values[0]["semantic_role"],
                "mean_margin_delta": margin,
                "margin_direction_fraction": sum(row["margin_delta"] > 0 for row in values) / len(values),
                "margin_recovery_of_complete_head": margin / complete_margin if complete_margin > 0 else None,
                "mean_donor_ce_gain": ce,
            })
        joint_rows = grouped[(cell_id, "joint_all_values", None)]
        joint_margin = statistics.fmean(row["margin_delta"] for row in joint_rows)
        joint_ce = statistics.fmean(row["donor_ce_gain"] for row in joint_rows)
        joint_direction = sum(row["margin_delta"] > 0 for row in joint_rows) / len(joint_rows)
        joint_recovery = joint_margin / complete_margin if complete_margin > 0 else None
        joint_pass = (joint_recovery is not None and
                      joint_recovery >= bars["minimum_joint_margin_recovery_each_cell"] and
                      joint_direction >= bars["minimum_joint_direction_fraction_each_cell"] and
                      joint_ce > 0)
        joint_passes.append(joint_pass)
        family = complete_rows[0]["target_family"]
        family_screen = None
        if family == "A1":
            subject = next(source for source in source_summaries if source["source_position"] == 8)
            family_screen = (subject["margin_recovery_of_complete_head"] >=
                             bars["minimum_fronted_subject_margin_recovery_each_cell"] and
                             subject["margin_direction_fraction"] >= .75 and
                             subject["mean_donor_ce_gain"] > 0)
            fronted_passes.append(family_screen)
        else:
            eligible = [source for source in source_summaries if source["source_position"] > 1]
            family_screen = any(
                source["margin_recovery_of_complete_head"] >=
                bars["minimum_relative_later_source_margin_recovery"] and
                source["margin_direction_fraction"] >=
                bars["minimum_relative_later_source_direction_fraction"] and
                source["mean_donor_ce_gain"] > 0 for source in eligible
            )
            relative_passes.append(family_screen)
        cells[cell_id] = {
            "row_count": len(complete_rows), "target_family": family,
            "native_accuracy": accuracy,
            "complete_head": {"mean_margin_delta": complete_margin,
                              "margin_direction_fraction": complete_direction,
                              "mean_donor_ce_gain": complete_ce},
            "sources": source_summaries,
            "joint_all_values": {"mean_margin_delta": joint_margin,
                                 "margin_direction_fraction": joint_direction,
                                 "margin_recovery_of_complete_head": joint_recovery,
                                 "mean_donor_ce_gain": joint_ce,
                                 "passed": joint_pass},
            "family_specific_screen_passed": family_screen,
        }
    instrument = (
        replay_error <= bars["maximum_native_replay_absolute_logit_error"] and
        identity_error <= bars["maximum_source_term_identity_absolute_error"] and
        noop_error <= bars["maximum_native_noop_absolute_logit_error"] and
        fronted_pre_error <= bars["maximum_fronted_pre_subject_noop_absolute_logit_error"] and
        fronted_subject_joint_error <= bars["maximum_fronted_subject_joint_absolute_logit_error"] and
        all(min(cell["native_accuracy"].values()) >=
            bars["minimum_native_accuracy_each_side_each_cell"] and
            cell["complete_head"]["mean_margin_delta"] > 0 and
            cell["complete_head"]["mean_donor_ce_gain"] > 0 and
            cell["complete_head"]["margin_direction_fraction"] >=
            bars["minimum_complete_head_direction_fraction_each_cell"]
            for cell in cells.values())
    )
    return {
        "native_replay_max_absolute_logit_error": replay_error,
        "source_term_identity_max_absolute_error": identity_error,
        "native_noop_max_absolute_logit_error": noop_error,
        "fronted_pre_subject_noop_max_absolute_logit_error": fronted_pre_error,
        "fronted_subject_joint_max_absolute_logit_error": fronted_subject_joint_error,
        "cells": cells,
        "predictions": {
            "pred_a_instrument_live": instrument,
            "pred_b_joint_value_sufficiency": instrument and all(joint_passes),
            "pred_c_fronted_final_subject_sufficiency": instrument and all(fronted_passes),
            "pred_d_two_attractor_relative_later_relay": instrument and all(relative_passes),
        },
    }


def evaluate(model, torch, F, facade, plan):
    rows = build_rows()
    device = next(model.parameters()).device
    length = max(len(row["base_ids"]) for row in rows)
    base_tokens, base_finals = test_atlas._pad(rows, "base_ids", length, torch, device)
    donor_tokens, donor_finals = test_atlas._pad(rows, "donor_ids", length, torch, device)
    combined_tokens = torch.cat((base_tokens, donor_tokens))
    combined_finals = torch.cat((base_finals, donor_finals))
    native = test_atlas.factor_parent._native_logits(model, combined_tokens, torch, F)
    replay, factors = test_atlas.factor_parent._factor_forward(
        model, combined_tokens, combined_finals, torch, F, facade,
    )
    count = len(rows)
    base, donor = test_atlas._split_factors(factors, count)
    patch = test_atlas._compile_patch_batch(
        base_tokens, base_finals, base, donor, rows, torch,
    )
    patched, patched_factors = test_atlas.factor_parent._factor_forward(
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
    capability = test_atlas._capability(
        rows, native[:count], native[count:], base_finals, donor_finals,
    )
    evidence, noop_error = [], 0.0
    single_logits, joint_logits = {}, {}
    fronted_pre_error = fronted_subject_joint_error = 0.0
    for patched_index, (row_index, condition, source_position) in enumerate(patch["specs"]):
        row = rows[row_index]
        q = int(base_finals[row_index])
        native_margin, native_ce = test_atlas._donor_metrics(replay[row_index], row, q, torch)
        margin, ce = test_atlas._donor_metrics(patched[patched_index], row, q, torch)
        item = {
            "row_id": row["row_id"], "group_id": row["group_id"],
            "target_family": row["target_family"], "atlas_cell_id": row["atlas_cell_id"],
            "condition": condition, "source_position": source_position,
            "semantic_role": row["source_roles"][source_position]
                if source_position is not None else None,
            "recipient_token_id": row["base_ids"][source_position]
                if source_position is not None else None,
            "donor_token_id": row["donor_ids"][source_position]
                if source_position is not None else None,
            "native_donor_margin": native_margin, "donor_margin": margin,
            "margin_delta": margin - native_margin,
            "native_donor_ce": native_ce, "donor_ce": ce,
            "donor_ce_gain": native_ce - ce,
        }
        evidence.append(item)
        if condition == "native_noop":
            noop_error = max(noop_error,
                             float((patched[patched_index] - replay[row_index]).abs().max()))
        if condition == "single_source_value":
            single_logits[(row_index, source_position)] = patched[patched_index]
            if row["target_family"] == "A1" and source_position < row["subject_position"]:
                fronted_pre_error = max(
                    fronted_pre_error,
                    float((patched[patched_index] - replay[row_index]).abs().max()),
                )
        elif condition == "joint_all_values":
            joint_logits[row_index] = patched[patched_index]
    for row_index, row in enumerate(rows):
        if row["target_family"] == "A1":
            fronted_subject_joint_error = max(
                fronted_subject_joint_error,
                float((single_logits[(row_index, row["subject_position"])]
                       - joint_logits[row_index]).abs().max()),
            )
    return (evidence, capability, replay_error, identity_error, noop_error,
            fronted_pre_error, fronted_subject_joint_error)


def main():
    plan = compile_plan()
    if "--dry-run" in sys.argv or os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    torch, F, facade = test_atlas.factor_parent._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True,
    )
    with torch.no_grad():
        values = evaluate(model, torch, F, facade, plan)
    scored = score(*values, plan["bars"])
    predictions = scored["predictions"]
    if not predictions["pred_a_instrument_live"]:
        terminal = "invalid"
    elif all(predictions.values()):
        terminal = "ood_same_syntax_source_value_atlas_screen"
    else:
        terminal = "ood_same_syntax_source_value_atlas_null"
    result = {
        "schema": "task14_head11_3_ood_same_syntax_source_value_atlas_result_v1",
        "plan": plan, "prior_art_sha256": PRIOR_ART_SHA256,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "terminal": terminal, "score": scored, "evidence": values[0],
        "evaluated_splits": ["OOD_TEXT_REUSE_NEW_INTERVENTION"],
        "forbidden_splits_opened": [], "model_forwards": 3,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"result": str(OUT), "terminal": terminal, "score": scored}, indent=2))


if __name__ == "__main__":
    main()
