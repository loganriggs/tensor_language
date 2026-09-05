#!/usr/bin/env python3
"""Exact all-source value-interchange atlas for Task14 L11H3."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_non_subject_source_signal pred_c_subject_only_among_single_sources

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

import circuit_battery_task14 as task14
import circuit_fast_screen_candidate_task14_test_cross_syntax as authority
import run_task14_head11_3_subject_attractor_score_payload_factorial as factor_parent


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits/fast_screens/task14_head11_3_same_syntax_source_value_atlas_v1_result.json"
PRIOR_ART_SHA256 = "1a22ec9bd2c46144742bdc7780996c4b0a30f6293a1b445db1c97e8cbc02bce4"
SUBJECT_PARENT_SHA256 = "e6f5e469cd561e5ee5f2b70031ab9d754ec40b51af0b2d7dfcc61b3c8e207603"
SPECIAL_CONDITIONS = ("native_noop", "joint_all_values", "complete_head")
ROLE_BY_FAMILY = {
    "A1": ("determiner", "subject", "preposition", "preposition", "preposition",
           "attractor_determiner", "attractor"),
    "A2": ("determiner", "subject", "relative_marker", "relative_pronoun",
           "relative_verb", "preposition", "attractor_determiner", "attractor"),
}


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def build_rows():
    parent_rows = authority.build_rows()
    authority.validate_rows(parent_rows)
    source_rows = {(str(row["group_id"]), str(row["transform_id"])): row
                   for row in authority._CANDIDATE._source_rows()}
    output = []
    for row in parent_rows:
        source = source_rows[(row["group_id"], row["target_family"])]
        donor = authority._CANDIDATE._endpoint(source, "donor")
        family = str(row["target_family"])
        roles = ROLE_BY_FAMILY[family]
        if len(row["base_ids"]) != len(donor["ids"]) or len(roles) != len(row["base_ids"]):
            raise ValueError("same-syntax source alignment changed")
        if any(left != right for position, (left, right) in
               enumerate(zip(row["base_ids"], donor["ids"])) if position != 1):
            raise ValueError("paired prompts differ outside subject token 1")
        if row["base_ids"][1] == donor["ids"][1] \
                or row["base_subject_number"] == donor["subject_number"]:
            raise ValueError("paired donor does not reverse subject number")
        syntax = "pp" if family == "A1" else "relative"
        augmented = dict(row)
        augmented.update(
            donor_ids=donor["ids"], donor_text=donor["text"],
            donor_answer_id=donor["answer_id"], donor_foil_id=donor["foil_id"],
            donor_subject_number=donor["subject_number"],
            donor_semantic_position=donor["position"],
            atlas_cell_id=f"{syntax}_{row['base_subject_number']}_to_{syntax}_{donor['subject_number']}",
            source_roles=list(roles),
            source_token_text=[task14.ENCODING.decode([token]) for token in row["base_ids"]],
            donor_source_token_text=[task14.ENCODING.decode([token]) for token in donor["ids"]],
        )
        output.append(augmented)
    cells = defaultdict(int)
    for row in output:
        cells[row["atlas_cell_id"]] += 1
    if len(output) != 64 or len(cells) != 4 or set(cells.values()) != {16}:
        raise ValueError(f"same-syntax atlas lost four balanced cells: {dict(cells)}")
    return output


def compile_plan():
    rows = build_rows()
    patched_examples = sum(len(row["base_ids"]) + len(SPECIAL_CONDITIONS) for row in rows)
    return {
        "schema": "task14_head11_3_same_syntax_source_value_atlas_plan_v1",
        "candidate_id": "subject_verb.number_agreement.head11_3_same_syntax_source_value_atlas",
        "split": "TEST_REUSE_NEW_INTERVENTION", "screen_tier": "BASIC",
        "row_count": len(rows), "paired_authority_sha256": _canonical(rows),
        "design": "32 lexical groups crossed with two recipient syntaxes; not 64 independent noun contrasts",
        "site": {"layer": factor_parent.LAYER, "head": factor_parent.HEAD,
                 "query": "final_prediction_position",
                 "sources": "every_causally_available_position"},
        "interventions": {
            "single_source_value": "recipient p_k times same-syntax donor u_k",
            "native_noop": "reinstall one native source term and native head",
            "joint_all_values": "sum over recipient p_k times donor u_k",
            "complete_head": "complete same-syntax donor head write",
        },
        "price": {"model_forwards": 3,
                  "example_evaluations": 4 * len(rows) + patched_examples,
                  "backwards": 0, "parameter_updates": 0},
        "outcomes": ["donor_directed_is_are_margin", "donor_answer_ce",
                     "joint_minus_sum_of_single_source_effects"],
        "bars": {
            "minimum_native_accuracy_each_side_each_cell": .85,
            "maximum_native_replay_absolute_logit_error": 5e-5,
            "maximum_source_term_identity_absolute_error": 5e-5,
            "maximum_native_noop_absolute_logit_error": 5e-5,
            "maximum_source_zero_absolute_logit_error": 5e-5,
            "maximum_subject_parent_reproduction_error": 5e-5,
            "minimum_complete_head_direction_fraction_each_cell": .75,
            "minimum_descriptive_source_margin_recovery": .10,
            "minimum_descriptive_source_direction_fraction": .75,
        },
        "scope": "TEST_REUSE_NEW_INTERVENTION",
        "closed_claims": ["OOD", "selectivity", "completeness"],
        "limits": "TEST text is reused. Source positions are exact summands, not yet complete circuits.",
    }


def _pad(rows, key, length, torch, device):
    tokens = torch.full((len(rows), length), 50256, dtype=torch.long, device=device)
    finals = []
    for index, row in enumerate(rows):
        ids = row[key]
        tokens[index, :len(ids)] = torch.tensor(ids, device=device)
        finals.append(len(ids) - 1)
    return tokens, torch.tensor(finals, device=device)


def _split_factors(factors, count):
    return ({key: value[:count] for key, value in factors.items()},
            {key: value[count:] for key, value in factors.items()})


def _compile_patch_batch(base_tokens, base_finals, base, donor, rows, torch):
    batch_rows, source_positions, terms, heads, specs = [], [], [], [], []
    for row_index, row in enumerate(rows):
        q = int(base_finals[row_index])
        for source_position in range(q + 1):
            batch_rows.append(row_index)
            source_positions.append(source_position)
            terms.append(base["p"][row_index, source_position] * donor["u"][row_index, source_position])
            heads.append(base["head"][row_index])
            specs.append((row_index, "single_source_value", source_position))
        native_term = base["p"][row_index, 0] * base["u"][row_index, 0]
        joint_head = torch.einsum("k,kd->d", base["p"][row_index, :q + 1],
                                  donor["u"][row_index, :q + 1])
        for condition, head in (("native_noop", base["head"][row_index]),
                                ("joint_all_values", joint_head),
                                ("complete_head", donor["head"][row_index])):
            batch_rows.append(row_index)
            source_positions.append(0)
            terms.append(native_term)
            heads.append(head)
            specs.append((row_index, condition, None))
    indices = torch.tensor(batch_rows, dtype=torch.long, device=base_tokens.device)
    return {
        "tokens": base_tokens[indices], "finals": base_finals[indices],
        "source_positions": torch.tensor(source_positions, dtype=torch.long,
                                         device=base_tokens.device),
        "replacement_terms": torch.stack(terms), "replacement_heads": torch.stack(heads),
        "specs": specs,
    }


def _donor_metrics(logits, row, q, torch):
    donor, recipient = int(row["donor_answer_id"]), int(row["base_answer_id"])
    return (float(logits[q, donor] - logits[q, recipient]),
            float(-torch.log_softmax(logits[q], dim=-1)[donor]))


def _capability(rows, native_base, native_donor, base_finals, donor_finals):
    cells = defaultdict(lambda: {"base": [], "donor": []})
    for index, row in enumerate(rows):
        for side, logits, finals, answer_key, foil_key in (
            ("base", native_base, base_finals, "base_answer_id", "base_foil_id"),
            ("donor", native_donor, donor_finals, "donor_answer_id", "donor_foil_id"),
        ):
            q = int(finals[index])
            cells[row["atlas_cell_id"]][side].append(
                float(logits[index, q, int(row[answer_key])]
                      - logits[index, q, int(row[foil_key])]) > 0
            )
    return {cell: {side: sum(values) / len(values) for side, values in groups.items()}
            for cell, groups in sorted(cells.items())}


def _subject_parent_evidence():
    path = ROOT / "circuits/fast_screens/task14_head11_3_subject_payload_lemma_direction_factorial_v1_result.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != SUBJECT_PARENT_SHA256:
        raise RuntimeError("same-lemma subject-payload parent changed")
    result = json.loads(path.read_text())
    return {row["row_id"]: row for row in result["evidence"]
            if row["condition"] == "same_lemma_payload"}


def score(evidence, capability, replay_error, identity_error, noop_error, source_zero_error,
          subject_reproduction_error, bars):
    grouped = defaultdict(list)
    for item in evidence:
        grouped[(item["atlas_cell_id"], item["condition"], item["source_position"])].append(item)
    cells = {}
    non_subject_signal = False
    for cell_id, accuracy in capability.items():
        complete_rows = grouped[(cell_id, "complete_head", None)]
        complete_margin = statistics.fmean(row["margin_delta"] for row in complete_rows)
        complete_ce = statistics.fmean(row["donor_ce_gain"] for row in complete_rows)
        complete_direction = sum(row["margin_delta"] > 0 for row in complete_rows) / len(complete_rows)
        source_summaries = []
        positions = sorted({key[2] for key in grouped if key[0] == cell_id
                            and key[1] == "single_source_value"})
        for position in positions:
            values = grouped[(cell_id, "single_source_value", position)]
            margin = statistics.fmean(row["margin_delta"] for row in values)
            ce = statistics.fmean(row["donor_ce_gain"] for row in values)
            recovery = margin / complete_margin if complete_margin > 0 else None
            direction = sum(row["margin_delta"] > 0 for row in values) / len(values)
            signal = (position != 1 and recovery is not None and
                      recovery >= bars["minimum_descriptive_source_margin_recovery"] and
                      direction >= bars["minimum_descriptive_source_direction_fraction"] and ce > 0)
            non_subject_signal |= signal
            source_summaries.append({
                "source_position": position,
                "semantic_role": values[0]["semantic_role"],
                "recipient_token_examples": sorted({row["recipient_token_text"] for row in values}),
                "donor_token_examples": sorted({row["donor_token_text"] for row in values}),
                "mean_margin_delta": margin, "margin_direction_fraction": direction,
                "margin_recovery_of_complete_head": recovery,
                "mean_donor_ce_gain": ce, "descriptive_non_subject_signal": signal,
            })
        joint_rows = grouped[(cell_id, "joint_all_values", None)]
        joint_margin = statistics.fmean(row["margin_delta"] for row in joint_rows)
        joint_ce = statistics.fmean(row["donor_ce_gain"] for row in joint_rows)
        summed_margin = sum(summary["mean_margin_delta"] for summary in source_summaries)
        summed_ce = sum(summary["mean_donor_ce_gain"] for summary in source_summaries)
        cells[cell_id] = {
            "row_count": len(complete_rows), "native_accuracy": accuracy,
            "complete_head": {"mean_margin_delta": complete_margin,
                              "margin_direction_fraction": complete_direction,
                              "mean_donor_ce_gain": complete_ce},
            "sources": source_summaries,
            "joint_all_values": {"mean_margin_delta": joint_margin,
                                 "mean_donor_ce_gain": joint_ce,
                                 "margin_recovery_of_complete_head":
                                     joint_margin / complete_margin if complete_margin > 0 else None},
            "joint_minus_sum_single": {"margin": joint_margin - summed_margin,
                                       "ce_gain": joint_ce - summed_ce},
        }
    instrument = (
        replay_error <= bars["maximum_native_replay_absolute_logit_error"] and
        identity_error <= bars["maximum_source_term_identity_absolute_error"] and
        noop_error <= bars["maximum_native_noop_absolute_logit_error"] and
        source_zero_error <= bars["maximum_source_zero_absolute_logit_error"] and
        subject_reproduction_error <= bars["maximum_subject_parent_reproduction_error"] and
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
        "source_zero_max_absolute_logit_error": source_zero_error,
        "subject_parent_max_absolute_reproduction_error": subject_reproduction_error,
        "cells": cells,
        "predictions": {
            "pred_a_instrument_live": instrument,
            "pred_b_non_subject_source_signal": instrument and non_subject_signal,
            "pred_c_subject_only_among_single_sources": instrument and not non_subject_signal,
        },
    }


def evaluate(model, torch, F, facade, plan):
    rows = build_rows()
    device = next(model.parameters()).device
    length = max(len(row["base_ids"]) for row in rows)
    base_tokens, base_finals = _pad(rows, "base_ids", length, torch, device)
    donor_tokens, donor_finals = _pad(rows, "donor_ids", length, torch, device)
    combined_tokens, combined_finals = torch.cat((base_tokens, donor_tokens)), \
        torch.cat((base_finals, donor_finals))
    native = factor_parent._native_logits(model, combined_tokens, torch, F)
    replay, factors = factor_parent._factor_forward(
        model, combined_tokens, combined_finals, torch, F, facade,
    )
    count = len(rows)
    base, donor = _split_factors(factors, count)
    patch = _compile_patch_batch(base_tokens, base_finals, base, donor, rows, torch)
    patched, patched_factors = factor_parent._factor_forward(
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
    native_base, native_donor = native[:count], native[count:]
    capability = _capability(rows, native_base, native_donor, base_finals, donor_finals)
    parent_subject = _subject_parent_evidence()
    evidence, noop_error, source_zero_error, subject_error = [], 0.0, 0.0, 0.0
    for patched_index, (row_index, condition, source_position) in enumerate(patch["specs"]):
        row = rows[row_index]
        q = int(base_finals[row_index])
        native_margin, native_ce = _donor_metrics(replay[row_index], row, q, torch)
        margin, ce = _donor_metrics(patched[patched_index], row, q, torch)
        item = {
            "row_id": row["row_id"], "group_id": row["group_id"],
            "target_family": row["target_family"],
            "atlas_cell_id": row["atlas_cell_id"],
            "condition": condition, "source_position": source_position,
            "semantic_role": row["source_roles"][source_position] if source_position is not None else None,
            "recipient_token_id": row["base_ids"][source_position] if source_position is not None else None,
            "recipient_token_text": row["source_token_text"][source_position] if source_position is not None else None,
            "donor_token_id": row["donor_ids"][source_position] if source_position is not None else None,
            "donor_token_text": row["donor_source_token_text"][source_position] if source_position is not None else None,
            "native_donor_margin": native_margin, "donor_margin": margin,
            "margin_delta": margin - native_margin,
            "native_donor_ce": native_ce, "donor_ce": ce, "donor_ce_gain": native_ce - ce,
        }
        evidence.append(item)
        if condition == "native_noop":
            noop_error = max(noop_error, float((patched[patched_index] - replay[row_index]).abs().max()))
        if condition == "single_source_value" and source_position == 0:
            source_zero_error = max(
                source_zero_error, float((patched[patched_index] - replay[row_index]).abs().max()),
            )
        if condition == "single_source_value" and source_position == 1:
            parent_row = parent_subject[row["row_id"]]
            subject_error = max(subject_error,
                                abs(item["margin_delta"] - parent_row["margin_delta"]),
                                abs(item["donor_ce"] - parent_row["donor_ce"]),
                                abs(item["native_donor_ce"] - parent_row["native_donor_ce"]))
    return (evidence, capability, replay_error, identity_error, noop_error,
            source_zero_error, subject_error)


def main():
    plan = compile_plan()
    if "--dry-run" in sys.argv or os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    torch, F, facade = factor_parent._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True,
    )
    with torch.no_grad():
        values = evaluate(model, torch, F, facade, plan)
    scored = score(*values, plan["bars"])
    predictions = scored["predictions"]
    if not predictions["pred_a_instrument_live"]:
        terminal = "invalid"
    elif predictions["pred_b_non_subject_source_signal"]:
        terminal = "non_subject_source_signal_screen"
    else:
        terminal = "subject_only_single_source_screen"
    result = {
        "schema": "task14_head11_3_same_syntax_source_value_atlas_result_v1",
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
