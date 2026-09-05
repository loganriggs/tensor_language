#!/usr/bin/env python3
"""Exact TEST-reuse score-context x subject-value factorial for Task14 L11H3."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_opposite_syntax_score_rescues_weak_cell pred_c_weak_pp_plural_opposite_syntax_score_no_interaction pred_d_score_context_changes_effect_without_full_rescue

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

import circuit_fast_screen_candidate_task14_test_cross_syntax as authority
import run_task14_head11_3_subject_attractor_score_payload_factorial as factor_parent
import run_task14_head11_3_subject_payload_test_transfer as transfer_parent


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits/fast_screens/task14_head11_3_subject_score_context_gate_factorial_v1_result.json"
PRIOR_ART_SHA256 = "4a6409fd58216afe34e8402463894f8f24fdc308c6575d5a472f293fcb019763"
PARENT_RESULT_SHA256 = "157c907abb796012f2b6ba2b0fb8bd302daa66455d01376467b2acdc283b3c1b"
WEAK_CELL = "pp_plural_to_relative_singular"
CONDITIONS = (
    "native_score_native_value", "opposite_syntax_score_native_value",
    "native_score_opposite_value", "opposite_syntax_score_opposite_value",
    "complete_head",
)


def _canonical(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    ).encode()).hexdigest()


def build_rows():
    rows = authority.build_rows()
    authority.validate_rows(rows)
    sources = {(str(row["group_id"]), str(row["transform_id"])): row
               for row in authority._CANDIDATE._source_rows()}
    output = []
    for row in rows:
        source = sources[(row["group_id"], row["donor_family"])]
        context = authority._CANDIDATE._endpoint(source, "base")
        if context["subject_number"] != row["base_subject_number"]:
            raise ValueError("opposite-syntax context changed grammatical number")
        if context["ids"][:2] != row["base_ids"][:2]:
            raise ValueError("opposite-syntax context changed the prefix through the subject")
        if context["ids"][-1] != row["base_ids"][-1]:
            raise ValueError("opposite-syntax context changed the final attractor token")
        if context["answer_id"] != row["base_answer_id"]:
            raise ValueError("opposite-syntax context changed the recipient answer")
        if context["ids"] == row["base_ids"]:
            raise ValueError("opposite-syntax context is identical to recipient")
        augmented = dict(row)
        augmented.update(
            score_context_ids=context["ids"], score_context_text=context["text"],
            score_context_answer_id=context["answer_id"],
            score_context_foil_id=context["foil_id"],
            score_context_semantic_position=context["position"],
            score_context_subject_number=context["subject_number"],
        )
        output.append(augmented)
    cells = defaultdict(int)
    for row in output:
        cells[row["cell_id"]] += 1
    if len(output) != 64 or len(cells) != 4 or set(cells.values()) != {16}:
        raise ValueError(f"score-context authority lost balance: {dict(cells)}")
    return output


def compile_plan():
    rows = build_rows()
    return {
        "schema": "task14_head11_3_subject_score_context_gate_factorial_plan_v1",
        "candidate_id": "subject_verb.number_agreement.head11_3_subject_score_context_gate_factorial",
        "split": "TEST_REUSE_NEW_INTERVENTION", "screen_tier": "BASIC",
        "row_count": len(rows),
        "parent_authority_sha256": authority.validate_rows(authority.build_rows()),
        "factorial_authority_sha256": _canonical(rows),
        "site": {"layer": factor_parent.LAYER, "head": factor_parent.HEAD,
                 "query": "final_prediction_position", "source": "subject_token_index_1"},
        "factors": {
            "score": ["native_recipient", "same_number_same_lemma_opposite_syntax_recipient"],
            "value": ["native_subject", "opposite_number_cross_noun_subject"],
        },
        "identification": (
            "The recipient and alternate-score prompts have the same tokens through source "
            "position 1. Causal masking therefore keeps the subject key fixed; their source-score "
            "difference comes from the final-query context. Gating is the 2x2 interaction "
            "(D-B)-(C-A), not the unadjusted difference D-C."
        ),
        "conditions": list(CONDITIONS),
        "price": {"model_forwards": 3, "example_evaluations": 704,
                  "backwards": 0, "parameter_updates": 0},
        "outcomes": ["donor_directed_is_are_margin", "donor_answer_ce",
                     "score_by_value_interaction"],
        "bars": {
            "minimum_native_accuracy_each_source_each_cell": .85,
            "maximum_native_replay_absolute_logit_error": 5e-5,
            "maximum_native_corner_absolute_logit_error": 5e-5,
            "maximum_source_term_identity_absolute_error": 5e-5,
            "maximum_score_context_subject_value_error": 5e-5,
            "maximum_parent_payload_reproduction_error": 5e-5,
            "score_contrast_absolute_epsilon": 1e-8,
            "minimum_weak_cell_relative_score_contrast_fraction": .75,
            "minimum_relative_score_contrast": .10,
            "minimum_complete_head_direction_fraction_each_cell": .75,
            "minimum_rescued_margin_recovery": .25,
            "minimum_rescued_direction_fraction": .75,
            "minimum_rescue_over_native_score": .10,
            "minimum_positive_interaction_recovery": .10,
            "maximum_context_independent_interaction_recovery": .05,
            "minimum_positive_ce_recovery": 0.0,
        },
        "scope": "TEST_REUSE_NEW_INTERVENTION", "closed_splits": ["OOD"],
        "limits": "TEST text is reused; no pristine final-test or OOD claim.",
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
    return tuple({key: value[start:start + count] for key, value in factors.items()}
                 for start in (0, count, 2 * count))


def _patch_batch(base_tokens, base_finals, base, score_context, opposite_value, torch):
    count = len(base_tokens)
    subject = torch.ones(count, dtype=torch.long, device=base_tokens.device)
    native_p, native_u = factor_parent._selected(base, subject, torch)
    context_p, _context_u = factor_parent._selected(score_context, subject, torch)
    _opposite_p, opposite_u = factor_parent._selected(opposite_value, subject, torch)
    terms = (
        native_p.unsqueeze(-1) * native_u,
        context_p.unsqueeze(-1) * native_u,
        native_p.unsqueeze(-1) * opposite_u,
        context_p.unsqueeze(-1) * opposite_u,
        native_p.unsqueeze(-1) * native_u,
    )
    return {
        "tokens": base_tokens.repeat(len(CONDITIONS), 1),
        "finals": base_finals.repeat(len(CONDITIONS)),
        "source_positions": subject.repeat(len(CONDITIONS)),
        "replacement_terms": torch.cat(terms),
        "replacement_heads": torch.cat((base["head"],) * 4 + (opposite_value["head"],)),
    }


def _capability(rows, native_parts, finals_parts):
    cells = defaultdict(lambda: {"base": [], "score_context": [], "opposite_value": []})
    specs = (
        ("base", "base_answer_id", "base_foil_id"),
        ("score_context", "score_context_answer_id", "score_context_foil_id"),
        ("opposite_value", "donor_answer_id", "donor_foil_id"),
    )
    for index, row in enumerate(rows):
        for logits, finals, (name, answer, foil) in zip(native_parts, finals_parts, specs):
            q = int(finals[index])
            cells[row["cell_id"]][name].append(
                float(logits[index, q, int(row[answer])] - logits[index, q, int(row[foil])]) > 0
            )
    return {cell: {name: sum(values) / len(values) for name, values in groups.items()}
            for cell, groups in sorted(cells.items())}


def _parent_payload_evidence():
    if hashlib.sha256(transfer_parent.OUT.read_bytes()).hexdigest() != PARENT_RESULT_SHA256:
        raise RuntimeError("parent TEST payload result changed")
    result = json.loads(transfer_parent.OUT.read_text())
    return {row["row_id"]: row for row in result["evidence"]
            if row["condition"] == "subject_payload"}


def score(evidence, capability, replay_error, corner_error, identity_error,
          context_value_error, score_contrast, reproduction_error, bars):
    grouped = defaultdict(list)
    for row in evidence:
        grouped[(row["cell_id"], row["condition"])].append(row)
    cells = {}
    for cell_id in sorted(capability):
        conditions = {}
        for condition in CONDITIONS:
            rows = grouped[(cell_id, condition)]
            conditions[condition] = {
                "mean_margin_delta": statistics.fmean(row["margin_delta"] for row in rows),
                "margin_direction_fraction": sum(row["margin_delta"] > 0 for row in rows) / len(rows),
                "mean_donor_ce_gain": statistics.fmean(row["donor_ce_gain"] for row in rows),
            }
        ceiling = conditions["complete_head"]
        ceiling_live = (ceiling["mean_margin_delta"] > 0 and ceiling["mean_donor_ce_gain"] > 0 and
                        ceiling["margin_direction_fraction"] >=
                        bars["minimum_complete_head_direction_fraction_each_cell"])
        for condition in CONDITIONS[:-1]:
            arm = conditions[condition]
            arm["margin_recovery_of_complete_head"] = (
                arm["mean_margin_delta"] / ceiling["mean_margin_delta"]
                if ceiling["mean_margin_delta"] > 0 else None
            )
            arm["ce_recovery_of_complete_head"] = (
                arm["mean_donor_ce_gain"] / ceiling["mean_donor_ce_gain"]
                if ceiling["mean_donor_ce_gain"] > 0 else None
            )
        interaction_margin = (
            conditions["opposite_syntax_score_opposite_value"]["mean_margin_delta"]
            - conditions["opposite_syntax_score_native_value"]["mean_margin_delta"]
            - conditions["native_score_opposite_value"]["mean_margin_delta"]
            + conditions["native_score_native_value"]["mean_margin_delta"]
        )
        interaction_ce = (
            conditions["opposite_syntax_score_opposite_value"]["mean_donor_ce_gain"]
            - conditions["opposite_syntax_score_native_value"]["mean_donor_ce_gain"]
            - conditions["native_score_opposite_value"]["mean_donor_ce_gain"]
            + conditions["native_score_native_value"]["mean_donor_ce_gain"]
        )
        native_value_rows = grouped[(cell_id, "native_score_opposite_value")]
        native_baseline_rows = grouped[(cell_id, "native_score_native_value")]
        alternate_value_rows = grouped[(cell_id, "opposite_syntax_score_opposite_value")]
        alternate_baseline_rows = grouped[(cell_id, "opposite_syntax_score_native_value")]
        native_margin_effects = [value["margin_delta"] - baseline["margin_delta"]
                                 for value, baseline in zip(native_value_rows, native_baseline_rows)]
        alternate_margin_effects = [value["margin_delta"] - baseline["margin_delta"]
                                    for value, baseline in zip(alternate_value_rows, alternate_baseline_rows)]
        native_ce_effects = [value["donor_ce_gain"] - baseline["donor_ce_gain"]
                             for value, baseline in zip(native_value_rows, native_baseline_rows)]
        alternate_ce_effects = [value["donor_ce_gain"] - baseline["donor_ce_gain"]
                                for value, baseline in zip(alternate_value_rows, alternate_baseline_rows)]
        value_effects = {}
        for name, margins, ces in (
            ("at_native_score", native_margin_effects, native_ce_effects),
            ("at_opposite_syntax_score", alternate_margin_effects, alternate_ce_effects),
        ):
            mean_margin = statistics.fmean(margins)
            mean_ce = statistics.fmean(ces)
            value_effects[name] = {
                "mean_margin_effect": mean_margin,
                "margin_direction_fraction": sum(value > 0 for value in margins) / len(margins),
                "margin_recovery_of_complete_head": mean_margin / ceiling["mean_margin_delta"],
                "mean_ce_benefit": mean_ce,
                "ce_recovery_of_complete_head": mean_ce / ceiling["mean_donor_ce_gain"],
            }
        cells[cell_id] = {
            "row_count": len(grouped[(cell_id, "complete_head")]),
            "native_accuracy": capability[cell_id],
            "native_capability_passed": min(capability[cell_id].values()) >=
                bars["minimum_native_accuracy_each_source_each_cell"],
            "complete_head_ceiling_passed": ceiling_live,
            "conditions": conditions,
            "opposite_syntax_score_contrast": score_contrast[cell_id],
            "opposite_value_effect": value_effects,
            "score_by_value_interaction": {
                "mean_margin_interaction": interaction_margin,
                "margin_interaction_recovery_of_complete_head": (
                    interaction_margin / ceiling["mean_margin_delta"]
                    if ceiling["mean_margin_delta"] > 0 else None
                ),
                "mean_ce_interaction": interaction_ce,
                "ce_interaction_recovery_of_complete_head": (
                    interaction_ce / ceiling["mean_donor_ce_gain"]
                    if ceiling["mean_donor_ce_gain"] > 0 else None
                ),
            },
        }
    instrument = (replay_error <= bars["maximum_native_replay_absolute_logit_error"] and
                  corner_error <= bars["maximum_native_corner_absolute_logit_error"] and
                  identity_error <= bars["maximum_source_term_identity_absolute_error"] and
                  context_value_error <= bars["maximum_score_context_subject_value_error"] and
                  reproduction_error <= bars["maximum_parent_payload_reproduction_error"] and
                  score_contrast[WEAK_CELL]["fraction_above_relative_threshold"] >=
                  bars["minimum_weak_cell_relative_score_contrast_fraction"] and
                  all(cell["native_capability_passed"] and cell["complete_head_ceiling_passed"]
                      for cell in cells.values()))
    weak = cells[WEAK_CELL]
    native = weak["opposite_value_effect"]["at_native_score"]
    alternate = weak["opposite_value_effect"]["at_opposite_syntax_score"]
    interaction = weak["score_by_value_interaction"]["margin_interaction_recovery_of_complete_head"]
    rescue = (instrument and
              alternate["margin_recovery_of_complete_head"] >= bars["minimum_rescued_margin_recovery"] and
              alternate["margin_direction_fraction"] >= bars["minimum_rescued_direction_fraction"] and
              alternate["ce_recovery_of_complete_head"] > bars["minimum_positive_ce_recovery"] and
              alternate["margin_recovery_of_complete_head"] >=
              native["margin_recovery_of_complete_head"] + bars["minimum_rescue_over_native_score"] and
              interaction >= bars["minimum_positive_interaction_recovery"])
    independent = (instrument and abs(interaction) <=
                   bars["maximum_context_independent_interaction_recovery"])
    mixed = instrument and not (rescue or independent)
    return {
        "native_replay_max_absolute_logit_error": replay_error,
        "native_corner_max_absolute_logit_error": corner_error,
        "source_term_identity_max_absolute_error": identity_error,
        "score_context_subject_value_max_absolute_error": context_value_error,
        "score_contrast": score_contrast,
        "parent_payload_max_absolute_reproduction_error": reproduction_error,
        "cells": cells,
        "predictions": {
            "pred_a_instrument_live": instrument,
            "pred_b_opposite_syntax_score_rescues_weak_cell": rescue,
            "pred_c_weak_pp_plural_opposite_syntax_score_no_interaction": independent,
            "pred_d_score_context_changes_effect_without_full_rescue": mixed,
        },
    }


def evaluate(model, torch, F, facade, plan):
    rows = build_rows()
    device = next(model.parameters()).device
    keys = ("base_ids", "score_context_ids", "donor_ids")
    length = max(len(row[key]) for row in rows for key in keys)
    padded = tuple(_pad(rows, key, length, torch, device) for key in keys)
    combined_tokens = torch.cat(tuple(item[0] for item in padded))
    combined_finals = torch.cat(tuple(item[1] for item in padded))
    native = factor_parent._native_logits(model, combined_tokens, torch, F)
    replay, factors = factor_parent._factor_forward(
        model, combined_tokens, combined_finals, torch, F, facade,
    )
    count = len(rows)
    native_parts = tuple(native[start:start + count] for start in (0, count, 2 * count))
    factor_parts = _split_factors(factors, count)
    patch = _patch_batch(padded[0][0], padded[0][1], *factor_parts, torch)
    patched, patched_factors = factor_parent._factor_forward(
        model, patch["tokens"], patch["finals"], torch, F, facade,
        source_positions=patch["source_positions"],
        replacement_terms=patch["replacement_terms"],
        replacement_heads=patch["replacement_heads"],
    )
    patched = patched.view(len(CONDITIONS), count, length, -1)
    replay_error = float((replay - native).abs().max())
    corner_error = float((patched[0] - replay[:count]).abs().max())
    identity_error = max(
        float((torch.einsum("bk,bkd->bd", item["p"], item["u"])-item["head"]).abs().max())
        for item in (*factor_parts, patched_factors)
    )
    subject = torch.ones(count, dtype=torch.long, device=device)
    native_subject_p, native_subject_u = factor_parent._selected(factor_parts[0], subject, torch)
    context_subject_p, context_subject_u = factor_parent._selected(factor_parts[1], subject, torch)
    context_value_error = float((native_subject_u - context_subject_u).abs().max())
    score_delta = (context_subject_p - native_subject_p).abs()
    score_contrast = {}
    for cell_id in sorted({row["cell_id"] for row in rows}):
        indices = torch.tensor([index for index, row in enumerate(rows)
                                if row["cell_id"] == cell_id], device=device)
        values = score_delta[indices]
        score_contrast[cell_id] = {
            "mean_absolute_difference": float(values.mean()),
            "maximum_absolute_difference": float(values.max()),
            "fraction_above_epsilon": float(
                (values > plan["bars"]["score_contrast_absolute_epsilon"]).float().mean()
            ),
            "mean_absolute_relative_difference": float(
                (values / (native_subject_p[indices].abs() +
                           plan["bars"]["score_contrast_absolute_epsilon"])).mean()
            ),
            "fraction_above_relative_threshold": float((
                values / (native_subject_p[indices].abs() +
                          plan["bars"]["score_contrast_absolute_epsilon"])
                > plan["bars"]["minimum_relative_score_contrast"]
            ).float().mean()),
        }
    capability = _capability(rows, native_parts, tuple(item[1] for item in padded))
    parent_rows = _parent_payload_evidence()
    reproduction_error = 0.0
    evidence = []
    for index, row in enumerate(rows):
        q = int(padded[0][1][index])
        native_margin, native_ce = transfer_parent._donor_metrics(replay[index], row, q, torch)
        for condition_index, condition in enumerate(CONDITIONS):
            margin, ce = transfer_parent._donor_metrics(
                patched[condition_index, index], row, q, torch,
            )
            record = {
                "row_id": row["row_id"], "group_id": row["group_id"],
                "cell_id": row["cell_id"],
                "condition": condition, "native_donor_margin": native_margin,
                "donor_margin": margin, "margin_delta": margin - native_margin,
                "native_donor_ce": native_ce, "donor_ce": ce,
                "donor_ce_gain": native_ce - ce,
            }
            evidence.append(record)
            if condition == "native_score_opposite_value":
                parent_row = parent_rows[row["row_id"]]
                reproduction_error = max(
                    reproduction_error,
                    *(abs(record[key] - parent_row[key]) for key in
                      ("native_donor_margin", "donor_margin", "margin_delta",
                       "native_donor_ce", "donor_ce")),
                )
    return (evidence, capability, replay_error, corner_error, identity_error,
            context_value_error, score_contrast, reproduction_error)


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
        args = evaluate(model, torch, F, facade, plan)
    scored = score(*args, plan["bars"])
    predictions = scored["predictions"]
    if not predictions["pred_a_instrument_live"]:
        terminal = "invalid"
    elif predictions["pred_b_opposite_syntax_score_rescues_weak_cell"]:
        terminal = "subject_score_context_gate_screen"
    elif predictions["pred_c_weak_pp_plural_opposite_syntax_score_no_interaction"]:
        terminal = "weak_pp_plural_opposite_syntax_score_no_interaction_null"
    else:
        terminal = "subject_score_context_mixed_result"
    result = {
        "schema": "task14_head11_3_subject_score_context_gate_factorial_result_v1",
        "plan": plan, "prior_art_sha256": PRIOR_ART_SHA256,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "terminal": terminal, "score": scored, "evidence": args[0],
        "evaluated_splits": ["TEST_REUSE_NEW_INTERVENTION"],
        "forbidden_splits_opened": [], "model_forwards": 3,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"result": str(OUT), "terminal": terminal, "score": scored}, indent=2))


if __name__ == "__main__":
    main()
