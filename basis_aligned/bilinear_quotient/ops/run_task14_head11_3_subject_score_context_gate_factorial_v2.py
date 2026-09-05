#!/usr/bin/env python3
"""Capability-repaired Task14 L11H3 score-context factorial (v2)."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_causal_score_span_live pred_c_opposite_syntax_score_rescues_weak_cell pred_d_weak_cell_natural_score_no_interaction pred_e_natural_score_mixed_result

from __future__ import annotations

from collections import defaultdict
import json
import os
import statistics
import sys

import run_task14_head11_3_subject_score_context_gate_factorial as v1


ROOT = v1.ROOT
OUT = ROOT / "circuits/fast_screens/task14_head11_3_subject_score_context_gate_factorial_v2_result.json"
PRIOR_ART_SHA256 = "7f5bb2186060f377cbb3d1d675e6e41b1b4d034f9a87327f4e12d80a2ab67df4"
CONDITIONS = v1.CONDITIONS[:-1] + (
    "zero_score_opposite_value", "twice_native_score_opposite_value", "complete_head",
)


def compile_plan():
    parent = v1.compile_plan()
    bars = {
        key: value for key, value in parent["bars"].items()
        if key not in {"minimum_weak_cell_relative_score_contrast_fraction",
                       "minimum_relative_score_contrast"}
    }
    bars.update(
        minimum_score_span_margin_recovery=.10,
        minimum_score_span_direction_fraction=.75,
        minimum_score_span_ce_benefit=0.0,
        minimum_weak_cell_nonzero_score_contrast_fraction=1.0,
    )
    return {
        **parent,
        "schema": "task14_head11_3_subject_score_context_gate_factorial_plan_v2",
        "candidate_id": (
            "subject_verb.number_agreement."
            "head11_3_subject_score_context_gate_factorial_capability_repair_v2"
        ),
        "conditions": list(CONDITIONS),
        "price": {"model_forwards": 3, "example_evaluations": 832,
                  "backwards": 0, "parameter_updates": 0},
        "bars": bars,
        "capability_repair": (
            "Replace v1's arbitrary relative natural-score cutoff with the task-level span "
            "between zero and twice the native score while holding the donor value fixed. "
            "The natural alternate score is required only to differ from native above epsilon "
            "on every weak-cell row."
        ),
    }


def _patch_batch(base_tokens, base_finals, base, score_context, opposite_value, torch):
    count = len(base_tokens)
    subject = torch.ones(count, dtype=torch.long, device=base_tokens.device)
    native_p, native_u = v1.factor_parent._selected(base, subject, torch)
    context_p, _context_u = v1.factor_parent._selected(score_context, subject, torch)
    _opposite_p, opposite_u = v1.factor_parent._selected(opposite_value, subject, torch)
    terms = (
        native_p.unsqueeze(-1) * native_u,
        context_p.unsqueeze(-1) * native_u,
        native_p.unsqueeze(-1) * opposite_u,
        context_p.unsqueeze(-1) * opposite_u,
        torch.zeros_like(opposite_u),
        (2 * native_p).unsqueeze(-1) * opposite_u,
        native_p.unsqueeze(-1) * native_u,
    )
    return {
        "tokens": base_tokens.repeat(len(CONDITIONS), 1),
        "finals": base_finals.repeat(len(CONDITIONS)),
        "source_positions": subject.repeat(len(CONDITIONS)),
        "replacement_terms": torch.cat(terms),
        "replacement_heads": torch.cat(
            (base["head"],) * (len(CONDITIONS) - 1) + (opposite_value["head"],)
        ),
    }


def _score_span(evidence, cells):
    grouped = defaultdict(list)
    for row in evidence:
        grouped[(row["cell_id"], row["condition"])].append(row)
    output = {}
    for cell_id, cell in cells.items():
        zero = grouped[(cell_id, "zero_score_opposite_value")]
        double = grouped[(cell_id, "twice_native_score_opposite_value")]
        margin_spans = [right["donor_margin"] - left["donor_margin"]
                        for left, right in zip(zero, double)]
        ce_spans = [left["donor_ce"] - right["donor_ce"]
                    for left, right in zip(zero, double)]
        denominator = cell["conditions"]["complete_head"]["mean_margin_delta"]
        mean_margin = statistics.fmean(margin_spans)
        output[cell_id] = {
            "row_count": len(margin_spans),
            "mean_donor_directed_margin_span": mean_margin,
            "margin_span_recovery_of_complete_head": mean_margin / denominator,
            "margin_span_direction_fraction": sum(value > 0 for value in margin_spans) /
                len(margin_spans),
            "mean_donor_answer_ce_benefit": statistics.fmean(ce_spans),
        }
    return output


def score(evidence, capability, replay_error, corner_error, identity_error,
          context_value_error, score_contrast, reproduction_error, bars):
    # Reuse v1's exact four-corner arithmetic. Its obsolete liveness prediction is discarded.
    natural = v1.score(
        evidence, capability, replay_error, corner_error, identity_error,
        context_value_error, score_contrast, reproduction_error, v1.compile_plan()["bars"],
    )
    cells = natural["cells"]
    span = _score_span(evidence, cells)
    for cell_id in cells:
        cells[cell_id]["zero_to_twice_native_score_span"] = span[cell_id]
    exact_preflight = (
        replay_error <= bars["maximum_native_replay_absolute_logit_error"] and
        corner_error <= bars["maximum_native_corner_absolute_logit_error"] and
        identity_error <= bars["maximum_source_term_identity_absolute_error"] and
        context_value_error <= bars["maximum_score_context_subject_value_error"] and
        reproduction_error <= bars["maximum_parent_payload_reproduction_error"] and
        score_contrast[v1.WEAK_CELL]["fraction_above_epsilon"] >=
        bars["minimum_weak_cell_nonzero_score_contrast_fraction"] and
        all(cell["native_capability_passed"] and cell["complete_head_ceiling_passed"]
            for cell in cells.values())
    )
    weak_span = span[v1.WEAK_CELL]
    span_live = (
        weak_span["margin_span_recovery_of_complete_head"] >=
        bars["minimum_score_span_margin_recovery"] and
        weak_span["margin_span_direction_fraction"] >=
        bars["minimum_score_span_direction_fraction"] and
        weak_span["mean_donor_answer_ce_benefit"] >
        bars["minimum_score_span_ce_benefit"]
    )
    instrument = exact_preflight and span_live
    weak = cells[v1.WEAK_CELL]
    native = weak["opposite_value_effect"]["at_native_score"]
    alternate = weak["opposite_value_effect"]["at_opposite_syntax_score"]
    interaction = weak["score_by_value_interaction"]["margin_interaction_recovery_of_complete_head"]
    rescue = (
        instrument and
        alternate["margin_recovery_of_complete_head"] >= bars["minimum_rescued_margin_recovery"] and
        alternate["margin_direction_fraction"] >= bars["minimum_rescued_direction_fraction"] and
        alternate["ce_recovery_of_complete_head"] > bars["minimum_positive_ce_recovery"] and
        alternate["margin_recovery_of_complete_head"] >=
        native["margin_recovery_of_complete_head"] + bars["minimum_rescue_over_native_score"] and
        interaction >= bars["minimum_positive_interaction_recovery"]
    )
    independent = instrument and abs(interaction) <= \
        bars["maximum_context_independent_interaction_recovery"]
    mixed = instrument and not (rescue or independent)
    return {
        "native_replay_max_absolute_logit_error": replay_error,
        "native_corner_max_absolute_logit_error": corner_error,
        "source_term_identity_max_absolute_error": identity_error,
        "score_context_subject_value_max_absolute_error": context_value_error,
        "natural_score_contrast": score_contrast,
        "parent_payload_max_absolute_reproduction_error": reproduction_error,
        "exact_preflight_live": exact_preflight,
        "cells": cells,
        "predictions": {
            "pred_a_instrument_live": instrument,
            "pred_b_causal_score_span_live": exact_preflight and span_live,
            "pred_c_opposite_syntax_score_rescues_weak_cell": rescue,
            "pred_d_weak_cell_natural_score_no_interaction": independent,
            "pred_e_natural_score_mixed_result": mixed,
        },
    }


def evaluate(model, torch, F, facade, plan):
    rows = v1.build_rows()
    device = next(model.parameters()).device
    keys = ("base_ids", "score_context_ids", "donor_ids")
    length = max(len(row[key]) for row in rows for key in keys)
    padded = tuple(v1._pad(rows, key, length, torch, device) for key in keys)
    combined_tokens = torch.cat(tuple(item[0] for item in padded))
    combined_finals = torch.cat(tuple(item[1] for item in padded))
    native = v1.factor_parent._native_logits(model, combined_tokens, torch, F)
    replay, factors = v1.factor_parent._factor_forward(
        model, combined_tokens, combined_finals, torch, F, facade,
    )
    count = len(rows)
    native_parts = tuple(native[start:start + count] for start in (0, count, 2 * count))
    factor_parts = v1._split_factors(factors, count)
    patch = _patch_batch(padded[0][0], padded[0][1], *factor_parts, torch)
    patched, patched_factors = v1.factor_parent._factor_forward(
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
    native_p, native_u = v1.factor_parent._selected(factor_parts[0], subject, torch)
    context_p, context_u = v1.factor_parent._selected(factor_parts[1], subject, torch)
    context_value_error = float((native_u - context_u).abs().max())
    score_delta = (context_p - native_p).abs()
    score_contrast = {}
    for cell_id in sorted({row["cell_id"] for row in rows}):
        indices = torch.tensor([index for index, row in enumerate(rows)
                                if row["cell_id"] == cell_id], device=device)
        values = score_delta[indices]
        relative = values / (native_p[indices].abs() + plan["bars"]["score_contrast_absolute_epsilon"])
        score_contrast[cell_id] = {
            "mean_absolute_difference": float(values.mean()),
            "maximum_absolute_difference": float(values.max()),
            "fraction_above_epsilon": float((
                values > plan["bars"]["score_contrast_absolute_epsilon"]
            ).float().mean()),
            "mean_absolute_relative_difference": float(relative.mean()),
            "fraction_above_relative_threshold": float((relative > .10).float().mean()),
            "relative_threshold_is_diagnostic_only": True,
        }
    capability = v1._capability(rows, native_parts, tuple(item[1] for item in padded))
    parent_rows = v1._parent_payload_evidence()
    reproduction_error = 0.0
    evidence = []
    for index, row in enumerate(rows):
        q = int(padded[0][1][index])
        native_margin, native_ce = v1.transfer_parent._donor_metrics(replay[index], row, q, torch)
        for condition_index, condition in enumerate(CONDITIONS):
            margin, ce = v1.transfer_parent._donor_metrics(
                patched[condition_index, index], row, q, torch,
            )
            record = {
                "row_id": row["row_id"], "group_id": row["group_id"],
                "cell_id": row["cell_id"], "condition": condition,
                "native_donor_margin": native_margin, "donor_margin": margin,
                "margin_delta": margin - native_margin,
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
    torch, F, facade = v1.factor_parent._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True,
    )
    with torch.no_grad():
        args = evaluate(model, torch, F, facade, plan)
    scored = score(*args, plan["bars"])
    predictions = scored["predictions"]
    if not predictions["pred_a_instrument_live"]:
        terminal = "invalid_score_span"
    elif predictions["pred_c_opposite_syntax_score_rescues_weak_cell"]:
        terminal = "subject_score_context_gate_screen"
    elif predictions["pred_d_weak_cell_natural_score_no_interaction"]:
        terminal = "weak_cell_natural_score_no_interaction_null"
    else:
        terminal = "natural_score_mixed_result"
    result = {
        "schema": "task14_head11_3_subject_score_context_gate_factorial_result_v2",
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
