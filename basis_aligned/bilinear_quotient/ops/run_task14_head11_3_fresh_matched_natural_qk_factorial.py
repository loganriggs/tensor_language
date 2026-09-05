#!/usr/bin/env python3
"""Licensed fresh matched-natural Task14 subject score-by-value factorial."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_number_score_discriminative pred_c_lexically_selective pred_d_bidirectional_task_use pred_e_directionally_asymmetric_task_use

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Mapping, Sequence

import circuit_fast_screen_managed_runner as managed
import native_capability_license as licensing
import run_task14_fresh_matched_natural_native_capability as capability
import run_task14_head11_3_subject_attractor_score_payload_factorial as factors


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_head11_3_fresh_matched_natural_qk_factorial_v1.json"
OUT = ROOT / "circuits/fast_screens/task14_head11_3_fresh_matched_natural_qk_factorial_v1_result.json"
PRIOR_ART_SHA256 = "d87ce6a857d1ac0dd58aee02822dad3a2fa99448bc3bfce4bddedd84a1034c44"
EXPECTED_LICENSE_SHA256 = "12d1835835612ce52629272309cbb49ff0af4d48dcabad45fe7e29e3fea94b4c"
CAUSAL_CANDIDATE_ID = \
    "subject_verb.number_agreement.head11_3_fresh_matched_natural_qk_factorial_v1"
SELF_POSITION = 8
CONDITIONS = (
    "same_score_same_value",
    "opposite_score_same_value",
    "same_score_opposite_value",
    "opposite_score_opposite_value",
    "lexical_score_same_value",
    "lexical_score_opposite_value",
    "complete_opposite_head",
)
BARS = {
    "maximum_native_replay_absolute_logit_error": 7e-5,
    "maximum_source_term_sum_absolute_error": 5e-5,
    "maximum_same_score_same_value_endpoint_error": 7e-5,
    "maximum_installed_term_absolute_error": 5e-5,
    "maximum_complete_head_vector_absolute_error": 5e-5,
    "minimum_complete_head_mean_donor_margin_improvement": .05,
    "minimum_complete_head_mean_donor_CE_improvement": 0.0,
    "minimum_complete_head_row_improvement_fraction": .75,
    "minimum_number_score_absolute_mean_are_minus_is_effect": .05,
    "minimum_number_score_expected_row_sign_fraction": .75,
    "maximum_same_number_lexical_over_number_margin_ratio": .25,
    "minimum_answer_helpful_mean_margin_effect": .05,
    "minimum_answer_helpful_mean_CE_improvement": 0.0,
    "minimum_answer_helpful_row_fraction": .75,
}


class FactorialError(ValueError):
    """The licensed causal factorial contract was violated."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rows() -> list[dict]:
    rows = [row for row in capability.authority.build_rows() if row["phase"] == "HOLDOUT"]
    if len(rows) != 16 or {row["phase"] for row in rows} != {"HOLDOUT"}:
        raise FactorialError("causal runner must use exactly sixteen licensed HOLDOUT rows")
    return rows


def validate_preflight() -> dict:
    if _sha256(PRIOR_ART) != PRIOR_ART_SHA256:
        raise FactorialError("causal prior-art receipt changed")
    if capability.authority.CAUSAL_CANDIDATE_ID != CAUSAL_CANDIDATE_ID:
        raise FactorialError("causal candidate ID changed")
    return licensing.validate_causal_preflight(
        capability.build_gate(), capability.CAPABILITY_RESULT, capability.LICENSE,
        expected_license_sha256=EXPECTED_LICENSE_SHA256,
        causal_candidate_id=CAUSAL_CANDIDATE_ID)


def compile_plan() -> dict:
    license_value = validate_preflight()
    rows = build_rows()
    return {
        "schema": "task14_head11_3_fresh_matched_natural_qk_factorial_plan_v1",
        "candidate_id": CAUSAL_CANDIDATE_ID,
        "split": "LICENSED_HOLDOUT", "screen_tier": "BASIC",
        "row_count": len(rows),
        "authority_logical_sha256": capability.AUTHORITY_LOGICAL_SHA256,
        "prior_art_sha256": PRIOR_ART_SHA256,
        "capability_license_sha256": EXPECTED_LICENSE_SHA256,
        "capability_result_sha256": license_value["capability_result_sha256"],
        "preflight_validated": True,
        "conditions": list(CONDITIONS), "bars": dict(BARS),
        "site": {"layer": 11, "head": 3, "source_position": SELF_POSITION,
                 "destination_position": SELF_POSITION},
        "price": {"model_forwards": 3, "example_evaluations": 208,
                  "backwards": 0, "parameter_updates": 0},
        "metrics": ["recipient_answer_margin", "donor_answer_margin",
                    "recipient_full_vocab_CE_improvement",
                    "donor_full_vocab_CE_improvement"],
        "ce_sign_rule": "positive improvement is better for the named answer in both directions",
        "old_foreign_prefix_contrast": "diagnostic only; not recomputed or gated",
        "closed_claims": ["individual_q_or_k_semantics", "whole_head_sufficiency",
                          "necessity_or_removal", "syntax_generality", "FIT_performance"],
    }


def _role_batch(rows, torch, device):
    blocks = {}
    for role in capability.authority.ROLES:
        blocks[role] = torch.tensor(
            [row["endpoints"][role]["ids"] for row in rows],
            dtype=torch.long, device=device)
    tokens = torch.cat([blocks[role] for role in capability.authority.ROLES])
    finals = torch.full((len(tokens),), SELF_POSITION, dtype=torch.long, device=device)
    return tokens, finals


def _split_roles(value, count):
    return {
        role: {key: tensor[index*count:(index+1)*count] for key, tensor in value.items()}
        for index, role in enumerate(capability.authority.ROLES)
    }


def _compile_patch_batch(tokens, recipient, opposite, lexical, rows, torch):
    score_value = {
        "same_score_same_value": (recipient["p"][:, SELF_POSITION],
                                  recipient["u"][:, SELF_POSITION]),
        "opposite_score_same_value": (opposite["p"][:, SELF_POSITION],
                                      recipient["u"][:, SELF_POSITION]),
        "same_score_opposite_value": (recipient["p"][:, SELF_POSITION],
                                      opposite["u"][:, SELF_POSITION]),
        "opposite_score_opposite_value": (opposite["p"][:, SELF_POSITION],
                                          opposite["u"][:, SELF_POSITION]),
        "lexical_score_same_value": (lexical["p"][:, SELF_POSITION],
                                     recipient["u"][:, SELF_POSITION]),
        "lexical_score_opposite_value": (lexical["p"][:, SELF_POSITION],
                                         opposite["u"][:, SELF_POSITION]),
    }
    indices, heads, specs, expected_terms = [], [], [], []
    native_term = recipient["p"][:, SELF_POSITION].unsqueeze(-1) \
        * recipient["u"][:, SELF_POSITION]
    for row_index, row in enumerate(rows):
        for condition in CONDITIONS:
            if condition == "complete_opposite_head":
                head = opposite["head"][row_index]
                expected_term = native_term[row_index]
            else:
                score, value = score_value[condition]
                expected_term = score[row_index] * value[row_index]
                head = recipient["head"][row_index] - native_term[row_index] + expected_term
            indices.append(row_index)
            heads.append(head)
            expected_terms.append(expected_term)
            specs.append((row_index, condition, f"{row['direction_id']}__{row['template_id']}"))
    index = torch.tensor(indices, dtype=torch.long, device=tokens.device)
    return {
        "tokens": tokens[index],
        "finals": torch.full((len(index),), SELF_POSITION, dtype=torch.long, device=tokens.device),
        "replacement_heads": torch.stack(heads),
        "expected_terms": torch.stack(expected_terms),
        "row_indices": index,
        "specs": specs,
    }


def _endpoint_metrics(logits, row, torch):
    recipient = row["endpoints"]["recipient"]
    opposite = row["endpoints"]["opposite_same_lemma"]
    recipient_answer, donor_answer = int(recipient["answer_id"]), int(opposite["answer_id"])
    recipient_margin = float(logits[recipient_answer] - logits[donor_answer])
    donor_margin = float(logits[donor_answer] - logits[recipient_answer])
    log_probs = torch.log_softmax(logits, dim=-1)
    return {
        "are_minus_is_margin": float(logits[389] - logits[318]),
        "recipient_margin": recipient_margin, "donor_margin": donor_margin,
        "recipient_CE": float(-log_probs[recipient_answer]),
        "donor_CE": float(-log_probs[donor_answer]),
    }


def evaluate(model, torch, F, facade):
    rows = build_rows(); count = len(rows); device = next(model.parameters()).device
    tokens, finals = _role_batch(rows, torch, device)
    native = factors._native_logits(model, tokens, torch, F)
    replay, captured = factors._factor_forward(model, tokens, finals, torch, F, facade)
    role_factors = _split_roles(captured, count)
    recipient = role_factors["recipient"]
    opposite = role_factors["opposite_same_lemma"]
    lexical = role_factors["same_number_different_lemma"]
    recipient_tokens = tokens[:count]
    patch = _compile_patch_batch(recipient_tokens, recipient, opposite, lexical, rows, torch)
    patched, patched_factors = factors._factor_forward(
        model, patch["tokens"], patch["finals"], torch, F, facade,
        replacement_heads=patch["replacement_heads"])

    source_sum_error = max(float((torch.einsum("bk,bkd->bd", side["p"], side["u"])
                                  - side["head"]).abs().max())
                           for side in role_factors.values())
    exactness = {
        "native_replay_max_absolute_logit_error": float((native-replay).abs().max()),
        "source_term_sum_max_absolute_error": source_sum_error,
        "same_score_same_value_endpoint_max_absolute_error": 0.0,
        "installed_term_max_absolute_error": 0.0,
        "complete_head_vector_max_absolute_error": 0.0,
    }
    native_recipient = replay[:count]
    evidence = []
    for output_index, (row_index, condition, cell_id) in enumerate(patch["specs"]):
        row = rows[row_index]
        baseline = _endpoint_metrics(native_recipient[row_index, SELF_POSITION], row, torch)
        observed = _endpoint_metrics(patched[output_index, SELF_POSITION], row, torch)
        evidence.append({
            "row_id": row["row_id"], "cell_id": cell_id, "condition": condition,
            "native_recipient_margin": baseline["recipient_margin"],
            "native_donor_margin": baseline["donor_margin"],
            "recipient_margin": observed["recipient_margin"],
            "donor_margin": observed["donor_margin"],
            "recipient_margin_improvement": observed["recipient_margin"]-baseline["recipient_margin"],
            "donor_margin_improvement": observed["donor_margin"]-baseline["donor_margin"],
            "native_are_minus_is_margin": baseline["are_minus_is_margin"],
            "are_minus_is_margin": observed["are_minus_is_margin"],
            "are_minus_is_margin_effect": observed["are_minus_is_margin"]
                - baseline["are_minus_is_margin"],
            "native_recipient_CE": baseline["recipient_CE"],
            "native_donor_CE": baseline["donor_CE"],
            "recipient_CE": observed["recipient_CE"], "donor_CE": observed["donor_CE"],
            "recipient_CE_improvement": baseline["recipient_CE"]-observed["recipient_CE"],
            "donor_CE_improvement": baseline["donor_CE"]-observed["donor_CE"],
        })
        base_index = row_index
        native_term = recipient["p"][base_index, SELF_POSITION] \
            * recipient["u"][base_index, SELF_POSITION]
        if condition == "complete_opposite_head":
            exactness["complete_head_vector_max_absolute_error"] = max(
                exactness["complete_head_vector_max_absolute_error"],
                float((patch["replacement_heads"][output_index]
                       - opposite["head"][base_index]).abs().max()))
        else:
            observed_term = patch["replacement_heads"][output_index] \
                - recipient["head"][base_index] + native_term
            exactness["installed_term_max_absolute_error"] = max(
                exactness["installed_term_max_absolute_error"],
                float((observed_term-patch["expected_terms"][output_index]).abs().max()))
        if condition == "same_score_same_value":
            exactness["same_score_same_value_endpoint_max_absolute_error"] = max(
                exactness["same_score_same_value_endpoint_max_absolute_error"],
                float((patched[output_index]-native_recipient[row_index]).abs().max()))
    # The patch dispatch must have captured the native recipient factors for every repeated row.
    expected_native_heads = recipient["head"][patch["row_indices"]]
    if float((patched_factors["head"]-expected_native_heads).abs().max()) > 5e-5:
        raise RuntimeError("patched dispatch did not retain recipient source factors")
    return evidence, exactness


def _effect(grouped, cell_id, target, baseline, metric):
    left = grouped[(cell_id, baseline)]
    right = grouped[(cell_id, target)]
    values = [float(r[metric])-float(l[metric]) for l, r in zip(left, right)]
    return {"mean": statistics.fmean(values), "values": values}


def _fraction(values, predicate):
    return sum(bool(predicate(value)) for value in values) / len(values)


def _validate_scoring_inputs(evidence, exactness):
    expected = {
        (row["row_id"], f"{row['direction_id']}__{row['template_id']}", condition)
        for row in build_rows() for condition in CONDITIONS
    }
    observed = []
    for item in evidence:
        identity = (item.get("row_id"), item.get("cell_id"), item.get("condition"))
        if identity not in expected:
            raise FactorialError("evidence lies outside the licensed HOLDOUT factorial")
        for metric in ("are_minus_is_margin_effect", "donor_margin_improvement",
                       "donor_CE_improvement"):
            value = item.get(metric)
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise FactorialError("scoring metric is missing or non-finite")
        observed.append(identity)
    if len(observed) != len(expected) or len(set(observed)) != len(expected) \
            or set(observed) != expected:
        raise FactorialError("evidence does not exactly cover the licensed factorial")
    expected_exactness = {
        "native_replay_max_absolute_logit_error", "source_term_sum_max_absolute_error",
        "same_score_same_value_endpoint_max_absolute_error",
        "installed_term_max_absolute_error", "complete_head_vector_max_absolute_error",
    }
    if set(exactness) != expected_exactness or any(
            type(value) not in (int, float) or not math.isfinite(float(value))
            for value in exactness.values()):
        raise FactorialError("exactness evidence is incomplete or non-finite")


def score(evidence: Sequence[Mapping[str, object]], exactness: Mapping[str, float], bars=BARS):
    _validate_scoring_inputs(evidence, exactness)
    grouped = defaultdict(list)
    for item in evidence:
        grouped[(item["cell_id"], item["condition"])].append(item)
    cells = {}
    for cell_id in sorted({item["cell_id"] for item in evidence}):
        direction_sign = 1 if cell_id.startswith("singular_to_plural") else -1
        same_number = _effect(grouped, cell_id, "opposite_score_same_value",
                              "same_score_same_value", "are_minus_is_margin_effect")
        opposite_number = _effect(grouped, cell_id, "opposite_score_opposite_value",
                                  "same_score_opposite_value", "are_minus_is_margin_effect")
        donor_same = _effect(grouped, cell_id, "opposite_score_same_value",
                             "same_score_same_value", "donor_margin_improvement")
        donor_opposite = _effect(grouped, cell_id, "opposite_score_opposite_value",
                                 "same_score_opposite_value", "donor_margin_improvement")
        same_ce = _effect(grouped, cell_id, "opposite_score_same_value",
                          "same_score_same_value", "donor_CE_improvement")
        opposite_ce = _effect(grouped, cell_id, "opposite_score_opposite_value",
                              "same_score_opposite_value", "donor_CE_improvement")
        lexical_same = _effect(grouped, cell_id, "lexical_score_same_value",
                               "same_score_same_value", "are_minus_is_margin_effect")
        lexical_opposite = _effect(grouped, cell_id, "lexical_score_opposite_value",
                                   "same_score_opposite_value", "are_minus_is_margin_effect")
        complete = grouped[(cell_id, "complete_opposite_head")]
        complete_margin = [float(item["donor_margin_improvement"]) for item in complete]
        complete_ce = [float(item["donor_CE_improvement"]) for item in complete]
        same_abs, opposite_abs = abs(same_number["mean"]), abs(opposite_number["mean"])
        cells[cell_id] = {
            "expected_number_score_sign_same_value": -direction_sign,
            "expected_number_score_sign_opposite_value": direction_sign,
            "number_score_same_value": {
                **same_number, "mean_donor_margin_improvement": donor_same["mean"],
                "mean_donor_CE_improvement": same_ce["mean"],
                "expected_margin_sign_fraction": _fraction(
                    same_number["values"], lambda x: x*(-direction_sign) > 0),
            },
            "number_score_opposite_value": {
                **opposite_number, "mean_donor_margin_improvement": donor_opposite["mean"],
                "donor_margin_improvement_values": donor_opposite["values"],
                "mean_donor_CE_improvement": opposite_ce["mean"],
                "donor_CE_improvement_values": opposite_ce["values"],
                "expected_margin_sign_fraction": _fraction(
                    opposite_number["values"], lambda x: x*direction_sign > 0),
                "donor_helpful_margin_fraction": _fraction(donor_opposite["values"], lambda x: x > 0),
                "donor_helpful_CE_fraction": _fraction(opposite_ce["values"], lambda x: x > 0),
            },
            "score_by_value_interaction": {
                "mean_are_minus_is_margin": opposite_number["mean"]-same_number["mean"],
                "mean_donor_margin_improvement": donor_opposite["mean"]-donor_same["mean"],
                "mean_donor_CE_improvement": opposite_ce["mean"]-same_ce["mean"],
            },
            "lexical_score_same_value": {
                **lexical_same,
                "absolute_mean_over_number": abs(lexical_same["mean"])/same_abs
                    if same_abs > 0 else None,
            },
            "lexical_score_opposite_value": {
                **lexical_opposite,
                "absolute_mean_over_number": abs(lexical_opposite["mean"])/opposite_abs
                    if opposite_abs > 0 else None,
            },
            "complete_opposite_head": {
                "mean_donor_margin_improvement": statistics.fmean(complete_margin),
                "mean_donor_CE_improvement": statistics.fmean(complete_ce),
                "donor_margin_improvement_fraction": _fraction(complete_margin, lambda x: x > 0),
                "donor_CE_improvement_fraction": _fraction(complete_ce, lambda x: x > 0),
            },
        }
    exact_live = (
        exactness["native_replay_max_absolute_logit_error"] <=
            bars["maximum_native_replay_absolute_logit_error"] and
        exactness["source_term_sum_max_absolute_error"] <=
            bars["maximum_source_term_sum_absolute_error"] and
        exactness["same_score_same_value_endpoint_max_absolute_error"] <=
            bars["maximum_same_score_same_value_endpoint_error"] and
        exactness["installed_term_max_absolute_error"] <=
            bars["maximum_installed_term_absolute_error"] and
        exactness["complete_head_vector_max_absolute_error"] <=
            bars["maximum_complete_head_vector_absolute_error"])
    complete_live = all(
        cell["complete_opposite_head"]["mean_donor_margin_improvement"] >=
            bars["minimum_complete_head_mean_donor_margin_improvement"] and
        cell["complete_opposite_head"]["mean_donor_CE_improvement"] >=
            bars["minimum_complete_head_mean_donor_CE_improvement"] and
        cell["complete_opposite_head"]["donor_margin_improvement_fraction"] >=
            bars["minimum_complete_head_row_improvement_fraction"] and
        cell["complete_opposite_head"]["donor_CE_improvement_fraction"] >=
            bars["minimum_complete_head_row_improvement_fraction"]
        for cell in cells.values())
    instrument = exact_live and complete_live
    number_discriminative = instrument and all(
        abs(cell[key]["mean"]) >= bars["minimum_number_score_absolute_mean_are_minus_is_effect"] and
        cell[key]["expected_margin_sign_fraction"] >=
            bars["minimum_number_score_expected_row_sign_fraction"]
        for cell in cells.values()
        for key in ("number_score_same_value", "number_score_opposite_value"))
    selective = number_discriminative and all(
        cell[key]["absolute_mean_over_number"] is not None and
        cell[key]["absolute_mean_over_number"] <=
            bars["maximum_same_number_lexical_over_number_margin_ratio"]
        for cell in cells.values()
        for key in ("lexical_score_same_value", "lexical_score_opposite_value"))
    bidirectional = instrument and all(
        cell["number_score_opposite_value"]["mean_donor_margin_improvement"] >=
            bars["minimum_answer_helpful_mean_margin_effect"] and
        cell["number_score_opposite_value"]["mean_donor_CE_improvement"] >=
            bars["minimum_answer_helpful_mean_CE_improvement"] and
        cell["number_score_opposite_value"]["donor_helpful_margin_fraction"] >=
            bars["minimum_answer_helpful_row_fraction"] and
        cell["number_score_opposite_value"]["donor_helpful_CE_fraction"] >=
            bars["minimum_answer_helpful_row_fraction"]
        for cell in cells.values())
    asymmetric = instrument
    for cell_id, cell in cells.items():
        effect = cell["number_score_opposite_value"]
        if cell_id.startswith("singular_to_plural"):
            asymmetric &= (effect["mean_donor_margin_improvement"] >=
                           bars["minimum_answer_helpful_mean_margin_effect"]
                           and effect["mean_donor_CE_improvement"] >=
                           bars["minimum_answer_helpful_mean_CE_improvement"]
                           and effect["donor_helpful_margin_fraction"] >=
                           bars["minimum_answer_helpful_row_fraction"]
                           and effect["donor_helpful_CE_fraction"] >=
                           bars["minimum_answer_helpful_row_fraction"])
        else:
            asymmetric &= (effect["mean_donor_margin_improvement"] <=
                           -bars["minimum_answer_helpful_mean_margin_effect"]
                           and effect["mean_donor_CE_improvement"] <= 0
                           and _fraction(effect["donor_margin_improvement_values"], lambda x: x < 0) >=
                           bars["minimum_answer_helpful_row_fraction"]
                           and _fraction(effect["donor_CE_improvement_values"], lambda x: x < 0) >=
                           bars["minimum_answer_helpful_row_fraction"])
    return {
        **exactness, "cells": cells,
        "predictions": {
            "pred_a_instrument_live": bool(instrument),
            "pred_b_number_score_discriminative": bool(number_discriminative),
            "pred_c_lexically_selective": bool(selective),
            "pred_d_bidirectional_task_use": bool(bidirectional),
            "pred_e_directionally_asymmetric_task_use": bool(asymmetric),
        },
    }


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("BQLIB_DRYRUN", "BQLIB_NO_MODEL"):
        if os.environ.get(name) not in {None, "1"}:
            raise FactorialError(f"{name} must be absent or exactly 1")
    # This validates the exact license and causal candidate ID before any model load.
    plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" \
            or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    if OUT.exists():
        raise FactorialError(f"refusing to overwrite {OUT}")
    torch, F, facade = factors._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        evidence, exactness = evaluate(model, torch, F, facade)
    scored = score(evidence, exactness, plan["bars"])
    terminal = "invalid" if not scored["predictions"]["pred_a_instrument_live"] \
        else "valid_causal_screen"
    result = {
        "schema": "task14_head11_3_fresh_matched_natural_qk_factorial_result_v1",
        "candidate_id": CAUSAL_CANDIDATE_ID, "terminal": terminal,
        "plan": plan, "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "score": scored, "evidence": evidence,
        "evaluated_splits": ["LICENSED_HOLDOUT"], "forbidden_splits_opened": [],
        "model_forwards": 3, "causal_interventions": len(evidence),
    }
    payload = managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": scored["predictions"],
                      "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
