#!/usr/bin/env python3
"""Licensed Task14 L11H3 subject current-value/cache-value factorial."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_current_branch_carries_task pred_c_cached_branch_carries_task pred_d_interaction_is_needed pred_e_lexical_leakage pred_f_number_specific

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
import attention_source_factor_primitive as source_factor


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_head11_3_fresh_matched_subject_current_cached_value_factorial_v1.json"
OUT = ROOT / "circuits/fast_screens/task14_head11_3_fresh_matched_subject_current_cached_value_factorial_v1_result.json"
LICENSE = ROOT / "circuits/fast_screens/task14_fresh_matched_subject_current_cached_value_factorial_v1_capability_license.json"
PRIOR_ART_SHA256 = "9ca110e1614c6bc8e3fb1dd771060ae6ac35c39088eb80728a4eece2160f9b9c"
LICENSE_SHA256 = "ce418412936aab8a8df37460dc8ef54b0555980d3625377903e88cb686f8070f"
CANDIDATE_ID = "subject_verb.number_agreement.head11_3_fresh_matched_subject_current_cached_value_factorial_v1"
LAYER, HEAD, SELF_POSITION = 11, 3, 8
CONDITIONS = (
    "native_value", "opposite_current_only", "opposite_cached_only", "opposite_both",
    "lexical_current_only", "lexical_cached_only", "lexical_both", "complete_opposite_head",
)
BARS = {
    "maximum_native_replay_absolute_logit_error": 7e-5,
    "maximum_source_term_sum_absolute_error": 5e-5,
    "maximum_value_branch_sum_absolute_error": 5e-5,
    "maximum_same_batch_native_noop_endpoint_error": 7e-5,
    "maximum_installed_head_absolute_error": 5e-5,
    "maximum_complete_head_vector_absolute_error": 5e-5,
    "minimum_complete_head_mean_target_margin_improvement": .05,
    "minimum_complete_head_mean_target_CE_improvement": 0.0,
    "minimum_complete_head_row_improvement_fraction": .75,
    "minimum_joint_value_mean_target_margin_improvement": .05,
    "minimum_joint_value_mean_target_CE_improvement": 0.0,
    "minimum_joint_value_row_improvement_fraction": .75,
    "minimum_branch_recovery_fraction": .70,
    "maximum_single_branch_recovery_for_interaction": .50,
    "minimum_interaction_recovery_fraction": .50,
    "maximum_number_specific_lexical_ratio": .25,
    "minimum_lexical_leakage_ratio": .50,
}


class CurrentCachedFactorialError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rows():
    rows = [row for row in capability.authority.build_rows() if row["phase"] == "HOLDOUT"]
    if len(rows) != 16 or {row["group_number"] for row in rows} != set(range(8, 16)):
        raise CurrentCachedFactorialError("runner must use exact licensed HOLDOUT")
    return rows


def validate_preflight():
    if _sha256(PRIOR_ART) != PRIOR_ART_SHA256:
        raise CurrentCachedFactorialError("prior-art receipt changed")
    return licensing.validate_causal_preflight(
        capability.build_gate(), capability.CAPABILITY_RESULT, LICENSE,
        expected_license_sha256=LICENSE_SHA256, causal_candidate_id=CANDIDATE_ID)


def compile_plan():
    license_value = validate_preflight()
    return {
        "schema": "task14_head11_3_fresh_matched_subject_current_cached_value_factorial_plan_v1",
        "candidate_id": CANDIDATE_ID, "split": "LICENSED_HOLDOUT",
        "row_count": len(build_rows()), "conditions": list(CONDITIONS),
        "prior_art_sha256": PRIOR_ART_SHA256, "license_sha256": LICENSE_SHA256,
        "capability_result_sha256": license_value["capability_result_sha256"],
        "preflight_validated": True, "bars": dict(BARS),
        "fixed_context": "recipient p_8 and recipient sum over j != 8 of p_j*u_j",
        "value_partition": {
            "current": "W_O,h3[(1-lambda_11)*V_11*xhat_11,8]",
            "cached": "W_O,h3[lambda_11*V_0*xhat_0,8]",
        },
        "price": {"model_forwards": 4, "example_evaluations": 352,
                  "backwards": 0, "parameter_updates": 0},
        "outcomes": ["answer_directed_target_margin_improvement",
                     "target_full_vocab_CE_improvement", "fixed_are_minus_is_change"],
        "closed_claims": ["individual_q_or_k", "necessity", "syntax_generality",
                          "FIT", "downstream_reader_identity", "rank"],
    }


def _role_batch(rows, torch, device):
    roles = ("recipient", "opposite_same_lemma", "same_number_different_lemma")
    tokens = torch.cat([torch.tensor([row["endpoints"][role]["ids"] for row in rows],
                                    dtype=torch.long, device=device) for role in roles])
    finals = torch.full((len(tokens),), SELF_POSITION, dtype=torch.long, device=device)
    return tokens, finals


def _projected_value_branches(state, first_value, attention, torch, F):
    if first_value is None:
        raise CurrentCachedFactorialError("L11 requires the block-0 cached-value bus")
    batch, length, width = state.shape
    heads, head_width = 9, width // 9
    raw = F.linear(state, attention.c_v.weight.to(state.dtype)).view(
        batch, length, heads, head_width)
    bus = first_value.view_as(raw)
    current_pre = (1 - attention.lamb) * raw
    cached_pre = attention.lamb * bus
    head_slice = attention.c_proj.weight[:, HEAD * head_width:(HEAD + 1) * head_width]
    current = F.linear(current_pre[:, :, HEAD].float(), head_slice.float())
    cached = F.linear(cached_pre[:, :, HEAD].float(), head_slice.float())
    return current, cached


def _factor_forward(model, tokens, finals, torch, F, facade, *, replacement_heads=None,
                    native_reinstall_mask=None):
    captured = {}

    def attention(event):
        if event.site != LAYER:
            return event.block.attn(event.state, event.first_value)
        write, base = source_factor.replay_attention_with_source_factors(
            event.state, event.first_value, event.block.attn, finals, HEAD, torch, F)
        current, cached = _projected_value_branches(
            event.state, event.first_value, event.block.attn, torch, F)
        captured.update({name: value.detach().clone() for name, value in base.items()})
        captured["current"] = current.detach().clone()
        captured["cached"] = cached.detach().clone()
        if replacement_heads is not None:
            rows = torch.arange(tokens.size(0), device=tokens.device)
            installed = factors._same_batch_native_heads(
                replacement_heads, base["head"], native_reinstall_mask, torch)
            write[rows, finals] += (installed - base["head"]).to(write.dtype)
        return write, event.first_value

    logits = facade.forward_with_dispatch(
        model, tokens, attention, lambda event: event.block.mlp(event.state),
        require_production=False).float()
    if set(captured) != {"p", "u", "head", "current", "cached"}:
        raise CurrentCachedFactorialError("failed to capture exact value branches")
    return logits, captured


def _compile(recipient_tokens, recipient, opposite, lexical, rows, torch):
    recipient_terms = recipient["p"].unsqueeze(-1) * recipient["u"]
    complement_mask = torch.arange(recipient_terms.shape[1], device=recipient_terms.device) != SELF_POSITION
    complement = recipient_terms[:, complement_mask].sum(1)
    native_p = recipient["p"][:, SELF_POSITION].unsqueeze(-1)

    def subject(current, cached):
        return native_p * (current[:, SELF_POSITION] + cached[:, SELF_POSITION])

    heads = {
        "native_value": complement + subject(recipient["current"], recipient["cached"]),
        "opposite_current_only": complement + subject(opposite["current"], recipient["cached"]),
        "opposite_cached_only": complement + subject(recipient["current"], opposite["cached"]),
        "opposite_both": complement + subject(opposite["current"], opposite["cached"]),
        "lexical_current_only": complement + subject(lexical["current"], recipient["cached"]),
        "lexical_cached_only": complement + subject(recipient["current"], lexical["cached"]),
        "lexical_both": complement + subject(lexical["current"], lexical["cached"]),
        "complete_opposite_head": opposite["head"],
    }
    indices, replacements, specs, reinstall = [], [], [], []
    for row_index, row in enumerate(rows):
        for condition in CONDITIONS:
            indices.append(row_index)
            replacements.append(heads[condition][row_index])
            specs.append((row_index, condition, f"{row['direction_id']}__{row['template_id']}"))
            reinstall.append(condition == "native_value")
    index = torch.tensor(indices, dtype=torch.long, device=recipient_tokens.device)
    return {
        "tokens": recipient_tokens[index],
        "finals": torch.full_like(index, SELF_POSITION),
        "replacement_heads": torch.stack(replacements),
        "native_reinstall_mask": torch.tensor(reinstall, dtype=torch.bool,
                                               device=recipient_tokens.device),
        "specs": specs, "heads": heads,
    }


def _metrics(logits, row, condition, torch):
    recipient_id = int(row["endpoints"]["recipient"]["answer_id"])
    target_role = "same_number_different_lemma" if condition.startswith("lexical_") \
        or condition == "native_value" else "opposite_same_lemma"
    target_id = int(row["endpoints"][target_role]["answer_id"])
    alternative_id = recipient_id if target_id != recipient_id else int(
        row["endpoints"]["opposite_same_lemma"]["answer_id"])
    by_number = {endpoint["subject_number"]: int(endpoint["answer_id"])
                 for endpoint in row["endpoints"].values()}
    if set(by_number) != {"singular", "plural"}:
        raise CurrentCachedFactorialError("row does not identify singular/plural answer IDs")
    lp = torch.log_softmax(logits, dim=-1)
    return {
        "target_margin": float(logits[target_id] - logits[alternative_id]),
        "target_CE": float(-lp[target_id]),
        "are_minus_is": float(logits[by_number["plural"]] - logits[by_number["singular"]]),
    }


def evaluate(model, torch, F, facade):
    rows = build_rows(); n = len(rows); device = next(model.parameters()).device
    tokens, finals = _role_batch(rows, torch, device)
    native = factors._native_logits(model, tokens, torch, F)
    replay, captured = _factor_forward(model, tokens, finals, torch, F, facade)
    recipient = {key: value[:n] for key, value in captured.items()}
    opposite = {key: value[n:2*n] for key, value in captured.items()}
    lexical = {key: value[2*n:] for key, value in captured.items()}
    patch = _compile(tokens[:n], recipient, opposite, lexical, rows, torch)
    native_patch = factors._native_logits(model, patch["tokens"], torch, F)
    patched, _ = _factor_forward(
        model, patch["tokens"], patch["finals"], torch, F, facade,
        replacement_heads=patch["replacement_heads"],
        native_reinstall_mask=patch["native_reinstall_mask"])
    sides = (recipient, opposite, lexical)
    exactness = {
        "native_replay_max_absolute_logit_error": float((native - replay).abs().max()),
        "source_term_sum_max_absolute_error": max(float((
            torch.einsum("bk,bkd->bd", side["p"], side["u"]) - side["head"]
        ).abs().max()) for side in sides),
        "value_branch_sum_max_absolute_error": max(float((
            side["current"] + side["cached"] - side["u"]
        ).abs().max()) for side in sides),
        "same_batch_native_noop_endpoint_max_absolute_error": 0.0,
        "installed_head_max_absolute_error": 0.0,
        "complete_head_vector_max_absolute_error": float((
            patch["heads"]["complete_opposite_head"] - opposite["head"]
        ).abs().max()),
    }
    evidence = []
    for out_index, (row_index, condition, cell_id) in enumerate(patch["specs"]):
        base = _metrics(native_patch[out_index, SELF_POSITION], rows[row_index], condition, torch)
        value = _metrics(patched[out_index, SELF_POSITION], rows[row_index], condition, torch)
        evidence.append({
            "row_id": rows[row_index]["row_id"], "cell_id": cell_id,
            "condition": condition,
            "target_margin_improvement": value["target_margin"] - base["target_margin"],
            "target_CE_improvement": base["target_CE"] - value["target_CE"],
            "fixed_are_minus_is_change": value["are_minus_is"] - base["are_minus_is"],
        })
        exactness["installed_head_max_absolute_error"] = max(
            exactness["installed_head_max_absolute_error"], float((
                patch["replacement_heads"][out_index] - patch["heads"][condition][row_index]
            ).abs().max()))
        if condition == "native_value":
            exactness["same_batch_native_noop_endpoint_max_absolute_error"] = max(
                exactness["same_batch_native_noop_endpoint_max_absolute_error"],
                float((patched[out_index] - native_patch[out_index]).abs().max()))
    return evidence, exactness


def _fraction(values, positive=True):
    return sum((value > 0) if positive else (value < 0) for value in values) / len(values)


def score(evidence: Sequence[Mapping[str, object]], exactness: Mapping[str, float], bars=BARS):
    expected = {(row["row_id"], f"{row['direction_id']}__{row['template_id']}", condition)
                for row in build_rows() for condition in CONDITIONS}
    observed = [(x.get("row_id"), x.get("cell_id"), x.get("condition")) for x in evidence]
    if len(observed) != len(expected) or set(observed) != expected or len(set(observed)) != len(expected):
        raise CurrentCachedFactorialError("evidence does not cover exact licensed factorial")
    metric_names = ("target_margin_improvement", "target_CE_improvement",
                    "fixed_are_minus_is_change")
    if any(type(x.get(key)) not in (int, float) or not math.isfinite(float(x[key]))
           for x in evidence for key in metric_names):
        raise CurrentCachedFactorialError("non-finite or missing task metric")
    grouped = defaultdict(dict)
    for item in evidence:
        grouped[item["cell_id"]].setdefault(item["condition"], []).append(item)
    cells = {}
    for cell_id, conditions in sorted(grouped.items()):
        def values(condition, metric):
            return [float(x[metric]) for x in conditions[condition]]
        summaries = {}
        for condition in CONDITIONS:
            margins = values(condition, "target_margin_improvement")
            ces = values(condition, "target_CE_improvement")
            summaries[condition] = {
                "mean_margin": statistics.fmean(margins), "mean_CE": statistics.fmean(ces),
                "margin_values": margins, "CE_values": ces,
                "mean_fixed_are_minus_is_change": statistics.fmean(
                    values(condition, "fixed_are_minus_is_change")),
            }
        joint_m = summaries["opposite_both"]["mean_margin"]
        joint_ce = summaries["opposite_both"]["mean_CE"]
        cur_m = summaries["opposite_current_only"]["mean_margin"]
        cache_m = summaries["opposite_cached_only"]["mean_margin"]
        cur_ce = summaries["opposite_current_only"]["mean_CE"]
        cache_ce = summaries["opposite_cached_only"]["mean_CE"]
        native_m = summaries["native_value"]["mean_margin"]
        native_ce = summaries["native_value"]["mean_CE"]
        interaction_m = joint_m - cur_m - cache_m + native_m
        interaction_ce = joint_ce - cur_ce - cache_ce + native_ce
        interaction_margin_values = [
            joint - current - cached + native for joint, current, cached, native in zip(
                summaries["opposite_both"]["margin_values"],
                summaries["opposite_current_only"]["margin_values"],
                summaries["opposite_cached_only"]["margin_values"],
                summaries["native_value"]["margin_values"])]
        interaction_CE_values = [
            joint - current - cached + native for joint, current, cached, native in zip(
                summaries["opposite_both"]["CE_values"],
                summaries["opposite_current_only"]["CE_values"],
                summaries["opposite_cached_only"]["CE_values"],
                summaries["native_value"]["CE_values"])]
        denom = max(abs(joint_m), 1e-12)
        summaries["derived"] = {
            "current_margin_recovery": cur_m / denom,
            "cached_margin_recovery": cache_m / denom,
            "interaction_margin_recovery": interaction_m / denom,
            "interaction_mean_margin": interaction_m,
            "interaction_mean_CE": interaction_ce,
            "interaction_margin_values": interaction_margin_values,
            "interaction_CE_values": interaction_CE_values,
            "maximum_lexical_over_number_joint_margin_ratio": max(
                abs(summaries[name]["mean_margin"]) / denom for name in
                ("lexical_current_only", "lexical_cached_only", "lexical_both")),
        }
        cells[cell_id] = summaries

    exact_live = all(exactness[name] <= bars[bar] for name, bar in (
        ("native_replay_max_absolute_logit_error", "maximum_native_replay_absolute_logit_error"),
        ("source_term_sum_max_absolute_error", "maximum_source_term_sum_absolute_error"),
        ("value_branch_sum_max_absolute_error", "maximum_value_branch_sum_absolute_error"),
        ("same_batch_native_noop_endpoint_max_absolute_error", "maximum_same_batch_native_noop_endpoint_error"),
        ("installed_head_max_absolute_error", "maximum_installed_head_absolute_error"),
        ("complete_head_vector_max_absolute_error", "maximum_complete_head_vector_absolute_error")))

    def helpful(summary, margin_bar):
        return (summary["mean_margin"] >= margin_bar and
                summary["mean_CE"] >= bars["minimum_joint_value_mean_target_CE_improvement"] and
                _fraction(summary["margin_values"]) >= bars["minimum_joint_value_row_improvement_fraction"] and
                _fraction(summary["CE_values"]) >= bars["minimum_joint_value_row_improvement_fraction"])

    complete_live = all(
        helpful(cell["complete_opposite_head"],
                bars["minimum_complete_head_mean_target_margin_improvement"])
        for cell in cells.values())
    joint_live = all(helpful(cell["opposite_both"],
                             bars["minimum_joint_value_mean_target_margin_improvement"])
                     for cell in cells.values())
    instrument = exact_live and complete_live and joint_live
    current = instrument and all(
        helpful(cell["opposite_current_only"],
                bars["minimum_joint_value_mean_target_margin_improvement"])
        and cell["derived"]["current_margin_recovery"] >= bars["minimum_branch_recovery_fraction"]
        for cell in cells.values())
    cached = instrument and all(
        helpful(cell["opposite_cached_only"],
                bars["minimum_joint_value_mean_target_margin_improvement"])
        and cell["derived"]["cached_margin_recovery"] >= bars["minimum_branch_recovery_fraction"]
        for cell in cells.values())
    interaction = instrument and all(
        cell["derived"]["current_margin_recovery"] <= bars["maximum_single_branch_recovery_for_interaction"]
        and cell["derived"]["cached_margin_recovery"] <= bars["maximum_single_branch_recovery_for_interaction"]
        and cell["derived"]["interaction_margin_recovery"] >= bars["minimum_interaction_recovery_fraction"]
        and cell["derived"]["interaction_mean_CE"] >= 0
        and _fraction(cell["derived"]["interaction_margin_values"]) >=
            bars["minimum_joint_value_row_improvement_fraction"]
        and _fraction(cell["derived"]["interaction_CE_values"]) >=
            bars["minimum_joint_value_row_improvement_fraction"]
        for cell in cells.values())
    ratios = [cell["derived"]["maximum_lexical_over_number_joint_margin_ratio"]
              for cell in cells.values()]
    leakage = instrument and any(ratio >= bars["minimum_lexical_leakage_ratio"] for ratio in ratios)
    number_specific = instrument and all(
        ratio <= bars["maximum_number_specific_lexical_ratio"] for ratio in ratios)
    return {**exactness, "cells": cells, "predictions": {
        "pred_a_instrument_live": bool(instrument),
        "pred_b_current_branch_carries_task": bool(current),
        "pred_c_cached_branch_carries_task": bool(cached),
        "pred_d_interaction_is_needed": bool(interaction),
        "pred_e_lexical_leakage": bool(leakage),
        "pred_f_number_specific": bool(number_specific),
    }}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    if OUT.exists():
        raise CurrentCachedFactorialError(f"refusing to overwrite {OUT}")
    torch, F, facade = factors._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        evidence, exactness = evaluate(model, torch, F, facade)
    scored = score(evidence, exactness, plan["bars"])
    terminal = "valid_causal_screen" if scored["predictions"]["pred_a_instrument_live"] else "invalid"
    result = {
        "schema": "task14_head11_3_fresh_matched_subject_current_cached_value_factorial_result_v1",
        "candidate_id": CANDIDATE_ID, "terminal": terminal, "plan": plan,
        "checkpoint_weights_sha256": checkpoint.weights_sha256, "score": scored,
        "evidence": evidence, "evaluated_splits": ["LICENSED_HOLDOUT"],
        "forbidden_splits_opened": [], "model_forwards": 4,
        "causal_interventions": len(evidence),
    }
    payload = managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": scored["predictions"],
                      "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
