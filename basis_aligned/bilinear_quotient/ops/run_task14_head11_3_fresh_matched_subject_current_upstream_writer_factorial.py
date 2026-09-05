#!/usr/bin/env python3
"""Exact E/A/M upstream-writer factorial for Task14 L11H3 current value."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_embedding_carries_task pred_c_attention_carries_task pred_d_MLP_carries_task pred_e_distributed_across_writer_families pred_f_interaction_is_needed pred_g_number_specific pred_h_lexical_collateral

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import circuit_fast_screen_managed_runner as managed
import native_capability_license as licensing
import run_task14_fresh_matched_natural_native_capability as capability
import run_task14_head11_3_fresh_matched_subject_current_cached_value_factorial as value_v1
import run_task14_head11_3_fresh_matched_subject_current_cached_value_factorial_v2 as value_v2
import run_task14_head11_3_subject_attractor_score_payload_factorial as factors
import attention_source_factor_primitive as source_factor


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_head11_3_fresh_matched_subject_current_upstream_writer_factorial_v1.json"
OUT = ROOT / "circuits/fast_screens/task14_head11_3_fresh_matched_subject_current_upstream_writer_factorial_v1_result.json"
LICENSE = ROOT / "circuits/fast_screens/task14_fresh_matched_subject_current_upstream_writer_factorial_v1_capability_license.json"
PRIOR_ART_SHA256 = "999884e00a3e730ccba9f60ae84aa99e55305cc9bcf537c77dbb2f971b6c7ea9"
LICENSE_SHA256 = "f0d1ebba9a7bc8ad5ec9936913a1d2dc5daa4e6f1f72a0571746b4aacc5b9f40"
CANDIDATE_ID = "subject_verb.number_agreement.head11_3_fresh_matched_subject_current_upstream_writer_factorial_v1"
LAYER, HEAD, SELF_POSITION = 11, 3, 8
CONDITIONS = (
    "recipient_EAM", "opposite_E", "opposite_A", "opposite_M",
    "opposite_EA", "opposite_EM", "opposite_AM", "opposite_EAM",
    "lexical_E", "lexical_A", "lexical_M", "lexical_EAM",
)
BARS = {
    "maximum_native_replay_absolute_logit_error": 7e-5,
    "maximum_state_sum_absolute_error": 5e-5,
    "maximum_normalized_state_absolute_error": 5e-5,
    "maximum_source_term_sum_absolute_error": 5e-5,
    "maximum_all_donor_current_head_absolute_error": 5e-5,
    "maximum_same_batch_native_noop_endpoint_error": 7e-5,
    "maximum_installed_head_absolute_error": 5e-5,
    "minimum_all_donor_mean_target_margin_improvement": .05,
    "minimum_all_donor_mean_target_CE_improvement": 0.0,
    "minimum_helpful_row_fraction": .75,
    "minimum_family_recovery_fraction": .70,
    "minimum_distributed_family_recovery_fraction": .25,
    "minimum_distributed_family_count": 2,
    "maximum_family_recovery_for_interaction": .50,
    "minimum_total_interaction_recovery_fraction": .50,
    "maximum_number_specific_lexical_ratio": .25,
    "minimum_lexical_collateral_ratio": .50,
}


class UpstreamWriterFactorialError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rows():
    return value_v1.build_rows()


def validate_preflight():
    if _sha256(PRIOR_ART) != PRIOR_ART_SHA256:
        raise UpstreamWriterFactorialError("prior-art receipt changed")
    return licensing.validate_causal_preflight(
        capability.build_gate(), capability.CAPABILITY_RESULT, LICENSE,
        expected_license_sha256=LICENSE_SHA256, causal_candidate_id=CANDIDATE_ID)


def compile_plan():
    license_value = validate_preflight()
    return {
        "schema": "task14_head11_3_fresh_matched_subject_current_upstream_writer_factorial_plan_v1",
        "candidate_id": CANDIDATE_ID, "split": "LICENSED_HOLDOUT",
        "row_count": len(build_rows()), "conditions": list(CONDITIONS),
        "prior_art_sha256": PRIOR_ART_SHA256, "license_sha256": LICENSE_SHA256,
        "capability_result_sha256": license_value["capability_result_sha256"],
        "preflight_validated": True, "bars": dict(BARS),
        "partition": "exact pre-attention-11 subject state E+A+M",
        "fixed_context": "recipient p_8, cached value branch, and non-subject head complement",
        "price": {"model_forwards": 4, "example_evaluations": 480,
                  "backwards": 0, "parameter_updates": 0},
        "outcomes": ["answer_directed_target_margin_improvement",
                     "target_full_vocab_CE_improvement"],
        "closed_claims": ["individual_attention_or_MLP_block", "downstream_reader",
                          "necessity", "syntax_generality", "FIT", "rank", "reconstruction"],
    }


def _role_batch(rows, torch, device):
    return value_v1._role_batch(rows, torch, device)


def _decomposed_factor_forward(model, tokens, finals, torch, F, facade, *,
                               replacement_heads=None, native_reinstall_mask=None):
    x0 = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
    embedding = x0.clone()
    attention_sum = torch.zeros_like(x0)
    mlp_sum = torch.zeros_like(x0)
    reference = x0.clone()
    captured = {}
    projection = None
    expected_attention_site = 0
    expected_mlp_site = 0
    state_sum_error = 0.0
    normalized_state_error = 0.0

    def attention(event):
        nonlocal embedding, attention_sum, mlp_sum, reference, projection
        nonlocal expected_attention_site, state_sum_error, normalized_state_error
        if event.site != expected_attention_site:
            raise UpstreamWriterFactorialError("attention sites are not sequential")
        expected_attention_site += 1
        residual_scale, skip_scale = event.block.lambdas[0], event.block.lambdas[1]
        reference = residual_scale * reference + skip_scale * x0
        embedding = residual_scale * embedding + skip_scale * x0
        attention_sum = residual_scale * attention_sum
        mlp_sum = residual_scale * mlp_sum
        rebuilt = embedding + attention_sum + mlp_sum
        rebuilt_normalized = F.rms_norm(rebuilt, (rebuilt.size(-1),))
        if event.site == LAYER:
            state_sum_error = float((rebuilt - reference).abs().max())
            normalized_state_error = float((rebuilt_normalized - event.state).abs().max())
            write, base = source_factor.replay_attention_with_source_factors(
                event.state, event.first_value, event.block.attn, finals, HEAD, torch, F)
            current, cached, effective, projection = value_v2._raw_value_branches(
                event.state, event.first_value, event.block.attn, torch, F)
            captured.update({name: value.detach().clone() for name, value in base.items()})
            captured.update({
                "E": embedding.detach().clone(), "A": attention_sum.detach().clone(),
                "M": mlp_sum.detach().clone(), "raw_state": reference.detach().clone(),
                "normalized_state": event.state.detach().clone(),
                "current_pre": current.detach().clone(),
                "cached_pre": cached.detach().clone(),
                "effective_pre": effective.detach().clone(),
            })
            if replacement_heads is not None:
                rows = torch.arange(tokens.size(0), device=tokens.device)
                installed = factors._same_batch_native_heads(
                    replacement_heads, base["head"], native_reinstall_mask, torch)
                write[rows, finals] += (installed - base["head"]).to(write.dtype)
            next_first_value = event.first_value
        else:
            write, next_first_value = event.block.attn(event.state, event.first_value)
        attention_sum = attention_sum + write
        reference = reference + write
        return write, next_first_value

    def mlp(event):
        nonlocal mlp_sum, reference, expected_mlp_site
        if event.site != expected_mlp_site:
            raise UpstreamWriterFactorialError("MLP sites are not sequential")
        expected_mlp_site += 1
        write = event.block.mlp(event.state)
        mlp_sum = mlp_sum + write
        reference = reference + write
        return write

    logits = facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=False).float()
    required = {"p", "u", "head", "E", "A", "M", "raw_state", "normalized_state",
                "current_pre", "cached_pre", "effective_pre"}
    if set(captured) != required or projection is None \
            or expected_attention_site != 18 or expected_mlp_site != 18:
        raise UpstreamWriterFactorialError("decomposed forward audit failed")
    return logits, captured, projection.detach().clone(), {
        "state_sum_max_absolute_error": state_sum_error,
        "normalized_state_max_absolute_error": normalized_state_error,
    }


def _current_from_state(embedding, attention_sum, mlp_sum, attention, torch, F):
    mixed = F.rms_norm(embedding + attention_sum + mlp_sum, (embedding.size(-1),))
    batch, length, width = mixed.shape
    head_width = width // 9
    raw = F.linear(mixed, attention.c_v.weight.to(mixed.dtype)).view(
        batch, length, 9, head_width)
    return (1 - attention.lamb) * raw[:, :, HEAD]


def _compile(recipient_tokens, recipient, opposite, lexical, attention, projection,
             rows, torch, F):
    recipient_terms = recipient["p"].unsqueeze(-1) * recipient["u"]
    complement_mask = torch.arange(recipient_terms.shape[1], device=recipient_terms.device) != SELF_POSITION
    complement = recipient_terms[:, complement_mask].sum(1)
    native_p = recipient["p"][:, SELF_POSITION].unsqueeze(-1)
    components = {"r": recipient, "o": opposite, "l": lexical}
    choices = {
        "recipient_EAM": "rrr",
        "opposite_E": "orr", "opposite_A": "ror", "opposite_M": "rro",
        "opposite_EA": "oor", "opposite_EM": "oro", "opposite_AM": "roo",
        "opposite_EAM": "ooo",
        "lexical_E": "lrr", "lexical_A": "rlr", "lexical_M": "rrl",
        "lexical_EAM": "lll",
    }
    heads = {}
    current_by_condition = {}
    for condition, choice in choices.items():
        current = _current_from_state(
            components[choice[0]]["E"], components[choice[1]]["A"],
            components[choice[2]]["M"], attention, torch, F)
        value = value_v2._project_once(current, recipient["cached_pre"], projection, F)
        heads[condition] = complement + native_p * value[:, SELF_POSITION]
        current_by_condition[condition] = current
    indices, replacements, specs, reinstall = [], [], [], []
    for row_index, row in enumerate(rows):
        for condition in CONDITIONS:
            indices.append(row_index)
            replacements.append(heads[condition][row_index])
            specs.append((row_index, condition, f"{row['direction_id']}__{row['template_id']}"))
            reinstall.append(condition == "recipient_EAM")
    index = torch.tensor(indices, dtype=torch.long, device=recipient_tokens.device)
    return {
        "tokens": recipient_tokens[index], "finals": torch.full_like(index, SELF_POSITION),
        "replacement_heads": torch.stack(replacements),
        "native_reinstall_mask": torch.tensor(reinstall, dtype=torch.bool,
                                               device=recipient_tokens.device),
        "specs": specs, "heads": heads, "current": current_by_condition,
    }


def _metrics(logits, row, condition, torch):
    if condition == "recipient_EAM" or condition.startswith("lexical_"):
        role = "recipient"
    else:
        role = "opposite_same_lemma"
    target = int(row["endpoints"][role]["answer_id"])
    foil = int(row["endpoints"][role]["foil_id"])
    lp = torch.log_softmax(logits, dim=-1)
    return {"target_margin": float(logits[target] - logits[foil]),
            "target_CE": float(-lp[target])}


def evaluate(model, torch, F, facade):
    rows = build_rows(); n = len(rows); device = next(model.parameters()).device
    tokens, finals = _role_batch(rows, torch, device)
    native = factors._native_logits(model, tokens, torch, F)
    replay, captured, projection, closure = _decomposed_factor_forward(
        model, tokens, finals, torch, F, facade)
    recipient = {key: value[:n] for key, value in captured.items()}
    opposite = {key: value[n:2*n] for key, value in captured.items()}
    lexical = {key: value[2*n:] for key, value in captured.items()}
    attention = model.transformer.h[LAYER].attn
    patch = _compile(tokens[:n], recipient, opposite, lexical, attention, projection,
                     rows, torch, F)
    native_patch = factors._native_logits(model, patch["tokens"], torch, F)
    patched, _, _, patch_closure = _decomposed_factor_forward(
        model, patch["tokens"], patch["finals"], torch, F, facade,
        replacement_heads=patch["replacement_heads"],
        native_reinstall_mask=patch["native_reinstall_mask"])
    sides = (recipient, opposite, lexical)
    direct_opposite_value = value_v2._project_once(
        opposite["current_pre"], recipient["cached_pre"], projection, F)
    recipient_terms = recipient["p"].unsqueeze(-1) * recipient["u"]
    complement_mask = torch.arange(recipient_terms.shape[1], device=device) != SELF_POSITION
    direct_opposite_head = recipient_terms[:, complement_mask].sum(1) + \
        recipient["p"][:, SELF_POSITION].unsqueeze(-1) * direct_opposite_value[:, SELF_POSITION]
    exactness = {
        "native_replay_max_absolute_logit_error": float((native - replay).abs().max()),
        "state_sum_max_absolute_error": max(
            closure["state_sum_max_absolute_error"], patch_closure["state_sum_max_absolute_error"]),
        "normalized_state_max_absolute_error": max(
            closure["normalized_state_max_absolute_error"],
            patch_closure["normalized_state_max_absolute_error"]),
        "source_term_sum_max_absolute_error": max(float((
            torch.einsum("bk,bkd->bd", side["p"], side["u"]) - side["head"]
        ).abs().max()) for side in sides),
        "all_donor_current_head_max_absolute_error": float((
            patch["heads"]["opposite_EAM"] - direct_opposite_head).abs().max()),
        "same_batch_native_noop_endpoint_max_absolute_error": 0.0,
        "installed_head_max_absolute_error": 0.0,
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
        })
        exactness["installed_head_max_absolute_error"] = max(
            exactness["installed_head_max_absolute_error"], float((
                patch["replacement_heads"][out_index] - patch["heads"][condition][row_index]
            ).abs().max()))
        if condition == "recipient_EAM":
            exactness["same_batch_native_noop_endpoint_max_absolute_error"] = max(
                exactness["same_batch_native_noop_endpoint_max_absolute_error"],
                float((patched[out_index] - native_patch[out_index]).abs().max()))
    return evidence, exactness


def _fraction(values):
    return sum(value > 0 for value in values) / len(values)


def score(evidence, exactness, bars=BARS):
    expected = {(row["row_id"], f"{row['direction_id']}__{row['template_id']}", condition)
                for row in build_rows() for condition in CONDITIONS}
    observed = [(x.get("row_id"), x.get("cell_id"), x.get("condition")) for x in evidence]
    if len(observed) != len(expected) or set(observed) != expected or len(set(observed)) != len(expected):
        raise UpstreamWriterFactorialError("evidence does not cover exact licensed factorial")
    if any(type(x.get(key)) not in (int, float) or not math.isfinite(float(x[key]))
           for x in evidence for key in ("target_margin_improvement", "target_CE_improvement")):
        raise UpstreamWriterFactorialError("non-finite or missing task metric")
    grouped = defaultdict(dict)
    for item in evidence:
        grouped[item["cell_id"]].setdefault(item["condition"], []).append(item)
    cells = {}
    for cell_id, conditions in sorted(grouped.items()):
        summaries = {}
        for condition in CONDITIONS:
            margins = [float(x["target_margin_improvement"]) for x in conditions[condition]]
            ces = [float(x["target_CE_improvement"]) for x in conditions[condition]]
            summaries[condition] = {"mean_margin": statistics.fmean(margins),
                                    "mean_CE": statistics.fmean(ces),
                                    "margin_values": margins, "CE_values": ces}
        native = summaries["recipient_EAM"]
        all_donor = summaries["opposite_EAM"]
        denominator_margin = max(abs(all_donor["mean_margin"]), 1e-12)
        denominator_CE = max(abs(all_donor["mean_CE"]), 1e-12)
        family_names = {"E": "opposite_E", "A": "opposite_A", "M": "opposite_M"}
        family_recovery = {name: summaries[condition]["mean_margin"] / denominator_margin
                           for name, condition in family_names.items()}

        def mobius(names, metric):
            values = {name: summaries[name][metric] for name in names}
            return values

        pair_values = {}
        for label, joint, left, right in (
                ("EA", "opposite_EA", "opposite_E", "opposite_A"),
                ("EM", "opposite_EM", "opposite_E", "opposite_M"),
                ("AM", "opposite_AM", "opposite_A", "opposite_M")):
            pair_values[label] = {
                metric: [j-l-r+n for j, l, r, n in zip(
                    summaries[joint][metric], summaries[left][metric],
                    summaries[right][metric], native[metric])]
                for metric in ("margin_values", "CE_values")}
        triple = {
            metric: [allv-ea-em-am+e+a+m-nativev for allv, ea, em, am, e, a, m, nativev in zip(
                all_donor[metric], summaries["opposite_EA"][metric],
                summaries["opposite_EM"][metric], summaries["opposite_AM"][metric],
                summaries["opposite_E"][metric], summaries["opposite_A"][metric],
                summaries["opposite_M"][metric], native[metric])]
            for metric in ("margin_values", "CE_values")}
        total_interaction = {
            metric: [allv-e-a-m+2*n for allv, e, a, m, n in zip(
                all_donor[metric], summaries["opposite_E"][metric],
                summaries["opposite_A"][metric], summaries["opposite_M"][metric],
                native[metric])]
            for metric in ("margin_values", "CE_values")}
        total_interaction_mean_margin = statistics.fmean(total_interaction["margin_values"])
        lexical_conditions = ("lexical_E", "lexical_A", "lexical_M", "lexical_EAM")
        lexical_margin_ratio = max(abs(summaries[name]["mean_margin"]) / denominator_margin
                                   for name in lexical_conditions)
        lexical_CE_ratio = max(abs(summaries[name]["mean_CE"]) / denominator_CE
                               for name in lexical_conditions)
        summaries["derived"] = {
            "family_margin_recovery": family_recovery,
            "pair_interactions": pair_values, "triple_interaction": triple,
            "total_interaction": total_interaction,
            "total_interaction_mean_margin": total_interaction_mean_margin,
            "total_interaction_mean_CE": statistics.fmean(total_interaction["CE_values"]),
            "total_interaction_margin_recovery": total_interaction_mean_margin / denominator_margin,
            "maximum_lexical_margin_ratio": lexical_margin_ratio,
            "maximum_lexical_CE_ratio": lexical_CE_ratio,
        }
        cells[cell_id] = summaries

    exact_live = all(exactness[name] <= bars[bar] for name, bar in (
        ("native_replay_max_absolute_logit_error", "maximum_native_replay_absolute_logit_error"),
        ("state_sum_max_absolute_error", "maximum_state_sum_absolute_error"),
        ("normalized_state_max_absolute_error", "maximum_normalized_state_absolute_error"),
        ("source_term_sum_max_absolute_error", "maximum_source_term_sum_absolute_error"),
        ("all_donor_current_head_max_absolute_error", "maximum_all_donor_current_head_absolute_error"),
        ("same_batch_native_noop_endpoint_max_absolute_error", "maximum_same_batch_native_noop_endpoint_error"),
        ("installed_head_max_absolute_error", "maximum_installed_head_absolute_error")))

    def helpful(summary):
        return (summary["mean_margin"] >= bars["minimum_all_donor_mean_target_margin_improvement"]
                and summary["mean_CE"] >= bars["minimum_all_donor_mean_target_CE_improvement"]
                and _fraction(summary["margin_values"]) >= bars["minimum_helpful_row_fraction"]
                and _fraction(summary["CE_values"]) >= bars["minimum_helpful_row_fraction"])

    instrument = exact_live and all(helpful(cell["opposite_EAM"]) for cell in cells.values())
    family_conditions = {"E": "opposite_E", "A": "opposite_A", "M": "opposite_M"}

    def family_holds(family):
        condition = family_conditions[family]
        return instrument and all(
            helpful(cell[condition]) and
            cell["derived"]["family_margin_recovery"][family] >= bars["minimum_family_recovery_fraction"]
            for cell in cells.values())

    family_predictions = {family: family_holds(family) for family in family_conditions}
    distributed = instrument and all(
        max(cell["derived"]["family_margin_recovery"].values()) <
            bars["minimum_family_recovery_fraction"]
        and sum(
            cell["derived"]["family_margin_recovery"][family] >=
                bars["minimum_distributed_family_recovery_fraction"]
            and helpful(cell[condition])
            for family, condition in family_conditions.items()) >=
                bars["minimum_distributed_family_count"]
        for cell in cells.values())
    interaction = instrument and all(
        max(cell["derived"]["family_margin_recovery"].values()) <=
            bars["maximum_family_recovery_for_interaction"]
        and cell["derived"]["total_interaction_margin_recovery"] >=
            bars["minimum_total_interaction_recovery_fraction"]
        and cell["derived"]["total_interaction_mean_CE"] >= 0
        and _fraction(cell["derived"]["total_interaction"]["margin_values"]) >=
            bars["minimum_helpful_row_fraction"]
        and _fraction(cell["derived"]["total_interaction"]["CE_values"]) >=
            bars["minimum_helpful_row_fraction"]
        for cell in cells.values())
    number_specific = instrument and all(
        cell["derived"]["maximum_lexical_margin_ratio"] <=
            bars["maximum_number_specific_lexical_ratio"]
        and cell["derived"]["maximum_lexical_CE_ratio"] <=
            bars["maximum_number_specific_lexical_ratio"]
        for cell in cells.values())
    collateral = instrument and any(
        cell["derived"]["maximum_lexical_margin_ratio"] >= bars["minimum_lexical_collateral_ratio"]
        or cell["derived"]["maximum_lexical_CE_ratio"] >= bars["minimum_lexical_collateral_ratio"]
        for cell in cells.values())
    return {**exactness, "cells": cells, "predictions": {
        "pred_a_instrument_live": bool(instrument),
        "pred_b_embedding_carries_task": bool(family_predictions["E"]),
        "pred_c_attention_carries_task": bool(family_predictions["A"]),
        "pred_d_MLP_carries_task": bool(family_predictions["M"]),
        "pred_e_distributed_across_writer_families": bool(distributed),
        "pred_f_interaction_is_needed": bool(interaction),
        "pred_g_number_specific": bool(number_specific),
        "pred_h_lexical_collateral": bool(collateral),
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
        raise UpstreamWriterFactorialError(f"refusing to overwrite {OUT}")
    torch, F, facade = factors._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        evidence, exactness = evaluate(model, torch, F, facade)
    scored = score(evidence, exactness, plan["bars"])
    terminal = "valid_causal_screen" if scored["predictions"]["pred_a_instrument_live"] else "invalid"
    result = {
        "schema": "task14_head11_3_fresh_matched_subject_current_upstream_writer_factorial_result_v1",
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
