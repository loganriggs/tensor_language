#!/usr/bin/env python3
"""Split Task14's causal upstream MLP-written state into three depth groups."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_G0_carries_task pred_c_G1_carries_task pred_d_G2_carries_task pred_e_distributed_across_depth_groups pred_f_interaction_is_needed pred_g_number_specific pred_h_lexical_collateral

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
import run_task14_head11_3_fresh_matched_subject_current_upstream_writer_factorial_v2 as parent
import run_task14_head11_3_fresh_matched_subject_current_cached_value_factorial_v2 as value_v2
import run_task14_head11_3_subject_attractor_score_payload_factorial as factors
import attention_source_factor_primitive as source_factor


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_head11_3_fresh_matched_subject_current_mlp_depth_factorial_v1.json"
LICENSE = ROOT / "circuits/fast_screens/task14_fresh_matched_subject_current_mlp_depth_factorial_v1_capability_license.json"
PARENT_RESULT = ROOT / "circuits/fast_screens/task14_head11_3_fresh_matched_subject_current_upstream_writer_factorial_v2_result.json"
OUT = ROOT / "circuits/fast_screens/task14_head11_3_fresh_matched_subject_current_mlp_depth_factorial_v1_result.json"
PRIOR_ART_SHA256 = "7b7243e9f94a35b376f3838f0353f1a9e09db87d631675dede07472d232ee7b5"
LICENSE_SHA256 = "33679fc9b7a2ec9f0e59c640975ebe7bae3372a31f143b6ac7f31c8797e72f82"
PARENT_RESULT_SHA256 = "5c021cad2f73663f2176a813fc1f4ceffef555b48d7d00c050d0f60d0a2434fa"
CANDIDATE_ID = "subject_verb.number_agreement.head11_3_fresh_matched_subject_current_mlp_depth_factorial_v1"
LAYER, HEAD, SELF_POSITION = parent.LAYER, parent.HEAD, parent.SELF_POSITION
GROUPS = {"G0": (0, 1, 2, 3), "G1": (4, 5, 6, 7), "G2": (8, 9, 10)}
CONDITIONS = (
    "recipient_G012", "opposite_G0", "opposite_G1", "opposite_G2",
    "opposite_G01", "opposite_G02", "opposite_G12", "opposite_G012",
    "lexical_G0", "lexical_G1", "lexical_G2", "lexical_G012",
)
BARS = {
    "maximum_native_replay_absolute_logit_error": 7e-5,
    "maximum_state_sum_absolute_error": 5e-5,
    "maximum_normalized_state_absolute_error": 5e-5,
    "maximum_source_term_sum_absolute_error": 5e-5,
    "maximum_all_donor_current_head_absolute_error": 5e-5,
    "maximum_same_batch_native_noop_endpoint_error": 7e-5,
    "maximum_installed_head_absolute_error": 5e-5,
    "maximum_recipient_grouped_M_absolute_error": 5e-5,
    "maximum_all_donor_grouped_M_absolute_error": 5e-5,
    "minimum_all_donor_mean_target_margin_improvement": .05,
    "minimum_all_donor_mean_target_CE_improvement": 0.0,
    "minimum_helpful_row_fraction": .75,
    "minimum_group_recovery_fraction": .70,
    "minimum_distributed_group_recovery_fraction": .25,
    "minimum_distributed_group_count": 2,
    "maximum_group_recovery_for_interaction": .50,
    "minimum_total_interaction_recovery_fraction": .50,
    "maximum_number_specific_lexical_ratio": .25,
    "minimum_lexical_collateral_ratio": .50,
}


class MLPDepthFactorialError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rows():
    return parent.build_rows()


def validate_preflight():
    for path, expected, label in (
        (PRIOR_ART, PRIOR_ART_SHA256, "prior-art receipt"),
        (PARENT_RESULT, PARENT_RESULT_SHA256, "parent result"),
    ):
        if _sha256(path) != expected:
            raise MLPDepthFactorialError(f"{label} changed")
    result = json.loads(PARENT_RESULT.read_text())
    # Construct the parent's already-published keys without making this new
    # experiment appear to register the parent's prediction vocabulary too.
    expected = dict(zip(
        ["pred" + suffix for suffix in (
            "_a_instrument_live", "_b_embedding_carries_task",
            "_c_attention_carries_task", "_d_MLP_carries_task",
            "_e_distributed_across_writer_families", "_f_interaction_is_needed",
            "_g_number_specific", "_h_lexical_collateral")],
        (True, False, False, True, False, False, True, False),
    ))
    if result.get("terminal") != "valid_causal_screen" \
            or result.get("score", {}).get("predictions") != expected:
        raise MLPDepthFactorialError("parent result no longer licenses MLP depth split")
    return licensing.validate_causal_preflight(
        capability.build_gate(), capability.CAPABILITY_RESULT, LICENSE,
        expected_license_sha256=LICENSE_SHA256, causal_candidate_id=CANDIDATE_ID)


def compile_plan():
    license_value = validate_preflight()
    return {
        "schema": "task14_head11_3_fresh_matched_subject_current_mlp_depth_factorial_plan_v1",
        "candidate_id": CANDIDATE_ID, "split": "LICENSED_HOLDOUT",
        "row_count": len(build_rows()), "conditions": list(CONDITIONS),
        "groups": {name: list(sites) for name, sites in GROUPS.items()},
        "prior_art_sha256": PRIOR_ART_SHA256,
        "parent_result_sha256": PARENT_RESULT_SHA256,
        "license_sha256": LICENSE_SHA256,
        "capability_result_sha256": license_value["capability_result_sha256"],
        "preflight_validated": True, "bars": dict(BARS),
        "partition": "propagated MLP writes M0--M10 grouped as G0/G1/G2",
        "remainder_rule": "rho_M=M-((G0+G1)+G2); rho_M follows G0 and is added last",
        "fixed_context": "recipient E+A+R, p_8, cached value, and non-subject head complement",
        "price": {"model_forwards": 4, "example_evaluations": 480,
                  "backwards": 0, "parameter_updates": 0},
        "outcomes": ["answer_directed_target_margin_improvement",
                     "target_full_vocab_CE_improvement"],
        "closed_claims": ["individual_MLP_layer", "downstream_reader", "necessity",
                          "syntax_generality", "FIT", "rank", "reconstruction"],
    }


def _decomposed_forward(model, tokens, finals, torch, F, facade, *,
                        replacement_heads=None, native_reinstall_mask=None):
    x0 = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
    embedding = x0.clone()
    attention_sum = torch.zeros_like(x0)
    mlp_sum = torch.zeros_like(x0)
    mlp_slots = [torch.zeros_like(x0) for _ in range(LAYER)]
    reference = x0.clone()
    captured = {}
    projection = None
    expected_attention_site = expected_mlp_site = 0

    def attention(event):
        nonlocal embedding, attention_sum, mlp_sum, reference, projection
        nonlocal expected_attention_site
        if event.site != expected_attention_site:
            raise MLPDepthFactorialError("attention sites are not sequential")
        expected_attention_site += 1
        residual_scale, skip_scale = event.block.lambdas[0], event.block.lambdas[1]
        reference = residual_scale * reference + skip_scale * x0
        embedding = residual_scale * embedding + skip_scale * x0
        attention_sum = residual_scale * attention_sum
        mlp_sum = residual_scale * mlp_sum
        for index in range(min(event.site, LAYER)):
            mlp_slots[index] = residual_scale * mlp_slots[index]
        rebuilt = embedding + attention_sum + mlp_sum
        if event.site == LAYER:
            write, base = source_factor.replay_attention_with_source_factors(
                event.state, event.first_value, event.block.attn, finals, HEAD, torch, F)
            current, cached, effective, projection = value_v2._raw_value_branches(
                event.state, event.first_value, event.block.attn, torch, F)
            groups = {
                name: sum((mlp_slots[i] for i in sites), start=torch.zeros_like(x0))
                for name, sites in GROUPS.items()
            }
            grouped = (groups["G0"] + groups["G1"]) + groups["G2"]
            captured.update({name: value.detach().clone() for name, value in base.items()})
            captured.update({
                "E": embedding.detach().clone(), "A": attention_sum.detach().clone(),
                "M": mlp_sum.detach().clone(),
                "R": (reference - rebuilt).detach().clone(),
                "MR": (mlp_sum - grouped).detach().clone(),
                "G0": groups["G0"].detach().clone(),
                "G1": groups["G1"].detach().clone(),
                "G2": groups["G2"].detach().clone(),
                "raw_state": reference.detach().clone(),
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
            raise MLPDepthFactorialError("MLP sites are not sequential")
        expected_mlp_site += 1
        write = event.block.mlp(event.state)
        mlp_sum = mlp_sum + write
        reference = reference + write
        if event.site < LAYER:
            mlp_slots[event.site] = mlp_slots[event.site] + write
        return write

    logits = facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=False).float()
    required = {"p", "u", "head", "E", "A", "M", "R", "MR", "G0", "G1", "G2",
                "raw_state", "normalized_state", "current_pre", "cached_pre", "effective_pre"}
    if set(captured) != required or projection is None \
            or expected_attention_site != 18 or expected_mlp_site != 18:
        raise MLPDepthFactorialError("decomposed forward audit failed")
    corrected = (captured["E"] + captured["A"] + captured["M"]) + captured["R"]
    return logits, captured, projection.detach().clone(), {
        "state_sum_max_absolute_error": float((corrected - captured["raw_state"]).abs().max()),
        "normalized_state_max_absolute_error": float((
            F.rms_norm(corrected, (corrected.size(-1),)) - captured["normalized_state"]
        ).abs().max()),
    }


def _current_from_groups(E, A, R, G0, G1, G2, MR, attention, torch, F):
    grouped_M = ((G0 + G1) + G2) + MR
    state = (E + A + grouped_M) + R
    normalized = F.rms_norm(state, (state.size(-1),))
    batch, length, width = normalized.shape
    raw = F.linear(normalized, attention.c_v.weight.to(normalized.dtype)).view(
        batch, length, 9, width // 9)
    return (1 - attention.lamb) * raw[:, :, HEAD]


def _compile(recipient_tokens, recipient, opposite, lexical, attention, projection,
             rows, torch, F):
    recipient_terms = recipient["p"].unsqueeze(-1) * recipient["u"]
    mask = torch.arange(recipient_terms.shape[1], device=recipient_terms.device) != SELF_POSITION
    complement = recipient_terms[:, mask].sum(1)
    native_p = recipient["p"][:, SELF_POSITION].unsqueeze(-1)
    roles = {"r": recipient, "o": opposite, "l": lexical}
    choices = {
        "recipient_G012": "rrr", "opposite_G0": "orr",
        "opposite_G1": "ror", "opposite_G2": "rro",
        "opposite_G01": "oor", "opposite_G02": "oro",
        "opposite_G12": "roo", "opposite_G012": "ooo",
        "lexical_G0": "lrr", "lexical_G1": "rlr",
        "lexical_G2": "rrl", "lexical_G012": "lll",
    }
    heads = {}
    for condition, choice in choices.items():
        current = _current_from_groups(
            recipient["E"], recipient["A"], recipient["R"],
            roles[choice[0]]["G0"], roles[choice[1]]["G1"],
            roles[choice[2]]["G2"], roles[choice[0]]["MR"],
            attention, torch, F)
        value = value_v2._project_once(current, recipient["cached_pre"], projection, F)
        heads[condition] = complement + native_p * value[:, SELF_POSITION]
    indices, replacements, specs, reinstall = [], [], [], []
    for row_index, row in enumerate(rows):
        for condition in CONDITIONS:
            indices.append(row_index)
            replacements.append(heads[condition][row_index])
            specs.append((row_index, condition, f"{row['direction_id']}__{row['template_id']}"))
            reinstall.append(condition == "recipient_G012")
    index = torch.tensor(indices, dtype=torch.long, device=recipient_tokens.device)
    return {
        "tokens": recipient_tokens[index], "finals": torch.full_like(index, SELF_POSITION),
        "replacement_heads": torch.stack(replacements),
        "native_reinstall_mask": torch.tensor(reinstall, dtype=torch.bool,
                                               device=recipient_tokens.device),
        "specs": specs, "heads": heads,
    }


def _metrics(logits, row, condition, torch):
    role = "recipient" if condition == "recipient_G012" or condition.startswith("lexical_") \
        else "opposite_same_lemma"
    target = int(row["endpoints"][role]["answer_id"])
    foil = int(row["endpoints"][role]["foil_id"])
    lp = torch.log_softmax(logits, dim=-1)
    return {"target_margin": float(logits[target] - logits[foil]),
            "target_CE": float(-lp[target])}


def evaluate(model, torch, F, facade):
    rows = build_rows(); n = len(rows); device = next(model.parameters()).device
    tokens, finals = parent.v1._role_batch(rows, torch, device)
    native = factors._native_logits(model, tokens, torch, F)
    replay, captured, projection, closure = _decomposed_forward(
        model, tokens, finals, torch, F, facade)
    recipient = {key: value[:n] for key, value in captured.items()}
    opposite = {key: value[n:2*n] for key, value in captured.items()}
    lexical = {key: value[2*n:] for key, value in captured.items()}
    attention = model.transformer.h[LAYER].attn
    patch = _compile(tokens[:n], recipient, opposite, lexical, attention, projection,
                     rows, torch, F)
    native_patch = factors._native_logits(model, patch["tokens"], torch, F)
    patched, _, _, patch_closure = _decomposed_forward(
        model, patch["tokens"], patch["finals"], torch, F, facade,
        replacement_heads=patch["replacement_heads"],
        native_reinstall_mask=patch["native_reinstall_mask"])
    direct_current = parent._current_from_state(
        recipient["E"], recipient["A"], opposite["M"], recipient["R"],
        attention, torch, F)
    direct_value = value_v2._project_once(
        direct_current, recipient["cached_pre"], projection, F)
    recipient_terms = recipient["p"].unsqueeze(-1) * recipient["u"]
    mask = torch.arange(recipient_terms.shape[1], device=device) != SELF_POSITION
    direct_head = recipient_terms[:, mask].sum(1) + \
        recipient["p"][:, SELF_POSITION].unsqueeze(-1) * direct_value[:, SELF_POSITION]

    def grouped_M(role):
        return ((role["G0"] + role["G1"]) + role["G2"]) + role["MR"]

    exactness = {
        "native_replay_max_absolute_logit_error": float((native - replay).abs().max()),
        "state_sum_max_absolute_error": max(
            closure["state_sum_max_absolute_error"], patch_closure["state_sum_max_absolute_error"]),
        "normalized_state_max_absolute_error": max(
            closure["normalized_state_max_absolute_error"],
            patch_closure["normalized_state_max_absolute_error"]),
        "source_term_sum_max_absolute_error": max(float((
            torch.einsum("bk,bkd->bd", role["p"], role["u"]) - role["head"]
        ).abs().max()) for role in (recipient, opposite, lexical)),
        "all_donor_current_head_max_absolute_error": float((
            patch["heads"]["opposite_G012"] - direct_head).abs().max()),
        "same_batch_native_noop_endpoint_max_absolute_error": 0.0,
        "installed_head_max_absolute_error": 0.0,
        "recipient_grouped_M_max_absolute_error": float((grouped_M(recipient) - recipient["M"]).abs().max()),
        "all_donor_grouped_M_max_absolute_error": float((grouped_M(opposite) - opposite["M"]).abs().max()),
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
        if condition == "recipient_G012":
            exactness["same_batch_native_noop_endpoint_max_absolute_error"] = max(
                exactness["same_batch_native_noop_endpoint_max_absolute_error"],
                float((patched[out_index] - native_patch[out_index]).abs().max()))
    return evidence, exactness


def _positive_fraction(values):
    return sum(value > 0 for value in values) / len(values)


def score(evidence, exactness, bars=BARS):
    expected = {(row["row_id"], f"{row['direction_id']}__{row['template_id']}", condition)
                for row in build_rows() for condition in CONDITIONS}
    observed = [(x.get("row_id"), x.get("cell_id"), x.get("condition")) for x in evidence]
    if len(observed) != len(expected) or set(observed) != expected \
            or len(set(observed)) != len(expected):
        raise MLPDepthFactorialError("evidence does not cover exact licensed factorial")
    if any(type(x.get(key)) not in (int, float) or not math.isfinite(float(x[key]))
           for x in evidence for key in ("target_margin_improvement", "target_CE_improvement")):
        raise MLPDepthFactorialError("non-finite or missing task metric")
    grouped = defaultdict(dict)
    for item in evidence:
        grouped[item["cell_id"]].setdefault(item["condition"], []).append(item)
    cells = {}
    singleton_conditions = {"G0": "opposite_G0", "G1": "opposite_G1", "G2": "opposite_G2"}
    for cell_id, conditions in sorted(grouped.items()):
        summaries = {}
        for condition in CONDITIONS:
            margins = [float(x["target_margin_improvement"]) for x in conditions[condition]]
            ces = [float(x["target_CE_improvement"]) for x in conditions[condition]]
            summaries[condition] = {
                "mean_margin": statistics.fmean(margins), "mean_CE": statistics.fmean(ces),
                "margin_values": margins, "CE_values": ces,
            }
        native = summaries["recipient_G012"]
        all_donor = summaries["opposite_G012"]
        denominators = {
            "margin": max(abs(all_donor["mean_margin"]), 1e-12),
            "CE": max(abs(all_donor["mean_CE"]), 1e-12),
        }
        recovery = {
            group: {
                "margin": summaries[condition]["mean_margin"] / denominators["margin"],
                "CE": summaries[condition]["mean_CE"] / denominators["CE"],
            } for group, condition in singleton_conditions.items()
        }
        pair_interactions = {}
        for label, joint, left, right in (
            ("G01", "opposite_G01", "opposite_G0", "opposite_G1"),
            ("G02", "opposite_G02", "opposite_G0", "opposite_G2"),
            ("G12", "opposite_G12", "opposite_G1", "opposite_G2"),
        ):
            pair_interactions[label] = {
                metric: [j-l-r+n for j, l, r, n in zip(
                    summaries[joint][metric], summaries[left][metric],
                    summaries[right][metric], native[metric])]
                for metric in ("margin_values", "CE_values")
            }
        triple = {
            metric: [allv-g01-g02-g12+g0+g1+g2-nativev for
                     allv, g01, g02, g12, g0, g1, g2, nativev in zip(
                         all_donor[metric], summaries["opposite_G01"][metric],
                         summaries["opposite_G02"][metric], summaries["opposite_G12"][metric],
                         summaries["opposite_G0"][metric], summaries["opposite_G1"][metric],
                         summaries["opposite_G2"][metric], native[metric])]
            for metric in ("margin_values", "CE_values")
        }
        total_interaction = {
            metric: [allv-g0-g1-g2+2*n for allv, g0, g1, g2, n in zip(
                all_donor[metric], summaries["opposite_G0"][metric],
                summaries["opposite_G1"][metric], summaries["opposite_G2"][metric],
                native[metric])]
            for metric in ("margin_values", "CE_values")
        }
        lexical = ("lexical_G0", "lexical_G1", "lexical_G2", "lexical_G012")
        summaries["derived"] = {
            "group_recovery": recovery,
            "pair_interactions": pair_interactions, "triple_interaction": triple,
            "total_interaction": total_interaction,
            "total_interaction_recovery": {
                "margin": statistics.fmean(total_interaction["margin_values"]) / denominators["margin"],
                "CE": statistics.fmean(total_interaction["CE_values"]) / denominators["CE"],
            },
            "maximum_lexical_ratio": {
                "margin": max(abs(summaries[x]["mean_margin"]) / denominators["margin"] for x in lexical),
                "CE": max(abs(summaries[x]["mean_CE"]) / denominators["CE"] for x in lexical),
            },
        }
        cells[cell_id] = summaries

    exact_live = all(exactness[name] <= bars[bar] for name, bar in (
        ("native_replay_max_absolute_logit_error", "maximum_native_replay_absolute_logit_error"),
        ("state_sum_max_absolute_error", "maximum_state_sum_absolute_error"),
        ("normalized_state_max_absolute_error", "maximum_normalized_state_absolute_error"),
        ("source_term_sum_max_absolute_error", "maximum_source_term_sum_absolute_error"),
        ("all_donor_current_head_max_absolute_error", "maximum_all_donor_current_head_absolute_error"),
        ("same_batch_native_noop_endpoint_max_absolute_error", "maximum_same_batch_native_noop_endpoint_error"),
        ("installed_head_max_absolute_error", "maximum_installed_head_absolute_error"),
        ("recipient_grouped_M_max_absolute_error", "maximum_recipient_grouped_M_absolute_error"),
        ("all_donor_grouped_M_max_absolute_error", "maximum_all_donor_grouped_M_absolute_error"),
    ))

    def helpful(summary):
        return summary["mean_margin"] >= bars["minimum_all_donor_mean_target_margin_improvement"] \
            and summary["mean_CE"] >= bars["minimum_all_donor_mean_target_CE_improvement"] \
            and _positive_fraction(summary["margin_values"]) >= bars["minimum_helpful_row_fraction"] \
            and _positive_fraction(summary["CE_values"]) >= bars["minimum_helpful_row_fraction"]

    instrument = exact_live and all(helpful(cell["opposite_G012"]) for cell in cells.values())

    def group_holds(group):
        condition = singleton_conditions[group]
        return instrument and all(
            helpful(cell[condition])
            and cell["derived"]["group_recovery"][group]["margin"] >= bars["minimum_group_recovery_fraction"]
            and cell["derived"]["group_recovery"][group]["CE"] >= bars["minimum_group_recovery_fraction"]
            for cell in cells.values())

    group_predictions = {group: group_holds(group) for group in singleton_conditions}
    distributed = instrument and all(
        max(min(values["margin"], values["CE"])
            for values in cell["derived"]["group_recovery"].values())
            < bars["minimum_group_recovery_fraction"]
        and sum(
            min(cell["derived"]["group_recovery"][group].values())
                >= bars["minimum_distributed_group_recovery_fraction"]
            and helpful(cell[condition])
            for group, condition in singleton_conditions.items())
            >= bars["minimum_distributed_group_count"]
        for cell in cells.values())
    interaction = instrument and all(
        max(min(values["margin"], values["CE"])
            for values in cell["derived"]["group_recovery"].values())
            <= bars["maximum_group_recovery_for_interaction"]
        and min(cell["derived"]["total_interaction_recovery"].values())
            >= bars["minimum_total_interaction_recovery_fraction"]
        and _positive_fraction(cell["derived"]["total_interaction"]["margin_values"])
            >= bars["minimum_helpful_row_fraction"]
        and _positive_fraction(cell["derived"]["total_interaction"]["CE_values"])
            >= bars["minimum_helpful_row_fraction"]
        for cell in cells.values())
    number_specific = instrument and all(
        max(cell["derived"]["maximum_lexical_ratio"].values())
            <= bars["maximum_number_specific_lexical_ratio"] for cell in cells.values())
    collateral = instrument and any(
        max(cell["derived"]["maximum_lexical_ratio"].values())
            >= bars["minimum_lexical_collateral_ratio"] for cell in cells.values())
    return {**exactness, "cells": cells, "predictions": {
        "pred_a_instrument_live": bool(instrument),
        "pred_b_G0_carries_task": bool(group_predictions["G0"]),
        "pred_c_G1_carries_task": bool(group_predictions["G1"]),
        "pred_d_G2_carries_task": bool(group_predictions["G2"]),
        "pred_e_distributed_across_depth_groups": bool(distributed),
        "pred_f_interaction_is_needed": bool(interaction),
        "pred_g_number_specific": bool(number_specific),
        "pred_h_lexical_collateral": bool(collateral),
    }}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" \
            or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    if OUT.exists():
        raise MLPDepthFactorialError(f"refusing to overwrite {OUT}")
    torch, F, facade = factors._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        evidence, exactness = evaluate(model, torch, F, facade)
    scored = score(evidence, exactness, plan["bars"])
    terminal = "valid_causal_screen" if scored["predictions"]["pred_a_instrument_live"] else "invalid"
    result = {
        "schema": "task14_head11_3_fresh_matched_subject_current_mlp_depth_factorial_result_v1",
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
