#!/usr/bin/env python3
"""Fixed H3+H7 list/digit exact-factor interchange, with active controls."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_shared_payload_private_router pred_c_shared_score_and_payload pred_d_generic_numeral_or_copy_bus

from __future__ import annotations

from collections import defaultdict
import json
import math
import os
from pathlib import Path
import statistics
import sys

import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import circuit_fast_screen_candidate_attn8_h3_h7_cross_behavior_factor_interchange as authority
import numbered_list_factor_localization_rung573 as r573
import numeric_sequence_complete_state_factor_localization_rung577 as r577


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits/fast_screens/attn8_h3_h7_cross_behavior_factor_interchange_v2_result.json"
PRIOR_ART_SHA256 = "63bc3120a8a694a80dee9587e27503798b0bd71cfc015ba5ea90098244692885"
ARMS = ("within_score", "within_cached", "within_joint", "cross_score",
        "cross_cached", "cross_joint", "cross_same_joint")
CONTROL_ARMS = ("cross_score", "cross_cached", "cross_joint")
HEADS = r573.HEADS


def build_rows():
    return authority.build_rows()


def compile_plan():
    rows = build_rows()
    return {"schema": "attn8_h3_h7_cross_behavior_factor_interchange_plan_v2",
            "candidate_id": "numeric_successor.attn8_h3_h7_cross_behavior_factor_interchange_v2",
            "authority_sha256": authority.validate_rows(rows), "screen_tier": "BASIC",
            "splits": ["FIT", "SELECT"], "target_arms": list(ARMS),
            "control_arms": list(CONTROL_ARMS), "fixed_heads": [3, 7],
            "price": {"model_forwards": 6, "example_evaluations": 720,
                      "backwards": 0, "parameter_updates": 0},
            "bars": {"minimum_native_accuracy": .85,
                     "maximum_native_replay_relative_squared_error": 1e-10,
                     "maximum_source_sum_relative_squared_error": 1e-10,
                     "maximum_cached_decomposition_relative_squared_error": 1e-10,
                     "maximum_installed_term_absolute_error": 1e-5,
                     "minimum_within_joint_natural_recovery": .50,
                     "minimum_target_direction_fraction": .75,
                     "minimum_cross_cached_over_within_cached": .70,
                     "minimum_cross_joint_over_within_joint": .70,
                     "minimum_cross_score_over_within_score": .50,
                     "minimum_control_intervention_norm_fraction": .10,
                     "minimum_control_preference_preservation": .75,
                     "maximum_control_absolute_mean_ce_change": .10,
                     "maximum_control_median_margin_change_fraction": .25},
            "closed_claims": ["abstract_numeral_concept", "single_head",
                              "earlier_source", "fresh_values_templates", "word_number_transfer"]}


def _pad(examples, device):
    length = max(len(item["ids"]) for item in examples)
    tokens = torch.full((len(examples), length), 50256, dtype=torch.long, device=device)
    finals, positions = [], []
    for index, item in enumerate(examples):
        tokens[index, :len(item["ids"])] = torch.tensor(item["ids"], device=device)
        finals.append(item["query_position"]); positions.append(item["source_positions"])
    return tokens, torch.tensor(finals, device=device), torch.tensor(positions, device=device)


@torch.no_grad()
def _capture_forward(model, tokens, finals, positions):
    captured, diagnostics = {}, {"head_source_sum_relative_squared_error": 0.,
                                  "value_split_relative_squared_error": 0.}
    rows = torch.arange(tokens.shape[0], device=tokens.device)

    def attention(event):
        if event.site != 8:
            return event.block.attn(event.state, event.first_value)
        write, tensors, errors = r573.replay_attention(
            event.state, event.first_value, event.block.attn, finals)
        diagnostics.update(errors)
        captured["complete"] = torch.stack(
            [tensors["head_output"][rows, finals, head] for head in HEADS], 1)
        captured["score"] = torch.stack([torch.stack(
            [tensors["pattern"][rows, head, finals, positions[:, ordinal]]
             for ordinal in range(3)], 1) for head in HEADS], 1)
        for name in ("own", "cached", "value"):
            captured[name] = torch.stack([torch.stack(
                [tensors[name][rows, positions[:, ordinal], head]
                 for ordinal in range(3)], 1) for head in HEADS], 1)
        return write, event.first_value
    logits = r573.facade.forward_with_dispatch(
        model, tokens, attention, lambda event: event.block.mlp(event.state),
        require_production=False).float()
    return logits[rows, finals], captured, diagnostics


@torch.no_grad()
def _patched_forward(model, tokens, finals, replacements):
    rows = torch.arange(tokens.shape[0], device=tokens.device)
    diagnostics = {"head_source_sum_relative_squared_error": 0.,
                   "value_split_relative_squared_error": 0.}
    norms = None
    installed_error = 0.0
    def attention(event):
        nonlocal norms, installed_error
        if event.site != 8:
            return event.block.attn(event.state, event.first_value)
        write, tensors, errors = r573.replay_attention(
            event.state, event.first_value, event.block.attn, finals)
        diagnostics.update(errors)
        native = torch.stack([tensors["head_output"][rows, finals, head] for head in HEADS], 1)
        changed = tensors["head_output"].clone()
        for slot, head in enumerate(HEADS):
            changed[rows, finals, head] = replacements[:, slot]
        installed = torch.stack([changed[rows, finals, head] for head in HEADS], 1)
        installed_error = float((installed.float() - replacements.float()).abs().max())
        delta = torch.zeros(tokens.shape[0], r573.N_HEAD, r573.HEAD_D, device=tokens.device)
        for slot, head in enumerate(HEADS): delta[:, head] = replacements[:, slot] - native[:, slot]
        norms = r573.linear(delta.reshape(tokens.shape[0], r573.D),
                            event.block.attn.c_proj.weight).float().norm(dim=-1)
        return r573.linear(changed.reshape(tokens.shape[0], tokens.shape[1], r573.D),
                           event.block.attn.c_proj.weight), event.first_value
    logits = r573.facade.forward_with_dispatch(
        model, tokens, attention, lambda event: event.block.mlp(event.state),
        require_production=False).float()
    return logits[rows, finals], norms, diagnostics, installed_error


def _replace(recipient, donor, kind):
    changed = recipient["complete"].clone()
    ordinal = 2
    for slot in range(2):
        native_score = recipient["score"][slot, ordinal]
        native_value = recipient["value"][slot, ordinal]
        if kind == "score": replacement = donor["score"][slot, ordinal] * native_value
        elif kind == "cached":
            replacement = native_score * (recipient["own"][slot, ordinal]
                                          + donor["cached"][slot, ordinal])
        elif kind == "joint":
            replacement = donor["score"][slot, ordinal] * (
                recipient["own"][slot, ordinal] + donor["cached"][slot, ordinal])
        else: raise ValueError(kind)
        changed[slot] += replacement - native_score * native_value
    return changed


def _materialize(capture, index):
    return {name: value[index] for name, value in capture.items()}


def _compile_split(rows, capture, endpoint_indices, tokens_by_key, torch):
    examples, finals, replacements, specs = [], [], [], []
    for row in rows:
        rec = _materialize(capture, endpoint_indices[(row["row_id"], "recipient")])
        donors = {"within": _materialize(capture, endpoint_indices[(row["row_id"], "within_donor")]),
                  "cross": _materialize(capture, endpoint_indices[(row["row_id"], "cross_opposite")]),
                  "cross_same": _materialize(capture, endpoint_indices[(row["row_id"], "cross_same")])}
        for arm in ARMS:
            relation, kind = arm.rsplit("_", 1)
            examples.append(row["recipient"]); finals.append(row["recipient"]["query_position"])
            replacements.append(_replace(rec, donors[relation], kind))
            specs.append(("target", row, arm))
    control_rows = [row for row in rows if row["recipient_format"] == "list"]
    for row in control_rows:
        for control_id, endpoint in row["controls"].items():
            rec = _materialize(capture, endpoint_indices[(row["row_id"], control_id)])
            donor_owner = row if control_id == "repeated_list_copy" else next(
                item for item in rows if item["group_id"] == row["group_id"]
                and item["direction"] == row["direction"] and item["recipient_format"] == "digit")
            donor = _materialize(capture, endpoint_indices[(donor_owner["row_id"], "cross_opposite")])
            for arm in CONTROL_ARMS:
                examples.append(endpoint); finals.append(endpoint["query_position"])
                replacements.append(_replace(rec, donor, arm.split("_", 1)[1]))
                specs.append(("control", row, arm, control_id, endpoint))
    padded, final_tensor, _positions = _pad(examples, replacements[0].device)
    return padded, final_tensor, torch.stack(replacements), specs


def _margin(logits, positive, negative):
    return float(logits[positive] - logits[negative])


def _ce(logits, answer):
    return float(-torch.log_softmax(logits, -1)[answer])


def _ratio(numerator, denominator):
    return numerator / denominator if abs(denominator) > 1e-12 else None


def score(evidence, control_evidence, capability, exactness, bars):
    split_results, all_target_live, all_controls = {}, True, True
    for split in ("FIT", "SELECT"):
        split_rows = [row for row in evidence if row["split"] == split]
        grouped = defaultdict(list)
        for row in split_rows: grouped[(row["format"], row["direction"], row["arm"])].append(row)
        cells = {}
        for format_id in ("list", "digit"):
            for direction in ("base_to_donor", "donor_to_base"):
                cell_id = f"{format_id}__{direction}"
                means = {arm: statistics.fmean(row["margin_effect"] for row in grouped[(format_id, direction, arm)])
                         for arm in ARMS}
                ces = {arm: statistics.fmean(row["donor_ce_gain"] for row in grouped[(format_id, direction, arm)])
                       for arm in ARMS}
                natural = statistics.fmean(row["natural_margin_effect"]
                                           for row in grouped[(format_id, direction, "within_joint")])
                directions = {arm: sum(row["margin_effect"] * row["natural_margin_effect"] > 0
                                       for row in grouped[(format_id, direction, arm)]) / 4 for arm in ARMS}
                cells[cell_id] = {"native_accuracy": capability[split]["target"][cell_id],
                    "natural_margin_effect": natural, "mean_margin_effects": means,
                    "mean_donor_ce_gains": ces, "direction_fractions": directions,
                    "within_joint_natural_recovery": _ratio(means["within_joint"], natural),
                    "cross_cached_over_within_cached": _ratio(means["cross_cached"], means["within_cached"]),
                    "cross_joint_over_within_joint": _ratio(means["cross_joint"], means["within_joint"]),
                    "cross_score_over_within_score": _ratio(means["cross_score"], means["within_score"])}
        target_live = all(min(cell["native_accuracy"].values()) >= bars["minimum_native_accuracy"] and
            cell["within_joint_natural_recovery"] is not None and
            cell["within_joint_natural_recovery"] >= bars["minimum_within_joint_natural_recovery"] and
            cell["direction_fractions"]["within_joint"] >= bars["minimum_target_direction_fraction"]
            for cell in cells.values())
        controls = {}
        for control_id in ("repeated_list_copy", "digit_copy", "step_two"):
            rows_c = [row for row in control_evidence if row["split"] == split and row["control_id"] == control_id]
            target_scale = statistics.median(abs(row["margin_effect"]) for row in split_rows
                                             if row["arm"] == "cross_joint")
            controls[control_id] = {}
            for arm in CONTROL_ARMS:
                arm_rows = [row for row in rows_c if row["arm"] == arm]
                controls[control_id][arm] = {
                    "native_preference_accuracy": sum(row["native_preference_margin"] > 0 for row in arm_rows)/len(arm_rows),
                    "post_preference_preservation": sum(row["preference_margin"] > 0 for row in arm_rows)/len(arm_rows),
                    "absolute_mean_ce_change": abs(statistics.fmean(row["answer_ce_change"] for row in arm_rows)),
                    "median_absolute_margin_change_fraction": statistics.median(
                        abs(row["preference_margin_change"]) for row in arm_rows) / max(target_scale, 1e-12),
                    "median_intervention_norm_fraction": statistics.median(row["intervention_norm"] for row in arm_rows)
                        / max(statistics.median(row["target_intervention_norm"] for row in arm_rows), 1e-12)}
        control_pass = all(item["native_preference_accuracy"] >= bars["minimum_native_accuracy"] and
            item["post_preference_preservation"] >= bars["minimum_control_preference_preservation"] and
            item["absolute_mean_ce_change"] <= bars["maximum_control_absolute_mean_ce_change"] and
            item["median_absolute_margin_change_fraction"] <= bars["maximum_control_median_margin_change_fraction"] and
            item["median_intervention_norm_fraction"] >= bars["minimum_control_intervention_norm_fraction"]
            for control in controls.values() for item in control.values())
        split_results[split] = {"cells": cells, "controls": controls,
                                "target_live": target_live, "controls_pass": control_pass}
        all_target_live &= target_live; all_controls &= control_pass
    exact_live = (exactness["native_replay_relative_squared_error"] <= bars["maximum_native_replay_relative_squared_error"] and
        exactness["head_source_sum_relative_squared_error"] <= bars["maximum_source_sum_relative_squared_error"] and
        exactness["value_split_relative_squared_error"] <= bars["maximum_cached_decomposition_relative_squared_error"] and
        exactness["installed_term_max_absolute_error"] <= bars["maximum_installed_term_absolute_error"])
    instrument = exact_live and all_target_live
    payload = instrument and all(cell["cross_cached_over_within_cached"] is not None and
        cell["cross_cached_over_within_cached"] >= bars["minimum_cross_cached_over_within_cached"] and
        cell["direction_fractions"]["cross_cached"] >= bars["minimum_target_direction_fraction"] and
        cell["mean_donor_ce_gains"]["cross_cached"] > 0
        for result in split_results.values() for cell in result["cells"].values())
    shared = instrument and all(cell["cross_joint_over_within_joint"] is not None and
        cell["cross_joint_over_within_joint"] >= bars["minimum_cross_joint_over_within_joint"] and
        cell["cross_score_over_within_score"] is not None and
        cell["cross_score_over_within_score"] >= bars["minimum_cross_score_over_within_score"]
        for result in split_results.values() for cell in result["cells"].values())
    target_transfer = payload or shared
    return {**exactness, "splits": split_results, "predictions": {
        "pred_a_instrument_live": instrument,
        "pred_b_shared_payload_private_router": payload and all_controls,
        "pred_c_shared_score_and_payload": shared and all_controls,
        "pred_d_generic_numeral_or_copy_bus": target_transfer and not all_controls}}


def evaluate_split(model, rows, split, torch):
    rows = [row for row in rows if row["split"] == split]
    examples, endpoint_indices = [], {}
    for row in rows:
        for role in ("recipient", "within_donor", "cross_same", "cross_opposite"):
            endpoint_indices[(row["row_id"], role)] = len(examples); examples.append(row[role])
    for row in [item for item in rows if item["recipient_format"] == "list"]:
        for control_id, endpoint in row["controls"].items():
            endpoint_indices[(row["row_id"], control_id)] = len(examples); examples.append(endpoint)
    device = next(model.parameters()).device
    tokens, finals, positions = _pad(examples, device)
    native_full = r573.native_logits(model, tokens)
    arange = torch.arange(len(examples), device=device); native = native_full[arange, finals]
    replay, capture, diagnostics = _capture_forward(model, tokens, finals, positions)
    patched_tokens, patched_finals, replacements, specs = _compile_split(
        rows, capture, endpoint_indices, None, torch)
    patched, norms, patch_diag, installed_error = _patched_forward(
        model, patched_tokens, patched_finals, replacements)
    diagnostics = {key: max(value, patch_diag[key]) for key, value in diagnostics.items()}
    replay_rse = float((replay-native).square().sum()) / max(float(native.square().sum()), 1e-30)
    evidence, control_evidence, target_norms = [], [], {}
    for output_index, spec in enumerate(specs):
        if spec[0] == "target":
            _, row, arm = spec; rec_i = endpoint_indices[(row["row_id"], "recipient")]
            donor_i = endpoint_indices[(row["row_id"], "within_donor")]
            positive, negative = row["within_donor"]["answer_id"], row["recipient"]["answer_id"]
            before_margin = _margin(replay[rec_i], positive, negative)
            effect = _margin(patched[output_index], positive, negative) - before_margin
            natural = _margin(replay[donor_i], positive, negative) - before_margin
            evidence.append({"split": split, "row_id": row["row_id"],
                "format": row["recipient_format"], "direction": row["direction"], "arm": arm,
                "margin_effect": effect, "natural_margin_effect": natural,
                "donor_ce_gain": _ce(replay[rec_i], positive)-_ce(patched[output_index], positive),
                "intervention_norm": float(norms[output_index])})
            target_norms[(row["group_id"], row["direction"], arm)] = float(norms[output_index])
        else:
            _, row, arm, control_id, endpoint = spec
            rec_i = endpoint_indices[(row["row_id"], control_id)]
            answer, foil = endpoint["answer_id"], endpoint["preference_foil_id"]
            native_margin = _margin(replay[rec_i], answer, foil)
            after_margin = _margin(patched[output_index], answer, foil)
            control_evidence.append({"split": split, "group_id": row["group_id"],
                "direction": row["direction"], "control_id": control_id, "arm": arm,
                "native_preference_margin": native_margin, "preference_margin": after_margin,
                "preference_margin_change": after_margin-native_margin,
                "answer_ce_change": _ce(patched[output_index], answer)-_ce(replay[rec_i], answer),
                "intervention_norm": float(norms[output_index]),
                "target_intervention_norm": target_norms.get((row["group_id"], row["direction"], arm),
                    statistics.median(target_norms.values()))})
    capability = {"target": {}, "controls": {}}
    for format_id in ("list", "digit"):
        for direction in ("base_to_donor", "donor_to_base"):
            selected = [row for row in rows if row["recipient_format"] == format_id and row["direction"] == direction]
            capability["target"][f"{format_id}__{direction}"] = {
                role: sum(r577.answer_is_best(replay[endpoint_indices[(row["row_id"], role)]],
                            row[role]["answer_id"], row[role]["answer_text"]) for row in selected)/len(selected)
                for role in ("recipient", "within_donor", "cross_same", "cross_opposite")}
    return evidence, control_evidence, capability, {"native_replay_relative_squared_error": replay_rse,
        "head_source_sum_relative_squared_error": diagnostics["head_source_sum_relative_squared_error"],
        "value_split_relative_squared_error": diagnostics["value_split_relative_squared_error"],
        "installed_term_max_absolute_error": installed_error}


def main():
    plan = compile_plan()
    if "--dry-run" in sys.argv or os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    if OUT.exists(): raise RuntimeError(f"refusing to overwrite {OUT}")
    model, checkpoint = r573.facade.load_bilin18(device="cuda", dtype=torch.float32,
                                                  verify_weights_sha256=True)
    evidence, controls, capability = [], [], {}
    exactness = {key: 0. for key in ("native_replay_relative_squared_error",
        "head_source_sum_relative_squared_error", "value_split_relative_squared_error",
        "installed_term_max_absolute_error")}
    for split in ("FIT", "SELECT"):
        split_e, split_c, split_cap, split_exact = evaluate_split(model, build_rows(), split, torch)
        evidence += split_e; controls += split_c; capability[split] = split_cap
        exactness = {key: max(exactness[key], split_exact[key]) for key in exactness}
    scored = score(evidence, controls, capability, exactness, plan["bars"])
    predictions = scored["predictions"]
    terminal = ("invalid" if not predictions["pred_a_instrument_live"] else
                "shared_score_and_payload" if predictions["pred_c_shared_score_and_payload"] else
                "shared_payload_private_router" if predictions["pred_b_shared_payload_private_router"] else
                "generic_numeral_or_copy_bus" if predictions["pred_d_generic_numeral_or_copy_bus"] else
                "location_only_null")
    result = {"schema": "attn8_h3_h7_cross_behavior_factor_interchange_result_v2",
        "plan": plan, "prior_art_sha256": PRIOR_ART_SHA256,
        "checkpoint_weights_sha256": checkpoint.weights_sha256, "terminal": terminal,
        "score": scored, "evidence": evidence, "control_evidence": controls,
        "evaluated_splits": ["FIT", "SELECT"], "forbidden_splits_opened": [],
        "model_forwards": 6}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(result, indent=1)+"\n")
    print(json.dumps({"result": str(OUT), "terminal": terminal,
                      "predictions": predictions}, indent=2))


if __name__ == "__main__": main()
