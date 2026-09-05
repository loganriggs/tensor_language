#!/usr/bin/env python3
"""Capability-first exact pair-centered opener-term test on R545 FINAL_TEST."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_pair_centered_open_term_held pred_c_transfer_without_selective_necessity pred_d_no_heldout_open_term_circuit

from __future__ import annotations

from collections import defaultdict
import json
import math
import os
from pathlib import Path
import statistics
import sys

import circuit_fast_screen_candidate_bracket_l13h8_pair_centered_open_term_final_test as authority
import run_bracket_l13h8_source_region_payload_factorial as exact


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits/fast_screens/bracket_l13h8_pair_centered_open_term_final_test_v1_result.json"


def _endpoints(rows):
    return [(row, side) for row in rows for side in ("base", "donor")]


def _pad(rows, torch, device):
    endpoints = _endpoints(rows)
    length = max(len(row[f"{side}_ids"]) for row, side in endpoints)
    tokens = torch.full((len(endpoints), length), 50256, dtype=torch.long, device=device)
    finals, sources = [], []
    for index, (row, side) in enumerate(endpoints):
        ids = row[f"{side}_ids"]
        tokens[index, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        finals.append(len(ids) - 1)
        sources.append(row[f"{side}_open_position"])
    return (endpoints, tokens, torch.tensor(finals, device=device),
            torch.tensor(sources, device=device))


def _ce(logits, answer, torch):
    return float(-torch.log_softmax(logits, dim=-1)[answer])


def _capability_cell(row, side):
    if row["role"] == "target":
        pair = f"{row['base_answer_id']}->{row['donor_answer_id']}"
        return f"{row['family_id']}|{pair}|{side}"
    return f"{row['family_id']}|{side}"


def score_native(evidence):
    bars = authority.compile_plan()["bars"]
    grouped = defaultdict(list)
    for item in evidence:
        grouped[item["cell_id"]].append(item)
    reports, passed = {}, True
    for cell, items in sorted(grouped.items()):
        threshold = (bars["native_target_cell_accuracy_min"]
                     if items[0]["role"] == "target"
                     else bars["native_control_cell_accuracy_min"])
        report = {
            "n": len(items),
            "accuracy": sum(item["correct"] for item in items) / len(items),
            "mean_answer_margin": statistics.fmean(item["answer_margin"] for item in items),
            "mean_full_vocab_CE": statistics.fmean(item["full_vocab_CE"] for item in items),
        }
        report["passed"] = report["accuracy"] >= threshold
        passed &= report["passed"]
        reports[cell] = report
    expected = 2 * 6 * 2 + 3 * 2
    if len(reports) != expected or any(report["n"] != 6 for report in reports.values()):
        raise ValueError("native capability cells differ from frozen FINAL_TEST")
    family_margins = {}
    for family in authority.TARGET_FAMILIES:
        values = [item["answer_margin"] for item in evidence if item["family_id"] == family]
        family_margins[family] = statistics.fmean(values)
        passed &= family_margins[family] > bars["native_target_family_mean_margin_min"]
    return {"cells": reports, "target_family_mean_margins": family_margins,
            "passed": bool(passed)}


def evaluate_native(model, rows, torch, F):
    endpoints, tokens, finals, _ = _pad(rows, torch, next(model.parameters()).device)
    logits = exact.native_logits(model, tokens, torch, F)
    evidence = []
    for index, (row, side) in enumerate(endpoints):
        q, answer = int(finals[index]), row[f"{side}_answer_id"]
        margin = exact.closer_margin(logits[index, q], answer)
        evidence.append({
            "row_id": row["row_id"], "family_id": row["family_id"],
            "role": row["role"], "side": side, "cell_id": _capability_cell(row, side),
            "correct": bool(margin > 0), "answer_margin": margin,
            "full_vocab_CE": _ce(logits[index, q], answer, torch),
        })
    return endpoints, tokens, finals, logits, evidence


def evaluate_causal(model, tokens, finals, native, torch, F, facade):
    endpoints = _endpoints(authority.ROWS)
    sources = torch.tensor([row[f"{side}_open_position"] for row, side in endpoints],
                           device=tokens.device)
    replay, factors = exact.factor_forward(model, tokens, finals, {}, torch, F, facade)
    arange = torch.arange(len(endpoints), device=tokens.device)
    terms = factors["p"][arange, sources].unsqueeze(-1) * factors["u"][arange, sources]
    donor_index = torch.tensor([index ^ 1 for index in range(len(endpoints))], device=tokens.device)
    donor = {key: value[donor_index] for key, value in factors.items()}
    complete = exact.factor_forward(
        model, tokens, finals, {}, torch, F, facade, donor=donor, complete=True)[0]
    swapped = exact.factor_forward(
        model, tokens, finals, {}, torch, F, facade,
        replacement_terms=terms[donor_index], source_positions=sources)[0]
    midpoint = exact.factor_forward(
        model, tokens, finals, {}, torch, F, facade,
        replacement_terms=(terms + terms[donor_index]) / 2, source_positions=sources)[0]
    replay_error = float((native - replay).abs().max())
    records = []
    for index, (row, side) in enumerate(endpoints):
        q = int(finals[index]); recipient = row[f"{side}_answer_id"]
        donor_side = "donor" if side == "base" else "base"
        donor_answer = row[f"{donor_side}_answer_id"]
        direction = "base_to_donor" if side == "base" else "donor_to_base"
        before = replay[index, q]
        base_recipient_margin = exact.closer_margin(before, recipient)
        base_recipient_ce = _ce(before, recipient, torch)
        base_donor_margin = exact.closer_margin(before, donor_answer)
        base_donor_ce = _ce(before, donor_answer, torch)
        record = {"row_id": row["row_id"], "family_id": row["family_id"],
                  "role": row["role"], "direction": direction,
                  "ordered_pair": f"{recipient}->{donor_answer}",
                  "open_term_norm": float(terms[index].norm())}
        for name, logits in (("complete", complete), ("open_swap", swapped),
                             ("midpoint", midpoint)):
            after = logits[index, q]
            record[f"{name}_donor_margin_effect"] = \
                exact.closer_margin(after, donor_answer) - base_donor_margin
            record[f"{name}_donor_CE_improvement"] = base_donor_ce - _ce(after, donor_answer, torch)
            record[f"{name}_recipient_margin_damage"] = \
                base_recipient_margin - exact.closer_margin(after, recipient)
            record[f"{name}_recipient_CE_damage"] = _ce(after, recipient, torch) - base_recipient_ce
            record[f"{name}_recipient_correct"] = bool(exact.closer_margin(after, recipient) > 0)
        records.append(record)
    return records, replay_error


def _group(records, fields):
    output = defaultdict(list)
    for row in records:
        output[tuple(row[field] for field in fields)].append(row)
    return output


def score_causal(records, replay_error, capability):
    bars = authority.compile_plan()["bars"]
    targets = [row for row in records if row["role"] == "target"]
    controls = [row for row in records if row["role"] == "control"]
    target_cells = _group(targets, ("family_id", "ordered_pair", "direction"))
    complete_cells, swap_cells = {}, {}
    for key, rows in sorted(target_cells.items()):
        complete_cells["|".join(key)] = sum(r["complete_donor_margin_effect"] > 0 for r in rows) / len(rows)
        swap_cells["|".join(key)] = sum(r["open_swap_donor_margin_effect"] > 0 for r in rows) / len(rows)
    complete_live = min(complete_cells.values()) >= \
        bars["complete_head_target_cell_positive_fraction_min"]
    swap_direction = min(swap_cells.values()) >= bars["open_swap_target_cell_positive_fraction_min"]

    family_direction = _group(targets, ("family_id", "direction"))
    target_reports = {}
    for key, rows in sorted(family_direction.items()):
        ratios = [r["open_swap_donor_margin_effect"] /
                  (r["complete_donor_margin_effect"] if abs(r["complete_donor_margin_effect"]) > 1e-9
                   else math.copysign(1e-9, r["complete_donor_margin_effect"] or 1)) for r in rows]
        midpoint_ratios = [r["midpoint_recipient_margin_damage"] /
                           (r["complete_recipient_margin_damage"]
                            if abs(r["complete_recipient_margin_damage"]) > 1e-9
                            else math.copysign(1e-9, r["complete_recipient_margin_damage"] or 1))
                           for r in rows]
        report = {
            "n": len(rows), "open_swap_median_fraction_complete": statistics.median(ratios),
            "midpoint_margin_damage_positive_fraction": sum(
                r["midpoint_recipient_margin_damage"] > 0 for r in rows) / len(rows),
            "midpoint_CE_damage_positive_fraction": sum(
                r["midpoint_recipient_CE_damage"] > 0 for r in rows) / len(rows),
            "midpoint_median_margin_fraction_complete": statistics.median(midpoint_ratios),
        }
        report["swap_passed"] = report["open_swap_median_fraction_complete"] >= \
            bars["open_swap_median_fraction_complete_each_family_direction_min"]
        report["midpoint_passed"] = (
            min(report["midpoint_margin_damage_positive_fraction"],
                report["midpoint_CE_damage_positive_fraction"])
            >= bars["midpoint_positive_margin_and_ce_damage_fraction_each_family_direction_min"]
            and report["midpoint_median_margin_fraction_complete"]
            >= bars["midpoint_median_margin_fraction_complete_each_family_direction_min"])
        target_reports["|".join(key)] = report

    # Direct task-level selectivity: controls are compared with the smallest
    # target cell effect on the same arm and metric, never with a whole-head partial sum.
    scale = {}
    for arm, metric in (("open_swap", "donor_margin_effect"),
                        ("open_swap", "donor_CE_improvement"),
                        ("midpoint", "recipient_margin_damage"),
                        ("midpoint", "recipient_CE_damage")):
        means = [statistics.fmean(abs(r[f"{arm}_{metric}"]) for r in rows)
                 for rows in target_cells.values()]
        scale[f"{arm}_{metric}"] = min(means)
    control_reports, controls_pass = {}, True
    for key, rows in sorted(_group(controls, ("family_id", "direction")).items()):
        report = {}
        for arm in ("open_swap", "midpoint"):
            report[f"{arm}_answer_preservation"] = sum(
                r[f"{arm}_recipient_correct"] for r in rows) / len(rows)
            for metric in (("donor_margin_effect" if arm == "open_swap" else "recipient_margin_damage"),
                           ("donor_CE_improvement" if arm == "open_swap" else "recipient_CE_damage")):
                observed = statistics.fmean(abs(r[f"{arm}_{metric}"]) for r in rows)
                report[f"{arm}_{metric}_fraction_target_min"] = observed / max(scale[f"{arm}_{metric}"], 1e-12)
            controls_pass &= report[f"{arm}_answer_preservation"] >= \
                bars["control_answer_preservation_each_family_direction_arm_min"]
            controls_pass &= max(report[f"{arm}_{metric}_fraction_target_min"] for metric in
                                 (("donor_margin_effect" if arm == "open_swap" else "recipient_margin_damage"),
                                  ("donor_CE_improvement" if arm == "open_swap" else "recipient_CE_damage"))) \
                <= bars["control_effect_fraction_of_smallest_target_cell_max"]
        control_reports["|".join(key)] = report
    replay_pass = replay_error <= bars["native_replay_max_absolute_logit_error_max"]
    instrument = capability["passed"] and replay_pass and complete_live
    transfer = swap_direction and all(x["swap_passed"] for x in target_reports.values())
    necessity = all(x["midpoint_passed"] for x in target_reports.values()) and controls_pass
    pred_b = instrument and transfer and necessity
    return {
        "instrument_live": instrument,
        "instrument_checks": {"native_capability": capability["passed"],
                              "native_replay": replay_pass,
                              "complete_head_ceiling": complete_live},
        "complete_head_positive_fraction_by_cell": complete_cells,
        "open_swap_positive_fraction_by_cell": swap_cells,
        "target_reports": target_reports, "control_reports": control_reports,
        "target_scale_for_selectivity": scale,
        "predictions": {
            "pred_a_instrument_live": instrument,
            "pred_b_pair_centered_open_term_held": pred_b,
            "pred_c_transfer_without_selective_necessity": instrument and transfer and not necessity,
            "pred_d_no_heldout_open_term_circuit": instrument and not transfer,
        },
    }


def run_staged(model, torch, F, facade):
    endpoints, tokens, finals, native, evidence = evaluate_native(model, authority.ROWS, torch, F)
    capability = score_native(evidence)
    if not capability["passed"]:
        return capability, evidence, [], None, 1
    records, replay_error = evaluate_causal(model, tokens, finals, native, torch, F, facade)
    return capability, evidence, records, score_causal(records, replay_error, capability), 5


def main():
    plan = authority.compile_plan()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1" \
            or "--dry-run" in sys.argv:
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    torch, F, facade = exact._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                             verify_weights_sha256=True)
    with torch.no_grad():
        capability, evidence, records, screen, forwards = run_staged(model, torch, F, facade)
    if not capability["passed"]:
        terminal, reason = "invalid", "native_capability_failed_causal_arms_not_opened"
    else:
        terminal = "screen" if screen["instrument_live"] else "invalid"
        reason = "scored" if terminal == "screen" else "causal_instrument_failed"
    result = {
        "schema": "bracket_l13h8_pair_centered_open_term_final_test_result_v1",
        "plan": plan, "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "capability": capability, "native_evidence": evidence,
        "raw": records, "screen": screen, "terminal": terminal, "reason": reason,
        "evaluated_splits": ["FINAL_TEST"], "forbidden_splits_opened": [],
        "model_forwards": forwards,
        "example_evaluations": forwards * len(evidence),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"result": str(OUT), "terminal": terminal, "reason": reason,
                      "model_forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
