#!/usr/bin/env python3
"""FINAL_TEST/OOD confirmation of the L8 H3+H7 cached successor value."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_selective_ood pred_c_cached_payload_not_selective pred_d_no_held_out_successor_transfer

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

import torch

import final_ood_authority_attn8_h3_h7_cached_successor_v1 as authority
import run_attn8_h3_h7_cross_behavior_factor_interchange_v2 as exact


ROOT = exact.ROOT
OUT = ROOT / "circuits/fast_screens/attn8_h3_h7_cached_successor_final_ood_v1_result.json"
PRIOR_ART = ROOT / "circuits/prior_art/attn8_h3_h7_cached_successor_final_ood_v1.json"
PRIOR_ART_SHA256 = "f4e261b98011701e36d682075874d9730f7b206b91b966d30094096fea86147d"
TARGET = ROOT / "circuits/fast_screens/attn8_h3_h7_cross_behavior_factor_interchange_v3_control_repair_result.json"
TARGET_SHA256 = "56dfa025df88192fb3f68e86b69cea6fdfe5164b9fc3c906edb52ca6b6f1fe1c"
TARGET_SOURCE_SPLIT = {"FINAL_TEST": "FIT", "OOD": "SELECT"}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_binding():
    if _sha(PRIOR_ART) != PRIOR_ART_SHA256 or _sha(TARGET) != TARGET_SHA256:
        raise RuntimeError("prior-art receipt or frozen v3 target evidence changed")
    result = json.loads(TARGET.read_text())
    margin, norm = {}, {}
    for heldout, source in TARGET_SOURCE_SPLIT.items():
        rows = [x for x in result["evidence"]
                if x["split"] == source and x["arm"] == "cross_cached"]
        margin[heldout] = statistics.median(abs(x["margin_effect"]) for x in rows)
        norm[heldout] = statistics.median(x["intervention_norm"] for x in rows)
    target_exact = (result["score"]["native_replay_relative_squared_error"] <= 1e-10 and
        result["score"]["head_source_sum_relative_squared_error"] <= 1e-10 and
        result["score"]["value_split_relative_squared_error"] <= 1e-10 and
        result["score"]["installed_term_max_absolute_error"] <= 1e-5)
    return {"artifact_sha256": TARGET_SHA256, "source_split": TARGET_SOURCE_SPLIT,
            "margin_scales": margin, "intervention_scales": norm,
            "target_exact": target_exact}


def compile_plan():
    pairs = authority.build_pairs()
    target = _target_binding()
    return {
        "schema": "attn8_h3_h7_cached_successor_final_ood_v1_plan",
        "candidate_id": "numeric_successor.attn8_h3_h7_cached_successor_final_ood_v1",
        "pairing_sha256": authority.canonical(pairs), "pair_count": len(pairs),
        "splits": list(authority.SPLITS), "positive_family": authority.POSITIVE,
        "negative_families": list(authority.NEGATIVE),
        "secondary_family": authority.SECONDARY, "arm": "cached_value_only",
        "fixed_layer": 8, "fixed_heads": [3, 7],
        "target_artifact_sha256": target["artifact_sha256"],
        "price": {"model_forwards": 6, "example_evaluations": 1280,
                  "backwards": 0, "parameter_updates": 0},
        "bars": {
            "minimum_native_accuracy": .85,
            "minimum_positive_donorward_fraction": .75,
            "minimum_positive_donor_answer_win_fraction": .60,
            "minimum_positive_mean_donor_ce_gain": 0.,
            "minimum_intervention_norm_fraction_of_target": .10,
            "minimum_negative_answer_preservation": .75,
            "maximum_negative_absolute_mean_recipient_ce_change": .10,
            "maximum_negative_median_margin_change_fraction_of_target": .25,
            "maximum_native_replay_relative_squared_error": 1e-10,
            "maximum_source_sum_relative_squared_error": 1e-10,
            "maximum_cached_decomposition_relative_squared_error": 1e-10,
            "maximum_installed_term_absolute_error": 1e-5,
        },
    }


def _compile_replacements(pairs, captures):
    replacements = []
    for index in range(len(pairs)):
        recipient = {k: v[2 * index] for k, v in captures.items()}
        donor = {k: v[2 * index + 1] for k, v in captures.items()}
        replacements.append(exact._replace(recipient, donor, "cached"))
    tokens, finals, _ = exact._pad([x["recipient"] for x in pairs], replacements[0].device)
    return tokens, finals, torch.stack(replacements)


def evaluate_split(model, pairs, split):
    pairs = [x for x in pairs if x["split"] == split]
    examples = [ep for pair in pairs for ep in (pair["recipient"], pair["donor"])]
    device = next(model.parameters()).device
    tokens, finals, positions = exact._pad(examples, device)
    native_full = exact.r573.native_logits(model, tokens)
    native = native_full[torch.arange(len(examples), device=device), finals]
    replay, captures, diagnostics = exact._capture_forward(model, tokens, finals, positions)
    patch_tokens, patch_finals, replacements = _compile_replacements(pairs, captures)
    patched, norms, patch_diagnostics, installed_error = exact._patched_forward(
        model, patch_tokens, patch_finals, replacements)
    replay_rse = float((native - replay).square().sum()) / max(float(native.square().sum()), 1e-30)
    evidence = []
    for i, pair in enumerate(pairs):
        before, donor_native, after = replay[2 * i], replay[2 * i + 1], patched[i]
        recipient_answer = pair["recipient"]["answer_id"]
        donor_answer = pair["donor"]["answer_id"]
        donor_margin_before = exact._margin(before, donor_answer, recipient_answer)
        donor_margin_after = exact._margin(after, donor_answer, recipient_answer)
        recipient_margin_before = -donor_margin_before
        recipient_margin_after = -donor_margin_after
        evidence.append({
            "pair_id": pair["pair_id"], "family_id": pair["family_id"],
            "split": split, "surface_side": pair["surface_side"],
            "direction": pair["direction"],
            "native_recipient_answer_correct": exact.r577.answer_is_best(
                before, recipient_answer, pair["recipient"]["answer_text"]),
            "native_donor_answer_correct": exact.r577.answer_is_best(
                donor_native, donor_answer, pair["donor"]["answer_text"]),
            "post_recipient_answer_correct": exact.r577.answer_is_best(
                after, recipient_answer, pair["recipient"]["answer_text"]),
            "donor_margin_before": donor_margin_before,
            "donor_margin_after": donor_margin_after,
            "donor_margin_effect": donor_margin_after - donor_margin_before,
            "donor_answer_win": donor_margin_after > 0.,
            "donor_ce_gain": exact._ce(before, donor_answer) - exact._ce(after, donor_answer),
            "recipient_margin_change": recipient_margin_after - recipient_margin_before,
            "recipient_ce_change": exact._ce(after, recipient_answer) - exact._ce(before, recipient_answer),
            "intervention_norm": float(norms[i]),
        })
    return evidence, {
        "native_replay_relative_squared_error": replay_rse,
        "head_source_sum_relative_squared_error": max(
            diagnostics["head_source_sum_relative_squared_error"],
            patch_diagnostics["head_source_sum_relative_squared_error"]),
        "value_split_relative_squared_error": max(
            diagnostics["value_split_relative_squared_error"],
            patch_diagnostics["value_split_relative_squared_error"]),
        "installed_term_max_absolute_error": installed_error,
    }


def _native_capability(rows, minimum):
    recipient = statistics.fmean(x["native_recipient_answer_correct"] for x in rows)
    donor = statistics.fmean(x["native_donor_answer_correct"] for x in rows)
    return recipient, donor, min(recipient, donor) >= minimum


def score(evidence, exactness, target, bars):
    grouped = defaultdict(list)
    for row in evidence:
        grouped[(row["family_id"], row["split"])].append(row)
    reports = {family: {} for family in authority.FAMILIES}
    positive_pass = negative_pass = capable = live = True
    for family in authority.FAMILIES:
        for split in authority.SPLITS:
            rows = grouped[(family, split)]
            recipient_accuracy, donor_accuracy, cell_capable = _native_capability(
                rows, bars["minimum_native_accuracy"])
            norm_fraction = statistics.median(x["intervention_norm"] for x in rows) / \
                max(target["intervention_scales"][split], 1e-12)
            report = {"row_count": len(rows),
                      "recipient_native_accuracy": recipient_accuracy,
                      "donor_native_accuracy": donor_accuracy,
                      "median_intervention_norm_fraction_of_target": norm_fraction}
            report["capable"] = cell_capable
            report["live"] = norm_fraction >= bars["minimum_intervention_norm_fraction_of_target"]
            if family == authority.POSITIVE:
                report.update({
                    "donorward_fraction": statistics.fmean(
                        x["donor_margin_effect"] > 0 for x in rows),
                    "donor_answer_win_fraction": statistics.fmean(x["donor_answer_win"] for x in rows),
                    "mean_donor_ce_gain": statistics.fmean(x["donor_ce_gain"] for x in rows),
                })
                report["passes"] = (report["donorward_fraction"] >= bars["minimum_positive_donorward_fraction"] and
                    report["donor_answer_win_fraction"] >= bars["minimum_positive_donor_answer_win_fraction"] and
                    report["mean_donor_ce_gain"] > bars["minimum_positive_mean_donor_ce_gain"])
                positive_pass &= report["passes"]
            else:
                report.update({
                    "recipient_answer_preservation": statistics.fmean(
                        x["post_recipient_answer_correct"] for x in rows),
                    "absolute_mean_recipient_ce_change": abs(statistics.fmean(
                        x["recipient_ce_change"] for x in rows)),
                    "median_absolute_margin_change_fraction_of_target": statistics.median(
                        abs(x["recipient_margin_change"]) for x in rows) /
                        max(target["margin_scales"][split], 1e-12),
                })
                report["passes"] = (report["recipient_answer_preservation"] >= bars["minimum_negative_answer_preservation"] and
                    report["absolute_mean_recipient_ce_change"] <= bars["maximum_negative_absolute_mean_recipient_ce_change"] and
                    report["median_absolute_margin_change_fraction_of_target"] <= bars["maximum_negative_median_margin_change_fraction_of_target"])
                if family in authority.NEGATIVE:
                    negative_pass &= report["passes"]
            reports[family][split] = report
            if family == authority.POSITIVE or family in authority.NEGATIVE:
                capable &= report["capable"]
                live &= report["live"]
    exact_live = (exactness["native_replay_relative_squared_error"] <= bars["maximum_native_replay_relative_squared_error"] and
        exactness["head_source_sum_relative_squared_error"] <= bars["maximum_source_sum_relative_squared_error"] and
        exactness["value_split_relative_squared_error"] <= bars["maximum_cached_decomposition_relative_squared_error"] and
        exactness["installed_term_max_absolute_error"] <= bars["maximum_installed_term_absolute_error"])
    instrument = target["target_exact"] and exact_live and capable and live
    return {**exactness, "target_binding": target, "family_reports": reports,
        "predictions": {
            "pred_a_instrument_live": instrument,
            "pred_b_selective_ood": instrument and positive_pass and negative_pass,
            "pred_c_cached_payload_not_selective": instrument and positive_pass and not negative_pass,
            "pred_d_no_held_out_successor_transfer": instrument and not positive_pass,
        }}


def _dry_requested():
    for name in ("BQLIB_DRYRUN", "BQLIB_NO_MODEL"):
        value = os.environ.get(name)
        if value not in (None, "0", "1"):
            raise RuntimeError(f"{name} must be 0 or 1")
    return "--dry-run" in sys.argv or any(os.environ.get(x) == "1" for x in (
        "BQLIB_DRYRUN", "BQLIB_NO_MODEL"))


def main():
    dry = _dry_requested()  # This gate intentionally precedes every model-loading path.
    plan = compile_plan()
    if dry:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    model, checkpoint = exact.r573.facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    pairs = authority.build_pairs()
    evidence = []
    exactness = {k: 0. for k in (
        "native_replay_relative_squared_error", "head_source_sum_relative_squared_error",
        "value_split_relative_squared_error", "installed_term_max_absolute_error")}
    for split in authority.SPLITS:
        rows, diagnostics = evaluate_split(model, pairs, split)
        evidence.extend(rows)
        exactness = {k: max(exactness[k], diagnostics[k]) for k in exactness}
    target = _target_binding()
    scored = score(evidence, exactness, target, plan["bars"])
    pred = scored["predictions"]
    terminal = ("invalid" if not pred["pred_a_instrument_live"] else
        "selective_ood" if pred["pred_b_selective_ood"] else
        "cached_payload_not_selective" if pred["pred_c_cached_payload_not_selective"] else
        "no_held_out_successor_transfer")
    result = {"schema": "attn8_h3_h7_cached_successor_final_ood_v1_result",
        "plan": plan, "prior_art_sha256": PRIOR_ART_SHA256,
        "checkpoint_weights_sha256": checkpoint.weights_sha256, "terminal": terminal,
        "score": scored, "evidence": evidence, "evaluated_splits": list(authority.SPLITS),
        "model_forwards": 6}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"result": str(OUT), "terminal": terminal,
                      "predictions": pred}, indent=2))


if __name__ == "__main__":
    main()
