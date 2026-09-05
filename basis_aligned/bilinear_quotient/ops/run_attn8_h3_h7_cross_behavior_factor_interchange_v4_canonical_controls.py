#!/usr/bin/env python3
"""Canonical R567 control-only evidence leg for cross-behavior H3+H7 factors."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_shared_cached_payload_private_router pred_c_broad_numeral_or_copy_service pred_d_score_or_joint_collateral_only

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import torch

import canonical_control_authority_attn8_h3_h7_cross_behavior_v4 as authority
import run_attn8_h3_h7_cross_behavior_factor_interchange_v2 as exact


ROOT = exact.ROOT
OUT = ROOT / "circuits/fast_screens/attn8_h3_h7_cross_behavior_factor_interchange_v4_canonical_controls_result.json"
PRIOR_ART = ROOT / "circuits/prior_art/attn8_h3_h7_cross_behavior_factor_interchange_v4_canonical_controls.json"
PRIOR_ART_SHA256 = "2276c565bf93eca0c0197d1541f703f5d2c962db3156292532de9b05ee1a19fe"
TARGET = ROOT / "circuits/fast_screens/attn8_h3_h7_cross_behavior_factor_interchange_v3_control_repair_result.json"
TARGET_SHA256 = "56dfa025df88192fb3f68e86b69cea6fdfe5164b9fc3c906edb52ca6b6f1fe1c"
ARMS = ("score", "cached", "joint")


def _sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_binding():
    if _sha(PRIOR_ART) != PRIOR_ART_SHA256 or _sha(TARGET) != TARGET_SHA256:
        raise RuntimeError("v4 receipt or retained v3 target evidence changed")
    result = json.loads(TARGET.read_text())
    target = {}
    target_live = True; cached_transfer = True
    for split, report in result["score"]["splits"].items():
        target[split] = {"cells": report["cells"]}
        target_live &= report["target_live"]
        for cell in report["cells"].values():
            cached_transfer &= (cell["cross_cached_over_within_cached"] is not None and
                cell["cross_cached_over_within_cached"] >= .70 and
                cell["direction_fractions"]["cross_cached"] >= .75 and
                cell["mean_donor_ce_gains"]["cross_cached"] > 0)
    exact_live = (result["score"]["native_replay_relative_squared_error"] <= 1e-10 and
        result["score"]["head_source_sum_relative_squared_error"] <= 1e-10 and
        result["score"]["value_split_relative_squared_error"] <= 1e-10 and
        result["score"]["installed_term_max_absolute_error"] <= 1e-5)
    scales = {split: {arm: statistics.median(
        row["intervention_norm"] for row in result["evidence"]
        if row["split"] == split and row["arm"] == f"cross_{arm}") for arm in ARMS}
        for split in ("FIT", "SELECT")}
    margin_scales = {split: {arm: statistics.median(abs(row["margin_effect"])
        for row in result["evidence"] if row["split"] == split and row["arm"] == f"cross_{arm}")
        for arm in ARMS} for split in ("FIT", "SELECT")}
    return {"artifact_sha256": TARGET_SHA256, "target_live": target_live,
            "target_exact": exact_live, "cached_transfer": cached_transfer,
            "reports": target, "intervention_scales": scales, "margin_scales": margin_scales}


def compile_plan():
    pairs = authority.build_pairs(); target = _target_binding()
    return {"schema": "attn8_h3_h7_cross_behavior_factor_interchange_v4_canonical_controls_plan",
        "candidate_id": "numeric_successor.attn8_h3_h7_cross_behavior_factor_interchange_v4_canonical_controls",
        "pairing_sha256": authority.canonical(pairs), "pair_count": len(pairs),
        "arms": list(ARMS), "fixed_heads": [3, 7], "target_artifact_sha256": target["artifact_sha256"],
        "target_endpoints_rerun": False,
        "price": {"model_forwards": 6, "example_evaluations": 2688,
                  "backwards": 0, "parameter_updates": 0},
        "bars": {"minimum_native_accuracy": .85,
                 "maximum_native_replay_relative_squared_error": 1e-10,
                 "maximum_source_sum_relative_squared_error": 1e-10,
                 "maximum_cached_decomposition_relative_squared_error": 1e-10,
                 "maximum_installed_term_absolute_error": 1e-5,
                 "minimum_intervention_norm_fraction_of_target": .10,
                 "minimum_preference_preservation": .75,
                 "maximum_absolute_mean_ce_change": .10,
                 "maximum_median_margin_change_fraction_of_target": .25},
        "decisive_families": list(authority.DECISIVE), "secondary_family": authority.SECONDARY}


def _compile_replacements(pairs, captures, torch):
    replacements, specs, examples = [], [], []
    for index, pair in enumerate(pairs):
        recipient = {name: value[2*index] for name, value in captures.items()}
        donor = {name: value[2*index+1] for name, value in captures.items()}
        for arm in ARMS:
            replacements.append(exact._replace(recipient, donor, arm))
            specs.append((index, arm)); examples.append(pair["recipient"])
    tokens, finals, _positions = exact._pad(examples, replacements[0].device)
    return tokens, finals, torch.stack(replacements), specs


def evaluate_split(model, pairs, split, torch):
    pairs = [pair for pair in pairs if pair["split"] == split]
    examples = [endpoint for pair in pairs for endpoint in (pair["recipient"], pair["donor"])]
    device = next(model.parameters()).device
    tokens, finals, positions = exact._pad(examples, device)
    native_full = exact.r573.native_logits(model, tokens)
    rows = torch.arange(len(examples), device=device); native = native_full[rows, finals]
    replay, captures, diagnostics = exact._capture_forward(model, tokens, finals, positions)
    patch_tokens, patch_finals, replacements, specs = _compile_replacements(pairs, captures, torch)
    patched, norms, patch_diagnostics, installed_error = exact._patched_forward(
        model, patch_tokens, patch_finals, replacements)
    replay_rse = float((native-replay).square().sum()) / max(float(native.square().sum()), 1e-30)
    evidence = []
    for output_index, (pair_index, arm) in enumerate(specs):
        pair = pairs[pair_index]; before = replay[2*pair_index]
        answer, foil = pair["recipient"]["answer_id"], pair["donor"]["answer_id"]
        native_margin = exact._margin(before, answer, foil)
        after_margin = exact._margin(patched[output_index], answer, foil)
        evidence.append({"pair_id": pair["pair_id"], "family_id": pair["family_id"],
            "decisive": pair["decisive"], "split": split, "surface_side": pair["surface_side"],
            "direction": pair["direction"], "arm": arm,
            "native_answer_correct": exact.r577.answer_is_best(
                before, answer, pair["recipient"]["answer_text"]),
            "post_answer_correct": exact.r577.answer_is_best(
                patched[output_index], answer, pair["recipient"]["answer_text"]),
            "native_margin": native_margin, "margin_change": after_margin-native_margin,
            "answer_ce_change": exact._ce(patched[output_index], answer)-exact._ce(before, answer),
            "intervention_norm": float(norms[output_index])})
    return evidence, {"native_replay_relative_squared_error": replay_rse,
        "head_source_sum_relative_squared_error": max(
            diagnostics["head_source_sum_relative_squared_error"],
            patch_diagnostics["head_source_sum_relative_squared_error"]),
        "value_split_relative_squared_error": max(diagnostics["value_split_relative_squared_error"],
            patch_diagnostics["value_split_relative_squared_error"]),
        "installed_term_max_absolute_error": installed_error}


def score(evidence, exactness, target, bars):
    grouped = defaultdict(list)
    for row in evidence: grouped[(row["family_id"], row["split"], row["arm"])].append(row)
    reports = {}; decisive_capable = decisive_live = True
    decisive_cached = True; decisive_other = True
    for family in authority.FAMILIES:
        reports[family] = {}
        for split in ("FIT", "SELECT"):
            reports[family][split] = {}
            for arm in ARMS:
                rows = grouped[(family, split, arm)]
                native_accuracy = sum(row["native_answer_correct"] for row in rows)/len(rows)
                preservation = sum(row["post_answer_correct"] for row in rows)/len(rows)
                target_norm = target["intervention_scales"][split][arm]
                target_margin = target["margin_scales"][split][arm]
                report = {"row_count": len(rows), "native_accuracy": native_accuracy,
                    "post_answer_preservation": preservation,
                    "absolute_mean_ce_change": abs(statistics.fmean(row["answer_ce_change"] for row in rows)),
                    "median_absolute_margin_change_fraction_of_target": statistics.median(
                        abs(row["margin_change"]) for row in rows)/max(target_margin, 1e-12),
                    "median_intervention_norm_fraction_of_target": statistics.median(
                        row["intervention_norm"] for row in rows)/max(target_norm, 1e-12)}
                report["capable"] = native_accuracy >= bars["minimum_native_accuracy"]
                report["live"] = report["median_intervention_norm_fraction_of_target"] >= \
                    bars["minimum_intervention_norm_fraction_of_target"]
                report["preserved"] = (preservation >= bars["minimum_preference_preservation"] and
                    report["absolute_mean_ce_change"] <= bars["maximum_absolute_mean_ce_change"] and
                    report["median_absolute_margin_change_fraction_of_target"] <=
                    bars["maximum_median_margin_change_fraction_of_target"])
                reports[family][split][arm] = report
                if family in authority.DECISIVE:
                    decisive_capable &= report["capable"]; decisive_live &= report["live"]
                    if arm == "cached": decisive_cached &= report["preserved"]
                    else: decisive_other &= report["preserved"]
    exact_live = (exactness["native_replay_relative_squared_error"] <= bars["maximum_native_replay_relative_squared_error"] and
        exactness["head_source_sum_relative_squared_error"] <= bars["maximum_source_sum_relative_squared_error"] and
        exactness["value_split_relative_squared_error"] <= bars["maximum_cached_decomposition_relative_squared_error"] and
        exactness["installed_term_max_absolute_error"] <= bars["maximum_installed_term_absolute_error"])
    instrument = target["target_exact"] and target["target_live"] and target["cached_transfer"] and \
        exact_live and decisive_capable and decisive_live
    shared = instrument and decisive_cached and decisive_other
    narrower = instrument and decisive_cached and not decisive_other
    broad = instrument and not decisive_cached
    return {**exactness, "target_binding": target, "family_reports": reports,
        "predictions": {"pred_a_instrument_live": instrument,
            "pred_b_shared_cached_payload_private_router": shared,
            "pred_c_broad_numeral_or_copy_service": broad,
            "pred_d_score_or_joint_collateral_only": narrower}}


def main():
    plan = compile_plan()
    if "--dry-run" in sys.argv:
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    if OUT.exists(): raise RuntimeError(f"refusing to overwrite {OUT}")
    model, checkpoint = exact.r573.facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    evidence, exactness = [], {key: 0. for key in (
        "native_replay_relative_squared_error", "head_source_sum_relative_squared_error",
        "value_split_relative_squared_error", "installed_term_max_absolute_error")}
    pairs = authority.build_pairs()
    for split in ("FIT", "SELECT"):
        split_evidence, split_exact = evaluate_split(model, pairs, split, torch)
        evidence += split_evidence
        exactness = {key: max(exactness[key], split_exact[key]) for key in exactness}
    target = _target_binding(); scored = score(evidence, exactness, target, plan["bars"])
    predictions = scored["predictions"]
    terminal = ("invalid" if not predictions["pred_a_instrument_live"] else
        "shared_cached_payload_private_router" if predictions["pred_b_shared_cached_payload_private_router"] else
        "broad_numeral_or_copy_service" if predictions["pred_c_broad_numeral_or_copy_service"] else
        "score_or_joint_collateral_only" if predictions["pred_d_score_or_joint_collateral_only"] else "invalid")
    result = {"schema": "attn8_h3_h7_cross_behavior_factor_interchange_v4_canonical_controls_result",
        "plan": plan, "prior_art_sha256": PRIOR_ART_SHA256,
        "checkpoint_weights_sha256": checkpoint.weights_sha256, "terminal": terminal,
        "score": scored, "evidence": evidence, "target_endpoints_rerun": False,
        "evaluated_splits": ["FIT", "SELECT"], "model_forwards": 6}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(result, indent=1)+"\n")
    print(json.dumps({"result": str(OUT), "terminal": terminal,
                      "predictions": predictions}, indent=2))


if __name__ == "__main__": main()
