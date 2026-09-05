#!/usr/bin/env python3
"""Thin exact runner for context-local semantic-opener shared/contrast terms."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import circuit_fast_screen_candidate_bracket_l13h8_shared_contrast as authority
import run_bracket_l13h8_source_region_payload_factorial as shared
import semantic_opener_effect_coding as coding


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits" / "fast_screens" / "bracket_l13h8_semantic_open_shared_contrast_v1_result.json"
CLOSERS = (8, 60, 1)


def pad_rows(rows, torch, device):
    length = max(len(row["ids"]) for row in rows)
    tokens = torch.full((len(rows), length), 50256, dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        tokens[index, :len(row["ids"])] = torch.tensor(row["ids"], device=device)
    finals = torch.tensor([row["final_position"] for row in rows], device=device)
    sources = torch.tensor([row["open_position"] for row in rows], device=device)
    return tokens, finals, sources


def evaluate(model, torch, F, facade):
    rows, device = authority.ROWS, next(model.parameters()).device
    tokens, finals, sources = pad_rows(rows, torch, device)
    masks = {}
    native = shared.native_logits(model, tokens, torch, F)
    replay, factors = shared.factor_forward(model, tokens, finals, masks, torch, F, facade)
    replay_error = float((native - replay).abs().max())
    arange = torch.arange(len(rows), device=device)
    terms = factors["p"][arange, sources].unsqueeze(-1) * factors["u"][arange, sources]
    by_group = defaultdict(dict)
    for index, row in enumerate(rows):
        by_group[row["group_id"]][row["delimiter_index"]] = index
    shared_terms = torch.empty_like(terms)
    donor_indices = []
    for index, row in enumerate(rows):
        indices = [by_group[row["group_id"]][delimiter] for delimiter in range(3)]
        mu = sum((terms[item] for item in indices)) / 3.0
        shared_terms[index] = mu
        donor_indices.append(by_group[row["group_id"]][(row["delimiter_index"] + 1) % 3])
    donor_indices = torch.tensor(donor_indices, device=device)
    natural_terms = terms[donor_indices]
    contrast_terms = terms - shared_terms
    natural = shared.factor_forward(model, tokens, finals, masks, torch, F, facade,
                                    replacement_terms=natural_terms, source_positions=sources)[0]
    contrast_removed = shared.factor_forward(model, tokens, finals, masks, torch, F, facade,
                                             replacement_terms=shared_terms, source_positions=sources)[0]
    shared_removed = shared.factor_forward(model, tokens, finals, masks, torch, F, facade,
                                           replacement_terms=contrast_terms, source_positions=sources)[0]
    records = []
    for index, row in enumerate(rows):
        q, donor_index = int(finals[index]), int(donor_indices[index])
        donor_answer = rows[donor_index]["answer_id"]
        native_type, native_common = coding.closer_type_and_common_axes(replay[index, q], CLOSERS, row["answer_id"])
        donor_before, _ = coding.closer_type_and_common_axes(replay[index, q], CLOSERS, donor_answer)
        donor_after, _ = coding.closer_type_and_common_axes(natural[index, q], CLOSERS, donor_answer)
        removed_type, removed_common = coding.closer_type_and_common_axes(
            contrast_removed[index, q], CLOSERS, row["answer_id"])
        shared_type, shared_common = coding.closer_type_and_common_axes(
            shared_removed[index, q], CLOSERS, row["answer_id"])
        type_damage = float(native_type - removed_type)
        common_change = abs(float(removed_common - native_common))
        shared_common_damage = abs(float(native_common - shared_common))
        shared_type_damage = abs(float(native_type - shared_type))
        records.append({
            "row_id": row["row_id"], "group_id": row["group_id"], "bundle_id": row["bundle_id"],
            "matched_direct_group_id": row["matched_direct_group_id"], "family_id": row["family_id"],
            "role": row["role"], "delimiter_index": row["delimiter_index"],
            "native_type_axis": float(native_type), "native_common_axis": float(native_common),
            "natural_swap_type_transfer": float(donor_after - donor_before),
            "contrast_removal_type_damage": type_damage,
            "contrast_removal_common_change": common_change,
            "contrast_type_to_common_ratio": abs(type_damage) / (common_change + 1e-6),
            "contrast_normalized_type_damage": type_damage / max(abs(float(native_type)), 1e-6),
            "shared_removal_common_damage": shared_common_damage,
            "shared_removal_type_damage": shared_type_damage,
            "shared_common_to_type_ratio": shared_common_damage / (shared_type_damage + 1e-6),
            "semantic_open_term_norm": float(terms[index].norm()),
        })
    return records, replay_error


def score(records, replay_error):
    bars = authority.compile_plan()["bars"]
    by_family = defaultdict(list)
    for row in records: by_family[row["family_id"]].append(row)
    native = {family: sum(row["native_type_axis"] > 0 for row in rows) / len(rows)
              for family, rows in by_family.items()}
    target = {}
    for family in authority.TARGET_FAMILIES:
        rows = by_family[family]
        target[family] = {
            "natural_swap_positive_fraction": sum(row["natural_swap_type_transfer"] > 0 for row in rows) / len(rows),
            "contrast_damage_positive_fraction": sum(row["contrast_removal_type_damage"] > 0 for row in rows) / len(rows),
            "median_type_to_common_ratio": statistics.median(row["contrast_type_to_common_ratio"] for row in rows),
        }
        target[family]["passed"] = bool(
            target[family]["natural_swap_positive_fraction"]
            >= bars["natural_swap_positive_type_transfer_fraction_each_target_family_min"]
            and target[family]["contrast_damage_positive_fraction"]
            >= bars["contrast_removal_positive_type_damage_fraction_each_target_family_min"]
            and target[family]["median_type_to_common_ratio"]
            >= bars["contrast_removal_type_to_common_median_ratio_each_target_family_min"])
    shared_ratio = statistics.median(row["shared_common_to_type_ratio"] for row in records)
    direct_by_bundle = {(row["bundle_id"], row["delimiter_index"]): row["contrast_normalized_type_damage"]
                        for row in by_family["direct_type"]}
    invariance_differences = [abs(row["contrast_normalized_type_damage"]
                                  - direct_by_bundle[(row["bundle_id"], row["delimiter_index"])])
                              for family in authority.CONTROL_FAMILIES for row in by_family[family]]
    invariance = statistics.median(invariance_differences)
    instrument = {"native_capability": min(native.values()) >= bars["native_positive_fraction_each_family_min"],
                  "native_replay": replay_error <= bars["native_replay_max_absolute_logit_error_max"],
                  "live_terms": min(row["semantic_open_term_norm"] for row in records)
                  >= bars["semantic_open_term_norm_min"]}
    pred_a = all(instrument.values())
    pred_b = (pred_a and all(item["passed"] for item in target.values())
              and shared_ratio >= bars["shared_removal_common_to_type_median_ratio_min"]
              and invariance <= bars["matched_invariance_median_absolute_ratio_difference_max"])
    pred_c = pred_a and not pred_b
    return {"instrument_checks": instrument, "target_families": target,
            "shared_removal_median_common_to_type_ratio": shared_ratio,
            "matched_invariance_median_absolute_ratio_difference": invariance,
            "shared_plus_contrast_held": pred_b,
            "predictions": {"pred_a_instrument_live": pred_a,
                            "pred_b_shared_plus_contrast_held": pred_b,
                            "pred_c_pure_shared_or_entangled": pred_c}}


def main():
    plan = authority.compile_plan()
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv:
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    if OUT.exists(): raise RuntimeError(f"refusing to overwrite {OUT}")
    shared.candidate = authority
    torch, F, facade = shared._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad(): records, replay_error = evaluate(model, torch, F, facade)
    screen = score(records, replay_error)
    instrument_live = all(screen["instrument_checks"].values())
    terminal = "screen" if screen["shared_plus_contrast_held"] else ("null" if instrument_live else "invalid")
    result = {"schema": "bracket_l13h8_semantic_open_shared_contrast_result_v1", "plan": plan,
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "native_replay_max_absolute_logit_error": replay_error, "raw": records, "screen": screen,
              "evaluated_splits": ["FRESH_BASIC"], "forbidden_splits_opened": [],
              "model_forwards": 5, "terminal": terminal}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"result": str(OUT), "terminal": terminal, "model_forwards": 5}, indent=2))


if __name__ == "__main__": main()
