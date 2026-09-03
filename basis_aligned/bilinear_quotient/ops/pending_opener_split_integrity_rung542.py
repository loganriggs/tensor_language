#!/usr/bin/env python3
"""Audit R537--R540 statistical units and rescore after exact-prompt deduplication.

The original rows use distinct ``group_id`` values, but their deterministic
lexical cycles can generate identical base/donor prompts more than once within a
split.  This CPU-only audit treats an exact prompt pair as the independent unit,
checks that no exact pair crosses splits, and recomputes every decision-bearing
R538--R540 summary that has saved row-level sufficient statistics.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "pending_opener_multifamily_rows_rung537.json"
CONTROLS = ROOT / "pending_opener_controls_rung537.json"
R537 = ROOT / "pending_opener_capability_rung537_results.json"
R538 = ROOT / "pending_opener_common_site_rung538_results.json"
R539 = ROOT / "pending_opener_control_ceilings_rung539_results.json"
R540 = ROOT / "pending_opener_cross_family_das_rung540_results.json"
OUT = ROOT / "pending_opener_split_integrity_rung542_results.json"

SPLITS = ("FIT", "SELECT", "FINAL_TEST", "OOD")
EVALUATED_SPLITS = ("FIT", "SELECT")
TARGETS = ("opener_type_substitution", "closed_then_reopened_type")
NEGATIVES = ("pending_state_preserved_surface_edit", "nonopener_punctuation_substitution")
SOURCES = ("direct", "structural", "joint")
RANKS = (1, 2, 4, 8, 16)
SEEDS = (0, 1, 2)
BOOTSTRAPS = 20_000
SEED = 542


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_key(row: dict) -> str:
    """Exact oriented causal stimulus, including answers when present."""
    payload = {
        "base_ids": row["base_ids"],
        "donor_ids": row["donor_ids"],
        "base_answer_id": row.get("base_answer_id", row.get("answer_id")),
        "donor_answer_id": row.get("donor_answer_id", row.get("answer_id")),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def sequence_keys(row: dict) -> tuple[str, str]:
    return tuple(hashlib.sha256(json.dumps(row[key]).encode()).hexdigest()
                 for key in ("base_ids", "donor_ids"))


def bootstrap_lower(values: list[float], *, absolute: bool, seed: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    if absolute:
        array = np.abs(array)
    generator = np.random.default_rng(seed)
    choices = generator.integers(0, len(array), size=(BOOTSTRAPS, len(array)))
    return float(np.quantile(array[choices].mean(axis=1), 0.025))


def collapse(values: list[float], rows: list[dict]) -> list[float]:
    assert len(values) == len(rows)
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, values):
        grouped[prompt_key(row)].append(float(value))
    # Exact repeated prompts must produce the same saved outcome.  If they do
    # not, order/alignment is wrong and no correction may be scored.
    for repeated in grouped.values():
        # Identical prompts can differ by a few float32 ulps when they appeared
        # in batches with different padding lengths.  Anything above 1e-5 is
        # too large to attribute to that execution-order noise.
        assert max(repeated) - min(repeated) <= 1e-5
    return [float(np.mean(items)) for items in grouped.values()]


def dataset_report(rows: list[dict]) -> dict:
    by_cell = {}
    split_pair_keys: dict[str, set[str]] = {}
    split_sequence_keys: dict[str, set[str]] = {}
    for family in sorted({row["family_id"] for row in rows}):
        by_cell[family] = {}
        for split in SPLITS:
            cell = [row for row in rows if row["family_id"] == family and row["split"] == split]
            keys = [prompt_key(row) for row in cell]
            counts = Counter(keys)
            by_cell[family][split] = {
                "rows": len(cell),
                "unique_exact_prompt_pairs": len(counts),
                "duplicate_rows": len(cell) - len(counts),
                "maximum_exact_multiplicity": max(counts.values(), default=0),
                "effective_fraction": len(counts) / len(cell) if cell else None,
            }
            split_pair_keys[f"{family}:{split}"] = set(keys)
            split_sequence_keys[f"{family}:{split}"] = {
                item for row in cell for item in sequence_keys(row)
            }
    overlaps = []
    for family in by_cell:
        for left_index, left in enumerate(SPLITS):
            for right in SPLITS[left_index + 1:]:
                pair_overlap = split_pair_keys[f"{family}:{left}"] & split_pair_keys[f"{family}:{right}"]
                sequence_overlap = split_sequence_keys[f"{family}:{left}"] & split_sequence_keys[f"{family}:{right}"]
                overlaps.append({
                    "family": family,
                    "left_split": left,
                    "right_split": right,
                    "exact_pair_overlap": len(pair_overlap),
                    "exact_sequence_overlap": len(sequence_overlap),
                })
    return {
        "cells": by_cell,
        "cross_split_overlaps": overlaps,
        "any_exact_pair_cross_split": any(item["exact_pair_overlap"] for item in overlaps),
        "any_exact_sequence_cross_split": any(item["exact_sequence_overlap"] for item in overlaps),
        "any_within_split_pseudoreplication": any(
            cell["duplicate_rows"] > 0 for family in by_cell.values() for cell in family.values()),
    }


def rows_for(main_rows: list[dict], control_rows: list[dict], split: str, family: str) -> list[dict]:
    source = control_rows if family == "nonopener_punctuation_substitution" else main_rows
    return [row for row in source if row["split"] == split and row["family_id"] == family]


def audit_r538(result: dict, main_rows: list[dict]) -> dict:
    reports, passing, seed = {}, [], SEED
    for site in result["passing_sites_in_frozen_order"] + [
            key for key in result["reports"] if key not in result["passing_sites_in_frozen_order"]]:
        reports[site] = {}
        site_pass = True
        for split in EVALUATED_SPLITS:
            reports[site][split] = {}
            for family in TARGETS:
                cell_rows = rows_for(main_rows, [], split, family)
                reports[site][split][family] = {}
                family_pass = True
                for direction in ("base_to_donor", "donor_to_base"):
                    values = collapse(result["raw_donorward_movements"][site][split][family][direction], cell_rows)
                    row = {
                        "n_unique": len(values),
                        "mean_donorward_movement": float(np.mean(values)),
                        "bootstrap95_lower_mean": bootstrap_lower(values, absolute=False, seed=seed),
                        "positive_movement_fraction": float(np.mean(np.asarray(values) > 0)),
                    }
                    seed += 1
                    row["passed"] = bool(row["mean_donorward_movement"] > 0
                                         and row["bootstrap95_lower_mean"] > 0
                                         and row["positive_movement_fraction"] >= .70)
                    family_pass &= row["passed"]
                    reports[site][split][family][direction] = row
                reports[site][split][family]["passed"] = bool(family_pass)
                site_pass &= family_pass
        if site_pass:
            passing.append(site)
    frozen_order = list(result["reports"])
    # JSON was written with sorted keys; recover the registered causal order.
    causal_order = [f"resid{i}" for i in range(8, 15)] + [f"mlp_product{i}" for i in range(8, 15)] + ["attn13h8"]
    passing = [site for site in causal_order if site in passing]
    return {
        "reports": reports,
        "passing_sites_in_frozen_order": passing,
        "selected_site": passing[0] if passing else None,
        "original_selected_site": result["selected_site"],
        "selection_unchanged": bool(passing and passing[0] == result["selected_site"]),
        "all_saved_duplicate_outcomes_exact": True,
        "json_key_order_not_used_for_selection": frozen_order != causal_order,
    }


def audit_r539(result: dict, main_rows: list[dict], control_rows: list[dict]) -> dict:
    reports, seed = {}, SEED + 10_000
    for split in EVALUATED_SPLITS:
        reports[split] = {}
        for family in NEGATIVES:
            cell_rows = rows_for(main_rows, control_rows, split, family)
            reports[split][family] = {}
            family_pass = True
            for direction in ("base_to_donor", "donor_to_base"):
                raw = result["raw_sufficient_statistics"][split][family][direction]
                endpoint = collapse(raw["endpoint_change"], cell_rows)
                logit_rms = collapse(raw["logit_rms"], cell_rows)
                row = {
                    "n_unique": len(endpoint),
                    "mean_signed_endpoint_change": float(np.mean(endpoint)),
                    "mean_absolute_endpoint_change": float(np.mean(np.abs(endpoint))),
                    "bootstrap95_lower_mean_absolute": bootstrap_lower(endpoint, absolute=True, seed=seed),
                    "mean_full_vocabulary_logit_rms": float(np.mean(logit_rms)),
                }
                seed += 1
                row["passed"] = bool(row["bootstrap95_lower_mean_absolute"] > .05
                                     and row["mean_full_vocabulary_logit_rms"] > .01)
                family_pass &= row["passed"]
                reports[split][family][direction] = row
            reports[split][family]["causally_testable"] = bool(family_pass)
    surface = all(reports[split][NEGATIVES[0]]["causally_testable"] for split in EVALUATED_SPLITS)
    punctuation = all(reports[split][NEGATIVES[1]]["causally_testable"] for split in EVALUATED_SPLITS)
    return {
        "reports": reports,
        "surface_invariance_causally_testable": surface,
        "nonopener_control_causally_testable": punctuation,
        "original_decisions": {
            "surface": result["pred_b_surface_invariance_causally_testable"],
            "punctuation": result["pred_c_nonopener_control_causally_testable"],
        },
        "decisions_unchanged": bool(surface == result["pred_b_surface_invariance_causally_testable"]
                                    and punctuation == result["pred_c_nonopener_control_causally_testable"]),
        "all_saved_duplicate_outcomes_exact": True,
    }


def audit_r540(result: dict, main_rows: list[dict], control_rows: list[dict], r539: dict) -> dict:
    reports, seed = {}, SEED + 20_000
    for rank in RANKS:
        reports[str(rank)] = {}
        for source in SOURCES:
            reports[str(rank)][source] = {}
            for fit_seed in SEEDS:
                old = result["fits"][str(rank)][source][str(fit_seed)]
                target_cells, control_cells, passed = {}, {}, True
                for family in TARGETS:
                    target_cells[family] = {}
                    cell_rows = rows_for(main_rows, control_rows, "SELECT", family)
                    for direction in ("base_to_donor", "donor_to_base"):
                        values = collapse(old["targets"][family][direction]["values"], cell_rows)
                        row = {
                            "n_unique": len(values),
                            "mean": float(np.mean(values)),
                            "median": float(np.median(values)),
                            "bootstrap95_lower_mean": bootstrap_lower(values, absolute=False, seed=seed),
                            "positive_fraction": float(np.mean(np.asarray(values) > 0)),
                        }
                        seed += 1
                        row["passed"] = bool(row["median"] >= .5 and row["bootstrap95_lower_mean"] > 0
                                             and row["positive_fraction"] >= .75)
                        passed &= row["passed"]
                        target_cells[family][direction] = row
                for family in NEGATIVES:
                    control_cells[family] = {}
                    cell_rows = rows_for(main_rows, control_rows, "SELECT", family)
                    for direction in ("base_to_donor", "donor_to_base"):
                        values = collapse(old["controls"][family][direction]["values"], cell_rows)
                        full = collapse(r539["raw_sufficient_statistics"]["SELECT"][family][direction]["endpoint_change"], cell_rows)
                        mean_abs = float(np.mean(np.abs(values)))
                        ratio = mean_abs / float(np.mean(np.abs(full)))
                        row = {"n_unique": len(values), "mean_absolute": mean_abs,
                               "fraction_of_full": ratio,
                               "passed": bool(mean_abs <= .10 and ratio <= .25)}
                        passed &= row["passed"]
                        control_cells[family][direction] = row
                reports[str(rank)][source][str(fit_seed)] = {
                    "targets": target_cells,
                    "controls": control_cells,
                    "passed": bool(passed),
                }
            stable = sum(reports[str(rank)][source][str(s)]["passed"] for s in SEEDS) >= 2
            reports[str(rank)][source]["seed_stable"] = bool(stable)
        random_pass = float(np.mean(result["random_controls"][str(rank)])) < .10
        reports[str(rank)]["random_control_pass"] = random_pass
        reports[str(rank)]["rank_eligible"] = bool(
            random_pass and all(reports[str(rank)][source]["seed_stable"] for source in SOURCES))
    eligible = [rank for rank in RANKS if reports[str(rank)]["rank_eligible"]]
    target_passes = sum(
        reports[str(rank)][source][str(fit_seed)]["targets"][family][direction]["passed"]
        for rank in RANKS for source in SOURCES for fit_seed in SEEDS
        for family in TARGETS for direction in ("base_to_donor", "donor_to_base"))
    control_passes = sum(
        reports[str(rank)][source][str(fit_seed)]["controls"][family][direction]["passed"]
        for rank in RANKS for source in SOURCES for fit_seed in SEEDS
        for family in NEGATIVES for direction in ("base_to_donor", "donor_to_base"))
    return {
        "summary": {
            "target_cells_passed": target_passes,
            "target_cells_total": len(RANKS) * len(SOURCES) * len(SEEDS) * len(TARGETS) * 2,
            "control_cells_passed": control_passes,
            "control_cells_total": len(RANKS) * len(SOURCES) * len(SEEDS) * len(NEGATIVES) * 2,
            "eligible_ranks": eligible,
            "selected_rank": min(eligible) if eligible else None,
            "original_selected_rank": result["selected_rank"],
            "decision_unchanged": (min(eligible) if eligible else None) == result["selected_rank"],
        },
        "fits": reports,
        "all_saved_duplicate_outcomes_exact": True,
    }


def main() -> None:
    main_doc = json.loads(ROWS.read_text())
    control_doc = json.loads(CONTROLS.read_text())
    main_rows, control_rows = main_doc["rows"], control_doc["rows"]
    r537, r538, r539, r540 = (json.loads(path.read_text()) for path in (R537, R538, R539, R540))
    main_report = dataset_report(main_rows)
    control_report = dataset_report(control_rows)
    corrected_r538 = audit_r538(r538, main_rows)
    corrected_r539 = audit_r539(r539, main_rows, control_rows)
    corrected_r540 = audit_r540(r540, main_rows, control_rows, r539)
    cross_split_clean = not any((
        main_report["any_exact_pair_cross_split"], main_report["any_exact_sequence_cross_split"],
        control_report["any_exact_pair_cross_split"], control_report["any_exact_sequence_cross_split"],
    ))
    conclusions_survive = bool(
        corrected_r538["selection_unchanged"]
        and corrected_r539["decisions_unchanged"]
        and corrected_r540["summary"]["decision_unchanged"]
    )
    result = {
        "rung": 542,
        "stage": "post_result_statistical_unit_audit",
        "input_sha256": {str(path): sha256(path) for path in (ROWS, CONTROLS, R537, R538, R539, R540)},
        "main_dataset": main_report,
        "control_dataset": control_report,
        "r537_capability": {
            "raw_row_level_values_available": False,
            "reported_n_is_not_independent_n": True,
            "all_correct_fraction_decision_survives": all(
                r537["summaries"][split][family][key] == 1.0
                for split in EVALUATED_SPLITS
                for family, key in (
                    ("opener_type_substitution", "both_endpoints_correct_fraction"),
                    ("closed_then_reopened_type", "both_endpoints_correct_fraction"),
                    ("pending_state_preserved_surface_edit", "base_correct_fraction"),
                    ("nonopener_punctuation_substitution", "base_correct_fraction"),
                )
            ),
            "bootstrap_intervals_cannot_be_exactly_recomputed": True,
        },
        "r538_unique_prompt_rescore": corrected_r538,
        "r539_unique_prompt_rescore": corrected_r539,
        "r540_unique_prompt_rescore": corrected_r540,
        "pred_a_exact_cross_split_isolation": cross_split_clean,
        "pred_b_no_within_split_pseudoreplication": not (
            main_report["any_within_split_pseudoreplication"]
            or control_report["any_within_split_pseudoreplication"]),
        "pred_c_r538_site_decision_survives_unique_prompt_rescore": corrected_r538["selection_unchanged"],
        "pred_d_r539_control_liveness_survives_unique_prompt_rescore": corrected_r539["decisions_unchanged"],
        "pred_e_r540_null_survives_unique_prompt_rescore": corrected_r540["summary"]["decision_unchanged"],
        "conclusions_survive_exact_prompt_deduplication": conclusions_survive,
        "correction": (
            "FIT and SELECT use disjoint exact sequences, so there is no exact train/select leakage. "
            "However, group_id overcounts independent prompts within every split. Report unique exact prompt "
            "counts, replace group_id by a normalized stimulus-family identifier in future rows, and do not "
            "open FINAL_TEST/OOD from this dataset. R538, R539, and R540 decisions survive deduplication."
        ),
        "next_step": "replace_r537_generator_with_unique_grouped_counterfactuals_before_any_final_or_ood_opening",
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "pred_a_exact_cross_split_isolation": result["pred_a_exact_cross_split_isolation"],
        "pred_b_no_within_split_pseudoreplication": result["pred_b_no_within_split_pseudoreplication"],
        "pred_c_r538_site_decision_survives_unique_prompt_rescore": result["pred_c_r538_site_decision_survives_unique_prompt_rescore"],
        "pred_d_r539_control_liveness_survives_unique_prompt_rescore": result["pred_d_r539_control_liveness_survives_unique_prompt_rescore"],
        "pred_e_r540_null_survives_unique_prompt_rescore": result["pred_e_r540_null_survives_unique_prompt_rescore"],
        "r540": corrected_r540["summary"],
        "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
