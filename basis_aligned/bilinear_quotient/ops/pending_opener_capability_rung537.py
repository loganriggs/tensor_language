#!/usr/bin/env python3
"""Managed FIT/SELECT-only capability gate for pending-opener counterfactuals."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import collections
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for search_path in (ROOT, ROOT / "ops"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))
import bqlib as B  # noqa: E402

ROWS = ROOT / "pending_opener_multifamily_rows_rung537.json"
ROWS_RECEIPT = ROOT / "pending_opener_multifamily_rows_rung537_receipt.json"
CONTROLS = ROOT / "pending_opener_controls_rung537.json"
CONTROLS_RECEIPT = ROOT / "pending_opener_controls_rung537_receipt.json"
PREREG = POLY / "PENDING_OPENER_MULTI_COUNTERFACTUAL_RUNG537_PREREGISTRATION.md"
OUT = ROOT / "pending_opener_capability_rung537_results.json"
HASHES = {
    ROWS: "c62cdf3929231e06de6883d74f3ab2c86bd524e02474bb2259267d6976e9e7d9",
    ROWS_RECEIPT: "d50528aa355ba89ab43edd43491c672a6aed88bd8a805ffda936afbfa4cc4816",
    CONTROLS: "f2693b9b78a9266619afc45ceb6f70e4f2339aa1980263ca22d3ea4453145494",
    CONTROLS_RECEIPT: "1ad594b4fc19abc3dab761f7cd09f3c9df764f7390ada09dc334f4b7268e626c",
    PREREG: "5a64b1f523f5e70973cb64ad6b5840fc15e252dc679591107d5f7ab3634f8b2c",
}
ALLOWED_SPLITS = {"FIT", "SELECT"}
BATCH = 16
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 537
EXPECTED_MAIN_ROWS = 192
EXPECTED_CONTROL_ROWS = 64
EXPECTED_SEQUENCES = 2 * (EXPECTED_MAIN_ROWS + EXPECTED_CONTROL_ROWS)
EXPECTED_FORWARDS = math.ceil(EXPECTED_SEQUENCES / BATCH)

PREDICTIONS = {
    "pred_a_valid_outcome_boundary": "all frozen hashes, split exclusion, row counts, and forward counts match",
    "pred_b_direct_type_capability": "direct opener substitution passes both-endpoint capability on FIT and SELECT",
    "pred_c_structural_type_capability": "closed-then-reopened construction passes both-endpoint capability on FIT and SELECT",
    "pred_d_invariance_capability": "surface/distance and non-opener punctuation controls retain the registered answer",
    "pred_e_dataset_authorized_for_site_screen": "all capability families pass without opening FINAL_TEST or OOD",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inputs() -> tuple[list[dict], list[dict]]:
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen input mismatch: {path}")
    main = json.loads(ROWS.read_text())
    controls = json.loads(CONTROLS.read_text())
    if main["status"] != "rows_frozen_outcomes_unopened" or controls["status"] != "controls_frozen_outcomes_unopened":
        raise RuntimeError("row authorities are not outcome-closed")
    selected_main = [row for row in main["rows"] if row["split"] in ALLOWED_SPLITS]
    selected_controls = [row for row in controls["rows"] if row["split"] in ALLOWED_SPLITS]
    if len(selected_main) != EXPECTED_MAIN_ROWS or len(selected_controls) != EXPECTED_CONTROL_ROWS:
        raise RuntimeError("FIT/SELECT row-count mismatch")
    if any(row["split"] not in ALLOWED_SPLITS for row in selected_main + selected_controls):
        raise RuntimeError("forbidden FINAL_TEST/OOD row reached capability stage")
    return selected_main, selected_controls


def bootstrap_lower(values: list[float], seed_offset: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    indices = generator.integers(0, len(array), size=(BOOTSTRAPS, len(array)))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025))


@torch.no_grad()
def score_sequences(entries: list[dict]) -> tuple[dict[tuple[str, str], dict[int, float]], int]:
    requests = []
    token_ids = set()
    for row in entries:
        for side in ("base", "donor"):
            ids = row[f"{side}_ids"]
            requests.append((row["row_id"], side, ids))
        if "base_answer_id" in row:
            token_ids.update((row["base_answer_id"], row["donor_answer_id"]))
        else:
            token_ids.add(row["answer_id"])
        token_ids.update(row.get("wrong_closer_ids", []))
    token_ids.update((1, 8, 60, 92))  # quote, parenthesis, square, curly closers
    ordered_tokens = sorted(token_ids)
    scores = {}
    calls = 0
    for start in range(0, len(requests), BATCH):
        chunk = requests[start:start + BATCH]
        length = max(len(item[2]) for item in chunk)
        batch = torch.full((len(chunk), length), 50256, dtype=torch.long, device=B.DEV)
        finals = []
        for index, (_, _, ids) in enumerate(chunk):
            batch[index, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=B.DEV)
            finals.append(len(ids) - 1)
        logits = B.forward_logits(batch)
        calls += 1
        for index, (row_id, side, _) in enumerate(chunk):
            vector = logits[index, finals[index]].float().cpu()
            scores[(row_id, side)] = {token: float(vector[token]) for token in ordered_tokens}
        del logits, batch
    return scores, calls


def interchange_summary(rows: list[dict], scores: dict, split: str, family_id: str, seed_offset: int) -> dict:
    selected = [row for row in rows if row["split"] == split and row["family_id"] == family_id]
    separations, base_margins, donor_margins = [], [], []
    for row in selected:
        base = scores[(row["row_id"], "base")]
        donor = scores[(row["row_id"], "donor")]
        base_margin = base[row["base_answer_id"]] - base[row["donor_answer_id"]]
        donor_margin = donor[row["donor_answer_id"]] - donor[row["base_answer_id"]]
        base_margins.append(base_margin)
        donor_margins.append(donor_margin)
        separations.append(0.5 * (base_margin + donor_margin))
    correct_both = np.logical_and(np.asarray(base_margins) > 0, np.asarray(donor_margins) > 0)
    correct_fraction = float(correct_both.mean())
    mean_separation = float(np.mean(separations))
    lower = bootstrap_lower(separations, seed_offset)
    passed = correct_fraction >= 0.75 and mean_separation > 0.5 and lower > 0
    return {
        "n": len(selected),
        "both_endpoints_correct_fraction": correct_fraction,
        "mean_symmetric_logit_separation": mean_separation,
        "bootstrap95_lower_symmetric_separation": lower,
        "mean_base_margin": float(np.mean(base_margins)),
        "mean_donor_margin": float(np.mean(donor_margins)),
        "passed": passed,
    }


def invariance_summary(rows: list[dict], scores: dict, split: str, family_id: str) -> dict:
    selected = [row for row in rows if row["split"] == split and row["family_id"] == family_id]
    base_margins, donor_margins, wrong_gaps = [], [], []
    for row in selected:
        base = scores[(row["row_id"], "base")]
        donor = scores[(row["row_id"], "donor")]
        answer_id = row.get("answer_id", row.get("base_answer_id"))
        comparison_id = 1 if answer_id == 8 else 8
        base_margins.append(base[answer_id] - base[comparison_id])
        donor_margins.append(donor[answer_id] - donor[comparison_id])
        wrong_ids = row.get("wrong_closer_ids", [60, 92])
        wrong_gaps.extend(
            [base[answer_id] - max(base[token] for token in wrong_ids), donor[answer_id] - max(donor[token] for token in wrong_ids)]
        )
    base_correct = float((np.asarray(base_margins) > 0).mean())
    donor_correct = float((np.asarray(donor_margins) > 0).mean())
    passed = base_correct >= 0.75 and donor_correct >= 0.75
    return {
        "n": len(selected),
        "base_correct_fraction": base_correct,
        "donor_correct_fraction": donor_correct,
        "mean_base_answer_margin": float(np.mean(base_margins)),
        "mean_donor_answer_margin": float(np.mean(donor_margins)),
        "mean_correct_minus_wrong_closer_margin": float(np.mean(wrong_gaps)),
        "passed": passed,
    }


def main() -> None:
    started = time.time()
    main_rows, control_rows = validate_inputs()
    if B.DRYRUN:
        print(json.dumps({
            "status": "dryrun_passed",
            "main_rows": len(main_rows),
            "control_rows": len(control_rows),
            "expected_sequences": EXPECTED_SEQUENCES,
            "expected_forwards": EXPECTED_FORWARDS,
            "allowed_splits": sorted(ALLOWED_SPLITS),
        }, indent=2))
        return

    scores, calls = score_sequences(main_rows + control_rows)
    summaries = {}
    offset = 0
    for split in ("FIT", "SELECT"):
        summaries[split] = {}
        for family_id in ("opener_type_substitution", "closed_then_reopened_type"):
            summaries[split][family_id] = interchange_summary(main_rows, scores, split, family_id, offset)
            offset += 1
        summaries[split]["pending_state_preserved_surface_edit"] = invariance_summary(
            main_rows, scores, split, "pending_state_preserved_surface_edit"
        )
        summaries[split]["nonopener_punctuation_substitution"] = invariance_summary(
            control_rows, scores, split, "nonopener_punctuation_substitution"
        )

    pred_a_valid_outcome_boundary = calls == EXPECTED_FORWARDS
    pred_b_direct_type_capability = all(
        summaries[split]["opener_type_substitution"]["passed"] for split in ("FIT", "SELECT")
    )
    pred_c_structural_type_capability = all(
        summaries[split]["closed_then_reopened_type"]["passed"] for split in ("FIT", "SELECT")
    )
    pred_d_invariance_capability = all(
        summaries[split][family]["passed"]
        for split in ("FIT", "SELECT")
        for family in ("pending_state_preserved_surface_edit", "nonopener_punctuation_substitution")
    )
    pred_e_dataset_authorized_for_site_screen = all((
        pred_a_valid_outcome_boundary,
        pred_b_direct_type_capability,
        pred_c_structural_type_capability,
        pred_d_invariance_capability,
    ))
    result = {
        "rung": 537,
        "stage": "fit_select_capability",
        "predictions": PREDICTIONS,
        "strong_null": not pred_e_dataset_authorized_for_site_screen,
        "summaries": summaries,
        "input_sha256": {str(path): expected for path, expected in HASHES.items()},
        "evaluated_splits": sorted(ALLOWED_SPLITS),
        "forbidden_splits_opened": [],
        "sequence_count": EXPECTED_SEQUENCES,
        "model_forwards": calls,
        "model_backwards": 0,
        "elapsed_seconds": time.time() - started,
        "next_step": "implement_common_site_ceiling" if pred_e_dataset_authorized_for_site_screen else "repair_or_reject_failed_counterfactual_family",
    }
    result.update(dict(zip(PREDICTIONS, (
        pred_a_valid_outcome_boundary,
        pred_b_direct_type_capability,
        pred_c_structural_type_capability,
        pred_d_invariance_capability,
        pred_e_dataset_authorized_for_site_screen,
    ))))
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
