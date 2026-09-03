#!/usr/bin/env python3
"""Checkpoint-verified rerun of the frozen R538 full-state site screen."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import torch


# Import the frozen computation without allowing its bqlib import to allocate a
# second, independently loaded model.  The verified facade below owns the only
# model used by this process.
os.environ["BQLIB_NO_MODEL"] = "1"
ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for search_path in (ROOT, ROOT / "ops", POLY):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))
import bilin18_observed_model_facade as facade  # noqa: E402
import pending_opener_common_site_rung538 as core  # noqa: E402

OUT = ROOT / "pending_opener_common_site_rung538_results.json"
INVALID_RESULT = ROOT / "pending_opener_common_site_rung538_invalid_unverified_checkpoint_results.json"
INVALID_RESULT_SHA256 = "6439e74b792b5a24fae8323f6b7fae97d2cc7b1f583a5b6e228f679537ccc62a"
DEVICE = "cuda"


def main() -> None:
    started = time.time()
    rows = core.selected_rows()
    if core.sha256(INVALID_RESULT) != INVALID_RESULT_SHA256:
        raise RuntimeError("the preserved invalid receipt changed")
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dryrun_passed", "correction": "verify actual checkpoint bytes",
            "pairs": len(rows), "sites": list(core.SITE_ORDER),
            "expected_forwards": core.EXPECTED_FORWARDS,
            "scientific_contract_changed": False,
        }, indent=2))
        return

    model, checkpoint = facade.load_bilin18(
        device=DEVICE, dtype=torch.float32, verify_weights_sha256=True)
    raw = {
        site: {
            split: {family: {"base_to_donor": [], "donor_to_base": []}
                    for family in core.FAMILIES}
            for split in core.SPLITS
        } for site in core.SITE_ORDER
    }
    calls, min_edit_rms = 0, float("inf")
    for start in range(0, len(rows), core.PAIR_BATCH):
        chunk = rows[start:start + core.PAIR_BATCH]
        length = max(len(row["base_ids"]) for row in chunk)
        base = torch.full((len(chunk), length), 50256, dtype=torch.long, device=DEVICE)
        donor = base.clone()
        finals = torch.tensor([len(row["base_ids"]) - 1 for row in chunk], device=DEVICE)
        for index, row in enumerate(chunk):
            base[index, :len(row["base_ids"])] = torch.tensor(row["base_ids"], device=DEVICE)
            donor[index, :len(row["donor_ids"])] = torch.tensor(row["donor_ids"], device=DEVICE)
        both = torch.cat((base, donor))
        both_finals = torch.cat((finals, finals))
        native, states = core.capture_all(model, both, both_finals)
        calls += 1
        arange = torch.arange(len(chunk), device=DEVICE)
        base_native = native[arange, finals]
        donor_native = native[arange + len(chunk), finals]
        for site in core.SITE_ORDER:
            base_state, donor_state = states[site].chunk(2)
            edit_rms = (donor_state.float() - base_state.float()).square().mean(-1).sqrt()
            min_edit_rms = min(min_edit_rms, float(edit_rms.min()))
            base_patch = core.patched(model, base, finals, site, donor_state)[arange, finals]
            donor_patch = core.patched(model, donor, finals, site, base_state)[arange, finals]
            calls += 2
            for index, row in enumerate(chunk):
                yb, yd = row["base_answer_id"], row["donor_answer_id"]
                b0 = float(base_native[index, yd] - base_native[index, yb])
                bp = float(base_patch[index, yd] - base_patch[index, yb])
                d0 = float(donor_native[index, yb] - donor_native[index, yd])
                dp = float(donor_patch[index, yb] - donor_patch[index, yd])
                cell = raw[site][row["split"]][row["family_id"]]
                cell["base_to_donor"].append(bp - b0)
                cell["donor_to_base"].append(dp - d0)
        del base, donor, both, native, states

    reports, passing = core.score(raw)
    instrument_valid = bool(
        calls == core.EXPECTED_FORWARDS and min_edit_rms > 0
        and checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
    )
    selected = passing[0] if instrument_valid and passing else None
    frozen_selection_valid = bool(
        selected is None or selected == next(site for site in core.SITE_ORDER if site in passing)
    )
    result = {
        "rung": 538, "version": 2,
        "stage": "common_site_full_state_interchange",
        "correction": "actual locally loaded checkpoint bytes verified before model construction",
        "scientific_contract_changed": False,
        "invalid_predecessor_sha256": INVALID_RESULT_SHA256,
        "pred_a_exact_instrument": instrument_valid,
        "pred_b_common_live_site": selected is not None,
        "pred_c_frozen_causal_order_selection": frozen_selection_valid,
        "strong_null": bool(instrument_valid and frozen_selection_valid and selected is None),
        "selected_site": selected, "passing_sites_in_frozen_order": passing,
        "reports": reports,
        # Row-level sufficient statistics make every mean, fraction, and
        # bootstrap result independently recomputable without another forward.
        "raw_donorward_movements": raw,
        "evaluated_splits": list(core.SPLITS), "forbidden_splits_opened": [],
        "evaluated_families": list(core.FAMILIES),
        "model_forwards": calls, "model_backwards": 0,
        "minimum_source_target_activation_rms": min_edit_rms,
        "checkpoint": {
            "revision": checkpoint.revision,
            "config_sha256": checkpoint.config_sha256,
            "weights_sha256": checkpoint.weights_sha256,
            "weights_bytes": checkpoint.weights_bytes,
        },
        "input_sha256": {str(path): expected for path, expected in core.HASHES.items()},
        "implementation_price": {
            "pairs": len(rows), "sites": len(core.SITE_ORDER),
            "baseline_forwards": core.EXPECTED_BASELINE_FORWARDS,
            "patched_forwards": core.EXPECTED_PATCHED_FORWARDS,
            "total_forwards": core.EXPECTED_FORWARDS, "backwards": 0,
        },
        "elapsed_seconds": time.time() - started,
        "next_step": (
            "measure_selected_site_invariance_and_control_ceilings_before_any_projector_fit"
            if selected else "redesign_causal_site_vocabulary_without_rank_search"
        ),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary = {key: value for key, value in result.items() if key.startswith("pred_")}
    for key in (
        "strong_null", "selected_site", "passing_sites_in_frozen_order",
        "model_forwards", "model_backwards", "minimum_source_target_activation_rms",
        "checkpoint", "next_step",
    ):
        summary[key] = result[key]
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
