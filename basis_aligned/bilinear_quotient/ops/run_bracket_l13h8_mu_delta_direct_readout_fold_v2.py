#!/usr/bin/env python3
"""Three-forward instrument-only successor for the direct L13H8 readout fold."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_direct_folded_factor_dominates pred_c_normalization_or_softcap_material

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import circuit_fast_screen_candidate_bracket_l13h8_mu_delta_direct_readout_fold_v2 as authority
import run_bracket_l13h8_mu_delta_direct_readout_fold as parent


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits" / "fast_screens" / "bracket_l13h8_mu_delta_direct_readout_fold_v2_result.json"


def main():
    plan = authority.compile_plan()
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    parent.authority = authority
    parent.parent.base.parent.shared.candidate = authority
    torch, F, facade = parent.parent.base.parent.shared._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        records = parent.evaluate(model, torch, F, facade)
    screen = parent.score(records)
    terminal = "invalid" if not screen["instrument_live"] else "screen"
    result = {
        "schema": "bracket_l13h8_mu_delta_direct_readout_fold_result_v2",
        "plan": plan,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "raw": records,
        "screen": screen,
        "evaluated_splits": ["FRESH_BASIC"],
        "forbidden_splits_opened": [],
        "model_forwards": 3,
        "terminal": terminal,
        "predictions": {
            "pred_a_instrument_live": screen["predictions"]["pred_a"],
            "pred_b_direct_folded_factor_dominates": screen["predictions"]["pred_b"],
            "pred_c_normalization_or_softcap_material": screen["predictions"]["pred_c"],
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"result": str(OUT), "terminal": terminal,
                      "model_forwards": 3}, indent=2))


if __name__ == "__main__":
    main()
