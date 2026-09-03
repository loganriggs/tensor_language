#!/usr/bin/env python3
"""Managed fail-closed v2 smoke for rung 532 after dtype-order correction."""

# BQGATE: EXPERIMENT

import hashlib
import os
from pathlib import Path

import equality_factor_companion_causal_equivalence_rung532 as rung532


PREDICTIONS = {
    "pred_a_exact_live_interaction_instrument": "v2 enforces replay, product identity, and edit liveness",
    "pred_b_product_control_transfers": "sealed during smoke",
    "pred_c_source_second_replaces_target_first": "sealed during smoke",
    "pred_d_source_first_replaces_target_second": "sealed during smoke",
    "pred_e_heldout_interaction_defined_factor": "sealed during smoke",
    "pred_f_factor_replacements_compose": "sealed during smoke",
}
CORE = Path(__file__).with_name("equality_factor_companion_causal_equivalence_rung532.py")
CORE_SHA256 = "2207288b731f69a5b540ab101d3b293d2f5ff7831f347d8e59ac00bf7e59e9e2"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if sha256(CORE) != CORE_SHA256:
        raise RuntimeError("rung532 core changed after v2 smoke freeze")
    os.environ["RUNG532_SMOKE"] = "1"
    rung532.main()


if __name__ == "__main__":
    main()
