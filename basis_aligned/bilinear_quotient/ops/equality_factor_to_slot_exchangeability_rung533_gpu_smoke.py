#!/usr/bin/env python3
"""Managed, outcome-closed structural smoke for rung 533."""

# BQGATE: EXPERIMENT

import hashlib
import os
from pathlib import Path

import equality_factor_to_slot_exchangeability_rung533 as rung533


PREDICTIONS = {
    "pred_a_valid_physical_instrument": "replay, edits, support path, calls, rows, and sources are live",
    "pred_b_product_level_positive_control": "sealed during smoke",
    "pred_c_both_source_factors_fill_target_first": "sealed during smoke",
    "pred_d_both_source_factors_fill_target_second": "sealed during smoke",
    "pred_e_branch_exchangeable_downstream_family": "sealed during smoke",
    "pred_f_donor_background_stability": "sealed during smoke",
}
CORE = Path(__file__).with_name("equality_factor_to_slot_exchangeability_rung533.py")
CORE_SHA256 = "6ba3a9e5fa4e0fa23c461610451bfc8d65eea909f14fe563131a1441228528fd"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if sha256(CORE) != CORE_SHA256:
        raise RuntimeError("rung533 core changed after smoke freeze")
    os.environ["RUNG533_SMOKE"] = "1"
    rung533.main()


if __name__ == "__main__":
    main()
