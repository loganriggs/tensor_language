#!/usr/bin/env python3
"""Managed structural v3 smoke for rung 532 after support-axis correction."""

# BQGATE: EXPERIMENT

import hashlib
import os
from pathlib import Path

import equality_factor_companion_causal_equivalence_rung532 as rung532


PREDICTIONS = {
    "pred_a_exact_live_interaction_instrument": "v3 exercises replay, edits, and support accumulation",
    "pred_b_product_control_transfers": "sealed during smoke",
    "pred_c_source_second_replaces_target_first": "sealed during smoke",
    "pred_d_source_first_replaces_target_second": "sealed during smoke",
    "pred_e_heldout_interaction_defined_factor": "sealed during smoke",
    "pred_f_factor_replacements_compose": "sealed during smoke",
}
CORE = Path(__file__).with_name("equality_factor_companion_causal_equivalence_rung532.py")
CORE_SHA256 = "142f4a0f05d582413fb6eac1820654dc6d4491690af9742e0a2d81eac719fdb8"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if sha256(CORE) != CORE_SHA256:
        raise RuntimeError("rung532 core changed after v3 smoke freeze")
    os.environ["RUNG532_SMOKE"] = "1"
    rung532.main()


if __name__ == "__main__":
    main()
