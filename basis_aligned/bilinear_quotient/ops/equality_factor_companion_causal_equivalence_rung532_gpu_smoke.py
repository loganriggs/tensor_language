#!/usr/bin/env python3
"""Managed 21-forward outcome-closed smoke for rung 532."""

# BQGATE: EXPERIMENT

import hashlib
import os
from pathlib import Path

import equality_factor_companion_causal_equivalence_rung532 as rung532


PREDICTIONS = {
    "pred_a_exact_live_interaction_instrument": "smoke exercises exact replay and all physical edits",
    "pred_b_product_control_transfers": "sealed during smoke",
    "pred_c_source_second_replaces_target_first": "sealed during smoke",
    "pred_d_source_first_replaces_target_second": "sealed during smoke",
    "pred_e_heldout_interaction_defined_factor": "sealed during smoke",
    "pred_f_factor_replacements_compose": "sealed during smoke",
}
CORE = Path(__file__).with_name("equality_factor_companion_causal_equivalence_rung532.py")
CORE_SHA256 = "877453b5471b167cf7b47a88219b7405b824fce349b69f435b035b8ebaa23f0b"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if sha256(CORE) != CORE_SHA256:
        raise RuntimeError("rung532 core changed after smoke freeze")
    os.environ["RUNG532_SMOKE"] = "1"
    rung532.main()


if __name__ == "__main__":
    main()
