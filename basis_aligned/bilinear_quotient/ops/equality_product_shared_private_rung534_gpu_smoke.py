#!/usr/bin/env python3
"""Managed, outcome-closed structural smoke for rung 534."""

# BQGATE: EXPERIMENT

import hashlib
import os
from pathlib import Path

import equality_product_shared_private_rung534 as rung534


PREDICTIONS = {
    "pred_a_exact_live_instrument": "replay, split, edits, supports, calls, rows, and sources are live",
    "pred_b_shared_signal_premise_reproduces": "sealed during smoke",
    "pred_c_private_correction_autonomous_on_code": "sealed during smoke",
    "pred_d_private_correction_key_specific": "sealed during smoke",
    "pred_e_private_correction_transfers_to_natural": "sealed during smoke",
    "pred_f_private_correction_survives_redundant_donor": "sealed during smoke",
}
CORE = Path(__file__).with_name("equality_product_shared_private_rung534.py")
CORE_SHA256 = "fdfb3b0ba8a7a5639cb75677e26e33e24b346f6bd7f45de20f40a70090ab5e88"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if sha256(CORE) != CORE_SHA256:
        raise RuntimeError("rung534 core changed after smoke freeze")
    os.environ["RUNG534_SMOKE"] = "1"
    rung534.main()


if __name__ == "__main__":
    main()
