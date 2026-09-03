#!/usr/bin/env python3
"""Managed one-forward smoke for rung 531; scientific outcomes remain closed."""

# BQGATE: EXPERIMENT

import hashlib
import os
from pathlib import Path

import equality_score_factor_branch_sharing_rung531 as rung531


PREDICTIONS = {
    "pred_a_exact_authorized_instrument": "smoke exercises exact capture and authority checks",
    "pred_b_both_score_factors_shared": "sealed during smoke",
    "pred_c_exactly_one_score_factor_shared": "sealed during smoke",
    "pred_d_factor_gauges_match_product": "sealed during smoke",
}
CORE = Path(__file__).with_name("equality_score_factor_branch_sharing_rung531.py")
CORE_SHA256 = "e2eb9bd2674247c1fa1c0e25a50d4e747b899a2883899f3074bf809bc676f71e"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if sha256(CORE) != CORE_SHA256:
        raise RuntimeError("rung531 core changed after the smoke was frozen")
    os.environ["RUNG531_SMOKE"] = "1"
    rung531.main()


if __name__ == "__main__":
    main()
