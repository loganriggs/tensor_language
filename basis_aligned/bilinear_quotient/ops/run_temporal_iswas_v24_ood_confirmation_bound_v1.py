#!/usr/bin/env python3
"""Managed fail-closed launcher for the preregistered unfiltered v24 OOD confirmation.

The launcher adds no scientific choices. It atomically binds an immutable, fully
passing capability receipt to the frozen confirmation runner and then delegates.
"""

# BQGATE: EXPERIMENT pred_a_authority_capability_alignment_closures_finiteness_and_exact_price pred_b_union_selectively_transfers_to_new_constructions pred_c_at_least_three_singletons_transfer_selectively pred_d_transfer_is_stable_across_frozen_reporter_halves
from __future__ import annotations

import json
import os

import run_temporal_iswas_v24_unfiltered_four_head_ood_confirmation_v1 as confirmation
import temporal_iswas_v24_ood_confirmation_binding as binding


REGISTERED_PREDICTIONS = {
    "pred_a_authority_capability_alignment_closures_finiteness_and_exact_price":
        "Delegate unchanged to the frozen v24 confirmation runner.",
    "pred_b_union_selectively_transfers_to_new_constructions":
        "Delegate unchanged to the frozen v24 confirmation runner.",
    "pred_c_at_least_three_singletons_transfer_selectively":
        "Delegate unchanged to the frozen v24 confirmation runner.",
    "pred_d_transfer_is_stable_across_frozen_reporter_halves":
        "Delegate unchanged to the frozen v24 confirmation runner.",
}


def main():
    dry_run = bool(os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"))
    status = binding.create_binding(dry_run=dry_run)
    if dry_run:
        print(json.dumps({
            "candidate_id": confirmation.CANDIDATE_ID,
            "conditional_launcher": True,
            "gpu_accessed": False,
            "model_loaded": False,
            "queue_touched": False,
            "binding_status": status,
            "registered_predictions": REGISTERED_PREDICTIONS,
            "price": confirmation.PRICE,
        }, sort_keys=True))
        return
    if status.get("status") != "bound":
        raise RuntimeError(f"v24 OOD confirmation dependency is not licensed: {status}")
    confirmation.main()


if __name__ == "__main__":
    main()
