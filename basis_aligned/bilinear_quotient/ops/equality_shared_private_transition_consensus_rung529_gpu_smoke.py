#!/usr/bin/env python3
"""Managed GPU smoke launcher for rung 529; retains no task/circuit outcome."""

# BQGATE: EXPERIMENT

import json
import os

import equality_shared_private_transition_consensus_rung529_run as rung


REGISTERED_PREDICTIONS = (
    "pred_a_exact_live_shared_private_instrument",
    "pred_b_consensus_beats_every_singleton",
    "pred_c_new_document_physical_consensus",
    "pred_d_heldout_circuits_and_documents",
    "pred_e_sufficient_selectively_removable_shared_state",
)


if os.environ.get("BQLIB_DRYRUN") == "1":
    print(json.dumps(rung.dry_run(), indent=2, sort_keys=True))
else:
    rung.gpu_smoke()
