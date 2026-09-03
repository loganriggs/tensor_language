#!/usr/bin/env python3
"""Managed full launcher for rung 529, frozen to the passing v2 smoke."""

# BQGATE: EXPERIMENT

import json
import os
from pathlib import Path

import equality_shared_private_transition_consensus_rung529_run as rung


REGISTERED_PREDICTIONS = (
    "pred_a_exact_live_shared_private_instrument",
    "pred_b_consensus_beats_every_singleton",
    "pred_c_new_document_physical_consensus",
    "pred_d_heldout_circuits_and_documents",
    "pred_e_sufficient_selectively_removable_shared_state",
)
MANAGED_SMOKE_V2_SHA256 = "03a039a0ea4735f196d9f84457803f88ac95eea46b19ae89de8b8eef5223d213"


if os.environ.get("BQLIB_DRYRUN") == "1":
    print(json.dumps(rung.dry_run(), indent=2, sort_keys=True))
else:
    rung.run_full(MANAGED_SMOKE_V2_SHA256, Path(__file__).resolve())
