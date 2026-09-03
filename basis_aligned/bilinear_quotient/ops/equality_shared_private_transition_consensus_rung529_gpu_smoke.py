#!/usr/bin/env python3
"""Managed GPU smoke launcher for rung 529; retains no task/circuit outcome."""

# BQGATE: EXPERIMENT

import json
import os

import equality_shared_private_transition_consensus_rung529_run as rung


if os.environ.get("BQLIB_DRYRUN") == "1":
    print(json.dumps(rung.dry_run(), indent=2, sort_keys=True))
else:
    rung.gpu_smoke()
