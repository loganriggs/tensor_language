#!/usr/bin/env python3
"""Instrument-only managed GPU smoke for rung 521 Stage A.

This smoke retains no task, circuit, or scientific outcome.  It checks exact
native replay, a no-op self donor, one live different-document attention8
swap, and the single-call hook before the liveness floor is frozen for the
separate Stage-A science run.
"""

# BQGATE: EXPERIMENT
# pred_a: native dispatched replay is exactly equal to the direct model
# pred_b: self-donor attention8 replacement is an exact activation and logit no-op
# pred_c: one different-document donor produces a nonzero attention8 edit and logit change

from __future__ import annotations

import json
import os


REGISTERED_PREDICTIONS = {
    "pred_a": "native dispatched replay is exactly equal to the direct model",
    "pred_b": "self-donor replacement is an exact activation and logit no-op",
    "pred_c": "a different-document donor gives a nonzero attention8 edit and logit change",
}


if os.environ.get("BQLIB_DRYRUN") == "1":
    print(
        "DRYRUN OK: rung521 instrument-only GPU smoke; no task/circuit metrics; "
        "no optimizer; result path is create-only",
        flush=True,
    )
    raise SystemExit(0)


import attention8_shared_private_das_rung521 as stage_a  # noqa: E402


OUT = stage_a.ROOT / "attention8_shared_private_das_rung521_gpu_smoke.json"


def main() -> dict:
    if os.environ.get("BQLIB_NO_MODEL") == "1":
        raise RuntimeError("BQLIB_NO_MODEL forbids the rung521 GPU smoke")
    result = stage_a._gpu_smoke()
    stage_a._atomic_json(OUT, result)
    print(json.dumps({
        "output": str(OUT),
        "native_replay_logits_exact": result["native_replay_logits_exact"],
        "self_donor_logits_exact": result["self_donor_logits_exact"],
        "self_donor_write_exact": result["self_donor_write_exact"],
        "real_donor_edit_rms": result["real_donor_edit_rms"],
        "real_donor_logits_changed": result["real_donor_logits_changed"],
        "suggested_frozen_floor": result["suggested_frozen_floor"],
        "scientific_metrics_retained": result["scientific_metrics_retained"],
    }, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
