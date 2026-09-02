#!/usr/bin/env python3
"""RUNG501 -- directed action graph among four equality-score components.

Implementation in progress. The dry-run freezes the candidate graph, action count,
data partitions, and calibration tripwires before any new candidate outcome opens.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os


TERMS = ("L5H5", "L7H3", "L8H3", "L8H4")
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (3, 2))
PAIR_NAMES = tuple(f"{TERMS[left]}->{TERMS[right]}" for left, right in PAIRS)
KNOWN_POSITIVE = "L5H5->L8H4"
KNOWN_NEGATIVE = "L7H3->L8H4"
PARTITIONS = ((0, 250), (250, 500), (500, 750), (750, 1000))
BATCH = 4
SCALE_FORWARDS = 24
FORWARDS_PER_BATCH = 2 + 9 * len(PAIRS)
DISCOVERY_FORWARDS = 125 * FORWARDS_PER_BATCH
VALIDATION_FORWARDS = 125 * FORWARDS_PER_BATCH


def main():
    if os.environ.get("BQLIB_DRYRUN") != "1":
        raise RuntimeError("rung501 implementation is not complete and must not run on GPU")
    assert len(PAIRS) == 7 and len(set(PAIRS)) == 7
    assert KNOWN_POSITIVE in PAIR_NAMES and KNOWN_NEGATIVE in PAIR_NAMES
    assert FORWARDS_PER_BATCH == 65
    print(json.dumps({
        "status": "implementation_in_progress_dry_run_passed", "rung": 501,
        "model_loaded": False, "candidate_outcomes_opened": False,
        "pairs": PAIR_NAMES, "partitions": PARTITIONS,
        "scale_forwards": SCALE_FORWARDS,
        "discovery_forwards": DISCOVERY_FORWARDS,
        "conditional_validation_forwards": VALIDATION_FORWARDS,
        'pred_a_instrument': None, 'pred_b_tripwires': None,
        'pred_c_new_edge': None, 'pred_d_typed_graph': None,
        'pred_e_validation': None, 'pred_f_interpretation': None,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
