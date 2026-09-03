#!/usr/bin/env python3
"""Register the pre-outcome list-middle role correction as claim v2."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import append_artifacts, append_claim_revision, circuit_path, file_sha256, rebuild_registry_v2, validate_v2  # noqa: E402

TAG = "task.numbered_list.index_successor"


def main() -> None:
    overlay_path = "basis_aligned/bilinear_quotient/increment_rung568_semantic_role_overlay.json"
    append_artifacts(TAG, {"r568_role_overlay": {"path": overlay_path, "sha256": file_sha256(REPO / overlay_path),
                                                   "kind": "semantic_correction", "status": "frozen"}})
    record = json.loads(circuit_path(TAG).read_text())
    if not any(claim["claim_id"] == "numbered_list_index_successor.v2" for claim in record["claims"]):
        old = next(claim for claim in record["claims"] if claim["claim_id"] == "numbered_list_index_successor.v1")
        claim = copy.deepcopy(old)
        claim.update({"claim_id": "numbered_list_index_successor.v2", "revision": 2,
                      "status": "specified", "supersedes": "numbered_list_index_successor.v1",
                      "next_missing": "run the R569 FIT/SELECT native gate with list_middle_index_break scored as invariance under the R568 overlay"})
        middle = next(family for family in claim["counterfactual_families"] if family["family_id"] == "list_middle_index_break")
        middle.update({
            "role": "invariance",
            "changes": ["an earlier middle label and visible sequence coherence"],
            "holds_fixed": ["final visible label", "last-label-plus-one operation", "registered successor"],
            "control_ids": ["surface edit", "state shift", "step-two conflict"],
        })
        append_claim_revision(TAG, claim)
    final = json.loads(circuit_path(TAG).read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({"tag": TAG, "claim_id": "numbered_list_index_successor.v2",
                      "corrected_family": "list_middle_index_break", "role": "invariance",
                      "outcomes_opened": []}, indent=2))


if __name__ == "__main__":
    main()
