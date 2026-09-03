#!/usr/bin/env python3
"""Bind the outcome-closed R538 site screen to the pending-opener dossier."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import (  # noqa: E402
    append_claim_revision, circuit_path, file_sha256, rebuild_registry_v2, validate_v2,
)

TAG = "task.bracket.pending_opener"
OLD_CLAIM = "pending_opener_state.v4"
NEW_CLAIM = "pending_opener_state.v5"
PREREG = "basis_aligned/polynomial_causal/PENDING_OPENER_COMMON_SITE_RUNG538_PREREGISTRATION.md"
IMPLEMENTATION = "basis_aligned/bilinear_quotient/ops/pending_opener_common_site_rung538.py"


def frozen(path: str, kind: str) -> dict:
    source = REPO / path
    assert source.is_file(), path
    return {"path": path, "sha256": file_sha256(source), "kind": kind, "status": "frozen"}


def main() -> None:
    path = circuit_path(TAG)
    record = json.loads(path.read_text())
    artifacts = {
        "r538_site_preregistration": frozen(PREREG, "preregistration"),
        "r538_site_implementation": frozen(IMPLEMENTATION, "implementation"),
    }
    existing = next((item for item in record["claims"] if item["claim_id"] == NEW_CLAIM), None)
    if existing is None:
        previous = next(item for item in record["claims"] if item["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM,
            "revision": 5,
            "status": "specified",
            "supersedes": OLD_CLAIM,
            "next_missing": (
                "execute the hash-frozen 15-site full-state interchange screen on FIT/SELECT; "
                "FINAL_TEST, OOD, controls, and every DAS rank remain unopened"
            ),
        })
        append_claim_revision(TAG, claim, artifacts=artifacts)
    else:
        assert all(record["artifacts"].get(key) == value for key, value in artifacts.items())
    final = json.loads(path.read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({
        "tag": TAG, "claim_id": NEW_CLAIM, "status": "specified",
        "preregistration_sha256": artifacts["r538_site_preregistration"]["sha256"],
        "implementation_sha256": artifacts["r538_site_implementation"]["sha256"],
        "outcomes_opened": False,
    }, indent=2))


if __name__ == "__main__":
    main()
