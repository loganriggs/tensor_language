#!/usr/bin/env python3
"""Rung 536 Stage-B1 CPU audit: distinguish frozen circuit masks from portable labels.

This script loads no model and performs no statistical selection. It verifies the row
authority of the census leaves and the specific proposed target r.2.0.2, then emits a
deterministic receipt used to constrain the later product-space DAS implementation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
BQ = ROOT / "bilinear_quotient"
STATE = BQ / "census_state_diverse.pt"
TARGET = BQ / "circuits" / "r_2_0_2.json"
OUT = BQ / "bilinear_product_das_rung536_label_boundary_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    state = torch.load(STATE, map_location="cpu", weights_only=False)
    target = json.loads(TARGET.read_text())

    assert tuple(state["rows"].shape) == (1000, 513)
    assert tuple(state["basev"].shape) == (256000,)
    leaves = state["leaves"]
    assert len(leaves) > 0
    assert all(isinstance(leaf["member"], torch.Tensor) for leaf in leaves)
    assert all(isinstance(leaf["slice"], torch.Tensor) for leaf in leaves)

    matches = [leaf for leaf in leaves if leaf["tag"] == target["tag"]]
    assert len(matches) == 1
    leaf = matches[0]
    state_members = leaf["member"].cpu().to(torch.int64)
    file_members = torch.tensor(target["members"]["indices"], dtype=torch.int64)
    assert torch.equal(state_members, file_members)
    assert target["tree"] == {"instance": "diverse-1000row-v1", "n_rows": 1000}

    story = target["story"]
    result = {
        "rung": 536,
        "stage": "B1_label_boundary_audit",
        "status": "passed",
        "model_loaded": False,
        "model_forwards": 0,
        "model_backwards": 0,
        "census": {
            "row_shape": list(state["rows"].shape),
            "scored_position_count": int(state["basev"].numel()),
            "leaf_count": len(leaves),
            "all_leaves_store_member_index_tensors": True,
            "all_leaves_store_slice_index_tensors": True,
        },
        "target": {
            "tag": target["tag"],
            "tree_instance": target["tree"]["instance"],
            "tree_n_rows": target["tree"]["n_rows"],
            "member_count": int(state_members.numel()),
            "state_and_circuit_file_members_bit_equal": True,
            "surface_program": story.get("program"),
            "surface_program_balanced_accuracy": story.get("program_bacc"),
            "surface_program_null": story.get("program_null"),
            "mechanism_level": story.get("mechanism_level"),
            "blind_name": story.get("blind_name"),
        },
        "decision": {
            "fresh_document_use_of_frozen_member_indices": "forbidden",
            "fresh_document_targets": ["exact_MLP0_token_only_T", "exact_MLP0_token_by_context_I"],
            "frozen_census_use": "32 discovery and 30 held-out circuit response fingerprints only",
            "real_model_DAS_authorized": False,
        },
        "input_sha256": {
            str(STATE.relative_to(ROOT)): sha256(STATE),
            str(TARGET.relative_to(ROOT)): sha256(TARGET),
        },
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
