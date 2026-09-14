#!/usr/bin/env python3
"""Freeze the outcome-blind 24-group R585 authority subset for the V1 screen."""
import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
OPS = HERE.parent / "bilinear_quotient" / "ops"
sys.path.insert(0, str(OPS))
import induction_centered_fixed_geometry_rung594 as r594


OUT = HERE / "INDUCTION_CONTEXTUAL_CONSUMER_RESPONSE_V1_ROWS.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    _, authority = r594.load_authority()
    eligible = [
        row for row in authority["directions"]
        if row["split"] == "FIT"
        and row["family"] == "selector_payload_joint_answer_preserved"
    ]
    group_ids = sorted({row["group_id"] for row in eligible})[:24]
    assert len(group_ids) == 24
    selected = [row for row in eligible if row["group_id"] in set(group_ids)]
    selected.sort(key=lambda row: (group_ids.index(row["group_id"]), row["directed_id"]))
    assert len(selected) == 96
    assert all(sum(row["group_id"] == group for row in selected) == 4 for group in group_ids)
    cells = {
        "|".join(str(row[key]) for key in ("family", "variant", "recipient_condition", "direction"))
        for row in selected
    }
    assert len(cells) == 4
    endpoint_ids = sorted({
        row[key] for row in selected
        for key in ("recipient_endpoint_id", "donor_endpoint_id")
    })
    out = {
        "schema": "induction_contextual_consumer_response_v1_rows",
        "selection_rule": "lexicographically first 24 FIT group hashes in frozen R585 authority",
        "discovery_group_ids": group_ids[:12],
        "confirm_group_ids": group_ids[12:],
        "rows": selected,
        "row_count": len(selected),
        "endpoint_ids": endpoint_ids,
        "endpoint_count": len(endpoint_ids),
        "cell_ids": sorted(cells),
        "authority_direction_manifest_sha256": authority["direction_manifest_sha256"],
        "builder_sha256": digest(Path(__file__)),
    }
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: out[k] for k in ("row_count", "endpoint_count", "cell_ids")}, indent=2))


if __name__ == "__main__":
    main()
