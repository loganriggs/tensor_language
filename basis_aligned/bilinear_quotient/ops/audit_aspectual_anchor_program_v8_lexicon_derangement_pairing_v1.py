#!/usr/bin/env python3
"""Zero-forward pre-execution audit of the frozen lexicon derangement."""

# BQLANE: cpu
# BQGATE: AUDIT pred_a_authority pred_b_exact_derangement pred_c_exact_coverage
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import run_aspectual_anchor_program_v8_cross_construction_variable_interchange_v1 as parent
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_program_v8_lexicon_deranged_variable_interchange_v1.json"
PARENT_RESULT = ROOT / "circuits/followups/aspectual_anchor_program_v8_cross_family_variable_interchange_v1_result.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_program_v8_lexicon_derangement_pairing_audit_v1_result.json"
EXPECTED_PRIOR_SHA256 = "9183760625e2de4165368966b4314e16b5f2abc6bdcabac0b884595f9c73fe3f"
EXPECTED_PARENT_SHA256 = "b10b919f4c3cf9d53d1a397326f5d46a195ecd710f3df24fb8acff35966ea031"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps({"dryrun": True, "model_forwards": 0, "pair_count": 64}))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    prior = json.loads(PRIOR.read_text())
    parent_result = json.loads(PARENT_RESULT.read_text())
    lexical_rows, _lexical_spec, fresh_rows, _fresh_spec = parent.validate_static()
    pairs = []
    for panel, rows in (("lexical", lexical_rows), ("fresh", fresh_rows)):
        indexed = {(row["transform_id"], row["group_number"]): row for row in rows}
        for target in rows:
            source = indexed[(target["transform_id"], (target["group_number"] + 6) % 16)]
            pairs.append({
                "panel": panel,
                "family": target["transform_id"],
                "target_group": target["group_number"],
                "source_group": source["group_number"],
                "target_row_id": str(target["row_id"]),
                "source_row_id": str(source["row_id"]),
                "same_direction": target["direction_id"] == source["direction_id"],
                "different_reporter": target["reporter"] != source["reporter"],
                "different_period": target["object_name"] != source["object_name"],
            })
    pred_a = sha(PRIOR) == EXPECTED_PRIOR_SHA256 and sha(PARENT_RESULT) == EXPECTED_PARENT_SHA256 and prior["candidate_id"] == "aspectual_anchor.has_vs_had.program_v8_lexicon_deranged_variable_interchange_v1" and parent_result["terminal"] == "screen"
    pred_b = all(pair["source_group"] == (pair["target_group"] + 6) % 16 and pair["same_direction"] and pair["different_reporter"] and pair["different_period"] and pair["source_row_id"] != pair["target_row_id"] for pair in pairs)
    pred_c = len(pairs) == 64 and len({(pair["panel"], pair["target_row_id"]) for pair in pairs}) == 64 and len({(pair["panel"], pair["source_row_id"]) for pair in pairs}) == 64
    predictions = {"pred_a_authority": pred_a, "pred_b_exact_derangement": pred_b, "pred_c_exact_coverage": pred_c}
    terminal = "screen" if all(predictions.values()) else "invalid"
    value = {
        "schema": "aspectual_anchor_program_v8_lexicon_derangement_pairing_audit_result_v1",
        "candidate_id": "aspectual_anchor.has_vs_had.program_v8_lexicon_derangement_pairing_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prior_art_sha256": sha(PRIOR),
        "parent_result_sha256": sha(PARENT_RESULT),
        "predictions": predictions,
        "pair_count": len(pairs),
        "pairs": pairs,
        "price": {"model_forwards": 0, "example_evaluations": 0, "fit_parameters": 0},
        "terminal": terminal,
    }
    atomic_create_json(OUT, value)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "pair_count": len(pairs), "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
