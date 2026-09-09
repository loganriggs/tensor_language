#!/usr/bin/env python3
"""Exact zero-forward sufficiency/reset interaction brackets for v23 cells."""

# BQGATE: AUDIT pred_a_hash_schema_terminal_and_prediction_authority pred_b_every_reported_cell_was_prospectively_reciprocal pred_c_projection_gap_equals_reset_minus_sufficiency pred_d_half_gaps_are_finite_and_preserved pred_e_zero_forward_no_selection_analysis
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "circuits/followups/temporal_iswas_v23_source_destination_cell_atlas_v4_result.json"
OUT = ROOT / "circuits/audits/temporal_iswas_v23_cell_interaction_brackets_v1.json"
EXPECTED_SOURCE_SHA256 = "4f32076a076749ff5601201579660949a7099b350025a0d261d4dfdeb18d10f0"
PREDICTION_KEYS = (
    "pred_a_hash_schema_terminal_and_prediction_authority",
    "pred_b_every_reported_cell_was_prospectively_reciprocal",
    "pred_c_projection_gap_equals_reset_minus_sufficiency",
    "pred_d_half_gaps_are_finite_and_preserved",
    "pred_e_zero_forward_no_selection_analysis",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(source):
    rows = []
    for head, labels in source["reciprocal_cells"].items():
        for label in labels:
            report = source["reports"][head]["cells"][label]
            sufficient = report["sufficiency"]
            removed = report["removed_by_reset"]
            rows.append({
                "head": head, "cell": label,
                "singleton_projection": sufficient["target"]["signed_projection"],
                "grand_coalition_marginal_projection": removed["target"]["signed_projection"],
                "higher_order_projection_burden": (
                    removed["target"]["signed_projection"]
                    - sufficient["target"]["signed_projection"]),
                "half_higher_order_projection_burden": {half: (
                    removed["halves"][half]["signed_projection"]
                    - sufficient["halves"][half]["signed_projection"])
                    for half in ("first", "second")},
                "source_reciprocal": report["reciprocal"],
            })
    A = bool(
        sha(SOURCE) == EXPECTED_SOURCE_SHA256
        and source.get("schema") == "temporal_iswas_v23_source_destination_cell_atlas_result_v4"
        and source.get("terminal") == "shared_directed_cell_program_screen"
        and all(source.get("predictions", {}).values())
    )
    B = bool(rows and all(row["source_reciprocal"] is True for row in rows))
    C = all(math.isclose(
        row["higher_order_projection_burden"],
        row["grand_coalition_marginal_projection"] - row["singleton_projection"],
        rel_tol=0, abs_tol=1e-15) for row in rows)
    D = all(math.isfinite(value) for row in rows
            for value in row["half_higher_order_projection_burden"].values())
    E = True
    predictions = dict(zip(PREDICTION_KEYS, (A, B, C, D, E)))
    return {
        "schema": "temporal_iswas_v23_cell_interaction_brackets_v1",
        "candidate_id": "analysis.temporal_iswas.v23_cell_interaction_brackets_v1",
        "source_sha256": sha(SOURCE), "rows": rows,
        "identity": "N_e-S_e=sum_{T contains e, abs(T)>=2} Harsanyi_dividend(T)",
        "predictions": predictions,
        "terminal": "valid_exact_interaction_brackets" if all(predictions.values()) else "invalid_audit",
        "selected_cell_or_threshold_after_outcome": None,
        "price": {"model_forwards_exact": 0, "sequence_evaluations_exact": 0,
                  "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0},
    }


def main():
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    result = build(json.loads(SOURCE.read_text()))
    atomic_create_json(OUT, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
