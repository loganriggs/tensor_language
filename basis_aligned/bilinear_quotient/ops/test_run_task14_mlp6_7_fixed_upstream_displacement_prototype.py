from __future__ import annotations

import math

import pytest

import run_task14_mlp6_7_fixed_upstream_displacement_prototype as prototype


def _evidence() -> list[dict[str, object]]:
    rows = []
    subsets = ("", "E", "A", "EA", "EAUW")
    for direction in ("plural_to_singular", "singular_to_plural"):
        for index in range(16):
            for subset in subsets:
                rows.append({
                    "row_id": f"{direction}.{index}",
                    "background": subset,
                    "direction": direction,
                    "template": "above_inside" if index % 2 else "inside_above",
                    "cardinality": len(subset),
                    "fixed_reader_q": float((index + 1) * (len(subset) + 1)),
                })
    return rows


def test_build_predictions_excludes_entire_held_row() -> None:
    rows = _evidence()
    # Pad each row to the real 16-subset cardinality profile.
    all_subsets = ("", "E", "A", "U", "W", "EA", "EU", "EW", "AU", "AW", "UW", "EAU", "EAW", "EUW", "AUW", "EAUW")
    rows = [
        {**row, "background": subset, "cardinality": len(subset), "fixed_reader_q": row["fixed_reader_q"] + len(subset)}
        for row in rows[::5]
        for subset in all_subsets
    ]
    predictions = prototype.build_predictions(rows)
    assert len(predictions) == 512
    assert all(item["exact_pool_rows"] == 15 for item in predictions)
    assert all(
        item["cardinality_pool_values"] == 15 * math.comb(4, item["cardinality"])
        for item in predictions
    )


def test_build_predictions_rejects_duplicate_keys() -> None:
    rows = _evidence()
    with pytest.raises(prototype.PrototypeError, match="512"):
        prototype.build_predictions(rows)


def test_score_recovers_perfect_prototype_and_control_gap() -> None:
    predictions = []
    causal = []
    all_subsets = ("", "E", "A", "U", "W", "EA", "EU", "EW", "AU", "AW", "UW", "EAU", "EAW", "EUW", "AUW", "EAUW")
    for index in range(32):
        direction = "plural_to_singular" if index < 16 else "singular_to_plural"
        template = "above_inside" if index % 2 else "inside_above"
        for subset in all_subsets:
            value = (-1.0 if direction == "plural_to_singular" else 1.0) * (len(subset) + 1)
            prediction = {
                "row_id": str(index), "background": subset, "direction": direction,
                "template": template, "cardinality": len(subset),
                "exact_subset_prototype_q": value, "cardinality_control_q": value * 0.1,
                "exact_pool_rows": 15,
                "cardinality_pool_values": 15 * math.comb(4, len(subset)),
                "held_row_excluded": True,
            }
            predictions.append(prediction)
            causal.append({
                "row_id": str(index), "background": subset, "direction": direction,
                "template": template, "cardinality": len(subset), "actual_q": value,
            })
    scored = prototype.score(predictions, causal)
    assert all(scored["predictions"].values())
    assert scored["sse_reduction_over_cardinality_control"] == pytest.approx(1.0)
