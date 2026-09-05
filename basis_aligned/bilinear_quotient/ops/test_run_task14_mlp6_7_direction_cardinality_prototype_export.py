from __future__ import annotations

import torch

import run_task14_mlp6_7_direction_cardinality_prototype_export as export


def test_prototype_group_counts_and_values() -> None:
    records = []
    for direction_index, direction in enumerate(("plural_to_singular", "singular_to_plural")):
        for cardinality in range(5):
            for index in range(16 * __import__("math").comb(4, cardinality)):
                records.append({
                    "direction": direction, "cardinality": cardinality,
                    "delta": torch.full((3,), float(direction_index + cardinality + 1)),
                })
    readers = {direction: torch.ones(3) for direction in ("plural_to_singular", "singular_to_plural")}
    prototypes = export.summarize_prototypes(records, readers, torch)
    assert len(prototypes) == 12
    assert prototypes["plural_to_singular.cardinality_2"]["coordinates"] == [3.0, 3.0, 3.0]
    assert prototypes["singular_to_plural.cardinality_4"]["frozen_reader_q"] == 18.0
    assert all(item["training_vectors"] > 0 for item in prototypes.values())


def test_plan_has_literal_storage_and_no_third_corpus() -> None:
    plan = export.compile_plan()
    assert plan["price"]["stored_scalars"] == 13824
    assert plan["third_corpus_rows_consumed"] == 0
    assert plan["third_corpus_outcomes_consumed"] == 0
