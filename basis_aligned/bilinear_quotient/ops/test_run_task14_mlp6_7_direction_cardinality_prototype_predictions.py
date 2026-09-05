from __future__ import annotations

import run_task14_mlp6_7_direction_cardinality_prototype_predictions as predictions


def test_evidence_uses_only_direction_and_cardinality_keys() -> None:
    prototypes = {}
    for direction_index, direction in enumerate(("plural_to_singular", "singular_to_plural")):
        for cardinality in range(5):
            prototypes[f"{direction}.cardinality_{cardinality}"] = {"frozen_reader_q": direction_index + cardinality}
        prototypes[f"{direction}.direction_only"] = {"frozen_reader_q": direction_index + 0.5}
    evidence = predictions.build_evidence(predictions.authority.build_rows(), prototypes)
    assert len(evidence) == 512
    assert len({(item["row_id"], item["background"]) for item in evidence}) == 512
    assert all(str(item["cardinality"]) in item["cardinality_prototype_key"] for item in evidence)


def test_plan_is_zero_execution_and_outcome_closed() -> None:
    plan = predictions.compile_plan()
    assert plan["price"]["physical_model_forwards"] == 0
    assert plan["price"]["sealed_predictions"] == 1024
    assert plan["target_exact_displacements_consumed"] == 0
    assert plan["causal_outcomes_opened"] is False
