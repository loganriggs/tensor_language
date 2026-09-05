from __future__ import annotations

import run_task14_mlp6_7_direction_cardinality_prototype_cross_circuit_collateral as collateral


def test_plan_tests_every_program_vector_without_pooling() -> None:
    plan = collateral.compile_plan()
    assert plan["prototype_count"] == 10
    assert plan["bars"]["pool_behaviors"] is False
    assert plan["bars"]["pool_prototypes"] is False
    assert plan["price"] == collateral.derive_price()


def test_batch_preserves_registered_positions() -> None:
    rows = [{
        "run_id": "a", "ids": [1, 2, 3], "answer_id": 7,
        "foil_id": 8, "semantic_position": 2,
    }]
    batch = collateral._batch(rows)
    assert batch.row_ids == ("a",)
    assert batch.semantic_positions == (2,)
    assert batch.answer_ids == (7,)


def test_price_covers_native_noop_and_all_installations() -> None:
    price = collateral.derive_price()
    assert price["native_evaluations"] + price["zero_add_replays"] + price["nonzero_program_installations"] == price["example_evaluations"] == 384
    assert price["physical_model_forwards"] == 3
