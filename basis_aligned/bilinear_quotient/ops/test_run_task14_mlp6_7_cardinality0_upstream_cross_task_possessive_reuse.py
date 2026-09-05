from __future__ import annotations

import run_task14_mlp6_7_cardinality0_upstream_cross_task_possessive_reuse as reuse


def test_selected_rows_are_balanced_answer_changing_families() -> None:
    rows = reuse.selected_rows()
    assert len(rows) == 64
    assert {row["transform_id"] for row in rows} == {"A1", "A2"}
    assert all(row["answer_changes"] for row in rows)
    counts = {}
    for row in rows:
        key = row["direction_id"], row["construction_id"]
        counts[key] = counts.get(key, 0) + 1
    assert set(counts.values()) == {16}


def test_plan_binds_cardinality_zero_without_fit() -> None:
    plan = reuse.compile_plan()
    assert plan["prototype_keys"] == ["plural_to_singular.cardinality_0", "singular_to_plural.cardinality_0"]
    assert plan["fit_operations"] == 0
    assert plan["price"]["example_evaluations"] == 256


def test_batch_scores_donor_minus_base_at_final_position() -> None:
    row = reuse.selected_rows()[0]
    batch = reuse._batch([row])
    assert batch.answer_ids == (row["donor_answer_id"],)
    assert batch.foil_ids == (row["base_answer_id"],)
    assert batch.semantic_positions == (row["base_semantic_position"],)
